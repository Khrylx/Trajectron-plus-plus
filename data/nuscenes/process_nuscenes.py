import sys
import os
import numpy as np
import pandas as pd
import pickle
import json
from tqdm import tqdm
from pyquaternion import Quaternion
from kalman_filter import LinearPointMass, NonlinearKinematicBicycle
from scipy.integrate import cumtrapz
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure

#op_path = './pytorch-openpose/python/'
sys.path.append("/home/yyuan2/Documents/repo/Trajectron-plus-plus/code_traj")
#sys.path.append(op_path)
from data import Environment, Scene, Node, BicycleNode, Position, Velocity, Acceleration, ActuatorAngle, Map, Scalar
from dataloader import data_generator
from utils.utils import Config


types = ['PEDESTRIAN',
         'BICYCLE',
         'VEHICLE']

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 25},
            'y': {'mean': 0, 'std': 25}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'heading': {
            'value': {'mean': 0, 'std': np.pi},
            'derivative': {'mean': 0, 'std': np.pi / 4}
        }
    },
    'BICYCLE': {
        'position': {
            'x': {'mean': 0, 'std': 50},
            'y': {'mean': 0, 'std': 50}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 6},
            'y': {'mean': 0, 'std': 6},
            'm': {'mean': 0, 'std': 6}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'm': {'mean': 0, 'std': 4}
        },
        'actuator_angle': {
            'steering_angle': {'mean': 0, 'std': np.pi/2}
        },
        'heading': {
            'value': {'mean': 0, 'std': np.pi},
            'derivative': {'mean': 0, 'std': np.pi / 4}
        }
    },
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 100},
            'y': {'mean': 0, 'std': 100}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 20},
            'y': {'mean': 0, 'std': 20},
            'm': {'mean': 0, 'std': 20}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'm': {'mean': 0, 'std': 4}
        },
        'actuator_angle': {
            'steering_angle': {'mean': 0, 'std': np.pi/2}
        },
        'heading': {
            'value': {'mean': 0, 'std': np.pi},
            'derivative': {'mean': 0, 'std': np.pi / 4}
        }
    }
}

def inverse_np_gradient(f, dx, F0=0.):
    N = f.shape[0]
    return F0 + np.hstack((np.zeros((N, 1)), cumtrapz(f, axis=1, dx=dx)))

def integrate_trajectory(v, x0, dt):
    xd_ = inverse_np_gradient(v[..., 0], dx=dt, F0=x0[0])
    yd_ = inverse_np_gradient(v[..., 1], dx=dt, F0=x0[1])
    integrated = np.stack([xd_, yd_], axis=2)
    return integrated

def integrate_heading_model(a, dh, h0, x0, v0, dt):
    h = inverse_np_gradient(dh, dx=dt, F0=h0)
    v_m = inverse_np_gradient(a, dx=dt, F0=v0)

    vx = np.cos(h) * v_m
    vy = np.sin(h) * v_m

    v = np.stack((vx, vy), axis=2)
    return integrate_trajectory(v, x0, dt)


if __name__ == "__main__":
    num_global_straight = 0
    num_global_curve = 0

    """data"""
    cfg, settings_show = Config('config.yml')
    log = open('log.txt', 'w')
    ph = cfg.past_frames

    for data_class in ['train', 'val']:
        print(f"Processing data class {data_class}")
        if data_class == 'train':
            generator = data_generator(cfg, log, split='train', phase='training')
        else:
            generator = data_generator(cfg, log, split='val', phase='testing', test_type='forecast')

        data_dict_path = os.path.join('../processed', '_'.join(['nuscenes', data_class, f'ph{ph}', 'v1.pkl']))
        env = Environment(node_type_list=types, standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.BICYCLE)] = 10.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.BICYCLE)] = 20.0
        attention_radius[(env.NodeType.BICYCLE, env.NodeType.PEDESTRIAN)] = 10.0
        attention_radius[(env.NodeType.BICYCLE, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.BICYCLE, env.NodeType.BICYCLE)] = 10.0
        env.attention_radius = attention_radius
        scenes = []
        while not generator.is_epoch_end():
            data = generator()
            if data[0] is None:
                continue
            
            print(generator.index)
            cur_motion_3D, pre_motion_3D, fut_motion_3D, fut_motion_mask, \
            pre_data, fut_data, gt_matrix, seq_name, frame = data

            start_data = pre_data[0]
            """pre_data""" 
            all_arr = []
            for i in range(start_data.shape[0]):    # number of objects in the most recent frame
                identity = start_data[i, 1]

                # """past frames for debug""" 
                most_recent_data = start_data[i].copy()
                first_data = start_data[i].copy()
                first_data[0] = ph - 1
                past_arr = [first_data]
                for j in range(1, cfg.past_frames):
                    cur_data = pre_data[j]              # past_data
                    if len(cur_data) > 0 and identity in cur_data[:, 1]:
                        data = cur_data[cur_data[:, 1] == identity].squeeze()
                    else:
                        # print('past missing', identity, ph - 1 - j)
                        data = most_recent_data.copy()
                    data[0] = ph - 1 - j
                    most_recent_data = data.copy()
                    past_arr.insert(0, data)
                all_arr += past_arr

                """future frames"""
                most_recent_data = start_data[i].copy()
                for j in range(1, cfg.future_frames + 1):
                    cur_data = fut_data[j]
                    if len(cur_data) > 0 and identity in cur_data[:, 1]:
                        data = cur_data[cur_data[:, 1] == identity].squeeze()
                    else:
                        # print('missing', identity, ph - 1 + j)
                        data = most_recent_data.copy()
                    data[0] = ph - 1 + j
                    most_recent_data = data.copy()
                    all_arr.append(data)

            all_data = np.vstack(all_arr)

            data = pd.DataFrame(columns=['frame_id',
                                         'type',
                                         'node_id',
                                         'robot',
                                         'x', 'y', 'z',
                                         'length',
                                         'width',
                                         'height',
                                         'heading',
                                         'orientation'])

            for obj in all_data:
                
                data_point = pd.Series({'frame_id': int(obj[0]),
                                            'type': env.NodeType.VEHICLE,
                                            'node_id': int(obj[1]),
                                            'robot': False,
                                            'x': obj[13],
                                        'y': obj[15],
                                        'z': obj[14],
                                            'length': obj[12],
                                            'width': obj[11],
                                            'height': obj[10],
                                            'heading': obj[16],
                                            'orientation': None})
                data = data.append(data_point, ignore_index=True)


            if len(data.index) == 0:
                continue

            data.sort_values('frame_id', inplace=True)
            max_timesteps = data['frame_id'].max()

            # x_min = np.round(data['x'].min() - 50)
            # x_max = np.round(data['x'].max() + 50)
            # y_min = np.round(data['y'].min() - 50)
            # y_max = np.round(data['y'].max() + 50)

            # data['x'] = data['x'] - x_min
            # data['y'] = data['y'] - y_min

            scene_id = f"{seq_name}_{frame}"
            scene = Scene(timesteps=max_timesteps + 1, dt=1.0, name=scene_id)

            # Generate Maps

        #     type_map = dict()
        #     homography = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])
        #    # VEHICLES
        #     map_mask_vehicle = np.zeros((783, 804, 3))
        #     type_map['VEHICLE'] = Map(data=map_mask_vehicle, homography=homography, description='')
        #     scene.map = type_map
        #     del map_mask_vehicle

            for node_id in pd.unique(data['node_id']):
                node_df = data[data['node_id'] == node_id]

                if node_df['x'].shape[0] < 2:
                    continue

                if not np.all(np.diff(node_df['frame_id']) == 1):
                    #print('Occlusion')
                    continue # TODO Make better

                node_values = node_df['x'].values
                if node_df.iloc[0]['type'] == env.NodeType.PEDESTRIAN:
                    node = Node(type=node_df.iloc[0]['type'])
                else:
                    node = BicycleNode(type=node_df.iloc[0]['type'])
                node.node_id = node_id
                node.first_timestep = node_df['frame_id'].iloc[0]
                node.position = Position(node_df['x'].values, node_df['y'].values)
                node.velocity = Velocity.from_position(node.position, scene.dt)
                node.velocity.m = np.linalg.norm(np.vstack((node.velocity.x, node.velocity.y)), axis=0)
                node.acceleration = Acceleration.from_velocity(node.velocity, scene.dt)
                node.heading = Scalar(node_df['heading'].values)
                heading_t = node_df['heading'].values.copy()
                shifted_heading = np.zeros_like(node.heading.value)
                shifted_heading[0] = node.heading.value[0]
                for i in range(1, len(node.heading.value)):
                    if not (np.sign(node.heading.value[i]) == np.sign(node.heading.value[i - 1])) and np.abs(
                            node.heading.value[i]) > np.pi / 2:
                        shifted_heading[i] = shifted_heading[i - 1] + (
                                node.heading.value[i] - node.heading.value[i - 1]) - np.sign(
                            (node.heading.value[i] - node.heading.value[i - 1])) * 2 * np.pi
                    else:
                        shifted_heading[i] = shifted_heading[i - 1] + (
                                node.heading.value[i] - node.heading.value[i - 1])
                node.heading.value = shifted_heading
                node.length = node_df.iloc[0]['length']
                node.width = node_df.iloc[0]['width']

                if node_df.iloc[0]['robot'] == True:
                    node.is_robot = True

                if node_df.iloc[0]['type'] == env.NodeType.PEDESTRIAN:
                    pass
                else:
                    filter_veh = NonlinearKinematicBicycle(lf=node.length*0.6, lr=node.length*0.4, dt=scene.dt)
                    for i in range(len(node.position.x)):
                        if i == 0:  # initalize KF
                            # initial P_matrix
                            P_matrix = np.identity(4)
                        elif i < len(node.position.x):
                            # assign new est values
                            node.position.x[i] = x_vec_est_new[0][0]
                            node.position.y[i] = x_vec_est_new[1][0]
                            node.heading.value[i] = x_vec_est_new[2][0]
                            node.velocity.m[i] = x_vec_est_new[3][0]

                        if i < len(node.position.x) - 1:  # no action on last data
                            # filtering
                            x_vec_est = np.array([[node.position.x[i]],
                                                  [node.position.y[i]],
                                                  [node.heading.value[i]],
                                                  [node.velocity.m[i]]])
                            z_new = np.array([[node.position.x[i+1]],
                                              [node.position.y[i+1]],
                                              [node.heading.value[i+1]],
                                              [node.velocity.m[i+1]]])
                            x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
                                x_vec_est=x_vec_est,
                                u_vec=np.array([[0.], [0.]]),
                                P_matrix=P_matrix,
                                z_new=z_new
                            )
                            P_matrix = P_matrix_new

                v_tmp = node.velocity.m
                node.velocity = Velocity.from_position(node.position, scene.dt)
                node.velocity.m = v_tmp
                #if (np.abs(np.linalg.norm(np.vstack((node.velocity.x, node.velocity.y)), axis=0) - v_tmp) > 0.4).any():
                #    print(np.abs(np.linalg.norm(np.vstack((node.velocity.x, node.velocity.y)), axis=0) - v_tmp))

                node.acceleration = Acceleration.from_velocity(node.velocity, scene.dt)
                node.acceleration.m = np.gradient(v_tmp, scene.dt)
                node.heading.derivative = np.gradient(node.heading.value, scene.dt)
                node.heading.value = (node.heading.value + np.pi) % (2.0 * np.pi) - np.pi

                scene.nodes.append(node)
                if node.is_robot is True:
                    scene.robot = node

            robot = False
            num_heading_changed = 0
            num_moving_vehicles = 0
            for node in scene.nodes:
                node.description = "straight"
                num_global_straight += 1
                if node.type == env.NodeType.VEHICLE:
                    if np.linalg.norm((node.position.x[0] - node.position.x[-1], node.position.y[0] - node.position.y[-1])) > 10:
                        num_moving_vehicles += 1
                    if np.abs(node.heading.value[0] - node.heading.value[-1]) > np.pi / 6:
                        if not np.sign(node.heading.value[0]) == np.sign(node.heading.value[-1]) and np.abs(node.heading.value[0] > 1/2 * np.pi):
                            if (node.heading.value[0] - node.heading.value[-1]) - np.sign((node.heading.value[0] - node.heading.value[-1])) * 2 * np.pi > np.pi / 6:
                                node.description = "curve"
                                num_global_curve += 1
                                num_global_straight -= 1
                                num_heading_changed += 1
                        else:
                            node.description = "curve"
                            num_global_curve += 1
                            num_global_straight -= 1
                            num_heading_changed += 1

                if node.is_robot:
                    robot = True

            if num_moving_vehicles > 0 and num_heading_changed / num_moving_vehicles > 0.4:
                scene.description = "curvy"
            else:
                scene.description = "straight"

            if len(scene.nodes) > 0:
                scenes.append(scene)

            del data

            # if len(scenes) > 5:
            #     break

        env.scenes = scenes

        if len(scenes) > 0:
            print(f'{data_class} num_scenes: {len(scenes)}')
            with open(data_dict_path, 'wb') as f:
                pickle.dump(env, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(num_global_straight)
    print(num_global_curve)