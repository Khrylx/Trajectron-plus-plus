import sys
sys.path.append('../../code_traj')
import os
import pickle
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.dyn_stg import SpatioTemporalGraphCVAEModel
import evaluation
from utils import prediction_output_to_trajectories
from scipy.interpolate import RectBivariateSpline

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str, default='../../data/kitti/logs/models_25_Feb_2020_20_56_04')
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int, default=1999)
parser.add_argument("--sample_num", type=int, default=20)
parser.add_argument("--data", help="full path to data file", type=str, default='../../data/processed/kitti_val_ph30_v1.pkl')
parser.add_argument("--output", help="output_folder", type=str, default='~/results/trajectron++')
parser.add_argument("--node_type", help="Node Type to evaluate", type=str, default='VEHICLE')
parser.add_argument("--prediction_horizon", nargs='+', help="prediction horizon", type=int, default=None)
args = parser.parse_args()


def load_model(model_dir, env, ts=99):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    stg = SpatioTemporalGraphCVAEModel(model_registrar,
                                       hyperparams,
                                       None, 'cuda:0')
    hyperparams['incl_robot_node'] = False

    stg.set_scene_graph(env)
    stg.set_annealing_params()
    return stg, hyperparams


def save_predictions(prediction_output_dict, scene, max_hl, ph):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict, scene.dt, max_hl, ph)

    prediction_dict = prediction_dict[max_hl]
    futures_dict = futures_dict[max_hl]

    seq, frame = scene.name.split('_')
    frame = int(frame)
    save_dir = os.path.join(output_dir, seq, f'frame_{frame:06d}')
    os.makedirs(save_dir, exist_ok=True)
    for i in range(args.sample_num):
        results = []
        nodes = list(prediction_dict.keys())
        nodes.sort(key=lambda x: x.node_id)
        for node in nodes:
            node_id = node.node_id
            pred = prediction_dict[node][i]
            pred = np.hstack([np.arange(ph)[:, None] + frame + 1, np.ones((ph, 1)) * node_id, pred])
            results.append(pred)
        results = np.concatenate(results, axis=0)
        np.savetxt(f'{save_dir}/sample_{i:03d}.txt', results, fmt="%.3f")
    return



if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = pickle.load(f, encoding='latin1')
    scenes = env.scenes

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['state'],
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    if args.prediction_horizon is None:
        args.prediction_horizon = [hyperparams['prediction_horizon']]

    output_dir = os.path.join(os.path.expanduser(args.output), os.path.basename(args.model))
    os.makedirs(output_dir, exist_ok=True)

    for ph in args.prediction_horizon:
        print(f"Prediction Horizon: {ph}")
        max_hl = hyperparams['maximum_history_length']
        node_type = env.NodeType[args.node_type]
        print(f"Node Type: {node_type.name}")
        print(f"Edge Radius: {hyperparams['edge_radius']}")

        with torch.no_grad():
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_kde_nll = np.array([])
            eval_obs_viols = np.array([])
            print("-- Evaluating Full")
            for i, scene in enumerate(tqdm(scenes)):
                timestep = hyperparams['maximum_history_length']
                predictions = eval_stg.predict(scene,
                                                np.array([timestep]),
                                                ph,
                                                num_samples_z=args.sample_num,
                                                most_likely_z=False,
                                                min_future_timesteps=ph)

                if not predictions:
                    continue

                save_predictions(predictions,
                                    scene,
                                    max_hl=max_hl,
                                    ph=ph)

                del predictions
