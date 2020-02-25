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
parser.add_argument("--model", help="model full path", type=str, default='../../data/kitti/logs/models_23_Feb_2020_19_05_34')
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int, default=1999)
parser.add_argument("--data", help="full path to data file", type=str, default='../../data/processed/kitti_val_ph10_v1.pkl')
parser.add_argument("--output", help="full path to output csv file", type=str)
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
                for timestep in range(scene.timesteps):
                    predictions = eval_stg.predict(scene,
                                                   np.array([timestep]),
                                                   ph,
                                                   num_samples_z=2000,
                                                   most_likely_z=False,
                                                   min_future_timesteps=ph)

                    if not predictions:
                        continue

                    eval_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                          scene.dt,
                                                                          node_type_enum=env.NodeType,
                                                                          max_hl=max_hl,
                                                                          ph=ph,
                                                                          map=None,
                                                                          obs=False)

                    eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, eval_error_dict[node_type]['ade']))
                    eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, eval_error_dict[node_type]['fde']))
                    eval_kde_nll = np.hstack((eval_kde_nll, eval_error_dict[node_type]['kde']))
                    eval_obs_viols = np.hstack((eval_obs_viols, eval_error_dict[node_type]['obs_viols']))

                    del predictions
                    del eval_error_dict

            print(f"Final Mean Displacement Error @{ph * scene.dt}s: {np.mean(eval_fde_batch_errors)}")
            print(f"Road Violations @{ph * scene.dt}s: {100 * np.sum(eval_obs_viols) / (eval_obs_viols.shape[0] * 2000)}%")
            # pd.DataFrame({'error_value': eval_ade_batch_errors, 'error_type': 'ade', 'type': 'full', 'ph': ph}).to_csv(args.output + '_ade_full_' + str(ph)+'ph' + '.csv')
            # pd.DataFrame({'error_value': eval_fde_batch_errors, 'error_type': 'fde', 'type': 'full', 'ph': ph}).to_csv(args.output + '_fde_full' + str(ph)+'ph' + '.csv')
            # pd.DataFrame({'error_value': eval_kde_nll, 'error_type': 'kde', 'type': 'full', 'ph': ph}).to_csv(args.output + '_kde_full' + str(ph)+'ph' + '.csv')
            # pd.DataFrame({'error_value': eval_obs_viols, 'error_type': 'obs', 'type': 'full', 'ph': ph}).to_csv(args.output + '_obs_full' + str(ph)+'ph' + '.csv')
