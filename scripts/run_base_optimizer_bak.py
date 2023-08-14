"""
@title

@description

"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import trange

from island_influence import project_properties
from island_influence.agent import Poi, AgentType
from island_influence.harvest_env import HarvestEnv
from island_influence.learn.optimizer.cceaV2 import ccea, rollout
from island_influence.learn.neural_network import NeuralNetwork
from island_influence.utils import load_config


def run_experiment(experiment_config, meta_vars):
    # todo  add noise to location of agents
    leaders = [
        Leader(
            agent_id=idx, location=each_pos, sensor_resolution=meta_vars['sensor_resolution'],
            value=meta_vars['leader_value'],
            max_velocity=meta_vars['leader_max_velocity'], weight=meta_vars['leader_weight'],
            observation_radius=meta_vars['leader_obs_rad'], policy=None)
        for idx, each_pos in enumerate(experiment_config['leader_positions'])
    ]
    followers = [
        Follower(
            agent_id=idx, location=each_pos, sensor_resolution=meta_vars['sensor_resolution'],
            value=meta_vars['follower_value'],
            max_velocity=meta_vars['follower_max_velocity'], weight=meta_vars['follower_weight'],
            repulsion_radius=meta_vars['repulsion_rad'], repulsion_strength=meta_vars['repulsion_strength'],
            attraction_radius=meta_vars['attraction_rad'], attraction_strength=meta_vars['attraction_strength'])
        for idx, each_pos in enumerate(experiment_config['follower_positions'])
    ]
    pois = [
        Poi(agent_id=idx, location=each_pos, sensor_resolution=meta_vars['sensor_resolution'],
            value=meta_vars['poi_value'],
            weight=meta_vars['poi_weight'],
            observation_radius=meta_vars['poi_obs_rad'], coupling=meta_vars['poi_coupling'])
        for idx, each_pos in enumerate(experiment_config['poi_positions'])
    ]
    env = HarvestEnv(leaders=leaders, followers=followers, pois=pois, max_steps=meta_vars['episode_length'])

    ######################################################
    # todo  allow for policy sharing
    agent_pops = {
        agent_name: [
            {
                'network': NeuralNetwork(
                    n_inputs=env.agent_mapping[agent_name].n_in,
                    n_hidden=meta_vars['n_hidden_layers'],
                    n_outputs=env.agent_mapping[agent_name].n_out,
                ),
                'fitness': None
            }
            for _ in range(meta_vars['population_size'], )
        ]
        for agent_name in env.agents
        if env.agent_mapping[agent_name].type == AgentType.Learner
    }

    # initial fitness evaluation of all networks in population
    print(f'Initializing fitness values for networks')
    for pop_idx in trange(meta_vars['population_size']):
        new_inds = {agent_name: policy_info[pop_idx] for agent_name, policy_info in agent_pops.items()}
        agent_rewards = rollout(env, new_inds, render=False)
        for agent_name, policy_info in agent_pops.items():
            policy_fitness = agent_rewards[agent_name]
            policy_info[pop_idx]['fitness'] = policy_fitness
    ########################################################
    print(f'Starting experiment: {meta_vars["config_name"]} | {meta_vars["reward_key"]}')
    start_time = time.time()
    best_solution = ccea(
        env, agent_pops, meta_vars['population_size'], meta_vars['n_gens'], meta_vars['num_simulations'],
        experiment_dir=meta_vars['experiment_dir']
    )
    end_time = time.time()
    print(f'Time to train: {end_time - start_time}')

    rewards = rollout(env, best_solution)
    print(f'{rewards=}')
    return



    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
