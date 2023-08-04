"""
@title

@description

"""
import argparse
import datetime
import math
from functools import partial
from pathlib import Path

import numpy as np

from island_influence import project_properties
from island_influence.agent import Agent, Poi, Obstacle, AgentType
from island_influence.harvest_env import HarvestEnv
from island_influence.learn.cceaV2 import ccea
from island_influence.learn.neural_network import NeuralNetwork
from island_influence.utils import random_ring


def main(main_args):
    obs_rad = 2
    max_vel = 1

    agent_weight = 1
    obs_weight = 1
    poi_weight = 1

    agent_value = 1
    obstacle_value = 1
    poi_value = 1

    env_scale_factor = 5
    agent_bounds = np.asarray([0, 3]) * env_scale_factor
    obstacle_bounds = np.asarray([5, 8]) * env_scale_factor
    poi_bounds = np.asarray([10, 13]) * env_scale_factor

    agent_locs = partial(random_ring, **{'center': (5, 5), 'min_rad': agent_bounds[0], 'max_rad': agent_bounds[1]})
    obstacle_locs = partial(random_ring, **{'center': (5, 5), 'min_rad': obstacle_bounds[0], 'max_rad': obstacle_bounds[1]})
    poi_locs = partial(random_ring, **{'center': (5, 5), 'min_rad': poi_bounds[0], 'max_rad': poi_bounds[1]})
    location_funcs = {
        'harvesters': agent_locs,
        'excavators': agent_locs,
        'obstacles': obstacle_locs,
        'pois': poi_locs,
    }

    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    experiment_dir = Path(project_properties.exps_dir, f'harvest_exp_{date_str}')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)

    num_gens = 10000
    # todo  make num_sim keyed to each agent type
    # to simulate all agents in both populations, choose a number larger than the population sizes of each agent type
    num_sims = 50
    # num_sims = {AgentType.Harvester: 20, AgentType.Excavators: 20}
    sen_res = 8
    delta_time = 1
    render_mode = None
    max_steps = 100

    population_sizes = {AgentType.Harvester: 50, AgentType.Excavators: 50}
    num_harvesters = 4
    num_excavators = 4
    num_obstacles = 100
    num_pois = 10

    n_inputs = sen_res * Agent.NUM_BINS
    n_outputs = 2
    n_hidden = math.ceil((n_inputs + n_outputs) / 2)
    policy = NeuralNetwork(n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden)

    agent_pops = {
        agent_type: [
            NeuralNetwork(n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden)
            for _ in range(pop_size // 5)
        ]
        for agent_type, pop_size in population_sizes.items()
    }

    harvesters = [
        Agent(idx, AgentType.Harvester, True, obs_rad, agent_weight, agent_value, max_vel, policy, sense_function='regions')
        for idx in range(num_harvesters)
    ]
    excavators = [
        Agent(idx, AgentType.Excavators, True, obs_rad, agent_weight, agent_value, max_vel, policy, sense_function='regions')
        for idx in range(num_excavators)
    ]

    obstacles = [
        Obstacle(idx, AgentType.Obstacle, obs_rad, obs_weight, obstacle_value)
        for idx in range(num_obstacles)
    ]
    pois = [
        Poi(idx, AgentType.StaticPoi, obs_rad, poi_weight, poi_value)
        for idx in range(num_pois)
    ]

    env = HarvestEnv(
        harvesters=harvesters, excavators=excavators, obstacles=obstacles, pois=pois,
        location_funcs=location_funcs, max_steps=max_steps, delta_time=delta_time, render_mode=render_mode
    )
    env.reset()

    top_inds = ccea(
        env, agent_policies=agent_pops, population_sizes=population_sizes,
        num_gens=num_gens, num_sims=num_sims, experiment_dir=experiment_dir
    )
    for agent_type, individuals in top_inds.items():
        for each_ind in individuals:
            print(f'{each_ind.fitness}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
