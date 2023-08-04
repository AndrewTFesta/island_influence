"""
@title

@description

"""
import argparse
import math
from functools import partial
from pathlib import Path

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

    agent_bounds = [0, 3]
    obstacle_bounds = [5, 8]
    poi_bounds = [10, 13]

    agent_locs = partial(random_ring, **{'center': (5, 5), 'min_rad': agent_bounds[0], 'max_rad': agent_bounds[1]})
    obstacle_locs = partial(random_ring, **{'center': (5, 5), 'min_rad': obstacle_bounds[0], 'max_rad': obstacle_bounds[1]})
    poi_locs = partial(random_ring, **{'center': (5, 5), 'min_rad': poi_bounds[0], 'max_rad': poi_bounds[1]})
    location_funcs = {
        'harvesters': agent_locs,
        'excavators': agent_locs,
        'obstacles': obstacle_locs,
        'pois': poi_locs,
    }

    # todo append exp_dir with date string
    experiment_dir = Path(project_properties.exps_dir)
    num_gens = 10
    num_sims = 10
    sen_res = 4
    delta_time = 1
    render_mode = None
    max_steps = 100

    population_sizes = {AgentType.Harvester: 20, AgentType.Excavators: 20}
    num_harvesters = 4
    num_excavators = 4
    num_obstacles = 10
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
