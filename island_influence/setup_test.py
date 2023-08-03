"""
@title

@description

"""
import math
from functools import partial

from island_influence.agent import Poi, Obstacle, Agent, AgentType
from island_influence.harvest_env import HarvestEnv
from island_influence.learn.neural_network import NeuralNetwork
from island_influence.utils import random_ring


def linear_setup():
    render_mode = 'human'
    delta_time = 1
    max_steps = 100

    obs_rad = 2
    max_vel = 1
    sen_res = 8

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

    num_harvesters = 4
    num_excavators = 4
    num_obstacles = 8
    num_pois = 8

    n_inputs = sen_res * Agent.NUM_BINS
    n_outputs = 2
    n_hidden = math.ceil((n_inputs + n_outputs) / 2)
    policy = NeuralNetwork(n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden)

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
    return env
