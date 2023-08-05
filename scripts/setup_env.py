"""
@title

@description

"""
import math
from functools import partial

import numpy as np

from island_influence.agent import Poi, Obstacle, Agent, AgentType
from island_influence.harvest_env import HarvestEnv
from island_influence.learn.neural_network import NeuralNetwork
from island_influence.utils import random_ring, deterministic_ring


def create_agent_policy(agent, learner):
    obs_space = agent.observation_space().shape
    action_space = agent.action_space().shape

    # state_size = agent.sensor_resolution * Agent.NUM_BINS
    # action_size = 2
    obs_size = np.prod(obs_space)
    action_size = action_space[0]
    num_hidden = math.ceil((obs_size + action_size) / 2)

    policy = NeuralNetwork(n_inputs=obs_size, n_outputs=action_size, n_hidden=num_hidden, learner=learner)
    return policy


def create_base_env(location_funcs):
    obs_rad = 2
    max_vel = 1

    agent_weight = 1
    obs_weight = 1
    poi_weight = 1

    agent_value = 1
    obstacle_value = 1
    poi_value = 1

    sen_res = 8
    delta_time = 1
    render_mode = None
    max_steps = 100

    num_harvesters = 4
    num_excavators = 4
    num_obstacles = 8
    num_pois = 8

    state_size = sen_res * Agent.NUM_BINS
    action_size = 2
    n_hidden = math.ceil((state_size + action_size) / 2)

    harvester_policy = NeuralNetwork(n_inputs=state_size, n_outputs=action_size, n_hidden=n_hidden)
    harvesters = [
        Agent(idx, AgentType.Harvester, obs_rad, agent_weight, agent_value, max_vel, harvester_policy, sense_function='regions')
        for idx in range(num_harvesters)
    ]

    excavator_policy = NeuralNetwork(n_inputs=state_size, n_outputs=action_size, n_hidden=n_hidden)
    excavators = [
        Agent(idx, AgentType.Excavator, obs_rad, agent_weight, agent_value, max_vel, excavator_policy, sense_function='regions')
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


def rand_ring_env():
    env_scale_factor = 5
    agent_bounds = np.asarray([0, 3]) * env_scale_factor
    obstacle_bounds = np.asarray([5, 8]) * env_scale_factor
    poi_bounds = np.asarray([10, 13]) * env_scale_factor
    center_loc = (5, 5)

    agent_locs = partial(random_ring, **{'center': center_loc, 'min_rad': agent_bounds[0], 'max_rad': agent_bounds[1]})
    obstacle_locs = partial(random_ring, **{'center': center_loc, 'min_rad': obstacle_bounds[0], 'max_rad': obstacle_bounds[1]})
    poi_locs = partial(random_ring, **{'center': center_loc, 'min_rad': poi_bounds[0], 'max_rad': poi_bounds[1]})
    location_funcs = {
        'harvesters': agent_locs,
        'excavators': agent_locs,
        'obstacles': obstacle_locs,
        'pois': poi_locs,
    }
    enc = create_base_env(location_funcs)
    return enc


def det_ring_env():
    num_agents = 10
    num_obstacles = 20
    num_pois = 10

    env_scale_factor = 5
    agent_bounds = np.asarray([0, 3]) * env_scale_factor
    obstacle_bounds = np.asarray([5, 8]) * env_scale_factor
    poi_bounds = np.asarray([10, 13]) * env_scale_factor
    center_loc = (5, 5)

    agent_locs = deterministic_ring(num_points=num_agents, center=center_loc, radius=np.average(agent_bounds))
    obstacle_locs = deterministic_ring(num_points=num_obstacles, center=center_loc, radius=np.average(obstacle_bounds))
    poi_locs = deterministic_ring(num_points=num_pois, center=center_loc, radius=np.average(poi_bounds))
    location_funcs = {
        'harvesters': agent_locs,
        'excavators': agent_locs,
        'obstacles': obstacle_locs,
        'pois': poi_locs,
    }
    env = create_base_env(location_funcs)
    return env
