"""
@title

@description

"""
import math
from functools import partial
from pathlib import Path

import numpy as np

from island_influence import project_properties
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


def create_base_env(
        location_funcs, num_harvesters, num_excavators, num_obstacles, num_pois, obs_rad, agent_size, obs_size, poi_size, agent_weight, obs_weight, poi_weight,
        agent_value, obstacle_value, poi_value, max_vel, sen_res, delta_time, render_mode, max_steps, collision_penalty_scalar, reward_type,
        normalize_rewards, save_dir
):
    state_size = sen_res * Agent.NUM_BINS
    action_size = 2
    n_hidden = math.ceil((state_size + action_size) / 2)

    harvester_policy = NeuralNetwork(n_inputs=state_size, n_outputs=action_size, n_hidden=n_hidden)
    harvesters = [
        Agent(idx, AgentType.Harvester, obs_rad, agent_size, agent_weight, agent_value, max_vel, harvester_policy, sense_function='regions')
        for idx in range(num_harvesters)
    ]

    excavator_policy = NeuralNetwork(n_inputs=state_size, n_outputs=action_size, n_hidden=n_hidden)
    excavators = [
        Agent(idx, AgentType.Excavator, obs_rad, agent_size, agent_weight, agent_value, max_vel, excavator_policy, sense_function='regions')
        for idx in range(num_excavators)
    ]

    obstacles = [
        Obstacle(idx, AgentType.Obstacle, obs_rad, obs_size, obs_weight, obstacle_value)
        for idx in range(num_obstacles)
    ]
    pois = [
        Poi(idx, AgentType.StaticPoi, obs_rad, poi_size, poi_weight, poi_value)
        for idx in range(num_pois)
    ]

    # normalize_rewards
    env_func = partial(HarvestEnv, harvesters=harvesters, excavators=excavators, obstacles=obstacles, pois=pois, location_funcs=location_funcs,
                       max_steps=max_steps, delta_time=delta_time, collision_penalty_scalar=collision_penalty_scalar, reward_type=reward_type,
                       normalize_rewards=normalize_rewards, render_mode=render_mode, save_dir=save_dir)
    return env_func


def rand_ring_env(
        scale_factor=1, num_harvesters=4, num_excavators=4, num_obstacles=8, num_pois=8, obs_rad=2, max_vel=1,
        agent_size=1, obs_size=1, poi_size=1, agent_weight=1, obs_weight=1, poi_weight=1, agent_value=1, obstacle_value=1, poi_value=1,
        sen_res=8, delta_time=1, max_steps=100, collision_penalty_scalar=0, reward_type='global', normalize_rewards=True,
        save_dir=Path(project_properties.env_dir), render_mode=None
):
    agent_bounds = np.asarray([0, 3]) * scale_factor
    obstacle_bounds = np.asarray([5, 8]) * scale_factor
    poi_bounds = np.asarray([10, 13]) * scale_factor
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
    env_func = create_base_env(
        location_funcs, num_harvesters=num_harvesters, num_excavators=num_excavators, num_obstacles=num_obstacles, num_pois=num_pois, obs_rad=obs_rad,
        agent_size=agent_size, obs_size=obs_size, poi_size=poi_size, agent_weight=agent_weight, obs_weight=obs_weight, poi_weight=poi_weight,
        agent_value=agent_value, obstacle_value=obstacle_value, poi_value=poi_value, max_vel=max_vel,  sen_res=sen_res, delta_time=delta_time,
        max_steps=max_steps, collision_penalty_scalar=collision_penalty_scalar, reward_type=reward_type, normalize_rewards=normalize_rewards,
        render_mode=render_mode, save_dir=save_dir
    )
    return env_func


def det_ring_env(
        scale_factor=1, num_harvesters=4, num_excavators=4, num_obstacles=8, num_pois=8, obs_rad=2, max_vel=1,
        agent_size=1, obs_size=1, poi_size=1, agent_weight=1, obs_weight=1, poi_weight=1, agent_value=1, obstacle_value=1, poi_value=1,
        sen_res=8, delta_time=1, max_steps=100, collision_penalty_scalar=0, reward_type='global', normalize_rewards=True,
        save_dir=Path(project_properties.env_dir), render_mode=None
):
    agent_bounds = np.asarray([0, 3]) * scale_factor
    obstacle_bounds = np.asarray([5, 8]) * scale_factor
    poi_bounds = np.asarray([10, 13]) * scale_factor
    center_loc = (5, 5)

    agent_locs = partial(deterministic_ring, center=center_loc, radius=np.average(agent_bounds))
    obstacle_locs = partial(deterministic_ring, center=center_loc, radius=np.average(obstacle_bounds))
    poi_locs = partial(deterministic_ring, center=center_loc, radius=np.average(poi_bounds))
    location_funcs = {
        'harvesters': agent_locs,
        'excavators': agent_locs,
        'obstacles': obstacle_locs,
        'pois': poi_locs,
    }
    env_func = create_base_env(
        location_funcs, num_harvesters=num_harvesters, num_excavators=num_excavators, num_obstacles=num_obstacles, num_pois=num_pois, obs_rad=obs_rad,
        agent_size=agent_size, obs_size=obs_size, poi_size=poi_size, agent_weight=agent_weight, obs_weight=obs_weight, poi_weight=poi_weight,
        agent_value=agent_value, obstacle_value=obstacle_value, poi_value=poi_value, max_vel=max_vel, sen_res=sen_res, delta_time=delta_time,
        max_steps=max_steps, collision_penalty_scalar=collision_penalty_scalar, reward_type=reward_type, normalize_rewards=normalize_rewards,
        render_mode=render_mode, save_dir=save_dir
    )
    return env_func
