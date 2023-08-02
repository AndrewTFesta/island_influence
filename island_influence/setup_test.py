"""
@title

@description

"""
import math

import numpy as np

from island_influence.agent import Poi, Obstacle, Agent, AgentType
from island_influence.harvest_env import HarvestEnv
from island_influence.learn.neural_network import NeuralNetwork


def linear_setup():
    render_mode = 'rgb_array'
    delta_time = 1

    obs_rad = 2
    max_vel = 1
    sen_res = 8

    agent_weight = 1
    obs_weight = 1
    poi_weight = 1

    agent_value = 1
    obstacle_value = 1
    poi_value = 1

    agent_config = [
        (AgentType.Harvester, False, np.asarray((5, 1))),
        (AgentType.Excavators, False, np.asarray((5, 2))),
        (AgentType.Harvester, False, np.asarray((5, 3))),
        (AgentType.Excavators, False, np.asarray((5, 4))),
        (AgentType.Harvester, False, np.asarray((5, 5))),
        (AgentType.Excavators, False, np.asarray((5, 6))),
        (AgentType.Harvester, False, np.asarray((5, 7))),
        (AgentType.Excavators, False, np.asarray((5, 8))),
    ]

    obstacle_config = [
        (AgentType.Obstacle, np.asarray((6, 1))),
        (AgentType.Obstacle, np.asarray((6, 2))),
        (AgentType.Obstacle, np.asarray((6, 3))),
        (AgentType.Obstacle, np.asarray((6, 4))),
        (AgentType.Obstacle, np.asarray((6, 5))),
        (AgentType.Obstacle, np.asarray((6, 6))),
        (AgentType.Obstacle, np.asarray((6, 7))),
        (AgentType.Obstacle, np.asarray((6, 8))),
    ]

    poi_config = [
        (AgentType.StaticPoi, np.asarray((4, 1))),
        # (AgentType.StaticPoi, np.asarray((4, 2))),
        (AgentType.StaticPoi, np.asarray((4, 3))),
        # (AgentType.StaticPoi, False, np.asarray((4, 4))),
        (AgentType.StaticPoi, np.asarray((4, 5))),
        # (AgentType.StaticPoi, np.asarray((4, 6))),
        (AgentType.StaticPoi, np.asarray((4, 7))),
        # (AgentType.StaticPoi, np.asarray((4, 8))),
    ]

    n_inputs = sen_res * Agent.NUM_BINS
    n_outputs = 2
    n_hidden = math.ceil((n_inputs + n_outputs) / 2)
    policy = NeuralNetwork(n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden)

    # agent_id: int, agent_type: AgentType, learner: bool, location: np.ndarray, observation_radius, weight: float, value: float,
    # max_velocity: float = 0.0, policy: NeuralNetwork | None = None, sense_function='regions'
    agents = [
        Agent(idx, agent_info[0], agent_info[1], agent_info[2], obs_rad, agent_weight, agent_value, max_vel, policy, sense_function='regions')
        for idx, agent_info in enumerate(agent_config)
    ]
    # agent_id, agent_type, location, observation_radius, weight, value
    obstacles = [
        Obstacle(idx, agent_info[0], agent_info[1], obs_rad, obs_weight, obstacle_value)
        for idx, agent_info in enumerate(obstacle_config)
    ]
    pois = [
        Poi(idx, agent_info[0], agent_info[1], obs_rad, poi_weight, poi_value)
        for idx, agent_info in enumerate(poi_config)
    ]

    env = HarvestEnv(agents=agents, obstacles=obstacles, pois=pois, max_steps=100, delta_time=delta_time, render_mode=render_mode)
    return env
