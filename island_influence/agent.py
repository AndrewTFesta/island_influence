"""
@title

@description

"""
import copy
from enum import Enum, auto

import numpy as np
import torch
from gym.vector.utils import spaces

from island_influence.learn.neural_network import NeuralNetwork


def relative(start_loc, end_loc):
    assert len(start_loc) == len(end_loc)

    dx = end_loc[0] - start_loc[0]
    dy = end_loc[1] - start_loc[1]
    angle = np.arctan2(dy, dx)
    angle = np.degrees(angle)
    angle = angle % 360

    dist = np.linalg.norm(np.asarray(end_loc) - np.asarray(start_loc))
    return angle, dist


class AgentType(Enum):
    Harvester = auto()
    Support = auto()
    Obstacle = auto()
    StaticPoi = auto()
    MovingPoi = auto()


class Agent:
    ROW_MAPPING = {
        AgentType.Harvester: 0,
        AgentType.Support: 0,
        AgentType.Obstacle: 1,
        AgentType.StaticPoi: 2,
        AgentType.MovingPoi: 2
    }

    NUM_BINS = 3

    def __init__(self, agent_id: int, agent_type: AgentType, location: np.ndarray, observation_radius, weight: float, value: float,
                 max_velocity: float = 0.0, policy: NeuralNetwork | None = None, sense_function='regions'):
        self.name = f'{agent_type.name}_{agent_id}'
        self.id = agent_id
        self.agent_type = agent_type

        self._initial_location = copy.copy(location)
        self.location = location

        self.observation_radius = observation_radius
        self.weight = weight
        self.value = value

        # lower/upper bounds agent is able to move
        # same for both x and y directions
        self.max_velocity = max_velocity
        self.policy = policy

        # input is one bin for each of
        #   other agents
        #   obstacles
        #   pois
        # self.n_in = self.sensor_resolution * self.num_bins
        self.sensor_resolution = int(policy.n_inputs / self.NUM_BINS) if policy is not None else None

        self.sense_functions = {
            'regions': self._sense_regions,
            'vision': self._sense_vision,
        }

        self._sense_func = self.sense_functions.get(sense_function, list(self.sense_functions.keys())[0])
        return

    def __repr__(self):
        return f'({self.name=}: {self.agent_type}: {self.location=})'

    def observation_space(self):
        sensor_range = spaces.Box(
            low=0, high=np.inf,
            shape=(self.sensor_resolution, self.NUM_BINS), dtype=np.float64
        )
        return sensor_range

    def action_space(self):
        action_range = spaces.Box(
            low=-1 * self.max_velocity, high=self.max_velocity,
            shape=(self.policy.n_outputs,), dtype=np.float64
        )
        return action_range

    def reset(self):
        self.location = copy.copy(self._initial_location)
        return

    def observable_agents(self, relative_agents, observation_radius):
        """
        observable_agents

        :param relative_agents:
        :param observation_radius:
        :return:
        """
        bins = []
        for idx, agent in enumerate(relative_agents):
            assert isinstance(agent, Agent)
            if agent == self:
                continue

            angle, dist = relative(self.location, agent.location)
            if dist <= observation_radius:
                bins.append((agent, angle, dist))
        return bins

    def _sense_regions(self, other_agents, offset=False):
        """
        Takes in the state of the worlds and counts how many agents are in each d-hyperoctant around the agent,
        with the agent being at the center of the observation.

        Calculates which pois, leaders, and follower go into which d-hyperoctant, where d is the state
        resolution of the environment.

        first set of (sensor_resolution) bins is for leaders/followers
        second set of (sensor_resolution) bins is for pois

        :param other_agents:
        :param offset:
        :return:
        """

        obs_agents = Agent.observable_agents(self, other_agents, self.observation_radius)

        bin_size = 360 / self.sensor_resolution
        if offset:
            offset = 360 / (self.sensor_resolution * 2)
            bin_size = offset * 2

        observation = np.zeros((self.NUM_BINS, self.sensor_resolution))
        counts = np.ones(observation.shape)
        for idx, entry in enumerate(obs_agents):
            agent, angle, dist = entry
            agent_type_idx = self.ROW_MAPPING[agent.agent_type]
            bin_idx = int(np.floor(angle / bin_size) % self.sensor_resolution)
            observation[agent_type_idx, bin_idx] += agent.value / max(dist, 0.01)
            counts[agent_type_idx, bin_idx] += 1

        observation = np.divide(observation, counts)
        observation = np.nan_to_num(observation)
        observation = observation.flatten()
        return observation

    def _sense_vision(self, other_agents, blur=False):
        """
        Computes an observation as the 2D rectangle surrounding the agent.

        :return:
        """
        # todo  implement
        return other_agents

    def sense(self, other_agents):
        return self._sense_func(other_agents)

    def get_action(self, observation):
        """
        Computes the x and y vectors using the active policy and the passed in observation.

        :param observation:
        :return:
        """
        active_policy = self.policy
        with torch.no_grad():
            action = active_policy(observation)
            action = action.numpy()

        mag = np.linalg.norm(action)
        if mag > self.max_velocity:
            action = action / mag
            action *= self.max_velocity
        return action


class Obstacle(Agent):

    def __init__(self, agent_id, agent_type, location, observation_radius, weight, value):
        super().__init__(agent_id, agent_type, location, observation_radius, weight, value)
        return


class Poi(Agent):

    def __init__(self, agent_id, agent_type, location, observation_radius, weight, value):
        super().__init__(agent_id, agent_type, location, observation_radius, weight, value)
        return

    def sense(self, other_agents):
        # sense nearby harvester agents
        return np.asarray([0, 0])

    def get_action(self, observation):
        return np.asarray([0, 0])
