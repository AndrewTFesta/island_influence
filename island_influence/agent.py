"""
@title

@description

"""
import copy
import math
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


def closest_agent_sets(origin_set, end_set, min_dist=math.inf):
    closest = {}
    for origin_name, origin_agent in origin_set.items():
        origin_location = origin_agent.location
        closest_agent = None
        closest_dist = math.inf
        for end_name, end_agent in end_set.items():
            end_location = end_agent.location
            angle, dist = relative(end_location, origin_location)
            if dist < closest_dist:
                closest_dist = dist
                closest_agent = end_agent
        if closest_dist <= min_dist:
            closest[origin_name] = (closest_agent, closest_dist)
    return closest


class AgentType(Enum):
    Harvester = auto()
    Excavators = auto()
    Obstacle = auto()
    StaticPoi = auto()
    MovingPoi = auto()


class Agent:
    ROW_MAPPING = {
        AgentType.Harvester: 0,
        AgentType.Excavators: 0,
        AgentType.Obstacle: 1,
        AgentType.StaticPoi: 2,
        AgentType.MovingPoi: 2
    }

    NUM_BINS = 3

    def __init__(self, agent_id: int, agent_type: AgentType, location: np.ndarray, observation_radius, weight: float, value: float,
                 max_velocity: float = 0.0, policy: NeuralNetwork | None = None, sense_function='regions'):
        """
        The weight of an agent is how many agents it counts as when checking if the agent has an effect on another agent.
        This essentially acts as a coupling value, where a single agent might be able to account for "more than one" agent when
        computing if the coupling requirement of a poi/obstacle has been satisfied.

        The value of an agent is how much the agent is capable of affecting another agent.
        This essentially affects how many interactions a poi/obstacle can have with other agents in the environment before
        no longer having an effect on the environment.

        For instance, a higher valued agent will be able to observe more of a POI whereas a higher weight agent will require fewer agents to observe it.

        :param agent_id:
        :param agent_type:
        :param location:
        :param observation_radius:
        :param weight:
        :param value:
        """
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

        self._sense_func = self.sense_functions.get(sense_function, self._sense_regions)
        return

    def __repr__(self):
        return f'({self.name}: {self.agent_type}: {self.weight=}: {self.value=}: {self.location=})'

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
        """
        The weight of an obstacle is how many agents it requires to remove it.

        The value of an obstacle is the potential reward that can be obtained from it when it is removed.
        The simulation is expected to reduce this value by one each time an excavator collides with it.

        :param agent_id:
        :param agent_type:
        :param location:
        :param observation_radius:
        :param weight:
        :param value:
        """
        super().__init__(agent_id, agent_type, location, observation_radius, weight, value)
        return

    def sense(self, other_agents):
        # sense nearby harvester agents
        return np.asarray([0, 0])

    def get_action(self, observation):
        return np.asarray([0, 0])


class Poi(Agent):

    def __init__(self, agent_id, agent_type, location, observation_radius, weight, value):
        """
        The weight of a poi is how many agents it requires to remove it.

        The value of a poi is the potential reward that can be obtained from it when it is observed.
        The simulation is expected to reduce this value by one each time a harvester collides with it.

        :param agent_id:
        :param agent_type:
        :param location:
        :param observation_radius:
        :param weight:
        :param value:
        """
        super().__init__(agent_id, agent_type, location, observation_radius, weight, value)
        return

    def sense(self, other_agents):
        # sense nearby harvester agents
        return np.asarray([0, 0])

    def get_action(self, observation):
        return np.asarray([0, 0])
