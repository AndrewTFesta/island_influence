"""
@title

@description

"""
from enum import Enum, auto

import numpy as np
import torch
from gym.vector.utils import spaces

from island_influence.learn.neural_network import NeuralNetwork
from island_influence.utils import relative


class AgentType(Enum):
    Harvester = auto()
    Excavator = auto()
    Obstacle = auto()
    StaticPoi = auto()
    # MovingPoi = auto()


class Agent:
    NUM_BINS = len(AgentType)

    def __init__(self, agent_id: int, agent_type: AgentType, observation_radius, size: float, weight: float, value: float,
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
        :param observation_radius:
        :param size:
        :param weight:
        :param value:
        """
        self.name = f'{agent_type.name}_{agent_id}'
        self.id = agent_id
        self.agent_type = agent_type

        self.location = None

        self.observation_radius = observation_radius
        self.size = size
        self.weight = weight
        self.value = value

        self._initial_value = value
        self._initial_weight = weight

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
        str_rep = f'({self.name}: {self.agent_type}: {self.weight=}: {self.value=}: {self.location=})'
        return str_rep

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
        self.value = self._initial_value
        self.weight = self._initial_weight
        # self._initial_location = copy.copy(location)
        return

    def observable_agents(self, agent_locations, observation_radius, min_distance=0.001):
        """
        observable_agents

        :param agent_locations:
        :param observation_radius:
        :param min_distance:
        :return:
        """
        obs_bins = []
        for idx, each_agent in enumerate(agent_locations):
            if np.isnan(each_agent).any():
                continue

            agent_location = each_agent[:2]
            angle, dist = relative(self.location, agent_location)
            if min_distance <= dist <= observation_radius:
                obs_bins.append((each_agent, angle, dist))
        return obs_bins

    def _sense_regions(self, state, offset=False):
        """
        Takes in the state of the worlds and counts how many agents are in each d-hyperoctant around the agent,
        with the agent being at the center of the observation.

        Calculates which pois, leaders, and follower go into which d-hyperoctant, where d is the state
        resolution of the environment.

        first set of (sensor_resolution) bins is for leaders/followers
        second set of (sensor_resolution) bins is for pois

        :param state:
        :param offset:
        :return:
        """
        layer_obs = np.zeros((Agent.NUM_BINS, self.sensor_resolution))
        counts = np.ones(layer_obs.shape)

        # each row in each layer is a list of
        #   [locations (2d), weight, value]
        obs_agents = Agent.observable_agents(self, state, self.observation_radius)
        bin_size = 360 / self.sensor_resolution
        if offset:
            offset = 360 / (self.sensor_resolution * 2)
            bin_size = offset * 2

        for idx, entry in enumerate(obs_agents):
            agent, angle, dist = entry
            if dist == 0.0:
                dist += 0.001
            obs_value = agent[3] / dist

            agent_type_idx = int(agent[-1])
            bin_idx = int(np.floor(angle / bin_size) % self.sensor_resolution)
            layer_obs[agent_type_idx, bin_idx] += obs_value
            counts[agent_type_idx, bin_idx] += 1

        layer_obs = np.divide(layer_obs, counts)
        layer_obs = np.nan_to_num(layer_obs)
        layer_obs = layer_obs.flatten()
        return layer_obs

    def _sense_vision(self, other_agents, blur=False):
        """
        Computes an observation as the 2D rectangle surrounding the agent.

        :return:
        """
        # todo  implement sense_vision perception
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

    def __init__(self, agent_id, agent_type, observation_radius, size, weight, value):
        """
        The weight of an obstacle is how many agents it requires to remove it.

        The value of an obstacle is the potential reward that can be obtained from it when it is removed.
        The simulation is expected to reduce this value by one each time an excavator collides with it.

        :param agent_id:
        :param agent_type:
        :param observation_radius:
        :param size:
        :param weight:
        :param value:
        """
        super().__init__(agent_id, agent_type, observation_radius, size, weight, value)
        return

    def __repr__(self):
        return f'({self.name}: {self.agent_type}: {self.size=}: {self.weight=}: {self.value=}: {self.location=})'

    def sense(self, other_agents):
        # sense nearby harvester agents
        return np.asarray([0, 0])

    def get_action(self, observation):
        return np.asarray([0, 0])


class Poi(Agent):

    # @property
    # def observed(self):
    #     max_seen = 0
    #     for each_step in self.observation_history:
    #         # using value allows for different agents to contribute different weights to observing the poi
    #         curr_seen = sum(each_agent[0].value for each_agent in each_step)
    #         max_seen = max(max_seen, curr_seen)
    #     obs = max_seen >= self.coupling
    #     return obs

    def __init__(self, agent_id, agent_type, observation_radius, size, weight, value):
        """
        The weight of a poi is how many agents it requires to remove it.

        The value of a poi is the potential reward that can be obtained from it when it is observed.
        The simulation is expected to reduce this value by one each time a harvester collides with it.

        :param agent_id:
        :param agent_type:
        :param observation_radius:
        :param size:
        :param weight:
        :param value:
        """
        super().__init__(agent_id, agent_type, observation_radius, size, weight, value)
        return

    def __repr__(self):
        return f'({self.name}: {self.agent_type}: {self.size=}: {self.weight=}: {self.value=}: {self.location=})'

    def observation_space(self):
        sensor_range = spaces.Box(low=0, high=self.weight, shape=(1,))
        return sensor_range

    def action_space(self):
        # static agents do not move during an episode
        action_range = spaces.Box(low=0, high=0, shape=(2,), dtype=np.float64)
        return action_range

    # def observed_idx(self, idx):
    #     idx_obs = self.observation_history[idx]
    #     idx_seen = len(idx_obs)
    #     observed = idx_seen >= self.weight
    #     return observed

    def sense(self, other_agents):
        # sense nearby harvester agents
        return np.asarray([0, 0])

    def get_action(self, observation):
        return np.asarray([0, 0])
