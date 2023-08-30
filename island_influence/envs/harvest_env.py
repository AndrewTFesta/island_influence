import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gym import spaces
from scipy.spatial import distance

from island_influence.learn.neural_network import NeuralNetwork
from island_influence.utils import relative


class HarvestEnv:
    metadata = {'render_modes': ['human', 'none'], 'render_fps': 4, 'name': 'HarvestEnv'}
    REWARD_TYPES = ['global', 'difference']
    AGENT_TYPES = ['harvester', 'excavator']
    # One bin for each agent type (harvester, excavator, obstacle, poi)
    NUM_BINS = 4

    @property
    def harvesters(self):
        return self._state[self._state['agent_type'] == 0]

    @property
    def excavators(self):
        return self._state[self._state['agent_type'] == 1]

    @property
    def obstacles(self):
        return self._state[self._state['agent_type'] == 2]

    @property
    def pois(self):
        return self._state[self._state['agent_type'] == 3]

    @property
    def remaining_obstacle_value(self):
        return self.obstacles['value'].sum()

    @property
    def remaining_poi_value(self):
        return self.pois['value'].sum()

    def __init__(self, actors: dict[str, list], num_harvesters, num_excavators, num_obstacles, num_pois, location_funcs: dict, max_steps, save_dir,
                 sensor_resolution=8, observation_radius=2, delta_time=1, normalize_rewards=False, collision_penalty_scalar: float = 0, reward_type='global',
                 render_mode=None):
        """
        Always make sure to add agents and call `HarvestEnv.reset()` after before using the environment

        :param actors:
        :param num_harvesters:
        :param num_excavators:
        :param num_obstacles:
        :param num_pois:
        :param location_funcs:
        :param max_steps:
        :param delta_time:
        :param render_mode:
        """
        self._reward_type = ['global', 'difference']
        self._current_step = 0
        self.max_steps = max_steps
        self.agent_action_size = 2
        self.delta_time = delta_time
        self.sensor_resolution = sensor_resolution
        self.observation_radius = observation_radius
        self.max_velocity = 1

        self.normalize_rewards = normalize_rewards
        self.collision_penalty_scalar = collision_penalty_scalar
        self.reward_type = reward_type if reward_type in self._reward_type else 'global'
        self.save_dir = save_dir

        self.num_dims = 2
        # observation radius controls how far the agent can "see"
        # size controls how far an agent is able to interact with another object
        self._fields = ['name', 'location_0', 'location_1', 'observation_radius', 'size', 'weight', 'value', 'agent_type']
        self._info = {
            'harvester': {'observation_radius': 5, 'size': 1, 'weight': 1, 'value': 1, 'agent_type': 0},
            'excavator': {'observation_radius': 5, 'size': 1, 'weight': 1, 'value': 1, 'agent_type': 1},
            'obstacle': {'observation_radius': 1, 'size': 1, 'weight': 1, 'value': 1, 'agent_type': 2},
            'poi': {'observation_radius': 1, 'size': 1, 'weight': 1, 'value': 1, 'agent_type': 3},
        }
        self.type_map = {
            info['agent_type']: agent_type
            for agent_type, info in self._info.items()
        }
        self.types_num = {
            'harvester': num_harvesters,
            'excavator': num_excavators,
            'obstacle': num_obstacles,
            'poi': num_pois,
        }

        self._action_spaces = {
            'harvester': spaces.Box(
                low=-1 * self.max_velocity, high=self.max_velocity,
                shape=(self.num_dims,), dtype=np.float64
            ),
            'excavator': spaces.Box(
                low=-1 * self.max_velocity, high=self.max_velocity,
                shape=(self.num_dims,), dtype=np.float64
            ),
        }
        self._observation_spaces = {
            'harvester': spaces.Box(
                low=0, high=np.inf,
                shape=(self.sensor_resolution, self.NUM_BINS), dtype=np.float64
            ),
            'excavator': spaces.Box(
                low=0, high=np.inf,
                shape=(self.sensor_resolution, self.NUM_BINS), dtype=np.float64
            ),
        }
        self._actors = {
            f'{agent_type}:{idx}': each_actor
            for agent_type, type_actors in actors.items()
            for idx, each_actor in enumerate(type_actors)
        }
        self.agents = []

        self.location_funcs = location_funcs
        self.initial_obstacle_value = None
        self.initial_poi_value = None
        self._state = None

        # note: state/observations history starts at len() == 1
        #       while action/reward history starts at len() == 0
        self.state_history = []
        self.action_history = []
        self.reward_history = []

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        # The size of the PyGame window
        self.window_size = 512
        self.render_scale = 2
        self.window = None
        self.clock = None

        self.render_bound = None
        self.location_offset = None
        return

    def create_env_model(self):
        # check agent locations to make sure all are set
        # network for learning state transitions
        #   input   num_agents * (num_dimensions (2) + action_size (2))
        #   output  num_agents * num_dimensions (2)
        state = self.state()
        state_size = state.flatten().size
        n_outputs = len(state) * self.num_dims
        n_hidden = math.ceil((state_size + n_outputs) / 2)
        env_model = NeuralNetwork(n_inputs=state_size, n_outputs=n_outputs, n_hidden=n_hidden)
        return env_model

    def set_policies(self, agent_type: str, agent_policies):
        assert len(agent_policies) == self.types_num[agent_type]
        for idx, each_policy in enumerate(agent_policies):
            self._actors[f'{agent_type}:{idx}'] = each_policy
        return

    def action_space(self, agent):
        agent_state = self.get_object_state(agent)
        agent_type = agent_state.loc['agent_type']
        agent_type = self.type_map[agent_type]
        act_space = self._action_spaces[agent_type]
        return act_space

    def observation_space(self, agent):
        agent_state = self.get_object_state(agent)
        agent_type = agent_state.loc['agent_type']
        agent_type = self.type_map[agent_type]
        act_space = self._observation_spaces[agent_type]
        return act_space

    def state(self):
        """
        Returns the state.

        State returns a global view of the environment appropriate for centralized
        training decentralized execution methods like QMIX
        :return:
        """
        reduced_state = self._state.drop(columns='name')
        return reduced_state

    def state_transition(self, idx):
        transition = {
            'initial_state': self.state_history[idx],
            'action': self.action_history[idx],
            # 'reward': self.reward_history[idx],
            'end_state': self.state_history[idx + 1]
        }
        return transition

    def all_state_transitions(self):
        num_transitions = len(self.action_history)
        state_transitions = [
            self.state_transition(idx)
            for idx in range(num_transitions)
        ]
        return state_transitions

    def clear_saved_transitions(self, save_dir=None, tag=''):
        if save_dir is None:
            save_dir = self.save_dir

        if tag != '':
            tag = f'_{tag}'

        transitions_fname = Path(save_dir, f'transitions{tag}.pkl')
        transitions = []
        with open(transitions_fname, 'wb') as save_file:
            pickle.dump(transitions, save_file, pickle.HIGHEST_PROTOCOL)
        return transitions

    def save_environment(self, save_dir=None, tag=''):
        # todo  use better methods of saving than pickling
        # https://docs.python.org/3/library/pickle.html#pickling-class-instances
        # https://stackoverflow.com/questions/37928794/which-is-faster-for-load-pickle-or-hdf5-in-python
        # https://marshmallow.readthedocs.io/en/stable/
        # https://developers.google.com/protocol-buffers
        # https://developers.google.com/protocol-buffers/docs/pythontutorial
        if save_dir is None:
            save_dir = self.save_dir

        if tag != '':
            tag = f'_{tag}'

        save_path = Path(save_dir, f'harvest_env{tag}.pkl')
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        self.reset()
        with open(save_path, 'wb') as save_file:
            self.close()
            pickle.dump(self, save_file, pickle.HIGHEST_PROTOCOL)
        return save_path

    def save_transitions(self, save_dir=None, tag=''):
        if save_dir is None:
            save_dir = self.save_dir

        if tag != '':
            tag = f'_{tag}'

        transitions_fname = Path(save_dir, f'transitions{tag}.pkl')
        existing_transitions = []
        if transitions_fname.exists():
            with open(transitions_fname, 'rb') as load_file:
                existing_transitions = pickle.load(load_file)
        transitions = self.all_state_transitions()
        existing_transitions.extend(transitions)
        with open(transitions_fname, 'wb') as save_file:
            pickle.dump(existing_transitions, save_file, pickle.HIGHEST_PROTOCOL)
        return transitions, transitions_fname

    @staticmethod
    def load_environment(env_path):
        with open(env_path, 'rb') as load_file:
            env = pickle.load(load_file)
        return env

    @staticmethod
    def load_transitions(transitions_fname):
        with open(transitions_fname, 'rb') as load_file:
            transitions_fname = pickle.load(load_file)
        return transitions_fname

    def get_object_state(self, object_name):
        object_state = self._state[self._state['name'] == object_name]
        object_state = object_state.iloc[0]
        return object_state

    def get_object_location(self, object_state: pd.Series):
        agent_locations = object_state.loc[['location_0', 'location_1']]
        return agent_locations

    def gen_locs(self, seed):
        harvester_locs = self.location_funcs['harvesters'](num_points=self.types_num['harvester'], seed=seed)
        excavator_locs = self.location_funcs['excavators'](num_points=self.types_num['excavator'], seed=seed)
        obstacle_locs = self.location_funcs['obstacles'](num_points=self.types_num['obstacle'], seed=seed)
        poi_locs = self.location_funcs['pois'](num_points=self.types_num['poi'], seed=seed)
        return {'harvester': harvester_locs, 'excavator': excavator_locs, 'obstacle': obstacle_locs, 'poi': poi_locs}

    def reset(self, seed: int | None = None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        And returns a dictionary of observations (keyed by the agent name).

        :param seed:
        """
        # todo  mark all the ways this can change
        #       different number of harvesters, excavators, obstacles, pois
        #       different feature attributes
        locs = self.gen_locs(None)
        keys = ['harvester', 'excavator', 'obstacle', 'poi']
        entries = [
            {'name': f'{each_key}:{loc_idx}', **{f'location_{idx}': val for idx, val in enumerate(each_loc)}, **self._info[each_key]}
            for each_key in keys
            for loc_idx, each_loc in enumerate(locs[each_key])
        ]
        self._state = pd.DataFrame(entries)

        self.agents = self._state[self._state['name'].str.startswith('harvester') | self._state['name'].str.startswith('excavator')]
        self.agents = self.agents['name'].tolist()
        self.initial_obstacle_value = self.obstacles['value'].sum()
        self.initial_poi_value = self.pois['value'].sum()

        self._current_step = 0
        # clear the action, observation, and state histories
        self.state_history = [self._state]
        self.action_history = []
        self.reward_history = []

        self.set_render_bounds()
        observations = self.get_observations()

        self.window = None
        self.clock = None
        return observations

    def observable_agents(self, agent, observation_radius, min_distance=0.001):
        """
        observable_agents

        :param agent:
        :param observation_radius:
        :param min_distance:
        :return:
        """
        agent_loc = self.get_object_location(agent)
        obs_bins = []
        for index, each_agent_state in self._state.iterrows():
            each_loc = self.get_object_location(each_agent_state)
            if all([val == 0 for val in np.subtract(each_loc, agent_loc)]):
                continue

            angle, dist = relative(agent_loc, each_loc)
            if min_distance <= dist <= observation_radius:
                obs_bins.append((each_agent_state, angle, dist))
        return obs_bins

    def sense(self, agent, offset=False):
        """
        Takes in the state of the worlds and counts how many agents are in each d-hyperoctant around the agent,
        with the agent being at the center of the observation.

        Calculates which pois, leaders, and follower go into which d-hyperoctant, where d is the state
        resolution of the environment.

        first set of (sensor_resolution) bins is for leaders/followers
        second set of (sensor_resolution) bins is for pois

        :param agent:
        :param offset:
        :return:
        """
        layer_obs = np.zeros((self.NUM_BINS, self.sensor_resolution))
        counts = np.ones(layer_obs.shape)

        # each row in each layer is a list of
        #   [locations (2d), size, weight, value, observation_radius, agent_type]
        obs_agents = self.observable_agents(agent, self.observation_radius)
        bin_size = 360 / self.sensor_resolution
        if offset:
            offset = 360 / (self.sensor_resolution * 2)
            bin_size = offset * 2

        for idx, entry in enumerate(obs_agents):
            other_agent, angle, dist = entry
            if dist == 0.0:
                dist += 0.001

            other_type = int(other_agent.loc['agent_type'])
            obs_value = other_agent.loc['value'] / max(dist, 0.01)

            bin_idx = int(np.floor(angle / bin_size) % self.sensor_resolution)
            layer_obs[other_type, bin_idx] += obs_value
            counts[other_type, bin_idx] += 1

        layer_obs = np.divide(layer_obs, counts)
        layer_obs = np.nan_to_num(layer_obs)
        layer_obs = layer_obs.flatten()
        return layer_obs

    def get_observations(self):
        """
        Returns a dictionary of observations (keyed by the agent name).

        :return:
        """
        observations = {}
        for agent_name in self.agents:
            agent = self.get_object_state(agent_name)
            agent_obs = self.sense(agent)
            observations[agent_name] = agent_obs
        return observations

    def agent_action(self, policy, observation):
        with torch.no_grad():
            action = policy(observation)
            action = action.numpy()

        mag = np.linalg.norm(action)
        if mag > self.max_velocity:
            action = action / mag
            action *= self.max_velocity
        return action

    def get_actions(self):
        """
        Returns a dictionary of actions (keyed by the agent name).

        :return:
        """
        observations = self.get_observations()
        actions = {}
        for agent_name in self.agents:
            policy = self._actors[agent_name]
            agent_obs = observations[agent_name]
            agent_action = self.agent_action(policy, agent_obs)
            actions[agent_name] = agent_action
        return actions

    def done(self):
        all_obs = self.remaining_poi_value == 0
        time_over = self._current_step >= self.max_steps
        episode_done = any([all_obs, time_over])

        agent_dones = {agent: episode_done for agent in self.agents}
        return agent_dones

    def cumulative_rewards(self):
        cum_rewards = {each_key: 0 for each_key in self.reward_history[0]}
        for agent_name in cum_rewards:
            step_rewards = []
            for each_reward in self.reward_history:
                step_rewards.append(each_reward[agent_name])
            agent_reward = sum(step_rewards)
            cum_rewards[agent_name] = agent_reward
        return cum_rewards

    def check_collision(self, moving_objects, static_objects):
        closest = {}
        for static_idx, static_object in static_objects.iterrows():
            static_value = static_object['value']
            if static_value == 0:
                continue

            static_name = static_object['name']
            static_loc = self.get_object_location(static_object)

            all_collide = []
            for moving_idx, moving_object in moving_objects.iterrows():
                moving_loc = self.get_object_location(moving_object)

                angle, dist = relative(moving_loc, static_loc)
                if dist <= moving_object['size']:
                    entry = {'agent': moving_object, 'distance': dist}
                    all_collide.append(entry)
            if len(all_collide) > 0:
                all_collide.sort(key=lambda x: x['distance'])
                closest[static_name] = all_collide
        return closest

    def eval_rewards(self, current_state, actions, next_state):
        # todo  consider how to normalize rewards
        #       reward based on obstacle and agent weights (analogous to physical impact between objects)
        # todo  incorporate penalty for harvesters being near obstacles
        return {agent_name: 0 for agent_name in self.agents}

    def step(self, actions):
        """
        actions of each agent are always the delta x, delta y

        obs, rew, terminated, truncated, info = par_env.step(actions)

        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts keyed by the agent name
            e.g. {agent_0: observation_0, agent_1: observation_1, ...}
        """
        initial_state = self._state
        team_rewards = {'harvester': 0, 'excavator': 0}

        remaining_obstacles = self.obstacles.loc[self.obstacles['value'] != 0]
        obstacle_locations = remaining_obstacles.loc[:, remaining_obstacles.columns.str.startswith('loc')]
        obstacle_radii = remaining_obstacles.loc[:, 'size']

        # step location and time of all agents
        # should not matter which order they are stepped in as long as dt is small enough
        default_action = np.asarray([0, 0])
        for idx, agent_name in enumerate(self.agents):
            agent_action = actions.get(agent_name, default_action)
            agent_state = self.get_object_state(agent_name)
            agent_location = self.get_object_location(agent_state)
            agent_type = agent_state.loc['agent_type']
            agent_type = self.type_map[agent_type]

            # check if agent is currently near any obstacles
            # do not restrict movement, instead, assign a reward based on the proximity of obstacles to harvesters
            #   possibly movement of agents in regions that are near obstacles is slower
            if agent_type == 'harvester' and len(remaining_obstacles) > 0:
                obstacle_dists = distance.cdist([agent_location.tolist()], obstacle_locations.to_numpy())[0]
                obstacle_dists -= obstacle_radii
                obstacle_dists = np.clip(obstacle_dists, 0, None)
                closest_dist = obstacle_dists.min()
                if closest_dist <= agent_state['size']:
                    agent_action /= 2
                    # penalty for being in a hazardous region based on the value of the closest obstacle
                    arg_closest = np.argmin(obstacle_dists)
                    closest_obstacle = remaining_obstacles.iloc[arg_closest]
                    each_reward = closest_obstacle['value']
                    each_reward *= self.collision_penalty_scalar
                    if self.normalize_rewards:
                        each_reward = each_reward / self.initial_obstacle_value
                    # todo  implement negative difference rewards for penalty
                    # todo  test difference rewards for accuracy
                    if self.reward_type == 'difference':
                        pass
                    team_rewards['harvester'] -= each_reward
            new_loc = agent_location + agent_action
            self._state.loc[self._state['name'] == agent_name, 'location_0'], self._state.loc[self._state['name'] == agent_name, 'location_1'] = new_loc

        # find the closest pairs of relevant agents after excavators and harvesters have moved
        # check closest agents based on observation radii of actors rather than stationary objects
        observed_obstacles_excavators = self.check_collision(self.excavators, self.obstacles)
        observed_pois_harvesters = self.check_collision(self.harvesters, self.pois)

        # apply the effects of harvesters and excavators on obstacles and pois
        # assign rewards to harvester for observing pois and excavators for removing obstacles
        # Compute for excavators and obstacles
        for obstacle_name, excavator_info in observed_obstacles_excavators.items():
            # need at least N excavators in excavator_info before reducing the obstacle's value and assigning rewards to any of the excavators
            obstacle = self.get_object_state(obstacle_name)
            coupling_req = int(obstacle['size'])
            if len(excavator_info) < coupling_req:
                continue

            # the reward for removing an obstacle is based on the current value of the obstacle
            each_reward = obstacle['value']
            if self.normalize_rewards:
                each_reward = each_reward / self.initial_obstacle_value
            # todo  consider if an obstacle takes longer to fully remove
            self._state.loc[self._state['name'] == obstacle_name, 'value'] = 0

            reward = each_reward
            if self.reward_type == 'difference':
                # look at the number of nearby excavators vs the N closest
                # if there is at least one additional excavator, then this
                # agent did not need to be present to remove the obstacle
                # todo  Note that in this case, the difference rewards are symmetric
                # todo  correlate the reward to a specific agent
                obs_excavators = excavator_info[:coupling_req]
                num_extra = len(obs_excavators) - len(excavator_info)
                reward = min(-num_extra + 1, 1)
            team_rewards['excavator'] += reward

        # Compute for harvesters and pois
        for poi_name, harvester_info in observed_pois_harvesters.items():
            # need at least N harvesters in harvester_info before reducing the obstacle's value and assigning rewards to any of the excavators
            poi = self.get_object_state(poi_name)
            coupling_req = int(poi['size'])
            if len(harvester_info) < coupling_req:
                continue

            # the reward for observing a poi is based on the current value of the obstacle
            each_reward = poi['value']
            if self.normalize_rewards:
                each_reward = each_reward / self.initial_poi_value
            # todo  consider if a poi takes longer to fully remove
            self._state.loc[self._state['name'] == poi_name, 'value'] = 0

            reward = each_reward
            if self.reward_type == 'difference':
                # look at the number of nearby harvesters vs the N closest
                # if there is at least one additional harvester, then this
                # agent did not need to be present to observe the poi
                # todo  Note that in this case, the difference rewards are symmetric
                # todo  correlate the reward to a specific agent
                obs_excavators = harvester_info[:coupling_req]
                num_extra = len(obs_excavators) - len(harvester_info)
                reward = min(-num_extra + 1, 1)
            team_rewards['harvester'] += reward

        self._current_step += self.delta_time
        # check if simulation is done
        dones = self.done()
        #############################################################################################
        # The effects of the actions have all been applied to the state
        # No more state changes should occur below this point
        # Global team reward is the sum of subteam (excavator team and harvest team) rewards
        curr_state = self._state
        eval_ = self.eval_rewards(initial_state, actions, curr_state)
        rewards = {
            agent_name: team_rewards['harvester'] if agent_name.startswith('harvester') else team_rewards['excavator']
            for agent_name in self.agents
        }
        rewards['global_harvester'] = team_rewards['harvester']
        rewards['global_excavator'] = team_rewards['excavator']

        self.action_history.append(actions)
        self.state_history.append(curr_state)
        self.reward_history.append(rewards)

        # Update infos and truncated for agents.
        # Not sure what would go here, but it seems to be expected by pettingzoo
        infos = {agent_name: {} for agent_name in self.agents}
        truncs = {agent_name: {} for agent_name in self.agents}

        observations = self.get_observations()
        return observations, rewards, dones, truncs, infos

    def set_render_bounds(self):
        agent_locations = self._state.loc[:, self._state.columns.str.startswith('loc')]

        min_loc = np.min(agent_locations, axis=0)
        max_loc = np.max(agent_locations, axis=0)

        delta_x = max_loc[0] - min_loc[0]
        delta_y = max_loc[1] - min_loc[1]

        self.render_bound = math.ceil(max(delta_x, delta_y))
        self.render_bound = self.render_bound * self.render_scale
        self.location_offset = self.render_bound // (self.render_scale * 2)
        return

    def __render_human(self):
        import pygame

        black = (0, 0, 0)
        white = (255, 255, 255)

        # The size of a single grid square in pixels
        pix_square_size = (self.window_size / self.render_bound)

        agent_colors = {'harvester': (0, 255, 0), 'excavator': (0, 0, 255), 'obstacle': black, 'poi': (255, 0, 0)}
        default_color = (128, 128, 128)

        agent_sizes = {'harvester': 0.25, 'excavator': 0.25, 'obstacle': 0.25, 'poi': 0.25}
        default_size = 0.1
        size_scalar = 1
        size_width = 2
        obs_width = 1

        text_size = 14
        write_values = False
        write_legend = True

        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(white)
        pygame.font.init()
        font = pygame.font.SysFont('arial', text_size)

        for agent in self.agents:
            agent = self.get_object_state(agent)
            agent_type = agent['agent_type']
            obs_radius = agent['observation_radius']
            size_radius = agent['size']
            agent_type = self.type_map[agent_type]
            location = self.get_object_location(agent).to_numpy()
            location = location + self.location_offset
            acolor = agent_colors.get(agent_type, default_color)
            asize = agent_sizes.get(agent_type, default_size)
            asize *= size_scalar

            pygame.draw.circle(canvas, acolor, (location + 0.5) * pix_square_size, pix_square_size * size_radius, width=size_width)
            pygame.draw.circle(canvas, black, (location + 0.5) * pix_square_size, pix_square_size * asize)
            pygame.draw.circle(canvas, acolor, (location + 0.5) * pix_square_size, pix_square_size * obs_radius, width=obs_width)

        for idx, agent in self.obstacles.iterrows():
            agent_value = agent['value']

            # only render the obstacle if it has not been removed by an excavator
            if agent_value <= 0:
                continue

            agent_type = agent['agent_type']
            agent_type = self.type_map[agent_type]
            obs_radius = agent['observation_radius']
            size_radius = agent['size']
            location = self.get_object_location(agent).to_numpy()
            location = location + self.location_offset
            asize = agent_sizes.get(agent_type, default_size)
            asize *= size_scalar

            # different colors to distinguish how much remains of the obstacle
            acolor = agent_colors.get(agent_type, default_color)
            acolor = np.divide(acolor, (agent_value + 1))

            # draw a circle at the location to represent the obstacle
            pygame.draw.circle(canvas, acolor, (location + 0.5) * pix_square_size, pix_square_size * size_radius, width=size_width)
            pygame.draw.circle(canvas, black, (location + 0.5) * pix_square_size, pix_square_size * asize)
            pygame.draw.circle(canvas, acolor, (location + 0.5) * pix_square_size, pix_square_size * obs_radius, width=obs_width)

        for idx, agent in self.pois.iterrows():
            agent_type = agent['agent_type']
            agent_type = self.type_map[agent_type]
            obs_radius = agent['observation_radius']
            size_radius = agent['size']
            agent_value = agent['value']
            location = self.get_object_location(agent).to_numpy()
            location = location + self.location_offset
            asize = agent_sizes.get(agent_type, default_size)
            asize *= size_scalar

            # different colors to distinguish how much of the poi is observed
            acolor = agent_colors.get(agent_type, default_color)
            acolor = np.divide(acolor, (agent_value + 1))
            # draw circle around poi indicating the observation radius
            # draw a circle at the location to represent the obstacle
            pygame.draw.circle(canvas, acolor, (location + 0.5) * pix_square_size, pix_square_size * size_radius, width=size_width)
            pygame.draw.circle(canvas, black, (location + 0.5) * pix_square_size, pix_square_size * asize)
            pygame.draw.circle(canvas, acolor, (location + 0.5) * pix_square_size, pix_square_size * obs_radius, width=obs_width)

        if write_values:
            for agent in self.agents:
                agent_value = agent['value']
                location = self.get_object_location(agent).to_numpy()
                location = location + self.location_offset
                # display text of the current value of an agent
                # do this in a loop after to make sure it displays on all pois, obstacles, and agents
                text_surface = font.render(f'{agent_value}', False, white)
                canvas.blit(text_surface, (location + 0.35) * pix_square_size)

            for agent in self.obstacles:
                agent_value = agent['value']
                location = self.get_object_location(agent).to_numpy()
                location = location + self.location_offset
                # display text of the current value of an obstacle
                # do this in a loop after to make sure it displays on all pois, obstacles, and agents
                text_surface = font.render(f'{agent_value}', False, white)
                canvas.blit(text_surface, (location + 0.35) * pix_square_size)

            for agent in self.pois:
                agent_value = agent['value']
                location = self.get_object_location(agent).to_numpy()
                location = location + self.location_offset
                # display text of the current value of a poi
                # do this in a loop after to make sure it displays on all pois, obstacles, and agents
                text_surface = font.render(f'{agent_value}', False, white)
                canvas.blit(text_surface, (location + 0.3) * pix_square_size)

        if write_legend:
            outline = 1
            legend_offset = 10
            for idx, (agent_type, color) in enumerate(agent_colors.items()):
                text = f'{agent_type}'
                text_surface = font.render(text, False, color)
                if outline > 0:
                    outline_surface = font.render(text, True, black)
                    outline_size = outline_surface.get_size()

                    text_surface = pygame.Surface((outline_size[0] + outline * 2, outline_size[1] + 2 * outline))
                    text_surface.fill(white)
                    text_rect = text_surface.get_rect()
                    offsets = [
                        (ox, oy)
                        # for ox in range(-outline, 2 * outline, outline)
                        # for oy in range(-outline, 2 * outline, outline)
                        for ox in range(-outline, outline, outline)
                        for oy in range(-outline, outline, outline)
                        if ox != 0 or ox != 0
                    ]
                    for ox, oy in offsets:
                        px, py = text_rect.center
                        text_surface.blit(outline_surface, outline_surface.get_rect(center=(px + ox, py + oy)))

                    inner_text = font.render(text, True, color).convert_alpha()
                    text_surface.blit(inner_text, inner_text.get_rect(center=text_rect.center))

                text_height = text_surface.get_height() + legend_offset
                text_width = text_surface.get_width() + legend_offset
                # noinspection PyTypeChecker
                canvas.blit(text_surface, (self.window_size - text_width, text_height * idx))

        if self.render_mode == 'human':
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata['render_fps'])
        return

    def render(self):
        """
        Displays a rendered frame from the environment, if supported.

        Alternate render modes in the default environments are ‘rgb_array’ which returns a numpy array and
        is supported by all environments outside of classic, and ‘ansi’ which returns the strings printed
        (specific to classic environments).

        :return:
        """
        frame = None
        match self.render_mode:
            case 'human':
                self.__render_human()
            case _:
                frame = None
        return frame

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections or any other
        environment data which should not be kept around after the user is no longer using the environment.
        """
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
        return
