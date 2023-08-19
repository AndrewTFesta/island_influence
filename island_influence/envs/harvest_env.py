import math
import pickle
from pathlib import Path

import numpy as np
import pygame

from island_influence.agent import Agent, Obstacle, Poi, AgentType
from island_influence.learn.neural_network import NeuralNetwork
from island_influence.utils import relative


class HarvestEnv:
    metadata = {'render_modes': ['human', 'none'], 'render_fps': 4, 'name': 'DiscreteHarvestEnvironment'}
    REWARD_TYPES = ['global', 'difference']

    @property
    def entities(self):
        entities = []
        entities.extend(self.harvesters)
        entities.extend(self.excavators)
        entities.extend(self.obstacles)
        entities.extend(self.pois)
        return entities

    @property
    def agents(self):
        all_agents = []
        all_agents.extend(self.harvesters)
        all_agents.extend(self.excavators)
        return all_agents

    @property
    def harvesters(self):
        return self._harvesters

    @property
    def excavators(self):
        return self._excavators

    @property
    def obstacles(self):
        return self._obstacles

    @property
    def pois(self):
        return self._pois

    @harvesters.setter
    def harvesters(self, new_harvesters):
        self._harvesters = new_harvesters
        return

    @excavators.setter
    def excavators(self, new_excavators):
        self._obstacles = new_excavators
        return

    @obstacles.setter
    def obstacles(self, new_obstacles):
        self._obstacles = new_obstacles
        return

    @pois.setter
    def pois(self, new_pois):
        self._pois = new_pois
        return

    @property
    def remaining_obstacle_value(self):
        return sum([each_obstacle.value for each_obstacle in self.obstacles])

    @property
    def remaining_poi_value(self):
        return sum([each_poi.value for each_poi in self.pois])

    def __init__(self, harvesters: list[Agent], excavators: list[Agent], obstacles: list[Obstacle], pois: list[Poi], location_funcs: dict, max_steps,
                 save_dir, delta_time=1, normalize_rewards=False, collision_penalty_scalar: int = 0, reward_type='global', render_mode=None):
        """
        Always make sure to add agents and call `HarvestEnv.reset()` after before using the environment

        :param harvesters:
        :param excavators:
        :param obstacles:
        :param pois:
        :param location_funcs:
        :param max_steps:
        :param delta_time:
        :param render_mode:
        """
        self._reward_type = ['global', 'difference']
        self._current_step = 0
        self.max_steps = max_steps
        self.num_dims = 2
        self.agent_action_size = 2
        self.delta_time = delta_time

        self.normalize_rewards = normalize_rewards
        self.collision_penalty_scalar = collision_penalty_scalar
        self.reward_type = reward_type if reward_type in self._reward_type else 'global'
        self.save_dir = save_dir

        # agents are the harvesters and the excavators
        # todo  add more types of harvesters
        # todo  add more types of supports
        self._harvesters = harvesters
        self._excavators = excavators
        self._obstacles = obstacles
        self._pois = pois

        self.type_mapping = {
            AgentType.Harvester: self.harvesters,
            AgentType.Excavator: self.excavators,
            AgentType.Obstacle: self.obstacles,
            AgentType.StaticPoi: self.pois,
        }

        self.location_funcs = location_funcs
        self.initial_obstacle_value = sum([each_obstacle.value for each_obstacle in self.obstacles])
        self.initial_poi_value = sum([each_poi.value for each_poi in self.pois])

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
        locations_set = True
        all_agents = self.agents
        all_agents.extend(self.obstacles)
        all_agents.extend(self.pois)
        for each_agent in all_agents:
            agent_loc = each_agent.location
            if agent_loc is None:
                locations_set = False
                break
        if not locations_set:
            self.reset()
        # network for learning state transitions
        #   input   num_agents * (num_dimensions (2) + action_size (2))
        #   output  num_agents * num_dimensions (2)
        state = self.state()
        state_size = state.flatten().size
        n_outputs = len(self.agents) * self.num_dims
        n_hidden = math.ceil((state_size + n_outputs) / 2)
        env_model = NeuralNetwork(n_inputs=state_size, n_outputs=n_outputs, n_hidden=n_hidden)
        return env_model

    def num_agent_types(self, agent_type: AgentType | str):
        if isinstance(agent_type, str):
            agent_name = agent_type.split('.')[-1]
            agent_type = AgentType[agent_name]

        agents = self.type_mapping[agent_type]
        num_agents = len(agents)
        return num_agents

    def set_policies(self, agent_type: AgentType | str, agent_policies):
        if isinstance(agent_type, str):
            agent_name = agent_type.split('.')[-1]
            agent_type = AgentType[agent_name]

        agents = self.type_mapping[agent_type]
        assert len(agents) == len(agent_policies)
        for each_agent, policy in zip(agents, agent_policies):
            each_agent.policy = policy
        return

    def get_agent(self, agent_name: str):
        agent = None
        for each_agent in self.agents:
            if each_agent.name == agent_name:
                agent = each_agent
                break
        return agent

    def get_harvester(self, harvester_name: str):
        harvester = None
        for each_harvester in self.harvesters:
            if each_harvester.name == harvester_name:
                harvester = each_harvester
                break
        return harvester

    def get_excavator(self, excavator_name: str):
        excavator = None
        for each_excavator in self.excavators:
            if each_excavator.name == excavator_name:
                excavator = each_excavator
                break
        return excavator

    def get_obstacle(self, obstacle_name: str):
        obstacle = None
        for each_obstacle in self.obstacles:
            if each_obstacle.name == obstacle_name:
                obstacle = each_obstacle
                break
        return obstacle

    def get_poi(self, poi_name: str):
        poi = None
        for each_poi in self.pois:
            if each_poi.name == poi_name:
                poi = each_poi
                break
        return poi

    def action_space(self, agent: Agent | str):
        if isinstance(agent, str):
            agent = self.get_agent(agent)
        act_space = agent.action_space()
        return act_space

    def observation_space(self, agent: Agent | str):
        if isinstance(agent, str):
            agent = self.get_agent(agent)
        obs_space = agent.observation_space()
        return obs_space

    def state(self):
        """
        Returns the state.

        State returns a global view of the environment appropriate for centralized
        training decentralized execution methods like QMIX
        :return:
        """
        # agent_types = list(AgentType)
        # all_agents = self.entities
        # env is the state and agents are how the updates are calculated based on current state
        # note that this may imply non-changing set of agents
        # agent_states = [
        #     [*agent.location, agent.size, agent.weight, agent.value, agent.observation_radius, agent_types.index(agent.agent_type)]
        #     for agent in all_agents
        # ]
        # states_lists = [states for states in agent_states if len(states) > 0]
        # all_states = np.vstack(states_lists)

        all_states = self.entities
        return all_states

    def state_transition(self, idx):
        start_state = self.state_history[idx]
        action = self.action_history[idx]
        reward = self.reward_history[idx]
        end_state = self.state_history[idx + 1]
        transition = {
            'initial_state': start_state,
            'action': action,
            # 'reward': reward,
            'end_state': end_state
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

    def reset(self, seed: int | None = None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        And returns a dictionary of observations (keyed by the agent name).

        :param seed:
        """
        if len(self.harvesters) > 0:
            harvester_locs = self.location_funcs['harvesters'](num_points=len(self.harvesters), seed=seed)
            for idx, agent in enumerate(self.harvesters):
                agent.reset()
                agent.location = harvester_locs[idx]

        if len(self.excavators) > 0:
            excavator_locs = self.location_funcs['excavators'](num_points=len(self.excavators), seed=seed)
            for idx, agent in enumerate(self.excavators):
                agent.reset()
                agent.location = excavator_locs[idx]

        if len(self.obstacles) > 0:
            obstacle_locs = self.location_funcs['obstacles'](num_points=len(self.obstacles), seed=seed)
            for idx, agent in enumerate(self.obstacles):
                agent.reset()
                agent.location = obstacle_locs[idx]

        if len(self.pois) > 0:
            poi_locs = self.location_funcs['pois'](num_points=len(self.pois), seed=seed)
            for idx, agent in enumerate(self.pois):
                agent.reset()
                agent.location = poi_locs[idx]

        self._current_step = 0
        # clear the action, observation, and state histories
        self.state_history = [self.state()]
        self.action_history = []
        self.reward_history = []

        self.set_render_bounds()
        observations = self.get_observations()

        self.window = None
        self.clock = None
        return observations

    def get_observations(self):
        """
        Returns a dictionary of observations (keyed by the agent name).

        :return:
        """
        curr_state = self.state()
        observations = {}
        for agent in self.agents:
            agent_obs = agent.sense(curr_state)
            observations[agent.name] = agent_obs
        return observations

    def get_actions(self):
        """
        Returns a dictionary of actions (keyed by the agent name).

        :return:
        """
        observations = self.get_observations()
        actions = {}
        for agent in self.agents:
            agent_obs = observations[agent.name]
            agent_action = agent.get_action(agent_obs)
            actions[agent.name] = agent_action
        return actions

    def done(self):
        all_obs = self.remaining_poi_value == 0
        time_over = self._current_step >= self.max_steps
        episode_done = any([all_obs, time_over])

        agent_dones = {agent.name: episode_done for agent in self.agents}
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

    @staticmethod
    def check_observed(observing_agents, observed_agents):
        closest = {}
        for observed_agent in observed_agents:
            observed_location = observed_agent.location
            all_observed = []
            for observing_agent in observing_agents:
                observing_location = observing_agent.location
                angle, dist = relative(observing_location, observed_location)
                if dist <= observing_agent.observation_radius:
                    entry = {'agent': observing_agent, 'distance': dist}
                    all_observed.append(entry)
            if len(all_observed) > 0:
                all_observed.sort(key=lambda x: x['distance'])
                closest[observed_agent] = all_observed
        return closest

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
        remaining_obstacles = [agent for agent in self.obstacles if agent.value > 0]
        remaining_pois = [agent for agent in self.pois if agent.value > 0]

        rewards = {agent.name: 0.0 for agent in self.agents}
        rewards['global_excavator'] = 0.0
        rewards['global_harvester'] = 0.0

        # step location and time of all agents
        # should not matter which order they are stepped in as long as dt is small enough
        obstacle_locations = np.asarray([each_obstacle.location for each_obstacle in remaining_obstacles])
        obstacle_radii = np.asarray([each_obstacle.observation_radius for each_obstacle in remaining_obstacles])
        # for agent_name, each_action in actions.items():
        default_action = np.asarray([0, 0])
        for agent in self.agents:
            agent_action = actions.get(agent.name, default_action)
            new_loc = agent.location + agent_action
            if len(obstacle_locations) > 0 and agent.agent_type == AgentType.Harvester:
                # Collision detection for obstacles and harvesters
                new_loc_mat = np.tile(new_loc, (len(remaining_obstacles), 1))
                obstacle_dists = np.linalg.norm(new_loc_mat - obstacle_locations, axis=1)
                obstacle_dists -= obstacle_radii
                collision = np.min(obstacle_dists) <= 0
                if collision:
                    new_loc = agent.location
                    colliding_obstacle_idx = np.argmin(obstacle_dists)
                    colliding_obstacle = remaining_obstacles[colliding_obstacle_idx]
                    obstacle_weight = colliding_obstacle.weight
                    # todo  consider how to normalize rewards
                    #       reward based on obstacle and agent weights (analogous to physical impact between objects)
                    # todo  differentiate between reward sources - positive and negative
                    collision_reward = agent.weight * obstacle_weight
                    collision_reward *= -1
                    collision_reward *= self.collision_penalty_scalar
                    rewards[agent.name] += collision_reward
            agent.location = new_loc

        self._current_step += self.delta_time

        # find the closest pairs of relevant agents after stepping the environment
        # check closest agents based on observation radii of active agents rather than passive agents
        observed_obstacles_excavators = self.check_observed(self.excavators, remaining_obstacles)
        observed_pois_harvesters = self.check_observed(self.harvesters, remaining_pois)
        # apply the effects of harvesters and excavators on obstacles and pois
        # assign rewards to harvester for observing pois and excavators for removing obstacles
        # Compute for excavators and obstacles
        for obstacle, excavator_info in observed_obstacles_excavators.items():
            # need at least N excavators in excavator_info before reducing the obstacle's value and assigning rewards to any of the excavators
            coupling_req = obstacle.size
            if len(excavator_info) < coupling_req:
                continue

            # the reward for removing an obstacle is based on the current value of the obstacle
            each_reward = obstacle.value
            if self.normalize_rewards:
                each_reward = each_reward / self.initial_obstacle_value
            # todo  consider if an obstacle takes longer to fully remove
            obstacle.value = 0
            rewards['global_excavator'] += each_reward

            obs_excavators = excavator_info[:coupling_req]
            for each_info in obs_excavators:
                excavator = each_info['agent']
                reward = 0
                if self.reward_type == 'global':
                    reward = each_reward
                elif self.reward_type == 'difference':
                    # look at the number of nearby excavators vs the N closest
                    # if there is at least one additional excavator, then this
                    # agent did not need to be present to remove the obstacle
                    num_extra = len(obs_excavators) - len(excavator_info)
                    reward = min(-num_extra + 1, 1)
                rewards[excavator.name] += reward

        # Compute for harvesters and pois
        for poi, harvester_info in observed_pois_harvesters.items():
            # need at least N harvesters in harvester_info before reducing the obstacle's value and assigning rewards to any of the excavators
            coupling_req = poi.size
            if len(harvester_info) < coupling_req:
                continue

            # the reward for removing an obstacle is based on the current value of the obstacle
            each_reward = poi.value
            if self.normalize_rewards:
                each_reward = each_reward / self.initial_poi_value
            # todo  consider if a poi takes longer to fully remove
            poi.value = 0
            rewards['global_harvester'] += each_reward

            obs_harvesters = harvester_info[:coupling_req]
            for each_info in obs_harvesters:
                harvester = each_info['agent']
                reward = 0
                if self.reward_type == 'global':
                    reward = each_reward
                elif self.reward_type == 'difference':
                    # look at the number of nearby harvesters vs the N closest
                    # if there is at least one additional harvester, then this
                    # agent did not need to be present to observe the poi
                    # todo  Note that in this case, the difference rewards are symmetric
                    num_extra = len(obs_harvesters) - len(harvester_info)
                    reward = min(-num_extra + 1, 1)
                rewards[harvester.name] += reward

        # check if simulation is done
        dones = self.done()
        #############################################################################################
        # The effects of the actions have all been applied to the state
        # No more state changes should occur below this point
        # Global team reward is the sum of subteam (excavator team and harvest team) rewards
        curr_state = self.state()
        rewards['global_excavator'] = sum([each_reward for each_agent, each_reward in rewards.items() if each_agent.startswith('Excavator')])
        rewards['global_harvester'] = sum([each_reward for each_agent, each_reward in rewards.items() if each_agent.startswith('Harvester')])
        rewards['global'] = rewards['global_excavator'] + rewards['global_harvester']

        self.action_history.append(actions)
        self.state_history.append(curr_state)
        self.reward_history.append(rewards)

        # Update infos and truncated for agents.
        # Not sure what would go here, but it seems to be expected by pettingzoo
        infos = {agent.name: {} for agent in self.agents}
        truncs = {agent.name: {} for agent in self.agents}

        observations = self.get_observations()
        return observations, rewards, dones, truncs, infos

    def set_render_bounds(self):
        agent_locations = np.asarray([agent.location for agent in self.agents])
        obstacle_locations = np.asarray([agent.location for agent in self.obstacles])
        poi_locations = np.asarray([agent.location for agent in self.pois])

        if len(obstacle_locations) > 0:
            agent_locations = np.concatenate((agent_locations, obstacle_locations), axis=0)
        if len(poi_locations) > 0:
            agent_locations = np.concatenate((agent_locations, poi_locations), axis=0)

        min_loc = np.min(agent_locations, axis=0)
        max_loc = np.max(agent_locations, axis=0)

        delta_x = max_loc[0] - min_loc[0]
        delta_y = max_loc[1] - min_loc[1]

        self.render_bound = math.ceil(max(delta_x, delta_y))
        self.render_bound = self.render_bound * self.render_scale
        self.location_offset = self.render_bound // (self.render_scale * 2)
        return

    def __render_human(self):
        black = (0, 0, 0)
        white = (255, 255, 255)

        # The size of a single grid square in pixels
        pix_square_size = (self.window_size / self.render_bound)

        agent_colors = {AgentType.Harvester: (0, 255, 0), AgentType.Excavator: (0, 0, 255), AgentType.Obstacle: black, AgentType.StaticPoi: (255, 0, 0)}
        default_color = (128, 128, 128)

        agent_sizes = {AgentType.Harvester: 0.5, AgentType.Excavator: 0.5, AgentType.Obstacle: 0.25, AgentType.StaticPoi: 0.25}
        default_size = 0.1
        size_scalar = 2

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
            location = np.array(agent.location) + self.location_offset
            acolor = agent_colors.get(agent.agent_type, default_color)
            asize = agent_sizes.get(agent.agent_type, default_size)
            asize *= size_scalar

            pygame.draw.rect(
                canvas, acolor, pygame.Rect(
                    pix_square_size * location, (pix_square_size * asize, pix_square_size * asize)
                )
            )

        for agent in self.obstacles:
            # only render the obstacle if it has not been removed by an excavator
            if agent.value <= 0:
                continue

            location = np.array(agent.location) + self.location_offset
            asize = agent_sizes.get(agent.agent_type, default_size)
            asize *= size_scalar

            # different colors to distinguish how much remains of the obstacle
            acolor = agent_colors.get(agent.agent_type, default_color)
            acolor = np.divide(acolor, (agent.value + 1))

            # draw a circle at the location to represent the obstacle
            pygame.draw.circle(canvas, acolor, (location + 0.5) * pix_square_size, pix_square_size * asize)
            pygame.draw.circle(canvas, acolor, (location + 0.5) * pix_square_size, pix_square_size * agent.observation_radius, width=1)

        for agent in self.pois:
            location = np.array(agent.location) + self.location_offset
            asize = agent_sizes.get(agent.agent_type, default_size)
            asize *= size_scalar

            # different colors to distinguish how much of the poi is observed
            acolor = agent_colors.get(agent.agent_type, default_color)
            acolor = np.divide(acolor, (agent.value + 1))
            # draw circle around poi indicating the observation radius
            # draw a circle at the location to represent the obstacle
            pygame.draw.circle(canvas, black, (location + 0.5) * pix_square_size, pix_square_size * asize)
            pygame.draw.circle(canvas, acolor, (location + 0.5) * pix_square_size, pix_square_size * asize * 0.75)
            pygame.draw.circle(canvas, acolor, (location + 0.5) * pix_square_size, pix_square_size * agent.observation_radius, width=1)

        if write_values:
            for agent in self.agents:
                location = np.array(agent.location) + self.location_offset
                # display text of the current value of an agent
                # do this in a loop after to make sure it displays on all pois, obstacles, and agents
                text_surface = font.render(f'{agent.value}', False, white)
                canvas.blit(text_surface, (location + 0.35) * pix_square_size)

            for agent in self.obstacles:
                location = np.array(agent.location) + self.location_offset
                # display text of the current value of an obstacle
                # do this in a loop after to make sure it displays on all pois, obstacles, and agents
                text_surface = font.render(f'{agent.value}', False, white)
                canvas.blit(text_surface, (location + 0.35) * pix_square_size)

            for agent in self.pois:
                location = np.array(agent.location) + self.location_offset
                # display text of the current value of a poi
                # do this in a loop after to make sure it displays on all pois, obstacles, and agents
                text_surface = font.render(f'{agent.value}', False, white)
                canvas.blit(text_surface, (location + 0.3) * pix_square_size)

        if write_legend:
            outline = 1
            legend_offset = 10
            for idx, (agent_type, color) in enumerate(agent_colors.items()):
                text = f'{agent_type.name}'
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
            pygame.display.quit()
            pygame.quit()
        return