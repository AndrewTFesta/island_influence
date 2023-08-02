import math
import pickle
from pathlib import Path

import numpy as np
import pygame

from island_influence import project_properties
from island_influence.agent import Agent, Obstacle, Poi, AgentType, closest_agent_sets
from island_influence.learn.neural_network import NeuralNetwork
from island_influence.utils import random_ring


class HarvestEnv:
    metadata = {'render_modes': ['human', 'rgb_array', 'none'], 'render_fps': 4, 'name': 'DiscreteHarvestEnvironment'}

    def __init__(self, num_agents: int, num_obstacles, num_pois, location_funcs, max_steps, delta_time=1, render_mode=None):
        self.num_dims = 2
        self.agent_action_size = 2
        self._current_step = 0
        self.max_steps = max_steps
        self.delta_time = delta_time

        # agents are the harvesters and the supports
        self.num_agents = num_agents
        self.num_obstacles = num_obstacles
        self.num_pois = num_pois

        self.agents: list | None = None
        self.obstacles: list | None = None
        self.pois: list | None = None

        self.locations = location_funcs

        # note: state/observations history starts at len() == 1
        #       while action/reward history starts at len() == 0
        self.state_history = []
        self.action_history = []
        self.reward_history = []

        # network for learning state transitions
        #   input   num_agents * (num_dimensions (2) + action_size (2))
        #   output  num_agents * num_dimensions (2)
        n_inputs = self.num_agents * (self.num_dims + self.agent_action_size)
        n_outputs = self.num_agents * self.num_dims
        n_hidden = math.ceil((n_inputs + n_outputs) / 2)
        self.env_model = NeuralNetwork(n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden)

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
        self.set_render_bounds()
        return

    def set_agents(self, agents: list[Agent]):
        assert len(agents) == self.num_agents
        self.agents = agents
        return

    def set_obstacles(self, obstacles: list[Obstacle]):
        assert len(obstacles) == self.obstacles
        self.obstacles = obstacles
        return

    def set_pois(self, pois: list[Poi]):
        assert len(pois) == self.num_pois
        self.pois = pois
        return

    def get_agent(self, agent_name: str):
        agent = None
        for each_agent in self.agents:
            if each_agent.name == agent_name:
                agent = each_agent
                break
        return agent

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
        # todo  store state as a matrix in environment rather than individually in agents
        # env is the state and agents are how the updates are calculated based on current state
        # note that this may imply non-changing set of agents
        agent_states = [agent for agent in self.agents]
        obstacles_states = [obstacle for obstacle in self.obstacles]
        poi_states = [poi for poi in self.pois]

        all_states = np.concatenate((agent_states, obstacles_states), axis=0)
        all_states = np.concatenate((all_states, poi_states), axis=0)
        return all_states

    def state_transition(self, idx):
        start_state = self.state_history[idx]
        action = self.action_history[idx]
        reward = self.reward_history[idx]
        end_state = self.state_history[idx + 1]
        return start_state, action, reward, end_state

    def all_state_transitions(self):
        state_transitions = []
        for idx in enumerate(self.state_history):
            transition = self.state_transition(idx)
            state_transitions.append(transition)
        return state_transitions

    def save_environment(self, save_dir=None, tag=''):
        # todo  use better methods of saving than pickling
        # https://docs.python.org/3/library/pickle.html#pickling-class-instances
        # https://stackoverflow.com/questions/37928794/which-is-faster-for-load-pickle-or-hdf5-in-python
        # https://marshmallow.readthedocs.io/en/stable/
        # https://developers.google.com/protocol-buffers
        # https://developers.google.com/protocol-buffers/docs/pythontutorial
        if save_dir is None:
            save_dir = project_properties.env_dir

        if tag != '':
            tag = f'_{tag}'

        save_path = Path(save_dir, f'discrete_harvest_env{tag}.pkl')
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        self.reset()
        with open(save_path, 'wb') as save_file:
            pickle.dump(self, save_file, pickle.HIGHEST_PROTOCOL)
        return save_path

    @staticmethod
    def load_environment(load_path):
        with open(load_path, 'rb') as load_file:
            env = pickle.load(load_file)
        return env

    def reset(self, seed: int | None = None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        And returns a dictionary of observations (keyed by the agent name).

        :param seed:
        """
        agent_bounds = [0, 3]
        obstacle_bounds = [5, 8]
        poi_bounds = [10, 13]

        self.agents = []
        self.agents = []
        self.agents = []

        # set agents, obstacles, and pois to the initial states
        # add all possible agents to the environment - agents are removed from the self.agents as they finish the task
        if num_agents > 0:
            agent_locs = random_ring(num_points=num_agents, center=(5, 5), min_rad=agent_bounds[0], max_rad=agent_bounds[1])
            for idx, (name, agent) in enumerate(self.agents.items()):
                agent.reset(location=agent_locs[idx])

        if num_obstacles > 0:
            obstacle_locs = random_ring(num_points=num_obstacles, center=(5, 5), min_rad=obstacle_bounds[0], max_rad=obstacle_bounds[1])
            for idx, (name, agent) in enumerate(self.obstacles.items()):
                agent.reset(location=obstacle_locs[idx])

        if num_pois > 0:
            poi_locs = random_ring(num_points=num_pois, center=(5, 5), min_rad=poi_bounds[0], max_rad=poi_bounds[1])
            for idx, (name, agent) in enumerate(self.pois.items()):
                agent.reset(location=poi_locs[idx])

        self.set_render_bounds()

        self._current_step = 0
        # clear the action, observation, and state histories
        self.state_history = [self.state_history[0]]
        self.action_history = []
        self.reward_history = []

        observations = self.get_observations()
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

    #
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
        # all_obs = len(self.observed_pois()) == len(self.pois.values())
        all_obs = len(self.pois) == 0
        time_over = self._current_step >= self.max_steps
        episode_done = any([all_obs, time_over])

        agent_dones = {agent.name: episode_done for agent in self.agents}
        return agent_dones

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
        # step location and time of all agents
        # should not matter which order they are stepped in as long as dt is small enough
        for agent_name, each_action in actions.items():
            agent = self.get_agent(agent_name)
            # each_action[0] is dx
            # each_action[1] is dy
            new_loc = agent.location + each_action
            agent.location = new_loc
        self._current_step += self.delta_time

        # apply the effects of harvesters and excavators on obstacles and pois
        # assign rewards to harvester for observing pois and excavators for removing obstacles
        rewards = {agent.name: 0.0 for agent in self.agents}

        # Compute for excavators and obstacles
        excavators = {agent.name: agent for agent in self.agents if agent.agent_type == AgentType.Excavators}
        remaining_obstacles = {agent.name: agent for agent in self.obstacles if agent.value > 0}
        closest_obstacles = closest_agent_sets(remaining_obstacles, excavators, min_dist=1)

        rewards['excavator_team'] = 0.0
        for obstacle_name, excavator_info in closest_obstacles.items():
            excavator = excavator_info[0]
            # remove an obstacle if an excavator collides with it
            # assign reward for removing the obstacle to the closest excavator
            #       the reward for an agent is based on the current value of the obstacle
            #       this value cannot exceed the remaining value of the obstacle being removed
            value_diff = min(excavator.value, self.obstacles[obstacle_name].value)
            self.obstacles[obstacle_name].value -= value_diff
            rewards[excavator.name] += value_diff
            rewards['excavator_team'] += value_diff

        # Compute for harvesters and pois
        harvesters = {agent.name: agent for agent in self.agents if agent.agent_type == AgentType.Harvester}
        remaining_pois = {agent.name: agent for agent in self.pois if agent.value > 0}
        closest_pois = closest_agent_sets(remaining_pois, harvesters, min_dist=1)

        rewards['harvest_team'] = 0.0
        for poi_name, harvester_info in closest_pois.items():
            harvester = harvester_info[0]
            # reduce the value of a poi from self.pois if it is observed
            # assign reward for observing the poi to the closest harvester
            #       the reward for an agent is based on the current value of the poi
            #       this value cannot exceed the remaining value of the poi being observed
            value_diff = min(harvester.value, self.pois[poi_name].value)
            self.pois[poi_name].value -= value_diff
            rewards[harvester.name] += value_diff
            rewards['harvest_team'] += value_diff

        # Global team reward is the sum of subteam (excavator team and harvest team) rewards
        rewards['team'] = rewards['excavator_team'] + rewards['harvest_team']

        # check if simulation is done
        dones = self.done()

        # The effects of the actions have all been applied to the state
        # No more state changes should occur below this point
        curr_state = self.state()
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

    def __render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # The size of a single grid square in pixels
        pix_square_size = (self.window_size / self.render_bound)

        agent_colors = {AgentType.Harvester: [0, 102, 0], AgentType.Excavators: [0, 0, 102], AgentType.Obstacle: [102, 51, 0], AgentType.StaticPoi: [102, 0, 0]}
        default_color = [128, 128, 128]

        agent_sizes = {AgentType.Harvester: 0.5, AgentType.Excavators: 0.5, AgentType.Obstacle: 0.25, AgentType.StaticPoi: 0.25}
        default_size = 0.1

        line_color = [0, 0, 0]
        text_color = [255, 255, 255]
        text_size = 14
        pygame.font.init()
        font = pygame.font.SysFont('arial', text_size)

        # draw some gridlines
        for x in range(self.render_bound + 1):
            pygame.draw.line(
                canvas, line_color, (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=1,
            )
            pygame.draw.line(
                canvas, line_color, (pix_square_size * x, 0), (pix_square_size * x, self.window_size), width=1,
            )

        for agent in self.agents:
            location = np.array(agent.location) + self.location_offset
            acolor = agent_colors.get(agent.agent_type, default_color)
            asize = agent_sizes.get(agent.agent_type, default_size)
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
            # different colors to distinguish how much remains of the obstacle
            acolor = agent_colors.get(agent.agent_type, default_color)
            acolor = np.divide(acolor, (agent.value + 1))
            # display text of the current value remaining of a poi
            text_surface = font.render(f'{agent.value}', False, text_color)
            canvas.blit(text_surface, (location + 0.35) * pix_square_size)
            # draw a circle at the location to represent the obstacle
            pygame.draw.circle(canvas, acolor, (location + 0.5) * pix_square_size, pix_square_size * asize)

        for agent in self.pois:
            location = np.array(agent.location) + self.location_offset
            asize = agent_sizes.get(agent.agent_type, default_size)
            # different colors to distinguish how much of the poi is observed
            acolor = agent_colors.get(agent.agent_type, default_color)
            acolor = np.divide(acolor, (agent.value + 1))
            # draw circle around poi indicating the observation radius
            # draw a circle at the location to represent the obstacle
            pygame.draw.circle(canvas, acolor, (location + 0.5) * pix_square_size, pix_square_size * asize)
            pygame.draw.circle(canvas, acolor, (location + 0.5) * pix_square_size, pix_square_size * agent.observation_radius, width=1)

        for agent in self.pois:
            location = np.array(agent.location) + self.location_offset
            # display text of the current value remaining of a poi
            # do this in a loop after to make sure it displays on all pois and agents
            text_surface = font.render(f'{agent.value}', False, text_color)
            canvas.blit(text_surface, (location + 0.3) * pix_square_size)

        if self.render_mode == 'human':
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata['render_fps'])
        else:  # rgb_array
            np_frame = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
            return np_frame
        return

    def render(self, mode: str | None = None):
        """
        Displays a rendered frame from the environment, if supported.

        Alternate render modes in the default environments are ‘rgb_array’ which returns a numpy array and
        is supported by all environments outside of classic, and ‘ansi’ which returns the strings printed
        (specific to classic environments).

        :param mode:
        :return:
        """
        if not mode:
            mode = self.render_mode

        match mode:
            case 'human':
                frame = self.__render_frame()
            case 'rgb_array':
                frame = self.__render_frame()
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
