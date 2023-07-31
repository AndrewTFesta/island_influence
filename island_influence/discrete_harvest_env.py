import math
import pickle
from pathlib import Path

import numpy as np
import pygame

from island_influence import project_properties
from island_influence.agent import Agent, Obstacle, Poi, AgentType
from island_influence.learn.neural_network import NeuralNetwork


class DiscreteHarvestEnv:
    metadata = {'render_modes': ['human', 'rgb_array', 'none'], 'render_fps': 4, 'name': 'DiscreteHarvestEnvironment'}

    def __init__(self, agents: list[Agent], obstacles: list[Obstacle], pois: list[Poi],
                 max_steps, delta_time=1, render_mode=None):
        self._current_step = 0
        self.max_steps = max_steps
        self.delta_time = delta_time

        # agents are the harvesters and the supports
        self.agents = {agent.name: agent for agent in agents}
        self.obstacles = {agent.name: agent for agent in obstacles}
        self.pois = {agent.name: agent for agent in pois}

        # note: state/observations history starts at len() == 1
        #       while action/reward history starts at len() == 0
        self.state_history = [self.state()]
        self.action_history = []
        self.reward_history = []

        # todo  implement a network for learning state transitions
        #       input   num_agents * (num_dimensions (2) + action_size (2))
        #       output  num_agents * num_dimensions (2)
        action_size = agents[0].action_space().shape[0]
        n_inputs = len(agents) * (action_size + len(agents[0].location))
        n_outputs = len(agents) * len(agents[0].location)
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
        self.render_bound = 100
        self.window_size = 512
        self.window = None
        self.clock = None
        return

    def state(self):
        """
        Returns the state.

        State returns a global view of the environment appropriate for centralized
        training decentralized execution methods like QMIX
        :return:
        """
        # todo  store state as a matrix in environment rather than individually in agents
        #       env is the state and agents are how the updates are calculated based on current state
        #       note that this may imply non-changing set of agents
        agent_states = [agent for name, agent in self.agents.items()]
        obstacles_states = [obstacle for name, obstacle in self.obstacles.items()]
        poi_states = [poi for name, poi in self.pois.items()]

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

    def __render_frame(self, window_size=None, render_bound=None):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        render_bound = self.render_bound if render_bound is None else render_bound
        window_size = self.window_size if window_size is None else window_size

        canvas = pygame.Surface((window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # The size of a single grid square in pixels
        pix_square_size = (window_size / render_bound)

        # todo  implement rendering
        agent_colors = {AgentType.Harvester: [0, 102, 0], AgentType.Support: [0, 0, 102], AgentType.Obstacle: [102, 51, 0], AgentType.StaticPoi: [102, 0, 0]}
        line_color = [0, 0, 0]
        default_color = [128, 128, 128]

        # draw some gridlines
        for x in range(render_bound + 1):
            pygame.draw.line(
                canvas, line_color, (0, pix_square_size * x), (window_size, pix_square_size * x), width=1,
            )
            pygame.draw.line(
                canvas, line_color, (pix_square_size * x, 0), (pix_square_size * x, window_size), width=1,
            )

        for name, agent in self.agents.items():
            acolor = agent_colors.get(agent.type, default_color)
            location = np.array(agent.location)
            pygame.draw.rect(
                canvas, acolor, pygame.Rect(pix_square_size * location, (pix_square_size, pix_square_size))
            )

        for name, agent in self.obstacles.items():
            acolor = agent_colors.get(agent.type, default_color)
            location = np.array(agent.location)
            pygame.draw.circle(canvas, acolor, (location + 0.5) * pix_square_size, pix_square_size / 1.5)

        for name, agent in self.pois.items():
            acolor = agent_colors.get(agent.type, default_color)
            location = np.array(agent.location)
            # todo  different colors to distinguish if the poi is captured
            # agent_color = acolor if agent.observed else (0, 0, 0)
            agent_color = acolor
            # todo  draw circle around poi indicating the observation radius
            pygame.draw.circle(canvas, agent_color, (location + 0.5) * pix_square_size, pix_square_size / 1.5)

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

    def __render_rgb(self):
        # todo set based on min/max agent locations
        render_resolution = (512, 512)
        render_bounds = (-5, 55)
        scaling = np.divide(render_resolution, render_bounds[1] - render_bounds[0])

        agent_colors = {AgentType.Harvester: [0, 102, 0], AgentType.Support: [0, 0, 102], AgentType.Obstacle: [102, 51, 0], AgentType.StaticPoi: [102, 0, 0]}
        agent_sizes = {AgentType.Harvester: 2, AgentType.Support: 2, AgentType.Obstacle: 3, AgentType.StaticPoi: 3}

        background_color = [255, 255, 255]
        line_color = [0, 0, 0]
        default_color = [128, 128, 128]
        default_size = 2
        num_lines = 10
        x_line_idxs = np.linspace(0, render_resolution[1], num=num_lines)
        y_line_idxs = np.linspace(0, render_resolution[0], num=num_lines)

        frame = np.full((render_resolution[0] + 1, render_resolution[1] + 1, 3), background_color)

        # draw a grid over the frame
        for each_line in x_line_idxs:
            each_line = int(each_line)
            frame[each_line] = line_color

        for each_line in y_line_idxs:
            each_line = int(each_line)
            frame[:, each_line] = line_color

        # place the agents in the frame based on the sizes and colors specified in agent_colors and agent_sizes
        for name, agent in self.agents.items():
            acolor = agent_colors.get(agent.type, default_color)
            asize = agent_sizes.get(agent.type, default_size)
            aloc = np.array(agent.location)

            scaled_loc = aloc - render_bounds[0]
            scaled_loc = np.multiply(scaled_loc, scaling)
            scaled_loc = np.rint(scaled_loc)
            scaled_loc = scaled_loc.astype(np.int)
            frame[
            scaled_loc[1] - asize: scaled_loc[1] + asize,
            scaled_loc[0] - asize: scaled_loc[0] + asize,
            ] = acolor
        frame = frame.astype(np.uint8)
        return frame

    def render(self, mode: str | None = None, **kwargs):
        """
        Displays a rendered frame from the environment, if supported.

        Alternate render modes in the default environments are ‘rgb_array’ which returns a numpy array and
        is supported by all environments outside of classic, and ‘ansi’ which returns the strings printed
        (specific to classic environments).

        :param mode:
        :param kwargs:
        :return:
        """
        if not mode:
            mode = self.render_mode

        match mode:
            case 'human':
                frame = self.__render_frame(**kwargs)
            case 'rgb_array':
                frame = self.__render_frame(**kwargs)
                # frame = self.__render_rgb()
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
        if seed is not None:
            np.random.seed(seed)

        # set agents, obstacles, and pois to the initial states
        # add all possible agents to the environment - agents are removed from the self.agents as they finish the task
        _ = [agent.reset() for name, agent in self.agents.items()]
        _ = [agent.reset() for name, agent in self.obstacles.items()]
        _ = [agent.reset() for name, agent in self.pois.items()]

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
        for name, agent in self.agents.items():
            agent_obs = agent.sense(curr_state)
            observations[name] = agent_obs
        return observations

    #
    def get_actions(self):
        """
        Returns a dictionary of actions (keyed by the agent name).

        :return:
        """
        observations = self.get_observations()
        actions = {}
        for name, agent in self.agents.items():
            agent_obs = observations[name]
            agent_action = agent.get_action(agent_obs)
            actions[name] = agent_action
        return actions

    # def observed_pois(self):
    #     observed = [poi for name, poi in self.pois.items() if poi.observed]
    #     return observed

    def done(self):
        # all_obs = len(self.observed_pois()) == len(self.pois.values())
        all_obs = len(self.pois) == 0
        time_over = self._current_step >= self.max_steps
        episode_done = any([all_obs, time_over])

        agent_dones = {name: episode_done for name, each_agent in self.agents.items()}
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
        # step leaders, followers, and pois
        # should not matter which order they are stepped in as long as dt is small enough
        for agent_name, each_action in actions.items():
            agent = self.agents[agent_name]
            # each_action[0] is dx
            # each_action[1] is dy
            new_loc = agent.location + each_action
            agent.location = new_loc

        # todo  remove an obstacle if an excavator collides with it
        # todo  reduce the value of a poi from self.pois if it is observed
        #       the reward for an agent is based on the current value of a poi
        curr_state = self.state()
        # for name, observation in observations.items():
        #     agent = self.agent_mapping[name]
        #     if isinstance(agent, Poi) and agent.observed:
        #         self.agents.remove(name)

        # Step forward and check if simulation is done
        # Update all agent dones with environment done
        self._current_step += self.delta_time
        dones = self.done()

        # The effects of the actions have all been applied to the state
        # No more state changes should occur below this point
        self.action_history.append(actions)
        curr_state = self.state()
        self.state_history.append(curr_state)

        # Update infos and truncated for agents.
        # Not sure what would go here, but it seems to be expected by pettingzoo
        infos = {name: {} for name, agent in self.agents.items()}
        truncs = {name: {} for name, agent in self.agents.items()}

        # todo  Calculate fitnesses
        rewards = {agent: 0.0 for name, agent in self.agents.items()}
        rewards['team'] = 0.0
        self.reward_history.append(rewards)

        observations = self.get_observations()
        if all(dones.values()):
            # self.completed_agents = self.possible_agents[:]
            # self.agents = []
            pass
        return observations, rewards, dones, truncs, infos
