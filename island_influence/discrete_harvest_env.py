import math
import pickle
from pathlib import Path

import numpy as np

from island_influence import project_properties
from island_influence.agent import Agent, Obstacle, Poi
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
