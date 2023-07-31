import pickle
from pathlib import Path

import numpy as np

from island_influence.agent import Poi, Agent, Obstacle


class DiscreteHarvestEnv:
    metadata = {'render_modes': ['human', 'rgb_array', 'none'], 'render_fps': 4, 'name': 'DiscreteHarvestEnvironment'}

    @property
    def all_entities(self):
        return self.agents | self.obstacles | self.pois

    def __init__(self, agents: list[tuple[Agent, np.ndarray]], obstacles: list[tuple[Obstacle, np.ndarray]], pois: list[tuple[Poi, np.ndarray]],
                 max_steps, delta_time=1, render_mode=None):
        self._current_step = 0
        self.max_steps = max_steps
        self.delta_time = delta_time

        # note: state/observations history starts at len() == 1
        #       while action/reward history starts at len() == 0
        # keep direct record of state history so it's faster to compute the update
        # agents are the harvesters and the supports
        self.agents = {
            agent.name: {
                'agent': agent,
                'states': [state],
                'actions': [],
                'rewards': [],
            } for agent, state in agents
        }
        self.obstacles = {
            agent.name: {
                'agent': agent,
                'states': [state],
                'actions': [],
                'rewards': [],
            } for agent, state in obstacles
        }
        self.pois = {
            agent.name: {
                'agent': agent,
                'states': [state],
                'actions': [],
                'rewards': [],
            } for agent, state in pois
        }
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
        agent_states = [agent['states'][-1] for name, agent in self.agents]
        obstacles_states = [obstacle['states'][-1] for name, obstacle in self.obstacles]
        poi_states = [poi['states'][-1] for name, poi in self.pois]

        all_states = np.concatenate((agent_states, obstacles_states), axis=0)
        all_states = np.concatenate((all_states, poi_states), axis=0)
        return all_states

    def save_environment(self, base_dir, tag=''):
        # todo  use better methods of saving than pickling
        # https://docs.python.org/3/library/pickle.html#pickling-class-instances
        # https://stackoverflow.com/questions/37928794/which-is-faster-for-load-pickle-or-hdf5-in-python
        # https://marshmallow.readthedocs.io/en/stable/
        # https://developers.google.com/protocol-buffers
        # https://developers.google.com/protocol-buffers/docs/pythontutorial
        if tag != '':
            tag = f'_{tag}'

        save_path = Path(base_dir, f'leader_follower_env{tag}.pkl')
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

        # # add all possible agents to the environment - agents are removed from the self.agents as they finish the task
        # self.agents = self.possible_agents[:]
        # self.completed_agents = []
        self._current_step = 0

        # todo  set agents, obstacles, and pois to the initial states
        #       clear the action, observation, and state histories
        # _ = [each_agent.reset() for each_agent in self.leaders.values()]
        # _ = [each_agent.reset() for each_agent in self.followers.values()]
        # _ = [each_agent.reset() for each_agent in self.pois.values()]

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
            agent = agent['agent']
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
            agent = agent['agent']
            agent_obs = observations[name]
            agent_action = agent.get_action(agent_obs)
            actions[name] = agent_action
        return actions

    # def observed_pois(self):
    #     observed = [poi for name, poi in self.pois.items() if poi.observed]
    #     return observed

    def done(self):
        all_obs = len(self.observed_pois()) == len(self.pois.values())
        time_over = self._current_step >= self.max_steps
        episode_done = any([all_obs, time_over])

        agent_dones = {each_agent: episode_done for each_agent in self.agents}
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
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # step leaders, followers, and pois
        # should not matter which order they are stepped in as long as dt is small enough
        for agent_name, each_action in actions.items():
            agent = self.agents[agent_name]
            # each_action[0] is dx
            # each_action[1] is dy
            new_loc = agent['states'] + each_action
            agent['states'].append(new_loc)

        # todo track actions and observations in step function, not when functions called in agent implementation
        # Get all observations
        # todo  remove a poi from self.agents if it is observed and add the poi to self.completed_agents
        observations = self.get_observations()
        # for name, observation in observations.items():
        #     agent = self.agent_mapping[name]
        #     if isinstance(agent, Poi) and agent.observed:
        #         self.agents.remove(name)

        # Step forward and check if simulation is done
        # Update all agent dones with environment done
        self._current_step += self.delta_time
        dones = self.done()

        # Update infos and truncated for agents.
        # Not sure what would go here, but it seems to be expected by pettingzoo
        infos = {agent: {} for agent in self.agents}
        truncs = {agent: {} for agent in self.agents}

        # Calculate fitnesses
        rewards = {agent: 0.0 for agent in self.agents}
        rewards['team'] = 0.0
        if all(dones.values()):
            self.completed_agents = self.possible_agents[:]
            self.agents = []
        return observations, rewards, dones, truncs, infos
