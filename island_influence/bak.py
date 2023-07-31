# class Harvester(Agent):
#
#     def __init__(self, agent_id, sensor_resolution, value, max_velocity, weight,
#                  observation_radius, policy: NeuralNetwork | None):
#
#         # agent_id: int, tuple, sensor_resolution: int, value: float, max_velocity: float, weight: float
#         super().__init__(agent_id, sensor_resolution, value, max_velocity, weight)
#         self.name = f'leader_{agent_id}'
#         self.type = AgentType.Learner
#
#         self.observation_radius = observation_radius
#         self.policy = policy
#         self._policy_history = []
#
#         self.n_in = self.sensor_resolution * Leader.NUM_ROWS
#         self.n_out = 2
#         return
#
#     def observation_space(self):
#         sensor_range = spaces.Box(
#             low=0, high=np.inf,
#             shape=(self.sensor_resolution, Leader.NUM_ROWS), dtype=np.float64
#         )
#         return sensor_range
#
#     def action_space(self):
#         action_range = spaces.Box(
#             low=-1 * self.max_velocity, high=self.max_velocity,
#             shape=(self.n_out,), dtype=np.float64
#         )
#         return action_range
#
#     def reset(self):
#         Agent.reset(self)
#         self._policy_history = []
#         return
#
#     def sense(self, other_agents, sensor_resolution=None, offset=False):
#         """
#         Takes in the state of the worlds and counts how many agents are in each d-hyperoctant around the agent,
#         with the agent being at the center of the observation.
#
#         Calculates which pois, leaders, and follower go into which d-hyperoctant, where d is the state
#         resolution of the environment.
#
#         first set of (sensor_resolution) bins is for leaders/followers
#         second set of (sensor_resolution) bins is for pois
#
#         :param other_agents:
#         :param sensor_resolution:
#         :param offset:
#         :return:
#         """
#         self.state_history.append(self.location)
#         obs_agents = Agent.observable_agents(self, other_agents, self.observation_radius)
#
#         bin_size = 360 / self.sensor_resolution
#         if offset:
#             offset = 360 / (self.sensor_resolution * 2)
#             bin_size = offset * 2
#
#         observation = np.zeros((2, self.sensor_resolution))
#         counts = np.ones((2, self.sensor_resolution))
#         for idx, entry in enumerate(obs_agents):
#             agent, angle, dist = entry[0]
#             agent_type_idx = Leader.ROW_MAPPING[agent.type]
#             bin_idx = int(np.floor(angle / bin_size) % self.sensor_resolution)
#             observation[agent_type_idx, bin_idx] += agent.value / max(dist, 0.01)
#             counts[agent_type_idx, bin_idx] += 1
#
#         observation = np.divide(observation, counts)
#         observation = np.nan_to_num(observation)
#         observation = observation.flatten()
#         self.observation_history.append(observation)
#         return observation
#
#     def get_action(self, observation):
#         """
#         Computes the x and y vectors using the active policy and the passed in observation.
#
#         :param observation:
#         :return:
#         """
#         active_policy = self.policy
#         with torch.no_grad():
#             action = active_policy(observation)
#             action = action.numpy()
#         self._policy_history.append(action)
#
#         mag = np.linalg.norm(action)
#         if mag > self.max_velocity:
#             action = action / mag
#             action *= self.max_velocity
#
#         self.action_history.append(action)
#         return action
#
#
# class Support(Agent):
#
#     def __init__(self, agent_id, sensor_resolution, value, max_velocity, weight,
#                  repulsion_radius, repulsion_strength, attraction_radius, attraction_strength):
#         # agent_id: int, location: tuple, sensor_resolution: int, value: float, max_velocity: float, weight: float
#         super().__init__(agent_id, sensor_resolution, value, max_velocity, weight)
#         self.name = f'follower_{agent_id}'
#         self.type = AgentType.Support
#
#         self.repulsion_radius = repulsion_radius
#         self.repulsion_strength = repulsion_strength
#
#         self.attraction_radius = attraction_radius
#         self.attraction_strength = attraction_strength
#
#         self.__obs_rule = self.rule_mass_center
#         self.rule_history = {'repulsion': [], 'attraction': []}
#         self.influence_history = {'repulsion': [], 'attraction': []}
#         return
#
#     def reset(self):
#         Agent.reset(self)
#
#         self.rule_history = {'repulsion': [], 'attraction': []}
#         self.influence_history = {'repulsion': [], 'attraction': []}
#         return
#
#     def observation_space(self):
#         # sum of vectors of agents in each radius
#         #   repulsion
#         #   attraction
#         sensor_range = spaces.Box(low=-np.inf, high=np.inf, shape=(2, 2), dtype=np.float64)
#         return sensor_range
#
#     def action_space(self):
#         action_range = spaces.Box(
#             low=-1 * self.max_velocity, high=self.max_velocity,
#             shape=(2,), dtype=np.float64
#         )
#         return action_range
#
#     def rule_mass_center(self, relative_agents, rule_radius):
#         obs_agents = Agent.observable_agents(self, relative_agents, rule_radius)
#         # adding self partially guards against when no other agents are nearby
#         obs_agents.append((self, 0, 0))
#
#         # weight center of mass by agent weight
#         rel_locs = [
#             ((each_agent[0].location[0] - self.location[0]) * each_agent[0].weight,
#              (each_agent[0].location[1] - self.location[1]) * each_agent[0].weight)
#             for each_agent in obs_agents
#         ]
#         avg_locs = np.average(rel_locs, axis=0)
#
#         obs_agents.remove((self, 0, 0))
#         return avg_locs, obs_agents
#
#     def sense(self, state):
#         """
#         agent_velocities
#         Finds the average velocity and acceleration of all the agents within the observation radius of the base agent.
#
#         :param state:
#         :return:
#         """
#         self.state_history.append(self.location)
#         repulsion_bins, repulsion_agents = self.__obs_rule(state, self.repulsion_radius)
#         attraction_bins, attraction_agents = self.__obs_rule(state, self.attraction_radius)
#
#         self.influence_history['repulsion'].extend(repulsion_agents)
#         self.influence_history['attraction'].extend(attraction_agents)
#
#         observation = np.vstack([repulsion_bins, attraction_bins])
#         self.observation_history.append(observation)
#         return observation
#
#     def influence_counts(self):
#         repulsion_names = [agent[0].name for agent in self.influence_history['repulsion']]
#         repulsion_counts = np.unique(repulsion_names, return_counts=True)
#
#         attraction_names = [agent[0].name for agent in self.influence_history['attraction']]
#         attraction_counts = np.unique(attraction_names, return_counts=True)
#
#         repulsion_names.extend(attraction_names)
#         total_counts = np.unique(repulsion_names, return_counts=True)
#
#         return total_counts, repulsion_counts, attraction_counts
#
#     def get_action(self, observation):
#         # todo  Followers should not have repulsion and attraction radii with xy update rules
#         repulsion_diff = observation[0]
#         self.rule_history['repulsion'].append(repulsion_diff)
#         repulsion_diff = np.nan_to_num(repulsion_diff)
#         # todo  weight repulsion and attraction by max radii for each. center of mass further
#         #       away will affect a follower less than if it is nearby
#         weighted_repulsion = repulsion_diff * self.repulsion_strength
#         weighted_repulsion *= -1
#
#         attraction_diff = observation[1]
#         self.rule_history['attraction'].append(attraction_diff)
#         attraction_diff = np.nan_to_num(attraction_diff)
#         weighted_attraction = attraction_diff * self.attraction_strength
#
#         action = weighted_attraction + weighted_repulsion
#         mag = np.linalg.norm(action)
#         if mag > self.max_velocity:
#             action = action / mag
#             action *= self.max_velocity
#
#         # add noise to updates because if two followers end up in the same location, they will not separate
#         rng = default_rng()
#         noise = rng.random(size=2) * self.max_velocity / 100
#         action += noise
#         self.action_history.append(action)
#         return action
#
#
# class Obstacle(Agent):
#
#     def __init__(self, agent_id, sensor_resolution, value, weight,
#                  observation_radius, coupling):
#         # agent_id: int, sensor_resolution: int, value: float, max_velocity: float, weight: float
#         super().__init__(agent_id, sensor_resolution, value, max_velocity=0, weight=weight)
#         self.name = f'obstacle_{agent_id}'
#         self.type = AgentType.Static
#
#         self.observation_radius = observation_radius
#         self.coupling = coupling
#         return
#
#     def observation_space(self):
#         sensor_range = spaces.Box(low=0, high=self.coupling, shape=(1,))
#         return sensor_range
#
#     def action_space(self):
#         # static agents do not move during an episode
#         action_range = spaces.Box(low=0, high=0, shape=(2,), dtype=np.float64)
#         return action_range
#
#     def sense(self, state):
#         """
#         Each entry in the observation is a tuple of (agent, angle, dist) where angle and dist are the
#         angle and distance to the agent in the tuple relative to this poi.
#
#         """
#         observation = []
#         self.observation_history.append(observation)
#         return observation
#
#     def get_action(self, observation):
#         action = np.array([0, 0])
#         self.action_history.append(action)
#         return action
#
# class Poi(Agent):
#
#     @property
#     def observed(self):
#         max_seen = 0
#         for each_step in self.observation_history:
#             # using value allows for different agents to contribute different weights to observing the poi
#             curr_seen = sum(each_agent[0].value for each_agent in each_step)
#             max_seen = max(max_seen, curr_seen)
#         obs = max_seen >= self.coupling
#         return obs
#
#     def __init__(self, agent_id, sensor_resolution, value, weight,
#                  observation_radius, coupling):
#         # agent_id: int, sensor_resolution: int, value: float, max_velocity: float, weight: float
#         super().__init__(agent_id, sensor_resolution, value, max_velocity=0, weight=weight)
#         self.name = f'poi_{agent_id}'
#         self.type = AgentType.Static
#
#         self.observation_radius = observation_radius
#         self.coupling = coupling
#         return
#
#     def __repr__(self):
#         parent_repr = Agent.__repr__(self)
#         return f'({parent_repr} -> {self.observed=})'
#
#     def observation_space(self):
#         sensor_range = spaces.Box(low=0, high=self.coupling, shape=(1,))
#         return sensor_range
#
#     def action_space(self):
#         # static agents do not move during an episode
#         action_range = spaces.Box(low=0, high=0, shape=(2,), dtype=np.float64)
#         return action_range
#
#     def observed_idx(self, idx):
#         idx_obs = self.observation_history[idx]
#         idx_seen = len(idx_obs)
#         observed = idx_seen >= self.coupling
#         return observed
#
#     def sense(self, state):
#         """
#         Each entry in the observation is a tuple of (agent, angle, dist) where angle and dist are the
#         angle and distance to the agent in the tuple relative to this poi.
#
#         """
#         self.state_history.append(self.location)
#         # filter out other POIs from the poi observation
#         # todo only store agent_names rather than full agent object
#         observable = self.observable_agents(state, self.observation_radius)
#         observation = [
#             each_obs
#             for each_obs in observable
#             if not isinstance(each_obs[0], Poi)
#         ]
#         self.observation_history.append(observation)
#         return observation
#
#     def get_action(self, observation):
#         action = np.array([0, 0])
#         self.action_history.append(action)
#         return action
