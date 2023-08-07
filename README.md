Leader Follower
=====

# License

See the [LICENSE file](LICENSE) for license rights and limitations (MIT).

# Index

- [Roadmap](#roadmap)
- [Quick State](#quick-start)
- [Environments](#environments)
  - [HarvestEnv](#harvestenv)
- [Agents](#agents)
  - [Harvester](#harvester)
  - [Excavator](#excavator)
  - [Obstacle](#obstacle)
  - [PoI](#poi)
- [Rewards](#rewards)
- [CCEA](#ccea)
  - [Mutation](#mutation)
  - [Selection](#selection)

# Roadmap

<img src="resources/local_state.png" alt="simulator" width="250">
<br>
<img src="resources/global_state.png" alt="simulator" width="250">
<br>
<img src="resources/island_hierarchy.png" alt="simulator" width="250">
<br>
<img src="resources/island_migrations.png" alt="simulator" width="250">

## Todo

- [ ] Write draft of introduction
  - Expected: 8/7/23
  - Actual:
- [ ] Write draft of background
  - Expected: 8/14/23
  - Actual:
- [ ] Write draft of approach
  - Expected: 8/8/23
  - Actual:
- [ ] Write draft of results
  - Expected: 8/10/23
  - Actual:
- [ ] Write draft of conclusion
  - Expected: 8/11/23
  - Actual:
- [ ] Write draft of future work
  - Expected: 8/11/23
  - Actual:
- [x] Script to restart a stat run
- [x] Multiprocessing subpop simulations in each generation
- [x] Combine multiple stat runs into the same plot
- [ ] Create trajectory graph visualization tool.
  - Include circles around POIs to indicate observation radius.
- [x] Animation to watch rollout of episode
- [ ] Documentation
  - Agent types
  - Environment dynamics
  - Reward structure
  - CCEA optimization
  - Island setup
  - Island migrations
- [ ] Add reward option to be based on average distance of n closest observing agents
- [x] Add agent weight parameter that affects how much an agent affects the environment (satisfy coupling for harvesters and excavators, coupling requirement for obstacles and pois)
- [x] Add agent value parameter that affects how much an agent can contribute to the environment (rewards from pois and harvesters)

# Quick Start

## Multiagent Reliance-Based Learning

[//]: # (introduce the actual task)

[//]: # (Consider a task where multiple agents must simultaneously observe a point of interest &#40;POI&#41; in order to receive any reward from the environment.There are multiple POIs scattered throughout the environment, some further awy than others. If a single POI requires three agents to observe it, then three agents would have to pick actions such that they are within the observation radius of the POI at the same time. The further the POI, the less likely a sequence of random actions from multiple agents, each with a different starting location, will bring them to a similar location. For two or three agents, this is unlikely. As this coupling requirement increases, this random coordination to receive any initial positive feedback becomes next to impossible.)

[//]: # ()
[//]: # (This work introduce multiagent leader-based learning as a method for addressing this necessity of agents having to randomly discover a set of coordinated behaviors for tightly coupled tasks requiring many agents. The method splits agents into two types: leaders and followers. Leaders take on the form of typical learning agents that take the state as input and produce an action as output at every time step. Followers have the same state and action spaces as the leaders, but instead they use a simple preset policy that causes them to move towards nearby agents while maintaining a minimal distance between each other.)

[//]: # ()
[//]: # (The key insight here is that the follower policy acts as a method of injecting domain knowledge about the task without fully specifying the behavior of the system. In a tightly coupled problem, multiple agents must work in close coordination to accomplish the task. The follower policy pushes some agents towards acting in a manner that is conducive to the agents working closely. Often, designers will shape the fitness functions to try and capture how well a task is performed, and it is this fitness shaping that is meant to drive the manifestation of a desired behavior. However, simple policies themselves can also serve as an effective means of guiding systems of agents to coordinate in complex manners.)

### Experiments and evaluation

# Environments
## HarvestEnv
# Agents
## Harvester
## Excavator
## Obstacle
## POI
# Rewards
# CCEA
## Mutation
## Selection
