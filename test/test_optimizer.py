"""
@title

@description

"""
import argparse
import datetime
from functools import partial
from pathlib import Path

from island_influence import project_properties
from island_influence.agent import AgentType
from island_influence.learn.cceaV2 import ccea
from scripts.setup_env import rand_ring_env, create_agent_policy


def main(main_args):
    # todo  test training single type of agent
    # todo  test unequal population sizes
    # todo  make num_sim keyed to each agent type
    # todo  check for unequal population sizes
    # todo  check for non-learning agents and populations
    # todo  check for restarting training
    env = rand_ring_env()
    num_gens = 100

    # to simulate all agents in both populations, choose a number larger than the population sizes of each agent type
    num_sims = 20
    # num_sims = {AgentType.Harvester: 20, AgentType.Excavators: 20}
    population_sizes = {AgentType.Harvester: 20, AgentType.Excavator: 20}
    num_agents = {AgentType.Harvester: 4, AgentType.Excavator: 4, AgentType.Obstacle: 100, AgentType.StaticPoi: 10}

    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    stat_run = 0
    experiment_dir = Path(project_properties.exps_dir, f'harvest_exp_{date_str}', f'stat_run_{stat_run}')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)

    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, env.excavators[0]),
    }

    agent_pops = {
        agent_type: [
            create_agent_policy(policy_funcs[agent_type]())
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in population_sizes.items()
    }

    top_inds = ccea(
        env, agent_policies=agent_pops, population_sizes=population_sizes,
        num_gens=num_gens, num_sims=num_sims, experiment_dir=experiment_dir
    )
    for agent_type, individuals in top_inds.items():
        for each_ind in individuals:
            print(f'{each_ind.fitness}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
