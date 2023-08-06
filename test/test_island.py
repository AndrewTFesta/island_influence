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
from island_influence.harvest_env import HarvestEnv
from island_influence.learn.cceaV2 import ccea
from island_influence.learn.island import MAIsland
from scripts.setup_env import create_agent_policy, rand_ring_env


def main(main_args):
    num_runs = 1
    # todo  make this return a factory function
    env = rand_ring_env()
    num_sims = 20
    num_gens = 100

    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, env.excavators[0]),
    }

    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    stat_run = 0
    # experiment_dir = Path(project_properties.exps_dir, f'harvest_exp_{date_str}', f'stat_run_{stat_run}')
    # experiment_dir = Path(project_properties.exps_dir, f'harvest_exp_test')
    experiment_dir = Path(project_properties.exps_dir, f'harvest_exp_{date_str}')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)

    population_sizes = {AgentType.Harvester: 20, AgentType.Excavator: 1}
    num_agents = {AgentType.Harvester: 4, AgentType.Excavator: 4, AgentType.Obstacle: 100, AgentType.StaticPoi: 10}

    agent_pops = {
        agent_type: [
            policy_funcs[agent_type](learner=agent_type == AgentType.Harvester)
            # need a minimum number of policies to satisfy the env requirements
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in population_sizes.items()
    }

    env_func = rand_ring_env()

    # trained_pops, top_inds = ccea(
    #     env, agent_policies=agent_pops, population_sizes=population_sizes,
    #     num_gens=num_gens, num_sims=num_sims, experiment_dir=experiment_dir
    # )
    # env: HarvestEnv, agent_policies, population_sizes, num_gens, num_sims, experiment_dir,
    #  initialize=True, starting_gen=0, direct_assign_fitness=True, fitness_update_eps=1, mutation_scalar=0.1, prob_to_mutate=0.05
    optimizer = partial(ccea, initialize=False, starting_gen=0, direct_assign_fitness=True, fitness_update_eps=1, mutation_scalar=0.1, prob_to_mutate=0.05)

    # island = MAIsland(optimizer=optimizer, env=env, actors=agents, evolving_agent_names=['red_harvesters'], neighbors=['blue_harvesters', 'excavators'])
    # island = MAIsland(optimizer=optimizer, env=env, actors=agents, evolving_agent_names=['blue_harvesters'], neighbors=['red_harvesters', 'excavators'])
    # island = MAIsland(optimizer=optimizer, env=env, actors=agents, evolving_agent_names=['excavators'], neighbors=['red_harvesters', 'blue_harvesters'])
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
