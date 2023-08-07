"""
@title

@description

"""
import argparse
import datetime
import time
from functools import partial
from pathlib import Path

from island_influence import project_properties
from island_influence.agent import AgentType
from island_influence.learn.cceaV2 import ccea
from island_influence.learn.island import MAIsland
from scripts.setup_env import create_agent_policy, rand_ring_env


def create_harvester_island(stat_run):
    env_func = rand_ring_env()
    harvester_env = env_func()

    num_sims = 20
    max_opt_iters = 100
    max_island_iters = 100

    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, harvester_env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, harvester_env.excavators[0]),
    }

    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    experiment_dir = Path(project_properties.exps_dir, f'island_exp_test_{date_str}', f'stat_run_{stat_run}')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)

    harvester_pop_sizes = {AgentType.Harvester: 20, AgentType.Excavator: 1}
    num_agents = {AgentType.Harvester: 4, AgentType.Excavator: 4, AgentType.Obstacle: 100, AgentType.StaticPoi: 10}

    harvester_pops = {
        agent_type: [
            policy_funcs[agent_type](learner=agent_type == AgentType.Harvester)
            # need a minimum number of policies to satisfy the env requirements
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in harvester_pop_sizes.items()
    }

    harvester_opt_kwargs = {
        'num_sims': num_sims, 'population_sizes': harvester_pop_sizes,
        'direct_assign_fitness': True, 'fitness_update_eps': 1, 'mutation_scalar': 0.1, 'prob_to_mutate': 0.05,
        'track_progress': False, 'use_mp': True, 'experiment_dir': Path(experiment_dir, 'harvester_island')
    }
    harvester_optimizer = partial(ccea, **harvester_opt_kwargs)

    harvester_island = MAIsland(
        agent_populations=harvester_pops, evolving_agent_names=[AgentType.Harvester], env=harvester_env, optimizer=harvester_optimizer,
        max_island_iters=max_island_iters, max_optimizer_iters=max_opt_iters
    )
    return harvester_island


def create_excavator_island(stat_run):
    env_func = rand_ring_env()
    harvester_env = env_func()
    excavator_env = env_func()

    num_sims = 20
    max_opt_iters = 100
    max_island_iters = 100

    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, harvester_env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, harvester_env.excavators[0]),
    }

    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    experiment_dir = Path(project_properties.exps_dir, f'island_exp_test_{date_str}', f'stat_run_{stat_run}')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)

    excavator_pop_sizes = {AgentType.Harvester: 1, AgentType.Excavator: 20}
    num_agents = {AgentType.Harvester: 4, AgentType.Excavator: 4, AgentType.Obstacle: 100, AgentType.StaticPoi: 10}

    excavator_pops = {
        agent_type: [
            policy_funcs[agent_type](learner=agent_type == AgentType.Excavator)
            # need a minimum number of policies to satisfy the env requirements
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in excavator_pop_sizes.items()
    }

    excavator_opt_kwargs = {
        'env': excavator_env, 'num_sims': num_sims, 'population_sizes': excavator_pop_sizes,
        'direct_assign_fitness': True, 'fitness_update_eps': 1, 'mutation_scalar': 0.1, 'prob_to_mutate': 0.05,
        'track_progress': False, 'use_mp': True, 'experiment_dir': Path(experiment_dir, 'excavator_island')
    }
    excavator_optimizer = partial(ccea, **excavator_opt_kwargs)

    excavator_island = MAIsland(
        agent_populations=excavator_pops, evolving_agent_names=[AgentType.Excavator], env=excavator_env, optimizer=excavator_optimizer,
        max_island_iters=max_island_iters, max_optimizer_iters=max_opt_iters
    )
    return excavator_island


def run_island_experiment(stat_run):
    harvester_island = create_harvester_island(stat_run)
    excavator_island = create_excavator_island(stat_run)

    harvester_island.add_neighbor(excavator_island)
    excavator_island.add_neighbor(harvester_island)

    harvester_island.run()
    excavator_island.run()

    time.sleep(600)

    harvester_island.stop()
    excavator_island.stop()
    return


def main(main_args):
    num_runs = 1
    for idx in range(num_runs):
        run_island_experiment(idx)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
