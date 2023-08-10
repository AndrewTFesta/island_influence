"""
@title

@description

"""
import argparse
import datetime
import threading
import time
from functools import partial
from pathlib import Path

from island_influence import project_properties
from island_influence.agent import AgentType
from island_influence.harvest_env import HarvestEnv
from island_influence.learn.cceaV2 import ccea
from island_influence.learn.island import MAIsland
from island_influence.setup_env import create_agent_policy, rand_ring_env


def create_harvester_island(
        stat_run, use_threading=True, use_mp=False, track_progress=False, num_sims=2, max_iters=3, migrate_every=1, base_pop_size=15, scale_env=1
):
    env_func = rand_ring_env(scale_factor=scale_env)
    harvester_env: HarvestEnv = env_func()

    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, harvester_env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, harvester_env.excavators[0]),
    }

    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    experiment_dir = Path(project_properties.exps_dir, f'island_exp_test_{date_str}', f'stat_run_{stat_run}')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)

    harvester_pop_sizes = {AgentType.Harvester: base_pop_size, AgentType.Excavator: harvester_env.num_agent_types(AgentType.Excavator)}
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
        'use_mp': use_mp, 'experiment_dir': Path(experiment_dir, 'harvester_island')
    }
    harvester_optimizer = partial(ccea, **harvester_opt_kwargs)

    harvester_island = MAIsland(
        agent_populations=harvester_pops, evolving_agent_names=[AgentType.Harvester], env=harvester_env, optimizer=harvester_optimizer,
        max_iters=max_iters, migrate_every=migrate_every, track_progress=track_progress, threaded=use_threading
    )
    return harvester_island


def create_excavator_island(
        stat_run, use_threading=True, use_mp=False, track_progress=False, num_sims=2, max_iters=3, migrate_every=1, base_pop_size=15, scale_env=1
):
    env_func = rand_ring_env(scale_factor=scale_env)
    excavator_env: HarvestEnv = env_func()

    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, excavator_env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, excavator_env.excavators[0]),
    }

    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    experiment_dir = Path(project_properties.exps_dir, f'island_exp_test_{date_str}', f'stat_run_{stat_run}')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)

    excavator_pop_sizes = {AgentType.Harvester: excavator_env.num_agent_types(AgentType.Harvester), AgentType.Excavator: base_pop_size}
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
        'use_mp': use_mp, 'experiment_dir': Path(experiment_dir, 'excavator_island')
    }
    excavator_optimizer = partial(ccea, **excavator_opt_kwargs)

    excavator_island = MAIsland(
        agent_populations=excavator_pops, evolving_agent_names=[AgentType.Excavator], env=excavator_env, optimizer=excavator_optimizer,
        max_iters=max_iters, migrate_every=migrate_every, track_progress=track_progress, threaded=use_threading
    )
    return excavator_island


def create_mainland(
        stat_run, use_threading=True, use_mp=False, track_progress=False, num_sims=2, max_iters=3, migrate_every=1, base_pop_size=15, scale_env=1
):
    env_func = rand_ring_env(scale_factor=scale_env)
    mainland_env = env_func()

    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, mainland_env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, mainland_env.excavators[0]),
    }

    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    experiment_dir = Path(project_properties.exps_dir, f'island_exp_test_{date_str}', f'stat_run_{stat_run}')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)

    mainland_pop_sizes = {AgentType.Harvester: base_pop_size, AgentType.Excavator: base_pop_size}
    num_agents = {AgentType.Harvester: 4, AgentType.Excavator: 4, AgentType.Obstacle: 100, AgentType.StaticPoi: 10}

    mainland_pops = {
        agent_type: [
            policy_funcs[agent_type](learner=agent_type == AgentType.Excavator)
            # need a minimum number of policies to satisfy the env requirements
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in mainland_pop_sizes.items()
    }

    mainland_opt_kwargs = {
        'env': mainland_env, 'num_sims': num_sims, 'population_sizes': mainland_pop_sizes,
        'direct_assign_fitness': True, 'fitness_update_eps': 1, 'mutation_scalar': 0.1, 'prob_to_mutate': 0.05,
        'use_mp': use_mp, 'experiment_dir': Path(experiment_dir, 'excavator_island')
    }
    mainland_optimizer = partial(ccea, **mainland_opt_kwargs)

    mainland = MAIsland(
        agent_populations=mainland_pops, evolving_agent_names=[AgentType.Excavator, AgentType.Harvester], env=mainland_env, optimizer=mainland_optimizer,
        max_iters=max_iters, migrate_every=migrate_every, track_progress=track_progress, threaded=use_threading
    )
    return mainland


def run_island_experiment(stat_run):
    island_params = {'num_sims': 2, 'max_iters': 3, 'scale_env': 1, 'migrate_every': 5, 'base_pop_size': 15}

    # todo  add logging
    #       logging.basicConfig(format="%(threadName)s:%(message)s")
    # todo  check saving policies
    # todo  check plotting learning trajectories
    # todo  create more islands that optimize same agent types but with different parameters (size)
    print(f'Running island experiment on thread: {threading.get_native_id()}')
    harvester_island = create_harvester_island(stat_run, **island_params)
    excavator_island = create_excavator_island(stat_run, **island_params)
    mainland = create_mainland(stat_run, **island_params)

    harvester_island.add_neighbor(excavator_island)
    harvester_island.add_neighbor(mainland)

    excavator_island.add_neighbor(harvester_island)
    excavator_island.add_neighbor(mainland)

    mainland.add_neighbor(harvester_island)
    mainland.add_neighbor(excavator_island)

    harvester_island.run()
    excavator_island.run()
    mainland.run()

    while harvester_island.running or excavator_island.running or mainland.running:
        # print(f'Harvester island running: {harvester_island.running} | Excavator island running: {excavator_island.running}')
        time.sleep(5)

    harvester_island.stop()
    excavator_island.stop()
    mainland.stop()
    print(f'All islands have finished running')
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
