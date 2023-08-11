"""
@title

@description

"""
import argparse
import datetime
import logging
import threading
import time
from functools import partial
from pathlib import Path

from island_influence import project_properties
from island_influence.agent import AgentType
from island_influence.harvest_env import HarvestEnv
from island_influence.learn.island.ThreadIsland import ThreadIsland
from island_influence.learn.optimizer.cceaV2 import ccea
from island_influence.learn.island.MAIsland import MAIsland
from island_influence.setup_env import create_agent_policy, rand_ring_env
from island_influence.utils import save_config


def create_harvester_island(
        island_class, experiment_dir, env_type, use_threading=True, use_mp=False, track_progress=False,
        num_harvesters=4, num_excavators=4, num_obstacles=8, num_pois=8, collision_penalty_scalar=0,
        num_sims=2, max_iters=3, migrate_every=1, base_pop_size=15, scale_env=1,
        direct_assign_fitness=True, fitness_update_eps=0.1, mutation_scalar=0.1, prob_to_mutate=0.05,
        logger=None,
):
    env_func = env_type(
        scale_factor=scale_env, num_harvesters=num_harvesters, num_excavators=num_excavators, num_obstacles=num_obstacles, num_pois=num_pois,
        collision_penalty_scalar=collision_penalty_scalar
    )
    harvester_env: HarvestEnv = env_func()

    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, harvester_env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, harvester_env.excavators[0]),
    }

    harvester_pop_sizes = {AgentType.Harvester: base_pop_size, AgentType.Excavator: harvester_env.num_agent_types(AgentType.Excavator)}
    num_agents = {AgentType.Harvester: num_harvesters, AgentType.Excavator: num_excavators, AgentType.Obstacle: num_obstacles, AgentType.StaticPoi: num_pois}

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
        'direct_assign_fitness': direct_assign_fitness, 'fitness_update_eps': fitness_update_eps,
        'mutation_scalar': mutation_scalar, 'prob_to_mutate': prob_to_mutate,
        'use_mp': use_mp, 'experiment_dir': experiment_dir
    }
    harvester_optimizer = partial(ccea, **harvester_opt_kwargs)

    island_params = {
        'env_type': rand_ring_env.__name__, 'num_harvesters': num_harvesters, 'num_excavators': num_excavators,
        'num_obstacles': num_obstacles, 'num_pois': num_pois, 'save_dir': str(experiment_dir),
        'max_iters': max_iters, 'scale_env': scale_env, 'migrate_every': migrate_every, 'use_threading': use_threading, 'track_progress': track_progress,
    }
    save_config(island_params, save_dir=experiment_dir, config_name='island_config')
    harvester_island = island_class(
        agent_populations=harvester_pops, evolving_agent_names=[AgentType.Harvester], env=harvester_env, optimizer=harvester_optimizer,
        max_iters=max_iters, migrate_every=migrate_every, track_progress=track_progress, name='Harvester', save_dir=experiment_dir,
        logger=logger,
    )
    return harvester_island


def create_excavator_island(
        island_class, experiment_dir, env_type, use_threading=True, use_mp=False, track_progress=False,
        num_harvesters=4, num_excavators=4, num_obstacles=8, num_pois=8, collision_penalty_scalar=0,
        num_sims=2, max_iters=3, migrate_every=1, base_pop_size=15, scale_env=1,
        direct_assign_fitness=True, fitness_update_eps=0.1, mutation_scalar=0.1, prob_to_mutate=0.05,
        logger=None,
):
    env_func = env_type(
        scale_factor=scale_env, num_harvesters=num_harvesters, num_excavators=num_excavators, num_obstacles=num_obstacles, num_pois=num_pois,
        collision_penalty_scalar=collision_penalty_scalar
    )
    excavator_env: HarvestEnv = env_func()

    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, excavator_env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, excavator_env.excavators[0]),
    }

    excavator_pop_sizes = {AgentType.Harvester: excavator_env.num_agent_types(AgentType.Harvester), AgentType.Excavator: base_pop_size}
    num_agents = {AgentType.Harvester: num_harvesters, AgentType.Excavator: num_excavators, AgentType.Obstacle: num_obstacles, AgentType.StaticPoi: num_pois}

    excavator_pops = {
        agent_type: [
            policy_funcs[agent_type](learner=agent_type == AgentType.Excavator)
            # need a minimum number of policies to satisfy the env requirements
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in excavator_pop_sizes.items()
    }

    excavator_opt_kwargs = {
        'num_sims': num_sims, 'population_sizes': excavator_pop_sizes,
        'direct_assign_fitness': direct_assign_fitness, 'fitness_update_eps': fitness_update_eps,
        'mutation_scalar': mutation_scalar, 'prob_to_mutate': prob_to_mutate,
        'use_mp': use_mp, 'experiment_dir': experiment_dir
    }
    excavator_optimizer = partial(ccea, **excavator_opt_kwargs)

    island_params = {
        'env_type': rand_ring_env.__name__, 'num_harvesters': num_harvesters, 'num_excavators': num_excavators,
        'num_obstacles': num_obstacles, 'num_pois': num_pois, 'save_dir': str(experiment_dir),
        'max_iters': max_iters, 'scale_env': scale_env, 'migrate_every': migrate_every, 'use_threading': use_threading, 'track_progress': track_progress,
    }
    save_config(island_params, save_dir=experiment_dir, config_name='island_config')
    excavator_island = island_class(
        agent_populations=excavator_pops, evolving_agent_names=[AgentType.Excavator], env=excavator_env, optimizer=excavator_optimizer,
        max_iters=max_iters, migrate_every=migrate_every, track_progress=track_progress, name='Excavator', save_dir=experiment_dir,
        logger=logger,
    )
    return excavator_island


def create_mainland(
        island_class, experiment_dir, env_type, use_threading=True, use_mp=False, track_progress=False,
        num_harvesters=4, num_excavators=4, num_obstacles=8, num_pois=8, collision_penalty_scalar=0,
        num_sims=2, max_iters=3, migrate_every=1, base_pop_size=15, scale_env=1,
        direct_assign_fitness=True, fitness_update_eps=0.1, mutation_scalar=0.1, prob_to_mutate=0.05,
        logger=None,
):
    env_func = env_type(
        scale_factor=scale_env, num_harvesters=num_harvesters, num_excavators=num_excavators, num_obstacles=num_obstacles, num_pois=num_pois,
        collision_penalty_scalar=collision_penalty_scalar
    )
    mainland_env = env_func()

    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, mainland_env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, mainland_env.excavators[0]),
    }

    mainland_pop_sizes = {AgentType.Harvester: base_pop_size, AgentType.Excavator: base_pop_size}
    num_agents = {AgentType.Harvester: num_harvesters, AgentType.Excavator: num_excavators, AgentType.Obstacle: num_obstacles, AgentType.StaticPoi: num_pois}

    mainland_pops = {
        agent_type: [
            policy_funcs[agent_type](learner=agent_type == AgentType.Excavator)
            # need a minimum number of policies to satisfy the env requirements
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in mainland_pop_sizes.items()
    }

    mainland_opt_kwargs = {
        'num_sims': num_sims, 'population_sizes': mainland_pop_sizes,
        'direct_assign_fitness': direct_assign_fitness, 'fitness_update_eps': fitness_update_eps,
        'mutation_scalar': mutation_scalar, 'prob_to_mutate': prob_to_mutate,
        'use_mp': use_mp, 'experiment_dir': experiment_dir
    }
    mainland_optimizer = partial(ccea, **mainland_opt_kwargs)

    island_params = {
        'env_type': rand_ring_env.__name__, 'num_harvesters': num_harvesters, 'num_excavators': num_excavators,
        'num_obstacles': num_obstacles, 'num_pois': num_pois, 'save_dir': str(experiment_dir),
        'max_iters': max_iters, 'scale_env': scale_env, 'migrate_every': migrate_every, 'use_threading': use_threading, 'track_progress': track_progress,
    }
    save_config(island_params, save_dir=experiment_dir, config_name='island_config')
    mainland = island_class(
        agent_populations=mainland_pops, evolving_agent_names=[AgentType.Excavator, AgentType.Harvester], env=mainland_env, optimizer=mainland_optimizer,
        max_iters=max_iters, migrate_every=migrate_every, track_progress=track_progress, name='Mainland', save_dir=experiment_dir,
        logger=logger,
    )
    return mainland


def run_island_experiment(experiment_dir):
    debug = False
    log_level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)

    island_params = {
        'env_type': rand_ring_env, 'num_harvesters': 4, 'num_excavators': 4, 'num_obstacles': 50, 'num_pois': 8,
        'num_sims': 15, 'max_iters': 500, 'scale_env': 1, 'migrate_every': 15, 'base_pop_size': 25,
        'use_threading': True, 'use_mp': True, 'track_progress': False,
        'direct_assign_fitness': True, 'fitness_update_eps': 0.1, 'mutation_scalar': 0.1, 'prob_to_mutate': 0.05,
    }
    if debug:
        island_params = {
            'env_type': rand_ring_env, 'num_harvesters': 4, 'num_excavators': 4, 'num_obstacles': 8, 'num_pois': 8,
            'num_sims': 5, 'max_iters': 500, 'scale_env': 1, 'migrate_every': 5, 'base_pop_size': 10,
            'use_threading': False, 'use_mp': False, 'track_progress': True,
            'direct_assign_fitness': True, 'fitness_update_eps': 0.1, 'mutation_scalar': 0.1, 'prob_to_mutate': 0.05,
        }
    # todo  fix issue with negative fitnesses
    collision_penalty_scalar = 0
    island_class = ThreadIsland if island_params['use_threading'] else MAIsland

    # todo  check plotting learning trajectories
    # todo  create more island that optimize same agent types but with different parameters (etc size, optimizer, num_agents)
    logging.debug(f'Running island experiment on thread: {threading.get_native_id()}')
    harvester_island = create_harvester_island(
        island_class=island_class, experiment_dir=Path(experiment_dir, 'harvester_island'),
        collision_penalty_scalar=collision_penalty_scalar, logger=logger, **island_params
    )
    excavator_island = create_excavator_island(
        island_class=island_class, experiment_dir=Path(experiment_dir, 'excavator_island'),
        collision_penalty_scalar=collision_penalty_scalar, logger=logger, **island_params
    )
    mainland = create_mainland(
        island_class=island_class, experiment_dir=Path(experiment_dir, 'mainland'),
        collision_penalty_scalar=collision_penalty_scalar, logger=logger, **island_params
    )

    harvester_island.add_neighbor(excavator_island)
    harvester_island.add_neighbor(mainland)

    excavator_island.add_neighbor(harvester_island)
    excavator_island.add_neighbor(mainland)

    mainland.add_neighbor(harvester_island)
    mainland.add_neighbor(excavator_island)

    harvester_island.optimize()
    excavator_island.optimize()
    mainland.optimize()

    while harvester_island.running or excavator_island.running or mainland.running:
        time.sleep(5)

    harvester_island.stop()
    excavator_island.stop()
    mainland.stop()
    print(f'All islands have finished running')

    # display final results
    print(f'=' * 80)
    print(f'Island {excavator_island.name} has finished running: {excavator_island.total_gens_run} after {sum(excavator_island.opt_times)} seconds')
    for agent_type, each_ind in excavator_island.top_inds.items():
        print(f'\t{agent_type}: {each_ind.name}: {each_ind.fitness}')
    print(f'=' * 80)
    print(f'Island {harvester_island.name} has finished running: {harvester_island.total_gens_run} after {sum(harvester_island.opt_times)} seconds')
    for agent_type, each_ind in harvester_island.top_inds.items():
        print(f'\t{agent_type}: {each_ind.name}: {each_ind.fitness}')
    print(f'=' * 80)
    print(f'Island {mainland.name} has finished running: {mainland.total_gens_run} after {sum(mainland.opt_times)} seconds')
    for agent_type, each_ind in mainland.top_inds.items():
        print(f'\t{agent_type}: {each_ind.name}: {each_ind.fitness}')
    print(f'=' * 80)
    return


def main(main_args):
    num_runs = 3
    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    for idx in range(num_runs):
        experiment_dir = Path(project_properties.exps_dir, f'island_exp_test_{date_str}', f'stat_run_{idx}')
        if not experiment_dir.exists():
            experiment_dir.mkdir(parents=True, exist_ok=True)
        try:
            print(f'Starting stat run {idx}')
            run_island_experiment(experiment_dir)
        except KeyboardInterrupt:
            print(f'Stopping stat run: {idx}')
        except Exception as e:
            print(f'Unexpected exception: {e}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
