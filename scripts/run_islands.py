"""
@title

@description

"""
import argparse
import copy
import datetime
import logging
import threading
import time
from functools import partial
from pathlib import Path

from island_influence import project_properties
from island_influence.agent import AgentType
from island_influence.learn.island.MAIsland import MAIsland
from island_influence.learn.island.ThreadIsland import ThreadIsland
from island_influence.learn.optimizer.cceaV2 import ccea
from island_influence.setup_env import create_agent_policy, rand_ring_env

DEBUG = False


def create_harvester_island(island_class, experiment_dir, island_params, ccea_params, env_params):
    env_type = env_params.pop('env_type')
    base_pop_size = env_params.pop('base_pop_size')
    env_func = env_type(**env_params)
    env = env_func()
    num_agents = {
        AgentType.Harvester: env_params['num_harvesters'], AgentType.Excavator: env_params['num_excavators'],
        AgentType.Obstacle: env_params['num_obstacles'], AgentType.StaticPoi: env_params['num_pois']
    }
    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, env.excavators[0]),
    }
    #############################################################################################################
    pop_sizes = {AgentType.Harvester: base_pop_size, AgentType.Excavator: env.num_agent_types(AgentType.Excavator)}
    agent_pops = {
        agent_type: [
            policy_funcs[agent_type](learner=agent_type == AgentType.Excavator)
            # need a minimum number of policies to satisfy the env requirements
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in pop_sizes.items()
    }
    island_params['name'] = 'Harvester'
    island_params['evolving_agent_names'] = [AgentType.Harvester]
    #############################################################################################################
    env_params['env_type'] = env_type.__name__
    ccea_params['population_sizes'] = pop_sizes
    island_params['optimizer'] = partial(ccea, **ccea_params)
    island_params['agent_populations'] = agent_pops
    island_params['env'] = env
    island_params['save_dir'] = experiment_dir

    island = island_class(**island_params)
    return island


def create_excavator_island(island_class, experiment_dir, island_params, ccea_params, env_params):
    env_type = env_params.pop('env_type')
    base_pop_size = env_params.pop('base_pop_size')
    env_func = env_type(**env_params)
    env = env_func()
    num_agents = {
        AgentType.Harvester: env_params['num_harvesters'], AgentType.Excavator: env_params['num_excavators'],
        AgentType.Obstacle: env_params['num_obstacles'], AgentType.StaticPoi: env_params['num_pois']
    }
    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, env.excavators[0]),
    }
    #############################################################################################################
    pop_sizes = {AgentType.Harvester: env.num_agent_types(AgentType.Harvester), AgentType.Excavator: base_pop_size}
    agent_pops = {
        agent_type: [
            policy_funcs[agent_type](learner=agent_type == AgentType.Excavator)
            # need a minimum number of policies to satisfy the env requirements
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in pop_sizes.items()
    }
    island_params['name'] = 'Excavator'
    island_params['evolving_agent_names'] = [AgentType.Excavator]
    #############################################################################################################
    env_params['env_type'] = env_type.__name__
    ccea_params['population_sizes'] = pop_sizes
    island_params['optimizer'] = partial(ccea, **ccea_params)
    island_params['agent_populations'] = agent_pops
    island_params['env'] = env
    island_params['save_dir'] = experiment_dir

    island = island_class(**island_params)
    return island


def create_mainland(island_class, experiment_dir, island_params, ccea_params, env_params):
    env_type = env_params.pop('env_type')
    base_pop_size = env_params.pop('base_pop_size')
    env_func = env_type(**env_params)
    env = env_func()
    num_agents = {
        AgentType.Harvester: env_params['num_harvesters'], AgentType.Excavator: env_params['num_excavators'],
        AgentType.Obstacle: env_params['num_obstacles'], AgentType.StaticPoi: env_params['num_pois']
    }
    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, env.excavators[0]),
    }
    #############################################################################################################
    pop_sizes = {AgentType.Harvester: base_pop_size, AgentType.Excavator: base_pop_size}
    agent_pops = {
        agent_type: [
            policy_funcs[agent_type](learner=agent_type == AgentType.Excavator)
            # need a minimum number of policies to satisfy the env requirements
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in pop_sizes.items()
    }

    island_params['name'] = 'Mainland'
    island_params['evolving_agent_names'] = [AgentType.Excavator, AgentType.Harvester]
    #############################################################################################################
    env_params['env_type'] = env_type.__name__
    ccea_params['population_sizes'] = pop_sizes
    island_params['optimizer'] = partial(ccea, **ccea_params)
    island_params['agent_populations'] = agent_pops
    island_params['env'] = env
    island_params['save_dir'] = experiment_dir

    island = island_class(**island_params)
    return island


def run_island_experiment(experiment_dir, island_params, ccea_params, env_params):
    use_threading = island_params.pop('use_threading')
    island_class = ThreadIsland if use_threading else MAIsland
    # todo  check plotting learning trajectories
    # todo  create more island that optimize same agent types but with different parameters (etc size, optimizer, num_agents)
    logging.debug(f'Running island experiment on thread: {threading.get_native_id()}')
    harvester_island = create_harvester_island(
        island_class=island_class, experiment_dir=Path(experiment_dir, 'harvester_island'),
        island_params=copy.deepcopy(island_params), ccea_params=copy.deepcopy(ccea_params), env_params=copy.deepcopy(env_params)
    )
    excavator_island = create_excavator_island(
        island_class=island_class, experiment_dir=Path(experiment_dir, 'excavator_island'),
        island_params=copy.deepcopy(island_params), ccea_params=copy.deepcopy(ccea_params), env_params=copy.deepcopy(env_params)
    )
    mainland = create_mainland(
        island_class=island_class, experiment_dir=Path(experiment_dir, 'mainland'),
        island_params=copy.deepcopy(island_params), ccea_params=copy.deepcopy(ccea_params), env_params=copy.deepcopy(env_params)
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
    log_level = logging.DEBUG if DEBUG else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)
    island_params = {'max_iters': 500, 'migrate_every': 15, 'use_threading': True, 'track_progress': True, 'logger': logger}
    ccea_params = {
        'num_sims': 15, 'starting_gen': 0, 'direct_assign_fitness': True, 'fitness_update_eps': 1, 'mutation_scalar': 0.1,
        'prob_to_mutate': 0.05, 'track_progress': True, 'use_mp': False,
    }
    env_params = {
        'env_type': rand_ring_env, 'scale_factor': 1, 'num_harvesters': 4, 'num_excavators': 4, 'num_obstacles': 8, 'num_pois': 8, 'obs_rad': 2,
        'collision_penalty_scalar': 0, 'max_vel': 1, 'agent_weight': 1, 'obs_weight': 1, 'poi_weight': 1, 'agent_value': 1, 'obstacle_value': 1,
        'poi_value': 1, 'sen_res': 8, 'delta_time': 1, 'render_mode': None, 'max_steps': 100,  'base_pop_size': 25,
    }

    num_runs = 3
    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    for idx in range(num_runs):
        experiment_dir = Path(project_properties.exps_dir, f'island_exp_test_{date_str}', f'stat_run_{idx}')
        if not experiment_dir.exists():
            experiment_dir.mkdir(parents=True, exist_ok=True)
        try:
            print(f'Starting stat run {idx}')
            run_island_experiment(experiment_dir, island_params, ccea_params, env_params)
        except KeyboardInterrupt:
            print(f'Stopping stat run: {idx}')
        except Exception as e:
            print(f'Unexpected exception: {e}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
