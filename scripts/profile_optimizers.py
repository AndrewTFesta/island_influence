"""
@title

@description

"""
import argparse
import cProfile
import datetime
import logging
from pathlib import Path
from pstats import SortKey

from island_influence import project_properties
from island_influence.learn.island.MAIsland import MAIsland
from island_influence.setup_env import rand_ring_env
from scripts.run_ccea import run_ccea
from scripts.run_islands import run_island_experiment

BASE_POP_SIZE = 25

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
DATE_STR = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

ISLAND_PARAMS = {'max_iters': 15, 'migrate_every': 15, 'track_progress': True, 'logger': LOGGER}
CCEA_PARAMS = {
    'starting_gen': 0, 'mutation_scalar': 0.1, 'prob_to_mutate': 0.05, 'num_sims': 5, 'fitness_update_eps': 0,
    'track_progress': True, 'use_mp': False,
}
ENV_PARAMS = {
    'scale_factor': 0.5, 'num_harvesters': 4, 'num_excavators': 4, 'num_obstacles': 16, 'num_pois': 8, 'obs_rad': 1, 'max_vel': 1,
    'agent_size': 1, 'obs_size': 1, 'poi_size': 1, 'agent_weight': 1, 'obs_weight': 1, 'poi_weight': 1, 'agent_value': 1, 'obstacle_value': 1,
    'poi_value': 1, 'sen_res': 8, 'delta_time': 1, 'max_steps': 25, 'collision_penalty_scalar': 0, 'reward_type': 'global', 'normalize_rewards': True,
    'render_mode': None
}


def profile_ccea():
    max_iters = 25
    experiment_dir = Path(project_properties.profile_dir, f'harvest_exp_{DATE_STR}')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)

    stat_dir = Path(experiment_dir, f'stat_run_{0}')
    run_ccea(rand_ring_env, ENV_PARAMS, CCEA_PARAMS, base_pop_size=BASE_POP_SIZE, experiment_dir=stat_dir, max_iters=max_iters)
    return


def profile_islands():
    experiment_dir = Path(project_properties.profile_dir, f'island_exp_{DATE_STR}', f'stat_run_{0}')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)
    run_island_experiment(experiment_dir, ISLAND_PARAMS, CCEA_PARAMS, ENV_PARAMS, base_pop_size=BASE_POP_SIZE, env_type=rand_ring_env, island_class=MAIsland)
    return


def main(main_args):
    """
    https://likegeeks.com/python-profiling/
    cProfile.run(statement, filename=None, sort=-1)

    ncalls: represents the number of calls.
    tottime: denotes the total time taken by a function. It excludes the time taken by calls made to sub-functions.
    percall: (tottime)/(ncalls)
    cumtime: represents the total time taken by a function as well as the time taken by subfunctions called by the parent function.
    percall: (cumtime)/( primitive calls)
    filename:lineno(function): gives the respective data of every function.

    :param main_args:
    :return:
    """
    # todo  create optimizer function calls to compare their execution bottlenecks
    # cProfile.run('profile_ccea()', sort=SortKey.CUMULATIVE)
    cProfile.run('profile_islands()', sort=SortKey.CUMULATIVE)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
