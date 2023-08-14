"""
@title

@description

"""
import argparse
import itertools
import logging
from datetime import datetime
from pathlib import Path

from island_influence import project_properties
from island_influence.learn.island.MAIsland import MAIsland
from island_influence.learn.island.ThreadIsland import ThreadIsland
from island_influence.setup_env import det_ring_env, rand_ring_env
from scripts.run_islands import run_island_experiment

DEBUG = True


def run_parameter_sweep(base_dir, island_params, ccea_params, env_params, param_ranges, island_class):
    keys, values = zip(*param_ranges.items())
    exp_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f'{len(exp_configs)=}')
    for idx, exp_params in enumerate(exp_configs):
        base_pop_size = exp_params.pop('base_pop_size')
        env_type = exp_params.pop('env_type')

        for each_param, param_val in exp_params.items():
            if each_param in island_params:
                island_params[each_param] = param_val
            if each_param in ccea_params:
                ccea_params[each_param] = param_val
            if each_param in env_params:
                env_params[each_param] = param_val

        experiment_dir = Path(base_dir, f'param_sweep_exp_{idx}')
        if not experiment_dir.exists():
            experiment_dir.mkdir(parents=True, exist_ok=True)

        run_island_experiment(experiment_dir, island_params, ccea_params, env_params, base_pop_size=base_pop_size, env_type=env_type, island_class=island_class)
    return


def main(main_args):
    stat_runs = 1
    use_threading = True
    island_class = ThreadIsland if use_threading else MAIsland
    log_level = logging.DEBUG if DEBUG else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)
    #############################################################
    island_params = {'max_iters': 15, 'track_progress': True, 'logger': logger, 'migrate_every': 5}
    ccea_params = {
        'starting_gen': 0, 'mutation_scalar': 0.1, 'prob_to_mutate': 0.05, 'track_progress': True, 'use_mp': False, 'num_sims': 5, 'fitness_update_eps': 0
    }
    env_params = {
        'scale_factor': 0.5, 'num_harvesters': 4, 'num_excavators': 4, 'num_obstacles': 50, 'num_pois': 8, 'sen_res': 8,
        'obs_rad': 1, 'max_vel': 1, 'agent_weight': 1, 'obs_weight': 1, 'poi_weight': 1,
        'agent_value': 1, 'obstacle_value': 1, 'poi_value': 1, 'delta_time': 1, 'render_mode': None, 'max_steps': 100,
        'reward_type': 'global', 'collision_penalty_scalar': 0,
    }

    param_ranges = {
        'migrate_every': [5, 25],
        'num_sims': [15, 25],
        'scale_factor': [1.0],
        'num_harvesters': [4],
        'num_excavators': [4],
        'num_obstacles': [15, 50],
        'num_pois': [4, 8],
        'sen_res': [8],
        'base_pop_size': [25],
        'env_type': [rand_ring_env, det_ring_env]
        # 'env_type': [det_ring_env]
    }
    ############################################################################
    now = datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    base_dir = Path(project_properties.exps_dir, f'island_param_sweeps_{date_str}')
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(stat_runs):
        stat_dir = Path(base_dir, f'stat_run_{idx}')
        try:
            print(f'Starting stat run {idx}')
            run_parameter_sweep(stat_dir, island_params, ccea_params, env_params, param_ranges, island_class)
        except KeyboardInterrupt:
            print(f'Stopping stat run: {idx}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
