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
from scripts.run_ccea import run_ccea
from scripts.run_islands import run_island_experiment


def run_parameter_sweep(base_dir, stat_runs, island_params, ccea_params, env_params, param_ranges, island_class):
    keys, values = zip(*param_ranges.items())
    exp_configs = [dict(zip(keys, vals)) for vals in itertools.product(*values)]
    print(f'{len(exp_configs)=}')
    for exp_idx, exp_params in enumerate(exp_configs):
        base_pop_size = exp_params.pop('base_pop_size')
        env_type = exp_params.pop('env_type')

        for each_param, param_val in exp_params.items():
            if each_param in island_params:
                island_params[each_param] = param_val
            if each_param in ccea_params:
                ccea_params[each_param] = param_val
            if each_param in env_params:
                env_params[each_param] = param_val

        experiment_dir = Path(base_dir, f'param_sweep_exp_{exp_idx}')
        if not experiment_dir.exists():
            experiment_dir.mkdir(parents=True, exist_ok=True)

        for stat_idx in range(stat_runs):
            print(f'Starting stat run {stat_idx}')
            stat_dir = Path(experiment_dir, f'stat_run_{stat_idx}')
            run_island_experiment(stat_dir, island_params, ccea_params, env_params, base_pop_size=base_pop_size, env_type=env_type, island_class=island_class)

            ccea_dir = Path(stat_dir, 'base_ccea')
            run_ccea(env_type, env_params, ccea_params, base_pop_size=base_pop_size, experiment_dir=ccea_dir, max_iters=island_params['max_iters'])
    return


def main(main_args):
    debug = False
    stat_runs = 2
    use_threading = True
    island_class = ThreadIsland if use_threading else MAIsland
    log_level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)
    #############################################################
    island_params = {'max_iters': 50, 'track_progress': True, 'logger': logger, 'migrate_every': 5}
    ccea_params = {
        'starting_gen': 0, 'mutation_scalar': 0.1, 'prob_to_mutate': 0.05, 'track_progress': True, 'use_mp': True, 'num_sims': 5, 'fitness_update_eps': 0
    }
    env_params = {
        'scale_factor': 0.5, 'num_harvesters': 4, 'num_excavators': 4, 'num_obstacles': 50, 'num_pois': 8, 'obs_rad': 1, 'max_vel': 1,
        'agent_size': 1, 'obs_size': 1, 'poi_size': 1, 'agent_weight': 1, 'obs_weight': 1, 'poi_weight': 1, 'agent_value': 1, 'obstacle_value': 1,
        'poi_value': 1, 'sen_res': 8, 'delta_time': 1, 'max_steps': 100, 'collision_penalty_scalar': 0, 'reward_type': 'global', 'normalize_rewards': True,
        'render_mode': None
    }

    param_ranges = {
        'migrate_every': [25],
        'num_sims': [25],
        'scale_factor': [1.0, 2.0],
        'num_harvesters': [4],
        'num_excavators': [4],
        'num_obstacles': [50],
        'num_pois': [8],
        'agent_size': [1],
        'obs_size': [1],
        'poi_size': [1],
        'sen_res': [8],
        'base_pop_size': [25],
        'collision_penalty_scalar': [0],
        'fitness_update_eps': [0],
        # 'env_type': [det_ring_env]
        'env_type': [rand_ring_env, det_ring_env]
    }
    ############################################################################
    now = datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    base_dir = Path(project_properties.exps_dir, f'island_param_sweeps_{date_str}')
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_parameter_sweep(base_dir, stat_runs, island_params, ccea_params, env_params, param_ranges, island_class)
    except KeyboardInterrupt:
        print(f'Stopping parameter sweep')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
