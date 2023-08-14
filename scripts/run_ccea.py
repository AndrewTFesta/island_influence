"""
@title

@description

"""
import argparse
import datetime
import time
from functools import partial
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from island_influence import project_properties
from island_influence.agent import AgentType
from island_influence.learn.optimizer.cceaV2 import ccea
from island_influence.setup_env import rand_ring_env, create_agent_policy


def run_ccea(env_type, env_params, ccea_params, base_pop_size, experiment_dir, max_iters):
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

    population_sizes = {AgentType.Harvester: base_pop_size, AgentType.Excavator: base_pop_size}
    agent_pops = {
        agent_type: [
            policy_funcs[agent_type](learner=True)
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in population_sizes.items()
    }
    for agent_type, policies in agent_pops.items():
        print(f'{agent_type}')
        for each_policy in policies:
            print(f'{each_policy}: {each_policy.fitness}')
    print(f'=' * 80)

    ccea_params['population_sizes'] = population_sizes
    ccea_params['experiment_dir'] = experiment_dir
    ccea_params['max_iters'] = max_iters

    writer = SummaryWriter(log_dir=experiment_dir)
    opt_start = time.process_time()
    trained_pops, top_inds, gens_run = ccea(env, agent_policies=agent_pops, **ccea_params)
    opt_end = time.process_time()
    writer.close()
    opt_time = opt_end - opt_start
    print(f'Optimization time: {opt_time} for {gens_run} generations')
    for agent_type, individuals in top_inds.items():
        print(f'{agent_type}')
        for each_ind in individuals:
            print(f'{each_ind}: {each_ind.fitness}')
    print(f'=' * 80)
    return


def main(main_args):
    num_runs = 3
    base_pop_size = 15
    max_iters= 100
    env_type = rand_ring_env

    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    experiment_dir = Path(project_properties.exps_dir, f'harvest_exp_{date_str}')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)

    ccea_params = {
        'starting_gen': 0, 'mutation_scalar': 0.1, 'prob_to_mutate': 0.05, 'track_progress': True, 'use_mp': False, 'num_sims': 15, 'fitness_update_eps': 0,
    }
    env_params = {
        'scale_factor': 0.5, 'num_harvesters': 4, 'num_excavators': 4, 'num_obstacles': 50, 'num_pois': 8, 'sen_res': 8,
        'obs_rad': 1, 'max_vel': 1, 'agent_weight': 1, 'obs_weight': 1, 'poi_weight': 1,
        'agent_value': 1, 'obstacle_value': 1, 'poi_value': 1, 'delta_time': 1, 'render_mode': None, 'max_steps': 100,
        'reward_type': 'global', 'collision_penalty_scalar': 1, 'normalize_rewards': True
    }

    for idx in range(num_runs):
        run_ccea(env_type, env_params, ccea_params, base_pop_size=base_pop_size, experiment_dir=experiment_dir, max_iters=max_iters)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
