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
from scripts.setup_env import rand_ring_env, create_agent_policy


def test_base_ccea(env, num_sims, num_gens, policy_funcs, exp_dir):
    print(f'=' * 80)
    print('Testing base ccea')
    print(f'=' * 80)
    # to simulate all agents in both populations, choose a number larger than the population sizes of each agent type
    population_sizes = {AgentType.Harvester: 20, AgentType.Excavator: 20}
    num_agents = {AgentType.Harvester: 4, AgentType.Excavator: 4, AgentType.Obstacle: 100, AgentType.StaticPoi: 10}

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

    opt_start = time.process_time()
    trained_pops, top_inds, gens_run = ccea(
        env, agent_policies=agent_pops, population_sizes=population_sizes,
        max_iters=num_gens, num_sims=num_sims, experiment_dir=exp_dir, track_progress=True, use_mp=True
    )
    opt_end = time.process_time()
    opt_time = opt_end - opt_start
    print(f'Optimization time: {opt_time} for {gens_run} generations')
    for agent_type, individuals in top_inds.items():
        print(f'{agent_type}')
        for each_ind in individuals:
            print(f'{each_ind}: {each_ind.fitness}')
    print(f'=' * 80)
    return


def test_unequal_pops(env, num_sims, num_gens, policy_funcs, exp_dir):
    print(f'=' * 80)
    print('Testing unequal population sizes')
    print(f'=' * 80)
    population_sizes = {AgentType.Harvester: 20, AgentType.Excavator: 30}
    num_agents = {AgentType.Harvester: 4, AgentType.Excavator: 4, AgentType.Obstacle: 100, AgentType.StaticPoi: 10}

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

    opt_start = time.process_time()
    trained_pops, top_inds, gens_run = ccea(
        env, agent_policies=agent_pops, population_sizes=population_sizes,
        max_iters=num_gens, num_sims=num_sims, experiment_dir=exp_dir, track_progress=True, use_mp=True
    )
    opt_end = time.process_time()
    opt_time = opt_end - opt_start
    print(f'Optimization time: {opt_time} for {gens_run} generations')
    for agent_type, individuals in top_inds.items():
        print(f'{agent_type}')
        for each_ind in individuals:
            print(f'{each_ind}: {each_ind.fitness}')
    print(f'=' * 80)
    return


def test_single_training(env, num_sims, num_gens, policy_funcs, exp_dir, agent_type=AgentType.Harvester):
    print(f'=' * 80)
    print('Testing single training population')
    print(f'=' * 80)
    population_sizes = {agent_type: 20}
    num_agents = {AgentType.Harvester: 4, AgentType.Excavator: 4, AgentType.Obstacle: 100, AgentType.StaticPoi: 10}

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

    opt_start = time.process_time()
    trained_pops, top_inds, gens_run = ccea(
        env, agent_policies=agent_pops, population_sizes=population_sizes,
        max_iters=num_gens, num_sims=num_sims, experiment_dir=exp_dir, track_progress=True, use_mp=True
    )
    opt_end = time.process_time()
    opt_time = opt_end - opt_start
    print(f'Optimization time: {opt_time} for {gens_run} generations')
    for agent_type, individuals in top_inds.items():
        print(f'{agent_type}')
        for each_ind in individuals:
            print(f'{each_ind}: {each_ind.fitness}')
    print(f'=' * 80)
    return


def test_non_learning_pop(env, num_sims, num_gens, policy_funcs, exp_dir):
    print(f'=' * 80)
    print('Testing training with non-learning population')
    print(f'=' * 80)
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
    for agent_type, policies in agent_pops.items():
        print(f'{agent_type}')
        for each_policy in policies:
            print(f'{each_policy}: {each_policy.fitness}')
    print(f'=' * 80)

    opt_start = time.process_time()
    trained_pops, top_inds, gens_run = ccea(
        env, agent_policies=agent_pops, population_sizes=population_sizes,
        max_iters=num_gens, num_sims=num_sims, experiment_dir=exp_dir, track_progress=True, use_mp=True
    )
    opt_end = time.process_time()
    opt_time = opt_end - opt_start
    print(f'Optimization time: {opt_time} for {gens_run} generations')
    for agent_type, individuals in top_inds.items():
        print(f'{agent_type}')
        for each_ind in individuals:
            print(f'{each_ind}: {each_ind.fitness}')
    print(f'=' * 80)
    return


def test_restart(env, num_gens, policy_funcs, exp_dir):
    # todo  test restarting training
    return


def main(main_args):
    # todo  setup experiments to run on laptop
    # todo  timeline for masters document
    # todo  go over notes, emails, and conversations to figure out all tests and story to convey
    num_runs = 1
    num_sims = 20
    num_gens = 25
    env_func = rand_ring_env()
    env = env_func()

    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, env.excavators[0]),
    }

    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    experiment_dir = Path(project_properties.exps_dir, f'harvest_exp_test_{date_str}')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(num_runs):
        # todo  make num_sim keyed to each agent type
        # the tag added to define the type of test essentially acts as the stat run
        test_base_ccea(env, num_sims, num_gens, policy_funcs, exp_dir=Path(experiment_dir, 'base_ccea', f'stat_run_{idx}'))
        test_unequal_pops(env, num_sims, num_gens, policy_funcs, exp_dir=Path(experiment_dir, 'unequal_pops', f'stat_run_{idx}'))
        test_single_training(env, num_sims, num_gens, policy_funcs, exp_dir=Path(experiment_dir, 'single_training', f'stat_run_{idx}'))
        test_non_learning_pop(env, num_sims, num_gens, policy_funcs, exp_dir=Path(experiment_dir, 'non_learning', f'stat_run_{idx}'))
        # test_restart(env, num_sims, num_gens, policy_funcs, exp_dir=Path(experiment_dir, 'restart'))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
