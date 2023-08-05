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
from island_influence.learn.cceaV2 import ccea
from scripts.setup_env import rand_ring_env, create_agent_policy


def test_base_ccea(env, num_gens, policy_funcs, exp_dir):
    # to simulate all agents in both populations, choose a number larger than the population sizes of each agent type
    num_sims = 20
    population_sizes = {AgentType.Harvester: 20, AgentType.Excavator: 20}
    num_agents = {AgentType.Harvester: 4, AgentType.Excavator: 4, AgentType.Obstacle: 100, AgentType.StaticPoi: 10}

    agent_pops = {
        agent_type: [
            policy_funcs[agent_type](learner=True)
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in population_sizes.items()
    }

    top_inds = ccea(
        env, agent_policies=agent_pops, population_sizes=population_sizes,
        num_gens=num_gens, num_sims=num_sims, experiment_dir=exp_dir
    )
    for agent_type, individuals in top_inds.items():
        for each_ind in individuals:
            print(f'{agent_type}: {each_ind.fitness}')
    return


def test_unequal_pops(env, num_gens, policy_funcs, exp_dir):
    num_sims = 20
    population_sizes = {AgentType.Harvester: 20, AgentType.Excavator: 30}
    num_agents = {AgentType.Harvester: 4, AgentType.Excavator: 4, AgentType.Obstacle: 100, AgentType.StaticPoi: 10}

    agent_pops = {
        agent_type: [
            policy_funcs[agent_type](learner=True)
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in population_sizes.items()
    }

    top_inds = ccea(
        env, agent_policies=agent_pops, population_sizes=population_sizes,
        num_gens=num_gens, num_sims=num_sims, experiment_dir=exp_dir
    )
    for agent_type, individuals in top_inds.items():
        for each_ind in individuals:
            print(f'{agent_type}: {each_ind.fitness}')
    return


def test_single_training(env, num_gens, policy_funcs, exp_dir, agent_type=AgentType.Harvester):
    num_sims = 20
    population_sizes = {agent_type: 20}
    num_agents = {AgentType.Harvester: 4, AgentType.Excavator: 4, AgentType.Obstacle: 100, AgentType.StaticPoi: 10}

    agent_pops = {
        agent_type: [
            policy_funcs[agent_type](learner=True)
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in population_sizes.items()
    }

    top_inds = ccea(
        env, agent_policies=agent_pops, population_sizes=population_sizes,
        num_gens=num_gens, num_sims=num_sims, experiment_dir=exp_dir
    )
    for agent_type, individuals in top_inds.items():
        for each_ind in individuals:
            print(f'{agent_type}: {each_ind.fitness}')
    return


def test_non_learning_pop(env, num_gens, policy_funcs, exp_dir):
    num_sims = 20
    population_sizes = {AgentType.Harvester: 20, AgentType.Excavator: 1}
    num_agents = {AgentType.Harvester: 4, AgentType.Excavator: 4, AgentType.Obstacle: 100, AgentType.StaticPoi: 10}

    agent_pops = {
        agent_type: [
            policy_funcs[agent_type](learner=agent_type == AgentType.Harvester)
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in population_sizes.items()
    }

    top_inds = ccea(
        env, agent_policies=agent_pops, population_sizes=population_sizes,
        num_gens=num_gens, num_sims=num_sims, experiment_dir=exp_dir
    )
    for agent_type, individuals in top_inds.items():
        for each_ind in individuals:
            print(f'{agent_type}: {each_ind.fitness}')
    return


def test_restart(env, num_gens, policy_funcs, exp_dir):
    # todo  test restarting training
    return


def main(main_args):
    env = rand_ring_env()
    num_gens = 5

    policy_funcs = {
        AgentType.Harvester: partial(create_agent_policy, env.harvesters[0]),
        AgentType.Excavator: partial(create_agent_policy, env.excavators[0]),
    }

    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    stat_run = 0
    # experiment_dir = Path(project_properties.exps_dir, f'harvest_exp_{date_str}', f'stat_run_{stat_run}')
    experiment_dir = Path(project_properties.exps_dir, f'harvest_exp_test')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)

    # todo  make num_sim keyed to each agent type
    # the tag added to define the type of test essentially acts as the stat run
    test_base_ccea(env, num_gens, policy_funcs, exp_dir=Path(experiment_dir, 'base_ccea'))
    test_unequal_pops(env, num_gens, policy_funcs, exp_dir=Path(experiment_dir, 'unequal_pops'))
    test_single_training(env, num_gens, policy_funcs, exp_dir=Path(experiment_dir, 'single_training'))
    test_non_learning_pop(env, num_gens, policy_funcs, exp_dir=Path(experiment_dir, 'non_learning'))
    # test_restart(env, num_gens, policy_funcs, exp_dir=Path(experiment_dir, 'restart'))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
