"""
@title

@description

"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from island_influence import project_properties
from island_influence.agent import AgentType
from island_influence.harvest_env import HarvestEnv
from island_influence.learn.cceaV2 import rollout
from island_influence.learn.neural_network import load_pytorch_model


def parse_stat_run(stat_run_dir):
    gen_dirs = list(stat_run_dir.glob('gen_*'))
    gen_dirs = sorted(gen_dirs, key=lambda x: int(x.stem.split('_')[-1]))
    generations = []
    for each_dir in gen_dirs:
        fitness_fname = Path(each_dir, 'fitnesses.json')
        with open(fitness_fname, 'r') as fitness_file:
            gen_data = json.load(fitness_file)
        generations.append(gen_data)

    condensed_gens = {name: [] for name in generations[0].keys()}
    for each_gen in generations:
        for each_name, vals in each_gen.items():
            condensed_gens[each_name].append(vals)

    np_gens = {
        name: data
        for name, data in condensed_gens.items()
    }
    return np_gens


def parse_experiment_fitnesses(experiment_dir: Path):
    stat_dirs = list(experiment_dir.glob('stat_run_*'))
    stat_dirs = sorted(stat_dirs, key=lambda x: int(x.stem.split('_')[-1]))
    stat_runs = []
    for each_dir in stat_dirs:
        fitness_data = parse_stat_run(each_dir)
        stat_runs.append(fitness_data)
    stat_keys = list(stat_runs[0].keys())
    exp_fitnesses = {each_key: [] for each_key in stat_keys}
    for each_run in stat_runs:
        for agent_type, fitnessses in each_run.items():
            exp_fitnesses[agent_type].append(fitnessses)
    return exp_fitnesses


def plot_fitnesses(fitness_data, save_dir, tag, save_format='svg'):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    agent_keys = [f'AgentType.{element.name}' for element in AgentType]
    for each_key in agent_keys:
        if each_key in fitness_data:
            agent_fitnesses = fitness_data[each_key]
            # issue arises due to first few generations not necessarily having the max number of policies
            max_fitnesses = []
            for each_stat_run in agent_fitnesses:
                stat_run_maxs = []
                for each_gen in each_stat_run:
                    max_fitness = max(each_gen)
                    stat_run_maxs.append(max_fitness)
                max_fitnesses.append(stat_run_maxs)

            means = np.mean(max_fitnesses, axis=0)
            stds = np.std(max_fitnesses, axis=0)
            stds /= math.sqrt(len(agent_fitnesses))

            gen_idxs = np.arange(0, len(means))
            axes.plot(gen_idxs, means, label=f'max {each_key}')
            axes.fill_between(gen_idxs, means + stds, means - stds, alpha=0.2)

    axes.set_xlabel(f'generation')
    axes.set_ylabel('fitness')

    axes.xaxis.grid()
    axes.yaxis.grid()

    axes.legend(loc='best')

    fig.suptitle(f'{tag}')

    fig.set_size_inches(7, 5)
    fig.set_dpi(100)

    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    plot_name = f'{tag}'
    save_name = Path(save_dir, f'{plot_name}_{tag}')
    plt.savefig(f'{save_name}.{save_format}')
    plt.close()
    return


def replay_episode(episode_dir: Path):
    stat_dirs = list(episode_dir.glob('stat_run_*'))
    stat_dirs = sorted(stat_dirs, key=lambda x: int(x.stem.split('_')[-1]))
    for idx, each_dir in enumerate(stat_dirs):
        fitness_data = parse_stat_run(each_dir)
        gen_dirs = list(each_dir.glob('gen_*'))
        gen_dirs = sorted(gen_dirs, key=lambda x: int(x.stem.split('_')[-1]))
        last_gen_idx = len(gen_dirs) - 1
        last_gen_dir = gen_dirs[last_gen_idx]

        env_path = list(each_dir.glob('harvest_env_initial.pkl'))
        env_path = env_path[0]
        env: HarvestEnv = HarvestEnv.load_environment(env_path)
        env.render_mode = 'human'

        agent_policies = {}
        policy_dirs = list(last_gen_dir.glob(f'*_networks'))
        for agent_policy_dir in policy_dirs:
            agent_name = agent_policy_dir.suffix.split('_')
            agent_name = f'AgentType{agent_name[0]}'

            # find the best policies for each agent based on fitnesses.json
            agent_fitnesses = fitness_data[agent_name][last_gen_idx]
            num_agent_types = env.num_agent_types(agent_name)
            arg_best_policies = np.argpartition(agent_fitnesses, -num_agent_types)[-num_agent_types:]

            policy_fnames = list(agent_policy_dir.glob(f'*_model*.pt'))
            policy_fnames = sorted(policy_fnames, key=lambda x: int(x.stem.split('_')[-1])) if len(policy_fnames) == len(gen_dirs) else policy_fnames

            agent_policies[agent_name] = []
            for policy_idx in arg_best_policies:
                best_policy_fn = policy_fnames[policy_idx] if len(policy_fnames) == len(gen_dirs) else policy_fnames[0]
                model = load_pytorch_model(best_policy_fn)
                agent_policies[agent_name].append(model)

        episode_rewards, policy_rewards = rollout(env, agent_policies, render=True)
        rewards = env.cumulative_rewards()
        print(f'stat_run: {idx} | {episode_rewards=}')
    return


def main(main_args):
    # todo  update to work for island experiments
    base_save_dir = Path(project_properties.output_dir, 'experiment_results', 'figs')
    if not base_save_dir.exists():
        base_save_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(project_properties.output_dir, 'exps')
    experiment_dirs = list(base_dir.glob('harvest_exp_*'))

    for each_dir in experiment_dirs:
        exp_types = list(each_dir.iterdir())
        for each_type in exp_types:
            print(f'Processing experiment: {each_dir.stem}: {each_type.stem}')
            fitness_data = parse_experiment_fitnesses(each_type)
            # replay_episode(each_dir)

            plot_fitnesses(fitness_data, save_dir=base_save_dir, tag=f'{each_type.stem}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
