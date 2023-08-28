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
# from island_influence.envs.harvest_env import HarvestEnv
# from island_influence.learn.optimizer.cceaV2 import rollout
# from island_influence.learn.neural_network import load_pytorch_model


def parse_generations(generations_dir):
    gen_dirs = list(generations_dir.glob('gen_*'))
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


def parse_harvest_fitnesses(experiment_dir: Path):
    stat_dirs = list(experiment_dir.glob('stat_run_*'))
    stat_dirs = sorted(stat_dirs, key=lambda x: int(x.stem.split('_')[-1]))
    stat_runs = []
    for each_dir in stat_dirs:
        fitness_data = parse_generations(each_dir)
        stat_runs.append(fitness_data)
    stat_keys = list(stat_runs[0].keys())
    exp_fitnesses = {each_key: [] for each_key in stat_keys}
    for each_run in stat_runs:
        for agent_type, fitnessses in each_run.items():
            exp_fitnesses[agent_type].append(fitnessses)
    return exp_fitnesses


def parse_harvest_exp(exp_dir, save_dir):
    exp_name = exp_dir.stem
    save_dir = Path(save_dir, f'harvest_{exp_name}')
    exp_types = list(exp_dir.iterdir())
    for each_type in exp_types:
        fitness_data = parse_harvest_fitnesses(each_type)
        # replay_episode(each_dir)
        plot_fitnesses(fitness_data, save_dir=save_dir, tag=f'{each_type.stem}')
    return


def parse_island_exp(exp_dir, save_dir, tag='None'):
    if len(tag) != 0:
        tag = f'{tag}_'

    stat_dirs = list(exp_dir.glob('stat_run_*'))
    stat_dirs = sorted(stat_dirs, key=lambda x: int(x.stem.split('_')[-1]))

    islands = {}
    for each_stat_run in stat_dirs:
        base_gen_dirs = [each_dir for each_dir in each_stat_run.iterdir() if each_dir.is_dir() and each_dir.stem.startswith('gen_')]
        base_gen_dirs = sorted(base_gen_dirs, key=lambda x: int(x.stem.split('_')[-1]))

        island_dirs = [each_dir for each_dir in each_stat_run.iterdir() if each_dir.is_dir() and not each_dir.stem.startswith('gen_')]
        for each_dir in island_dirs:
            island_type = each_dir.stem
            fitness_data = parse_generations(each_dir)
            # todo  replace with dict.get
            if island_type not in islands:
                islands[island_type] = {agent_type: [] for agent_type in fitness_data}

            island_fitnesses = islands[island_type]
            for agent_type, fitnesses in fitness_data.items():
                agent_fitnesses = island_fitnesses[agent_type]
                agent_fitnesses.append(fitnesses)

    for each_island, fitnesses in islands.items():
        island_tag = f'{tag}{each_island}'
        plot_fitnesses(fitnesses, save_dir=save_dir, tag=island_tag)
    return


def parse_param_sweep_exp(exp_dir, save_dir):
    param_exp_dirs = [each_dir for each_dir in exp_dir.iterdir() if each_dir.is_dir()]
    for param_dir in param_exp_dirs:
        param_name = param_dir.stem.split('_')
        param_name = param_name[-2:]
        param_name = '_'.join(param_name)
        tag = f'{exp_dir.stem}_{param_name}'
        parse_island_exp(param_dir, save_dir, tag=tag)
    return


def plot_fitnesses(fitness_data, save_dir, tag, save_format='svg'):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    plot_keys = [
        # 'global',
        'global_harvester',
        'global_excavator'
    ]
    agent_keys = [f'AgentType.{element.name}' for element in AgentType]
    # plot_keys.extend(agent_keys)

    for each_key in plot_keys:
        if each_key in fitness_data:
            agent_fitnesses = fitness_data[each_key]
            # issue arises due to first few generations not necessarily having the max number of policies
            max_runs = []
            for each_stat_run in agent_fitnesses:
                stat_run_maxs = []
                for gen_fitnesses in each_stat_run:
                    max_fitness = max(gen_fitnesses) if isinstance(gen_fitnesses, list) else gen_fitnesses
                    stat_run_maxs.append(max_fitness)
                max_runs.append(stat_run_maxs)

            min_gens = [len(each_fits) for each_fits in max_runs]
            min_gens = np.min(min_gens)
            max_runs = [each_run[:min_gens] for each_run in max_runs]
            means = np.mean(max_runs, axis=0)
            stds = np.std(max_runs, axis=0)
            stds /= math.sqrt(len(agent_fitnesses))

            gen_idxs = np.arange(0, len(means))
            axes.plot(gen_idxs, means, label=f'{each_key}')
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
    save_name = Path(save_dir, f'{plot_name}')
    plt.savefig(f'{save_name}.{save_format}')
    # plt.show()
    plt.close()
    return


def replay_episode(episode_dir: Path):
    stat_dirs = list(episode_dir.glob('stat_run_*'))
    stat_dirs = sorted(stat_dirs, key=lambda x: int(x.stem.split('_')[-1]))
    for idx, each_dir in enumerate(stat_dirs):
        fitness_data = parse_generations(each_dir)
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
    base_save_dir = Path(project_properties.output_dir, 'experiment_results', 'figs')
    harvest_save_dir = Path(base_save_dir, 'harvest_exp')
    island_save_dir = Path(base_save_dir, 'island_exp')
    param_save_dir = Path(base_save_dir, 'param_sweep')
    if not harvest_save_dir.exists():
        harvest_save_dir.mkdir(parents=True, exist_ok=True)
    if not island_save_dir.exists():
        island_save_dir.mkdir(parents=True, exist_ok=True)
    if not param_save_dir.exists():
        param_save_dir.mkdir(parents=True, exist_ok=True)

    base_exp_dir = Path(project_properties.output_dir, 'exps')
    # base_exp_dir = Path(r'D:\output\exps')
    # exp_dirs = list(base_dir.glob('*_exp_*'))
    exp_dirs = list(base_exp_dir.iterdir())
    for each_exp in exp_dirs:
        exp_type = '_'.join(each_exp.stem.split('_')[:-6])
        if exp_type.startswith('island_exp'):
            continue
            # parse_island_exp(each_exp, save_dir=island_save_dir)
        elif exp_type.startswith('harvest_exp'):
            continue
            # parse_harvest_exp(each_exp, save_dir=harvest_save_dir)
        elif exp_type.startswith('island_param_sweep'):
            parse_param_sweep_exp(each_exp, save_dir=param_save_dir)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
