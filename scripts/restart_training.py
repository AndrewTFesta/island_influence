"""
@title

@description

"""
import argparse
import time
from pathlib import Path

from island_influence import project_properties
from island_influence.harvest_env import HarvestEnv
from island_influence.learn.optimizer.cceaV2 import ccea
from island_influence.learn.neural_network import load_pytorch_model
from island_influence.utils import load_config


def restart_stat_run(stat_run_dir):
    ccea_config = load_config(stat_run_dir, config_stem='ccea_config')

    gen_dirs = list(stat_run_dir.glob('gen_*'))
    gen_dirs = sorted(gen_dirs, key=lambda x: int(x.stem.split('_')[-1]))
    last_gen_dir = gen_dirs[len(gen_dirs) - 1]

    env_path = list(stat_run_dir.glob('harvest_env.pkl'))
    env_path = env_path[0]
    env: HarvestEnv = HarvestEnv.load_environment(env_path)
    env.render_mode = None

    agent_pops = {}
    policy_dirs = list(last_gen_dir.glob(f'*_networks'))
    for agent_policy_dir in policy_dirs:
        agent_name = agent_policy_dir.suffix.split('_')
        agent_name = f'AgentType{agent_name[0]}'

        policy_fnames = list(agent_policy_dir.glob(f'*_model*.pt'))
        policy_fnames = sorted(policy_fnames, key=lambda x: int(x.stem.split('_')[-1])) if len(policy_fnames) == len(gen_dirs) else policy_fnames
        agent_pops[agent_name] = [load_pytorch_model(each_fname) for each_fname in policy_fnames]

    population_sizes = ccea_config['population_sizes']
    max_iters = ccea_config['max_iters']
    num_sims = ccea_config['num_sims']
    experiment_dir = Path(ccea_config['experiment_dir'])

    track_progress = ccea_config['track_progress']
    use_mp = ccea_config['use_mp']

    # start neuro_evolve from specified generation
    stat_run = stat_run_dir.stem
    print(f'Restarting stat run {stat_run} experiment: {experiment_dir.parent.stem}')
    opt_start = time.process_time()
    trained_pops, top_inds, gens_run = ccea(
        env, agent_policies=agent_pops, population_sizes=population_sizes, starting_gen=len(gen_dirs),
        max_iters=max_iters, num_sims=num_sims, experiment_dir=experiment_dir, track_progress=track_progress, use_mp=use_mp
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


def main(main_args):
    # todo  update for env and experiment changes
    stat_run_dirs = list(Path(project_properties.output_dir, 'exps').rglob('**/stat_run*'))
    for each_dir in stat_run_dirs:
        restart_stat_run(each_dir)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
