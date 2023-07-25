"""
@title

@description

"""
import argparse

from island_influence.discrete_harvest_env import DiscreteHarvestEnv
from island_influence.learn.cceaV2 import ccea


def main(main_args):
    env = DiscreteHarvestEnv()
    agents = {'red_harvesters': []}
    supports = {'supports': None, 'blue_harvesters': None}
    agent_pops = agents | supports

    # env, agent_pops, population_size, n_gens, reward_func, experiment_dir,
    # selection_func, sim_func, downselect_func, starting_gen=0,
    optimizer = ccea(env, agent_pops)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
