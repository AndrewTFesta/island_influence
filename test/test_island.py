"""
@title

@description

"""
import argparse

from island_influence.discrete_harvest_env import DiscreteHarvestEnv
from island_influence.learn.cceaV2 import ccea
from island_influence.learn.island import MAIsland


def main(main_args):
    optimizer = ccea
    env = DiscreteHarvestEnv
    agents = {'red_harvesters': [], 'blue_harvesters': [], 'excavators': []}

    island = MAIsland(optimizer=optimizer, env=env, actors=agents, evolving_agent_names=['red_harvesters'], neighbors=['blue_harvesters', 'excavators'])
    island = MAIsland(optimizer=optimizer, env=env, actors=agents, evolving_agent_names=['blue_harvesters'], neighbors=['red_harvesters', 'excavators'])
    island = MAIsland(optimizer=optimizer, env=env, actors=agents, evolving_agent_names=['excavators'], neighbors=['red_harvesters', 'blue_harvesters'])
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
