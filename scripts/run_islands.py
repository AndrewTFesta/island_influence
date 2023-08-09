"""
@title

@description

"""
import argparse

from island_influence.learn.neural_network import NeuralNetwork


def harvester_island(env):
    return


def excavator_island(env):
    return


def mainland(env):
    return


def main(main_args):
    agent_pops = {
        agent_type: [
            NeuralNetwork(n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden, learner=True)
            for _ in range(max(pop_size // 5, num_agents[agent_type]))
        ]
        for agent_type, pop_size in population_sizes.items()
    }
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
