"""
@title

@description

"""
import argparse

import numpy as np


def main(main_args):
    noise = 0.01
    num_policies = 4
    fitness_vals = np.asarray([0 for _ in range(num_policies)])
    fitness_noise = np.random.uniform(-noise / 2, noise / 2, len(fitness_vals))

    # fitness_vals += noise
    fitness_vals = np.add(fitness_vals, fitness_noise)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
