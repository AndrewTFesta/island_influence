"""
@title

@description

"""
import argparse

import pygmo as pg


def verify_installation():
    pg.test.run_test_suite()
    return


def test_evolution():
    # The problem
    # The initial population
    # The algorithm (a self-adaptive form of Differential Evolution (sade - jDE variant)
    # The actual optimization process
    # Getting the best individual in the population
    prob = pg.problem(pg.rosenbrock(dim=10))
    pop = pg.population(prob, size=20)
    algo = pg.algorithm(pg.sade(gen=1000))
    pop = algo.evolve(pop)
    best_fitness = pop.get_f()[pop.best_idx()]
    print(best_fitness)
    # [1.31392565e-07]
    return


def main(main_args):
    verify_installation()
    test_evolution()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
