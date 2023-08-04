"""
@title

@description

"""
import argparse
import itertools

import numpy as np
from matplotlib import pyplot as plt

from island_influence.utils import random_ring, deterministic_ring


def test_deterministic_ring():
    # num_points, center, radius, start_angle = 0
    ring_points = deterministic_ring(num_points=10, center=(10, 10), radius=5, start_proportion=0)
    offset_ring_points = deterministic_ring(num_points=10, center=(10, 10), radius=4, start_proportion=0.2)
    print(ring_points)
    print(offset_ring_points)

    colors = itertools.cycle(['r', 'b', 'g'])
    plt.scatter(*zip(*ring_points), c=next(colors))
    plt.scatter(*zip(*offset_ring_points))

    plt.show()
    plt.close()

    num_agents = 10
    num_obstacles = 20
    num_pois = 10

    agent_bounds = [0, 3]
    obstacle_bounds = [5, 8]
    poi_bounds = [10, 13]

    agent_locs = deterministic_ring(num_points=num_agents, center=(5, 5), radius=np.average(agent_bounds))
    obstacle_locs = deterministic_ring(num_points=num_obstacles, center=(5, 5), radius=np.average(obstacle_bounds))
    poi_locs = deterministic_ring(num_points=num_pois, center=(5, 5), radius=np.average(poi_bounds))

    plt.scatter(*zip(*agent_locs))
    plt.scatter(*zip(*obstacle_locs))
    plt.scatter(*zip(*poi_locs))

    plt.show()
    plt.close()
    return


def test_random_ring():
    ring_points = random_ring(num_points=500, center=(10, 10), min_rad=5, max_rad=8, seed=42)
    print(ring_points)

    plt.scatter(*zip(*ring_points))
    plt.show()
    plt.close()

    # num_points, center, min_rad, max_rad
    num_agents = 10
    num_obstacles = 20
    num_pois = 10

    agent_bounds = [0, 3]
    obstacle_bounds = [5, 8]
    poi_bounds = [10, 13]

    agent_locs = random_ring(num_points=num_agents, center=(5, 5), min_rad=agent_bounds[0], max_rad=agent_bounds[1])
    obstacle_locs = random_ring(num_points=num_obstacles, center=(5, 5), min_rad=obstacle_bounds[0], max_rad=obstacle_bounds[1])
    poi_locs = random_ring(num_points=num_pois, center=(5, 5), min_rad=poi_bounds[0], max_rad=poi_bounds[1])

    plt.scatter(*zip(*agent_locs))
    plt.scatter(*zip(*obstacle_locs))
    plt.scatter(*zip(*poi_locs))

    plt.show()
    plt.close()
    return


def main(main_args):
    # test_random_ring()
    test_deterministic_ring()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
