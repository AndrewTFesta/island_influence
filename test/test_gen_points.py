"""
@title

@description

"""
import argparse

from matplotlib import pyplot as plt

from island_influence.utils import random_ring


def main(main_args):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
