import json
import math
from pathlib import Path

import numpy as np


def network_distance(network_0, network_1):
    # Frobenius Norm
    # Kullback–Leibler divergence
    # Jensen–Shannon divergence
    # Jordan normal form
    # Frobenius normal form
    # RV coefficient
    # Matrix similarity

    return


def relative(start_loc, end_loc):
    assert len(start_loc) == len(end_loc)

    dx = end_loc[0] - start_loc[0]
    dy = end_loc[1] - start_loc[1]
    angle = np.arctan2(dy, dx)
    angle = np.degrees(angle)
    angle = angle % 360

    dist = np.linalg.norm(np.asarray(end_loc) - np.asarray(start_loc))
    return angle, dist


def observed_agents(observing_set, observed_set):
    closest = {}
    for observed_agent in observed_set:
        observed_location = observed_agent.location
        closest_agent = None
        closest_dist = math.inf
        for observing_agent in observing_set:
            observing_location = observing_agent.location
            angle, dist = relative(observing_location, observed_location)
            if dist < closest_dist and dist <= observing_agent.observation_radius:
                closest_dist = dist
                closest_agent = observing_agent
        if closest_agent is not None:
            closest[observed_agent.name] = (closest_agent, closest_dist)
    return closest


def pol2cart(angle, radius):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return x, y


def deterministic_ring(num_points, center, radius, start_proportion=0, seed=None):
    angles = np.linspace(start=start_proportion, stop=1, num=num_points, endpoint=False)
    angles += start_proportion
    angles *= 2 * np.pi

    # polar_coords = np.vstack((radius, angles))
    # polar_coords = np.transpose(polar_coords)

    # calculating coordinates
    vector_pol2cart = np.vectorize(pol2cart, )
    cart_coords = vector_pol2cart(angles, radius)
    cart_coords = np.transpose(cart_coords)
    center_arr = np.tile(center, (cart_coords.shape[0], 1))

    cart_coords = cart_coords + center_arr
    return cart_coords


def random_ring(num_points, center, min_rad, max_rad, seed=None):
    rng = np.random.default_rng(seed=seed)
    angles = rng.normal(size=num_points)
    angles *= 2 * np.pi

    radius = rng.uniform(low=min_rad, high=max_rad, size=num_points)

    # polar_coords = np.vstack((radius, angles))
    # polar_coords = np.transpose(polar_coords)

    # calculating coordinates
    vector_pol2cart = np.vectorize(pol2cart, )
    cart_coords = vector_pol2cart(angles, radius)
    cart_coords = np.transpose(cart_coords)
    center_arr = np.tile(center, (cart_coords.shape[0], 1))

    cart_coords = cart_coords + center_arr
    return cart_coords


def euclidean(positions_a: np.ndarray, positions_b: np.ndarray, axis=0):
    """
    Calculate the distance between positions A and positions B

    :param positions_a:
    :param positions_b:
    :param axis:
    :return:
    """
    return np.linalg.norm(positions_a - positions_b, axis=axis)


def save_config(config, save_dir, config_name='config', indent=2):
    if not save_dir.exists():
        save_dir.mkdir(exist_ok=True, parents=True)
    save_path = Path(save_dir, f'{config_name}.json')
    with open(save_path, 'w') as config_file:
        json.dump(config, config_file, indent=indent)
    return save_path


def load_config(experiment_dir, config_stem='config'):
    config_fname = Path(experiment_dir, f'{config_stem}.json')
    with open(config_fname, 'r') as config_file:
        ccea_config = json.load(config_file)
    return ccea_config