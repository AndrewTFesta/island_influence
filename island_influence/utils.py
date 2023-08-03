import math
import pickle
from pathlib import Path
from typing import Dict, Optional

import myaml
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


def closest_agent_sets(origin_set, end_set, min_dist=math.inf):
    closest = {}
    for origin_agent in origin_set:
        origin_location = origin_agent.location
        closest_agent = None
        closest_dist = math.inf
        for end_agent in end_set:
            end_location = end_agent.location
            angle, dist = relative(end_location, origin_location)
            if dist < closest_dist:
                closest_dist = dist
                closest_agent = end_agent
        if closest_dist <= min_dist:
            closest[origin_agent.name] = (closest_agent, closest_dist)
    return closest


def dl2ld(dict_of_lists: dict[object, list]):
    # all lists in the dictionary must be of the same length
    first_val = list(dict_of_lists.values())[0]
    num_vals = len(first_val)

    list_of_dicts = [
        {key: val[idx] for key, val in dict_of_lists.items()}
        for idx in range(num_vals)
    ]
    return list_of_dicts


def ld2dl(list_of_dicts: list[dict]):
    # all lists must contain the same dict keys
    first_element = list_of_dicts[0]
    keys = list(first_element.keys())

    dict_of_lists = {
        [each_element[each_key] for each_element in list_of_dicts]
        for each_key in keys
    }
    return dict_of_lists


def pol2cart(angle, radius):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return x, y


def random_ring(num_points, center, min_rad, max_rad, seed=None):
    rng = np.random.default_rng(seed=seed)
    angles = rng.uniform(size=num_points)
    angles *= 2 * np.pi

    radius = rng.uniform(low=min_rad, high=max_rad, size=num_points)

    polar_coords = np.vstack((radius, angles))
    polar_coords = np.transpose(polar_coords)

    # calculating coordinates
    vector_pol2cart = np.vectorize(pol2cart, )
    cart_coords = vector_pol2cart(angles, radius)
    cart_coords = np.transpose(cart_coords)
    center_arr = np.tile(center, (cart_coords.shape[0], 1))

    cart_coords = cart_coords + center_arr
    return cart_coords


def euclidean(positions_a: np.ndarray, positions_b: np.ndarray, axis=0):
    """Calculate the distance between positions A and B"""
    return np.linalg.norm(positions_a - positions_b, axis=axis)


def calc_delta_heading(current_heading: float, desired_heading: float) -> float:
    """ Calculate delta headings such that delta is the shortest path from
    current heading to the desired heading.
    """
    if desired_heading == current_heading:
        delta_heading = 0
    else:
        # Case 1: Desired heading greater than current heading
        if desired_heading > current_heading:
            desired_heading_prime = desired_heading - 2 * np.pi

        # Case 2: Desired heading less than current heading
        else:
            desired_heading_prime = desired_heading + 2 * np.pi

        delta0 = desired_heading - current_heading
        delta1 = desired_heading_prime - current_heading
        which_delta = np.argmin([np.abs(delta0), np.abs(delta1)])
        delta_heading = np.array([delta0, delta1])[which_delta]
    return delta_heading


def bound_angle(heading, bound=np.pi):
    bounded_heading = heading
    # Bound heading from [0,2pi]
    if bounded_heading > 2 * bound or bounded_heading < 0:
        bounded_heading %= 2 * bound

    # Bound heading from [-pi,+pi]
    if bounded_heading > bound:
        bounded_heading -= 2 * bound
    return bounded_heading


def calc_centroid(positions):
    if positions.size == 0:
        return None
    else:
        return np.average(positions, axis=0)


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def load_trial(base_dir, trial_name: str) -> Dict:
    trial_path = Path(base_dir, 'trials', trial_name)
    with open(trial_path, 'rb') as trial_file:
        trial_data = pickle.load(trial_file)
    return trial_data


def load_population(base_dir, trial_name: str):
    trial_data = load_trial(base_dir, trial_name)
    return trial_data['final_population']


def latest_trial_num(base_dir) -> int:
    trials_dir = Path(base_dir, 'trials')
    trial_nums = [
        int(each_file.stem.split('_')[-1])
        for each_file in trials_dir.glob('*.pkl')
        if each_file.is_file() and each_file.suffix == '.pkl' and each_file.stem.startswith('trial_')
    ]
    if len(trial_nums) == 0:
        return -1
    return max(trial_nums)


def latest_trial_name(base_dir):
    return f'trial_{latest_trial_num(base_dir)}'


def new_trial_name(base_dir) -> str:
    return f'trial_{int(latest_trial_num(base_dir)) + 1}'


def save_trial(base_dir, save_data: Dict, trial_num: Optional[str] = None):
    if trial_num is None:
        trial_name = new_trial_name(base_dir)
    else:
        trial_name = f'trial_{trial_num}'

    # todo save as json for readability
    trial_name = f'{trial_name}.pkl'
    trial_path = Path(base_dir, 'trials', trial_name)
    if not trial_path.parent.exists() or not trial_path.parent.is_dir():
        trial_path.parent.mkdir(parents=True, exist_ok=True)

    with open(trial_path, 'wb') as file:
        pickle.dump(save_data, file)

    return trial_path


def load_config(config_name):
    config_path = str(Path(config_name))
    return myaml.safe_load(config_path)


def setup_initial_population(base_dir, meta_params):
    if meta_params['load_population'] is None:
        return None

    if meta_params["load_population"] == "latest":
        meta_params["load_population"] = latest_trial_name(base_dir)
    return load_population(base_dir, meta_params["load_population"])
