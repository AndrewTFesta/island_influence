"""
@title

@description

"""
import argparse
import datetime
import logging
import time
import uuid
from pathlib import Path

from island_influence import project_properties
from island_influence.learn.island.MpIsland import MpIsland
from island_influence.setup_env import rand_ring_env
from scripts.run_islands import create_mainland


def quit_script(mp_island: MpIsland):
    mp_island.close()
    return


def add_neighbor(mp_island: MpIsland):
    exit_code = 0
    valid = False
    while not valid:
        try:
            neighbor_host = '127.0.0.1'
            # neighbor_host = input('Enter the host of the neighbor:')
            # if len(neighbor_host) == 0:
            #     neighbor_host = '127.0.0.1'

            neighbor_port = input('Enter the port of the neighbor: ')
            neighbor_port = int(neighbor_port)
            valid = True

            print(f'Trying to connect to neighbor at address: {neighbor_host}:{neighbor_port}')
            mp_island.connect(neighbor_host, neighbor_port)
            exit_code = 1
        except Exception as e:
            print(f'Invalid input: {e}')
    return exit_code


def remove_neighbor(mp_island: MpIsland):
    exit_code = 0
    valid = False
    while not valid:
        try:
            neighbor_ids = list(mp_island.neighbors.keys())
            for idx, each_island in enumerate(neighbor_ids):
                print(f'{idx}: {each_island}')
            user_in = input(f'Please select the index of the island to remove: ')
            user_in = int(user_in)
            island_to_remove = neighbor_ids[user_in]
            valid = mp_island.remove_neighbor(island_to_remove)
            exit_code = 1
        except Exception as e:
            print(f'Invalid input: {e}')

    return exit_code


def start_optimizer(mp_island: MpIsland):
    mp_island.optimize()
    return 1


def stop_optimizer(mp_island: MpIsland):
    mp_island.stop()
    return 1


def display_state(mp_island: MpIsland):
    neighbor_ids = list(mp_island.neighbors.keys())
    str_end = '' if len(neighbor_ids) == 1 else 's'
    print(f'Island {mp_island.name} is connected to {len(neighbor_ids)} island{str_end}')
    for island_name, neighbor in mp_island.neighbors.items():
        print(f'\t{island_name}: {neighbor["last_heartbeat"]}')
    return 1


def send_data(mp_island):
    mp_island.send_data('test')
    return 1


options = {
    'quit': quit_script,
    'Add neighbor': add_neighbor,
    'Remove neighbor': remove_neighbor,
    'Start optimizer': start_optimizer,
    'Stop optimizer': stop_optimizer,
    'Display state': display_state,
    'Send data': send_data
}


def print_menu():
    for idx, each_option in enumerate(options):
        print(f'{idx}: {each_option}')
    return


def main(main_args):
    debug = False
    log_level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)

    island_params = {
        'env_type': rand_ring_env, 'num_harvesters': 4, 'num_excavators': 4, 'num_obstacles': 15, 'num_pois': 8,
        'num_sims': 15, 'max_iters': 500, 'scale_env': 1, 'migrate_every': 15, 'base_pop_size': 25,
        'use_threading': True, 'use_mp': True, 'track_progress': True,
        'direct_assign_fitness': True, 'fitness_update_eps': 0.1, 'mutation_scalar': 0.1, 'prob_to_mutate': 0.05,
    }
    collision_penalty_scalar = 0

    now = datetime.datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    island_id = str(uuid.uuid4())[-4:]
    experiment_dir = Path(project_properties.exps_dir, f'mpisland_exp_test_{date_str}', f'mpisland_{island_id}')
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)

    mp_island = create_mainland(
        island_class=MpIsland, experiment_dir=Path(experiment_dir, 'mpisland'),
        collision_penalty_scalar=collision_penalty_scalar, logger=logger, **island_params
    )
    mp_island.start_listeners()
    time.sleep(2)
    ########################################################################################

    # todo  implement communicating between socket-based islands
    running = True
    while running:
        try:
            print_menu()
            user_in = input(f'Select desired option: ')
            user_in = int(user_in)
            print(f'You entered: {user_in}: {list(options.keys())[user_in]}')
            func = list(options.values())[user_in]
            return_val = func(mp_island)
            # print(f'Results of {func.__name__}: {return_val}')
            if user_in == 0:
                running = False
            time.sleep(2)
        except KeyboardInterrupt as ki:
            print(f'Keyboard interrupt received: closing connections')
            quit_script(mp_island)
            running = False
        except Exception as e:
            print(f'Unexpected exception: {e}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
