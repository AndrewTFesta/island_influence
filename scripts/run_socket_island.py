"""
@title

@description

"""
import argparse
import time
import uuid

from island_influence.learn.island.MpIsland import MpIsland


def add_neighbor(mp_island):
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


def remove_neighbor(mp_island):
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


def start_optimizer(mp_island):
    return 0


def stop_optimizer(mp_island):
    return 0


def display_state(mp_island):
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
    'quit': None,
    'Add neighbor': add_neighbor,
    'Remove neighbor': remove_neighbor,
    'Start optimizer': stop_optimizer,
    'Stop optimizer': start_optimizer,
    'Display state': display_state,
    'Send data': send_data
}


def print_menu():
    for idx, each_option in enumerate(options):
        print(f'{idx}: {each_option}')
    return


def main(main_args):
    island_id = str(uuid.uuid4())[-4:]
    mp_island = MpIsland(f'island {island_id}')
    mp_island.start_listeners()
    time.sleep(2)

    # todo  implement communicating between socket-based islands
    running = True
    while running:
        print_menu()
        user_in = input(f'Select desired option: ')
        user_in = int(user_in)
        print(f'You entered: {user_in}: {list(options.keys())[user_in]}')
        if user_in == 0:
            running = False
        else:
            func = list(options.values())[user_in]
            return_val = func(mp_island)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
