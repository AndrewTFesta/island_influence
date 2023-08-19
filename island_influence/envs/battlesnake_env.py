"""
@title

@description

"""
import threading
import uuid

import numpy as np

from island_influence.learn.neural_network import NeuralNetwork


class BattlesnakeEnv:

    def __init__(self, policy: NeuralNetwork, board_size=11, health=100, game_id=None):
        if game_id is None:
            game_id = uuid.uuid4()

        self.game_id = game_id
        self.policy = policy
        self.actions = ['left', 'right', 'up', 'down']

        rng = np.random.default_rng()
        positions = rng.random(size=(1, 2))

        self.board_size = board_size
        self.positions = positions
        self.health = health
        return

    def __repr__(self):
        return f'BS server: {self.game_id}'

    def state(self):
        return

    def get_actions(self):
        return

    def _send_action(self):
        return


class BattlesnakeServer:

    def __init__(self, server_id=None):
        if server_id is None:
            server_id = uuid.uuid4()

        self.server_id = server_id
        self.games: dict[BattlesnakeEnv] = {}

        self.endpoints = {
            'start': {'endpoint': r'https://your.battlesnake.com/start', 'running': False},
            'move': {'endpoint': r'https://your.battlesnake.com/move', 'running': False},
            'end': {'endpoint': r'https://your.battlesnake.com/end', 'running': False},
        }
        self.listen_threads = {
            'start': threading.Thread(target=self._listen_start, args=(), daemon=True),
            # 'move': threading.Thread(target=self._listen, args=(self.endpoints['move'],), daemon=True),
            'end': threading.Thread(target=self._listen_stop, args=(), daemon=True)
        }
        return

    def __repr__(self):
        return f'BS server: {self.server_id}'

    def run(self):
        self.listen_threads['start'].start()
        # for thread_type, thread in self.listen_threads.items():
        #     thread.start()
        return

    def stop(self):
        for end_type, info in self.endpoints.items():
            info['running'] = False
        return

    def _listen_start(self):
        # todo  if this endpoint receives a post request, then add the info to the list of running games
        print(f'{self}: start')
        while self.endpoints['start']['running']:
            pass
        return

    def _listen_stop(self):
        # todo  if this endpoint receives a post request, then remove the info from the list of running games
        print(f'{self}: stop')
        while self.endpoints['stop']['running']:
            pass
        return
