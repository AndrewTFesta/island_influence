"""
@title

@description

"""
import argparse
import threading


class MAIsland:

    def __init__(self, optimizer, env, agents: dict[str, list], evolve_agents: list[str], neighbors: list[str]):
        self.optimizer = optimizer
        self.env = env
        self.agents = agents
        self.evolve_agents = evolve_agents
        self.neighbors = neighbors
        self._updates_agents = {}

        self.optimize_thread = threading.Thread(target=self.optimizer)
        self.evolving = False
        self._update_lock = threading.Lock()
        return

    def __repr__(self):
        return f':'.join(self .evolve_agents)

    def add_neighbors(self, neighbors):
        self.neighbors.append(neighbors)
        return

    def evolve(self):
        self.evolving = True
        self.optimize_thread.start()
        return

    def replace_agents(self, pop_id, population):
        with self._update_lock:
            # todo  remove/replace old population?
            self._updates_agents[pop_id] = population
        return

    def migrate_population(self):
        for each_neighbor in self.neighbors:
            neighbor_island = self.agents[each_neighbor]
            for pop_id, each_pop in self.agents.items():
                neighbor_island.replace_agents({pop_id: each_pop})
        return


def main(main_args):
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
