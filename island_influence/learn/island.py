"""
@title

@description

"""
import argparse
import threading


class MAIsland:

    @property
    def evolving_agents(self):
        return {name: population for name, population in self.agents.items() if name in self.evolving_agent_names}

    def __init__(self, optimizer, env, actors: dict[str, list], evolving_agent_names: list[str]):
        self.optimizer = optimizer
        self.env = env

        # populations of all the agents (and actors) in the system
        self.agents = actors

        # list of agents types that this island is evolving
        self.evolving_agent_names = evolving_agent_names

        # neighbors are a list of "neighboring" islands where an island is able to migrate populations to its neighbors
        # able to know which agents to migrate to which island by looking at the agents being evolved on the current island and the neighbor island
        # note that there is no restriction that any given island may be the only island evolving a certain "type" of agent
        #   or that "neighboring" must be symmetric
        self.neighbors: list[MAIsland] = []
        self._updates_agents = {}

        self.optimize_thread = threading.Thread(target=self.optimizer)
        self.evolving = False
        self._update_lock = threading.Lock()
        return

    def __repr__(self):
        return f':'.join(self .evolving_agent_names)

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)
        return

    def evolve(self):
        self.evolving = True
        self.optimize_thread.start()
        # todo  incorporate migrated populations in optimization process
        return

    def replace_agents(self, pop_id, population):
        # make sure pop_id is evolved by this island
        if pop_id not in self.evolving_agent_names:
            print(f'{pop_id=} not evolved by island {self}')
            return

        with self._update_lock:
            # todo  remove/replace or augment old populations?
            self._updates_agents[pop_id] = population
        return

    def migrate_population(self):
        for each_neighbor in self.neighbors:
            neighbor_evolves = each_neighbor.evolving_agent_names
            for name, population in self.evolving_agents.items():
                if name in neighbor_evolves:
                    each_neighbor.replace_agents(name, population)
        return


def main(main_args):
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
