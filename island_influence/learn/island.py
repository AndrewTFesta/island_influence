"""
@title

@description

"""
import threading
import time


class MAIsland:

    def __init__(self, agent_populations, evolving_agent_names, env, optimizer, max_optimizer_iters, max_island_iters, name=None):
        if name is None:
            name = ':'.join([str(agent_type) for agent_type in evolving_agent_names])
        self.name = name
        self.agent_populations = agent_populations
        self.evolving_agent_names = evolving_agent_names
        self.env = env

        self.optimizer_func = optimizer
        # self.env_func = env_func
        self.max_optimizer_iters = max_optimizer_iters
        self.max_island_iters = max_island_iters

        # neighbors are a list of "neighboring" islands where an island is able to migrate populations to its neighbors
        # able to know which agents to migrate to which island by looking at the agents being evolved on the current island and the neighbor island
        # note that there is no restriction that any given island may be the only island evolving a certain "type" of agent
        #   or that "neighboring" must be symmetric
        self.neighbors: list[MAIsland] = []
        self.migrated_populations = {}

        self.optimize_thread = threading.Thread(target=self._optimize, daemon=True)
        self.running = False
        self._update_lock = threading.Lock()
        return

    def __repr__(self):
        return f'{self.name}'

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)
        return

    def run(self):
        self.running = True
        self._optimize()
        # self.optimize_thread.start()
        return

    def _optimize(self):
        # todo  at the start of any iteration, try to incorporate any neighbors that may be been migrated to this island
        # run the optimize function to completion (as defined by the optimize function)
        # todo  at the end of every optimization loop (when the optimizer finishes), migrate the champion from all learning population to every neighbor island
        iter_count = 0
        running_gen_idx = 0
        opt_times = []
        trained_pops = None
        top_inds = None

        while self.running and iter_count <= self.max_island_iters:
            remaining_possible_gens = self.max_optimizer_iters - running_gen_idx
            opt_start = time.process_time()
            trained_pops, top_inds, gens_run = self.optimizer_func(
                agent_policies=self.agent_populations, env=self.env, starting_gen=running_gen_idx, max_iters=remaining_possible_gens,
                completion_criteria=self.interrupt_criteria
            )
            opt_end = time.process_time()
            opt_time = opt_end - opt_start
            opt_times.append(opt_time)

            running_gen_idx += gens_run

            iter_count += 1
        return trained_pops, top_inds, opt_times

    def interrupt_criteria(self):
        agent_migrated = [len(policies) > 0 for agent_type, policies in self.migrated_populations]
        completed = any(agent_migrated)
        return completed

    def stop(self):
        self.running = False
        return

    def replace_agents(self, pop_id, population):
        # todo  replace agents in islands
        # make sure pop_id is evolved by this island
        if pop_id not in self.evolving_agent_names:
            print(f'{pop_id=} not evolved by island {self}')
            return

        with self._update_lock:
            # todo  remove/replace or augment old populations?
            self._updates_agents[pop_id] = population
        return

    def migrate_population(self):
        # todo  migrate populations to neighboring islands
        # todo  need to know how many agents are required in the env on other islands to know how many policies to migrate
        for each_neighbor in self.neighbors:
            neighbor_evolves = each_neighbor.evolving_agent_names
            for name, population in self.evolving_agents.items():
                if name in neighbor_evolves:
                    each_neighbor.replace_agents(name, population)
        return
