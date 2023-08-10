"""
@title

@description

"""
import threading
import time

from tqdm import tqdm


class MAIsland:

    def __init__(self, agent_populations, evolving_agent_names, env, optimizer, max_iters, migrate_every=1, name=None, track_progress=False, threaded=True):
        if name is None:
            name = ':'.join([str(agent_type) for agent_type in evolving_agent_names])
        self.name = f'MAIsland: {name}'
        self.agent_populations = agent_populations
        self.evolving_agent_names = evolving_agent_names
        self.env = env

        self.migrate_every = migrate_every
        self.since_last_migration = 0
        self.optimizer_func = optimizer
        self.max_iters = max_iters
        self.track_progress = track_progress

        # neighbors are a list of "neighboring" islands where an island is able to migrate populations to its neighbors
        # able to know which agents to migrate to which island by looking at the agents being evolved on the current island and the neighbor island
        # note that there is no restriction that any given island may be the only island evolving a certain "type" of agent
        #   or that "neighboring" must be symmetric
        self.neighbors: list[MAIsland] = []
        self.migrated_from_neighbors = {}

        self.threaded = threaded
        self.optimize_thread = threading.Thread(target=self._optimize, daemon=True) if self.threaded else None
        self.optimize_call = self.optimize_thread.start if self.threaded else self._optimize
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
        self.optimize_call()
        return

    def _optimize(self):
        print(f'Running island optimizer on thread: {threading.get_native_id()}')
        # todo  how would this work if it were it's own process?
        #       have to create a socket communication and then connect to other islands
        # run the optimize function to completion (as defined by the optimize function)
        running_gen_idx = 0
        opt_times = []
        trained_pops = None
        top_inds = None
        pbar = tqdm(total=self.max_iters, desc=f'Generation') if self.track_progress else None
        while self.running and running_gen_idx < self.max_iters:
            if self.agents_migrated():
                self.incorporate_migrations()

            opt_start = time.process_time()
            trained_pops, top_inds, gens_run = self.optimizer_func(
                agent_policies=self.agent_populations, env=self.env, starting_gen=running_gen_idx, max_iters=running_gen_idx + self.migrate_every,
                completion_criteria=self.interrupt_criteria, track_progress=pbar
            )
            opt_end = time.process_time()
            opt_time = opt_end - opt_start
            opt_times.append(opt_time)

            running_gen_idx += gens_run
            self.since_last_migration += gens_run
            # at the end of every optimization loop (when the optimizer finishes),
            # migrate the champion from all learning population to every neighbor island
            if self.since_last_migration >= self.migrate_every:
                # this guards against an early migration when we stop the optimizer in order
                # to incorporate a new population that has been migrated from another island
                self.migrate_to_neighbors()
                self.since_last_migration = 0
        if isinstance(pbar, tqdm):
            pbar.close()
        self.running = False
        print(f'Island {self.name} has finished running')
        return trained_pops, top_inds, opt_times

    def interrupt_criteria(self):
        completion_criteria = (
            self.agents_migrated(),
            self.migration_criteria()
        )
        completed = any(completion_criteria)
        return completed

    def agents_migrated(self):
        agents_migrated = [len(policies) > 0 for agent_type, policies in self.migrated_from_neighbors.items()]
        agents_migrated = any(agents_migrated)
        return agents_migrated

    @staticmethod
    def migration_criteria():
        # todo  this could also include mcc criteria (criteria to migrate pops to another island)
        return False

    def stop(self):
        self.running = False
        return

    def incorporate_migrations(self):
        # todo  add new agents to current population and keep top N agents of all off original agents and new agents
        with self._update_lock:
            for agent_type, population in self.migrated_from_neighbors.items():
                num_agents = self.env.num_agent_types(agent_type)
                self.sort_population(agent_type)
                top_agents = self.agent_populations[agent_type]
                top_agents = top_agents[:num_agents]

                print(f'Island {self.name}: {time.time()}: incorporating {len(top_agents)} agents in population {agent_type}')
                # todo  replace or augment old populations?
                self.agent_populations[agent_type] = top_agents

            # reset view of migrated agents so that the same populations are not migrated repeatedly
            self.migrated_from_neighbors = {}
        return

    def add_from_neighbor(self, pop_id, population):
        print(f'Island {self.name}: {time.time()}: adding {len(population)} agents from neighbor')
        # todo  track who sent this population?
        with self._update_lock:
            if pop_id not in self.migrated_from_neighbors:
                self.migrated_from_neighbors[pop_id] = []

            self.migrated_from_neighbors[pop_id].extend(population)
        return

    def sort_population(self, agent_type):
        sorted_pop = sorted(self.agent_populations[agent_type], key=lambda x: x.fitness, reverse=True)
        self.agent_populations[agent_type] = sorted_pop
        return

    def migrate_to_neighbors(self):
        for each_neighbor in self.neighbors:
            for agent_type, population in self.agent_populations.items():
                if agent_type in self.evolving_agent_names:
                    num_agents = each_neighbor.env.num_agent_types(agent_type)
                    self.sort_population(agent_type)
                    top_agents = self.agent_populations[agent_type]
                    top_agents = top_agents[:num_agents]

                    print(f'Island {self.name}: {time.time()}: migrating {len(top_agents)} agents to neighbor {each_neighbor.name}')
                    each_neighbor.add_from_neighbor(agent_type, top_agents)
        return
