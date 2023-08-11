"""
@title

@description

"""
import csv
import logging
import threading
import time
from pathlib import Path

from tqdm import tqdm


class MAIsland:

    def __init__(self, agent_populations, evolving_agent_names, env, optimizer, max_iters, save_dir, migrate_every=1,
                 name=None, track_progress=False, threaded=True, logger=None):
        if name is None:
            name = ':'.join([str(agent_type) for agent_type in evolving_agent_names])
            name = f'[{name}]'
        if logger is None:
            logger = logging.getLogger()

        self.name = f'MAIsland: {name}'
        self.logger = logger

        self.agent_populations = agent_populations
        self.evolving_agent_names = evolving_agent_names
        self.env = env

        self.migrate_every = migrate_every
        self.since_last_migration = 0
        self.optimizer_func = optimizer
        self.max_iters = max_iters
        self.track_progress = track_progress
        self.save_dir = save_dir
        self.times_fname = Path(self.save_dir, 'opt_times.csv')
        # todo  mark where migrations happen

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

        self.num_migrations = 0
        self.total_gens_run = 0
        self.opt_times = None
        self.final_pops = None
        self.top_inds = None
        return

    def __repr__(self):
        return f'{self.name}'

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)
        return

    def run(self):
        logging.info(msg=f'{self.name}: Starting optimizer')
        self.running = True
        self.optimize_call()
        return

    def _optimize(self):
        logging.debug(msg=f'Running island optimizer on thread: {threading.get_native_id()}')
        # todo  how would this work if it were it's own process?
        #       have to create a socket communication and then connect to other islands
        # run the optimize function to completion (as defined by the optimize function)
        self.total_gens_run = 0
        self.opt_times = []
        self.final_pops = None
        self.top_inds = None
        pbar = tqdm(total=self.max_iters, desc=f'{self.name}') if self.track_progress else None
        self.num_migrations = 0
        while self.running and self.total_gens_run < self.max_iters:
            if self.agents_migrated():
                self.incorporate_migrations()

            # todo  check calculating remaining gens
            #       seems to oboe on every migration
            opt_start = time.process_time()
            self.final_pops, self.top_inds, gens_run = self.optimizer_func(
                agent_policies=self.agent_populations, env=self.env, starting_gen=self.total_gens_run, max_iters=self.total_gens_run + self.migrate_every,
                completion_criteria=self.interrupt_criteria, track_progress=pbar, experiment_dir=self.save_dir
            )
            opt_end = time.process_time()
            opt_time = opt_end - opt_start
            self.opt_times.append(opt_time)
            self.total_gens_run += gens_run
            self.since_last_migration += gens_run

            total_time = sum(self.opt_times)
            iter_time_per_gen = opt_time / (gens_run + 0.001)
            overall_time_per_gen = total_time / self.total_gens_run

            remaining_gens = self.max_iters - self.total_gens_run
            expected_time_remaining = remaining_gens * overall_time_per_gen

            max_fitnesses = {agent_type: max([each_policy.fitness for each_policy in policies])for agent_type, policies in self.top_inds.items()}
            logging.info(msg=f'Island {self.name}:{self.total_gens_run}/{self.max_iters}:{total_time} seconds')
            logging.debug(msg=f'Island {self.name}:{max_fitnesses}')
            logging.debug(msg=f'Island {self.name}:{overall_time_per_gen} seconds per generation overall')
            logging.debug(msg=f'Island {self.name}:{gens_run} generations completed after {opt_time} seconds: {iter_time_per_gen} per generation')
            logging.debug(msg=f'Island {self.name}:{remaining_gens=}: {expected_time_remaining=}')
            with open(self.times_fname, 'w+') as times_file:
                writer = csv.writer(times_file)
                writer.writerow(self.opt_times)

            # at the end of every optimization loop (when the optimizer finishes),
            # migrate the champion from all learning population to every neighbor island
            if self.since_last_migration >= self.migrate_every:
                # this guards against an early migration when we stop the optimizer in order
                # to incorporate a new population that has been migrated from another island
                self.migrate_to_neighbors()
                self.num_migrations += 1
                self.since_last_migration = 0
        if isinstance(pbar, tqdm):
            pbar.close()
        self.running = False
        return self.final_pops, self.top_inds, self.opt_times

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
        logging.info(msg=f'{self.name}: stopping optimizer')
        self.running = False
        return

    def incorporate_migrations(self):
        with self._update_lock:
            for agent_type, population in self.migrated_from_neighbors.items():
                # determine how many old policies must be kept to satisfy env requirements
                # add the new policies with the current population
                num_agents = self.env.num_agent_types(agent_type)
                num_agents -= max(len(population), 0)
                self.sort_population(agent_type)
                new_agents = self.agent_populations[agent_type]
                if agent_type not in self.evolving_agent_names:
                    # keep the top N previous policies if this island is not evolving this type of agent
                    # this ensures that if integrating a new population, the current populations being
                    # evolved do not get thrown out and can still have an influence on the learning on this island
                    new_agents = new_agents[:num_agents]
                new_agents.extend(population)

                logging.debug(f'Incorporate: {time.time()}: {agent_type}: {len(new_agents)}: Island {self.name}')
                self.agent_populations[agent_type] = new_agents

            # reset view of migrated agents so that the same populations are not migrated repeatedly
            self.migrated_from_neighbors = {}
        return

    def add_from_neighbor(self, pop_id, population, from_neighbor):
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

                    logging.debug(f'Migration: {time.time()}: {len(top_agents)} agents: {self.name} -> {each_neighbor.name}')
                    each_neighbor.add_from_neighbor(agent_type, top_agents, self)
        return
