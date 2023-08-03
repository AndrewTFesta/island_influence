"""
@title

@description

"""
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.random import default_rng
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import trange

from island_influence.agent import AgentType
from island_influence.harvest_env import HarvestEnv


# selection_functions
def select_roulette(agent_pops, select_sizes: dict[AgentType, int], noise=0.01):
    """
    :param agent_pops:
    :param select_sizes:
    :param noise:
    :return:
    """
    chosen_agent_pops = {}
    for agent_type, policy_population in agent_pops.items():
        fitness_vals = np.asarray([each_policy.fitness for each_policy in policy_population])

        # add small amount of noise to each fitness value (help deal with all same value)
        noise = np.random.uniform(0, noise, len(fitness_vals))
        fitness_vals += noise

        fitness_vals = fitness_vals / sum(fitness_vals)
        select_size = select_sizes[agent_type]
        rand_pop = np.random.choice(policy_population, select_size, replace=True, p=fitness_vals)
        chosen_agent_pops[agent_type] = rand_pop

    return chosen_agent_pops


# def select_egreedy(agent_pops, select_sizes: dict[AgentType, int], epsilon):
#     # todo implement egreedy selection
#     rng = default_rng()
#     chosen_agent_pops = {}
#     for agent_name, policy_population in agent_pops.items():
#         rand_val = rng.random()
#         if rand_val <= epsilon:
#             pass
#         else:
#             pass
#         # chosen_agent_pops[agent_name] = rand_pop
#
#     agent_names = list(chosen_agent_pops.keys())
#     select_size = select_sizes[agent_type]
#     chosen_pops = [
#         [{agent_name: pops[idx] for agent_name, pops in chosen_agent_pops.items()}, agent_names]
#         for idx in range(0, select_size)
#     ]
#     return chosen_pops


# def select_leniency(agent_pops, select_sizes: dict[AgentType, int]):
#     # todo  implement leniency
#     rng = default_rng()
#     best_policies = select_top_n(agent_pops, select_size=1)[0]
#     chosen_pops = []
#     for agent_name, policies in agent_pops.items():
#         # todo  weight select based on fitness
#         policies = rng.choice(policies, size=select_size)
#         for each_policy in policies:
#             entry = {
#                 name: policy if name != agent_name else each_policy
#                 for name, policy in best_policies.items()
#             }
#             chosen_pops.append([entry, [agent_name]])
#     return chosen_pops


def select_hall_of_fame(agent_pops, env, num_sims):
    best_policies = select_top_n(agent_pops, select_sizes={name: env.get_num_agent_type(name) for name, pop in agent_pops.items()})
    all_selected_policies = select_roulette(agent_pops, select_sizes={agent_type: num_sims for agent_type in agent_pops.keys()})

    chosen_teams = []
    for agent_type, agent_selected_policies in all_selected_policies.items():
        for each_agent in agent_selected_policies:
            best_team = {each_type: [each_member for each_member in team_members] for each_type, team_members in best_policies.items()}
            collaborators = best_team[agent_type]
            collaborators.insert(0, each_agent)
            # todo  check for duplicates
            #       an agent should not be paired with itself
            collaborators = collaborators[:-1]
            best_team[agent_type] = collaborators
            chosen_teams.append((best_team, (each_agent,)))
    return chosen_teams


def select_top_n(agent_pops, select_sizes: dict[AgentType, int]):
    chosen_agent_pops = {}
    for agent_type, population in agent_pops.items():
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        select_size = select_sizes[agent_type]
        top_pop = sorted_pop[:select_size]
        chosen_agent_pops[agent_type] = top_pop
    return chosen_agent_pops


# Mutation Functions
def mutate_gaussian(agent_policies, mutation_scalar=0.1, probability_to_mutate=0.05):
    mutated_agents = {}
    for agent_name, individual in agent_policies.items():
        model = individual['network']
        model_copy = copy.deepcopy(model)

        rng = default_rng()
        with torch.no_grad():
            param_vector = parameters_to_vector(model_copy.parameters())

            for each_val in param_vector:
                rand_val = rng.random()
                if rand_val <= probability_to_mutate:
                    # todo  base proportion on current weight rather than scaled random sample
                    noise = torch.randn(each_val.size()) * mutation_scalar
                    each_val.add_(noise)

            vector_to_parameters(param_vector, model_copy.parameters())
        new_ind = {
            'network': model_copy,
            'fitness': None
        }
        mutated_agents[agent_name] = new_ind
    return mutated_agents


# def simulate_subpop(agent_policies, env, mutate_func):
#     mutated_policies = mutate_func(agent_policies[0])
#
#     # rollout and evaluate
#     agent_rewards = rollout(env, mutated_policies, render=False)
#     for agent_name, policy_info in mutated_policies.items():
#         policy_fitness = agent_rewards[agent_name]
#         policy_info['fitness'] = policy_fitness
#     return mutated_policies, agent_policies[1]


def downselect_top_n(agent_pops, select_sizes: dict[AgentType, int]):
    chosen_agent_pops = {}
    for agent_type, population in agent_pops.items():
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        select_size = select_sizes[agent_type]
        top_pop = sorted_pop[:select_size]
        chosen_agent_pops[agent_type] = top_pop
    return chosen_agent_pops


def rollout(env: HarvestEnv, individuals, render: bool | dict = False):
    render_func = partial(env.render, **render) if isinstance(render, dict) else env.render

    # set policy to use for each learning agent
    env.set_agents(individuals)
    env.reset()

    _ = env.reset()
    agent_dones = env.done()
    done = all(agent_dones.values())

    all_rewards = []
    while not done:
        next_actions = env.get_actions()
        observations, rewards, agent_dones, truncs, infos = env.step(next_actions)
        all_rewards.append(rewards)
        done = all(agent_dones.values())
        if render:
            render_func()

    # assign rewards based on the cumulative rewards of the env
    episode_rewards = env.cumulative_rewards()
    # episode_rewards = all_rewards[-1]
    # episode_rewards = reward_func(env)
    return episode_rewards


def save_agent_policies(experiment_dir, gen_idx, env, agent_pops, fitnesses):
    gen_path = Path(experiment_dir, f'gen_{gen_idx}')
    if not gen_path:
        gen_path.mkdir(parents=True, exist_ok=True)

    env.save_environment(gen_path, tag=f'gen_{gen_idx}')
    for agent_name, policy_info in agent_pops.items():
        network_save_path = Path(gen_path, f'{agent_name}_networks')
        if not network_save_path:
            network_save_path.mkdir(parents=True, exist_ok=True)

        for idx, each_policy in enumerate(policy_info):
            # fitnesses[agent_name].append(each_policy['fitness'])
            network = each_policy['network']
            network.save_model(save_dir=network_save_path, tag=f'{idx}')

    fitnesses_path = Path(gen_path, 'fitnesses.csv')
    fitnesses_df = pd.DataFrame.from_dict(fitnesses, orient='index')
    fitnesses_df.to_csv(fitnesses_path, header=True, index_label='agent_name')
    return


SELECTION_FUNCTIONS = {
    'select_roulette': select_roulette,
    # 'select_egreedy': select_egreedy,
    # 'select_leniency': select_leniency,
    'select_hall_of_fame': select_hall_of_fame,
    'select_top_n': select_top_n,
}

MUTATION_FUNCTIONS = {
    'mutate_gaussian': mutate_gaussian,
}

# SIMULATION_FUNCTIONS = {
#     'simulate_subpop': simulate_subpop,
# }

DOWNSELECT_FUNCTIONS = {
    'downselect_top_n': downselect_top_n,
}


def ccea(env: HarvestEnv, agent_pops, population_sizes, num_gens, sims_per_atype, experiment_dir,
         starting_gen=0, direct_assign_fitness=True, fitness_update_eps=1):
    direct_assign_fitness = direct_assign_fitness
    max_fitness = 100.0

    # selection_func = partial(select_roulette, **{'select_size': num_simulations, 'noise': 0.01})
    selection_func = partial(select_hall_of_fame, **{'env': env, 'num_sims': sims_per_atype})

    mutate_func = partial(mutate_gaussian, mutation_scalar=0.1, probability_to_mutate=0.05)
    # sim_func = partial(simulate_subpop, **{'env': env, 'mutate_func': mutate_func})
    downselect_func = partial(downselect_top_n, **{'select_sizes': population_sizes})

    env.save_environment(experiment_dir, tag='initial')

    # todo  initial rollout to assign fitnesses of individuals on random teams
    #       pair each agent in a population with first agent in other agents' populations
    # todo  UCB style selection for networks?
    for agent_type, population in agent_pops.items():
        for individual in population:
            individual.fitness = max_fitness

    # team_members = {
    #     agent_type: population[0]
    #     for agent_type, population in agent_pops.items()
    # }
    # initial_teams = []
    # for agent_type, population in agent_pops.items():
    #     for individual in population:
    #         ind_team = {each_type: each_member for each_type, each_member in team_members.items()}
    #         ind_team[agent_type] = individual
    #         initial_teams.append(ind_team)
    #
    # for each_team in initial_teams:
    #     agent_rewards = rollout(env, each_team, render=False)

    num_cores = multiprocessing.cpu_count()
    mp_pool = ProcessPoolExecutor(max_workers=num_cores - 1)
    for gen_idx in trange(starting_gen, num_gens):
        selected_policies = selection_func(agent_pops)
        # todo  mutate selected policies

        results = []
        for individuals, update_fitnesses in selected_policies:
            episode_rewards = rollout(env, individuals, render=False)
            results.append((episode_rewards, update_fitnesses))
        # results = map(sim_func, selected_policies)
        # results = mp_pool.map(sim_func, selected_policies)

        all_agent_fitnesses = {}
        for fitnesses, eval_agents in results:
            # each_result stores the rewards of the agents assigned to be evaluated
            #   e.g. for hall of fame or leniency, each entry in selected_policies should only be evaluating a single policy
            for each_agent in eval_agents:
                if each_agent not in all_agent_fitnesses:
                    all_agent_fitnesses[each_agent] = []

                agent_fitness = fitnesses[each_agent.name]
                all_agent_fitnesses[each_agent].append(agent_fitness)
        # average all fitnesses and assign back to agent
        avg_fitnesses = {each_agent: np.average(fitnesses) for each_agent, fitnesses in all_agent_fitnesses.items()}

        if direct_assign_fitness:
            for agent, fitness in avg_fitnesses.items():
                agent.fitness = fitness
        else:
            for agent, fitness in avg_fitnesses.items():
                fitness_delta = fitness - agent.fitness
                agent.fitness += fitness_delta * fitness_update_eps

        # downselect
        agent_pops = downselect_func(agent_pops)

        # top_inds = select_top_n(agent_pops, select_size=1)[0]
        # _ = rollout(env, top_inds, render=False)
        # g_reward = env.calc_global()
        # g_reward = list(g_reward.values())[0]

        # fitnesses = {
        #     agent_name: [each_individual['fitness'] for each_individual in policy_info]
        #     for agent_name, policy_info in agent_pops.items()
        # }
        # fitnesses['G'] = [g_reward for _ in range(population_sizes)]

        # # save all policies of each agent and save fitnesses mapping policies to fitnesses
        # save_agent_policies(experiment_dir, gen_idx, env, agent_pops, fitnesses)
    # mp_pool.shutdown()

    top_inds = select_top_n(agent_pops, select_sizes={agent_type: 1 for agent_type in population_sizes.keys()})[0]
    return top_inds
