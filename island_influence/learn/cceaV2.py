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

from island_influence.harvest_env import HarvestEnv


# selection_functions
def select_roulette(agent_pops, select_size, noise=0.01):
    """
    :param agent_pops:
    :param select_size:
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
        rand_pop = np.random.choice(policy_population, select_size, replace=True, p=fitness_vals)
        chosen_agent_pops[agent_type] = rand_pop

    return chosen_agent_pops


def select_egreedy(agent_pops, select_size, epsilon):
    # todo implement egreedy selection
    rng = default_rng()
    chosen_agent_pops = {}
    for agent_name, policy_population in agent_pops.items():
        rand_val = rng.random()
        if rand_val <= epsilon:
            pass
        else:
            pass
        # chosen_agent_pops[agent_name] = rand_pop

    agent_names = list(chosen_agent_pops.keys())
    chosen_pops = [
        [{agent_name: pops[idx] for agent_name, pops in chosen_agent_pops.items()}, agent_names]
        for idx in range(0, select_size)
    ]
    return chosen_pops


def select_leniency(agent_pops, select_size):
    # todo  implement leniency
    rng = default_rng()
    best_policies = select_top_n(agent_pops, select_size=1)[0]
    chosen_pops = []
    for agent_name, policies in agent_pops.items():
        # todo  weight select based on fitness
        policies = rng.choice(policies, size=select_size)
        for each_policy in policies:
            entry = {
                name: policy if name != agent_name else each_policy
                for name, policy in best_policies.items()
            }
            chosen_pops.append([entry, [agent_name]])
    return chosen_pops


def select_hall_of_fame(agent_pops, num_sims):
    rng = default_rng()
    best_policies = select_top_n(agent_pops, select_sizes={name: 1 for name, pop in agent_pops.items()})
    best_policies = {agent_type: policies[0] for agent_type, policies in best_policies.items()}
    all_selected_policies = select_roulette(agent_pops, select_size=num_sims)
    # todo  finish implementing hof

    chosen_pops = []
    for agent_type, agent_selected_policies in all_selected_policies.items():
        pass

    for agent_name, policies in agent_pops.items():
        # todo  weight select based on fitness
        policies = rng.choice(policies, size=num_sims)
        for each_policy in policies:
            entry = {
                name: policy if name != agent_name else each_policy
                for name, policy in best_policies.items()
            }
            chosen_pops.append([entry, [agent_name]])
    return chosen_pops


def select_top_n(agent_pops, select_sizes):
    chosen_agent_pops = {}
    for agent_name, population in agent_pops.items():
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        top_pop = sorted_pop[:select_sizes[agent_name]]
        chosen_agent_pops[agent_name] = top_pop
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


def downselect_top_n(agent_pops, select_size):
    chosen_agent_pops = {}
    for agent_name, population in agent_pops.items():
        sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
        top_pop = sorted_pop[:select_size]
        chosen_agent_pops[agent_name] = top_pop
    return chosen_agent_pops


def rollout(env: HarvestEnv, individuals, render: bool | dict = False):
    render_func = partial(env.render, **render) if isinstance(render, dict) else env.render

    observations = env.reset()
    agent_dones = env.done()
    done = all(agent_dones.values())

    # set policy to use for each learning agent
    for agent_name, agent in individuals.items():
        env.agents[agent_name].policy = agent.policy

    all_rewards = []
    while not done:
        next_actions = env.get_actions()
        observations, rewards, agent_dones, truncs, infos = env.step(next_actions)
        all_rewards.append(rewards)
        done = all(agent_dones.values())
        if render:
            render_func()

    episode_rewards = all_rewards[-1]
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
    'select_egreedy': select_egreedy,
    'select_leniency': select_leniency,
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


def ccea(env: HarvestEnv, agent_pops, population_sizes, num_gens, num_sims, experiment_dir, starting_gen=0):
    # selection_func = partial(select_roulette, **{'select_size': num_simulations, 'noise': 0.01})
    selection_func = partial(select_hall_of_fame, **{'num_sims': num_sims})

    mutate_func = partial(mutate_gaussian, mutation_scalar=0.1, probability_to_mutate=0.05)
    # sim_func = partial(simulate_subpop, **{'env': env, 'mutate_func': mutate_func})
    downselect_func = partial(downselect_top_n, **{'select_sizes': population_sizes})

    env.save_environment(experiment_dir, tag='initial')

    # todo  initial rollout to assign fitnesses of individuals on random teams
    #       pair each agent in a population with first agent in other agents' populations
    team_members = {
        agent_type: population[0]
        for agent_type, population in agent_pops.items()
    }
    initial_teams = []
    for agent_type, population in agent_pops.items():
        for individual in population:
            ind_team = {each_type: each_member for each_type, each_member in team_members.items()}
            ind_team[agent_type] = individual
            initial_teams.append(ind_team)
    for each_team in initial_teams:
        agent_rewards = rollout(env, each_team, render=False)

    # num_cores = multiprocessing.cpu_count()
    # mp_pool = ProcessPoolExecutor(max_workers=num_cores - 1)
    for gen_idx in trange(starting_gen, num_gens):
        selected_policies = selection_func(agent_pops)

        # results = []
        # for each_selection in selected_policies:
        #     each_result = sim_func(each_selection)
        #     results.append(each_result)
        # # results = map(sim_func, selected_policies)
        # # results = mp_pool.map(sim_func, selected_policies)
        #
        # for each_result in results:
        #     eval_agents = each_result[1]
        #     # reinsert new individual into population of policies if this result was meant to be
        #     # evaluating a particular agent
        #     # e.g. for hall of fame or leniency, each entry in selected_policies should only be
        #     # evaluating a single policy
        #     for name, policy in each_result[0].items():
        #         if name in eval_agents:
        #             agent_pops[name].append(policy)
        #
        # # downselect
        # agent_pops = downselect_func(agent_pops)
        #
        # top_inds = select_top_n(agent_pops, select_size=1)[0]
        # _ = rollout(env, top_inds, render=False)
        # g_reward = env.calc_global()
        # g_reward = list(g_reward.values())[0]
        # # todo  bug fix sometimes there is more than population_size policies in the population
        # fitnesses = {
        #     agent_name: [each_individual['fitness'] for each_individual in policy_info]
        #     for agent_name, policy_info in agent_pops.items()
        # }
        # fitnesses['G'] = [g_reward for _ in range(population_sizes)]
        #
        # # save all policies of each agent and save fitnesses mapping policies to fitnesses
        # save_agent_policies(experiment_dir, gen_idx, env, agent_pops, fitnesses)
    # mp_pool.shutdown()

    top_inds = select_top_n(agent_pops, select_size=1)[0]
    return top_inds
