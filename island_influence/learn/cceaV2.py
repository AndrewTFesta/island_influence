"""
@title

@description

"""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import default_rng
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
#     """
#
#     :param agent_pops:
#     :param select_sizes:
#     :param epsilon:
#     :return: list with tuple of agents and indices, where the indices are the agents selected for evaluation
#     """
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


# def select_leniency(agent_pops, env, select_sizes: dict[AgentType, int]):
#     """
#
#     :param agent_pops:
#     :param env:
#     :param select_sizes:
#     :return: list with tuple of agents and indices, where the indices are the agents selected for evaluation
#     """
#     # todo  implement leniency
#     rng = default_rng()
#     best_policies = select_top_n(agent_pops, select_sizes=1)[0]
#     chosen_pops = []
#     for agent_name, policies in agent_pops.items():
#         # todo  weight select based on fitness
#         policies = rng.choice(policies, size=select_sizes)
#         for each_policy in policies:
#             entry = {
#                 name: policy if name != agent_name else each_policy
#                 for name, policy in best_policies.items()
#             }
#             chosen_pops.append([entry, [agent_name]])
#     return chosen_pops


def select_hall_of_fame(agent_pops, env, num_sims):
    """


    :param agent_pops:
    :param env:
    :param num_sims:
    :return: list with tuple of agents and indices, where the indices are the agents selected for evaluation
    """
    best_policies = select_top_n(agent_pops, select_sizes={name: env.num_agent_types(name) for name, pop in agent_pops.items()})
    all_selected_policies = select_roulette(agent_pops, select_sizes={agent_type: num_sims for agent_type in agent_pops.keys()})

    chosen_teams = []
    for agent_type, selected_policies in all_selected_policies.items():
        for each_policy in selected_policies:
            best_team = {each_type: [each_member for each_member in team_members] for each_type, team_members in best_policies.items()}
            collaborators = best_team[agent_type]
            collaborators.insert(0, each_policy)
            collaborators = collaborators[:-1]
            best_team[agent_type] = collaborators
            chosen_teams.append((best_team, {agent_type: (0,)}))
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
# def mutate_gaussian(agent_policies, mutation_scalar=0.1, probability_to_mutate=0.05):
#     mutated_agents = {}
#     for agent_name, individual in agent_policies.items():
#         # model_copy = individual.copy()
#         model_copy = copy.deepcopy(individual)
#         model_copy.mutate_gaussian(mutation_scalar=mutation_scalar, probability_to_mutate=probability_to_mutate)
#
#         model_copy.fitness = None
#         mutated_agents[agent_name] = model_copy
#     return mutated_agents


# def simulate_subpop(agent_policies, env, mutate_func):
#     mutated_policies = mutate_func(agent_policies[0])
#
#     # rollout and evaluate
#     agent_rewards = rollout(env, mutated_policies, render=False)
#     for agent_name, policy_info in mutated_policies.items():
#         policy_fitness = agent_rewards[agent_name]
#         policy_info['fitness'] = policy_fitness
#     return mutated_policies, agent_policies[1]


# def downselect_top_n(agent_pops, select_sizes: dict[AgentType, int]):
#     chosen_agent_pops = {}
#     for agent_type, population in agent_pops.items():
#         sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
#         select_size = select_sizes[agent_type]
#         top_pop = sorted_pop[:select_size]
#         chosen_agent_pops[agent_type] = top_pop
#     return chosen_agent_pops


def rollout(env: HarvestEnv, agent_policies, render: bool | dict = False):
    render_func = partial(env.render, **render) if isinstance(render, dict) else env.render

    # set policy to use for each learning agent
    for agent_type, policies in agent_policies.items():
        env.set_policies(agent_type, policies)

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
    agent_rewards = env.cumulative_rewards()
    # correlate policies to rewards
    policy_rewards = {each_agent.policy: agent_rewards[each_agent.name] for each_agent in env.agents}
    # episode_rewards = all_rewards[-1]
    # episode_rewards = reward_func(env)
    return agent_rewards, policy_rewards


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
    # 'mutate_gaussian': mutate_gaussian,
}

SIMULATION_FUNCTIONS = {
    # 'simulate_subpop': simulate_subpop,
}

DOWNSELECT_FUNCTIONS = {
    'downselect_top_n': select_top_n,
}


def ccea(env: HarvestEnv, agent_policies, population_sizes, num_gens, num_sims, experiment_dir,
         starting_gen=0, direct_assign_fitness=True, fitness_update_eps=1):
    # direct_assign_fitness = direct_assign_fitness

    # selection_func = partial(select_roulette, **{'select_size': num_simulations, 'noise': 0.01})
    selection_func = partial(select_hall_of_fame, **{'env': env, 'num_sims': num_sims})

    # mutate_func = partial(mutate_gaussian, mutation_scalar=0.1, probability_to_mutate=0.05)
    # sim_func = partial(simulate_subpop, **{'env': env, 'mutate_func': mutate_func})
    downselect_func = partial(select_top_n, **{'select_sizes': population_sizes})

    mutation_scalar = 0.1
    probability_to_mutate = 0.05

    env.save_environment(experiment_dir, tag='initial')

    # initial rollout to assign fitnesses of individuals on random teams
    #   pair first policies in each population together enough times to make a full team
    #   UCB style selection for networks?
    all_populations = list(agent_policies.values())
    all_lens = np.asarray([len(each_pop) for each_pop in all_populations])
    max_len = np.max(all_lens)
    all_teams = [
        {
            agent_type: [
                policies[idx % len(policies)]
                for _ in range(env.num_agent_types(agent_type))
            ]
            for agent_type, policies in agent_policies.items()
        }
        for idx in range(max_len)]
    for individuals in all_teams:
        agent_rewards, policy_rewards = rollout(env, individuals, render=False)
        for policy, reward in policy_rewards.items():
            policy.fitness = reward

    num_cores = multiprocessing.cpu_count()
    mp_pool = ProcessPoolExecutor(max_workers=num_cores - 1)
    for gen_idx in trange(starting_gen, num_gens):
        print(f'{gen_idx=}')

        # for agent_type, population in agent_policies.items():
        #     print(f'\t{agent_type}: {len(population)}')
        for each_obstacle in env.obstacles:
            print(f'{each_obstacle.value}', end=',')
        print()
        for each_poi in env.pois:
            print(f'{each_poi.value}', end=',')
        print()

        selected_policies = selection_func(agent_policies)

        teams = []
        for individuals, update_fitnesses in selected_policies:
            networks = []
            # new_team = individuals
            for agent_type, policy_idxs in update_fitnesses.items():
                for each_idx in policy_idxs:
                    policy_to_mutate = individuals[agent_type][each_idx]
                    model_copy = policy_to_mutate.copy()
                    model_copy.mutate_gaussian(mutation_scalar=mutation_scalar, probability_to_mutate=probability_to_mutate)
                    model_copy.fitness = None
                    # add mutated policies into respective agent_population
                    agent_policies[agent_type].append(model_copy)
                    networks.append(model_copy)
                    individuals[agent_type][each_idx] = model_copy
                # new_team[agent_type] = new_team
            teams.append((individuals, networks))

        # results = map(sim_func, selected_policies)
        # results = mp_pool.map(sim_func, selected_policies)ss)
        results = {}
        for individuals, update_fitnesses in teams:
            agent_rewards, policy_rewards = rollout(env, individuals, render=False)
            for policy_id in update_fitnesses:
                update_policy_reward = policy_rewards[policy_id]
                if policy_id not in results:
                    results[policy_id] = []
                results[policy_id].append(update_policy_reward)
        # average all fitnesses and assign back to agent
        avg_fitnesses = {each_agent: np.average(fitnesses) for each_agent, fitnesses in results.items()}

        if direct_assign_fitness:
            for agent, fitness in avg_fitnesses.items():
                agent.fitness = fitness
        else:
            for agent, fitness in avg_fitnesses.items():
                fitness_delta = fitness - agent.fitness
                agent.fitness += fitness_delta * fitness_update_eps

        # downselect
        agent_policies = downselect_func(agent_policies)

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
        # save_agent_policies(experiment_dir, gen_idx, env, agent_policies, fitnesses)
    # mp_pool.shutdown()

    best_policies = select_top_n(agent_policies, select_sizes={name: env.num_agent_types(name) for name, pop in agent_policies.items()})
    return best_policies
