"""
@title

@description

"""
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

from island_influence.agent import AgentType
from island_influence.harvest_env import HarvestEnv


def filter_learning_policies(agent_policies):
    filtered_pops = {}
    for agent_type, population in agent_policies.items():
        learners = [individual for individual in population if individual.learner]
        if len(learners) > 0:
            filtered_pops[agent_type] = learners
    return filtered_pops


def filter_no_fitness(agent_policies):
    filtered_pops = {
        agent_type: [individual for individual in population if individual.fitness is None]
        for agent_type, population in agent_policies.items()
    }
    return filtered_pops


def find_policy(policy_name, agent_populations):
    for agent_type, population in agent_populations:
        for each_policy in population:
            if each_policy.name == policy_name:
                return each_policy
    return None


# selection_functions
def select_roulette(agent_pops, select_sizes: dict[AgentType, int], noise=0.01, filter_learners=False):
    """
    :param agent_pops:
    :param select_sizes:
    :param noise:
    :param filter_learners:
    :return:
    """
    if filter_learners:
        agent_pops = filter_learning_policies(agent_pops)

    chosen_agent_pops = {}
    for agent_type, policy_population in agent_pops.items():
        fitness_vals = np.asarray([each_policy.fitness for each_policy in policy_population])

        # add small amount of noise to each fitness value (help deal with all same value)
        fitness_noise = np.random.uniform(0, noise, len(fitness_vals))
        fitness_vals += fitness_noise

        fitness_vals = fitness_vals / sum(fitness_vals)
        select_size = min(len(policy_population), select_sizes[agent_type])
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


def select_hall_of_fame(agent_pops, env, num_sims, filter_learners=False):
    """


    :param agent_pops:
    :param env:
    :param num_sims:
    :param filter_learners:
    :return: list with tuple of agents and indices, where the indices are the agents selected for evaluation
    """
    best_policies = select_top_n(agent_pops, select_sizes={name: env.num_agent_types(name) for name, pop in agent_pops.items()})
    test_policies = select_roulette(agent_pops, select_sizes={agent_type: num_sims for agent_type in agent_pops.keys()}, filter_learners=filter_learners)

    chosen_teams = []
    for agent_type, selected_policies in test_policies.items():
        for each_policy in selected_policies:
            best_team = {each_type: [each_member for each_member in team_members] for each_type, team_members in best_policies.items()}
            collaborators = best_team[agent_type]
            collaborators.insert(0, each_policy)
            collaborators = collaborators[:-1]
            best_team[agent_type] = collaborators
            chosen_teams.append((best_team, {agent_type: (0,)}))
    return chosen_teams


def select_top_n(agent_pops, select_sizes: dict[AgentType, int], filter_learners=False):
    if filter_learners:
        agent_pops = filter_learning_policies(agent_pops)

    chosen_agent_pops = {}
    for agent_type, policy_population in agent_pops.items():
        sorted_pop = sorted(policy_population, key=lambda x: x.fitness, reverse=True)
        select_size = min(len(policy_population), select_sizes[agent_type])
        top_pop = sorted_pop[:select_size]
        chosen_agent_pops[agent_type] = top_pop
    return chosen_agent_pops


def evaluate_agents(agent_team, env):
    team_members, update_policy_ids = agent_team
    agent_rewards, policy_rewards = rollout(env, team_members, render=False)
    eval_policy_rewards = {policy.name: policy_reward for policy, policy_reward in policy_rewards.items() if policy in update_policy_ids}
    return eval_policy_rewards


def rollout(env: HarvestEnv, agent_policies, render: bool = False):
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
            # TypeError: HarvestEnv.render() got an unexpected keyword argument 'window_size'
            render_func()

    # assign rewards based on the cumulative rewards of the env
    agent_rewards = env.cumulative_rewards()
    # correlate policies to rewards
    policy_rewards = {each_agent.policy: agent_rewards[each_agent.name] for each_agent in env.agents}
    # episode_rewards = all_rewards[-1]
    # episode_rewards = reward_func(env)
    return agent_rewards, policy_rewards


def save_agent_policies(experiment_dir, gen_idx, env, agent_pops, fitnesses, human_readable=False):
    indent = 2 if human_readable else None
    gen_path = Path(experiment_dir, f'gen_{gen_idx}')
    if not gen_path:
        gen_path.mkdir(parents=True, exist_ok=True)

    env.save_environment(gen_path, tag=f'gen_{gen_idx}')
    for agent_name, policies in agent_pops.items():
        network_save_path = Path(gen_path, f'{agent_name}_networks')
        if not network_save_path:
            network_save_path.mkdir(parents=True, exist_ok=True)

        for idx, each_policy in enumerate(policies):
            # fitnesses[agent_name].append(each_policy.fitness)
            each_policy.save_model(save_dir=network_save_path, tag=f'{idx}')

    fitnesses_path = Path(gen_path, 'fitnesses.json')
    with open(fitnesses_path, 'w') as fitness_file:
        json.dump(fitnesses, fitness_file, indent=indent)
    return


def ccea(env: HarvestEnv, agent_policies, population_sizes, max_iters, num_sims, experiment_dir, completion_criteria=lambda: False,
         starting_gen=0, direct_assign_fitness=True, fitness_update_eps=1, mutation_scalar=0.1, prob_to_mutate=0.05, track_progress=True, use_mp=False):
    """
    agents in agent_policies are the actual agents being optimized
    the non-learning agents are generally expected to be an inherent part of the environment

    :param env:
    :param agent_policies:
    :param population_sizes:
    :param max_iters:
    :param num_sims:
    :param experiment_dir:
    :param completion_criteria:
    :param starting_gen:
    :param direct_assign_fitness:
    :param fitness_update_eps:
    :param mutation_scalar:
    :param prob_to_mutate:
    :param track_progress:
    :param use_mp:
    :return:
    """
    population_sizes = {agent_type: max(pop_size, env.num_agent_types(agent_type)) for agent_type, pop_size in population_sizes.items()}
    selection_func = partial(select_hall_of_fame, **{'env': env, 'num_sims': num_sims, 'filter_learners': True})
    eval_func = partial(evaluate_agents, env=env)
    downselect_func = partial(select_top_n, **{'select_sizes': population_sizes})
    ##########################################################################################
    # only run initialize if there are any policies with no fitness assigned
    # initial rollout to assign fitnesses of individuals on random teams
    #   pair first policies in each population together enough times to make a full team
    #   UCB style selection for networks?
    filtered_pops = filter_no_fitness(agent_policies)
    filtered_lens = {agent_type: len(each_pop) for agent_type, each_pop in filtered_pops.items()}
    max_len = np.max(np.asarray(list(filtered_lens.values())))
    if max_len != 0:
        # if a population has no unassigned fitnesses, add in the best policy from that initial population
        all_teams = [
            {
                agent_type: [
                    policies[idx % len(policies)] if len(policies) > 0 else filtered_pops[agent_type][0]
                    for _ in range(env.num_agent_types(agent_type))
                ]
                for agent_type, policies in agent_policies.items()
            }
            for idx in range(max_len)]
        for individuals in all_teams:
            agent_rewards, best_policy_rewards = rollout(env, individuals, render=False)
            for policy, fitness in best_policy_rewards.items():
                if policy.learner:
                    policy.fitness = fitness
    ##########################################################################################
    best_agent_fitnesses = {agent.name: 0 for agent in env.agents}
    best_agent_fitnesses['harvest_team'] = 0
    best_agent_fitnesses['excavator_team'] = 0
    team_fitness = {'team': 0}

    map_func = map
    mp_pool = None
    if use_mp:
        num_cores = multiprocessing.cpu_count()
        max_workers = num_cores - 1
        mp_pool = ProcessPoolExecutor(max_workers=max_workers)
        map_func = mp_pool.map
        print(f'Running mp pool with {max_workers} workers')

    gen_idx = 0
    pbar = None
    if track_progress:
        pbar = tqdm(total=max_iters, desc=f'Generation', postfix=team_fitness)
        pbar.update(starting_gen)

    for gen_idx in range(starting_gen, max_iters):
        selected_policies = selection_func(agent_policies)

        ###############################################################################################
        # results = map(eval_func, selected_policies)
        # # results = mp_pool.map(sim_func, selected_policies)
        teams = []
        for individuals, update_fitnesses in selected_policies:
            networks = []
            for agent_type, policy_idxs in update_fitnesses.items():
                for each_idx in policy_idxs:
                    policy_to_mutate = individuals[agent_type][each_idx]
                    model_copy = policy_to_mutate.copy()
                    model_copy.mutate_gaussian(mutation_scalar=mutation_scalar, probability_to_mutate=prob_to_mutate)
                    model_copy.fitness = None
                    # add mutated policies into respective agent_population
                    agent_policies[agent_type].append(model_copy)
                    networks.append(model_copy)
                    individuals[agent_type][each_idx] = model_copy
            teams.append((individuals, networks))

        ###############################################################################################
        rollout_results = map_func(eval_func, teams)
        # rollout_results = []
        # for individuals, update_fitnesses in teams:
        #     agent_rewards, best_policy_rewards = rollout(env, individuals, render=False)
        #     eval_policy_rewards = {policy_id: policy_reward for policy_id, policy_reward in best_policy_rewards.items() if policy_id in update_fitnesses}
        #     rollout_results.append(eval_policy_rewards)
        ###############################################################################################
        agent_results = {}
        for each_result in rollout_results:
            for each_policy_name, fitness in each_result.items():
                if each_policy_name not in agent_results:
                    agent_results[each_policy_name] = []
                agent_results[each_policy_name].append(fitness)
        ###############################################################################################
        # average all fitnesses and assign back to agent
        avg_fitnesses = {each_agent: np.average(fitnesses) for each_agent, fitnesses in agent_results.items()}

        for agent_type, population in agent_policies.items():
            for policy in population:
                if policy.name in avg_fitnesses and policy.learner:
                    fitness = avg_fitnesses[policy.name]
                    if direct_assign_fitness:
                        policy.fitness = fitness
                    else:
                        fitness_delta = fitness - policy.fitness
                        policy.fitness += fitness_delta * fitness_update_eps

        # downselect
        agent_policies = downselect_func(agent_policies)

        # save generation progress
        best_policies = select_top_n(agent_policies, select_sizes={name: env.num_agent_types(name) for name, pop in agent_policies.items()})
        best_agent_fitnesses, best_policy_rewards = rollout(env, best_policies, render=False)

        fitnesses = {
            str(agent_name): [each_individual.fitness for each_individual in policy]
            for agent_name, policy in agent_policies.items()
        }
        fitnesses['harvest_team'] = best_agent_fitnesses['harvest_team']
        fitnesses['excavator_team'] = best_agent_fitnesses['excavator_team']
        fitnesses['team'] = best_agent_fitnesses['team']
        team_fitness['team'] = best_agent_fitnesses['team']

        # save all policies of each agent and save fitnesses mapping policies to fitnesses
        save_agent_policies(experiment_dir, gen_idx, env, agent_policies, fitnesses)
        if isinstance(pbar, tqdm):
            pbar.update(1)
        if completion_criteria():
            break
    if mp_pool:
        mp_pool.shutdown()
    if isinstance(pbar, tqdm):
        pbar.close()

    best_policies = select_top_n(agent_policies, select_sizes={name: env.num_agent_types(name) for name, pop in agent_policies.items()})
    return agent_policies, best_policies, gen_idx
