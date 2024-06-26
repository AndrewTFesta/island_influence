"""
@title

@description

"""
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from island_influence.agent import AgentType
from island_influence.envs.harvest_env import HarvestEnv
from island_influence.utils import save_config


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
        fitness_noise = np.random.uniform(-noise / 2, noise / 2, len(fitness_vals))
        # numpy.core._exceptions._UFuncOutputCastingError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'
        fitness_vals = np.add(fitness_vals, fitness_noise)

        fitness_vals -= np.min(fitness_vals)
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


def select_hall_of_fame(agent_pops, env: HarvestEnv, num_sims, filter_learners=False):
    """


    :param agent_pops:
    :param env:
    :param num_sims:
    :param filter_learners:
    :return: list with tuple of agents and indices, where the indices are the agents selected for evaluation
    """
    best_policies = select_top_n(agent_pops, select_sizes={name: env.types_num[name] for name, pop in agent_pops.items()})
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


def evaluate_agents(agent_team, env: HarvestEnv):
    team_members, update_policy_ids = agent_team
    agent_rewards, policy_rewards = rollout(env, team_members, render=False)
    eval_policy_rewards = {
        policy.name: {'individual_reward': policy_reward, 'policy_rewards': policy_rewards, 'agent_rewards': agent_rewards}
        for policy, policy_reward in policy_rewards.items()
        if policy in update_policy_ids
    }
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
            render_func()

    # assign rewards based on the cumulative rewards of the env
    agent_rewards = env.cumulative_agent_rewards()
    policy_rewards = {env.get_actor(agent_name): {'reward': agent_rewards[agent_name], 'agent_name': agent_name} for agent_name in env.agents}
    return agent_rewards, policy_rewards


def get_policy(population, policy_name):
    policy = None
    for each_policy in population:
        if each_policy.name == policy_name:
            policy = each_policy
            break
    return policy


def save_agent_policies(experiment_dir, gen_idx, env: HarvestEnv, agent_pops, best_policies, human_readable=False):
    indent = 2 if human_readable else None
    networks_dir = Path(experiment_dir, f'networks')
    fitnesses_path = Path(experiment_dir, f'gen_{gen_idx}_fitnesses.json')

    if not networks_dir.exists():
        networks_dir.mkdir(parents=True, exist_ok=True)

    if not fitnesses_path.parent.exists():
        fitnesses_path.parent.mkdir(parents=True, exist_ok=True)

    fitnesses = {
        str(agent_name): [each_individual.fitness for each_individual in policy]
        for agent_name, policy in agent_pops.items()
    }

    best_agent_fitnesses, policy_rewards = rollout(env, best_policies, render=False)
    fitnesses['global_harvester'] = best_agent_fitnesses['global_harvester']
    fitnesses['global_excavator'] = best_agent_fitnesses['global_excavator']
    fitnesses['global'] = best_agent_fitnesses['global_harvester']

    for agent_name, policies in agent_pops.items():
        network_save_path = Path(networks_dir, f'{agent_name}_networks')
        if not network_save_path:
            network_save_path.mkdir(parents=True, exist_ok=True)

        # also overwrite any policies already in pool
        # in case they might have been updated in some way
        for idx, each_policy in enumerate(policies):
            each_policy.save_model(save_dir=network_save_path)

    with open(fitnesses_path, 'w') as fitness_file:
        json.dump(fitnesses, fitness_file, indent=indent)
    return


def generation(agent_policies, map_func, selection_func, mutation_scalar, prob_to_mutate, eval_func, fitness_update_eps, downselect_func):
    # todo  currently, only new policies are evaluated as the policies are selected, mutated, and the mutated policies are evaluated
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
    best_global = -math.inf
    best_result = {}
    policy_results = {}
    for each_result in rollout_results:
        for each_policy_name, result_info in each_result.items():
            if each_policy_name not in policy_results:
                policy_results[each_policy_name] = []
            policy_results[each_policy_name].append(result_info)

            global_reward = result_info['agent_rewards']
            global_reward = global_reward['global']
            if global_reward > best_global:
                best_global = global_reward
                best_result = result_info['policy_rewards']
    best_team = {}
    for policy, policy_info in best_result.items():
        agent_type = policy_info['agent_name'].split(':')[0]
        if agent_type not in best_team:
            best_team[agent_type] = []
        best_team[agent_type].append(policy)
    ###############################################################################################
    # need to have the eval function pass back the policy name rather than the policy directly because when passing
    # a network to a new process, it creates a new object and so is no longer the original policy
    eval_agent_names = [each_policy for each_policy in policy_results.keys()]

    for agent_type, population in agent_policies.items():
        for policy in population:
            # todo  verify - any agent name in assignment_agents should correspond to a learning agent
            #       should never select a non-learning agent for evaluation
            # if policy.name in eval_agent_names and policy.learner:
            if policy.name in eval_agent_names:
                assert policy.learner
                each_result = policy_results[policy.name]
                policy.fitness_history.append(each_result)

                # average all fitnesses and assign back to agent
                fitness = np.average([rollout_eval['individual_reward']['reward'] for rollout_eval in each_result])

                # if fitness update eps is 0, then evaluating a fitness would have no change on the current fitness of the agent
                # can also pass a negative value to force the evaluated fitness to replace the fitness value of the policy
                if fitness_update_eps <= 0:
                    policy.fitness = fitness
                else:
                    # todo  dan't have a delta if the policy does not have a fitness yet
                    fitness_delta = fitness - policy.fitness
                    policy.fitness += fitness_delta * fitness_update_eps

    agent_policies = downselect_func(agent_policies)
    return agent_policies, best_team


def ccea(env: HarvestEnv, agent_policies, population_sizes, max_iters, num_sims, experiment_dir, completion_criteria=lambda: False,
         starting_gen=0, fitness_update_eps: float = 0, mutation_scalar=0.1, prob_to_mutate=0.05, track_progress=True,
         use_mp=False):
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
    :param fitness_update_eps:
    :param mutation_scalar:
    :param prob_to_mutate:
    :param track_progress:
    :param use_mp:
    :return:
    """
    population_sizes = {agent_type: max(pop_size, env.types_num[agent_type]) for agent_type, pop_size in population_sizes.items()}
    selection_func = partial(select_hall_of_fame, **{'env': env, 'num_sims': num_sims, 'filter_learners': True})
    eval_func = partial(evaluate_agents, env=env)
    downselect_func = partial(select_top_n, **{'select_sizes': population_sizes})
    generation_func = partial(
        generation, selection_func=selection_func, mutation_scalar=mutation_scalar, prob_to_mutate=prob_to_mutate,
        eval_func=eval_func, fitness_update_eps=fitness_update_eps, downselect_func=downselect_func
    )
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
                    for _ in range(env.types_num[agent_type])
                ]
                for agent_type, policies in agent_policies.items()
            }
            for idx in range(max_len)]
        for individuals in all_teams:
            agent_rewards, policy_rewards = rollout(env, individuals, render=False)
            for policy, fitness_info in policy_rewards.items():
                if policy.learner:
                    policy.fitness = fitness_info['reward']
    ##########################################################################################
    ccea_config_fname = Path(experiment_dir, 'ccea_config.json')
    exp_started = ccea_config_fname.exists() and ccea_config_fname.is_file()
    if not exp_started:
        ccea_config = {
            'max_iters': max_iters, 'starting_gen': starting_gen,
            'num_sims': num_sims,
            # 'num_sims': {str(agent_type): size for agent_type, size in num_sims.items()},
            'population_sizes': {str(agent_type): size for agent_type, size in population_sizes.items()},
            'fitness_update_eps': fitness_update_eps, 'mutation_scalar': mutation_scalar, 'prob_to_mutate': prob_to_mutate,

        }
        save_config(config=ccea_config, save_dir=experiment_dir, config_name='ccea_config')
        env.save_environment(save_dir=experiment_dir)

        best_policies = select_top_n(agent_policies, select_sizes={name: env.types_num[name] for name, pop in agent_policies.items()})
        # save initial fitnesses
        save_agent_policies(experiment_dir, 0, env, agent_policies, best_policies)
    ##########################################################################################
    map_func = map
    mp_pool = None
    if use_mp:
        num_cores = multiprocessing.cpu_count()
        max_workers = num_cores - 1
        mp_pool = ProcessPoolExecutor(max_workers=max_workers)
        map_func = mp_pool.map

    pbar = None
    if track_progress:
        pbar = track_progress if isinstance(track_progress, tqdm) else tqdm(total=max_iters, desc=f'Generation')
        if not isinstance(track_progress, tqdm):
            pbar.update(starting_gen)

    num_iters = 0
    for gen_idx in range(starting_gen, max_iters):
        # agent_policies, map_func, env, experiment_dir, gen_idx
        agent_policies, best_team = generation_func(agent_policies=agent_policies, map_func=map_func)

        # save generation progress
        # save all policies of each agent and save fitnesses mapping policies to fitnesses
        # save before downselecting in case a policy in the best team gets pruned
        save_agent_policies(experiment_dir, gen_idx + 1, env, agent_policies, best_team)

        num_iters += 1
        if isinstance(pbar, tqdm):
            pbar.update(1)
        if completion_criteria():
            break
    if mp_pool:
        mp_pool.shutdown()
    if isinstance(pbar, tqdm) and not track_progress:
        pbar.close()

    best_policies = select_top_n(agent_policies, select_sizes={name: env.types_num[name] for name, pop in agent_policies.items()})
    return agent_policies, best_policies, num_iters
