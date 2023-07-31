"""
@title

@description

"""
import argparse
import math
import time

import matplotlib.pyplot as plt
import numpy as np

from island_influence.agent import Poi, Agent, Obstacle, AgentType
from island_influence.discrete_harvest_env import DiscreteHarvestEnv
from island_influence.learn.neural_network import NeuralNetwork


def display_final_agents(env):
    print(f'Remaining agents: {len(env.agents)}')
    for agent_name in env.agents:
        agent = env.agent_mapping[agent_name]
        print(f'{agent_name=}: {agent.location=}')
    print(f'Completed agents: {len(env.completed_agents)}')
    for agent_name, agent_reward in env.completed_agents.items():
        agent = env.agent_mapping[agent_name]
        print(f'{agent_name=} | {agent_reward=} | {agent.location=}')
    return


def test_observations(env):
    print(f'=' * 80)
    env.reset()
    print(f'Running observation tests')
    obs_space = env.observation_space(env.agents[0])
    print(f'{obs_space=}')

    for agent_name in env.agents:
        each_obs = env.observation_space(agent_name)
        print(f'{agent_name}: {each_obs}')
    all_obs = env.get_observations()
    for agent_name, each_obs in all_obs.items():
        print(f'{agent_name}: {each_obs}')
    print(f'=' * 80)
    return


def test_actions(env):
    print(f'=' * 80)
    env.reset()
    print(f'Running action tests')
    act_space = env.action_space(env.agents[0])
    print(f'{act_space=}')

    for each_agent in env.agents:
        each_act = env.action_space(each_agent)
        print(f'{each_agent}: {each_act}')

    all_obs = env.get_observations()
    for agent_name, obs in all_obs.items():
        agent = env.agent_mapping[agent_name]
        action = agent.get_action(obs)
        print(f'{agent_name=}: {obs=} | {action=}')
    print(f'=' * 80)
    return


def test_render(env):
    print(f'=' * 80)
    env.reset()
    print(f'Running render tests')
    mode = env.render_mode
    print(f'{mode=}')

    frame = env.render(mode='rgb_array')
    env.close()

    plt.imshow(frame)
    plt.show()
    plt.close()

    print(f'=' * 80)
    return


def test_step(env, render):
    render_delay = 0.1

    # action is (dx, dy)
    forward_action = np.array((0, 1.5))
    backwards_action = np.array((0, -1.5))
    left_action = np.array((-1.5, 0))
    right_action = np.array((1.5, 0))

    tests = [
        {agent: forward_action for agent in env.agents},
        {agent: backwards_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: left_action for agent in env.agents},

        {agent: forward_action for agent in env.agents},
        {agent: forward_action for agent in env.agents},
        {agent: forward_action for agent in env.agents},
        {agent: forward_action for agent in env.agents},

        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
    ]

    print(f'=' * 80)
    env.reset()
    print(f'Running step tests')
    obs_space = env.observation_space(env.agents[0])
    act_space = env.action_space(env.agents[0])
    print(f'{obs_space=}')
    print(f'{act_space=}')

    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env.step(each_action)
        if render:
            frame = env.render(render)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)

    # reset and do it again
    env.reset()
    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env.step(each_action)
        if render:
            frame = env.render(render)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)
    display_final_agents(env)
    print(f'=' * 80)
    influence_counts = [
        follower.influence_counts()
        for follower_name, follower in env.__followers.items()
    ]
    print(f'{influence_counts=}')
    return


def test_random(env, render):
    render_delay = 0.1
    counter = 0
    done = False
    print(f'=' * 80)
    env.reset()
    print(f'Running random step tests')

    # noinspection DuplicatedCode
    init_observations = env.reset()
    print(f'{init_observations=}')
    while not done:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncs, infos = env.step(actions)
        done = all(terminations.values())
        counter += 1
        if counter % 100 == 0:
            print(f'{counter=}')
        if render:
            frame = env.render(render)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)

    # reset and do it again
    # noinspection DuplicatedCode
    init_observations = env.reset()
    print(f'{init_observations=}')
    while not done:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncs, infos = env.step(actions)
        done = all(terminations.values())
        counter += 1
        if counter % 100 == 0:
            print(f'{counter=}')
        if render:
            frame = env.render(render)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)

    display_final_agents(env)
    print(f'=' * 80)
    return


def test_rollout(env, render):
    render_delay = 0.1
    env.reset()
    agent_dones = env.done()
    done = all(agent_dones.values())

    while not done:
        observations = env.get_observations()
        next_actions = env.get_actions_from_observations(observations=observations)
        observations, rewards, agent_dones, truncs, infos = env.step(next_actions)
        done = all(agent_dones.values())
        if render:
            frame = env.render(render)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)
    return


def test_persistence(env: DiscreteHarvestEnv):
    save_path = env.save_environment()
    test_env = DiscreteHarvestEnv.load_environment(save_path)
    # todo  inspect object
    #       agents
    #           histories
    #           leaders
    #               policies
    #           followers
    #           pois
    #       reward history
    #       state history
    return


def main(main_args):
    render_mode = 'rgb_array'
    delta_time = 1

    obs_rad = 100
    max_vel = 1
    sen_res = 8

    agent_weight = 1
    obs_weight = 1
    poi_weight = 1

    agent_value = 1
    obstacle_value = 1
    poi_value = 0

    agent_config = [
        (AgentType.Harvester, np.asarray((0, 1))),
        (AgentType.Harvester, np.asarray((0, 2))),
        (AgentType.Harvester, np.asarray((0, 3))),
        (AgentType.Harvester, np.asarray((0, 4))),
    ]

    obstacle_config = [
        (AgentType.Obstacle, np.asarray((1, 1))),
        (AgentType.Obstacle, np.asarray((1, 2))),
        (AgentType.Obstacle, np.asarray((1, 3))),
        (AgentType.Obstacle, np.asarray((1, 4))),
    ]

    poi_config = [
        (AgentType.StaticPoi, np.asarray((2, 1))),
        (AgentType.StaticPoi, np.asarray((2, 2))),
        (AgentType.StaticPoi, np.asarray((2, 3))),
        (AgentType.StaticPoi, np.asarray((2, 4))),
    ]

    n_inputs = sen_res * Agent.NUM_BINS
    n_outputs = 2
    n_hidden = math.ceil((n_inputs + n_outputs) / 2)
    policy = NeuralNetwork(n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden)

    agents = [
        Agent(idx, agent_info[0], agent_info[1], obs_rad, agent_weight, agent_value, max_vel, policy, sense_function='regions')
        for idx, agent_info in enumerate(agent_config)
    ]
    obstacles = [
        Obstacle(idx, agent_info[0], agent_info[1], obs_rad, obs_weight, obstacle_value)
        for idx, agent_info in enumerate(obstacle_config)
    ]
    pois = [
        Poi(idx, agent_info[0], agent_info[1], obs_rad, poi_weight, poi_value)
        for idx, agent_info in enumerate(poi_config)
    ]

    env = DiscreteHarvestEnv(agents=agents, obstacles=obstacles, pois=pois, max_steps=100, delta_time=delta_time, render_mode=render_mode)

    # test_observations(env)
    # test_actions(env)
    # test_render(env)uhu

    # test_step(env, render=None)
    # test_step(env, render='rgb_array')

    # test_random(env, render=None)
    # test_random(env, render='rgb_array')

    # test_rollout(env, render=None)
    # test_rollout(env, render='rgb_array')

    test_persistence(env)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
