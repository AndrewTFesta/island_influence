"""
@title

@description

"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from island_influence.harvest_env import HarvestEnv
from island_influence.setup_test import linear_setup


# def display_final_agents(env: DiscreteHarvestEnv):
#     print(f'Remaining agents: {len(env.agents)}')
#     for agent_name in env.agents:
#         agent = env.agent_mapping[agent_name]
#         print(f'{agent_name=}: {agent.location=}')
#     print(f'Completed agents: {len(env.completed_agents)}')
#     for agent_name, agent_reward in env.completed_agents.items():
#         agent = env.agent_mapping[agent_name]
#         print(f'{agent_name=} | {agent_reward=} | {agent.location=}')
#     return


def test_observations(env: HarvestEnv):
    print(f'=' * 80)
    env.reset()
    print(f'Running observation tests')
    first_agent = list(env.agents.values())[0]
    obs_space = env.observation_space(first_agent)
    print(f'{obs_space=}')

    for name, agent in env.agents.items():
        each_obs = env.observation_space(agent)
        print(f'{name}: {each_obs}')
    all_obs = env.get_observations()
    for name, each_obs in all_obs.items():
        print(f'{name}: {each_obs}')
    print(f'=' * 80)
    return


def test_actions(env: HarvestEnv):
    print(f'=' * 80)
    env.reset()
    print(f'Running action tests')
    first_agent = list(env.agents.values())[0]
    act_space = env.action_space(first_agent)
    print(f'{act_space=}')

    for name, agent in env.agents.items():
        each_act = env.action_space(agent)
        print(f'{name}: {each_act}')

    all_obs = env.get_observations()
    for agent_name, obs in all_obs.items():
        agent = env.agents[agent_name]
        action = agent.get_action(obs)
        print(f'{agent_name=}: {obs=} | {action=}')
    print(f'=' * 80)
    return


def test_render(env: HarvestEnv):
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


def test_step(env: HarvestEnv, render_mode):
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
    first_agent = list(env.agents.values())[0]
    obs_space = env.observation_space(first_agent)
    act_space = env.action_space(first_agent)
    print(f'{obs_space=}')
    print(f'{act_space=}')

    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env.step(each_action)
        if render_mode:
            frame = env.render(render_mode)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)

    # reset and do it again
    env.reset()
    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env.step(each_action)
        if render_mode:
            frame = env.render(render_mode)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)
    # display_final_agents(env)
    print(f'=' * 80)
    return


def test_random(env: HarvestEnv, render_mode):
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
        actions = {name: env.action_space(agent).sample() for name, agent in env.agents.items()}
        observations, rewards, terminations, truncs, infos = env.step(actions)
        done = all(terminations.values())
        counter += 1
        if counter % 100 == 0:
            print(f'{counter=}')
        if render_mode:
            frame = env.render(render_mode)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)

    # reset and do it again
    # noinspection DuplicatedCode
    init_observations = env.reset()
    print(f'{init_observations=}')
    while not done:
        actions = {name: env.action_space(agent).sample() for name, agent in env.agents.items()}
        observations, rewards, terminations, truncs, infos = env.step(actions)
        done = all(terminations.values())
        counter += 1
        if counter % 100 == 0:
            print(f'{counter=}')
        if render_mode:
            frame = env.render(render_mode)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)

    # display_final_agents(env)
    print(f'=' * 80)
    return


def test_rollout(env: HarvestEnv, render_mode):
    render_delay = 0.1
    env.reset()
    agent_dones = env.done()
    done = all(agent_dones.values())

    while not done:
        observations = env.get_observations()
        next_actions = env.get_actions()
        observations, rewards, agent_dones, truncs, infos = env.step(next_actions)
        done = all(agent_dones.values())
        if render_mode:
            frame = env.render(render_mode)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)
    return


def test_persistence(env: HarvestEnv):
    save_path = env.save_environment()
    test_env = HarvestEnv.load_environment(save_path)
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


def test_reset(env: HarvestEnv, render_mode):
    render_delay = 0.1
    num_resets = 10
    state = env.state()
    for idx in range(num_resets):
        observations = env.reset()
        state = env.state()
        if render_mode:
            frame = env.render(render_mode)
            plt.imshow(frame)
            plt.show()
            time.sleep(render_delay)
    return


def main(main_args):
    env = linear_setup()

    test_observations(env)
    test_actions(env)

    test_reset(env, render_mode=None)
    # test_step(env, render_mode=None)
    # test_random(env, render_mode=None)
    # test_rollout(env, render_mode=None)

    # test_render(env)
    test_reset(env, render_mode='rgb_array')
    # test_step(env, render_mode='rgb_array')
    # test_random(env, render_mode='rgb_array')
    # test_rollout(env, render_mode='rgb_array')

    test_persistence(env)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
