"""
@title

@description

"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from island_influence.harvest_env import HarvestEnv
from island_influence.setup_env import rand_ring_env, det_ring_env


def display_env(env, render_mode, render_delay=1.0):
    frame = env.render()
    if render_mode == 'rgb_array':
        plt.imshow(frame)
        plt.pause(render_delay)
    else:
        time.sleep(render_delay)
    plt.close()
    return


def test_observations(env: HarvestEnv):
    print(f'=' * 80)
    env.reset()
    print(f'Running observation tests')
    first_agent = env.agents[0]
    obs_space = env.observation_space(first_agent)
    print(f'{obs_space=}')

    for agent in env.agents:
        each_obs = env.observation_space(agent)
        print(f'{agent.name}: {each_obs}')
    all_obs = env.get_observations()
    for name, each_obs in all_obs.items():
        print(f'{name}: {each_obs}')
    print(f'=' * 80)
    return


def test_actions(env: HarvestEnv):
    print(f'=' * 80)
    env.reset()
    print(f'Running action tests')
    first_agent = env.agents[0]
    act_space = env.action_space(first_agent)
    print(f'{act_space=}')

    for agent in env.agents:
        each_act = env.action_space(agent)
        print(f'{agent.name}: {each_act}')

    all_obs = env.get_observations()
    for agent_name, obs in all_obs.items():
        agent = env.get_agent(agent_name)
        action = agent.get_action(obs)
        print(f'{agent_name=}: {obs=} | {action=}')
    print(f'=' * 80)
    return


def test_render(env: HarvestEnv, render_mode='rgb_array'):
    print(f'=' * 80)
    env.reset()
    env.render_mode = render_mode
    print(f'Running render tests')
    mode = env.render_mode
    print(f'{mode=}')

    frame = env.render()
    env.close()

    if render_mode == 'rgb_array':
        plt.imshow(frame)
        plt.show()
        plt.close()

    print(f'=' * 80)
    return


def test_step(env: HarvestEnv, render_mode):
    render_delay = 1

    # action is (dx, dy)
    step_size = 0.5
    forward_action = np.array((0, step_size))
    backwards_action = np.array((0, -step_size))
    left_action = np.array((-step_size, 0))
    right_action = np.array((step_size, 0))

    tests = [
        {agent.name: forward_action for agent in env.agents},
        {agent.name: backwards_action for agent in env.agents},
        {agent.name: right_action for agent in env.agents},
        {agent.name: left_action for agent in env.agents},

        {agent.name: forward_action for agent in env.agents},
        {agent.name: forward_action for agent in env.agents},
        {agent.name: forward_action for agent in env.agents},
        {agent.name: forward_action for agent in env.agents},

        {agent.name: right_action for agent in env.agents},
        {agent.name: right_action for agent in env.agents},
        {agent.name: right_action for agent in env.agents},
        {agent.name: right_action for agent in env.agents},
    ]

    print(f'=' * 80)
    env.reset()
    env.render_mode = render_mode
    print(f'Running step tests')
    first_agent = env.agents[0]
    obs_space = env.observation_space(first_agent)
    act_space = env.action_space(first_agent)
    print(f'{obs_space=}')
    print(f'{act_space=}')

    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env.step(each_action)
        if render_mode:
            display_env(env, render_mode, render_delay)

    # reset and do it again
    env.reset()
    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env.step(each_action)
        if render_mode:
            display_env(env, render_mode, render_delay)
    # display_final_agents(env)
    print(f'=' * 80)
    return


def test_random(env: HarvestEnv, render_mode):
    render_delay = 1
    counter = 0
    done = False
    print(f'=' * 80)
    env.reset()
    env.render_mode = render_mode
    print(f'Running random step tests')

    # noinspection DuplicatedCode
    init_observations = env.reset()
    print(f'{init_observations=}')
    while not done:
        actions = {agent.name: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncs, infos = env.step(actions)
        done = all(terminations.values())
        counter += 1
        if counter % 100 == 0:
            print(f'{counter=}')
        if render_mode:
            display_env(env, render_mode, render_delay)

    # reset and do it again
    # noinspection DuplicatedCode
    init_observations = env.reset()
    print(f'{init_observations=}')
    while not done:
        actions = {agent.name: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncs, infos = env.step(actions)
        done = all(terminations.values())
        counter += 1
        if counter % 100 == 0:
            print(f'{counter=}')
        if render_mode:
            display_env(env, render_mode, render_delay)

    # display_final_agents(env)
    print(f'=' * 80)
    return


def test_rollout(env: HarvestEnv, render_mode):
    render_delay = 1
    env.reset()
    agent_dones = env.done()
    done = all(agent_dones.values())

    while not done:
        observations = env.get_observations()
        next_actions = env.get_actions()
        observations, rewards, agent_dones, truncs, infos = env.step(next_actions)
        done = all(agent_dones.values())
        if render_mode:
            display_env(env, render_mode, render_delay)
    return


def test_collisions(render_mode):
    print(f'Running collision tests')
    env_obstacles_func = det_ring_env(scale_factor=0.5, num_excavators=0)
    env_pois_func = det_ring_env(scale_factor=0.5, num_excavators=0, num_obstacles=0)

    env_obstacles = env_obstacles_func()
    env_pois = env_pois_func()

    env_obstacles.reset()
    env_obstacles.render_mode = render_mode
    env_obstacles.normalize_rewards = True

    env_pois.reset()
    env_pois.render_mode = render_mode
    env_pois.normalize_rewards = True

    render_delay = 0.5

    # action is (dx, dy)
    step_size = 0.5
    forward_action = np.array((0, step_size))
    backwards_action = np.array((0, -step_size))
    left_action = np.array((-step_size, 0))
    right_action = np.array((step_size, 0))

    tests = [
        {agent.name: right_action for agent in env_obstacles.agents},
        {agent.name: right_action for agent in env_obstacles.agents},
        {agent.name: right_action for agent in env_obstacles.agents},
        {agent.name: right_action for agent in env_obstacles.agents},

        {agent.name: right_action for agent in env_obstacles.agents},
        {agent.name: right_action for agent in env_obstacles.agents},
        {agent.name: right_action for agent in env_obstacles.agents},
        {agent.name: right_action for agent in env_obstacles.agents},

        {agent.name: left_action for agent in env_obstacles.agents},
        {agent.name: left_action for agent in env_obstacles.agents},
        {agent.name: left_action for agent in env_obstacles.agents},
        {agent.name: left_action for agent in env_obstacles.agents},

        {agent.name: right_action for agent in env_obstacles.agents},
        {agent.name: right_action for agent in env_obstacles.agents},
        {agent.name: right_action for agent in env_obstacles.agents},
        {agent.name: right_action for agent in env_obstacles.agents},
    ]

    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env_obstacles.step(each_action)
        if render_mode:
            display_env(env_obstacles, render_mode, render_delay)
    cum_rewards = env_obstacles.cumulative_rewards()
    print(f'Obstacle environment: {env_obstacles.collision_penalty_scalar}: {cum_rewards}')
    env_obstacles.reset()
    env_obstacles.collision_penalty_scalar = 1
    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env_obstacles.step(each_action)
        if render_mode:
            display_env(env_obstacles, render_mode, render_delay)
    cum_rewards = env_obstacles.cumulative_rewards()
    print(f'Obstacle environment: {env_obstacles.collision_penalty_scalar}: {cum_rewards}')
    print(f'=' * 80)

    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env_pois.step(each_action)
        if render_mode:
            display_env(env_pois, render_mode, render_delay)
    cum_rewards = env_pois.cumulative_rewards()
    print(f'Poi environment: {env_pois.collision_penalty_scalar}: {cum_rewards}')
    env_pois.reset()
    env_pois.collision_penalty_scalar = 1
    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env_pois.step(each_action)
        if render_mode:
            display_env(env_pois, render_mode, render_delay)
    print(f'=' * 80)
    return


def test_persistence(env: HarvestEnv):
    print(f'Running persistence tests')
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
    print(f'Running reset tests')
    render_delay = 1
    env.reset()
    env.render_mode = render_mode
    num_resets = 10
    state = env.state()
    for idx in range(num_resets):
        observations = env.reset()
        state = env.state()
        if render_mode:
            display_env(env, render_mode, render_delay)
    return


def main(main_args):
    env_params = {'scale_factor': 2, 'num_harvesters': 4, 'num_excavators': 4, 'num_obstacles': 16, 'num_pois': 8, 'collision_penalty_scalar': 0}
    env_func = rand_ring_env(**env_params)
    env = env_func()
    env.reset()
    env.normalize_rewards = True

    test_observations(env)
    test_actions(env)
    test_collisions(render_mode=None)
    # test_collisions(render_mode='rgb_array')
    # test_collisions(render_mode='human')

    # test_reset(env, render_mode=None)
    # test_step(env, render_mode=None)
    # test_random(env, render_mode=None)
    # test_rollout(env, render_mode=None)

    # test_render(env, render_mode='rgb_array')
    # test_reset(env, render_mode='rgb_array')
    # test_step(env, render_mode='rgb_array')
    # test_random(env, render_mode='rgb_array')
    # test_rollout(env, render_mode='rgb_array')

    test_render(env, render_mode='human')
    test_reset(env, render_mode='human')
    test_step(env, render_mode='human')
    test_random(env, render_mode='human')
    test_rollout(env, render_mode='human')

    test_persistence(env)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
