"""
@title

@description

"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from island_influence import project_properties
from island_influence.envs.harvest_env import HarvestEnv
from island_influence.setup_env import det_ring_env


def display_env(env, render_mode, render_delay=0.1):
    env.render()
    time.sleep(render_delay)
    return


def test_observations(env: HarvestEnv, display=False):
    env.reset()
    # first_agent = env.agents[0]
    # obs_space = env.observation_space(first_agent)
    if display:
        print(f'=' * 80)
        print(f'Running observation tests')
        # print(f'{obs_space=}')

    for agent in env.agents:
        obs_space = env.observation_space(agent)
        if display:
            print(f'{agent.name}: {obs_space}')

    observations = env.get_observations()
    for name, obs in observations.items():
        if display:
            print(f'{name}: {obs}')
    return


def test_actions(env: HarvestEnv, display=False):
    env.reset()
    # first_agent = env.agents[0]
    # act_space = env.action_space(first_agent)
    if display:
        print(f'=' * 80)
        print(f'Running action tests')
        # print(f'{act_space=}')

    for agent in env.agents:
        act_space = env.action_space(agent)
        if display:
            print(f'{agent.name}: {act_space}')

    observations = env.get_observations()
    actions = env.get_actions()
    for name, act in actions.items():
        if display:
            print(f'{name}: {act}')
    return


def test_rewards(env: HarvestEnv, render_mode):
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
    env.reward_type = 'global'
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

    # reset and do it again with difference rewards
    env.reset()
    env.reward_type = 'difference'
    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env.step(each_action)
        if render_mode:
            display_env(env, render_mode, render_delay)
    # display_final_agents(env)
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
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},
        {agent: right_action for agent in env.agents},

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
    agent_rewards = env.cumulative_agent_rewards()
    policy_rewards = env.cumulative_policy_rewards()

    # reset and do it again
    env.reset()
    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env.step(each_action)
        if render_mode:
            display_env(env, render_mode, render_delay)
    agent_rewards = env.cumulative_agent_rewards()
    policy_rewards = env.cumulative_policy_rewards()
    print(f'=' * 80)
    return


def test_random(env: HarvestEnv, render_mode, display=False):
    render_delay = 1
    counter = 0
    env.reset()
    env.render_mode = render_mode

    # noinspection DuplicatedCode
    init_observations = env.reset()
    if display:
        print(f'=' * 80)
        print(f'Running random step tests')
        print(f'{init_observations=}')

    done = False
    while not done:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncs, infos = env.step(actions)
        done = all(terminations.values())
        counter += 1
        if counter % 100 == 0 and display:
            print(f'{counter=}')
        if render_mode:
            display_env(env, render_mode, render_delay)

    # reset and do it again
    # noinspection DuplicatedCode
    init_observations = env.reset()
    if display:
        print(f'{init_observations=}')
    done = False
    while not done:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncs, infos = env.step(actions)
        done = all(terminations.values())
        counter += 1
        if counter % 100 == 0 and display:
            print(f'{counter=}')
        if render_mode:
            display_env(env, render_mode, render_delay)
    return


def test_rollout(env: HarvestEnv, render_mode):
    render_delay = 1
    env.reset()
    env.render_mode = render_mode
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


def test_collisions(render_mode, display=False):
    if display:
        print(f'Running collision tests')
    env_obstacles_func = det_ring_env(scale_factor=0.5, num_excavators=1)
    env_pois_func = det_ring_env(scale_factor=0.5, num_excavators=1, num_obstacles=1)

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
        {agent: right_action for agent in env_obstacles.agents},
        {agent: right_action for agent in env_obstacles.agents},
        {agent: right_action for agent in env_obstacles.agents},
        {agent: right_action for agent in env_obstacles.agents},

        {agent: right_action for agent in env_obstacles.agents},
        {agent: right_action for agent in env_obstacles.agents},
        {agent: right_action for agent in env_obstacles.agents},
        {agent: right_action for agent in env_obstacles.agents},

        {agent: left_action for agent in env_obstacles.agents},
        {agent: left_action for agent in env_obstacles.agents},
        {agent: left_action for agent in env_obstacles.agents},
        {agent: left_action for agent in env_obstacles.agents},

        {agent: right_action for agent in env_obstacles.agents},
        {agent: right_action for agent in env_obstacles.agents},
        {agent: right_action for agent in env_obstacles.agents},
        {agent: right_action for agent in env_obstacles.agents},
    ]

    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env_obstacles.step(each_action)
        if render_mode:
            display_env(env_obstacles, render_mode, render_delay)
    cum_rewards = env_obstacles.cumulative_agent_rewards()
    if display:
        print(f'Obstacle environment: {env_obstacles.collision_penalty_scalar}: {cum_rewards}')
    env_obstacles.reset()
    env_obstacles.collision_penalty_scalar = 1
    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env_obstacles.step(each_action)
        if render_mode:
            display_env(env_obstacles, render_mode, render_delay)
    cum_rewards = env_obstacles.cumulative_agent_rewards()
    if display:
        print(f'Obstacle environment: {env_obstacles.collision_penalty_scalar}: {cum_rewards}')
        print(f'=' * 80)

    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env_pois.step(each_action)
        if render_mode:
            display_env(env_pois, render_mode, render_delay)
    cum_rewards = env_pois.cumulative_agent_rewards()
    if display:
        print(f'Poi environment: {env_pois.collision_penalty_scalar}: {cum_rewards}')
    env_pois.reset()
    env_pois.collision_penalty_scalar = 1
    for each_action in tests:
        observations, rewards, terminations, truncs, infos = env_pois.step(each_action)
        if render_mode:
            display_env(env_pois, render_mode, render_delay)
    return


def test_save_env(env: HarvestEnv, display=False):
    if display:
        print(f'Running persistence tests')
    save_path = env.save_environment()
    test_env = HarvestEnv.load_environment(save_path)

    assert len(env.state()) == len(test_env.state())
    assert len(env.state_history) == len(test_env.state_history)
    assert len(env.action_history) == len(test_env.action_history)
    assert len(env.reward_history) == len(test_env.reward_history)

    for env_entity, test_entity in zip(env.state(), test_env.state()):
        assert env_entity == test_entity
    return


def test_save_transitions(env: HarvestEnv, display=False):
    if display:
        print(f'Running save transitions tests')
    env.clear_saved_transitions()
    test_random(env, render_mode=None, display=display)
    initial_trans, initial_path = env.save_transitions()
    env.reset()

    test_random(env, render_mode=None, display=display)
    second_trans, second_path = env.save_transitions()
    env.reset()

    load_trans = HarvestEnv.load_transitions(initial_path)

    initial_trans.extend(second_trans)
    assert initial_path == second_path
    assert len(load_trans) == len(initial_trans)
    return


def test_reset(env: HarvestEnv, render_mode, display=False):
    if display:
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
    env_params = {
        'scale_factor': 1, 'num_harvesters': 15, 'num_excavators': 1, 'num_obstacles': 15, 'num_pois': 8, 'collision_penalty_scalar': 0,
        'max_steps': 25, 'save_dir': Path(project_properties.env_dir, 'harvest_env_test')
    }
    env_func = det_ring_env(**env_params)
    env = env_func()
    env.reset()
    env.normalize_rewards = True

    test_observations(env)
    test_actions(env)
    # test_save_env(env)
    # test_save_transitions(env)

    # test_collisions(render_mode=None)
    # test_collisions(render_mode='human')

    # test_rewards(env, render_mode=None)
    # test_rewards(env, render_mode='human')

    # test_reset(env, render_mode=None)
    test_step(env, render_mode=None)
    # test_random(env, render_mode=None)
    # test_rollout(env, render_mode=None)

    # test_render(env, render_mode='human')
    # test_reset(env, render_mode='human')
    # test_step(env, render_mode='human')
    # test_random(env, render_mode='human')
    test_rollout(env, render_mode='human')

    env.close()
    return


if __name__ == '__main__':
    # https://docs.python.org/3/library/profile.html
    # python test\test_environment.py
    # python -m cProfile -s cumulative test\test_environment.py
    # python -m cProfile -s filename test\test_environment.py
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
