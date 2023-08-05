"""
@title

@description

"""
import argparse
from pathlib import Path

from matplotlib import pyplot as plt

from island_influence import project_properties
from island_influence.agent import Agent, Obstacle, Poi
from island_influence.harvest_env import HarvestEnv
from island_influence.learn.neural_network import NeuralNetwork
from island_influence.utils import load_config


def generate_plot(config_path: Path):
    render_mode = 'rgb_array'
    experiment_config = load_config(str(config_path))
    config_name = config_path.stem

    harvesters = [
        Agent(idx, AgentType.Harvester, obs_rad, agent_weight, agent_value, max_vel, policy, sense_function='regions')
        for idx in range(num_harvesters)
    ]
    excavators = [
        Agent(idx, AgentType.Excavator, obs_rad, agent_weight, agent_value, max_vel, policy, sense_function='regions')
        for idx in range(num_excavators)
    ]

    obstacles = [
        Obstacle(idx, AgentType.Obstacle, obs_rad, obs_weight, obstacle_value)
        for idx in range(num_obstacles)
    ]
    pois = [
        Poi(idx, AgentType.StaticPoi, obs_rad, poi_weight, poi_value)
        for idx in range(num_pois)
    ]

    env = HarvestEnv(
        leaders=leaders, followers=followers, pois=pois, max_steps=100, render_mode=render_mode, delta_time=0
    )
    frame = env.render()

    config_name = config_name.split(".")[0]
    fig_name = f'{config_name}'

    fg_color = 'white'
    bg_color = 'black'
    fig, axes = plt.subplots(1, 1)

    img = axes.imshow(frame)
    # set visibility of x-axis as False
    xax = axes.get_xaxis()
    xax.set_visible(False)

    # set visibility of y-axis as False
    yax = axes.get_yaxis()
    yax.set_visible(False)

    axes.set_title(fig_name, color=fg_color)
    axes.patch.set_facecolor(bg_color)

    # set tick and ticklabel color
    img.axes.tick_params(color=fg_color, labelcolor=fg_color)

    # set imshow outline
    for spine in img.axes.spines.values():
        spine.set_edgecolor(fg_color)
    fig.patch.set_facecolor(bg_color)
    plt.tight_layout()

    save_name = Path(project_properties.doc_dir, 'configs', f'{fig_name}.png')
    if not save_name.parent.exists():
        save_name.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(str(save_name))
    plt.close()
    return


def main(main_args):
    config_paths = Path(project_properties.config_dir).glob('*.yaml')
    for each_path in config_paths:
        if each_path.stem == 'meta_params':
            continue

        # config_fn = Path(project_properties.test_dir, 'configs', config_name)
        generate_plot(each_path)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))