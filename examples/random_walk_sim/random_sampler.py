# Example script to generate samples following random instructions using actions with predefined step sizes

import sys
import random
import json
import re
from PIL import Image

import os

import habitat
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationTask
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config as get_baselines_config
from habitat_sim.utils.common import d3_40_colors_rgb
import argparse
import git

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir

def display_sample(rgb_obs): 
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    return rgb_img


def generate_habitat_env(env_config_path):
    # Config initialized at pointnav_habitat_test.yaml
    if env_config_path == "":
        config = habitat.get_config(
            config_path=f"{dir_path}/benchmark/nav/pointnav/pointnav_habitat_test.yaml",
            overrides=[
                "habitat.environment.max_episode_steps=1000",
                "habitat.environment.iterator_options.shuffle=False",
            ],
        )
    else:
        config = habitat.get_config(
            config_path=env_config_path
        )

    try:
        env.close()  # type: ignore[has-type]
    except NameError:
        pass

    env = habitat.Env(config=config)
    return env

def random_action_sampler(env):
    directionset = ["no_change", "turn_left", "turn_right"]
    rotation = random.choice(directionset)
    if rotation=="no_change":
        translation = "move_forward"
    else:
        translation = random.choice(["stay", "move_forward"])
    
    rinst = re.sub("_", " ", rotation)
    tinst = re.sub("_", " ", translation)
    instruction = f"{rinst} in direction and {tinst}"
    return (rotation, translation, instruction)

def generate_instruction_images(env_config_path, n_samples, source_img_path, target_img_path, instruction_path):
    env = generate_habitat_env(env_config_path)
    obs = env.reset()
    data = []
    for i in range(n_samples):
        img1 = display_sample(obs["rgb"])
        rotation, translation, instruction = random_action_sampler(env)
        if rotation!="no_change":
            obs = env.step({"action": rotation})
        if translation=="move_forward":
            obs = env.step({"action": translation})
        img2 = display_sample(obs["rgb"])

        img1.save(f"{source_img_path}/img_{i}.jpg")
        img2.save(f"{target_img_path}/img_{i}.jpg")
        # Add row in instructions.json with base_img_path, transformed_img_path, and instruction
        data.append({"source_img_path": f"{source_img_path}/img_{i}.jpg", "target_img_path": f"{target_img_path}/img_{i}.jpg", "instruction": instruction})
    
    env.close()
    with open(instruction_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Generate instruction images")

    # Add arguments
    parser.add_argument("config_path", type=str, help="Config to initialize habitat environment", default='')
    parser.add_argument("source_img_path", type=str, help="Path to save the source images")
    parser.add_argument("target_img_path", type=str, help="Path to save the target images")
    parser.add_argument("instruction_path", type=str, help="Path to save the instructions")
    parser.add_argument("n_samples", type=int, help="Number of samples to generate")

    # Parse the arguments
    args = parser.parse_args()

    # Get runtime arguments
    source_img_path = args.source_img_path
    if not os.path.exists(source_img_path):
        os.makedirs(source_img_path)
    target_img_path = args.target_img_path
    if not os.path.exists(target_img_path):
        os.makedirs(target_img_path)
    instruction_path = args.instruction_path
    if not os.path.exists(instruction_path):
        os.makedirs(instruction_path)
    instruction_path = f"{instruction_path}/instructions.json"

    n_samples = int(sys.argv[4])
    # Call the function to generate instruction images
    generate_instruction_images(args.config_path, args.n_samples, args.source_img_path, args.target_img_path, args.instruction_path)
