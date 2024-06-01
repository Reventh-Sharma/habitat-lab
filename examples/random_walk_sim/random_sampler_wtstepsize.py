import sys
import random
import json
import re
from PIL import Image
import numpy as np
import os
import git
import habitat
import argparse
import git

from os.path import join

from parametrized_action import add_param_actions

from loguru import logger

repo = git.Repo(".", search_parent_directories=True)
dir_path = repo.working_tree_dir
logger.info(f"Directory path: {dir_path}")

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

    config = add_param_actions(config)

    try:
        env.close()  # type: ignore[has-type]
    except NameError:
        pass

    env = habitat.Env(config=config)
    return env

def random_action_sampler(env):
    actionset = ["translate", "turn", "straff"]
    action = random.choice(actionset)

    if action == "translate":
        step_length = 0
        while step_length == 0:
            step_length = random.uniform(-5.0, 5.0)
        action_wt_args = ["translate", (step_length,)]
        if step_length < 0:
            instruction = f"Move backward {-step_length} meters"
        else:
            instruction = f"Move forward {step_length} meters" 
    
    elif action == "turn":
        angle = 0
        while angle == 0:
            angle = random.uniform(-180, 180)
        action_wt_args = ["turn", (angle,)]
        if angle < 0:
            instruction = f"Turn left {-angle} degrees"
        else:
            instruction = f"Turn right {angle} degrees"
    
    else:
        step_length = 0
        while step_length == 0:
            step_length = random.uniform(-5.0, 5.0)

        angle = 0
        while angle == 0:
            angle = random.uniform(-180, 180)
        action_wt_args = ["straff", (step_length, angle)]
        instruction = f"Move {step_length} meters at {angle} degrees"

    return (action_wt_args, instruction)


def generate_instruction_images(env_config_path, n_samples, source_img_path, target_img_path, instruction_path):
    env = generate_habitat_env(env_config_path)
    obs = env.reset()
    data = []
    for i in range(n_samples):
        img1 = display_sample(obs["rgb"])
        action_wt_args, instruction = random_action_sampler(env)
        if action_wt_args=="translate":
            obs = env.step({"action":("param_translate"), "action_args":{'move_amount': action_wt_args[1][0]}})
        elif action_wt_args=="turn":
            obs = env.step({"action":("param_rotate"), "action_args":{'turn_angle': action_wt_args[1][0]}})
        elif action_wt_args=="straff":
            obs = env.step({"action":("param_straffing"), "action_args":{'straff_angle': action_wt_args[1][1], 'move_amount': action_wt_args[1][0]}})

        img2 = display_sample(obs["rgb"])

        img1.save(f"{source_img_path}/img_{i}.jpg")
        img2.save(f"{target_img_path}/img_{i}.jpg")
        # Add row in instructions.json with base_img_path, transformed_img_path, and instruction
        data.append({"source_img_path": f"{source_img_path}/img_{i}.jpg", "target_img_path": f"{target_img_path}/img_{i}.jpg", "instruction": instruction})
    
    env.close()
    with open(join(instruction_path), "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Generate instruction images")

    # Add arguments
    parser.add_argument("--config_path", type=str, help="Config to initialize habitat environment", default='')
    parser.add_argument("--source_img_path", type=str, help="Path to save the source images")
    parser.add_argument("--target_img_path", type=str, help="Path to save the target images")
    parser.add_argument("--instruction_path", type=str, help="Path to save the instructions")
    parser.add_argument("--n_samples", type=int, help="Number of samples to generate")

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
    instruction_path = join(instruction_path, "instructions.json")

    n_samples = int(args.n_samples)
    # Call the function to generate instruction images
    generate_instruction_images(args.config_path, n_samples, source_img_path, target_img_path, instruction_path)
