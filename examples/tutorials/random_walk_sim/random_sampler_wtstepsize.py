import sys
import random
import json
import re
from PIL import Image
import numpy as np
import os
import git
import habitat


from new_actions_modif import StrafeActionConfig


def display_sample(rgb_obs): 
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    return rgb_img


def generate_instruction_images(n_samples, source_img_path, target_img_path, instruction_path):
    data = []
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            "habitat.environment.iterator_options.shuffle=False",
        ]
    )

    with habitat.config.read_write(config):
        # Add a simple action config to the config.habitat.task.actions dictionary
        # Here we do it via code, but you can easily add them to a yaml config as well
        config.habitat.task.actions["STRAFE_RAND"] = StrafeActionConfig(type="StrafeRand")
    
    env = habitat.Env(config=config)
    obj1 = env.reset()
    
    for i in range(n_samples):
        obj2 = env.step("STRAFE_RAND")
        img1 = display_sample(obj1["rgb"])
        img2 = display_sample(obj2["rgb"])
        img1.save(f"{source_img_path}/img_{i}.jpg")
        img2.save(f"{target_img_path}/img_{i}.jpg")
        instr = env.task.actions["STRAFE_RAND"].instruction
        # Add row in instructions.json with base_img_path, transformed_img_path, and instruction
        data.append({"source_img_path": f"{source_img_path}/img_{i}.jpg", "target_img_path": f"{target_img_path}/img_{i}.jpg", "instruction": instr})
        obj1 = obj2
    env.close()
    with open(instruction_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    repo = git.Repo(".", search_parent_directories=True)
    dir_path = repo.working_tree_dir
    os.chdir(dir_path)
    # Get runtime arguments
    source_img_path = sys.argv[1]
    if not os.path.exists(source_img_path):
        os.makedirs(source_img_path)
    target_img_path = sys.argv[2]
    if not os.path.exists(target_img_path):
        os.makedirs(target_img_path)
    instruction_path = sys.argv[3]
    if not os.path.exists(instruction_path):
        os.makedirs(instruction_path)
    instruction_path = f"{instruction_path}/instructions.json"

    n_samples = int(sys.argv[4])
    generate_instruction_images(n_samples, source_img_path, target_img_path, instruction_path)