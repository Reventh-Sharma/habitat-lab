import os
import random

import git
import numpy as np
from gym import spaces

from matplotlib import pyplot as plt

from PIL import Image

import habitat
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationTask
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config as get_baselines_config
import habitat_sim
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import images_to_video

import shutil

import sys
sys.path.append(os.path.join(os.getcwd()))

from add_objects import add_object, remove_object
from initialize_env import create_env
from topdownmap import draw_top_down_map, add_path_to_map

def display_sample(
    obs
):  # noqa: B006
    rgb_img = Image.fromarray(obs[0]['rgb'], mode="RGB")
    return rgb_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_save_dir", type=str, required=True)
    parser.add_argument("--video_save_dir", type=str, required=True)
    args = parser.parse_args()

    PLOTSAVE_DIR = args.plot_save_dir
    IMAGE_DIR = args.video_save_dir

    env = create_env()
    goal_radius = env.episodes[0].goals[0].radius
    if goal_radius is None:
        goal_radius = 0.25 #Default step size in config
    follower = ShortestPathFollower(
        env.habitat_env.sim, goal_radius, False
    )
    print(goal_radius)
    follower = ShortestPathFollower(env.habitat_env.sim, goal_radius, False)
    env.reset()
    default_spawn = env.habitat_env.sim.get_agent_state().position, env.habitat_env.sim.get_agent_state().rotation

    # Traverse path with and without object on scene
    for episode in range(3):
        # Without object
        env.reset()
        env.step({"action":"param_change_loc", "action_args": {"pos_crd": default_spawn[0], "angle_crd": default_spawn[1]}})
        dirname = os.path.join(
            IMAGE_DIR, "shortest_path_example", "%02d" % episode+"_no_object"
        )
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)
        print("Agent stepping around inside environment.")
        images = []
        path = []
        draw_top_down_map(env, args.plot_save_dir+"%02d" % episode+".png")
        while not env.habitat_env.episode_over:
            best_action = follower.get_next_action(
                env.habitat_env.current_episode.goals[0].view_points[0].agent_state.position

            )
            if best_action is None:
                break
            observations, reward, done, info = env.step(best_action)
            im = observations["rgb"]
            path.append((env.habitat_env.sim.get_agent_state().position, env.habitat_env.sim.get_agent_state().rotation))
            print(info)
            # top_down_map = draw_top_down_map(info, im.shape[0])
            # output_im = np.concatenate((im, top_down_map), axis=1)
            output_im = im
            images.append(output_im)
        images_to_video(images, dirname, "trajectory")
        print("Episode finished")
        add_path_to_map(path, env, args.plot_save_dir+"%02d" % episode+"withoutobj.png")

        # With object
        env.reset()
        env.step({"action":"param_change_loc", "action_args": {"pos_crd": default_spawn[0], "angle_crd": default_spawn[1]}})
        dirname = os.path.join(
            IMAGE_DIR, "shortest_path_example", "%02d" % episode+"_wt_object"
        )
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)
        print("Agent stepping around inside environment.")
        obj_loc = path[random.randint(0, len(path))]
        objid = add_object(env, obj_loc, scale=1, template_id=2) # Add cone
        images = []
        path = []
        while not env.habitat_env.episode_over:
            best_action = follower.get_next_action(
                env.habitat_env.current_episode.goals[0].view_points[0].agent_state.position

            )
            if best_action is None:
                break
            observations, reward, done, info = env.step(best_action)
            im = observations["rgb"]
            path.append((env.habitat_env.sim.get_agent_state().position, env.habitat_env.sim.get_agent_state().rotation))
            print(info)
            # top_down_map = draw_top_down_map(info, im.shape[0])
            # output_im = np.concatenate((im, top_down_map), axis=1)
            output_im = im
            images.append(output_im)
        images_to_video(images, dirname, "trajectory")
        print("Episode finished")
        remove_object(env, objid)
        add_path_to_map(path, env, args.plot_save_dir+"%02d" % episode+"withoutobj.png")
