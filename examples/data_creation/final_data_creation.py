# @title Colab Setup and Imports { display-mode: "form" }
# @markdown (double click to see the code)

import os
import random
import io
from os.path import join
import git
import numpy as np
from gym import spaces
import json
import shutil
from matplotlib import pyplot as plt

import argparse
from loguru import logger

logger.info(f"Inside final_data_creation.py: Current working directory: {os.getcwd()}")

from PIL import Image
from loguru import logger
import habitat
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationTask
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config as get_baselines_config

from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import HeadingSensorConfig, TopRGBSensorConfig, TopDownMapMeasurementConfig, FogOfWarConfig, CollisionsMeasurementConfig
from habitat.config import read_write
from habitat.utils.visualizations.utils import (
    images_to_video
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from add_objects import recompute_navmesh, add_object, remove_object
from initialize_env import create_env
from topdownmap import draw_top_down_map, get_3drendered_top_down_map, add_path_to_map

import argparse

logger.info(f"Inside main.py: Current working directory: {os.getcwd()}")

actionmap = {v: k for k, v in HabitatSimActions._known_actions.items()}
def generate_path_traversal_data(env):
    images = []
    traversal = []
    actions = []
    agent_coordinates = []
    nopath = False
    isfirst = True
    startpos, startrot = env.habitat_env.current_episode.start_position, env.habitat_env.current_episode.start_rotation
    # startpos, startrot = env.habitat_env.sim.get_agent_state().position, env.habitat_env.sim.get_agent_state().rotation
    observations, reward, done, info = env.step({'action': 'param_change_loc', 'action_args': {'pos_crd': startpos, 'angle_crd': startrot}})
    # env.habitat_env.sim.set_agent_state(startpos, startrot)
    goal = env.habitat_env.current_episode.goals[0]
    logger.info(f"Goal: {goal}")
    goal_radius = goal.radius
    if goal_radius is None:
        goal_radius = 0.25
    follower = ShortestPathFollower(
        env.habitat_env.sim, goal_radius, False
    )
    logger.info(f"Goal: {goal}")
    while not env.habitat_env.episode_over:
        best_action = follower.get_next_action(
            goal.view_points[0].agent_state.position

        )
        if best_action is None:
            nopath = True
            break
         
        # if isfirst:
        #     im = env.habitat_env.sim.get_sensor_observations()["rgb"][:, :, :3]
        #     tdv = draw_top_down_map(env.habitat_env.get_metrics(), size=im.shape[0])
        #     isfirst = False
        # else:
        im = observations["rgb"]   
        # tdv = draw_top_down_map(info, size=im.shape[0])
        tdv = add_path_to_map(agent_coordinates, env, info, goal)
        images.append(im)
        traversal.append(tdv)
        actions.append(actionmap[best_action])
        agent_coordinates.append((env.habitat_env.sim.get_agent_state().position.tolist(), env.habitat_env.sim.get_agent_state().rotation.components.tolist()))
        observations, reward, done, info = env.step(best_action)
        # path.append((env.habitat_env.sim.get_agent_state().position, env.habitat_env.sim.get_agent_state().rotation))
        # print(info)
    im = observations["rgb"]   
    tdv = add_path_to_map(agent_coordinates, env, info, goal, '')
    images.append(im)
    traversal.append(tdv)
    agent_coordinates.append((env.habitat_env.sim.get_agent_state().position.tolist(), env.habitat_env.sim.get_agent_state().rotation.components.tolist()))
    return nopath, goal.object_category, images, traversal, actions, agent_coordinates

def save_episode_data(savelocation, images, traversal):
    savepaths_rgb = []
    savepaths_topdown = []
    for i, (im, tdv) in enumerate(zip(images, traversal)):
        Image.fromarray(im.astype(np.uint8)).save(f"{savelocation}/image_{i}.png")
        Image.fromarray(tdv.astype(np.uint8)).save(f"{savelocation}/top_down_map_{i}.png")
        savepaths_rgb.append(f"{savelocation}/image_{i}.png")
        savepaths_topdown.append(f"{savelocation}/top_down_map_{i}.png")
    return savepaths_rgb, savepaths_topdown

def create_data(savelocation, topdown3drender, targetobj, images, traversal, actions, agent_coordinates):
    if os.path.exists(savelocation):
        shutil.rmtree(savelocation)
    os.makedirs(savelocation)
    topdown3drender.save(f"{savelocation}/top_down_map_3drender.png")
    savepaths_rgb, savepaths_topdown = save_episode_data(savelocation, images, traversal)
    images_to_video(images, f"{savelocation}/trajectoryvideo", "rgb_traj")
    images_to_video(traversal, f"{savelocation}/trajectoryvideo", "topdown_traj")
    agent_positions = [x[0] for x in agent_coordinates]
    agent_orientations = [x[1] for x in agent_coordinates]
    data = {"target_object": targetobj, "video_rgb_traj":f"{savelocation}/trajectoryvideo/rgb_traj.mp4", "video_topdown_traj":f"{savelocation}/trajectoryvideo/topdown_traj.mp4", "env_topdown_3d_render": f"{savelocation}/top_down_map_3drender.png", "source_rgb": savepaths_rgb[0], "source_topdown": savepaths_topdown[0], "actions": actions, "targets_rgb": savepaths_rgb[1:], "targets_topdown": savepaths_topdown[1:], "agent_positions": agent_positions, "agent_orientations": agent_orientations}
    return data

def create_final_data(N, M, step_size, turn_angle, svpath):
    env = create_env(step_size, turn_angle)
    env.habitat_env._episode_from_iter_on_reset = False
    obs = env.reset()
    data = {}
    i = 0
    while i<N: # Number of different scenes
        env.habitat_env.episode_iterator._forced_scene_switch()
        j = 0
        for ep in env.habitat_env._episode_iterator:
            if j>=M:# Number of different episodes per scene
                break
            env.habitat_env.current_episode = ep
            env.reset()
            recompute_navmesh(env.habitat_env.sim)
            topdown3drender = get_3drendered_top_down_map(env)
            nopath, goal_cat, images, traversal, actions, agent_coordinates = generate_path_traversal_data(env)
            if not nopath:
                data[f'scene:{ep.scene_id}, episode:{ep.episode_id}, goalid: {ep.goals[0].object_id}'] = create_data(f"{svpath}/episode_{i}", topdown3drender, goal_cat, images, traversal, actions, agent_coordinates)
            j += 1
        i += 1
    with open(f"{svpath}/data.json", "w") as f:
        json.dump(data, f)

