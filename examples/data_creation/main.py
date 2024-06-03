import os
from loguru import logger
# import random

# import git
# import numpy as np
# from gym import spaces

# from matplotlib import pyplot as plt

# from PIL import Image

# import habitat
# from habitat.core.logging import logger
# from habitat.core.registry import registry
# from habitat.sims.habitat_simulator.actions import HabitatSimActions
# from habitat.tasks.nav.nav import NavigationTask
# from habitat_baselines.common.baseline_registry import baseline_registry
# from habitat_baselines.config.default import get_config as get_baselines_config
# import habitat_sim
# from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
# from habitat.utils.visualizations.utils import images_to_video

# import shutil

# from add_objects import recompute_navmesh, add_object, remove_object
# from initialize_env import create_env
# from topdownmap import draw_top_down_map, add_path_to_map

import argparse
from final_data_creation import create_final_data

logger.info(f"Inside main.py: Current working directory: {os.getcwd()}")

# def display_sample(
#     obs
# ):  # noqa: B006
#     rgb_img = Image.fromarray(obs[0]['rgb'], mode="RGB")
#     return rgb_img

# def explore_paths(plot_save_dir, video_save_dir):
    
#     PLOTSAVE_DIR = plot_save_dir
#     IMAGE_DIR = video_save_dir

#     env = create_env()
#     goal_radius = env.episodes[0].goals[0].radius
#     if goal_radius is None:
#         goal_radius = 0.25 #Default step size in config
#     follower = ShortestPathFollower(
#         env.habitat_env.sim, goal_radius, False
#     )

#     logger.info(f"Goal Radius: {goal_radius}")
#     env.reset()
#     recompute_navmesh(env.habitat_env.sim)
#     default_spawn = env.habitat_env.sim.get_agent_state().position, env.habitat_env.sim.get_agent_state().rotation

#     logger.info("Inside main.py: explore_paths(): Current working directory: {}".format(os.getcwd()))

#     # Traverse path with and without object on scene
#     for episode in range(3):
#         # Without object
#         env.reset()
#         env.step({"action":"param_change_loc", "action_args": {"pos_crd": default_spawn[0], "angle_crd": default_spawn[1]}})
#         dirname = os.path.join(
#             IMAGE_DIR, "shortest_path_example", "%02d" % episode+"_no_object"
#         )
#         if os.path.exists(dirname):
#             shutil.rmtree(dirname)
#         os.makedirs(dirname)
#         print("Agent stepping around inside environment.")
#         images = []
#         path = []
#         draw_top_down_map(env, image_save_path=os.path.join(PLOTSAVE_DIR, "%02d" % episode+"topdown.png"))
#         envgoal = env.habitat_env.current_episode.goals[0]
#         while not env.habitat_env.episode_over:
#             best_action = follower.get_next_action(
#                 envgoal.view_points[0].agent_state.position

#             )
#             if best_action is None:
#                 break
#             observations, reward, done, info = env.step(best_action)
#             im = observations["rgb"]
#             path.append((env.habitat_env.sim.get_agent_state().position, env.habitat_env.sim.get_agent_state().rotation))
#             print(info)
#             # top_down_map = draw_top_down_map(info, im.shape[0])
#             # output_im = np.concatenate((im, top_down_map), axis=1)
#             output_im = im
#             images.append(output_im)
#         images_to_video(images, dirname, "trajectory")
#         print("Episode finished")
#         add_path_to_map(path, env, image_save_path=os.path.join(PLOTSAVE_DIR, "%02d" % episode+"_withoutobj.png"))

#         # With object
#         env.reset()
#         env.step({"action":"param_change_loc", "action_args": {"pos_crd": default_spawn[0], "angle_crd": default_spawn[1]}})
#         dirname = os.path.join(
#             IMAGE_DIR, "shortest_path_example", "%02d" % episode+"_wt_object"
#         )
#         if os.path.exists(dirname):
#             shutil.rmtree(dirname)
#         os.makedirs(dirname)
#         print("Agent stepping around inside environment.")
#         obj_loc = path[random.randint(0, len(path))][0]
#         objid = add_object(env, obj_loc, scale=2.0, template_id=2) # Add cone
#         images = []
#         path = []
#         while not env.habitat_env.episode_over:
#             best_action = follower.get_next_action(
#                 envgoal.view_points[0].agent_state.position

#             )
#             if best_action is None:
#                 break
#             observations, reward, done, info = env.step(best_action)
#             im = observations["rgb"]
#             path.append((env.habitat_env.sim.get_agent_state().position, env.habitat_env.sim.get_agent_state().rotation))
#             print(info)
#             # top_down_map = draw_top_down_map(info, im.shape[0])
#             # output_im = np.concatenate((im, top_down_map), axis=1)
#             output_im = im
#             images.append(output_im)
#         images_to_video(images, dirname, "trajectory")
#         print("Episode finished")
#         add_path_to_map(path, env, image_save_path=os.path.join(PLOTSAVE_DIR, "%02d" % episode+"_wtobject.png"))
#         remove_object(env, objid)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--plot_save_dir", type=str, required=True)
    # parser.add_argument("--video_save_dir", type=str, required=True)
    # parser.add_argument("--config", type=str, required=True)
    # parser.add_argument("--num_episodes", type=int, default=3)

    parser.add_argument("--num_scenes", type=int, required=True)
    parser.add_argument("--num_ep_per_scene", type=int, required=True)
    parser.add_argument("--step_size", type=float, default=0.25)
    parser.add_argument("--turn_angle", type=int, default=30)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    create_final_data(args.num_scenes, args.num_ep_per_scene, args.step_size, args.turn_angle, args.save_path)

    # if os.path.exists(args.plot_save_dir):
    #     shutil.rmtree(args.plot_save_dir)
    # os.makedirs(args.plot_save_dir)
    # if os.path.exists(args.video_save_dir):
    #     shutil.rmtree(args.video_save_dir)
    # os.makedirs(args.video_save_dir)
    # explore_paths(args.plot_save_dir, args.video_save_dir)
