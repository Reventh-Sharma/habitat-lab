import os
from loguru import logger
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from habitat.core.utils import try_cv2_import
cv2 = try_cv2_import()

logger.info("Inside topdownmap.py: Current working directory: {}".format(os.getcwd()))

def draw_top_down_map(env, image_save_path='images/top_down_map.png'):
    top_down_map = maps.get_topdown_map_from_sim(env.habitat_env.sim, map_resolution=512)
    # By default, `get_topdown_map_from_sim` returns image
    # containing 0 if occupied, 1 if unoccupied, and 2 if border
    # The line below recolors returned image so that
    # occupied regions are colored in [255, 255, 255],
    # unoccupied in [128, 128, 128] and border is [0, 0, 0]
    recolor_map = np.array(
                [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
            )
    top_down_map = recolor_map[top_down_map]
    plt.imshow(top_down_map)
    plt.title("Top Down Map")
    plt.show()
    if image_save_path!='':
        plt.savefig(image_save_path)
    return top_down_map

# def draw_top_down_map(info, size):
#     top_down_map_key = "top_down_map"
#     if top_down_map_key in info:
#         top_down_map = maps.colorize_draw_agent_and_fit_to_height(
#             info[top_down_map_key], size
#         )
#     fig = plt.figure()
#     img = plt.imshow(top_down_map)
#     plt.close(fig)
#     return img.get_array().filled(fill_value=0)

def display_sample(
    rgb_obs
):  # noqa: B006
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    return rgb_img

def get_3drendered_top_down_map(env):
    top_down_map = maps.get_topdown_map_from_sim(env.habitat_env.sim)
    z, x = maps.from_grid(top_down_map.shape[0]//2, top_down_map.shape[1]//2, top_down_map.shape, env.habitat_env.sim)
    pos, _ = env.habitat_env.sim.get_agent_state().position, env.habitat_env.sim.get_agent_state().rotation
    obs = env.step({'action': 'param_change_loc', 'action_args': {'pos_crd': np.array([x, pos[1], z]), 'angle_crd': np.array([0.0, 1.0, 0.0, 0.0])}})[0]
    logger.info(f"{obs.keys()}")
    img = display_sample(obs["top_rgb"])
    bbox = img.getbbox()
    return img.crop(bbox)

def add_path_to_map(path, env, info, goal, image_save_path='images/top_down_mapwtpath.png'):
    top_down_map = draw_top_down_map(env, '')
    x = env.habitat_env.sim.get_agent_state().position
    agent_map_coord = maps.to_grid(x[2], x[0], (top_down_map.shape[0], top_down_map.shape[1]), env.habitat_env.sim)
    image = maps.draw_agent(top_down_map, agent_map_coord, info['top_down_map']['agent_angle'][0])
    goalpos = goal.position
    goal_map_coord = maps.to_grid(goalpos[2], goalpos[0], (image.shape[0], image.shape[1]), env.habitat_env.sim)
    image = cv2.circle(image, goal_map_coord, radius=5, color=(0, 255, 0), thickness=6)
    reformatted_path = [maps.to_grid(x[0][2], x[0][0], (image.shape[0], image.shape[1]), env.habitat_env.sim) for x in path]
    maps.draw_path(image, reformatted_path, color=10, thickness=2)
    return image
    # plt.imshow(top_down_map)
    # plt.title("top_down_map.png")
    # plt.show()
    # plt.savefig(image_save_path)