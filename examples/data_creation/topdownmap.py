import os
from loguru import logger
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt
import numpy as np

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
    if image_save_path!='':
        plt.imshow(top_down_map)
        plt.title("Top Down Map")
        plt.show()
        plt.savefig(image_save_path)
    return top_down_map

def add_path_to_map(path, env, image_save_path='images/top_down_mapwtpath.png'):
    top_down_map = draw_top_down_map(env, '')
    reformatted_path = [maps.to_grid(x[0][2], x[0][0], (top_down_map.shape[0], top_down_map.shape[1]), env.habitat_env.sim) for x in path]
    maps.draw_path(top_down_map, reformatted_path, color=10, thickness=2)
    plt.imshow(top_down_map)
    plt.title("top_down_map.png")
    plt.show()
    plt.savefig(image_save_path)