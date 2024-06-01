import os
from loguru import logger

import git

import habitat

# # Set paths
# repo = git.Repo(".", search_parent_directories=True)
# dir_path = repo.working_tree_dir
# data_path = os.path.join(dir_path, "data")
# os.chdir(dir_path)
logger.info(f"Inside initialize_env.py: Current working directory: {os.getcwd()}")

# import sys
# sys.path.append(os.getcwd())
# Add parametrized actions to the config
from parametrized_action import add_param_actions

def define_config():
    # Config for HSSD Dataset and DDPPO model with pretrained weights (we don't use this model for now)
    dir_path = os.getcwd()
    config = habitat.get_config(
    config_path=os.path.join(
        dir_path,
        "habitat-baselines/habitat_baselines/config/objectnav/hssd-200_ver_clip_hssd-hab.yaml"
        # "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hssd-hab.yaml",
    ),
    overrides=[
        # "habitat.environment.iterator_options.shuffle=False",
        "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=512",
        "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=512",
        # "habitat.simulator.create_renderer=False",
        "habitat.simulator.habitat_sim_v0.enable_hbao=True",
        "habitat_baselines.eval.video_option=[\"disk\"]",
        "habitat_baselines.rl.ddppo.pretrained_weights=data/ddppo-models/hssd_pretrained_best_on_hm3d_ckpt.pth",
        "habitat.simulator.habitat_sim_v0.allow_sliding=false"])
    config = add_param_actions(config)
    return config

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
    
def create_env(dataset=None):
    config = define_config()
    if dataset is not None:
        env = SimpleRLEnv(config=config, dataset=dataset)
    else:
        env = SimpleRLEnv(config=config)
    return env
