import os
from loguru import logger

import git

import habitat

from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import HeadingSensorConfig, TopRGBSensorConfig, TopDownMapMeasurementConfig, FogOfWarConfig, CollisionsMeasurementConfig
from habitat.config import read_write
# # Set paths
# repo = git.Repo(".", search_parent_directories=True)
# dir_path = repo.working_tree_dir
# data_path = os.path.join(dir_path, "data")
# os.chdir(dir_path)
logger.info(f"Inside initialize_env.py: Current working directory: {os.getcwd()}")

import sys
sys.path.append(os.getcwd())
# Add parametrized actions to the config
from examples.random_walk_sim.parametrized_action import add_param_actions

def define_config(step_size=0.25, turn_angle=30):
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
            f"habitat.simulator.forward_step_size={step_size}",
            f"habitat.simulator.turn_angle={turn_angle}",
            "habitat_baselines.eval.split=train",
            "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=512",
            "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=512",
            # "habitat.simulator.create_renderer=False",
            "habitat.simulator.habitat_sim_v0.enable_hbao=True",
            "habitat.simulator.habitat_sim_v0.enable_physics=True",
            "habitat_baselines.eval.video_option=[\"disk\"]",
            "habitat_baselines.rl.ddppo.pretrained_weights=data/ddppo-models/hssd_pretrained_best_on_hm3d_ckpt.pth",
            "habitat.simulator.habitat_sim_v0.allow_sliding=false",
            # "habitat.environment.iterator_options.num_episode_sample=1",
            # "habitat.environment.iterator_options.shuffle=False" Don't use these two else random episodes won't get generated
        ],
    )
    with read_write(config):
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        agent_config.sim_sensors.update(
            {"toprgb_sensor": TopRGBSensorConfig(),
            }
        )
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=False,
                    draw_border=True,
                    draw_shortest_path=False,
                    draw_view_points=False,
                    draw_goal_positions=True,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(
                        draw=False,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                )
            }
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.toprgb_sensor.height=1920
        config.habitat.simulator.agents.main_agent.sim_sensors.toprgb_sensor.width=1080
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
    
def create_env(step_size=0.25, turn_angle=30):
    config = define_config(step_size, turn_angle)
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset, 
    )
    env = SimpleRLEnv(config=config, dataset=dataset)
    return env
