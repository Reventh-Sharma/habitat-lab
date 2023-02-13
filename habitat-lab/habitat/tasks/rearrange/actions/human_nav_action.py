# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from gym import spaces

import habitat_sim
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.rearrange.actions.actions import BaseVelAction, HumanJointAction
from habitat.tasks.rearrange.utils import get_robot_spawns
from habitat.tasks.utils import get_angle

import magnum as mn
import human_controllers.amass_human_controller as amass_human_controller




@registry.register_task_action
class HumanNavAction(HumanJointAction):
    """
    An action that will convert the index of an entity (in the sense of
    `PddlEntity`) to navigate to and convert this to base control to move the
    robot to the closest navigable position to that entity. The entity index is
    the index into the list of all available entities in the current scene.
    """
     
    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, **kwargs)
        self._task = task
        self.human_controller = None
        self._poss_entities = (
            self._task.pddl_problem.get_ordered_entities_list()
        )

        self._prev_ep_id = None
        self._targets = {}
        self.gen = 0
        self.curr_ind_map = {'cont': 0}

    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "human_nav_action": spaces.Box(
                    shape=(1,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._task._episode_id != self._prev_ep_id:
            self._targets = {}
            self._prev_ep_id = self._task._episode_id

    def _get_target_for_idx(self, nav_to_target_idx: int):
        if nav_to_target_idx not in self._targets:
            nav_to_obj = self._poss_entities[nav_to_target_idx]
            obj_pos = self._task.pddl_problem.sim_info.get_entity_pos(
                nav_to_obj
            )
            start_pos, _, _ = get_robot_spawns(
                np.array(obj_pos),
                0.0,
                self._config.spawn_max_dist_to_obj,
                self._sim,
                self._config.num_spawn_attempts,
                1,
            )
            # TODO: not sure if it is clean to access this here
            # breakpoint()
            self.human_controller.reset(self.cur_human.base_pos)
            # breakpoint()
            self._targets[nav_to_target_idx] = (start_pos, np.array(obj_pos))
        return self._targets[nav_to_target_idx]

    def _path_to_point(self, point):
        agent_pos = self.cur_human.base_pos
        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = point
        found_path = self._sim.pathfinder.find_path(path)

        
        if not found_path:
            return None
        return path.points

    def step(self, *args, is_last_action, **kwargs):
        nav_to_target_idx = kwargs[
            self._action_arg_prefix + "human_nav_action"
        ]
        if nav_to_target_idx <= 0 or nav_to_target_idx > len(
            self._poss_entities
        ):
            if is_last_action:
                return self._sim.step(HabitatSimActions.base_velocity)
            else:
                return {}
        nav_to_target_idx = int(nav_to_target_idx[0]) - 1

        final_nav_targ, obj_targ_pos = self._get_target_for_idx(
            nav_to_target_idx
        )
        curr_path_points = self._path_to_point(final_nav_targ)

        sim = self._sim

        if curr_path_points is None:
            # path not found
            new_pos, new_trans = self.human_controller.stop()
        else:
            index = 0
            distance = 0

            robot_pos = np.array(self.cur_human.base_pos)
            index = 1

            cur_nav_targ = curr_path_points[index]
            base_T = self.cur_human.base_transformation
            # TODO: change this so that front is aligned with x, not z
            forward = np.array([0, 0, 1.00])
            robot_forward = np.array(base_T.transform_vector(forward))
            
            p1 = robot_pos
            p2 = robot_pos + robot_forward
            robot_forward = robot_forward[[0, 2]]
            
            rel_targ = cur_nav_targ - robot_pos

            rel_pos = (obj_targ_pos - robot_pos)[[0, 2]]


            dist_to_final_nav_targ = np.linalg.norm(
                (final_nav_targ - robot_pos)[[0, 2]]
            )
            
            angle_to_obj = get_angle(robot_forward, rel_pos)
            at_goal = (
                dist_to_final_nav_targ < self._config.dist_thresh
                and angle_to_obj < self._config.turn_thresh
            )
            if not at_goal:
                if dist_to_final_nav_targ < self._config.dist_thresh:

                    new_pos, new_trans = self.human_controller.compute_turn(
                        rel_pos
                    )
                else:
                    new_pos, new_trans = self.human_controller.walk(rel_targ)
            else:
                new_pos, new_trans = self.human_controller.stop()
            # breakpoint()
            # print("DISTANCE AND MOTION", dist_to_final_nav_targ, rel_targ, new_trans.translation, 'offset', self.human_controller.translation_offset)
        base_action = amass_human_controller.AmassHumanController.transformAction(new_pos, new_trans)
        # print(new_trans, robot_pos)
        kwargs[f"{self._action_arg_prefix}human_joints_trans"] = base_action
        return super().step(*args, is_last_action=is_last_action, **kwargs)
