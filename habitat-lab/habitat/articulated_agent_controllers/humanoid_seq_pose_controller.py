#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import magnum as mn
import numpy as np

from habitat.articulated_agent_controllers import (
    HumanoidBaseController,
    Motion,
)


class HumanoidSeqPoseController(HumanoidBaseController):
    """
    Humanoid Seq Pose Controller, replays a sequence of humanoid poses.
        :param motion_pose_path: file containing the motion poses we want to play.
        :param motion_fps: the FPS at which we should be advancing the pose.
        :base_offset: what is the offset between the root of the character and their feet.
    """

    def __init__(
        self,
        motion_pose_path,
        motion_fps=30,
        base_offset=(0, 0.9, 0),
    ):
        super().__init__(motion_fps, base_offset)

        if not os.path.isfile(motion_pose_path):
            raise RuntimeError(
                f"Path does {motion_pose_path} not exist. Reach out to the paper authors to obtain this data."
            )

        motion_info = np.load(motion_pose_path, allow_pickle=True)
        motion_info = motion_info["pose_motion"]
        self.humanoid_motion = Motion(
            motion_info["joints_array"],
            motion_info["transform_array"],
            motion_info["displacement"],
            motion_info["fps"],
        )
        self.walk_mocap_frame = 0
        self.ref_pose = mn.Matrix4()
        self.first_pose = mn.Matrix4(
            self.humanoid_motion.poses[0].root_transform
        )

    def reset(self, base_transformation) -> None:
        """Reset the joints on the human. (Put in rest state)"""
        super().reset(base_transformation)
        self.walk_mocap_frame = 0

        # The first pose of the motion file will be set at base_transformation
        self.ref_pose = base_transformation

        self.calculate_pose_mdm()

    def next_pose(self, cycle=False):
        if cycle:
            self.walk_mocap_frame = (
                self.walk_mocap_frame + 1
            ) % self.humanoid_motion.num_poses
        else:
            self.walk_mocap_frame = min(
                self.walk_mocap_frame + 1, self.humanoid_motion.num_poses - 1
            )

    def prev_pose(self, cycle=False):
        if cycle:
            self.walk_mocap_frame = (
                self.walk_mocap_frame - 1
            ) % self.humanoid_motion.num_poses
        else:
            self.walk_mocap_frame = max(0, self.walk_mocap_frame - 1)

    def calculate_pose_mdm(self):
        curr_transform = mn.Matrix4(
            self.humanoid_motion.poses[self.walk_mocap_frame].root_transform
        )
        curr_transform.translation = (
            curr_transform.translation
            - self.first_pose.translation
            + self.ref_pose.translation
        )
        curr_transform.translation = curr_transform.translation - mn.Vector3(
            0, 0.9, 0
        )
        curr_poses = self.humanoid_motion.poses[self.walk_mocap_frame].joints
        self.obj_transform_offset = curr_transform
        self.joint_pose = curr_poses
        self.walk_mocap_frame = (
            self.walk_mocap_frame + self.humanoid_motion.fps
        ) % self.humanoid_motion.num_poses