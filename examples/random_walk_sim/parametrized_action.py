#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


r"""
Class to add parametrized actions for agent
"""

from dataclasses import dataclass

import magnum as mn
import numpy as np
import math

import habitat
from habitat.config.default_structured_configs import ActionConfig
from habitat.tasks.nav.nav import SimulatorTaskAction
from loguru import logger

# This is the configuration for our action.
@dataclass
class ParamaterizedActionConfig(ActionConfig):
    # We will change these in the configuration
    move_amount: float = 0.0  
    turn_angle: int = 0
    straff_angle: int = 0

# We define and register our actions as follows.
# the __init__ method receives a sim and config argument.
@habitat.registry.register_task_action
class ParamaterizedTranslation(SimulatorTaskAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.move_amount = config.move_amount

    # Change state of simulator by specified translation amounts
    def parametrized_translation(self):
        logger.info(f"Move amount: {self.move_amount}")
        agent_state = self._sim.get_agent_state()
        normalized_quaternion = agent_state.rotation
        agent_mn_quat = mn.Quaternion(
            normalized_quaternion.imag, normalized_quaternion.real
        )
        forward = agent_mn_quat.transform_vector(-mn.Vector3.z_axis())
        translation = forward * self.move_amount
        final_position = self._sim.pathfinder.try_step(  # type: ignore
            agent_state.position, agent_state.position + translation
        )
        self._sim.set_agent_state(
            final_position,
            [*agent_mn_quat.vector, agent_mn_quat.scalar],
            reset_sensors=False,
        )
        
        
    def _get_uuid(self, *args, **kwargs) -> str:
        return "param_translation"

    # Step method for translation
    def step(self, move_amount=0, **kwargs):
        print(
            f"Calling {self._get_uuid()}"
        )
        self.move_amount = move_amount
        logger.info(f"Move amount: {self.move_amount}")
        # This is where the code for the new action goes. Here we use a
        # helper method but you could directly modify the simulation here.
        self.parametrized_translation()

@habitat.registry.register_task_action
class ParamaterizedRotation(SimulatorTaskAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.turn_angle = config.turn_angle
    
    # Change state of simulator by specified rotation amount
    def parametrized_rotation(self):
        logger.info(f"Turn angle: {self.turn_angle}")
        agent_state = self._sim.get_agent_state()
        normalized_quaternion = agent_state.rotation
        agent_mn_quat = mn.Quaternion(
            normalized_quaternion.imag, normalized_quaternion.real
        )
        turn_angle = mn.Deg(self.turn_angle)
        rotation = mn.Quaternion.rotation(turn_angle, mn.Vector3.y_axis())
        turned_state = agent_mn_quat * rotation
        turned_state = turned_state.normalized()
        self._sim.set_agent_state(
            agent_state.position,
            [*turned_state.vector, turned_state.scalar],
            reset_sensors=False,
        )
        
    def _get_uuid(self, *args, **kwargs) -> str:
        return "param_rotate"
    
    # Step method for rotation, angles in anticlockwise direction
    def step(self, turn_angle=0, **kwargs):
        print(
            f"Calling {self._get_uuid()}"
        )
        self.turn_angle = turn_angle
        logger.info(f"Turn angle: {self.turn_angle}")
        # This is where the code for the new action goes. Here we use a
        # helper method but you could directly modify the simulation here.
        self.parametrized_rotation()


# We define and register our actions as follows.
# the __init__ method receives a sim and config argument.
@habitat.registry.register_task_action
class ParamaterizedStraffing(SimulatorTaskAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.move_amount = config.move_amount
        self.straff_angle = config.straff_angle

    # Change state of simulator by specified translation amount at an angle defined by straff_angle
    def parametrized_straffing(self):
        logger.info(f"Straff angle: {self.straff_angle}, Move amount: {self.move_amount}")
        agent_state = self._sim.get_agent_state()
        normalized_quaternion = agent_state.rotation
        agent_mn_quat = mn.Quaternion(
            normalized_quaternion.imag, normalized_quaternion.real
        )
        forward = agent_mn_quat.transform_vector(-mn.Vector3.z_axis())
        straff_angle = mn.Deg(self.straff_angle)
        rotation = mn.Quaternion.rotation(straff_angle, mn.Vector3.y_axis())
        delta_position = rotation.transform_vector(forward) * self.move_amount
        final_position = self._sim.pathfinder.try_step(  # type: ignore
        agent_state.position, agent_state.position + delta_position
        )

        self._sim.set_agent_state(
            final_position,
            [*agent_mn_quat.vector, agent_mn_quat.scalar],
            reset_sensors=False,
        )
        
        
    def _get_uuid(self, *args, **kwargs) -> str:
        return "param_straffing"

    # Step method for straffing, angles in anticlockwise direction
    def step(self, straff_angle=0, move_amount=0, **kwargs):
        print(
            f"Calling {self._get_uuid()}"
        )
        self.straff_angle = straff_angle
        self.move_amount = move_amount
        logger.info(f"Straff angle: {self.straff_angle}, Move amount: {self.move_amount}")
        # This is where the code for the new action goes. Here we use a
        # helper method but you could directly modify the simulation here.
        if self.move_amount != 0 and self.straff_angle != 0:
            self.parametrized_straffing()
        else:
            self.straff_angle = 0
            self.move_amount = 0
            raise ValueError("Parameter move_amount and/or straffe_angle needs to be non-zero for straffing")

# This is the configuration for our action.
@dataclass
class ParamaterizedChangeLocConfig(ActionConfig):
    pos_cord = [0.0, 0.0, 0.0]
    angle_cord = [0.0, 0.0, 0.0, 0.0]

@habitat.registry.register_task_action
class ParamaterizedChangeLocation(SimulatorTaskAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.pos_cord = np.array([0.0, 0.0, 0.0])
        self.angle_cord = np.array([0.0, 0.0, 0.0, 0.0])

    # Change state of simulator by specified translation amountS
    def param_change_loc(self):
        self._sim.set_agent_state(
            self.pos_cord,
            self.angle_cord,
            reset_sensors=False,
        )
        
        
    def _get_uuid(self, *args, **kwargs) -> str:
        return "param_change_loc"

    # Step method for translation
    def step(self, pos_crd, angle_crd, **kwargs):
        self.pos_cord = pos_crd
        self.angle_cord = angle_crd
        self.param_change_loc()

# Returns updated config with new actions registered
def add_param_actions(config):

    with habitat.config.read_write(config):
        # Add a simple action config to the config.habitat.task.actions dictionary
        # Here we do it via code, but you can easily add them to a yaml config as well
        config.habitat.task.actions["param_translate"] = ParamaterizedActionConfig(type="ParamaterizedTranslation")
        config.habitat.task.actions["param_rotate"] = ParamaterizedActionConfig(type="ParamaterizedRotation")
        config.habitat.task.actions["param_straff"] = ParamaterizedActionConfig(type="ParamaterizedStraffing")
        config.habitat.task.actions["param_change_loc"] = ParamaterizedChangeLocConfig(type="ParamaterizedChangeLocation")
    
    return config


if __name__ == "__main__":
    add_param_actions(habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml"))
