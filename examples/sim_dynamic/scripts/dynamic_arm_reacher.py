#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic obstacle arm reacher rollout."""

from __future__ import annotations

from storm_kit.mpc.rollout.arm_reacher import ArmReacher

from examples.sim_dynamic.scripts.dynamic_primitive_collision_cost import DynamicPrimitiveCollisionCost


class DynamicArmReacher(ArmReacher):
    def __init__(self, exp_params, tensor_args=None, world_params=None):
        if tensor_args is None:
            tensor_args = {"device": "cpu", "dtype": None}
        super().__init__(exp_params=exp_params, tensor_args=tensor_args, world_params=world_params)
        if exp_params["cost"]["primitive_collision"]["weight"] > 0.0:
            self.primitive_collision_cost = DynamicPrimitiveCollisionCost(
                world_params=world_params,
                robot_params=exp_params["robot_params"],
                tensor_args=self.tensor_args,
                **exp_params["cost"]["primitive_collision"],
            )

    def set_dynamic_sphere_position_world(self, sphere_name: str, position_world) -> None:
        self.primitive_collision_cost.set_dynamic_sphere_position_world(sphere_name, position_world)
