#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAGE rollout with predictive dynamic primitive collision."""

from __future__ import annotations

from storm_kit.mpc.rollout.sage_arm_reacher import SageArmReacher

from examples.sage_sim_dynamic.scripts.sage_dynamic_primitive_collision_cost_predict import (
    SageDynamicPrimitiveCollisionCostPredict,
)


class SageDynamicArmReacherPredict(SageArmReacher):
    def __init__(self, exp_params, tensor_args=None, world_params=None):
        if tensor_args is None:
            tensor_args = {"device": "cpu", "dtype": None}
        super().__init__(exp_params=exp_params, tensor_args=tensor_args, world_params=world_params)
        if exp_params["cost"]["primitive_collision"]["weight"] > 0.0:
            safety_margin = float(exp_params.get("task", {}).get("dynamic_ball_safety_margin", 0.03))
            self.primitive_collision_cost = SageDynamicPrimitiveCollisionCostPredict(
                world_params=world_params,
                robot_params=exp_params["robot_params"],
                tensor_args=self.tensor_args,
                traj_dt=self.traj_dt,
                dynamic_ball_safety_margin=safety_margin,
                **exp_params["cost"]["primitive_collision"],
            )

    def set_dynamic_sphere_state_world(self, sphere_name: str, position_world, vel_y) -> None:
        self.primitive_collision_cost.set_dynamic_sphere_state_world(sphere_name, position_world, vel_y)

    def get_dynamic_ball_metrics(self):
        cost = self.primitive_collision_cost
        return {
            'predictive_dynamic_obstacle_enabled': bool(getattr(cost, 'predictive_dynamic_obstacle_enabled', False)),
            'dynamic_ball_pos': getattr(cost, 'last_dynamic_ball_pos', None),
            'dynamic_ball_vel_y': float(getattr(cost, 'last_dynamic_ball_vel_y', 0.0)),
            'rollout_min_dynamic_ball_distance': float(getattr(cost, 'last_rollout_min_dynamic_ball_distance', float('nan'))),
            'rollout_min_dynamic_ball_margin': float(getattr(cost, 'last_rollout_min_dynamic_ball_margin', float('nan'))),
            'rollout_sample_violation_count': int(getattr(cost, 'last_rollout_sample_violation_count', 0)),
            'rollout_time_violation_count': int(getattr(cost, 'last_rollout_time_violation_count', 0)),
        }
