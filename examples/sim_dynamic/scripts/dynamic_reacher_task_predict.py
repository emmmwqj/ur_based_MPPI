#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predictive dynamic Gazebo reacher task wrapper."""

from __future__ import annotations

from examples.sim_dynamic.scripts.dynamic_arm_reacher_predict import DynamicArmReacherPredict
from examples.sim_gazebo.reach_static_ur7e import GazeboReacherTask


class DynamicGazeboReacherTaskPredict(GazeboReacherTask):
    def get_rollout_fn(self, **kwargs):
        return DynamicArmReacherPredict(**kwargs)

    def set_dynamic_sphere_state_world(self, sphere_name: str, position_world, vel_y) -> None:
        self.controller.rollout_fn.set_dynamic_sphere_state_world(sphere_name, position_world, vel_y)

    def get_dynamic_ball_metrics(self):
        return self.controller.rollout_fn.get_dynamic_ball_metrics()
