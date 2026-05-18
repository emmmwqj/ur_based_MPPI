#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic Gazebo reacher task wrapper."""

from __future__ import annotations

from examples.sim_dynamic.scripts.dynamic_arm_reacher import DynamicArmReacher
from examples.sim_gazebo.reach_static_ur7e import GazeboReacherTask


class DynamicGazeboReacherTask(GazeboReacherTask):
    def get_rollout_fn(self, **kwargs):
        return DynamicArmReacher(**kwargs)

    def set_dynamic_sphere_position_world(self, sphere_name: str, position_world) -> None:
        self.controller.rollout_fn.set_dynamic_sphere_position_world(sphere_name, position_world)
