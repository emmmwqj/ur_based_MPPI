#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predictive dynamic primitive collision cost with horizon ball forecast."""

from __future__ import annotations

import copy
from typing import Dict, Iterable

import torch
import torch.nn as nn

from storm_kit.geom.sdf.robot_world import RobotWorldCollisionPrimitive


class DynamicPrimitiveCollisionCostPredict(nn.Module):
    """Primitive collision cost with horizon-wise future ball centers.

    Static primitives are evaluated through the existing analytic primitive world.
    The dynamic ball is evaluated separately with a future center sequence so each
    rollout time step uses its own predicted obstacle position.
    """

    def __init__(
        self,
        weight=None,
        world_params=None,
        robot_params=None,
        gaussian_params=None,
        distance_threshold: float = 0.1,
        tensor_args=None,
        traj_dt=None,
        dynamic_ball_name: str = 'dynamic_ball',
        dynamic_ball_safety_margin: float = 0.03,
    ):
        super().__init__()
        if tensor_args is None:
            tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}
        if gaussian_params is None:
            gaussian_params = {}

        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight, **self.tensor_args)
        self.distance_threshold = float(distance_threshold)
        self.traj_dt = None if traj_dt is None else torch.as_tensor(traj_dt, **self.tensor_args).view(-1)

        world_collision_params = copy.deepcopy(world_params["world_model"])
        dynamic_cfg = world_collision_params.get("dynamic_obstacles", {}).get(dynamic_ball_name, {})
        dynamic_sphere_cfg = world_collision_params.get("coll_objs", {}).get("sphere", {}).get(dynamic_ball_name, {})
        if dynamic_ball_name not in world_collision_params.get("coll_objs", {}).get("sphere", {}):
            raise KeyError(f"Dynamic sphere '{dynamic_ball_name}' is missing from world_model.coll_objs.sphere")

        # Keep only static primitives inside the shared primitive world model.
        world_collision_params["coll_objs"]["sphere"].pop(dynamic_ball_name, None)

        robot_collision_params = robot_params["robot_collision_params"]
        self.robot_world_coll = RobotWorldCollisionPrimitive(
            robot_collision_params,
            world_collision_params,
            tensor_args=self.tensor_args,
            bounds=robot_params["world_collision_params"]["bounds"],
            grid_resolution=robot_params["world_collision_params"]["grid_resolution"],
        )

        self.dynamic_ball_name = dynamic_ball_name
        self.dynamic_ball_radius = float(dynamic_sphere_cfg.get("radius", dynamic_cfg.get("radius", 0.06)))
        self.dynamic_ball_safety_margin = float(dynamic_ball_safety_margin)
        self.dynamic_ball_effective_radius = self.dynamic_ball_radius + self.dynamic_ball_safety_margin
        self.dynamic_ball_y_min, self.dynamic_ball_y_max = [
            float(v) for v in dynamic_cfg.get("y_limits", [-0.18, 0.08])
        ]
        self.dynamic_ball_speed_nominal = float(dynamic_cfg.get("speed", 0.1))
        self.dynamic_ball_position_world = torch.as_tensor(
            dynamic_cfg.get("initial_position", dynamic_sphere_cfg.get("position", [0.4, -0.05, 0.4])),
            **self.tensor_args,
        ).view(3)
        self.dynamic_ball_vel_y = torch.as_tensor(self.dynamic_ball_speed_nominal, **self.tensor_args)

        self.batch_size = -1
        self.predictive_dynamic_obstacle_enabled = True
        self.last_dynamic_ball_pos = self.dynamic_ball_position_world.detach().cpu().numpy().copy()
        self.last_dynamic_ball_vel_y = float(self.dynamic_ball_vel_y.item())
        self.last_predicted_ball_seq = None
        self.last_min_dynamic_ball_distance = float('nan')
        self.last_min_dynamic_ball_margin = float('nan')
        self.last_dynamic_collision_violation_count = 0

    def list_dynamic_spheres(self) -> Iterable[str]:
        return (self.dynamic_ball_name,)

    def set_dynamic_sphere_position_world(self, sphere_name: str, position_world) -> None:
        self.set_dynamic_sphere_state_world(sphere_name, position_world, None)

    def set_dynamic_sphere_state_world(self, sphere_name: str, position_world, vel_y) -> None:
        if sphere_name != self.dynamic_ball_name:
            raise KeyError(f"Predictive cost only supports '{self.dynamic_ball_name}', got '{sphere_name}'")
        pos = torch.as_tensor(position_world, **self.tensor_args).view(3)
        self.dynamic_ball_position_world = pos
        self.last_dynamic_ball_pos = pos.detach().cpu().numpy().copy()
        if vel_y is not None:
            vel_y_tensor = torch.as_tensor(float(vel_y), **self.tensor_args)
            if torch.isfinite(vel_y_tensor):
                self.dynamic_ball_vel_y = vel_y_tensor
                self.last_dynamic_ball_vel_y = float(vel_y_tensor.item())

    def _get_time_sequence(self, horizon: int) -> torch.Tensor:
        if self.traj_dt is not None and self.traj_dt.numel() >= horizon:
            return self.traj_dt[:horizon]
        dt = 0.02
        return torch.arange(1, horizon + 1, **self.tensor_args) * dt

    def _reflect_y(self, y_values: torch.Tensor) -> torch.Tensor:
        span = self.dynamic_ball_y_max - self.dynamic_ball_y_min
        if span <= 0.0:
            return torch.clamp(y_values, min=self.dynamic_ball_y_min, max=self.dynamic_ball_y_max)
        period = 2.0 * span
        shifted = y_values - self.dynamic_ball_y_min
        wrapped = torch.remainder(shifted, period)
        return torch.where(
            wrapped <= span,
            self.dynamic_ball_y_min + wrapped,
            self.dynamic_ball_y_max - (wrapped - span),
        )

    def predict_dynamic_ball_center_seq(self, horizon: int) -> torch.Tensor:
        t_seq = self._get_time_sequence(horizon)
        pos_seq = self.dynamic_ball_position_world.view(1, 3).repeat(horizon, 1)
        y_future = self.dynamic_ball_position_world[1] + self.dynamic_ball_vel_y * t_seq
        pos_seq[:, 1] = self._reflect_y(y_future)
        return pos_seq

    def _dynamic_sphere_distance(self, spheres, dynamic_centers, obstacle_radius: float):
        center_dist = torch.norm(spheres[..., :3] - dynamic_centers.unsqueeze(1), dim=-1)
        return obstacle_radius + spheres[..., 3] - center_dist

    def _compute_link_distance(self, w_link_spheres, dynamic_centers):
        total_batch = dynamic_centers.shape[0]
        n_links = len(w_link_spheres)
        dist = torch.empty((total_batch, n_links), **self.tensor_args)

        min_distance = None
        min_margin = None
        violation_count = 0

        for i, spheres in enumerate(w_link_spheres):
            static_d = self.robot_world_coll.world_coll.get_sphere_distance(spheres)
            static_link_dist = torch.max(torch.max(static_d, dim=-1)[0], dim=-1)[0]

            dynamic_margin_d = self._dynamic_sphere_distance(spheres, dynamic_centers, self.dynamic_ball_effective_radius)
            dynamic_margin_link_dist = torch.max(dynamic_margin_d, dim=-1)[0]
            dist[:, i] = torch.maximum(static_link_dist, dynamic_margin_link_dist)

            dynamic_distance_clearance = -self._dynamic_sphere_distance(spheres, dynamic_centers, self.dynamic_ball_radius)
            dynamic_margin_clearance = -dynamic_margin_d
            local_min_distance = torch.min(dynamic_distance_clearance)
            local_min_margin = torch.min(dynamic_margin_clearance)
            if min_distance is None or local_min_distance < min_distance:
                min_distance = local_min_distance
            if min_margin is None or local_min_margin < min_margin:
                min_margin = local_min_margin
            violation_count += int(torch.count_nonzero(dynamic_margin_clearance < 0.0).item())

        self.last_min_dynamic_ball_distance = float(min_distance.item()) if min_distance is not None else float('nan')
        self.last_min_dynamic_ball_margin = float(min_margin.item()) if min_margin is not None else float('nan')
        self.last_dynamic_collision_violation_count = int(violation_count)
        return dist

    def forward(self, link_pos_seq, link_rot_seq):
        inp_device = link_pos_seq.device
        batch_size = link_pos_seq.shape[0]
        horizon = link_pos_seq.shape[1]
        total_batch = batch_size * horizon

        if self.batch_size != batch_size:
            self.batch_size = batch_size
            self.robot_world_coll.build_batch_features(total_batch, clone_pose=True, clone_points=True)

        pred_seq = self.predict_dynamic_ball_center_seq(horizon)
        dynamic_centers = pred_seq.unsqueeze(0).repeat(batch_size, 1, 1).reshape(total_batch, 3)
        self.last_predicted_ball_seq = pred_seq.detach().cpu().numpy()

        n_links = link_pos_seq.shape[2]
        link_pos_batch = link_pos_seq.view(total_batch, n_links, 3)
        link_rot_batch = link_rot_seq.view(total_batch, n_links, 3, 3)

        self.robot_world_coll.robot_coll.update_batch_robot_collision_objs(link_pos_batch, link_rot_batch)
        w_link_spheres = self.robot_world_coll.robot_coll.get_batch_robot_link_spheres()
        dist = self._compute_link_distance(w_link_spheres, dynamic_centers)

        dist = dist.view(batch_size, horizon, n_links)
        dist = dist + self.distance_threshold
        dist = torch.clamp(dist, min=0.0, max=0.2)
        dist = dist / 0.25
        cost = torch.sum(dist, dim=-1)
        cost = self.weight * cost
        return cost.to(inp_device)
