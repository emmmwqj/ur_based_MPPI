#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic primitive collision cost for online sphere updates."""

from __future__ import annotations

import copy
from typing import Dict, Iterable

import torch
import torch.nn as nn

from storm_kit.geom.sdf.robot_world import RobotWorldCollisionPrimitive


class DynamicPrimitiveCollisionCost(nn.Module):
    """Primitive collision cost that supports online obstacle sphere center updates.

    This mirrors the existing PrimitiveCollisionCost behavior, but evaluates the
    primitive distances analytically through ``get_robot_env_sdf`` so that the
    obstacle sphere position can be updated every control step without rebuilding
    a static world SDF grid.
    """

    def __init__(
        self,
        weight=None,
        world_params=None,
        robot_params=None,
        gaussian_params=None,
        distance_threshold: float = 0.1,
        tensor_args=None,
    ):
        super().__init__()
        if tensor_args is None:
            tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}
        if gaussian_params is None:
            gaussian_params = {}

        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight, **self.tensor_args)
        self.distance_threshold = float(distance_threshold)

        robot_collision_params = robot_params["robot_collision_params"]
        world_collision_params = copy.deepcopy(world_params["world_model"])
        self.robot_world_coll = RobotWorldCollisionPrimitive(
            robot_collision_params,
            world_collision_params,
            tensor_args=self.tensor_args,
            bounds=robot_params["world_collision_params"]["bounds"],
            grid_resolution=robot_params["world_collision_params"]["grid_resolution"],
        )

        sphere_names = list(world_collision_params.get("coll_objs", {}).get("sphere", {}).keys())
        self._sphere_name_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(sphere_names)}
        self.batch_size = -1

    def list_dynamic_spheres(self) -> Iterable[str]:
        return tuple(self._sphere_name_to_index.keys())

    def set_dynamic_sphere_position_world(self, sphere_name: str, position_world) -> None:
        if sphere_name not in self._sphere_name_to_index:
            raise KeyError(f"Unknown dynamic sphere: {sphere_name}")
        idx = self._sphere_name_to_index[sphere_name]
        pos = torch.as_tensor(position_world, **self.tensor_args).view(1, 1, 3)
        self.robot_world_coll.world_coll._world_spheres[:, idx : idx + 1, :3] = pos

    def forward(self, link_pos_seq, link_rot_seq):
        inp_device = link_pos_seq.device
        batch_size = link_pos_seq.shape[0]
        horizon = link_pos_seq.shape[1]
        n_links = link_pos_seq.shape[2]

        if self.batch_size != batch_size:
            self.batch_size = batch_size
            self.robot_world_coll.build_batch_features(self.batch_size * horizon, clone_pose=True, clone_points=True)

        link_pos_batch = link_pos_seq.view(batch_size * horizon, n_links, 3)
        link_rot_batch = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)

        dist = self.robot_world_coll.get_robot_env_sdf(link_pos_batch, link_rot_batch)
        dist = dist.view(batch_size, horizon, n_links)
        dist = dist + self.distance_threshold
        dist = torch.clamp(dist, min=0.0, max=0.2)
        dist = dist / 0.25
        cost = torch.sum(dist, dim=-1)
        cost = self.weight * cost
        return cost.to(inp_device)
