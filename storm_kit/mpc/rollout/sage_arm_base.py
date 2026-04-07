#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#

"""
SAGE rollout extension for arm tasks.

This file keeps the original ArmBase rollout semantics, but additionally emits
native signed safety-margin outputs:

- `safety_margin_seq`: per-rollout, per-step minimum signed safety margin
- `delta_safe`: per-rollout minimum signed safety margin over the full horizon

This is the preferred, paper-aligned path for SAGE safe-elite selection.
If a controller consumes these native fields, it no longer needs to reconstruct
delta_n internally from rollout state tensors.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.autograd.profiler as profiler

from .arm_base import ArmBase


class SageArmBase(ArmBase):
    """
    Arm rollout with native safety-margin outputs for SAGE.

    The underlying dynamics/cost stack is unchanged. The only semantic
    extension is that rollout outputs now expose the signed safety margin
    directly, instead of forcing the controller to reconstruct it.
    """

    def _compute_primitive_safety_margin_seq(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not hasattr(self, "primitive_collision_cost"):
            return None
        if "link_pos_seq" not in state_dict or "link_rot_seq" not in state_dict:
            return None

        p_cost = self.primitive_collision_cost
        link_pos_seq = state_dict["link_pos_seq"]
        link_rot_seq = state_dict["link_rot_seq"]
        batch_size, horizon, n_links = link_pos_seq.shape[:3]

        if p_cost.batch_size != batch_size:
            p_cost.batch_size = batch_size
            p_cost.robot_world_coll.build_batch_features(
                batch_size * horizon,
                clone_pose=True,
                clone_points=True,
            )

        link_pos_batch = link_pos_seq.view(batch_size * horizon, n_links, 3)
        link_rot_batch = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)
        raw_signed_dist = p_cost.robot_world_coll.check_robot_sphere_collisions(
            link_pos_batch,
            link_rot_batch,
        ).view(batch_size, horizon, n_links)

        # Positive margin means safely outside the collision threshold.
        primitive_margin_seq = -(raw_signed_dist + p_cost.distance_threshold)
        return primitive_margin_seq.amin(dim=-1)

    def _compute_self_safety_margin_seq(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not hasattr(self, "robot_self_collision_cost"):
            return None
        if "state_seq" not in state_dict:
            return None

        self_cost = self.robot_self_collision_cost
        q_seq = state_dict["state_seq"][:, :, : self.n_dofs]
        batch_size, horizon = q_seq.shape[:2]
        q_flat = q_seq.reshape(batch_size * horizon, self.n_dofs)

        raw_signed_dist = self_cost.coll.check_self_collisions_nn(q_flat).view(
            batch_size,
            horizon,
        )

        # Positive margin means safely outside the self-collision threshold.
        self_margin_seq = -(raw_signed_dist + self_cost.distance_threshold)
        return self_margin_seq

    def _build_native_safety_outputs(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        margin_terms = []
        primitive_margin_seq = self._compute_primitive_safety_margin_seq(state_dict)
        self_margin_seq = self._compute_self_safety_margin_seq(state_dict)

        outputs: Dict[str, torch.Tensor] = {}
        if primitive_margin_seq is not None:
            outputs["primitive_safety_margin_seq"] = primitive_margin_seq
            margin_terms.append(primitive_margin_seq)
        if self_margin_seq is not None:
            outputs["self_safety_margin_seq"] = self_margin_seq
            margin_terms.append(self_margin_seq)

        batch_size = state_dict["state_seq"].shape[0]
        horizon = state_dict["state_seq"].shape[1]

        if margin_terms:
            safety_margin_seq = torch.stack(margin_terms, dim=0).amin(dim=0)
        else:
            # If no safety model is active, expose a trivially safe margin so the
            # rollout API remains well-defined. The controller can still decide
            # whether to use or ignore it.
            safety_margin_seq = torch.full(
                (batch_size, horizon),
                float("inf"),
                **self.tensor_args,
            )

        outputs["safety_margin_seq"] = safety_margin_seq
        outputs["delta_safe"] = safety_margin_seq.amin(dim=1)
        outputs["collision_margin_seq"] = safety_margin_seq
        return outputs

    def rollout_fn(self, start_state, act_seq):
        with profiler.record_function("robot_model"):
            state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq)

        with profiler.record_function("cost_fns"):
            cost_seq = self.cost_fn(state_dict, act_seq)

        sim_trajs = dict(
            actions=act_seq,
            costs=cost_seq,
            ee_pos_seq=state_dict["ee_pos_seq"],
            state_seq=state_dict["state_seq"],
            state_dict=state_dict,
            rollout_time=0.0,
        )
        sim_trajs.update(self._build_native_safety_outputs(state_dict))
        return sim_trajs
