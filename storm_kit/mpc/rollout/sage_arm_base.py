#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#

"""
SAGE rollout extension for arm tasks.

This file keeps the original ArmBase rollout semantics while returning the
expanded rollout dictionary (`state_dict`, `state_seq`, `ee_pos_seq`) that the
clean SAGE controller expects.

Important performance note:
- this rollout no longer emits extra collision-derived bookkeeping tensors
- the earlier implementation recomputed collision distances a second time only
  for controller-side bookkeeping, which duplicated the work already done
  inside `cost_fn()`
- the current SAGE controller is purely cost-driven for proposal updates, so
  that extra collision pass is unnecessary on the main path
"""

from __future__ import annotations

import torch.autograd.profiler as profiler

from .arm_base import ArmBase


class SageArmBase(ArmBase):
    """
    Arm rollout for clean SAGE.

    The underlying dynamics/cost stack is unchanged. Compared with ArmBase, the
    rollout returns `state_dict` so controller-side logging/debugging can still
    access rollout internals without re-running dynamics.
    """

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
        return sim_trajs
