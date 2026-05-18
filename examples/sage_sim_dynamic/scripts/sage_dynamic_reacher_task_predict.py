#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task wrapper for SAGE predictive dynamic-ball reaching demo."""

from __future__ import annotations

import torch
import numpy as np

from storm_kit.mpc.task.sage_arm_task_impl import SageArmTaskV3
from examples.sage_sim_dynamic.scripts.sage_dynamic_arm_reacher_predict import SageDynamicArmReacherPredict


class SageDynamicReacherTaskPredict(SageArmTaskV3):
    def get_rollout_fn(self, **kwargs):
        return SageDynamicArmReacherPredict(**kwargs)

    def set_dynamic_sphere_state_world(self, sphere_name: str, position_world, vel_y) -> None:
        self.controller.rollout_fn.set_dynamic_sphere_state_world(sphere_name, position_world, vel_y)

    def get_dynamic_ball_metrics(self):
        return self.controller.rollout_fn.get_dynamic_ball_metrics()

    def get_selected_action_sequence(self):
        controller = self.controller
        latest_stats = {}
        if hasattr(controller, 'get_latest_stats'):
            latest_stats = controller.get_latest_stats() or {}
        output_mode = str(latest_stats.get('output_mode_used', 'mean'))
        if output_mode == 'best' and getattr(controller, 'best_traj', None) is not None:
            return controller.best_traj.detach().cpu().numpy().copy()
        if getattr(controller, 'mean_action', None) is not None:
            return controller.mean_action.detach().cpu().numpy().copy()
        return None

    def evaluate_action_sequence_dynamic_margin(self, start_state, action_seq, t_step=0.0, pred_mpc_dt=0.0):
        rollout_fn = self.controller.rollout_fn
        dyn_model = rollout_fn.dynamics_model
        robot_model = dyn_model.robot_model
        ee_link_name = rollout_fn.exp_params['model']['ee_link_name']

        start_state = np.asarray(start_state, dtype=np.float64).reshape(-1)
        n_dof = dyn_model.n_dofs
        curr_state = start_state[: 3 * n_dof].copy()
        action_seq = np.asarray(action_seq, dtype=np.float64)
        if action_seq.ndim != 2:
            raise ValueError(f'action_seq must be [H,d], got shape={action_seq.shape}')

        dt_seq = dyn_model._dt_h.detach().cpu().numpy().reshape(-1)
        horizon = min(action_seq.shape[0], dt_seq.shape[0])
        link_pos_seq = []
        link_rot_seq = []

        for h in range(horizon):
            act = torch.as_tensor(action_seq[h], **self.tensor_args)
            state_t = torch.as_tensor(curr_state, **self.tensor_args).clone()
            next_state = dyn_model.get_next_state(state_t, act, float(dt_seq[h]))
            curr_state = next_state.detach().cpu().numpy().copy()

            q_t = next_state[:n_dof].unsqueeze(0)
            dq_t = next_state[n_dof:2 * n_dof].unsqueeze(0)
            robot_model.compute_fk_and_jacobian(q_t, dq_t, ee_link_name)
            link_pos_step = []
            link_rot_step = []
            for link_name in dyn_model.link_names:
                pos, rot = robot_model.get_link_pose(link_name)
                link_pos_step.append(pos[0])
                link_rot_step.append(rot[0])
            link_pos_seq.append(torch.stack(link_pos_step, dim=0))
            link_rot_seq.append(torch.stack(link_rot_step, dim=0))

        link_pos_seq = torch.stack(link_pos_seq, dim=0).unsqueeze(0)
        link_rot_seq = torch.stack(link_rot_seq, dim=0).unsqueeze(0)
        return rollout_fn.primitive_collision_cost.evaluate_link_pose_sequence(
            link_pos_seq,
            link_rot_seq,
        )

    def evaluate_current_state_dynamic_margin(self, q, dq):
        rollout_fn = self.controller.rollout_fn
        q_t = torch.as_tensor(np.asarray(q, dtype=np.float64), **self.tensor_args).unsqueeze(0)
        dq_t = torch.as_tensor(np.asarray(dq, dtype=np.float64), **self.tensor_args).unsqueeze(0)
        robot_model = rollout_fn.dynamics_model.robot_model
        robot_model.compute_fk_and_jacobian(q_t, dq_t, rollout_fn.exp_params['model']['ee_link_name'])
        link_pos = []
        link_rot = []
        for link_name in rollout_fn.dynamics_model.link_names:
            pos, rot = robot_model.get_link_pose(link_name)
            link_pos.append(pos[0])
            link_rot.append(rot[0])
        link_pos = torch.stack(link_pos, dim=0).unsqueeze(0)
        link_rot = torch.stack(link_rot, dim=0).unsqueeze(0)
        return rollout_fn.primitive_collision_cost.evaluate_current_link_poses(link_pos, link_rot)

    def set_position_only_goal_mode(self) -> None:
        rollout_fn = getattr(self.controller, 'rollout_fn', None)
        goal_cost = getattr(rollout_fn, 'goal_cost', None)
        if goal_cost is None:
            return
        weight = goal_cost.weight
        if isinstance(weight, torch.Tensor):
            flat = weight.reshape(-1).clone()
            if flat.numel() >= 2:
                flat[0] = 0.0
                goal_cost.weight = flat.reshape(weight.shape)
            return
        if isinstance(weight, (list, tuple)) and len(weight) >= 2:
            goal_cost.weight = [0.0, float(weight[1])]
