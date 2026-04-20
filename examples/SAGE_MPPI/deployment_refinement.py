#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deployment refinement helpers for SAGE Gazebo examples.

These utilities are intentionally separate from the controller core. The clean
controller/task pipeline should remain usable with this entire module disabled.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

import numpy as np
import torch


def _reset_controller_distribution(mpc) -> None:
    controller = getattr(mpc, "controller", None)
    if controller is not None and hasattr(controller, "reset"):
        controller.reset()


class GoalHoldController:
    def __init__(
        self,
        success_threshold: float,
        enter_threshold: Optional[float] = None,
        exit_threshold: Optional[float] = None,
        enter_count: int = 5,
        exit_count: int = 6,
        velocity_threshold: float = 0.08,
    ) -> None:
        self.success_threshold = float(success_threshold)
        self.enter_threshold = float(
            self.success_threshold if enter_threshold is None else enter_threshold
        )
        self.exit_threshold = float(
            max(self.enter_threshold * 1.5, self.success_threshold * 1.5)
            if exit_threshold is None
            else exit_threshold
        )
        self.enter_count = int(enter_count)
        self.exit_count = int(exit_count)
        self.velocity_threshold = float(velocity_threshold)
        self.active = False
        self.hold_positions = None
        self._enter_streak = 0
        self._exit_streak = 0

    def reset(self) -> None:
        self.active = False
        self.hold_positions = None
        self._enter_streak = 0
        self._exit_streak = 0

    def force_activate(self, q: np.ndarray) -> None:
        self.active = True
        self.hold_positions = np.asarray(q, dtype=np.float64).copy()
        self._enter_streak = self.enter_count
        self._exit_streak = 0

    def update(self, error: float, q: np.ndarray, dq: np.ndarray):
        just_entered = False
        just_released = False

        if self.active:
            if error > self.exit_threshold:
                self._exit_streak += 1
                if self._exit_streak >= self.exit_count:
                    self.reset()
                    just_released = True
            else:
                self._exit_streak = 0
            hold_positions = None if self.hold_positions is None else self.hold_positions.copy()
            return self.active, hold_positions, just_entered, just_released

        velocity_norm = float(np.linalg.norm(dq))
        if error <= self.enter_threshold and velocity_norm <= self.velocity_threshold:
            self._enter_streak += 1
            if self._enter_streak >= self.enter_count:
                self.active = True
                self.hold_positions = np.asarray(q, dtype=np.float64).copy()
                self._exit_streak = 0
                just_entered = True
        else:
            self._enter_streak = 0

        hold_positions = None if self.hold_positions is None else self.hold_positions.copy()
        return self.active, hold_positions, just_entered, just_released


class NearGoalRefinementController:
    def __init__(
        self,
        mpc,
        controller,
        rollout_fn,
        enter_threshold: float = 0.08,
        exit_threshold: float = 0.11,
        sigma_scale: float = 0.2,
        stagnation_alpha: float = 0.0,
        goal_weight_scale: float = 1.5,
        stop_cost_scale: float = 1.0,
        stop_cost_acc_scale: float = 1.0,
        tau_p: Optional[float] = None,
        step_size_mean: Optional[float] = None,
    ) -> None:
        self.mpc = mpc
        self.controller = controller
        self.rollout_fn = rollout_fn
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.sigma_scale = float(sigma_scale)
        self.refine_stagnation_alpha = float(stagnation_alpha)
        self.goal_weight_scale = float(goal_weight_scale)
        self.stop_cost_scale = float(stop_cost_scale)
        self.stop_cost_acc_scale = float(stop_cost_acc_scale)
        self.refine_tau_p = None if tau_p is None else float(tau_p)
        self.refine_step_size_mean = None if step_size_mean is None else float(step_size_mean)
        self.active = False

        self.base_sigma_0 = float(controller.sigma_0)
        self.base_stagnation_alpha = float(controller.stagnation_alpha)
        self.base_tau_p = float(controller.tau_p)
        self.base_step_size_mean = float(controller.step_size_mean)
        self.base_goal_weight = self._get_goal_position_weight()
        self.base_stop_cost_weight = self._get_scalar_cost_weight("stop_cost")
        self.base_stop_cost_acc_weight = self._get_scalar_cost_weight("stop_cost_acc")
        self.base_retract_state = np.asarray(
            self.mpc.exp_params["cost"]["retract_state"],
            dtype=np.float64,
        ).copy()

    def _get_goal_position_weight(self) -> Optional[float]:
        goal_cost = getattr(self.rollout_fn, "goal_cost", None)
        if goal_cost is None or not hasattr(goal_cost, "weight"):
            return None
        weight = goal_cost.weight
        if isinstance(weight, torch.Tensor):
            if weight.numel() < 2:
                return None
            return float(weight.detach().reshape(-1)[1].item())
        if len(weight) < 2:
            return None
        return float(weight[1])

    def _set_goal_position_weight(self, value: float) -> None:
        goal_cost = getattr(self.rollout_fn, "goal_cost", None)
        if goal_cost is None or not hasattr(goal_cost, "weight"):
            return
        weight = goal_cost.weight
        if isinstance(weight, torch.Tensor):
            weight = weight.clone()
            weight.reshape(-1)[1] = float(value)
            goal_cost.weight = weight
            return
        weight = list(weight)
        if len(weight) >= 2:
            weight[1] = float(value)
            goal_cost.weight = weight

    def _get_scalar_cost_weight(self, attr_name: str) -> Optional[float]:
        cost_obj = getattr(self.rollout_fn, attr_name, None)
        if cost_obj is None or not hasattr(cost_obj, "weight"):
            return None
        weight = cost_obj.weight
        if isinstance(weight, torch.Tensor):
            if weight.numel() == 0:
                return None
            return float(weight.detach().reshape(-1)[0].item())
        return float(weight)

    def _set_scalar_cost_weight(self, attr_name: str, exp_key: str, value: float) -> None:
        cost_obj = getattr(self.rollout_fn, attr_name, None)
        if cost_obj is not None and hasattr(cost_obj, "weight"):
            weight = cost_obj.weight
            if isinstance(weight, torch.Tensor):
                cost_obj.weight = torch.as_tensor(float(value), **self.controller.tensor_args)
            else:
                cost_obj.weight = float(value)
        try:
            self.rollout_fn.exp_params["cost"][exp_key]["weight"] = float(value)
        except Exception:
            pass
        try:
            self.mpc.exp_params["cost"][exp_key]["weight"] = float(value)
        except Exception:
            pass

    def _apply_refine_params(self) -> None:
        self.controller.sigma_0 = self.base_sigma_0 * self.sigma_scale
        self.controller.stagnation_alpha = self.refine_stagnation_alpha
        if self.refine_tau_p is not None:
            self.controller.tau_p = self.refine_tau_p
        if self.refine_step_size_mean is not None:
            self.controller.step_size_mean = self.refine_step_size_mean
        if self.base_goal_weight is not None:
            self._set_goal_position_weight(self.base_goal_weight * self.goal_weight_scale)
        if self.base_stop_cost_weight is not None:
            self._set_scalar_cost_weight(
                "stop_cost",
                "stop_cost",
                self.base_stop_cost_weight * self.stop_cost_scale,
            )
        if self.base_stop_cost_acc_weight is not None:
            self._set_scalar_cost_weight(
                "stop_cost_acc",
                "stop_cost_acc",
                self.base_stop_cost_acc_weight * self.stop_cost_acc_scale,
            )

    def _restore_nominal_params(self) -> None:
        self.controller.sigma_0 = self.base_sigma_0
        self.controller.stagnation_alpha = self.base_stagnation_alpha
        self.controller.tau_p = self.base_tau_p
        self.controller.step_size_mean = self.base_step_size_mean
        self.mpc.update_params(retract_state=self.base_retract_state)
        if self.base_goal_weight is not None:
            self._set_goal_position_weight(self.base_goal_weight)
        if self.base_stop_cost_weight is not None:
            self._set_scalar_cost_weight("stop_cost", "stop_cost", self.base_stop_cost_weight)
        if self.base_stop_cost_acc_weight is not None:
            self._set_scalar_cost_weight(
                "stop_cost_acc",
                "stop_cost_acc",
                self.base_stop_cost_acc_weight,
            )

    def reset(self) -> None:
        self.active = False
        self._restore_nominal_params()

    def update(self, error: float, current_q: np.ndarray):
        just_entered = False
        just_exited = False

        if self.active:
            if error > self.exit_threshold:
                self.active = False
                self._restore_nominal_params()
                just_exited = True
            return self.active, just_entered, just_exited

        if error <= self.enter_threshold:
            self.active = True
            self._apply_refine_params()
            self.mpc.update_params(retract_state=np.asarray(current_q, dtype=np.float64).copy())
            just_entered = True
        return self.active, just_entered, just_exited


class CartesianGoalRefiner:
    def __init__(
        self,
        rollout_fn,
        tensor_args,
        enter_threshold: float = 0.05,
        exit_threshold: float = 0.07,
        damping: float = 0.05,
        gain: float = 0.7,
        max_joint_step: float = 0.02,
        blend_steps: int = 6,
        max_command_slew: float = 0.01,
        output_mode: str = "blend",
    ) -> None:
        self.rollout_fn = rollout_fn
        self.tensor_args = tensor_args
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.damping = float(damping)
        self.gain = float(gain)
        self.max_joint_step = float(max_joint_step)
        self.blend_steps = max(int(blend_steps), 1)
        self.max_command_slew = float(max_command_slew)
        self.output_mode = str(output_mode)
        self.active = False
        self.transition_step = 0
        self.last_command = None
        self.last_step_norm = 0.0
        self.last_gain_used = float(gain)
        self.last_blend_ratio = 0.0

        self.robot_model = rollout_fn.dynamics_model.robot_model
        self.ee_link_name = rollout_fn.exp_params["model"]["ee_link_name"]
        dyn_model = rollout_fn.dynamics_model
        self.q_lower = dyn_model.state_lower_bounds[: dyn_model.n_dofs].detach().cpu().numpy()
        self.q_upper = dyn_model.state_upper_bounds[: dyn_model.n_dofs].detach().cpu().numpy()

    def reset(self) -> None:
        self.active = False
        self.transition_step = 0
        self.last_command = None
        self.last_step_norm = 0.0
        self.last_blend_ratio = 0.0

    def update(self, error: float):
        just_entered = False
        just_exited = False
        if self.active:
            if error > self.exit_threshold:
                self.active = False
                self.transition_step = 0
                self.last_command = None
                just_exited = True
            return self.active, just_entered, just_exited
        if error <= self.enter_threshold:
            self.active = True
            self.transition_step = 0
            self.last_command = None
            just_entered = True
        return self.active, just_entered, just_exited

    def compute_command(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        goal_ee_pos_robot: np.ndarray,
    ) -> np.ndarray:
        q_t = torch.as_tensor(q, **self.tensor_args).unsqueeze(0)
        qd_t = torch.as_tensor(dq, **self.tensor_args).unsqueeze(0)
        goal_t = torch.as_tensor(goal_ee_pos_robot, **self.tensor_args).reshape(1, 3)

        ee_pos, _, lin_jac, _ = self.robot_model.compute_fk_and_jacobian(
            q_t,
            qd_t,
            link_name=self.ee_link_name,
        )
        pos_err = (goal_t - ee_pos).reshape(3)
        jac = lin_jac.reshape(3, -1)
        ident = torch.eye(3, **self.tensor_args)
        dls_step = jac.transpose(-2, -1) @ torch.linalg.solve(
            jac @ jac.transpose(-2, -1) + (self.damping ** 2) * ident,
            pos_err.unsqueeze(-1),
        )
        joint_step = self.gain * dls_step.squeeze(-1)
        joint_step = torch.clamp(joint_step, -self.max_joint_step, self.max_joint_step)
        self.last_step_norm = float(torch.norm(joint_step).detach().cpu().item())
        self.last_gain_used = float(self.gain)
        q_cmd = q_t.reshape(-1) + joint_step
        q_cmd = torch.max(
            torch.min(q_cmd, torch.as_tensor(self.q_upper, **self.tensor_args)),
            torch.as_tensor(self.q_lower, **self.tensor_args),
        )
        return q_cmd.detach().cpu().numpy()

    def smooth_command(
        self,
        desired_command: np.ndarray,
        nominal_command: Optional[np.ndarray],
        current_q: np.ndarray,
    ) -> np.ndarray:
        desired = np.asarray(desired_command, dtype=np.float64).copy()
        nominal = (
            np.asarray(current_q, dtype=np.float64).copy()
            if nominal_command is None
            else np.asarray(nominal_command, dtype=np.float64).copy()
        )

        alpha = min(float(self.transition_step + 1) / float(self.blend_steps), 1.0)
        blended = nominal + alpha * (desired - nominal)
        self.last_blend_ratio = float(alpha)

        prev = nominal if self.last_command is None else self.last_command
        if self.max_command_slew > 0.0:
            delta = np.clip(blended - prev, -self.max_command_slew, self.max_command_slew)
            blended = prev + delta

        self.last_command = blended.copy()
        self.transition_step += 1
        return blended

    def postprocess_command(
        self,
        desired_command: np.ndarray,
        nominal_command: Optional[np.ndarray],
        current_q: np.ndarray,
    ) -> np.ndarray:
        if self.output_mode == "override":
            desired = np.asarray(desired_command, dtype=np.float64).copy()
            prev = (
                np.asarray(current_q, dtype=np.float64).copy()
                if self.last_command is None
                else self.last_command
            )
            if self.max_command_slew > 0.0:
                delta = np.clip(desired - prev, -self.max_command_slew, self.max_command_slew)
                desired = prev + delta
            self.last_blend_ratio = 1.0
            self.last_command = desired.copy()
            self.transition_step += 1
            return desired
        return self.smooth_command(
            desired_command=desired_command,
            nominal_command=nominal_command,
            current_q=current_q,
        )


class StallMonitor:
    def __init__(
        self,
        history_len: int = 50,
        min_runtime: float = 8.0,
        error_threshold: float = 0.12,
        motion_threshold: float = 0.01,
        velocity_threshold: float = 0.08,
        cooldown: float = 8.0,
    ) -> None:
        self.history = deque(maxlen=history_len)
        self.min_runtime = float(min_runtime)
        self.error_threshold = float(error_threshold)
        self.motion_threshold = float(motion_threshold)
        self.velocity_threshold = float(velocity_threshold)
        self.cooldown = float(cooldown)
        self.last_recovery_t = -1e9

    def reset(self) -> None:
        self.history.clear()
        self.last_recovery_t = -1e9

    def _history_motion(self) -> float:
        if len(self.history) < 2:
            return np.inf
        points = np.asarray(self.history, dtype=np.float64)
        ref = points[0]
        disp = np.linalg.norm(points - ref.reshape(1, 3), axis=1)
        return float(np.max(disp))

    def update(self, ee_pos_world: np.ndarray) -> None:
        self.history.append(np.asarray(ee_pos_world, dtype=np.float64).copy())

    def should_recover(
        self,
        t_step: float,
        ee_pos_world: np.ndarray,
        goal_world: np.ndarray,
        joint_velocity: np.ndarray,
    ) -> bool:
        self.update(ee_pos_world)
        if t_step < self.min_runtime:
            return False
        if len(self.history) < self.history.maxlen:
            return False
        if (t_step - self.last_recovery_t) < self.cooldown:
            return False

        ee_error = float(np.linalg.norm(ee_pos_world - goal_world))
        history_motion = self._history_motion()
        velocity_norm = float(np.linalg.norm(joint_velocity))

        if ee_error <= self.error_threshold:
            return False
        if history_motion >= self.motion_threshold:
            return False
        if velocity_norm >= self.velocity_threshold:
            return False

        self.last_recovery_t = float(t_step)
        return True


class DeploymentRefinementStack:
    """
    Runtime wrapper for deployment-specific heuristics.

    This stack is intentionally separate from the controller core. It can be
    disabled entirely without changing the controller/rollout/task path.
    """

    def __init__(
        self,
        mpc,
        tensor_args,
        refinement_cfg: Optional[dict],
        reset_timing_fn: Callable[[object, float, float], None],
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.mpc = mpc
        self.controller = mpc.controller
        self.rollout_fn = self.controller.rollout_fn
        self.tensor_args = tensor_args
        self.refinement_cfg = dict(refinement_cfg or {})
        self.reset_timing_fn = reset_timing_fn
        self.log_fn = log_fn or (lambda msg: None)
        self.enabled = bool(self.refinement_cfg.get("enabled", False))

        self.goal_hold = None
        self.near_goal = None
        self.local_refinement = None
        self.cartesian = None
        self.stall_monitor = None
        self.cart_hold_threshold = 0.01
        self.cart_hold_count = 10
        self.cart_hold_streak = 0
        self.goal_change_boost_cfg = dict(self.refinement_cfg.get("goal_change_boost", {}))
        self._boost_active = False
        self._boost_restore_error_threshold = None
        self._base_sigma_0 = float(self.controller.sigma_0)
        self._base_step_size_mean = float(self.controller.step_size_mean)
        self.latest_local_refinement_stats = {
            "local_refinement_active": False,
            "local_refinement_mode": "off",
            "local_refinement_step_norm": 0.0,
            "local_refinement_gain_used": 0.0,
            "local_refinement_blend_ratio": 0.0,
        }

        if not self.enabled:
            return

        hold_cfg = dict(self.refinement_cfg.get("hold", {}))
        near_cfg = dict(self.refinement_cfg.get("near_goal_refinement", {}))
        local_cfg = dict(self.refinement_cfg.get("local_refinement", {}))
        if not local_cfg:
            local_cfg = dict(self.refinement_cfg.get("cartesian_refinement", {}))
        stall_cfg = dict(self.refinement_cfg.get("stall_recovery", {}))

        success_threshold = float(
            hold_cfg.get(
                "success_threshold",
                self.mpc.exp_params.get("task_metrics", {}).get("success_threshold", 0.05),
            )
        )
        if hold_cfg.get("enabled", False):
            self.goal_hold = GoalHoldController(
                success_threshold=success_threshold,
                enter_threshold=hold_cfg.get("enter_threshold"),
                exit_threshold=hold_cfg.get("exit_threshold"),
                enter_count=hold_cfg.get("enter_count", 5),
                exit_count=hold_cfg.get("exit_count", 6),
                velocity_threshold=hold_cfg.get("velocity_threshold", 0.08),
            )
            self.cart_hold_threshold = float(hold_cfg.get("cart_hold_threshold", 0.01))
            self.cart_hold_count = int(hold_cfg.get("cart_hold_count", 10))

        if near_cfg.get("enabled", False):
            self.near_goal = NearGoalRefinementController(
                mpc=mpc,
                controller=self.controller,
                rollout_fn=self.rollout_fn,
                enter_threshold=near_cfg.get("enter_threshold", 0.08),
                exit_threshold=near_cfg.get("exit_threshold", 0.11),
                sigma_scale=near_cfg.get("sigma_scale", 0.2),
                stagnation_alpha=near_cfg.get("stagnation_alpha", 0.0),
                goal_weight_scale=near_cfg.get("goal_weight_scale", 1.5),
                stop_cost_scale=near_cfg.get("stop_cost_scale", 1.0),
                stop_cost_acc_scale=near_cfg.get("stop_cost_acc_scale", 1.0),
                tau_p=near_cfg.get("tau_p"),
                step_size_mean=near_cfg.get("step_size_mean"),
            )

        if local_cfg.get("enabled", False):
            self.local_refinement = CartesianGoalRefiner(
                rollout_fn=self.rollout_fn,
                tensor_args=tensor_args,
                enter_threshold=local_cfg.get("dist_threshold", local_cfg.get("enter_threshold", success_threshold)),
                exit_threshold=local_cfg.get("exit_threshold", 0.07),
                damping=local_cfg.get("damping", 0.05),
                gain=local_cfg.get("gain", 0.7),
                max_joint_step=local_cfg.get("max_step", local_cfg.get("max_joint_step", 0.02)),
                blend_steps=local_cfg.get("blend_steps", 6),
                max_command_slew=local_cfg.get("max_command_slew", 0.01),
                output_mode=local_cfg.get("output_mode", "blend"),
            )
        self.cartesian = self.local_refinement

        if stall_cfg.get("enabled", False):
            self.stall_monitor = StallMonitor(
                history_len=stall_cfg.get("history_len", 50),
                min_runtime=stall_cfg.get("min_runtime", 8.0),
                error_threshold=stall_cfg.get("error_threshold", 0.12),
                motion_threshold=stall_cfg.get("motion_threshold", 0.01),
                velocity_threshold=stall_cfg.get("velocity_threshold", 0.08),
                cooldown=stall_cfg.get("cooldown", 8.0),
            )

    def _apply_exploration_boost(
        self,
        sigma_scale: float,
        step_size_mean: Optional[float],
        restore_error_threshold: Optional[float],
    ) -> None:
        sigma_scale = float(max(sigma_scale, 1.0))
        self.controller.sigma_0 = self._base_sigma_0 * sigma_scale
        if step_size_mean is not None:
            self.controller.step_size_mean = float(step_size_mean)
        self._boost_active = True
        self._boost_restore_error_threshold = (
            None if restore_error_threshold is None else float(restore_error_threshold)
        )

    def _restore_exploration_boost(self) -> None:
        if not self._boost_active:
            return
        self.controller.sigma_0 = self._base_sigma_0
        self.controller.step_size_mean = self._base_step_size_mean
        self._boost_active = False
        self._boost_restore_error_threshold = None

    def _sync_reset(self, t_step: float, control_dt: float, message: Optional[str] = None) -> None:
        _reset_controller_distribution(self.mpc)
        self.reset_timing_fn(self.mpc.control_process, t_step, control_dt)
        if message:
            self.log_fn(message)

    def reset_all(self) -> None:
        if self.goal_hold is not None:
            self.goal_hold.reset()
        if self.near_goal is not None:
            self.near_goal.reset()
        if self.cartesian is not None:
            self.cartesian.reset()
        if self.stall_monitor is not None:
            self.stall_monitor.reset()
        self.cart_hold_streak = 0
        self._restore_exploration_boost()
        self.latest_local_refinement_stats = {
            "local_refinement_active": False,
            "local_refinement_mode": "off",
            "local_refinement_step_norm": 0.0,
            "local_refinement_gain_used": 0.0,
            "local_refinement_blend_ratio": 0.0,
        }

    def on_goal_changed(self, t_step: float, control_dt: float) -> None:
        if not self.enabled:
            return
        self.reset_all()
        if self.goal_change_boost_cfg.get("enabled", False):
            self._apply_exploration_boost(
                sigma_scale=self.goal_change_boost_cfg.get("sigma_scale", 9.0),
                step_size_mean=self.goal_change_boost_cfg.get("step_size_mean", 0.45),
                restore_error_threshold=self.goal_change_boost_cfg.get(
                    "restore_error_threshold", 0.18
                ),
            )
        self._sync_reset(t_step, control_dt, "[CleanRefine] goal changed, reset refinement stack")

    def update_modes(
        self,
        error: float,
        q: np.ndarray,
        dq: np.ndarray,
        t_step: float,
        control_dt: float,
    ) -> None:
        if not self.enabled:
            return

        if (
            self._boost_active
            and self._boost_restore_error_threshold is not None
            and error <= self._boost_restore_error_threshold
        ):
            self._restore_exploration_boost()
            self._sync_reset(
                t_step,
                control_dt,
                f"[CleanRefine] restore nominal exploration @ {error:.4f}",
            )

        if self.near_goal is not None:
            _, entered, exited = self.near_goal.update(error, q)
            if entered:
                self._sync_reset(t_step, control_dt, f"[CleanRefine] enter near-goal refinement @ {error:.4f}")
            elif exited:
                self._sync_reset(t_step, control_dt, f"[CleanRefine] exit near-goal refinement @ {error:.4f}")

        if self.cartesian is not None:
            _, entered, exited = self.cartesian.update(error)
            if entered:
                self._sync_reset(t_step, control_dt, f"[CleanRefine] enter local refinement @ {error:.4f}")
            elif exited:
                self.cart_hold_streak = 0
                self._sync_reset(t_step, control_dt, f"[CleanRefine] exit local refinement @ {error:.4f}")

        if self.cartesian is not None and self.goal_hold is not None:
            if self.cartesian.active and (not self.goal_hold.active):
                if error <= self.cart_hold_threshold:
                    self.cart_hold_streak += 1
                    if self.cart_hold_streak >= self.cart_hold_count:
                        self.goal_hold.force_activate(q)
                        self.cartesian.reset()
                        self._sync_reset(
                            t_step,
                            control_dt,
                            f"[CleanRefine] cartesian convergence latched hold @ {error:.4f}",
                        )
                else:
                    self.cart_hold_streak = 0
            else:
                self.cart_hold_streak = 0

    def maybe_get_override_command(
        self,
        error: float,
        q: np.ndarray,
        dq: np.ndarray,
        goal_ee_pos_robot: np.ndarray,
        t_step: float,
        control_dt: float,
        nominal_position_cmd: Optional[np.ndarray] = None,
    ):
        if not self.enabled:
            return None

        if self.goal_hold is not None:
            hold_active, hold_q, entered, released = self.goal_hold.update(error=error, q=q, dq=dq)
            if entered:
                if self.cartesian is not None:
                    self.cartesian.reset()
                self._sync_reset(t_step, control_dt, f"[CleanRefine] enter hold @ {error:.4f}")
            elif released:
                self._sync_reset(t_step, control_dt, f"[CleanRefine] exit hold @ {error:.4f}")

            if hold_active and hold_q is not None:
                return {"position": hold_q}

        if self.cartesian is not None and self.cartesian.active:
            try:
                desired_command = self.cartesian.compute_command(
                    q=q,
                    dq=dq,
                    goal_ee_pos_robot=goal_ee_pos_robot,
                )
                refined_command = self.cartesian.postprocess_command(
                    desired_command=desired_command,
                    nominal_command=nominal_position_cmd,
                    current_q=q,
                )
                self.latest_local_refinement_stats = {
                    "local_refinement_active": True,
                    "local_refinement_mode": str(self.cartesian.output_mode),
                    "local_refinement_step_norm": float(self.cartesian.last_step_norm),
                    "local_refinement_gain_used": float(self.cartesian.last_gain_used),
                    "local_refinement_blend_ratio": float(self.cartesian.last_blend_ratio),
                }
                return {
                    "position": refined_command
                }
            except Exception as exc:
                self.log_fn(f"[CleanRefine] cartesian refinement failed, fallback to MPC: {exc}")
        self.latest_local_refinement_stats = {
            "local_refinement_active": bool(self.cartesian is not None and self.cartesian.active),
            "local_refinement_mode": (
                str(self.cartesian.output_mode)
                if self.cartesian is not None and self.cartesian.active
                else "off"
            ),
            "local_refinement_step_norm": 0.0,
            "local_refinement_gain_used": (
                float(self.cartesian.gain)
                if self.cartesian is not None and self.cartesian.active
                else 0.0
            ),
            "local_refinement_blend_ratio": 0.0,
        }

        return None

    def maybe_trigger_recovery(
        self,
        t_step: float,
        ee_pos_world: np.ndarray,
        goal_world: np.ndarray,
        joint_velocity: np.ndarray,
        control_dt: float,
    ) -> bool:
        if not self.enabled or self.stall_monitor is None:
            return False

        if self.stall_monitor.should_recover(
            t_step=t_step,
            ee_pos_world=ee_pos_world,
            goal_world=goal_world,
            joint_velocity=joint_velocity,
        ):
            stall_cfg = dict(self.refinement_cfg.get("stall_recovery", {}))
            self._apply_exploration_boost(
                sigma_scale=stall_cfg.get("sigma_scale", 16.0),
                step_size_mean=stall_cfg.get("step_size_mean", 0.35),
                restore_error_threshold=stall_cfg.get("restore_error_threshold", 0.18),
            )
            self._sync_reset(t_step, control_dt, "[CleanRefine] stall recovery triggered")
            return True
        return False
