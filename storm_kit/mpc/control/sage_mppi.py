#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#

"""
Standalone clean SAGE controller.

This file is the canonical public SAGE controller entry. It intentionally does
not inherit from any legacy controller implementation.

Design goals:
- keep the same external task/control-process interface used by STORM
- update proposal mean/shape from all samples using MPPI weights
- expose the SAGE stage-scale and stagnation mechanisms as explicit switches
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.autograd.profiler as profiler

from .control_utils import cost_to_go, gaussian_entropy, scale_ctrl
from .sample_libs import HaltonSampleLib, MultipleSampleLib, RandomSampleLib, StompSampleLib


class SAGE_MPPI:
    """
    Independent clean SAGE controller.

    This class is the canonical SAGE controller implementation.
    """

    def __init__(
        self,
        d_action,
        horizon,
        init_cov,
        init_mean,
        base_action,
        beta,
        num_particles,
        step_size_mean,
        step_size_cov,
        alpha,
        gamma,
        kappa,
        n_iters,
        action_lows,
        action_highs,
        null_act_frac=0.0,
        rollout_fn=None,
        sample_mode="mean",
        hotstart=True,
        squash_fn="clamp",
        update_cov=False,
        cov_type="diag_AxA",
        seed=0,
        sample_params=None,
        tensor_args=None,
        visual_traj="state_seq",
        sigma_0=None,
        sigma_1=0.0,
        sigma_2=0.0,
        tau_p=1.0e-4,
        stagnation_alpha=0.0,
        execute_best=False,
        enable_stage_scale=True,
        enable_anisotropic_shape_update=True,
        enable_stagnation_amplification=True,
        enable_runtime_stats=False,
        enable_shape_collapse_guard=True,
        shape_update_min_normalized_entropy=0.04,
        shape_update_max_fallback_fraction=0.85,
        shape_update_last_iter_only=True,
        shape_update_random_only=True,
        shape_update_ema=0.6,
        shape_temperature_multiplier=1.5,
        near_goal_dist_threshold=0.2,
        near_goal_stagnation_disable_threshold=None,
        near_goal_scale_threshold=None,
        near_goal_scale_min_factor=0.5,
        near_goal_scale_floor=8.0e-4,
        near_goal_update_shape_each_iter=True,
        near_goal_execute_best=True,
        near_goal_shape_update_min_normalized_entropy=0.005,
        near_goal_shape_temperature_multiplier=3.0,
        near_goal_preserve_previous_shape=True,
        near_goal_allow_low_entropy_shape_update=False,
        near_goal_previous_shape_identity_mix=0.15,
    ):
        if sample_params is None:
            sample_params = {
                "type": "halton",
                "fixed_samples": True,
                "seed": 0,
                "filter_coeffs": None,
            }
        if tensor_args is None:
            tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}

        self.tensor_args = tensor_args
        self.device = tensor_args["device"]
        self.dtype = tensor_args["dtype"]

        self.d_action = int(d_action)
        self.horizon = int(horizon)
        self.init_cov = float(init_cov)
        self.init_mean = init_mean.clone().to(**self.tensor_args)
        self.base_action = base_action
        self.num_particles = int(num_particles)
        self.step_size_mean = float(step_size_mean)
        self.step_size_cov = float(step_size_cov)  # kept for interface parity
        self.alpha = alpha  # kept for interface parity
        self.gamma = float(gamma)
        self.kappa = float(kappa)  # kept for hotstart parity
        self.n_iters = int(n_iters)
        self.sample_mode = sample_mode
        self.hotstart = bool(hotstart)
        self.squash_fn = squash_fn
        self.update_cov = bool(update_cov)  # kept for interface parity
        self.cov_type = cov_type  # kept for interface parity
        self.seed_val = int(seed)
        self.sample_params = sample_params
        self.visual_traj = visual_traj
        self.execute_best = bool(execute_best)

        # SAGE core parameters.
        self.lambda_ = float(beta)
        self.sigma_0 = float(self.init_cov if sigma_0 is None else sigma_0)
        self.sigma_1 = float(sigma_1)
        self.sigma_2 = float(sigma_2)
        self.tau_p = float(tau_p)
        self.stagnation_alpha = float(stagnation_alpha)
        self.enable_stage_scale = bool(enable_stage_scale)
        self.enable_anisotropic_shape_update = bool(enable_anisotropic_shape_update)
        self.enable_stagnation_amplification = bool(enable_stagnation_amplification)
        self.enable_runtime_stats = bool(enable_runtime_stats)
        self.enable_shape_collapse_guard = bool(enable_shape_collapse_guard)
        self.shape_update_min_normalized_entropy = float(shape_update_min_normalized_entropy)
        self.shape_update_max_fallback_fraction = float(shape_update_max_fallback_fraction)
        self.shape_update_last_iter_only = bool(shape_update_last_iter_only)
        self.shape_update_random_only = bool(shape_update_random_only)
        self.shape_update_ema = float(shape_update_ema)
        self.shape_temperature_multiplier = float(shape_temperature_multiplier)
        self.near_goal_dist_threshold = float(near_goal_dist_threshold)
        self.near_goal_stagnation_disable_threshold = float(
            near_goal_dist_threshold
            if near_goal_stagnation_disable_threshold is None
            else near_goal_stagnation_disable_threshold
        )
        self.near_goal_scale_threshold = float(
            near_goal_dist_threshold if near_goal_scale_threshold is None else near_goal_scale_threshold
        )
        self.near_goal_scale_min_factor = float(near_goal_scale_min_factor)
        self.near_goal_scale_floor = float(near_goal_scale_floor)
        self.near_goal_update_shape_each_iter = bool(near_goal_update_shape_each_iter)
        self.near_goal_execute_best = bool(near_goal_execute_best)
        self.near_goal_shape_update_min_normalized_entropy = float(
            near_goal_shape_update_min_normalized_entropy
        )
        self.near_goal_shape_temperature_multiplier = float(
            near_goal_shape_temperature_multiplier
        )
        self.near_goal_preserve_previous_shape = bool(near_goal_preserve_previous_shape)
        self.near_goal_allow_low_entropy_shape_update = bool(
            near_goal_allow_low_entropy_shape_update
        )
        self.near_goal_previous_shape_identity_mix = float(
            near_goal_previous_shape_identity_mix
        )

        if self.sigma_0 <= 0.0:
            raise ValueError("sigma_0 must be positive for SAGE proposal scaling")
        if self.lambda_ <= 0.0:
            raise ValueError("beta/lambda must be positive")
        if not (0.0 < self.shape_update_ema <= 1.0):
            raise ValueError("shape_update_ema must be in (0, 1]")
        if self.shape_temperature_multiplier <= 0.0:
            raise ValueError("shape_temperature_multiplier must be positive")
        if not (0.0 < self.near_goal_scale_min_factor <= 1.0):
            raise ValueError("near_goal_scale_min_factor must be in (0, 1]")
        if self.near_goal_scale_floor < 0.0:
            raise ValueError("near_goal_scale_floor must be non-negative")
        if self.near_goal_shape_temperature_multiplier <= 0.0:
            raise ValueError("near_goal_shape_temperature_multiplier must be positive")
        if not (0.0 <= self.near_goal_previous_shape_identity_mix <= 1.0):
            raise ValueError("near_goal_previous_shape_identity_mix must be in [0, 1]")

        self.action_lows = action_lows.to(**self.tensor_args)
        self.action_highs = action_highs.to(**self.tensor_args)
        self.rollout_fn = rollout_fn
        self._rollout_fn = rollout_fn

        self.gamma_seq = torch.cumprod(
            torch.tensor([1.0] + [self.gamma] * (self.horizon - 1), **self.tensor_args),
            dim=0,
        ).reshape(1, self.horizon)

        self.I = torch.eye(self.d_action, **self.tensor_args)
        self.Z_seq = torch.zeros(1, self.horizon, self.d_action, **self.tensor_args)

        self.null_act_frac = float(null_act_frac)
        self.num_null_particles = round(int(self.null_act_frac * self.num_particles))
        self.num_neg_particles = round(int(self.null_act_frac * self.num_particles)) - self.num_null_particles
        self.num_nonzero_particles = (
            self.num_particles - self.num_null_particles - self.num_neg_particles
        )
        self.sample_shape = torch.Size([max(self.num_nonzero_particles - 2, 0)])

        self._build_sample_lib()

        self.trace_tol = 1.0e-8
        self.cholesky_jitter = (1.0e-8, 1.0e-6, 1.0e-4)

        self.num_steps = 0
        self.trajectories = None
        self.best_idx = None
        self.best_traj = None
        self.top_values = None
        self.top_idx = None
        self.top_trajs = None
        self.total_costs = None

        self.prev_goal_dist = None
        self._goal_signature = None
        self._last_goal_progress = 0.0
        self._last_goal_dist = None
        self._last_stage_scale = None
        self._last_scale_tril = None
        self._shape_tril = None
        self._last_weight_entropy = 0.0
        self._last_full_weight_entropy = 0.0
        self._last_shape_weight_entropy = 0.0
        self._last_full_normalized_entropy = 0.0
        self._last_shape_normalized_entropy = 0.0
        self._last_shape_entropy_used_for_skip = "none"
        self._last_shape_temperature_used = 1.0
        self._last_shape_weight_entropy_after_flatten = 0.0
        self._last_covariance_trace_mean = 0.0
        self._last_shape_condition_number = 1.0
        self._last_covariance_fallback_count = 0
        self._last_proposal_scale_min = float(self.sigma_0)
        self._last_proposal_scale_max = float(self.sigma_0)
        self._last_shape_update_skipped = False
        self._last_shape_skip_reason = ""
        self._last_success = None
        self._last_failure = None
        self._last_near_goal_active = False
        self._last_near_goal_scale_factor = 1.0
        self._last_stagnation_amplification_applied = False
        self._last_shape_update_sample_count = 0
        self._last_output_mode_used = "mean"
        self._last_near_goal_used_previous_shape = False
        self._last_near_goal_shape_condition = 1.0
        self._last_near_goal_proposal_scale = float(self.sigma_0)
        self._last_near_goal_scale_floor_active = False
        self._last_near_goal_scale_after_floor = float(self.sigma_0)
        self.latest_stats = {}

        self.reset_distribution()

    def _build_sample_lib(self):
        sample_type = self.sample_params["type"]
        if sample_type == "stomp":
            self.sample_lib = StompSampleLib(
                self.horizon,
                self.d_action,
                tensor_args=self.tensor_args,
                **self.sample_params,
            )
        elif sample_type == "halton":
            self.sample_lib = HaltonSampleLib(
                self.horizon,
                self.d_action,
                tensor_args=self.tensor_args,
                **self.sample_params,
            )
        elif sample_type == "random":
            self.sample_lib = RandomSampleLib(
                self.horizon,
                self.d_action,
                tensor_args=self.tensor_args,
                **self.sample_params,
            )
        elif sample_type == "multiple":
            self.sample_lib = MultipleSampleLib(
                self.horizon,
                self.d_action,
                tensor_args=self.tensor_args,
                **self.sample_params,
            )
        else:
            raise ValueError(f"Unsupported sample library type: {sample_type}")

    def reset_mean(self):
        self.mean_action = self.init_mean.clone()
        self.best_traj = self.mean_action.clone()

    def reset_covariance(self):
        self.shape_matrices = self.I.unsqueeze(0).repeat(self.horizon, 1, 1).clone()
        self._shape_tril = self.I.unsqueeze(0).repeat(self.horizon, 1, 1).clone()
        self._last_scale_tril = torch.sqrt(
            torch.full((self.horizon,), self.sigma_0, **self.tensor_args)
        ).view(self.horizon, 1, 1) * self.I.unsqueeze(0)

    def reset_distribution(self):
        self.reset_mean()
        self.reset_covariance()
        if self.num_null_particles > 0:
            self.null_act_seqs = torch.zeros(
                self.num_null_particles,
                self.horizon,
                self.d_action,
                **self.tensor_args,
            )

    def reset(self):
        self.num_steps = 0
        self.reset_distribution()
        self.prev_goal_dist = None
        self._goal_signature = None
        self._last_goal_progress = 0.0
        self._last_goal_dist = None
        self._last_stage_scale = None
        self._last_scale_tril = None
        self._last_weight_entropy = 0.0
        self._last_full_weight_entropy = 0.0
        self._last_shape_weight_entropy = 0.0
        self._last_full_normalized_entropy = 0.0
        self._last_shape_normalized_entropy = 0.0
        self._last_shape_entropy_used_for_skip = "none"
        self._last_shape_temperature_used = 1.0
        self._last_shape_weight_entropy_after_flatten = 0.0
        self._last_covariance_trace_mean = 0.0
        self._last_shape_condition_number = 1.0
        self._last_covariance_fallback_count = 0
        self._last_proposal_scale_min = float(self.sigma_0)
        self._last_proposal_scale_max = float(self.sigma_0)
        self._last_shape_update_skipped = False
        self._last_shape_skip_reason = ""
        self._last_success = None
        self._last_failure = None
        self._last_near_goal_active = False
        self._last_near_goal_scale_factor = 1.0
        self._last_stagnation_amplification_applied = False
        self._last_shape_update_sample_count = 0
        self._last_output_mode_used = "mean"
        self._last_near_goal_used_previous_shape = False
        self._last_near_goal_shape_condition = 1.0
        self._last_near_goal_proposal_scale = float(self.sigma_0)
        self._last_near_goal_scale_floor_active = False
        self._last_near_goal_scale_after_floor = float(self.sigma_0)
        self.latest_stats = {}

    def _to_python_float(self, value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            value = value.detach().reshape(-1)[0].item()
        return float(value)

    def _to_python_bool(self, value) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            value = value.detach().reshape(-1)[0].item()
        return bool(value)

    def _get_success_threshold(self) -> Optional[float]:
        threshold_attr_names = (
            "success_threshold",
            "goal_success_threshold",
            "goal_dist_threshold",
            "reach_goal_threshold",
            "goal_threshold",
            "hinge_val",
        )
        owners = (self.rollout_fn, getattr(self.rollout_fn, "goal_cost", None))
        for owner in owners:
            if owner is None:
                continue
            for attr_name in threshold_attr_names:
                if hasattr(owner, attr_name):
                    threshold = self._to_python_float(getattr(owner, attr_name))
                    if threshold is not None:
                        return threshold
        return None

    def _infer_task_outcome(
        self,
        trajectories: Dict[str, torch.Tensor],
        goal_dist: Optional[float],
    ) -> Tuple[Optional[bool], Optional[bool]]:
        direct_fields = (("success", "failure"), ("task_success", "task_failure"))
        for success_key, failure_key in direct_fields:
            if success_key in trajectories or failure_key in trajectories:
                success = self._to_python_bool(trajectories.get(success_key))
                failure = self._to_python_bool(trajectories.get(failure_key))
                if success is None and failure is not None:
                    success = not failure
                if failure is None and success is not None:
                    failure = not success
                if success is not None or failure is not None:
                    return success, failure

        success_threshold = self._get_success_threshold()
        if goal_dist is not None and success_threshold is not None:
            success = bool(goal_dist <= success_threshold)
            return success, (not success)

        return None, None

    def get_latest_stats(self) -> Dict[str, object]:
        return dict(self.latest_stats)

    def _goal_signature_tensor(self) -> Optional[torch.Tensor]:
        goal_parts = []
        if hasattr(self.rollout_fn, "goal_ee_pos") and self.rollout_fn.goal_ee_pos is not None:
            goal_parts.append(self.rollout_fn.goal_ee_pos.reshape(-1))
        if hasattr(self.rollout_fn, "goal_ee_quat") and self.rollout_fn.goal_ee_quat is not None:
            goal_parts.append(self.rollout_fn.goal_ee_quat.reshape(-1))
        elif hasattr(self.rollout_fn, "goal_ee_rot") and self.rollout_fn.goal_ee_rot is not None:
            goal_parts.append(self.rollout_fn.goal_ee_rot.reshape(-1))
        if hasattr(self.rollout_fn, "goal_state") and self.rollout_fn.goal_state is not None:
            goal_parts.append(self.rollout_fn.goal_state.reshape(-1))
        if not goal_parts:
            return None
        return torch.cat(goal_parts).detach().to(device="cpu", dtype=torch.float32)

    def _refresh_goal_cache_if_needed(self):
        current_sig = self._goal_signature_tensor()
        if current_sig is None:
            return
        if self._goal_signature is None:
            self._goal_signature = current_sig
            self.prev_goal_dist = None
            return
        if current_sig.shape != self._goal_signature.shape or not torch.allclose(
            current_sig, self._goal_signature
        ):
            self._goal_signature = current_sig
            self.prev_goal_dist = None

    def _compute_goal_progress(self, state: torch.Tensor) -> Tuple[float, Optional[float]]:
        self._refresh_goal_cache_if_needed()

        current_goal_dist = None
        if (
            hasattr(self.rollout_fn, "get_ee_pose")
            and hasattr(self.rollout_fn, "goal_ee_pos")
            and self.rollout_fn.goal_ee_pos is not None
        ):
            ee_state = self.rollout_fn.get_ee_pose(state[:, : self.rollout_fn.dynamics_model.d_state])
            ee_pos = ee_state["ee_pos_seq"]
            goal_pos = self.rollout_fn.goal_ee_pos.to(**self.tensor_args)
            current_goal_dist = torch.norm(ee_pos - goal_pos, dim=-1).mean().item()
        elif hasattr(self.rollout_fn, "goal_state") and self.rollout_fn.goal_state is not None:
            n_dofs = self.rollout_fn.dynamics_model.n_dofs
            q = state[:, :n_dofs]
            goal_q = self.rollout_fn.goal_state[:, :n_dofs].to(**self.tensor_args)
            current_goal_dist = torch.norm(q - goal_q, dim=-1).mean().item()

        if current_goal_dist is None:
            self._last_goal_progress = 0.0
            self._last_goal_dist = None
            return 0.0, None

        if self.prev_goal_dist is None:
            delta_goal = 0.0
        else:
            delta_goal = float(self.prev_goal_dist - current_goal_dist)

        self.prev_goal_dist = current_goal_dist
        self._last_goal_progress = delta_goal
        self._last_goal_dist = current_goal_dist
        return delta_goal, current_goal_dist

    def _compute_stage_scale(
        self,
        iter_idx: int,
        n_total: int,
        stagnated: bool,
        goal_dist: Optional[float] = None,
    ) -> torch.Tensor:
        if self.enable_stage_scale:
            H = self.horizon
            h_idx = torch.arange(1, H + 1, **self.tensor_args)
            k = float(iter_idx)
            K = float(max(n_total, 1))
            stage_scale = self.sigma_0 * torch.exp(
                self.sigma_1 * (h_idx - H) / H - self.sigma_2 * (k / K)
            )
        else:
            stage_scale = torch.full((self.horizon,), self.sigma_0, **self.tensor_args)

        near_goal_active, near_goal_scale_factor = self._compute_near_goal_scale_factor(goal_dist)
        self._last_near_goal_active = bool(near_goal_active)
        self._last_near_goal_scale_factor = float(near_goal_scale_factor)
        stage_scale = stage_scale * near_goal_scale_factor

        if (
            self.enable_stagnation_amplification
            and stagnated
            and not self._should_disable_stagnation_amplification(goal_dist)
        ):
            stage_scale = (1.0 + self.stagnation_alpha) * stage_scale
            self._last_stagnation_amplification_applied = True
        else:
            self._last_stagnation_amplification_applied = False

        self._last_near_goal_scale_floor_active = False
        if near_goal_active and self.near_goal_scale_floor > 0.0:
            self._last_near_goal_scale_floor_active = bool(
                torch.any(stage_scale < self.near_goal_scale_floor).item()
            )
            stage_scale = torch.clamp(stage_scale, min=self.near_goal_scale_floor)
        self._last_near_goal_scale_after_floor = (
            float(stage_scale.mean().item()) if near_goal_active else float(self.sigma_0)
        )
        return stage_scale

    def _compute_near_goal_scale_factor(self, goal_dist: Optional[float]) -> Tuple[bool, float]:
        if goal_dist is None:
            return False, 1.0
        threshold = max(self.near_goal_scale_threshold, self.trace_tol)
        if goal_dist >= threshold:
            return False, 1.0
        ratio = max(0.0, min(float(goal_dist) / threshold, 1.0))
        scale_factor = self.near_goal_scale_min_factor + (1.0 - self.near_goal_scale_min_factor) * ratio
        return True, float(scale_factor)

    def _should_disable_stagnation_amplification(self, goal_dist: Optional[float]) -> bool:
        if goal_dist is None:
            return False
        return bool(goal_dist <= self.near_goal_stagnation_disable_threshold)

    def _batch_stabilize_shape_matrices(
        self,
        matrices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Stabilize a batch of [H, A, A] proposal shape matrices and return both
        the stabilized matrices and their Cholesky factors.

        This keeps the hot path fully batched:
        - no per-step Python loop for Cholesky attempts
        - numerical fallback count only reflects stability handling
        """
        matrices = 0.5 * (matrices + matrices.transpose(-2, -1))
        H = matrices.shape[0]

        stable = matrices.clone()
        shape_tril = torch.empty_like(stable)
        used_fallback = torch.zeros(H, dtype=torch.bool, device=self.device)

        finite_mask = torch.isfinite(stable).all(dim=(-2, -1))
        invalid_idx = torch.where(~finite_mask)[0]
        if invalid_idx.numel() > 0:
            stable[invalid_idx] = self.I
            shape_tril[invalid_idx] = self.I
            used_fallback[invalid_idx] = True

        pending = finite_mask.clone()
        if pending.any():
            pending_idx = torch.where(pending)[0]
            chol, info = torch.linalg.cholesky_ex(stable[pending_idx], check_errors=False)
            ok = info.eq(0)
            if ok.any():
                ok_idx = pending_idx[ok]
                shape_tril[ok_idx] = chol[ok]
                pending[ok_idx] = False

        for jitter in self.cholesky_jitter:
            if not pending.any():
                break
            pending_idx = torch.where(pending)[0]
            candidate = stable[pending_idx] + jitter * self.I
            chol, info = torch.linalg.cholesky_ex(candidate, check_errors=False)
            ok = info.eq(0)
            if ok.any():
                ok_idx = pending_idx[ok]
                stable[ok_idx] = candidate[ok]
                shape_tril[ok_idx] = chol[ok]
                used_fallback[ok_idx] = True
                pending[ok_idx] = False

        if pending.any():
            pending_idx = torch.where(pending)[0]
            stable[pending_idx] = self.I
            shape_tril[pending_idx] = self.I
            used_fallback[pending_idx] = True

        return stable, shape_tril, int(used_fallback.sum().item())

    def _build_proposal_scale_tril(
        self,
        iter_idx: int,
        n_total: int,
        stagnated: bool,
        goal_dist: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stage_scale = self._compute_stage_scale(iter_idx, n_total, stagnated, goal_dist)
        proposal_scale_tril = torch.sqrt(stage_scale).view(self.horizon, 1, 1) * self._shape_tril

        self._last_stage_scale = stage_scale
        self._last_scale_tril = proposal_scale_tril
        self._last_proposal_scale_min = float(stage_scale.amin().item())
        self._last_proposal_scale_max = float(stage_scale.amax().item())
        self._last_near_goal_proposal_scale = (
            float(stage_scale.mean().item()) if self._last_near_goal_active else float(self.sigma_0)
        )
        return proposal_scale_tril, stage_scale

    def _sample_standard_noise(self, base_seed: int) -> torch.Tensor:
        delta = self.sample_lib.get_samples(
            sample_shape=self.sample_shape,
            base_seed=base_seed,
        )
        delta = torch.cat((delta, self.Z_seq), dim=0)
        return delta

    def _sample_actions(
        self,
        proposal_scale_tril: torch.Tensor,
        base_seed: int,
    ) -> torch.Tensor:
        delta = self._sample_standard_noise(base_seed=base_seed)
        scaled_delta = torch.einsum("nha,hac->nhc", delta, proposal_scale_tril)

        act_seq = self.mean_action.unsqueeze(0) + scaled_delta
        act_seq = scale_ctrl(
            act_seq,
            self.action_lows,
            self.action_highs,
            squash_fn=self.squash_fn,
        )

        append_acts = self.best_traj.unsqueeze(0)
        if self.num_null_particles > 0:
            neg_action = -1.0 * self.mean_action.unsqueeze(0)
            neg_act_seqs = neg_action.expand(self.num_neg_particles, -1, -1)
            append_acts = torch.cat((append_acts, self.null_act_seqs, neg_act_seqs), dim=0)
        return torch.cat((act_seq, append_acts), dim=0)

    def _get_shape_update_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Shape learning uses only the stochastic proposal body.

        The appended deterministic helper samples (best/null/negative actions)
        are useful for candidate evaluation and mean update, but they can
        distort covariance estimation near the goal by collapsing the proposal
        onto hand-crafted directions.
        """
        if not self.shape_update_random_only:
            return actions

        random_count = max(self.num_nonzero_particles - 2, 0)
        if random_count > 0:
            return actions[:random_count]

        # If the random body is disabled, fall back to whatever non-appended
        # proposal samples exist so the controller remains numerically valid.
        non_appended_count = max(
            actions.shape[0] - (1 + self.num_null_particles + self.num_neg_particles),
            1,
        )
        return actions[:non_appended_count]

    def _standardize_rollout_dict(
        self,
        rollout: Dict[str, torch.Tensor],
        act_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        rollout = dict(rollout)
        rollout.setdefault("actions", act_seq)
        rollout.setdefault("rollout_time", 0.0)

        state_dict = rollout.get("state_dict")
        if state_dict is not None:
            if self.visual_traj not in rollout and self.visual_traj in state_dict:
                rollout[self.visual_traj] = state_dict[self.visual_traj]
            elif "state_seq" not in rollout and "state_seq" in state_dict:
                rollout["state_seq"] = state_dict["state_seq"]
            if "ee_pos_seq" not in rollout and "ee_pos_seq" in state_dict:
                rollout["ee_pos_seq"] = state_dict["ee_pos_seq"]
        return rollout

    def _build_rollout_dict(
        self,
        state: torch.Tensor,
        act_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        rollout = self.rollout_fn(state, act_seq)
        if isinstance(rollout, dict):
            rollout = self._standardize_rollout_dict(rollout, act_seq)
            if "state_dict" in rollout and "costs" in rollout:
                return rollout

        # Backward-compatible path for old rollouts that do not emit state_dict.
        if hasattr(self.rollout_fn, "dynamics_model") and hasattr(self.rollout_fn, "cost_fn"):
            with profiler.record_function("sage/rollout/model"):
                state_dict = self.rollout_fn.dynamics_model.rollout_open_loop(state, act_seq)
            with profiler.record_function("sage/rollout/cost"):
                costs = self.rollout_fn.cost_fn(state_dict, act_seq)

            fallback_rollout = {
                "actions": act_seq,
                "costs": costs,
                "rollout_time": 0.0,
                "state_dict": state_dict,
            }
            if self.visual_traj in state_dict:
                fallback_rollout[self.visual_traj] = state_dict[self.visual_traj]
            elif "state_seq" in state_dict:
                fallback_rollout["state_seq"] = state_dict["state_seq"]
            if "ee_pos_seq" in state_dict:
                fallback_rollout["ee_pos_seq"] = state_dict["ee_pos_seq"]
            return fallback_rollout

        return rollout

    def _compute_total_costs(self, costs: torch.Tensor) -> torch.Tensor:
        traj_costs = cost_to_go(costs, self.gamma_seq)[:, 0]
        self.total_costs = traj_costs
        return traj_costs

    def _compute_mppi_weights(self, total_costs: torch.Tensor) -> torch.Tensor:
        return torch.softmax((-1.0 / self.lambda_) * total_costs, dim=0)

    def _get_shape_temperature(self, near_goal_active: bool) -> float:
        if near_goal_active:
            return self.near_goal_shape_temperature_multiplier
        return self.shape_temperature_multiplier

    def _compute_shape_update_weights(
        self,
        total_costs: torch.Tensor,
        shape_count: int,
        near_goal_active: bool,
    ) -> Tuple[torch.Tensor, float]:
        """
        Shape update uses its own flatter weighting than the mean update.

        Mean update keeps the original MPPI weights over all candidates.
        Shape update instead uses only the shape-sample subset, with a higher
        temperature to reduce weight collapse near the goal.
        """
        shape_temperature = self._get_shape_temperature(near_goal_active)
        shape_total_costs = total_costs[:shape_count]
        shape_weights = torch.softmax(
            (-1.0 / (self.lambda_ * shape_temperature)) * shape_total_costs,
            dim=0,
        )
        return shape_weights, float(shape_temperature)

    def _compute_weight_entropy(self, weights: torch.Tensor) -> float:
        clipped_weights = torch.clamp(weights, min=self.trace_tol)
        entropy = -torch.sum(weights * torch.log(clipped_weights))
        return float(entropy.item())

    def _compute_normalized_weight_entropy(
        self,
        weight_entropy: float,
        sample_count: Optional[int] = None,
    ) -> float:
        if sample_count is None:
            sample_count = self.num_particles
        norm = math.log(max(int(sample_count), 2))
        if norm <= 0.0:
            return 0.0
        return float(weight_entropy / norm)

    def _compute_shape_condition_number(self, shape_matrices: torch.Tensor) -> float:
        if shape_matrices.numel() == 0:
            return 1.0
        evals = torch.linalg.eigvalsh(shape_matrices)
        evals = torch.clamp(evals, min=self.trace_tol)
        cond = evals[..., -1] / evals[..., 0]
        return float(cond.mean().item())

    def _make_identity_shape(self) -> Tuple[torch.Tensor, torch.Tensor]:
        identity_shape = self.I.unsqueeze(0).repeat(self.horizon, 1, 1)
        return identity_shape, identity_shape.clone()

    def _get_near_goal_reference_shape(
        self,
        previous_shape_matrices: torch.Tensor,
        previous_shape_tril: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Near the goal, preserving the previous shape is usually better than
        collapsing to identity, but carrying a very sharp old shape unchanged
        can also lock the proposal into a narrow funnel.

        Blend the previous shape slightly back toward identity before reuse so
        the controller keeps a stable anisotropic prior without becoming overly
        brittle in the last few centimeters.
        """
        mix = self.near_goal_previous_shape_identity_mix
        if mix <= 0.0:
            return previous_shape_matrices, previous_shape_tril

        identity_shape = self.I.unsqueeze(0).repeat(self.horizon, 1, 1)
        blended = (1.0 - mix) * previous_shape_matrices + mix * identity_shape
        stabilized, stabilized_tril, _ = self._batch_stabilize_shape_matrices(blended)
        return stabilized, stabilized_tril

    def _get_shape_entropy_threshold(self, near_goal_active: bool) -> float:
        if near_goal_active:
            return self.near_goal_shape_update_min_normalized_entropy
        return self.shape_update_min_normalized_entropy

    def _should_skip_shape_update(
        self,
        shape_normalized_entropy: float,
        near_goal_active: bool,
    ) -> Tuple[bool, str]:
        if not self.enable_anisotropic_shape_update:
            return True, "anisotropic_disabled"
        if not self.enable_shape_collapse_guard:
            return False, ""
        if near_goal_active and self.near_goal_allow_low_entropy_shape_update:
            return False, ""
        if shape_normalized_entropy <= self._get_shape_entropy_threshold(near_goal_active):
            return True, "low_entropy"
        return False, ""

    def _should_update_shape_this_iter(
        self,
        iter_idx: int,
        n_total: int,
        near_goal_active: bool,
    ) -> bool:
        if not self.enable_anisotropic_shape_update:
            return False
        if near_goal_active and self.near_goal_update_shape_each_iter:
            return True
        if not self.shape_update_last_iter_only:
            return True
        return iter_idx == (max(n_total, 1) - 1)

    def _compute_all_sample_temporal_shape(
        self,
        actions: torch.Tensor,
        weights: torch.Tensor,
        temporal_mean: torch.Tensor,
        near_goal_active: bool = False,
        previous_shape_matrices: Optional[torch.Tensor] = None,
        previous_shape_tril: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, int, float, bool]:
        """
        Compute per-step anisotropic proposal shape from all rollout samples.

        Inputs:
        - actions: [N, H, A]
        - weights: [N], MPPI weights over all samples
        - temporal_mean: [H, A], usually the weighted temporal action mean

        Output:
        - shape_matrices: [H, A, A]
        - covariance_trace_mean: mean trace of the raw weighted covariance
        - covariance_fallback_count: number of steps that fell back for purely
          numerical reasons
        - shape_condition_number: mean condition number over stabilized shapes

        The raw weighted covariance is trace-normalized to preserve the
        stage-scale / shape separation: stage_scale carries scalar exploration
        magnitude, while shape_matrices capture anisotropy.
        """
        H = actions.shape[1]
        d = self.d_action
        identity_shape = self.I.unsqueeze(0).repeat(H, 1, 1)
        if near_goal_active and previous_shape_matrices is not None and previous_shape_tril is not None:
            (
                previous_shape_matrices,
                previous_shape_tril,
            ) = self._get_near_goal_reference_shape(previous_shape_matrices, previous_shape_tril)
        fallback_reference_shape = (
            previous_shape_matrices
            if near_goal_active and previous_shape_matrices is not None
            else identity_shape
        )
        fallback_reference_tril = (
            previous_shape_tril
            if near_goal_active and previous_shape_tril is not None
            else identity_shape.clone()
        )
        used_previous_shape = False

        centered = actions - temporal_mean.unsqueeze(0)
        raw_covariance = torch.einsum("n,nha,nhb->hab", weights, centered, centered)
        raw_covariance = 0.5 * (raw_covariance + raw_covariance.transpose(-2, -1))

        traces = raw_covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        valid_trace_mask = torch.isfinite(traces) & (traces > self.trace_tol)

        normalized_covariance = fallback_reference_shape.clone()
        if valid_trace_mask.any():
            denom = (traces[valid_trace_mask] / d).view(-1, 1, 1)
            normalized_valid = raw_covariance[valid_trace_mask] / denom
            normalized_valid = 0.5 * (
                normalized_valid + normalized_valid.transpose(-2, -1)
            )
            normalized_covariance[valid_trace_mask] = normalized_valid

        invalid_count = int((~valid_trace_mask).sum().item())
        shape_matrices, shape_tril, stabilize_fallback_count = self._batch_stabilize_shape_matrices(
            normalized_covariance
        )

        if not valid_trace_mask.all():
            invalid_idx = torch.where(~valid_trace_mask)[0]
            shape_matrices[invalid_idx] = fallback_reference_shape[invalid_idx]
            shape_tril[invalid_idx] = fallback_reference_tril[invalid_idx]
            if near_goal_active and previous_shape_matrices is not None:
                used_previous_shape = True

        fallback_count = invalid_count + stabilize_fallback_count
        fallback_fraction = float(fallback_count) / float(max(H, 1))

        if self.enable_shape_collapse_guard and fallback_fraction >= self.shape_update_max_fallback_fraction:
            if near_goal_active and self.near_goal_preserve_previous_shape and previous_shape_matrices is not None:
                shape_matrices = previous_shape_matrices.clone()
                shape_tril = fallback_reference_tril.clone()
                used_previous_shape = True
                shape_condition_number = self._compute_shape_condition_number(shape_matrices)
            else:
                shape_matrices = identity_shape
                shape_tril = identity_shape.clone()
                shape_condition_number = 1.0
            fallback_count = H
            covariance_trace_mean = 0.0
            return (
                shape_matrices,
                shape_tril,
                covariance_trace_mean,
                fallback_count,
                shape_condition_number,
                used_previous_shape,
            )

        if self.enable_runtime_stats and valid_trace_mask.any():
            covariance_trace_mean = float(traces[valid_trace_mask].mean().item())
        else:
            covariance_trace_mean = 0.0
        shape_condition_number = (
            self._compute_shape_condition_number(shape_matrices)
            if self.enable_runtime_stats
            else 1.0
        )
        return (
            shape_matrices,
            shape_tril,
            covariance_trace_mean,
            fallback_count,
            shape_condition_number,
            used_previous_shape,
        )

    def _update_distribution(
        self,
        trajectories: Dict[str, torch.Tensor],
        iter_idx: int,
        n_total: int,
        near_goal_active: bool,
    ):
        actions = trajectories["actions"].to(**self.tensor_args)
        costs = trajectories["costs"].to(**self.tensor_args)

        total_costs = self._compute_total_costs(costs)
        weights = self._compute_mppi_weights(total_costs)
        self._last_full_weight_entropy = self._compute_weight_entropy(weights)
        self._last_full_normalized_entropy = self._compute_normalized_weight_entropy(
            self._last_full_weight_entropy,
            weights.shape[0],
        )
        self._last_weight_entropy = self._last_full_weight_entropy
        self._last_shape_entropy_used_for_skip = "none"
        should_update_shape_this_iter = self._should_update_shape_this_iter(
            iter_idx,
            n_total,
            near_goal_active,
        )

        best_idx = torch.argmin(total_costs)
        self.best_idx = best_idx
        self.best_traj = torch.index_select(actions, 0, best_idx).squeeze(0)

        if iter_idx == (max(n_total, 1) - 1):
            if self.visual_traj in trajectories:
                vis_seq = trajectories[self.visual_traj].to(**self.tensor_args)
            elif "state_seq" in trajectories:
                vis_seq = trajectories["state_seq"].to(**self.tensor_args)
            else:
                vis_seq = actions

            k_top = min(10, actions.shape[0])
            top_values, top_idx = torch.topk(-total_costs, k_top)
            self.top_values = -top_values
            self.top_idx = top_idx
            self.top_trajs = torch.index_select(vis_seq, 0, top_idx).squeeze(0)

        new_mean = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * actions, dim=0)
        previous_shape_matrices = self.shape_matrices.clone()
        previous_shape_tril = self._shape_tril.clone()
        if near_goal_active and self.near_goal_preserve_previous_shape:
            (
                previous_shape_matrices,
                previous_shape_tril,
            ) = self._get_near_goal_reference_shape(previous_shape_matrices, previous_shape_tril)
        self._last_near_goal_used_previous_shape = False
        shape_actions = self._get_shape_update_actions(actions)
        shape_count = int(shape_actions.shape[0])
        shape_weights, self._last_shape_temperature_used = self._compute_shape_update_weights(
            total_costs=total_costs,
            shape_count=shape_count,
            near_goal_active=near_goal_active,
        )
        shape_mean = torch.sum(
            shape_weights.unsqueeze(-1).unsqueeze(-1) * shape_actions,
            dim=0,
        )
        self._last_shape_update_sample_count = shape_count
        self._last_shape_weight_entropy = self._compute_weight_entropy(shape_weights)
        self._last_shape_weight_entropy_after_flatten = self._last_shape_weight_entropy
        self._last_shape_normalized_entropy = self._compute_normalized_weight_entropy(
            self._last_shape_weight_entropy,
            shape_count,
        )
        if not should_update_shape_this_iter:
            self._last_shape_update_skipped = True
            self._last_shape_skip_reason = "deferred_until_last_iter"
            self._last_covariance_trace_mean = 0.0
            self._last_covariance_fallback_count = 0
            self._last_shape_condition_number = 1.0
            self._last_shape_entropy_used_for_skip = "shape"
        else:
            skip_shape_update, skip_reason = self._should_skip_shape_update(
                self._last_shape_normalized_entropy,
                near_goal_active=near_goal_active,
            )
            self._last_shape_update_skipped = bool(skip_shape_update)
            self._last_shape_skip_reason = str(skip_reason)
            self._last_shape_entropy_used_for_skip = "shape"
            if skip_shape_update:
                if near_goal_active and self.near_goal_preserve_previous_shape:
                    self.shape_matrices = previous_shape_matrices
                    self._shape_tril = previous_shape_tril
                    self._last_near_goal_used_previous_shape = True
                else:
                    self.shape_matrices, self._shape_tril = self._make_identity_shape()
                self._last_covariance_trace_mean = 0.0
                self._last_covariance_fallback_count = 0
                self._last_shape_condition_number = (
                    self._compute_shape_condition_number(self.shape_matrices)
                    if near_goal_active or self.enable_runtime_stats
                    else 1.0
                )
            else:
                (
                    new_shape_matrices,
                    new_shape_tril,
                    self._last_covariance_trace_mean,
                    self._last_covariance_fallback_count,
                    self._last_shape_condition_number,
                    used_previous_shape,
                ) = self._compute_all_sample_temporal_shape(
                    actions=shape_actions,
                    weights=shape_weights,
                    temporal_mean=shape_mean,
                    near_goal_active=near_goal_active,
                    previous_shape_matrices=previous_shape_matrices,
                    previous_shape_tril=previous_shape_tril,
                )
                self._last_near_goal_used_previous_shape = bool(used_previous_shape)
                if self.shape_update_ema < 1.0:
                    blended_shape = (
                        (1.0 - self.shape_update_ema) * self.shape_matrices
                        + self.shape_update_ema * new_shape_matrices
                    )
                    (
                        self.shape_matrices,
                        self._shape_tril,
                        ema_fallback_count,
                    ) = self._batch_stabilize_shape_matrices(blended_shape)
                    self._last_covariance_fallback_count += int(ema_fallback_count)
                else:
                    self.shape_matrices = new_shape_matrices
                    self._shape_tril = new_shape_tril
                if (
                    self.enable_shape_collapse_guard
                    and self._last_covariance_fallback_count >= self.horizon
                ):
                    self._last_shape_update_skipped = True
                    self._last_shape_skip_reason = "fallback_fraction"
                    if near_goal_active and self.near_goal_preserve_previous_shape:
                        self.shape_matrices = previous_shape_matrices
                        self._shape_tril = previous_shape_tril
                        self._last_near_goal_used_previous_shape = True
                        self._last_shape_condition_number = self._compute_shape_condition_number(
                            self.shape_matrices
                        )
        self._last_near_goal_shape_condition = (
            self._compute_shape_condition_number(self.shape_matrices)
            if near_goal_active
            else 1.0
        )
        self.mean_action = (1.0 - self.step_size_mean) * self.mean_action + self.step_size_mean * new_mean
        self.mean_action = scale_ctrl(
            self.mean_action,
            self.action_lows,
            self.action_highs,
            squash_fn=self.squash_fn,
        )

    def _shift(self, shift_steps: int = 1):
        if shift_steps <= 0:
            return
        if shift_steps >= self.horizon:
            self.reset_distribution()
            return

        self.mean_action = self.mean_action.roll(-shift_steps, dims=0)
        self.best_traj = self.best_traj.roll(-shift_steps, dims=0)
        self.shape_matrices = self.shape_matrices.roll(-shift_steps, dims=0)
        self._shape_tril = self._shape_tril.roll(-shift_steps, dims=0)

        if self.base_action == "random":
            torch.manual_seed(self.seed_val + 123 * self.num_steps)
            random_tail = torch.randn(shift_steps, self.d_action, **self.tensor_args)
            random_tail = math.sqrt(self.init_cov) * random_tail
            self.mean_action[-shift_steps:] = random_tail
            self.best_traj[-shift_steps:] = random_tail
        elif self.base_action == "null":
            self.mean_action[-shift_steps:].zero_()
            self.best_traj[-shift_steps:].zero_()
        elif self.base_action == "repeat":
            tail_source = max(self.horizon - shift_steps - 1, 0)
            self.mean_action[-shift_steps:] = self.mean_action[tail_source].clone()
            self.best_traj[-shift_steps:] = self.best_traj[tail_source].clone()
        else:
            raise NotImplementedError(f"Unsupported base_action for SAGE shift: {self.base_action}")

        identity_tail = self.I.unsqueeze(0).repeat(shift_steps, 1, 1)
        self.shape_matrices[-shift_steps:] = identity_tail
        self._shape_tril[-shift_steps:] = identity_tail

    def _get_action_seq(self, mode: str = "mean") -> torch.Tensor:
        if mode == "mean":
            act_seq = self.mean_action.clone()
        elif mode == "sample":
            proposal_scale_tril = self._last_scale_tril
            if proposal_scale_tril is None:
                proposal_scale_tril, _ = self._build_proposal_scale_tril(
                    iter_idx=max(self.n_iters - 1, 0),
                    n_total=max(self.n_iters, 1),
                    stagnated=False,
                )
            delta = self.sample_lib.get_samples(
                sample_shape=torch.Size([1]),
                base_seed=self.seed_val + 1009 * max(self.num_steps, 1),
            )
            scaled_delta = torch.einsum("nha,hac->nhc", delta, proposal_scale_tril)
            act_seq = self.mean_action.unsqueeze(0) + scaled_delta
            act_seq = act_seq.squeeze(0)
        else:
            raise ValueError(f"Unsupported sample_mode: {mode}")

        return scale_ctrl(
            act_seq,
            self.action_lows,
            self.action_highs,
            squash_fn=self.squash_fn,
        )

    def generate_rollouts(
        self,
        state: torch.Tensor,
        proposal_scale_tril: torch.Tensor,
        base_seed: int,
    ) -> Dict[str, torch.Tensor]:
        act_seq = self._sample_actions(proposal_scale_tril=proposal_scale_tril, base_seed=base_seed)
        return self._build_rollout_dict(state, act_seq)

    @property
    def entropy(self) -> torch.Tensor:
        if self._last_scale_tril is None:
            return torch.tensor(0.0, **self.tensor_args)
        entropies = []
        for h in range(self.horizon):
            entropies.append(gaussian_entropy(L=self._last_scale_tril[h]))
        return torch.stack(entropies).mean()

    def _calc_val(self, trajectories: Dict[str, torch.Tensor]) -> torch.Tensor:
        costs = trajectories["costs"].to(**self.tensor_args)
        total_costs = self._compute_total_costs(costs)
        return -self.lambda_ * torch.logsumexp((-1.0 / self.lambda_) * total_costs, dim=0)

    def optimize(
        self,
        state: torch.Tensor,
        calc_val: bool = False,
        shift_steps: int = 1,
        n_iters: Optional[int] = None,
    ):
        n_total = self.n_iters if n_iters is None else int(n_iters)
        inp_device = state.device
        inp_dtype = state.dtype
        state = state.to(**self.tensor_args)

        if self.hotstart:
            self._shift(shift_steps)
        else:
            self.reset_distribution()

        goal_progress, goal_dist = self._compute_goal_progress(state)
        stagnated = goal_dist is not None and goal_progress < self.tau_p

        info = {
            "rollout_time": 0.0,
            "entropy": [],
            "iteration_costs": [],
            "stage_scale_mean": [],
            "weight_entropy_seq": [],
            "full_weight_entropy_seq": [],
            "shape_weight_entropy_seq": [],
            "full_normalized_entropy_seq": [],
            "shape_normalized_entropy_seq": [],
            "covariance_trace_mean_seq": [],
            "shape_condition_number_seq": [],
            "covariance_fallback_count_seq": [],
            "shape_update_skipped_seq": [],
            "shape_skip_reason_seq": [],
            "goal_progress": goal_progress,
            "goal_dist": goal_dist,
            "final_goal_distance": goal_dist,
            "success": None,
            "failure": None,
            "z_t": int(bool(stagnated)),
            "stagnation": float(bool(stagnated)),
            "near_goal_active": False,
            "near_goal_scale_factor": 1.0,
            "stagnation_amplification_applied": False,
            "shape_skip_count_near_goal": 0,
            "low_entropy_trigger_count_near_goal": 0,
            "fallback_fraction_trigger_count_near_goal": 0,
            "near_goal_shape_condition": 1.0,
            "near_goal_proposal_scale": float(self.sigma_0),
            "near_goal_shape_update_used_previous_shape": False,
            "covariance_fallback": False,
            "covariance_fallback_count": 0,
            "weight_entropy": 0.0,
            "full_weight_entropy": 0.0,
            "shape_weight_entropy": 0.0,
            "full_normalized_entropy": 0.0,
            "shape_normalized_entropy": 0.0,
            "shape_entropy_used_for_skip": "none",
            "shape_temperature_used": 1.0,
            "shape_weight_entropy_after_flatten": 0.0,
            "covariance_trace_mean": 0.0,
            "shape_condition_number": 1.0,
            "proposal_scale_min": float(self.sigma_0),
            "proposal_scale_max": float(self.sigma_0),
            "near_goal_scale_floor_active": False,
            "near_goal_scale_after_floor": float(self.sigma_0),
            "shape_update_skipped": False,
            "shape_skip_reason": "",
            "shape_update_sample_count": 0,
            "shape_update_used_previous_shape": False,
            "output_mode_used": self.sample_mode,
            "enable_runtime_stats": self.enable_runtime_stats,
        }

        with torch.amp.autocast("cuda", enabled=True):
            with torch.no_grad():
                for iter_idx in range(n_total):
                    proposal_scale_tril, stage_scale = self._build_proposal_scale_tril(
                        iter_idx=iter_idx,
                        n_total=n_total,
                        stagnated=bool(stagnated),
                        goal_dist=goal_dist,
                    )
                    trajectories = self.generate_rollouts(
                        state,
                        proposal_scale_tril=proposal_scale_tril,
                        base_seed=self.seed_val + 97 * self.num_steps + 7919 * iter_idx,
                    )

                    with profiler.record_function("sage_update"):
                        self._update_distribution(
                            trajectories=trajectories,
                            iter_idx=iter_idx,
                            n_total=n_total,
                            near_goal_active=self._last_near_goal_active,
                        )

                    info["shape_update_skipped_seq"].append(bool(self._last_shape_update_skipped))
                    info["shape_skip_reason_seq"].append(str(self._last_shape_skip_reason))
                    info["near_goal_active"] = bool(self._last_near_goal_active)
                    info["near_goal_scale_factor"] = float(self._last_near_goal_scale_factor)
                    info["stagnation_amplification_applied"] = bool(
                        self._last_stagnation_amplification_applied
                    )
                    info["full_weight_entropy"] = float(self._last_full_weight_entropy)
                    info["shape_weight_entropy"] = float(self._last_shape_weight_entropy)
                    info["full_normalized_entropy"] = float(self._last_full_normalized_entropy)
                    info["shape_normalized_entropy"] = float(
                        self._last_shape_normalized_entropy
                    )
                    info["shape_entropy_used_for_skip"] = str(
                        self._last_shape_entropy_used_for_skip
                    )
                    info["shape_temperature_used"] = float(self._last_shape_temperature_used)
                    info["shape_weight_entropy_after_flatten"] = float(
                        self._last_shape_weight_entropy_after_flatten
                    )
                    info["near_goal_shape_condition"] = float(self._last_near_goal_shape_condition)
                    info["near_goal_proposal_scale"] = float(self._last_near_goal_proposal_scale)
                    info["near_goal_scale_floor_active"] = bool(
                        self._last_near_goal_scale_floor_active
                    )
                    info["near_goal_scale_after_floor"] = float(
                        self._last_near_goal_scale_after_floor
                    )
                    info["near_goal_shape_update_used_previous_shape"] = bool(
                        info["near_goal_shape_update_used_previous_shape"]
                        or self._last_near_goal_used_previous_shape
                    )
                    info["shape_update_used_previous_shape"] = bool(
                        info["shape_update_used_previous_shape"]
                        or self._last_near_goal_used_previous_shape
                    )
                    info["shape_update_sample_count"] = int(self._last_shape_update_sample_count)
                    if self._last_near_goal_active and self._last_shape_update_skipped:
                        info["shape_skip_count_near_goal"] += 1
                    if self._last_near_goal_active and self._last_shape_skip_reason == "low_entropy":
                        info["low_entropy_trigger_count_near_goal"] += 1
                    if self._last_near_goal_active and self._last_shape_skip_reason == "fallback_fraction":
                        info["fallback_fraction_trigger_count_near_goal"] += 1
                    if self.enable_runtime_stats:
                        info["rollout_time"] += float(trajectories.get("rollout_time", 0.0))
                        info["iteration_costs"].append(float(self.total_costs.min().item()))
                        info["stage_scale_mean"].append(float(stage_scale.mean().item()))
                        info["weight_entropy_seq"].append(float(self._last_weight_entropy))
                        info["full_weight_entropy_seq"].append(
                            float(self._last_full_weight_entropy)
                        )
                        info["shape_weight_entropy_seq"].append(
                            float(self._last_shape_weight_entropy)
                        )
                        info["full_normalized_entropy_seq"].append(
                            float(self._last_full_normalized_entropy)
                        )
                        info["shape_normalized_entropy_seq"].append(
                            float(self._last_shape_normalized_entropy)
                        )
                        info["covariance_trace_mean_seq"].append(
                            float(self._last_covariance_trace_mean)
                        )
                        info["shape_condition_number_seq"].append(
                            float(self._last_shape_condition_number)
                        )
                        info["covariance_fallback_count_seq"].append(
                            int(self._last_covariance_fallback_count)
                        )

        self.trajectories = trajectories
        success, failure = self._infer_task_outcome(trajectories, goal_dist)
        self._last_success = success
        self._last_failure = failure
        info["covariance_fallback_count"] = (
            int(sum(info["covariance_fallback_count_seq"]))
            if self.enable_runtime_stats
            else int(self._last_covariance_fallback_count)
        )
        info["covariance_fallback"] = bool(info["covariance_fallback_count"] > 0)
        info["success"] = success
        info["failure"] = failure
        info["weight_entropy"] = (
            info["weight_entropy_seq"][-1]
            if self.enable_runtime_stats and info["weight_entropy_seq"]
            else float(self._last_weight_entropy)
        )
        info["full_weight_entropy"] = (
            info["full_weight_entropy_seq"][-1]
            if self.enable_runtime_stats and info["full_weight_entropy_seq"]
            else float(self._last_full_weight_entropy)
        )
        info["shape_weight_entropy"] = (
            info["shape_weight_entropy_seq"][-1]
            if self.enable_runtime_stats and info["shape_weight_entropy_seq"]
            else float(self._last_shape_weight_entropy)
        )
        info["full_normalized_entropy"] = (
            info["full_normalized_entropy_seq"][-1]
            if self.enable_runtime_stats and info["full_normalized_entropy_seq"]
            else float(self._last_full_normalized_entropy)
        )
        info["shape_normalized_entropy"] = (
            info["shape_normalized_entropy_seq"][-1]
            if self.enable_runtime_stats and info["shape_normalized_entropy_seq"]
            else float(self._last_shape_normalized_entropy)
        )
        info["shape_temperature_used"] = float(self._last_shape_temperature_used)
        info["shape_weight_entropy_after_flatten"] = float(
            self._last_shape_weight_entropy_after_flatten
        )
        info["covariance_trace_mean"] = (
            info["covariance_trace_mean_seq"][-1]
            if self.enable_runtime_stats and info["covariance_trace_mean_seq"]
            else 0.0
        )
        info["shape_condition_number"] = (
            info["shape_condition_number_seq"][-1]
            if self.enable_runtime_stats and info["shape_condition_number_seq"]
            else 1.0
        )
        info["proposal_scale_min"] = float(self._last_proposal_scale_min)
        info["proposal_scale_max"] = float(self._last_proposal_scale_max)
        info["shape_update_skipped"] = bool(
            info["shape_update_skipped_seq"][-1] if info["shape_update_skipped_seq"] else False
        )
        info["shape_skip_reason"] = (
            info["shape_skip_reason_seq"][-1] if info["shape_skip_reason_seq"] else ""
        )
        info["near_goal_scale_floor_active"] = bool(self._last_near_goal_scale_floor_active)
        info["near_goal_scale_after_floor"] = float(self._last_near_goal_scale_after_floor)
        info["shape_update_used_previous_shape"] = bool(
            info["shape_update_used_previous_shape"]
            or info["near_goal_shape_update_used_previous_shape"]
        )
        if self.enable_runtime_stats:
            info["entropy"].append(float(self.entropy.item()))
        info["controller_core"] = {
            "enable_stage_scale": self.enable_stage_scale,
            "enable_anisotropic_shape_update": self.enable_anisotropic_shape_update,
            "enable_stagnation_amplification": self.enable_stagnation_amplification,
            "enable_runtime_stats": self.enable_runtime_stats,
            "shape_update_last_iter_only": self.shape_update_last_iter_only,
            "shape_update_random_only": self.shape_update_random_only,
            "shape_temperature_multiplier": self.shape_temperature_multiplier,
            "near_goal_update_shape_each_iter": self.near_goal_update_shape_each_iter,
            "near_goal_execute_best": self.near_goal_execute_best,
            "near_goal_scale_floor": self.near_goal_scale_floor,
            "near_goal_shape_update_min_normalized_entropy": self.near_goal_shape_update_min_normalized_entropy,
            "near_goal_shape_temperature_multiplier": self.near_goal_shape_temperature_multiplier,
            "near_goal_preserve_previous_shape": self.near_goal_preserve_previous_shape,
            "near_goal_allow_low_entropy_shape_update": self.near_goal_allow_low_entropy_shape_update,
            "near_goal_previous_shape_identity_mix": self.near_goal_previous_shape_identity_mix,
        }
        info["stats"] = {
            "success": success,
            "failure": failure,
            "final_goal_distance": goal_dist,
            "weight_entropy": info["weight_entropy"],
            "full_weight_entropy": info["full_weight_entropy"],
            "shape_weight_entropy": info["shape_weight_entropy"],
            "full_normalized_entropy": info["full_normalized_entropy"],
            "shape_normalized_entropy": info["shape_normalized_entropy"],
            "shape_entropy_used_for_skip": info["shape_entropy_used_for_skip"],
            "shape_temperature_used": info["shape_temperature_used"],
            "shape_weight_entropy_after_flatten": info["shape_weight_entropy_after_flatten"],
            "covariance_trace_mean": info["covariance_trace_mean"],
            "shape_condition_number": info["shape_condition_number"],
            "proposal_scale_min": info["proposal_scale_min"],
            "proposal_scale_max": info["proposal_scale_max"],
            "covariance_fallback_count": info["covariance_fallback_count"],
            "z_t": info["z_t"],
            "near_goal_active": info["near_goal_active"],
            "near_goal_scale_factor": info["near_goal_scale_factor"],
            "stagnation_amplification_applied": info["stagnation_amplification_applied"],
            "near_goal_scale_floor_active": info["near_goal_scale_floor_active"],
            "near_goal_scale_after_floor": info["near_goal_scale_after_floor"],
            "shape_skip_count_near_goal": info["shape_skip_count_near_goal"],
            "low_entropy_trigger_count_near_goal": info["low_entropy_trigger_count_near_goal"],
            "fallback_fraction_trigger_count_near_goal": info["fallback_fraction_trigger_count_near_goal"],
            "near_goal_shape_condition": info["near_goal_shape_condition"],
            "near_goal_proposal_scale": info["near_goal_proposal_scale"],
            "near_goal_shape_update_used_previous_shape": info["near_goal_shape_update_used_previous_shape"],
            "shape_update_used_previous_shape": info["shape_update_used_previous_shape"],
            "covariance_fallback": info["covariance_fallback"],
            "shape_update_skipped": info["shape_update_skipped"],
            "shape_skip_reason": info["shape_skip_reason"],
            "shape_update_sample_count": info["shape_update_sample_count"],
            "output_mode_used": info["output_mode_used"],
            **info["controller_core"],
        }
        self.latest_stats = dict(info["stats"])

        output_mode = self.sample_mode
        if self._last_near_goal_active and self.near_goal_execute_best:
            output_mode = "best"
        if self.execute_best and self.best_traj is not None:
            curr_action_seq = self.best_traj.clone()
            output_mode = "best"
        elif output_mode == "best" and self.best_traj is not None:
            curr_action_seq = self.best_traj.clone()
        else:
            curr_action_seq = self._get_action_seq(mode=self.sample_mode)
        info["output_mode_used"] = output_mode
        info["stats"]["output_mode_used"] = output_mode
        self.latest_stats["output_mode_used"] = output_mode

        value = 0.0
        if calc_val:
            value = self._calc_val(trajectories)

        self.num_steps += 1
        return curr_action_seq.to(inp_device, dtype=inp_dtype), value, info

    def get_optimal_value(self, state: torch.Tensor):
        self.reset()
        _, value, _ = self.optimize(state, calc_val=True, shift_steps=0)
        return value
