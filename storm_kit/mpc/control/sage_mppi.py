#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
# Additional standalone SAGE-MPPI implementation for whole_control branch.
#

"""
Standalone SAGE-MPPI controller.

This file intentionally does not inherit from the existing STORM controllers.
It only mirrors the controller/task/control-process interface that the current
project already expects:

    task -> ControlProcess -> controller.optimize()

Design goals:
- Keep the constructor and public attributes close to the existing MPPI style.
- Reuse trusted utility code (sampling libs, rollout invocation conventions,
  scaling helpers, discounted cost helper).
- Implement only the three requested SAGE pieces:
  1. stage-scaled proposal
  2. safe-elite anisotropic covariance
  3. stagnation-triggered amplification

Current whole_control branch note:
- Existing arm rollouts do not expose a per-rollout signed safety margin.
- To obtain delta_n without modifying rollout/task files, this controller
  reuses the already-built collision modules on the rollout object and derives
  a conservative signed safety margin from those tensors.
- Future task/rollout integration can avoid this extra work by directly
  attaching `delta_safe` or `safety_margin_seq` to the rollout dictionary.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.autograd.profiler as profiler

from .control_utils import cost_to_go, gaussian_entropy, scale_ctrl
from .sample_libs import (
    HaltonSampleLib,
    MultipleSampleLib,
    RandomSampleLib,
    StompSampleLib,
)


class SAGE_MPPI:
    """
    Standalone SAGE-MPPI controller.

    This class does not inherit from the current STORM controller hierarchy.
    It only matches the interface and state that `ControlProcess` and the task
    wrappers already consume.

    Compatibility expectations satisfied by this class:
    - `controller.rollout_fn`
    - `controller.tensor_args`
    - `controller.optimize(state, shift_steps=...)`
    - `controller.reset()`
    - `controller.reset_covariance()`
    - `controller.top_idx`, `controller.top_values`, `controller.top_trajs`
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
        eta=0.2,
        tau_p=1.0e-4,
        stagnation_alpha=0.0,
        execute_best=False,
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
        self.step_size_cov = float(step_size_cov)  # kept for signature parity
        self.alpha = alpha  # baseline MPPI compatibility field, intentionally unused
        self.gamma = float(gamma)
        self.kappa = float(kappa)  # kept for hotstart compatibility; not used in SAGE math
        self.n_iters = int(n_iters)
        self.sample_mode = sample_mode
        self.hotstart = bool(hotstart)
        self.squash_fn = squash_fn
        self.update_cov = bool(update_cov)  # kept for signature parity
        self.cov_type = cov_type  # kept for signature parity
        self.seed_val = int(seed)
        self.sample_params = sample_params
        self.visual_traj = visual_traj
        self.execute_best = bool(execute_best)

        # SAGE parameters.
        self.lambda_ = float(beta)
        self.sigma_0 = float(self.init_cov if sigma_0 is None else sigma_0)
        self.sigma_1 = float(sigma_1)
        self.sigma_2 = float(sigma_2)
        self.eta = float(eta)
        self.tau_p = float(tau_p)
        self.stagnation_alpha = float(stagnation_alpha)

        if self.sigma_0 <= 0.0:
            raise ValueError("sigma_0 must be positive for SAGE proposal scaling")
        if not (0.0 < self.eta <= 1.0):
            raise ValueError("eta must be in (0, 1]")
        if self.lambda_ <= 0.0:
            raise ValueError("beta/lambda must be positive")

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
        self._last_safe_elite_fraction = 0.0
        self._last_safe_weight_mass = 0.0
        self._last_stage_scale = None
        self._last_scale_tril = None
        self._used_margin_fallback = False

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
        self._last_safe_elite_fraction = 0.0
        self._last_safe_weight_mass = 0.0
        self._last_stage_scale = None
        self._used_margin_fallback = False

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
        """
        Compute Delta_goal_t = prev_goal_dist - current_goal_dist.

        Preferred reaching-task path:
            g(x) = || p_ee(q) - p_goal ||_2

        Fallback path:
            if only goal_state exists, use joint-space distance on q.
        """
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
        self, iter_idx: int, n_total: int, stagnated: bool
    ) -> torch.Tensor:
        """
        s_{k,h} = sigma_0 * exp( sigma_1 * (h - H) / H - sigma_2 * k / K )

        Here `iter_idx` is zero-based and `h` is implemented as 1..H to match the
        requested formula.
        """
        H = self.horizon
        h_idx = torch.arange(1, H + 1, **self.tensor_args)
        k = float(iter_idx)
        K = float(max(n_total, 1))

        stage_scale = self.sigma_0 * torch.exp(
            self.sigma_1 * (h_idx - H) / H - self.sigma_2 * (k / K)
        )
        if stagnated:
            stage_scale = (1.0 + self.stagnation_alpha) * stage_scale
        return stage_scale

    def _stable_cholesky(self, matrix: torch.Tensor) -> torch.Tensor:
        sym = 0.5 * (matrix + matrix.transpose(-2, -1))
        try:
            return torch.linalg.cholesky(sym)
        except RuntimeError:
            pass

        for jitter in self.cholesky_jitter:
            try:
                return torch.linalg.cholesky(sym + jitter * self.I)
            except RuntimeError:
                continue
        # Final fallback keeps the controller alive instead of crashing the MPC loop.
        return torch.diag(torch.sqrt(torch.clamp(torch.diagonal(sym), min=0.0)))

    def _build_proposal_scale_tril(
        self, iter_idx: int, n_total: int, stagnated: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stage_scale = self._compute_stage_scale(iter_idx, n_total, stagnated)
        proposal_scale_tril = torch.empty(
            self.horizon, self.d_action, self.d_action, **self.tensor_args
        )

        for h in range(self.horizon):
            shape_tril = self._stable_cholesky(self.shape_matrices[h])
            proposal_scale_tril[h] = torch.sqrt(stage_scale[h]) * shape_tril

        self._last_stage_scale = stage_scale
        self._last_scale_tril = proposal_scale_tril
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
            append_acts = torch.cat(
                (append_acts, self.null_act_seqs, neg_act_seqs), dim=0
            )
        return torch.cat((act_seq, append_acts), dim=0)

    def _build_rollout_dict(
        self, state: torch.Tensor, act_seq: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Prefer the trusted rollout components directly, so we can keep the
        intermediate `state_dict` for safety-margin extraction without changing
        any existing rollout file.
        """
        if hasattr(self.rollout_fn, "dynamics_model") and hasattr(self.rollout_fn, "cost_fn"):
            with profiler.record_function("sage/rollout/model"):
                state_dict = self.rollout_fn.dynamics_model.rollout_open_loop(state, act_seq)
            with profiler.record_function("sage/rollout/cost"):
                costs = self.rollout_fn.cost_fn(state_dict, act_seq)

            rollout = {
                "actions": act_seq,
                "costs": costs,
                "rollout_time": 0.0,
                "state_dict": state_dict,
            }
            if self.visual_traj in state_dict:
                rollout[self.visual_traj] = state_dict[self.visual_traj]
            elif "state_seq" in state_dict:
                rollout[self.visual_traj] = state_dict["state_seq"]
            return rollout

        rollout = self.rollout_fn(state, act_seq)
        return rollout

    def _compute_rollout_safety_margin(
        self, rollout_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute delta_n = min signed safety margin along each rollout.

        Current whole_control limitation:
        - Arm rollouts return costs/actions/visuals, but do not expose a direct
          signed safety margin field.
        - Therefore this method derives a conservative margin by reusing the
          rollout object's already-instantiated collision modules and the
          `state_dict` retained by `_build_rollout_dict`.

        Preferred direct fields if future rollout integration adds them:
        - `delta_safe`
        - `safety_margin_seq`
        - `collision_margin_seq`

        Sign convention used here:
        - existing collision modules operate on signed distances where positive
          means collision / penetration and negative means safe separation.
        - the paper's safe elite uses delta_n > 0 for safe rollouts.
        - therefore we convert each raw signed distance d_raw into a safety
          margin m = -(d_raw + distance_threshold), so:
              m > 0   => still safely outside the threshold
              m <= 0  => colliding or too close to the threshold
        """
        if "delta_safe" in rollout_dict:
            return rollout_dict["delta_safe"].to(**self.tensor_args)

        if "safety_margin_seq" in rollout_dict:
            return rollout_dict["safety_margin_seq"].to(**self.tensor_args).amin(dim=-1)

        if "collision_margin_seq" in rollout_dict:
            return rollout_dict["collision_margin_seq"].to(**self.tensor_args).amin(dim=-1)

        state_dict = rollout_dict.get("state_dict")
        if state_dict is None:
            self._used_margin_fallback = True
            return -torch.ones(self.num_particles, **self.tensor_args)

        margins = []
        batch_size = rollout_dict["actions"].shape[0]
        horizon = rollout_dict["actions"].shape[1]

        if (
            hasattr(self.rollout_fn, "primitive_collision_cost")
            and "link_pos_seq" in state_dict
            and "link_rot_seq" in state_dict
        ):
            p_cost = self.rollout_fn.primitive_collision_cost
            n_links = state_dict["link_pos_seq"].shape[2]

            if p_cost.batch_size != batch_size:
                p_cost.batch_size = batch_size
                p_cost.robot_world_coll.build_batch_features(
                    batch_size * horizon,
                    clone_pose=True,
                    clone_points=True,
                )

            link_pos_batch = state_dict["link_pos_seq"].view(batch_size * horizon, n_links, 3)
            link_rot_batch = state_dict["link_rot_seq"].view(batch_size * horizon, n_links, 3, 3)
            raw_signed_dist = p_cost.robot_world_coll.check_robot_sphere_collisions(
                link_pos_batch, link_rot_batch
            ).view(batch_size, horizon, n_links)
            primitive_margin = -(raw_signed_dist + p_cost.distance_threshold)
            margins.append(primitive_margin.amin(dim=(1, 2)))

        if hasattr(self.rollout_fn, "robot_self_collision_cost") and "state_seq" in state_dict:
            self_cost = self.rollout_fn.robot_self_collision_cost
            n_dofs = self.rollout_fn.dynamics_model.n_dofs
            q_seq = state_dict["state_seq"][:, :, :n_dofs]
            q_flat = q_seq.reshape(batch_size * horizon, n_dofs)
            raw_signed_dist = self_cost.coll.check_self_collisions_nn(q_flat).view(
                batch_size, horizon
            )
            self_margin = -(raw_signed_dist + self_cost.distance_threshold)
            margins.append(self_margin.amin(dim=1))

        if margins:
            self._used_margin_fallback = False
            return torch.stack(margins, dim=0).amin(dim=0)

        # Conservative fallback:
        # if the current rollout/task does not expose enough information to
        # recover a true signed safety margin, return a negative margin for all
        # rollouts so the safe-elite set becomes empty and the shape falls back
        # to identity.
        self._used_margin_fallback = True
        return -torch.ones(batch_size, **self.tensor_args)

    def _compute_total_costs(self, costs: torch.Tensor) -> torch.Tensor:
        traj_costs = cost_to_go(costs, self.gamma_seq)[:, 0]
        self.total_costs = traj_costs
        return traj_costs

    def _compute_mppi_weights(self, total_costs: torch.Tensor) -> torch.Tensor:
        return torch.softmax((-1.0 / self.lambda_) * total_costs, dim=0)

    def _compute_safe_elite_set(
        self, total_costs: torch.Tensor, delta_safe: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sorted_costs, _ = torch.sort(total_costs)
        quantile_idx = max(int(math.ceil(self.eta * total_costs.numel())) - 1, 0)
        cost_threshold = sorted_costs[quantile_idx]
        safe_mask = (total_costs <= cost_threshold) & (delta_safe > 0.0)
        return safe_mask, cost_threshold

    def _compute_safe_elite_covariance(
        self,
        actions: torch.Tensor,
        proposal_mean: torch.Tensor,
        weights: torch.Tensor,
        safe_mask: torch.Tensor,
        iter_idx: int,
        n_total: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the normalized anisotropic proposal shape.

        Formula implemented per requested spec:
            Chat_{k,h}   = sum_n wbar_n (u_n - mu)(u_n - mu)^T
            Ctilde_{k,h} = Chat_{k,h} / (trace(Chat_{k,h}) / d)
            rho_k        = (k / K) * sum_n w_n * 1[n in E_t]
            C_{k,h}      = (1 - rho_k) I + rho_k Ctilde_{k,h}

        Important trust behavior:
        - early iterations should not over-trust a noisy anisotropic estimate
        - if the safe elite carries little weight mass, stay closer to I
        """
        H = actions.shape[1]
        d = self.d_action
        identity_shape = self.I.unsqueeze(0).repeat(H, 1, 1)

        safe_weight_mass = torch.sum(weights * safe_mask.to(weights.dtype))
        self._last_safe_weight_mass = float(safe_weight_mass.item())
        self._last_safe_elite_fraction = float(
            safe_mask.to(torch.float32).mean().item()
        )

        if not torch.any(safe_mask):
            return identity_shape, safe_weight_mass

        safe_weights = weights[safe_mask]
        safe_actions = actions[safe_mask]
        safe_weights_sum = torch.sum(safe_weights)
        if safe_weights_sum <= 0.0:
            return identity_shape, safe_weight_mass

        # Requested formula uses normalized safe-elite weights without epsilon.
        wbar = safe_weights / safe_weights_sum
        centered = safe_actions - proposal_mean.unsqueeze(0)

        chat = torch.zeros(H, d, d, **self.tensor_args)
        for h in range(H):
            delta_h = centered[:, h, :]
            chat[h] = torch.einsum("n,ni,nj->ij", wbar, delta_h, delta_h)

        ctilde = identity_shape.clone()
        for h in range(H):
            trace_h = torch.trace(chat[h])
            if trace_h <= self.trace_tol:
                ctilde[h] = self.I
            else:
                ctilde[h] = chat[h] / (trace_h / d)
                ctilde[h] = 0.5 * (ctilde[h] + ctilde[h].transpose(-2, -1))

        rho_k = (float(iter_idx) / float(max(n_total, 1))) * float(safe_weight_mass.item())
        rho_k = max(0.0, min(1.0, rho_k))
        shape_matrices = (1.0 - rho_k) * identity_shape + rho_k * ctilde
        return shape_matrices, safe_weight_mass

    def _update_distribution(
        self,
        trajectories: Dict[str, torch.Tensor],
        iter_idx: int,
        n_total: int,
    ):
        actions = trajectories["actions"].to(**self.tensor_args)
        costs = trajectories["costs"].to(**self.tensor_args)

        if self.visual_traj in trajectories:
            vis_seq = trajectories[self.visual_traj].to(**self.tensor_args)
        elif "state_seq" in trajectories:
            vis_seq = trajectories["state_seq"].to(**self.tensor_args)
        else:
            vis_seq = actions

        proposal_mean = self.mean_action.clone()
        total_costs = self._compute_total_costs(costs)
        weights = self._compute_mppi_weights(total_costs)
        delta_safe = self._compute_rollout_safety_margin(trajectories)
        trajectories["delta_safe"] = delta_safe

        best_idx = torch.argmin(total_costs)
        self.best_idx = best_idx
        self.best_traj = torch.index_select(actions, 0, best_idx).squeeze(0)

        k_top = min(10, actions.shape[0])
        top_values, top_idx = torch.topk(-total_costs, k_top)
        self.top_values = -top_values
        self.top_idx = top_idx
        self.top_trajs = torch.index_select(vis_seq, 0, top_idx).squeeze(0)

        new_mean = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * actions, dim=0)
        self.mean_action = (
            (1.0 - self.step_size_mean) * self.mean_action
            + self.step_size_mean * new_mean
        )
        self.mean_action = scale_ctrl(
            self.mean_action,
            self.action_lows,
            self.action_highs,
            squash_fn=self.squash_fn,
        )

        safe_mask, _ = self._compute_safe_elite_set(total_costs, delta_safe)
        self.shape_matrices, _ = self._compute_safe_elite_covariance(
            actions=actions,
            proposal_mean=proposal_mean,
            weights=weights,
            safe_mask=safe_mask,
            iter_idx=iter_idx,
            n_total=n_total,
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
            raise NotImplementedError(
                f"Unsupported base_action for SAGE_MPPI shift: {self.base_action}"
            )

        self.shape_matrices[-shift_steps:] = self.I.unsqueeze(0).repeat(shift_steps, 1, 1)

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
        act_seq = self._sample_actions(
            proposal_scale_tril=proposal_scale_tril,
            base_seed=base_seed,
        )
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
            "safe_elite_fraction": [],
            "safe_weight_mass": [],
            "goal_progress": goal_progress,
            "goal_dist": goal_dist,
            "stagnation": float(bool(stagnated)),
            "margin_fallback": False,
        }

        with torch.amp.autocast("cuda", enabled=True):
            with torch.no_grad():
                for iter_idx in range(n_total):
                    proposal_scale_tril, stage_scale = self._build_proposal_scale_tril(
                        iter_idx=iter_idx,
                        n_total=n_total,
                        stagnated=bool(stagnated),
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
                        )

                    info["rollout_time"] += float(trajectories.get("rollout_time", 0.0))
                    info["iteration_costs"].append(float(self.total_costs.min().item()))
                    info["stage_scale_mean"].append(float(stage_scale.mean().item()))
                    info["safe_elite_fraction"].append(float(self._last_safe_elite_fraction))
                    info["safe_weight_mass"].append(float(self._last_safe_weight_mass))

        self.trajectories = trajectories
        info["margin_fallback"] = bool(self._used_margin_fallback)
        info["entropy"].append(float(self.entropy.item()))

        if self.execute_best and self.best_traj is not None:
            curr_action_seq = self.best_traj.clone()
        else:
            curr_action_seq = self._get_action_seq(mode=self.sample_mode)

        value = 0.0
        if calc_val:
            value = self._calc_val(trajectories)

        self.num_steps += 1
        return curr_action_seq.to(inp_device, dtype=inp_dtype), value, info

    def get_optimal_value(self, state: torch.Tensor):
        self.reset()
        _, value, _ = self.optimize(state, calc_val=True, shift_steps=0)
        return value
