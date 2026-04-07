#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#

"""
Clean SAGE controller variant for rollout-native safety margins and ablation.

This file intentionally leaves the original `sage_mppi.py` untouched. It builds
on top of the existing SAGE implementation, but makes two paper-alignment
changes:

1. Prefer rollout-native `delta_safe` / `safety_margin_seq` when available.
   This is the ideal, rollout-semantic path.
2. Expose the three SAGE core ideas as explicit ablation-ready switches:
   - stage scale
   - safe-elite anisotropic shape
   - stagnation amplification

If a rollout does not provide native safety-margin outputs, this controller
falls back to the legacy compatibility path implemented in `sage_mppi.py`.
"""

from __future__ import annotations

from typing import Dict

import torch

from .sage_mppi import SAGE_MPPI


class SAGE_MPPI_CLEAN(SAGE_MPPI):
    """
    Paper-aligned SAGE controller variant with explicit ablation switches.
    """

    def __init__(
        self,
        *args,
        enable_stage_scale=True,
        enable_safe_elite_shape=True,
        enable_stagnation_amplification=True,
        prefer_rollout_native_margin=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_stage_scale = bool(enable_stage_scale)
        self.enable_safe_elite_shape = bool(enable_safe_elite_shape)
        self.enable_stagnation_amplification = bool(enable_stagnation_amplification)
        self.prefer_rollout_native_margin = bool(prefer_rollout_native_margin)

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

            # Preferred, paper-aligned path: consume rollout-native signed
            # safety margins directly from the rollout outputs.
            native_margin_ready = any(
                key in rollout for key in ("delta_safe", "safety_margin_seq", "collision_margin_seq")
            )
            if native_margin_ready:
                return rollout

            # Compatibility path: if the rollout already exposes `state_dict`,
            # keep it and allow the legacy controller fallback to reconstruct
            # the safety margin from the trusted collision modules.
            if "state_dict" in rollout and "costs" in rollout:
                return rollout

        return super()._build_rollout_dict(state, act_seq)

    def _compute_rollout_safety_margin(
        self,
        rollout_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Preferred, paper-aligned path: consume rollout-native signed safety
        # margins emitted directly by the rollout.
        if "delta_safe" in rollout_dict:
            self._used_margin_fallback = False
            return rollout_dict["delta_safe"].to(**self.tensor_args)

        if "safety_margin_seq" in rollout_dict:
            self._used_margin_fallback = False
            safety_margin_seq = rollout_dict["safety_margin_seq"].to(**self.tensor_args)
            return safety_margin_seq.amin(dim=-1)

        if "collision_margin_seq" in rollout_dict:
            self._used_margin_fallback = False
            collision_margin_seq = rollout_dict["collision_margin_seq"].to(**self.tensor_args)
            return collision_margin_seq.amin(dim=-1)

        # Backward-compatible fallback: reconstruct the signed safety margin
        # from rollout state tensors and collision modules, as implemented in
        # the legacy `sage_mppi.py`.
        return super()._compute_rollout_safety_margin(rollout_dict)

    def _compute_stage_scale(
        self,
        iter_idx: int,
        n_total: int,
        stagnated: bool,
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

        if self.enable_stagnation_amplification and stagnated:
            stage_scale = (1.0 + self.stagnation_alpha) * stage_scale
        return stage_scale

    def _compute_safe_elite_covariance(
        self,
        actions: torch.Tensor,
        proposal_mean: torch.Tensor,
        weights: torch.Tensor,
        safe_mask: torch.Tensor,
        iter_idx: int,
        n_total: int,
    ):
        if self.enable_safe_elite_shape:
            return super()._compute_safe_elite_covariance(
                actions=actions,
                proposal_mean=proposal_mean,
                weights=weights,
                safe_mask=safe_mask,
                iter_idx=iter_idx,
                n_total=n_total,
            )

        identity_shape = self.I.unsqueeze(0).repeat(actions.shape[1], 1, 1)
        safe_weight_mass = torch.sum(weights * safe_mask.to(weights.dtype))
        self._last_safe_weight_mass = float(safe_weight_mass.item())
        self._last_safe_elite_fraction = float(
            safe_mask.to(torch.float32).mean().item()
        )
        return identity_shape, safe_weight_mass, 0.0, False

    def optimize(
        self,
        state: torch.Tensor,
        calc_val: bool = False,
        shift_steps: int = 1,
        n_iters=None,
    ):
        action, value, info = super().optimize(
            state=state,
            calc_val=calc_val,
            shift_steps=shift_steps,
            n_iters=n_iters,
        )
        info["controller_core"] = {
            "enable_stage_scale": self.enable_stage_scale,
            "enable_safe_elite_shape": self.enable_safe_elite_shape,
            "enable_stagnation_amplification": self.enable_stagnation_amplification,
            "prefer_rollout_native_margin": self.prefer_rollout_native_margin,
        }
        self.latest_stats.update(info["controller_core"])
        return action, value, info
