#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
# Modified for DIAL-MPC diffusion-inspired sampling.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
DIAL-MPC Diffusion MPPI Controller

Implements DIAL-MPC (ICRA 2025) within STORM's MPPI framework.

Key Design Insight — faithfully following the original DIAL-MPC code:
  Original DIAL-MPC samples as:  Y0s = eps * noise_scale + Ybar
  where noise_scale is the DIRECT standard deviation, controlled externally
  by traj_diffuse_factor^i (geometric decay across iterations).
  Mean is updated via pure weighted average: Ybar = sum(w * Y0s).
  There is NO adaptive covariance — noise_scale is never updated from data.

This implementation fuses DIAL-MPC with STORM by:
  1. Using STORM's full sampling pipeline (Halton, B-spline, null particles)
     for the LAST diffusion iteration (i=0), preserving STORM's adaptive
     covariance and all convergence mechanisms.
  2. For earlier iterations (i>0), applying diffusion noise_scale DIRECTLY
     as the sampling std (like original DIAL-MPC), bypassing scale_tril.
     cov_action is NOT updated during these iterations.
  3. This ensures STORM's cov_action sees only "normal" scale samples and
     evolves correctly, while diffusion iterations provide coarse exploration.

Equation 7: σ_{i,h} = σ_base * exp(-(N-i)/(β₁N) - (H-h)/(β₂H))
"""

import copy
import math
import numpy as np
import torch
import torch.autograd.profiler as profiler

from .mppi import MPPI
from .control_utils import scale_ctrl, matrix_cholesky


class DiffusionMPPI(MPPI):
    """
    DIAL-MPC Diffusion MPPI Controller
    
    Extends STORM's MPPI with diffusion-style variance annealing.
    
    Early diffusion iterations (large noise_scale) explore broadly using
    DIAL-MPC's direct-noise sampling. The final iteration uses STORM's
    native sampling with adaptive covariance for local refinement.
    
    Attributes:
        beta_1 (float): Iteration-level annealing rate
        beta_2 (float): Horizon-level annealing rate
        sigma_base (float): Base standard deviation for Eq.7 schedule
        n_diffuse (int): Number of diffusion iterations per step
        n_diffuse_init (int): Number of iterations for first control step
    """
    
    def __init__(self,
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
                 # Diffusion-specific parameters
                 beta_1=1.0,
                 beta_2=1.0,
                 sigma_base=1.0,
                 n_diffuse=4,
                 n_diffuse_init=10,
                 execute_best=True,
                 # Standard MPPI parameters
                 null_act_frac=0.,
                 rollout_fn=None,
                 sample_mode='mean',
                 hotstart=True,
                 squash_fn='clamp',
                 update_cov=False,
                 cov_type='sigma_I',
                 seed=0,
                 sample_params={'type': 'halton', 'fixed_samples': True, 'seed': 0, 'filter_coeffs': None},
                 tensor_args={'device': torch.device('cpu'), 'dtype': torch.float32},
                 visual_traj='state_seq'):
        # Initialize parent MPPI
        super(DiffusionMPPI, self).__init__(
            d_action=d_action,
            horizon=horizon,
            init_cov=init_cov,
            init_mean=init_mean,
            base_action=base_action,
            beta=beta,
            num_particles=num_particles,
            step_size_mean=step_size_mean,
            step_size_cov=step_size_cov,
            alpha=alpha,
            gamma=gamma,
            kappa=kappa,
            n_iters=n_iters,
            action_lows=action_lows,
            action_highs=action_highs,
            null_act_frac=null_act_frac,
            rollout_fn=rollout_fn,
            sample_mode=sample_mode,
            hotstart=hotstart,
            squash_fn=squash_fn,
            update_cov=update_cov,
            cov_type=cov_type,
            seed=seed,
            sample_params=sample_params,
            tensor_args=tensor_args,
            visual_traj=visual_traj
        )
        
        # Diffusion-specific parameters
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.sigma_base = sigma_base
        self.n_diffuse = n_diffuse
        self.n_diffuse_init = n_diffuse_init
        self.execute_best = execute_best
        
        # Track if this is first optimization step
        self._is_first_step = True
        
        # Pre-compute horizon weights for efficiency
        self._precompute_horizon_weights()
            
    def _precompute_horizon_weights(self):
        """Pre-compute horizon-level annealing weights for efficiency."""
        H = self.horizon
        h_indices = torch.arange(1, H + 1, **self.tensor_args)  # h from 1 to H
        self._horizon_exponent = -(H - h_indices) / (self.beta_2 * H)
        
    def compute_variance_schedule(self, iteration, n_total):
        """
        Compute variance schedule per DIAL-MPC Equation 7.
        
        σ_{i,h} = σ_base * exp(-(N-i)/(β₁*N) - (H-h)/(β₂*H))
        
        Args:
            iteration: Current iteration i (N-1 down to 0)
            n_total: Total iterations N
            
        Returns:
            Tensor of shape (horizon,) with σ for each horizon step
        """
        iter_exponent = -(n_total - iteration) / (self.beta_1 * n_total)
        total_exponent = iter_exponent + self._horizon_exponent
        return self.sigma_base * torch.exp(total_exponent)

    def _diffusion_sample_actions(self, state, noise_scale):
        """
        Sample actions using DIAL-MPC's direct-noise approach.
        
        Like original DIAL-MPC: Y0s = eps * noise_scale + Ybar
        Uses STORM's Halton/B-spline sampler for eps, but applies
        noise_scale DIRECTLY as std instead of going through scale_tril.
        
        Args:
            state: Current state (unused, kept for API compatibility)
            noise_scale: Tensor (horizon,) — per-horizon-step std from Eq.7
            
        Returns:
            act_seq: Sampled action sequences (N+extras, horizon, d_action)
        """
        # Get base samples (standard normal via Halton + B-spline)
        delta = self.sample_lib.get_samples(
            sample_shape=self.sample_shape,
            base_seed=self.seed_val + self.num_steps
        )
        # Add zero-noise seq so mean is always a part of samples
        delta = torch.cat((delta, self.Z_seq), dim=0)
        
        # Apply noise_scale DIRECTLY per horizon step (like DIAL-MPC)
        # delta: (N, H, A), noise_scale: (H,) → broadcast to (1, H, 1)
        scaled_delta = delta * noise_scale.unsqueeze(0).unsqueeze(-1)
        
        # Add mean action
        act_seq = self.mean_action.unsqueeze(0) + scaled_delta
        
        # Clamp to action bounds
        act_seq = scale_ctrl(act_seq, self.action_lows, self.action_highs,
                             squash_fn=self.squash_fn)
        
        # Append best_traj, null particles, negative particles
        append_acts = self.best_traj.unsqueeze(0)
        if self.num_null_particles > 0:
            neg_action = -1.0 * self.mean_action.unsqueeze(0)
            neg_act_seqs = neg_action.expand(self.num_neg_particles, -1, -1)
            append_acts = torch.cat(
                (append_acts, self.null_act_seqs, neg_act_seqs), dim=0
            )
        act_seq = torch.cat((act_seq, append_acts), dim=0)
        return act_seq

    def _diffusion_update_mean(self, trajectories):
        """
        Update mean_action only (no cov update) — like DIAL-MPC's weighted avg.
        
        Uses STORM's MPPI weighting (temperature, cost normalization) but
        only updates mean_action, leaving cov_action untouched.
        Also updates best_traj and top_trajs for diagnostics.
        """
        costs = trajectories["costs"].to(**self.tensor_args)
        vis_seq = trajectories[self.visual_traj].to(**self.tensor_args)
        actions = trajectories["actions"].to(**self.tensor_args)
        
        # Compute MPPI weights (same as parent)
        w = self._exp_util(costs, actions)
        
        # Update best action
        best_idx = torch.argmax(w)
        self.best_idx = best_idx
        self.best_traj = torch.index_select(actions, 0, best_idx).squeeze(0)
        
        top_values, top_idx = torch.topk(self.total_costs, 10)
        self.top_values = top_values
        self.top_idx = top_idx
        self.top_trajs = torch.index_select(vis_seq, 0, top_idx).squeeze(0)
        
        # Weighted mean of actions
        weighted_seq = w.unsqueeze(-1).unsqueeze(-1) * actions
        new_mean = torch.sum(weighted_seq, dim=0)
        
        # Blend with current mean (STORM's step_size_mean)
        self.mean_action = (1.0 - self.step_size_mean) * self.mean_action + \
            self.step_size_mean * new_mean
            
    def optimize(self, state, calc_val=False, shift_steps=1, n_iters=None):
        """
        DIAL-MPC + STORM hybrid optimization.
        
        Flow:
        1. _shift (hotstart + kappa covariance growth) — STORM standard
        2. For iterations i = N-1 down to 1 (diffusion exploration):
           - Sample with DIAL-MPC noise_scale (Eq.7) directly as std
           - Update mean_action only (no cov update)
        3. Final iteration i=0 (STORM refinement):
           - Use STORM's native sample_actions with current scale_tril
           - Full _update_distribution (mean + cov update if enabled)
        
        This preserves STORM's adaptive covariance evolution while adding
        DIAL-MPC's coarse-to-fine exploration.
        """
        # Determine number of iterations
        if self._is_first_step:
            n_total = self.n_diffuse_init
        else:
            n_total = self.n_diffuse
        
        inp_device = state.device
        inp_dtype = state.dtype
        state = state.to(**self.tensor_args)

        info = dict(
            rollout_time=0.0, 
            entropy=[],
            iteration_costs=[],
            variance_schedule=[]
        )
        
        # Shift distribution (hotstart: shift mean forward, grow cov via kappa)
        if self.hotstart:
            self._shift(shift_steps)
        else:
            self.reset_distribution()

        with torch.amp.autocast('cuda', enabled=True):
            with torch.no_grad():
                
                # Track global best across ALL iterations
                global_best_cost = float('inf')
                global_best_traj = None
                
                # ── Phase 1: Diffusion iterations (i = N-1 down to 1) ──
                # Use DIAL-MPC direct-noise sampling, update mean only
                for iter_idx in range(n_total - 1, 0, -1):
                    # Compute noise_scale from Eq.7
                    noise_scale = self.compute_variance_schedule(iter_idx, n_total)
                    
                    info['variance_schedule'].append(noise_scale.mean().item())
                    
                    # Sample with diffusion noise directly
                    act_seq = self._diffusion_sample_actions(state, noise_scale)
                    
                    # Rollout
                    trajectory = self._rollout_fn(state, act_seq)
                    
                    # Update mean only (like DIAL-MPC), no cov update
                    self._diffusion_update_mean(trajectory)
                    
                    info['rollout_time'] += trajectory['rollout_time']
                    iter_min_cost = self.total_costs.min().item()
                    info['iteration_costs'].append(iter_min_cost)
                    
                    # Update global best
                    if iter_min_cost < global_best_cost:
                        global_best_cost = iter_min_cost
                        global_best_traj = self.best_traj.clone()
                
                # ── Phase 2: Final STORM iteration (i = 0) ──
                # Use STORM's native sampling with current scale_tril/cov_action
                # This is a standard STORM MPPI iteration
                noise_scale_last = self.compute_variance_schedule(0, n_total)
                info['variance_schedule'].append(noise_scale_last.mean().item())
                
                trajectory = self.generate_rollouts(state)
                
                with profiler.record_function("mppi_update"):
                    self._update_distribution(trajectory)
                
                info['rollout_time'] += trajectory['rollout_time']
                iter_min_cost = self.total_costs.min().item()
                info['iteration_costs'].append(iter_min_cost)
                
                # Update global best with Phase 2 result
                if iter_min_cost < global_best_cost:
                    global_best_cost = iter_min_cost
                    global_best_traj = self.best_traj.clone()

        self.trajectories = trajectory
        
        # Choose execution mode based on execute_best flag
        # mean_action is always updated via weighted average for next step's sampling center
        if self.execute_best and global_best_traj is not None:
            # Execute the global best particle (lowest cost across all iterations)
            curr_action_seq = global_best_traj
        else:
            # Execute the weighted-average mean_action (original STORM behavior)
            curr_action_seq = self._get_action_seq(mode=self.sample_mode)
        
        value = 0.0
        if calc_val:
            trajectories = self.generate_rollouts(state)
            value = self._calc_val(trajectories)

        info['entropy'].append(self.entropy)
        self.num_steps += 1
        self._is_first_step = False

        return curr_action_seq.to(inp_device, dtype=inp_dtype), value, info
        
    def reset(self):
        """Reset controller state."""
        super().reset()
        self._is_first_step = True
        
    def get_diffusion_info(self):
        """Get information about the diffusion configuration."""
        sample_schedule_init = []
        for i in range(self.n_diffuse_init - 1, -1, -1):
            avg_var = self.compute_variance_schedule(i, self.n_diffuse_init).mean().item()
            sample_schedule_init.append(avg_var)
            
        sample_schedule_normal = []
        for i in range(self.n_diffuse - 1, -1, -1):
            avg_var = self.compute_variance_schedule(i, self.n_diffuse).mean().item()
            sample_schedule_normal.append(avg_var)
            
        return {
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'sigma_base': self.sigma_base,
            'n_diffuse': self.n_diffuse,
            'n_diffuse_init': self.n_diffuse_init,
            'horizon': self.horizon,
            'cov_type': self.cov_type,
            'init_schedule_preview': sample_schedule_init,
            'normal_schedule_preview': sample_schedule_normal
        }
