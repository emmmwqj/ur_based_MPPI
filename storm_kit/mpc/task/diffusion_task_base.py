#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
# Modified for DIAL-MPC diffusion-inspired sampling.
#

"""
DIAL-MPC Diffusion Task Base

This module provides the base task class for DIAL-MPC style controllers.
It extends STORM's BaseTask with diffusion-specific functionality while
preserving ALL of STORM's engineering techniques for convergence:

- ControlProcess: async optimization with state prediction and time-based
  command truncation
- JointStateFilter: state filtering and acceleration integration
- Proper shift_steps calculation via find_first_idx
- Null particles for coasting/stopping at goal
- B-spline smoothed Halton-knot sampling
"""

import torch
import yaml
import numpy as np
import copy

from ..utils.state_filter import JointStateFilter
from ..utils.mpc_process_wrapper import ControlProcess
from ...util_file import get_mpc_configs_path as mpc_configs_path
from ...util_file import get_assets_path, join_path, load_yaml, get_gym_configs_path
from .task_base import BaseTask


class DiffusionTaskBase(BaseTask):
    """
    Base class for tasks using DIAL-MPC Diffusion MPPI.
    
    This class extends BaseTask using the SAME architecture as STORM:
    - Uses ControlProcess for optimization (handles state prediction,
      time-based command truncation, shift_steps calculation)
    - DiffusionMPPI's optimize() is called by ControlProcess just like
      MPPI's optimize() is called for standard tasks
    - Tracks diffusion optimization statistics
    
    The key insight: DiffusionMPPI is a DROP-IN replacement for MPPI.
    Its optimize() method follows the exact same interface, so it works
    seamlessly with ControlProcess.
    
    Attributes:
        controller: DiffusionMPPI controller instance
        diffusion_stats: Statistics from diffusion optimization
    """
    
    def __init__(self, tensor_args={'device': "cpu", 'dtype': torch.float32}):
        """Initialize DiffusionTaskBase."""
        super().__init__(tensor_args=tensor_args)
        self.diffusion_stats = {
            'iteration_costs': [],
            'variance_schedules': [],
            'total_steps': 0
        }
        
    def init_aux(self):
        """Initialize auxiliary components.
        
        Uses the SAME init_aux as BaseTask, including ControlProcess.
        DiffusionMPPI is a drop-in replacement for MPPI, so ControlProcess
        works with it identically.
        """
        # Same as BaseTask.init_aux() - creates ControlProcess, state filters, etc.
        self.state_filter = JointStateFilter(
            filter_coeff=self.exp_params['state_filter_coeff'], 
            dt=self.exp_params['control_dt']
        )
        self.command_filter = JointStateFilter(
            filter_coeff=self.exp_params['cmd_filter_coeff'],
            dt=self.exp_params['control_dt']
        )
        # ControlProcess handles async optimization, state prediction,
        # time-based command truncation, and shift_steps calculation.
        # DiffusionMPPI.optimize() has the same interface as MPPI.optimize(),
        # so ControlProcess works with it seamlessly.
        self.control_process = ControlProcess(self.controller)
        self.n_dofs = self.controller.rollout_fn.dynamics_model.n_dofs
        self.zero_acc = np.zeros(self.n_dofs)

    def get_command(self, t_step, curr_state, control_dt, WAIT=False):
        """
        Get command using STORM's standard flow with DiffusionMPPI.
        
        This is IDENTICAL to BaseTask.get_command(). The diffusion variance
        scheduling happens inside DiffusionMPPI.optimize(), which is called
        by ControlProcess transparently.
        
        Args:
            t_step: Current time step
            curr_state: Current state dictionary  
            control_dt: Control timestep
            WAIT: Whether to wait for optimization to complete
            
        Returns:
            cmd_des: Command dictionary with position, velocity, acceleration
        """
        # Filter state - same as BaseTask
        if self.state_filter.cmd_joint_state is None:
            curr_state['velocity'] *= 0.0
        filt_state = self.state_filter.filter_joint_state(curr_state)
        state_tensor = self._state_to_tensor(filt_state)

        # Use ControlProcess for optimization - handles state prediction,
        # timestamp appending, shift_steps, command truncation
        if WAIT:
            next_command, val, info, best_action = self.control_process.get_command_debug(
                t_step, state_tensor.numpy(), control_dt=control_dt
            )
        else:
            next_command, val, info, best_action = self.control_process.get_command(
                t_step, state_tensor.numpy(), control_dt=control_dt
            )

        qdd_des = next_command
        self.prev_qdd_des = qdd_des
        cmd_des = self.state_filter.integrate_acc(qdd_des)
        
        # Track diffusion stats
        self.diffusion_stats['total_steps'] += 1
        
        # Expose optimization info for diagnostics
        # control_process.command = [action_seq, value, info_dict]
        if hasattr(self.control_process, 'command') and self.control_process.command is not None:
            opt_info = self.control_process.command[2] if len(self.control_process.command) > 2 else {}
        else:
            opt_info = {}
        self._last_opt_info = opt_info
        
        # Record scale_tril (STORM adaptive noise level)
        if hasattr(self.controller, 'scale_tril'):
            st = self.controller.scale_tril
            self._last_scale_tril = st.mean().item() if torch.is_tensor(st) else float(st)
        else:
            self._last_scale_tril = 0.0

        return cmd_des
        
    def update_params(self, **kwargs):
        """Update task parameters (e.g., goal state).
        
        Updates both the controller's rollout_fn AND the ControlProcess,
        same as BaseTask.
        """
        self.controller.rollout_fn.update_params(**kwargs)
        self.control_process.update_params(**kwargs)
        return True

    def reset(self):
        """Reset the task state for a new episode."""
        self.controller.reset()
        self.state_filter.cmd_joint_state = None
        self.command_filter.cmd_joint_state = None
        self.diffusion_stats = {
            'iteration_costs': [],
            'variance_schedules': [],
            'total_steps': 0
        }
        
    def get_diffusion_info(self):
        """
        Get diffusion configuration and statistics.
        
        Returns:
            dict: Combined controller config and runtime stats
        """
        info = self.controller.get_diffusion_info()
        info['runtime_stats'] = self.diffusion_stats
        return info
        
    def print_diffusion_summary(self):
        """Print a summary of the diffusion optimization."""
        info = self.get_diffusion_info()
        print("\n=== DIAL-MPC Diffusion Summary ===")
        print(f"β₁ (iteration annealing): {info['beta_1']}")
        print(f"β₂ (horizon annealing): {info['beta_2']}")
        print(f"σ_base: {info['sigma_base']}")
        print(f"n_diffuse: {info['n_diffuse']}")
        print(f"n_diffuse_init: {info['n_diffuse_init']}")
        print(f"Horizon: {info['horizon']}")
        print(f"Covariance type: {info['cov_type']}")
        print(f"Total control steps: {info['runtime_stats']['total_steps']}")

    @property
    def mpc_dt(self):
        return self.control_process.mpc_dt
    
    @property
    def opt_dt(self):
        return self.control_process.opt_dt
    
    def close(self):
        """Close the ControlProcess."""
        self.control_process.close()
        
    @property
    def top_trajs(self):
        """Get top trajectories from last optimization."""
        return self.control_process.top_trajs