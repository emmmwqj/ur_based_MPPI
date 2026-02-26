#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
# Modified for DIAL-MPC diffusion-inspired sampling.
#

"""
DIAL-MPC Diffusion Simple Task

This module implements DiffusionSimpleTask for the simple_reacher environment,
demonstrating the full DIAL-MPC algorithm with Equation 7 variance scheduling.

DiffusionSimpleTask is a DROP-IN replacement for SimpleTask. It uses the same
ControlProcess, state filtering, and command flow — the only difference is 
DiffusionMPPI replaces MPPI, adding per-iteration variance annealing.

Usage:
    from storm_kit.mpc.task.diffusion_simple_task import DiffusionSimpleTask
    
    task = DiffusionSimpleTask(
        robot_file='simple_reacher.yml',
        diffusion_params={'beta_1': 1.0, 'beta_2': 1.0, 'n_diffuse': 4}
    )
"""

import torch
import yaml
import numpy as np

from ...util_file import get_mpc_configs_path as mpc_configs_path
from ...mpc.rollout.simple_reacher import SimpleReacher
from ...mpc.control.diffusion_mppi import DiffusionMPPI
from ...mpc.utils.state_filter import JointStateFilter
from ...mpc.utils.mpc_process_wrapper import ControlProcess
from ...util_file import get_assets_path, join_path, load_yaml, get_gym_configs_path
from .diffusion_task_base import DiffusionTaskBase


class DiffusionSimpleTask(DiffusionTaskBase):
    """
    Diffusion MPPI task for the simple_reacher environment.
    
    This is analogous to SimpleTask but uses DiffusionMPPI instead of MPPI.
    It is a DROP-IN replacement: same get_command interface, same 
    ControlProcess flow, same state filtering. The only change is the
    controller's optimize() adds diffusion variance scheduling.
    
    Args:
        robot_file: Path to robot configuration YAML
        diffusion_params: Dictionary with diffusion parameters
        tensor_args: Device and dtype settings
    """
    
    def __init__(self, 
                 robot_file='simple_reacher.yml',
                 diffusion_params=None,
                 tensor_args={'device': "cpu", 'dtype': torch.float32}):
        """Initialize DiffusionSimpleTask."""
        super().__init__(tensor_args=tensor_args)
        
        # Default diffusion parameters
        self.diffusion_params = {
            'beta_1': 1.0,
            'beta_2': 1.0,
            'n_diffuse': 4,
            'n_diffuse_init': 10,
            'sigma_base': 1.0
        }
        
        # Override with user-provided params
        if diffusion_params is not None:
            self.diffusion_params.update(diffusion_params)
            
        # Initialize controller (DiffusionMPPI)
        self.controller = self.init_diffusion_mppi(robot_file)
        # Initialize ControlProcess, state filters, etc. — same as SimpleTask
        self.init_aux()
        
    def get_rollout_fn(self, **kwargs):
        """Create rollout function for simple reacher."""
        rollout_fn = SimpleReacher(**kwargs)
        return rollout_fn
        
    def init_diffusion_mppi(self, robot_file):
        """
        Initialize DiffusionMPPI controller.
        
        This is identical to SimpleTask.init_mppi but creates a DiffusionMPPI
        controller with additional diffusion parameters.
        """
        # Load robot configuration
        mpc_yml_file = join_path(mpc_configs_path(), robot_file)
        
        with open(mpc_yml_file) as file:
            exp_params = yaml.safe_load(file)
            
        # Create rollout function
        rollout_fn = self.get_rollout_fn(
            exp_params=exp_params, 
            tensor_args=self.tensor_args
        )
        
        # Build MPPI parameters — identical to SimpleTask.init_mppi
        mppi_params = exp_params['mppi']
        dynamics_model = rollout_fn.dynamics_model
        
        mppi_params['d_action'] = dynamics_model.d_action
        mppi_params['action_lows'] = -exp_params['model']['max_action'] * \
            torch.ones(dynamics_model.d_action, **self.tensor_args)
        mppi_params['action_highs'] = exp_params['model']['max_action'] * \
            torch.ones(dynamics_model.d_action, **self.tensor_args)
            
        init_action = torch.zeros(
            (mppi_params['horizon'], dynamics_model.d_action), 
            **self.tensor_args
        )
        mppi_params['init_mean'] = init_action
        mppi_params['rollout_fn'] = rollout_fn
        mppi_params['tensor_args'] = self.tensor_args
        
        # Add diffusion parameters
        mppi_params['beta_1'] = self.diffusion_params['beta_1']
        mppi_params['beta_2'] = self.diffusion_params['beta_2']
        mppi_params['n_diffuse'] = self.diffusion_params['n_diffuse']
        mppi_params['n_diffuse_init'] = self.diffusion_params['n_diffuse_init']
        mppi_params['sigma_base'] = self.diffusion_params['sigma_base']
        
        # Create DiffusionMPPI controller
        controller = DiffusionMPPI(**mppi_params)
        
        self.exp_params = exp_params
        return controller
        
    def _state_to_tensor(self, state):
        """Convert state dict to tensor."""
        state_tensor = np.concatenate((
            state['position'], 
            state['velocity'], 
            state['acceleration']
        ))
        state_tensor = torch.tensor(state_tensor)
        return state_tensor
        
    def get_current_error(self, curr_state):
        """Get current tracking error."""
        state_tensor = self._state_to_tensor(curr_state).to(
            **self.controller.tensor_args
        ).unsqueeze(0)
        
        ee_error, _ = self.controller.rollout_fn.current_cost(state_tensor)
        ee_error = [x.detach().cpu().item() for x in ee_error]
        return ee_error
