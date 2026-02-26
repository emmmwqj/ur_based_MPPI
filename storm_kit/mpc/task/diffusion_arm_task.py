#!/usr/bin/env python
#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
# Modified for DIAL-MPC diffusion-inspired sampling.
#

"""
DIAL-MPC Diffusion Arm Task

This module implements DiffusionArmTask for robotic arm manipulation tasks,
demonstrating the full DIAL-MPC algorithm with Equation 7 variance scheduling.

This is designed for use with UR robots or other articulated arms, extending
STORM's ArmTask with DiffusionMPPI capabilities.

Usage:
    from storm_kit.mpc.task.diffusion_arm_task import DiffusionArmTask
    
    task = DiffusionArmTask(
        task_file='ur10.yml',
        robot_file='ur10_reacher.yml',
        world_file='collision_env.yml',
        diffusion_params={'beta_1': 1.0, 'beta_2': 1.0, 'n_diffuse': 4}
    )
"""

import torch
import yaml
import numpy as np

from ...util_file import get_mpc_configs_path as mpc_configs_path
from ...mpc.rollout.arm_reacher import ArmBase
from ...mpc.control.diffusion_mppi import DiffusionMPPI
from ...mpc.utils.state_filter import JointStateFilter
from ...mpc.utils.mpc_process_wrapper import ControlProcess
from ...util_file import get_assets_path, join_path, load_yaml, get_gym_configs_path
from .diffusion_task_base import DiffusionTaskBase


class DiffusionArmTask(DiffusionTaskBase):
    """
    Diffusion MPPI task for robotic arm manipulation.
    
    This is analogous to ArmTask but uses DiffusionMPPI instead of MPPI,
    implementing the full DIAL-MPC algorithm with Equation 7 variance scheduling.
    
    Can be used with any robotic arm supported by STORM (UR5, UR10, Franka, etc.).
    
    Example:
        >>> task = DiffusionArmTask(
        ...     task_file='ur10.yml',
        ...     robot_file='ur10_reacher.yml',
        ...     world_file='collision_env.yml',
        ...     diffusion_params={
        ...         'beta_1': 1.0,
        ...         'beta_2': 1.0,
        ...         'n_diffuse': 4,
        ...         'n_diffuse_init': 10
        ...     }
        ... )
        >>> task.update_params(goal_ee_pos=target_pos, goal_ee_quat=target_quat)
        >>> cmd = task.get_command(0.0, current_state, 0.02)
    
    Args:
        task_file: MPC configuration file (e.g., 'ur10.yml')
        robot_file: Robot configuration file (e.g., 'ur10_reacher.yml')
        world_file: Collision environment file (e.g., 'collision_env.yml')
        diffusion_params: Dictionary with diffusion parameters:
            - beta_1: Iteration annealing rate (default: 1.0)
            - beta_2: Horizon annealing rate (default: 1.0)
            - n_diffuse: Iterations per step (default: 4)
            - n_diffuse_init: Iterations for first step (default: 10)
            - sigma_base: Base sigma for variance (default: 1.0)
        tensor_args: Device and dtype settings
    """
    
    def __init__(self, 
                 task_file='ur10.yml',
                 robot_file='ur10_reacher.yml',
                 world_file='collision_env.yml',
                 diffusion_params=None,
                 tensor_args={'device': "cpu", 'dtype': torch.float32}):
        """Initialize DiffusionArmTask."""
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
            
        # Initialize controller
        self.controller = self.init_diffusion_mppi(task_file, robot_file, world_file)
        self.init_aux()
        
    def get_rollout_fn(self, **kwargs):
        """Create rollout function for arm reacher."""
        rollout_fn = ArmBase(**kwargs)
        return rollout_fn
        
    def init_diffusion_mppi(self, task_file, robot_file, collision_file):
        """
        Initialize DiffusionMPPI controller for arm task.
        
        This is similar to ArmTask.init_mppi but creates a DiffusionMPPI
        controller with additional diffusion parameters.
        """
        # Load robot configuration
        robot_yml = join_path(get_gym_configs_path(), robot_file)
        with open(robot_yml) as file:
            robot_params = yaml.safe_load(file)
            
        # Load world/collision configuration
        world_yml = join_path(get_gym_configs_path(), collision_file)
        with open(world_yml) as file:
            world_params = yaml.safe_load(file)
            
        # Load MPC configuration
        mpc_yml_file = join_path(mpc_configs_path(), task_file)
        with open(mpc_yml_file) as file:
            exp_params = yaml.safe_load(file)
            
        exp_params['robot_params'] = exp_params['model']
        
        # Create rollout function
        rollout_fn = self.get_rollout_fn(
            exp_params=exp_params,
            tensor_args=self.tensor_args,
            world_params=world_params
        )
        
        # Build MPPI parameters
        mppi_params = exp_params['mppi']
        dynamics_model = rollout_fn.dynamics_model
        
        mppi_params['d_action'] = dynamics_model.d_action
        mppi_params['action_lows'] = -exp_params['model']['max_acc'] * \
            torch.ones(dynamics_model.d_action, **self.tensor_args)
        mppi_params['action_highs'] = exp_params['model']['max_acc'] * \
            torch.ones(dynamics_model.d_action, **self.tensor_args)
            
        init_q = torch.tensor(exp_params['model']['init_state'], **self.tensor_args)
        init_action = torch.zeros(
            (mppi_params['horizon'], dynamics_model.d_action),
            **self.tensor_args
        )
        init_action[:, :] += init_q
        
        if exp_params['control_space'] == 'acc':
            mppi_params['init_mean'] = init_action * 0.0
        elif exp_params['control_space'] == 'pos':
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
        
    def update_params(self, **kwargs):
        """Update task parameters (e.g., goal pose)."""
        self.controller.rollout_fn.update_params(**kwargs)
        # Note: we bypass ControlProcess for diffusion optimization
        return True


class DiffusionReacherTask(DiffusionArmTask):
    """
    Diffusion MPPI task specifically for arm reaching tasks.
    
    This extends DiffusionArmTask with the ArmReacher rollout function
    for end-effector reaching tasks.
    
    Example:
        >>> task = DiffusionReacherTask(
        ...     task_file='ur10.yml',
        ...     robot_file='ur10_reacher.yml',
        ...     world_file='collision_env.yml',
        ...     diffusion_params={'beta_1': 1.0, 'beta_2': 1.0}
        ... )
    """
    
    def __init__(self,
                 task_file='ur10.yml',
                 robot_file='ur10_reacher.yml',
                 world_file='collision_env.yml',
                 diffusion_params=None,
                 tensor_args={'device': "cpu", 'dtype': torch.float32}):
        """Initialize DiffusionReacherTask."""
        super().__init__(
            task_file=task_file,
            robot_file=robot_file,
            world_file=world_file,
            diffusion_params=diffusion_params,
            tensor_args=tensor_args
        )
        
    def get_rollout_fn(self, **kwargs):
        """Create rollout function for arm reacher."""
        from ...mpc.rollout.arm_reacher import ArmReacher
        rollout_fn = ArmReacher(**kwargs)
        return rollout_fn
