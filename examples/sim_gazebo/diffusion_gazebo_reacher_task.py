#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
import yaml

from storm_kit.mpc.control import DiffusionMPPI
from storm_kit.mpc.rollout.arm_reacher import ArmReacher
from storm_kit.mpc.task.diffusion_task_base import DiffusionTaskBase
from storm_kit.util_file import get_gym_configs_path, get_mpc_configs_path, join_path


def _log(message: str) -> None:
    print(message, flush=True)


class DiffusionGazeboReacherTask(DiffusionTaskBase):
    """Gazebo tall-scene reach task using DiffusionMPPI and primitive world collision."""

    def __init__(self, task_file, robot_file, world_file, tensor_args):
        super().__init__(tensor_args=tensor_args)
        self.diffusion_params = {
            'beta_1': 1.0,
            'beta_2': 1.0,
            'n_diffuse': 4,
            'n_diffuse_init': 8,
            'sigma_base': 0.45,
            'execute_best': True,
        }
        self.controller = self.init_diffusion_mppi(task_file, robot_file, world_file)
        self.init_aux()

    def get_rollout_fn(self, **kwargs):
        return ArmReacher(**kwargs)

    def _resolve_yaml(self, path, base_dir_getter):
        if os.path.isabs(path):
            return path
        return join_path(base_dir_getter(), path)

    def init_diffusion_mppi(self, task_file, robot_file, world_file):
        robot_yml = self._resolve_yaml(robot_file, get_gym_configs_path)
        world_yml = self._resolve_yaml(world_file, get_gym_configs_path)
        task_yml = self._resolve_yaml(task_file, get_mpc_configs_path)

        with open(robot_yml) as f:
            robot_params = yaml.safe_load(f)
        with open(world_yml) as f:
            world_params = yaml.safe_load(f)
        with open(task_yml) as f:
            exp_params = yaml.safe_load(f)

        exp_params['robot_params'] = exp_params['model']
        self.diffusion_params.update(exp_params.get('diffusion', {}))

        rollout_fn = self.get_rollout_fn(
            exp_params=exp_params,
            tensor_args=self.tensor_args,
            world_params=world_params,
        )

        mppi_params = exp_params['mppi']
        self.runtime_overrides = {}
        if float(mppi_params.get('step_size_mean', 0.0)) > 0.35:
            self.runtime_overrides['step_size_mean'] = (
                float(mppi_params['step_size_mean']),
                0.35,
            )
            mppi_params['step_size_mean'] = 0.35
        dynamics_model = rollout_fn.dynamics_model
        mppi_params['d_action'] = dynamics_model.d_action
        mppi_params['action_lows'] = -exp_params['model']['max_acc'] * torch.ones(
            dynamics_model.d_action,
            **self.tensor_args,
        )
        mppi_params['action_highs'] = exp_params['model']['max_acc'] * torch.ones(
            dynamics_model.d_action,
            **self.tensor_args,
        )

        init_q = torch.tensor(exp_params['model']['init_state'], **self.tensor_args)
        init_action = torch.zeros(
            (mppi_params['horizon'], dynamics_model.d_action),
            **self.tensor_args,
        )
        init_action[:, :] += init_q
        if exp_params['control_space'] == 'acc':
            mppi_params['init_mean'] = init_action * 0.0
        elif exp_params['control_space'] == 'pos':
            mppi_params['init_mean'] = init_action

        mppi_params['rollout_fn'] = rollout_fn
        mppi_params['tensor_args'] = self.tensor_args
        mppi_params['beta_1'] = self.diffusion_params['beta_1']
        mppi_params['beta_2'] = self.diffusion_params['beta_2']
        mppi_params['n_diffuse'] = self.diffusion_params['n_diffuse']
        mppi_params['n_diffuse_init'] = self.diffusion_params['n_diffuse_init']
        mppi_params['sigma_base'] = self.diffusion_params['sigma_base']
        mppi_params['execute_best'] = self.diffusion_params['execute_best']

        controller = DiffusionMPPI(**mppi_params)
        self.exp_params = exp_params
        self.robot_params = robot_params
        self.world_params = world_params

        diff_info = controller.get_diffusion_info()
        preview = [round(v, 4) for v in diff_info['normal_schedule_preview']]
        _log('[DiffusionGazeboReacherTask] Controller summary:')
        _log('  controller_type             = DiffusionMPPI')
        _log('  environment_collision       = primitive world')
        _log('  task_config                 = %s' % task_yml)
        _log('  world_config                = %s' % world_yml)
        _log(
            '  primitive_collision.weight  = %.1f'
            % float(exp_params['cost']['primitive_collision']['weight'])
        )
        _log(
            '  robot_self_collision.weight = %.1f'
            % float(exp_params['cost']['robot_self_collision']['weight'])
        )
        _log('  goal_pose.weight            = %s' % exp_params['cost']['goal_pose']['weight'])
        _log('  diffusion.beta_1            = %.3f' % self.diffusion_params['beta_1'])
        _log('  diffusion.beta_2            = %.3f' % self.diffusion_params['beta_2'])
        _log('  diffusion.sigma_base        = %.3f' % self.diffusion_params['sigma_base'])
        _log('  diffusion.n_diffuse         = %d' % self.diffusion_params['n_diffuse'])
        _log('  diffusion.n_diffuse_init    = %d' % self.diffusion_params['n_diffuse_init'])
        _log('  diffusion.execute_best      = %s' % self.diffusion_params['execute_best'])
        _log('  diffusion.schedule_preview  = %s' % preview)
        if 'step_size_mean' in self.runtime_overrides:
            old_value, new_value = self.runtime_overrides['step_size_mean']
            _log('  override.step_size_mean     = %.3f -> %.3f' % (old_value, new_value))
        return controller
