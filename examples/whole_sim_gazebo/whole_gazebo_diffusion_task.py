import queue

import torch
import yaml

from storm_kit.mpc.control import DiffusionMPPI
from storm_kit.mpc.task.diffusion_task_base import DiffusionTaskBase

from examples.whole_sim_gazebo.arm_reacher_esdf import ArmReacherESDF


def _log(message: str) -> None:
    print(message, flush=True)


def _drain_mp_queue(mp_queue) -> None:
    if mp_queue is None:
        return
    while True:
        try:
            mp_queue.get_nowait()
        except queue.Empty:
            break
        except Exception:
            break


def _close_mp_queue(mp_queue) -> None:
    for method_name in ('close', 'cancel_join_thread'):
        method = getattr(mp_queue, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass


def _shutdown_control_process(control_process, join_timeout: float = 2.0) -> None:
    if control_process is None:
        return

    control_process.done = True
    _drain_mp_queue(getattr(control_process, 'result_queue', None))

    done_message = {'state': None, 'dt': None, 'done': True, 'params': None}
    opt_queue = getattr(control_process, 'opt_queue', None)
    if opt_queue is not None:
        try:
            opt_queue.put_nowait(done_message)
        except queue.Full:
            _drain_mp_queue(opt_queue)
            try:
                opt_queue.put_nowait(done_message)
            except Exception:
                pass
        except Exception:
            pass

    opt_process = getattr(control_process, 'opt_process', None)
    if opt_process is not None:
        opt_process.join(timeout=join_timeout)
        if opt_process.is_alive():
            _log('后台 Diffusion MPC 进程未在超时内退出，强制终止...')
            opt_process.terminate()
            opt_process.join(timeout=join_timeout)

    if opt_queue is not None:
        _close_mp_queue(opt_queue)
    result_queue = getattr(control_process, 'result_queue', None)
    if result_queue is not None:
        _close_mp_queue(result_queue)


class WholeGazeboDiffusionReacherTask(DiffusionTaskBase):
    def __init__(self, task_file, robot_file, world_file, tensor_args):
        super().__init__(tensor_args=tensor_args)
        self.diffusion_params = {
            'beta_1': 1.0,
            'beta_2': 1.0,
            'n_diffuse': 4,
            'n_diffuse_init': 8,
            'sigma_base': 0.25,
            'execute_best': False,
        }
        self.controller = self.init_diffusion_mppi(task_file, robot_file, world_file)
        self.init_aux()

    def get_rollout_fn(self, **kwargs):
        return ArmReacherESDF(**kwargs)

    def init_diffusion_mppi(self, task_file, robot_file, world_file):
        with open(robot_file) as f:
            robot_params = yaml.safe_load(f)
        with open(world_file) as f:
            world_params = yaml.safe_load(f)
        with open(task_file) as f:
            exp_params = yaml.safe_load(f)

        exp_params['robot_params'] = exp_params['model']
        self.diffusion_params.update(exp_params.get('diffusion', {}))

        rollout_fn = self.get_rollout_fn(
            exp_params=exp_params,
            tensor_args=self.tensor_args,
            world_params=world_params,
        )

        mppi_params = exp_params['mppi']
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
        _log('[WholeGazeboDiffusionReacherTask] Controller summary:')
        _log('  controller_type            = DiffusionMPPI')
        _log(
            '  primitive_collision.weight = %.1f'
            % float(exp_params['cost']['primitive_collision']['weight'])
        )
        _log(
            '  voxel_collision.weight     = %.1f'
            % float(exp_params['cost']['voxel_collision']['weight'])
        )
        _log(
            '  esdf_collision.weight      = %.1f'
            % float(exp_params['cost']['esdf_collision']['weight'])
        )
        _log(
            '  esdf_snapshot_path         = %s'
            % world_params['world_model']['esdf_snapshot_path']
        )
        _log('  environment_collision      = ESDF snapshot')
        _log('  diffusion.beta_1           = %.3f' % self.diffusion_params['beta_1'])
        _log('  diffusion.beta_2           = %.3f' % self.diffusion_params['beta_2'])
        _log('  diffusion.sigma_base       = %.3f' % self.diffusion_params['sigma_base'])
        _log('  diffusion.n_diffuse        = %d' % self.diffusion_params['n_diffuse'])
        _log('  diffusion.n_diffuse_init   = %d' % self.diffusion_params['n_diffuse_init'])
        _log('  diffusion.execute_best     = %s' % self.diffusion_params['execute_best'])
        _log('  diffusion.schedule_preview = %s' % preview)
        return controller

    def close(self):
        control_process = getattr(self, 'control_process', None)
        _shutdown_control_process(control_process)
