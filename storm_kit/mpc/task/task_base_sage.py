#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#

import numpy as np
import torch

from ...mpc.utils.state_filter import JointStateFilter
from ...mpc.utils.mpc_process_wrapper_sage import ControlProcessSage


class BaseTaskSage:
    """
    SAGE-specific task base.

    This mirrors the original task base API, but keeps ``control_dt``
    synchronized across:
    - task-side state filters
    - SAGE-specific control process
    - final command integration
    """

    def __init__(self, tensor_args={"device": "cpu", "dtype": torch.float32}):
        self.tensor_args = tensor_args
        self.prev_qdd_des = None

    def init_aux(self):
        control_dt = float(self.exp_params["control_dt"])
        self.state_filter = JointStateFilter(
            filter_coeff=self.exp_params["state_filter_coeff"],
            dt=control_dt,
        )
        self.command_filter = JointStateFilter(
            filter_coeff=self.exp_params["cmd_filter_coeff"],
            dt=control_dt,
        )
        self.control_process = ControlProcessSage(
            self.controller,
            control_space=self.exp_params.get("control_space", "acc"),
            control_dt=control_dt,
        )
        self.n_dofs = self.controller.rollout_fn.dynamics_model.n_dofs
        self.zero_acc = np.zeros(self.n_dofs)

    def get_rollout_fn(self, **kwargs):
        raise NotImplementedError

    def init_mppi(self, **kwargs):
        raise NotImplementedError

    def update_params(self, **kwargs):
        self.controller.rollout_fn.update_params(**kwargs)
        self.control_process.update_params(**kwargs)
        return True

    def _sync_control_dt(self, control_dt):
        resolved_dt = float(self.exp_params["control_dt"] if control_dt is None else control_dt)
        self.state_filter.dt = resolved_dt
        self.command_filter.dt = resolved_dt
        self.control_process.control_dt = resolved_dt
        return resolved_dt

    def get_command(self, t_step, curr_state, control_dt, WAIT=False):
        resolved_dt = self._sync_control_dt(control_dt)

        if self.state_filter.cmd_joint_state is None:
            curr_state["velocity"] *= 0.0
        filt_state = self.state_filter.filter_joint_state(curr_state)
        state_tensor = self._state_to_tensor(filt_state)

        if WAIT:
            next_command, val, info, best_action = self.control_process.get_command_debug(
                t_step,
                state_tensor.numpy(),
                control_dt=resolved_dt,
            )
        else:
            next_command, val, info, best_action = self.control_process.get_command(
                t_step,
                state_tensor.numpy(),
                control_dt=resolved_dt,
            )

        qdd_des = next_command
        self.prev_qdd_des = qdd_des
        cmd_des = self.state_filter.integrate_acc(qdd_des, dt=resolved_dt)
        return cmd_des

    def _state_to_tensor(self, state):
        state_tensor = np.concatenate((state["position"], state["velocity"], state["acceleration"]))
        state_tensor = torch.tensor(state_tensor)
        return state_tensor

    def get_current_error(self, curr_state):
        state_tensor = self._state_to_tensor(curr_state).to(**self.controller.tensor_args).unsqueeze(0)
        ee_error, _ = self.controller.rollout_fn.current_cost(state_tensor)
        ee_error = [x.detach().cpu().item() for x in ee_error]
        return ee_error

    @property
    def mpc_dt(self):
        return self.control_process.mpc_dt

    @property
    def opt_dt(self):
        return self.control_process.opt_dt

    def close(self):
        self.control_process.close()

    @property
    def top_trajs(self):
        return self.control_process.top_trajs
