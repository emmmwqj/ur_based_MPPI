#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#

import copy
import time

import numpy as np
import torch
from torch.multiprocessing import Process, Queue

from ..utils.torch_utils import find_first_idx


class ControlProcessSage(object):
    """
    SAGE-specific ControlProcess variant.

    The legacy STORM wrapper keeps an internal ``self.control_dt`` that may
    diverge from the task-level ``control_dt`` passed into ``get_command``.
    This wrapper keeps the SAGE chain internally consistent without changing
    the original STORM files.
    """

    def __init__(self, controller, control_space="acc", control_dt=0.01):
        try:
            controller.rollout_fn.dynamics_model.robot_model.delete_lxml_objects()
        except Exception:
            pass

        torch.save(controller, "control_instance_sage.p")

        try:
            controller.rollout_fn.dynamics_model.robot_model.load_lxml_objects()
        except Exception:
            pass

        self.command = None
        self.current_state = None
        self.done = False
        self.opt_data = 0.0
        self.n_dofs = controller.rollout_fn.dynamics_model.n_dofs

        self.traj_tstep = copy.deepcopy(controller.rollout_fn.dynamics_model._traj_tstep.detach().cpu())
        self.command_tstep = self.traj_tstep
        self.mpc_dt = 0.0
        self.params = None
        self.top_trajs = None
        self.top_values = None
        self.top_idx = None
        self.control_space = control_space

        self.result_queue = Queue(maxsize=1)
        self.opt_queue = Queue(maxsize=1)
        self.opt_process = Process(
            target=optimize_process_sage,
            args=("control_instance_sage.p", self.opt_queue, self.result_queue),
        )
        self.opt_process.daemon = True
        self.opt_process.start()

        self.controller = controller
        self.control_dt = float(control_dt)
        self.prev_mpc_tstep = 0.0

    def _sync_control_dt(self, control_dt=None):
        if control_dt is not None:
            self.control_dt = float(control_dt)
        return self.control_dt

    def predict_next_state(self, t_step, curr_state):
        t1_idx = find_first_idx(self.command_tstep, t_step) - 1
        t2_idx = find_first_idx(self.command_tstep, t_step + self.mpc_dt)

        for i in range(t1_idx, t2_idx):
            command = self.command[0][i]
            curr_state = self.controller.rollout_fn.dynamics_model.get_next_state(
                curr_state,
                command,
                self.mpc_dt,
            )

        return curr_state

    def get_command_debug(self, t_step, curr_state, debug=False, control_dt=0.01):
        control_dt = self._sync_control_dt(control_dt)
        if self.command is not None:
            curr_state = self.predict_next_state(t_step, curr_state)

        current_state = np.append(curr_state, t_step + self.mpc_dt)
        shift_steps = find_first_idx(self.command_tstep, t_step + self.mpc_dt)
        state_tensor = torch.as_tensor(current_state, **self.controller.tensor_args).unsqueeze(0)

        mpc_time = time.time()
        command = list(self.controller.optimize(state_tensor, shift_steps=shift_steps))
        mpc_time = time.time() - mpc_time
        command[0] = command[0].cpu().numpy()
        self.command_tstep = self.traj_tstep + t_step

        self.opt_dt = mpc_time
        self.mpc_dt = t_step - self.prev_mpc_tstep
        self.prev_mpc_tstep = copy.deepcopy(t_step)

        self.top_idx = self.controller.top_idx
        self.top_values = self.controller.top_values
        self.top_trajs = self.controller.top_trajs
        self.command = command

        command_buffer, command_tstep_buffer = self.truncate_command(
            self.command[0], t_step, self.command_tstep
        )
        act = self.controller.rollout_fn.dynamics_model.integrate_action_step(
            command_buffer[0], control_dt
        )
        return act, command_tstep_buffer, self.command[1], command_buffer

    def get_command(self, t_step, curr_state, debug=False, control_dt=0.01):
        control_dt = self._sync_control_dt(control_dt)
        if self.opt_queue.empty():
            if self.command is not None:
                curr_state = self.predict_next_state(t_step, curr_state)

            curr_state = np.append(curr_state, t_step + self.mpc_dt)
            shift_steps = find_first_idx(self.command_tstep, t_step + self.mpc_dt)
            if shift_steps < 0:
                shift_steps = 0

            opt_data = {
                "state": curr_state,
                "t_step": t_step + self.mpc_dt,
                "done": self.done,
                "params": self.params,
                "shift_steps": shift_steps,
                "pred_mpc_dt": self.mpc_dt,
            }

            self.start_time = time.time()
            self.mpc_dt = t_step - self.prev_mpc_tstep
            self.opt_queue.put(opt_data)
            self.params = None

        while self.command is None and self.result_queue.empty():
            time.sleep(0.01)

        if not self.result_queue.empty():
            command_data = self.result_queue.get()
            self.command_tstep = self.traj_tstep + command_data["t_step"]
            self.command = command_data["command"]
            self.opt_dt = command_data["mpc_dt"]
            self.prev_mpc_tstep = copy.deepcopy(t_step)
            self.top_values = command_data["top_values"]
            self.top_trajs = command_data["top_trajs"]
            self.top_idx = command_data["top_idx"]

        if self.command is None:
            raise ValueError

        command_buffer, command_tstep_buffer = self.truncate_command(
            self.command[0], t_step, self.command_tstep
        )
        act = self.controller.rollout_fn.dynamics_model.integrate_action_step(
            command_buffer[0], control_dt
        )
        return act, command_tstep_buffer, self.command[1], command_buffer

    def truncate_command(self, command, trunc_tstep, command_tstep):
        f_idx = find_first_idx(command_tstep, trunc_tstep)
        if f_idx == -1:
            f_idx = 0
        return command[f_idx:], command_tstep[f_idx:]

    def update_params(self, **kwargs):
        self.params = kwargs

    def close(self):
        self.done = True
        opt_data = {"state": None, "dt": None, "done": self.done, "params": None}
        self.opt_queue.put(opt_data)
        self.opt_process.join()


def optimize_process_sage(control_string, opt_queue, result_queue):
    controller = torch.load(control_string, weights_only=False)
    try:
        controller.rollout_fn.dynamics_model.robot_model.load_lxml_objects()
    except Exception:
        pass

    start_time = time.time()
    while True:
        opt_data = opt_queue.get()
        if opt_data["done"]:
            break

        current_state = opt_data["state"]
        state_tensor = torch.as_tensor(current_state, **controller.tensor_args).unsqueeze(0)
        shift_steps = opt_data["shift_steps"]

        if opt_data["params"] is not None:
            controller.rollout_fn.update_params(**opt_data["params"])

        mpc_time = time.time()
        command = list(controller.optimize(state_tensor, shift_steps=shift_steps))
        mpc_time = time.time() - mpc_time

        top_idx = controller.top_idx
        top_values = controller.top_values
        top_trajs = controller.top_trajs

        command[0] = command[0].cpu().numpy()
        result = {
            "command": command,
            "t_step": opt_data["t_step"],
            "mpc_dt": mpc_time,
            "top_values": top_values,
            "top_trajs": top_trajs,
            "top_idx": top_idx,
        }
        result_queue.put(result)
        _ = time.time() - start_time
    return True
