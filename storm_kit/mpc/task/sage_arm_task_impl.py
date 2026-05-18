#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#

import os

import numpy as np
import torch
import yaml

from ...util_file import get_gym_configs_path, get_mpc_configs_path, join_path
from ..control.sage_mppi import SAGE_MPPI
from ..rollout.sage_arm_base import SageArmBase
from ..utils.state_filter import JointStateFilter
from ..utils.mpc_process_wrapper_sage import ControlProcessSage
from .task_base_sage import BaseTaskSage


class SageArmTaskV3(BaseTaskSage):
    """
    Final clean SAGE task assembly.

    This task is the intended clean pipeline entry:
    - independent clean controller core
    - controller/deployment config groups kept separate
    """

    def __init__(
        self,
        task_file="ur10.yml",
        robot_file="ur10_reacher.yml",
        world_file="collision_env.yml",
        tensor_args={"device": "cpu", "dtype": torch.float32},
    ):
        super().__init__(tensor_args=tensor_args)
        self.latest_task_stats = {}
        self.success_threshold = None
        self.controller_core_config = {}
        self.deployment_refinement_config = {}
        self.controller = self.init_sage_mppi(task_file, robot_file, world_file)
        self.init_aux()

    def _resolve_yaml(self, path, base_dir_getter):
        if os.path.isabs(path):
            return path
        return join_path(base_dir_getter(), path)

    def get_rollout_fn(self, **kwargs):
        return SageArmBase(**kwargs)

    def _build_controller_params(self, exp_params, rollout_fn):
        mppi_params = dict(exp_params["mppi"])
        controller_core = dict(exp_params.get("sage_controller_core", {}))
        mppi_params.pop("execution_mode", None)

        mppi_params.update(controller_core)
        dynamics_model = rollout_fn.dynamics_model
        mppi_params["d_action"] = dynamics_model.d_action
        mppi_params["action_lows"] = -exp_params["model"]["max_acc"] * torch.ones(
            dynamics_model.d_action,
            **self.tensor_args,
        )
        mppi_params["action_highs"] = exp_params["model"]["max_acc"] * torch.ones(
            dynamics_model.d_action,
            **self.tensor_args,
        )

        init_q = torch.tensor(exp_params["model"]["init_state"], **self.tensor_args)
        init_action = torch.zeros(
            (mppi_params["horizon"], dynamics_model.d_action),
            **self.tensor_args,
        )
        init_action[:, :] += init_q
        if exp_params["control_space"] == "acc":
            mppi_params["init_mean"] = init_action * 0.0
        elif exp_params["control_space"] == "pos":
            mppi_params["init_mean"] = init_action
        else:
            raise ValueError(
                f"Unsupported control_space for SageArmTaskV3: {exp_params['control_space']}"
            )

        mppi_params["rollout_fn"] = rollout_fn
        mppi_params["tensor_args"] = self.tensor_args
        self.controller_core_config = controller_core
        self.deployment_refinement_config = dict(
            exp_params.get("sage_deployment_refinement", {})
        )
        return mppi_params

    def init_sage_mppi(self, task_file, robot_file, world_file):
        robot_yml = self._resolve_yaml(robot_file, get_gym_configs_path)
        world_yml = self._resolve_yaml(world_file, get_gym_configs_path)
        task_yml = self._resolve_yaml(task_file, get_mpc_configs_path)

        with open(robot_yml) as f:
            robot_params = yaml.safe_load(f)
        with open(world_yml) as f:
            world_params = yaml.safe_load(f)
        with open(task_yml) as f:
            exp_params = yaml.safe_load(f)

        exp_params["robot_params"] = exp_params["model"]
        rollout_fn = self.get_rollout_fn(
            exp_params=exp_params,
            tensor_args=self.tensor_args,
            world_params=world_params,
        )
        controller_params = self._build_controller_params(exp_params, rollout_fn)

        self.exp_params = exp_params
        self.robot_params = robot_params
        self.world_params = world_params
        self.success_threshold = exp_params.get("task_metrics", {}).get(
            "success_threshold",
            exp_params.get("sage", {}).get("success_threshold"),
        )
        self.task_file = task_yml
        self.robot_file = robot_yml
        self.world_file = world_yml
        return SAGE_MPPI(**controller_params)

    def init_aux(self):
        self.state_filter = JointStateFilter(
            filter_coeff=self.exp_params["state_filter_coeff"],
            dt=self.exp_params["control_dt"],
        )
        self.command_filter = JointStateFilter(
            filter_coeff=self.exp_params["cmd_filter_coeff"],
            dt=self.exp_params["control_dt"],
        )
        self.control_process = ControlProcessSage(
            self.controller,
            control_space=self.exp_params.get("control_space", "acc"),
            control_dt=self.exp_params["control_dt"],
        )
        self.n_dofs = self.controller.rollout_fn.dynamics_model.n_dofs
        self.zero_acc = np.zeros(self.n_dofs)

    def _augment_task_stats(self, stats):
        stats = {} if stats is None else dict(stats)
        stats.setdefault("success", None)
        stats.setdefault("failure", None)
        stats.setdefault("final_goal_distance", None)
        stats.setdefault("z_t", 0)
        stats.setdefault("covariance_fallback", False)
        stats.setdefault("covariance_fallback_count", 0)
        stats.setdefault("weight_entropy", 0.0)
        stats.setdefault("covariance_trace_mean", 0.0)
        stats.setdefault("shape_condition_number", 1.0)
        stats.setdefault("proposal_scale_min", None)
        stats.setdefault("proposal_scale_max", None)
        stats.setdefault("shape_update_skipped", False)
        stats.setdefault("shape_skip_reason", "")
        stats.setdefault("enable_runtime_stats", False)

        if self.success_threshold is not None and stats["final_goal_distance"] is not None:
            success = bool(stats["final_goal_distance"] <= self.success_threshold)
            stats["success"] = success
            stats["failure"] = not success

        stats["success_threshold"] = self.success_threshold
        stats["controller_type"] = "SAGE_MPPI"
        stats["controller_core_config"] = dict(self.controller_core_config)
        stats["deployment_refinement_config"] = dict(self.deployment_refinement_config)
        return stats

    def get_latest_stats(self):
        if hasattr(self.controller, "get_latest_stats"):
            return self._augment_task_stats(self.controller.get_latest_stats())
        return self._augment_task_stats(self.latest_task_stats)

    def get_command_and_stats(self, t_step, curr_state, control_dt=None, WAIT=True):
        control_dt = self.exp_params["control_dt"] if control_dt is None else control_dt
        cmd_des = BaseTaskSage.get_command(
            self,
            t_step,
            curr_state,
            control_dt=control_dt,
            WAIT=WAIT,
        )
        stats = self.get_latest_stats() if WAIT else self._augment_task_stats(self.latest_task_stats)
        self.latest_task_stats = dict(stats)
        return cmd_des, stats
