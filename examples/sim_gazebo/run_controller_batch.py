#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import torch
import yaml

torch.multiprocessing.set_start_method("spawn", force=True)

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from experiment_logging import (
    CsvExperimentLogger,
    extract_raw_info,
    normalize_step_record,
    summarize_episode,
)
from storm_kit.mpc.control import MPPI
from storm_kit.mpc.rollout.arm_reacher import ArmReacher
from storm_kit.mpc.task.sage_reacher_task import SageReacherTask
from storm_kit.mpc.task.task_base import BaseTask
from storm_kit.mpc.utils.mpc_process_wrapper import ControlProcess
from storm_kit.mpc.utils.state_filter import JointStateFilter
from storm_kit.util_file import get_gym_configs_path, get_mpc_configs_path, join_path

try:
    from examples.sim_gazebo.diffusion_gazebo_reacher_task import DiffusionGazeboReacherTask
except ImportError:
    DiffusionGazeboReacherTask = None


def _repo_path(*parts):
    return os.path.join(REPO_ROOT, *parts)


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_yaml(path, base_dir_getter):
    if os.path.isabs(path):
        return path
    return join_path(base_dir_getter(), path)


def _build_tensor_args(use_cuda):
    cuda_enabled = bool(use_cuda and torch.cuda.is_available())
    if use_cuda and not cuda_enabled:
        print("CUDA 不可用，自动回退到 CPU", flush=True)
    device = torch.device("cuda", 0) if cuda_enabled else torch.device("cpu")
    return {"device": device, "dtype": torch.float32}


def _default_paths(controller_name, scene):
    robot_file = _repo_path("examples", "sim_gazebo", "config", "ur7e_robot_gazebo.yml")
    world_default = _repo_path("examples", "sim_gazebo", "config", "collision_world_gazebo.yml")
    world_tall = _repo_path("examples", "sim_gazebo", "config", "collision_world_gazebo_tall.yml")
    if controller_name == "sage":
        task_default = _repo_path("content", "configs", "mpc", "ur7e_reacher_sage.yml")
        task_tall = task_default
    elif controller_name == "baseline":
        task_default = _repo_path("content", "configs", "mpc", "ur7e_reacher.yml")
        task_tall = task_default
    elif controller_name == "diffusion":
        task_default = _repo_path("examples", "sim_gazebo", "config", "ur7e_reacher_gazebo_tall.yml")
        task_tall = task_default
    else:
        raise ValueError(f"Unsupported controller_name: {controller_name}")

    if scene == "tall":
        return task_tall, robot_file, world_tall
    return task_default, robot_file, world_default


def _goal_joint_positions(args):
    if args.goal is not None:
        return np.asarray(args.goal, dtype=np.float64)
    return np.asarray([0.5, -1.2, 1.2, -1.57, -1.57, 0.0], dtype=np.float64)


def _goal_state(goal_positions):
    return np.concatenate([goal_positions, np.zeros_like(goal_positions)])


def _initial_state_dict(task, robot_file):
    robot_cfg = _load_yaml(robot_file)
    init_state = robot_cfg.get("sim_params", {}).get(
        "init_state",
        task.exp_params["model"]["init_state"],
    )
    init_state = np.asarray(init_state, dtype=np.float64)
    n_dofs = task.n_dofs
    return {
        "position": init_state[:n_dofs].copy(),
        "velocity": np.zeros(n_dofs, dtype=np.float64),
        "acceleration": np.zeros(n_dofs, dtype=np.float64),
    }


def _command_to_state(command):
    return {
        "position": np.asarray(command["position"], dtype=np.float64).copy(),
        "velocity": np.asarray(command["velocity"], dtype=np.float64).copy(),
        "acceleration": np.asarray(command["acceleration"], dtype=np.float64).copy(),
    }


def _apply_seed(task, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    controller = task.controller
    if hasattr(controller, "seed_val"):
        controller.seed_val = int(seed)
    if hasattr(controller, "sample_params") and isinstance(controller.sample_params, dict):
        controller.sample_params["seed"] = int(seed)
    sample_lib = getattr(controller, "sample_lib", None)
    if sample_lib is not None:
        for attr_name in ("seed", "seed_val", "base_seed"):
            if hasattr(sample_lib, attr_name):
                setattr(sample_lib, attr_name, int(seed))


class LoggedBaselineReacherTask(BaseTask):
    def __init__(self, task_file, robot_file, world_file, tensor_args):
        super().__init__(tensor_args=tensor_args)
        self.controller = self.init_mppi(task_file, robot_file, world_file)
        self.init_aux()

    def get_rollout_fn(self, **kwargs):
        return ArmReacher(**kwargs)

    def init_mppi(self, task_file, robot_file, world_file):
        robot_yml = _resolve_yaml(robot_file, get_gym_configs_path)
        world_yml = _resolve_yaml(world_file, get_gym_configs_path)
        task_yml = _resolve_yaml(task_file, get_mpc_configs_path)

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

        mppi_params = dict(exp_params["mppi"])
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
        else:
            mppi_params["init_mean"] = init_action
        mppi_params["rollout_fn"] = rollout_fn
        mppi_params["tensor_args"] = self.tensor_args

        self.exp_params = exp_params
        self.robot_file = robot_yml
        self.world_file = world_yml
        self.task_file = task_yml
        return MPPI(**mppi_params)

    def init_aux(self):
        self.state_filter = JointStateFilter(
            filter_coeff=self.exp_params["state_filter_coeff"],
            dt=self.exp_params["control_dt"],
        )
        self.command_filter = JointStateFilter(
            filter_coeff=self.exp_params["cmd_filter_coeff"],
            dt=self.exp_params["control_dt"],
        )
        self.control_process = ControlProcess(
            self.controller,
            control_space=self.exp_params.get("control_space", "acc"),
            control_dt=self.exp_params["control_dt"],
        )
        self.n_dofs = self.controller.rollout_fn.dynamics_model.n_dofs
        self.zero_acc = np.zeros(self.n_dofs)

    def get_command_and_stats(self, t_step, curr_state, control_dt=None, WAIT=True):
        control_dt = self.exp_params["control_dt"] if control_dt is None else control_dt
        cmd_des = BaseTask.get_command(
            self,
            t_step,
            curr_state,
            control_dt=control_dt,
            WAIT=WAIT,
        )
        return cmd_des, extract_raw_info(self)


class LoggedDiffusionReacherTask(DiffusionGazeboReacherTask):
    def init_aux(self):
        self.state_filter = JointStateFilter(
            filter_coeff=self.exp_params["state_filter_coeff"],
            dt=self.exp_params["control_dt"],
        )
        self.command_filter = JointStateFilter(
            filter_coeff=self.exp_params["cmd_filter_coeff"],
            dt=self.exp_params["control_dt"],
        )
        self.control_process = ControlProcess(
            self.controller,
            control_space=self.exp_params.get("control_space", "acc"),
            control_dt=self.exp_params["control_dt"],
        )
        self.n_dofs = self.controller.rollout_fn.dynamics_model.n_dofs
        self.zero_acc = np.zeros(self.n_dofs)

    def get_command_and_stats(self, t_step, curr_state, control_dt=None, WAIT=True):
        control_dt = self.exp_params["control_dt"] if control_dt is None else control_dt
        cmd_des = super().get_command(
            t_step=t_step,
            curr_state=curr_state,
            control_dt=control_dt,
            WAIT=WAIT,
        )
        return cmd_des, extract_raw_info(self)


def _build_task(controller_name, task_file, robot_file, world_file, tensor_args):
    if controller_name == "baseline":
        task = LoggedBaselineReacherTask(task_file, robot_file, world_file, tensor_args)
    elif controller_name == "sage":
        task = SageReacherTask(task_file, robot_file, world_file, tensor_args)
    elif controller_name == "diffusion":
        if DiffusionGazeboReacherTask is None:
            raise RuntimeError("DiffusionGazeboReacherTask is not available in this environment")
        task = LoggedDiffusionReacherTask(task_file, robot_file, world_file, tensor_args)
    else:
        raise ValueError(f"Unsupported controller: {controller_name}")

    task.task_file = getattr(task, "task_file", task_file)
    task.robot_file = getattr(task, "robot_file", robot_file)
    task.world_file = getattr(task, "world_file", world_file)
    return task


def _update_goal(task, args):
    task.update_params(goal_state=_goal_state(_goal_joint_positions(args)))


def _task_command_and_raw_stats(task, t_step, current_state, control_dt):
    if hasattr(task, "get_command_and_stats"):
        return task.get_command_and_stats(t_step, current_state, control_dt=control_dt, WAIT=True)
    cmd_des = task.get_command(t_step, current_state, control_dt=control_dt, WAIT=True)
    return cmd_des, extract_raw_info(task)


def _run_headless_episode(controller_name, task, args, seed, episode_id, logger):
    current_state = _initial_state_dict(task, task.robot_file)
    control_dt = task.exp_params["control_dt"]
    t_step = 0.0
    step_records = []

    for step_id in range(args.steps):
        command, raw_info = _task_command_and_raw_stats(task, t_step, current_state, control_dt)
        step_record = normalize_step_record(
            controller_name=controller_name,
            task=task,
            current_state=current_state,
            episode_id=episode_id,
            step_id=step_id,
            seed=seed,
            raw_info=raw_info,
            success_threshold=args.success_threshold,
        )
        logger.log_step(step_record)
        step_records.append(step_record)
        print(
            "[%s][headless] episode=%s step=%03d goal_dist=%s min_margin=%s"
            % (
                controller_name,
                episode_id,
                step_id,
                step_record["final_goal_distance"],
                step_record["minimum_safety_margin"],
            ),
            flush=True,
        )
        current_state = _command_to_state(command)
        t_step += control_dt
        if args.stop_on_success and bool(step_record["success"]):
            break

    logger.log_episode(summarize_episode(step_records))
    return step_records


def _init_ros():
    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
    except ImportError as exc:
        raise RuntimeError(
            "Gazebo mode requires ROS2 Python packages. Please source /opt/ros/humble/setup.bash."
        ) from exc
    return rclpy, MultiThreadedExecutor, Node, JointState, Float64MultiArray


def _make_robot_interface(Node, JointState, Float64MultiArray):
    class GazeboRobotInterface(Node):
        def __init__(self, joint_names, control_rate):
            super().__init__("storm_controller_batch")
            self.joint_names = joint_names
            self.n_dof = len(joint_names)
            self.current_positions = None
            self.current_velocities = None
            self.prev_velocities = None
            self.prev_time = None
            self.state_received = False
            self.sub_joint_states = self.create_subscription(
                JointState,
                "/joint_states",
                self._joint_state_callback,
                10,
            )
            self.pub_position_cmd = self.create_publisher(
                Float64MultiArray,
                "/forward_position_controller/commands",
                10,
            )

        def _joint_state_callback(self, msg):
            positions = np.zeros(self.n_dof)
            velocities = np.zeros(self.n_dof)
            for i, name in enumerate(self.joint_names):
                if name in msg.name:
                    idx = msg.name.index(name)
                    positions[i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        velocities[i] = msg.velocity[idx]
            self.current_positions = positions
            self.current_velocities = velocities
            self.state_received = True

        def wait_for_state(self, rclpy_module, timeout=10.0):
            start = time.time()
            while not self.state_received:
                rclpy_module.spin_once(self, timeout_sec=0.1)
                if time.time() - start > timeout:
                    return False
            return True

        def get_state(self):
            if not self.state_received:
                return None
            current_time = time.time()
            if self.prev_velocities is not None and self.prev_time is not None:
                dt = max(current_time - self.prev_time, 0.001)
                acceleration = (self.current_velocities - self.prev_velocities) / dt
            else:
                acceleration = np.zeros(self.n_dof)
            self.prev_velocities = self.current_velocities.copy()
            self.prev_time = current_time
            return {
                "position": self.current_positions.copy(),
                "velocity": self.current_velocities.copy(),
                "acceleration": acceleration,
            }

        def send_position_command(self, positions):
            msg = Float64MultiArray()
            msg.data = positions.tolist()
            self.pub_position_cmd.publish(msg)

    return GazeboRobotInterface


def _run_gazebo_episode(controller_name, task, args, seed, episode_id, logger):
    rclpy, MultiThreadedExecutor, Node, JointState, Float64MultiArray = _init_ros()
    GazeboRobotInterface = _make_robot_interface(Node, JointState, Float64MultiArray)
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    robot = None
    executor = None
    spin_thread = None
    spin_running = True
    step_records = []
    try:
        rclpy.init(args=None)
        robot = GazeboRobotInterface(joint_names, args.rate)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(robot)

        def _spin():
            while spin_running and rclpy.ok():
                executor.spin_once(timeout_sec=0.1)

        spin_thread = threading.Thread(target=_spin, daemon=True)
        spin_thread.start()

        if not robot.wait_for_state(rclpy, timeout=15.0):
            raise RuntimeError("Unable to receive /joint_states from Gazebo")

        control_dt = task.exp_params["control_dt"]
        t_step = 0.0
        for step_id in range(args.steps):
            current_state = robot.get_state()
            if current_state is None:
                time.sleep(0.01)
                continue
            command, raw_info = _task_command_and_raw_stats(task, t_step, current_state, control_dt)
            target_positions = np.asarray(command["position"]).flatten()[: task.n_dofs]
            robot.send_position_command(target_positions)
            step_record = normalize_step_record(
                controller_name=controller_name,
                task=task,
                current_state=current_state,
                episode_id=episode_id,
                step_id=step_id,
                seed=seed,
                raw_info=raw_info,
                success_threshold=args.success_threshold,
            )
            logger.log_step(step_record)
            step_records.append(step_record)
            print(
                "[%s][gazebo] episode=%s step=%03d goal_dist=%s min_margin=%s"
                % (
                    controller_name,
                    episode_id,
                    step_id,
                    step_record["final_goal_distance"],
                    step_record["minimum_safety_margin"],
                ),
                flush=True,
            )
            t_step += control_dt
            if args.stop_on_success and bool(step_record["success"]):
                break
            time.sleep(max(1.0 / args.rate, 0.02))

        logger.log_episode(summarize_episode(step_records))
        return step_records
    finally:
        spin_running = False
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)
        if robot is not None:
            robot.destroy_node()
        if executor is not None:
            executor.shutdown(timeout_sec=1.0)
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


def run_controller_batch(args):
    controllers = [args.controller] if args.controller != "all" else ["baseline", "sage", "diffusion"]
    seeds = args.seed if args.seed else [0]
    mode_name = "gazebo" if args.gazebo else "headless"
    run_root = os.path.join(
        os.path.abspath(args.output_dir),
        f"run_{_timestamp()}_{mode_name}",
    )
    os.makedirs(run_root, exist_ok=True)

    for controller_name in controllers:
        task_default, robot_default, world_default = _default_paths(controller_name, args.scene)
        task_file = args.task_file or task_default
        robot_file = args.robot_file or robot_default
        world_file = args.world_file or world_default

        for seed in seeds:
            run_dir = os.path.join(
                run_root,
                f"controller={controller_name}",
                f"seed={seed}",
            )
            os.makedirs(run_dir, exist_ok=True)
            logger = CsvExperimentLogger(run_dir)

            for episode_idx in range(args.episodes):
                episode_id = f"{controller_name}_seed{seed}_ep{episode_idx:03d}"
                tensor_args = _build_tensor_args(args.cuda)
                task = _build_task(controller_name, task_file, robot_file, world_file, tensor_args)
                try:
                    _apply_seed(task, seed)
                    _update_goal(task, args)
                    if args.gazebo:
                        _run_gazebo_episode(controller_name, task, args, seed, episode_id, logger)
                    else:
                        _run_headless_episode(controller_name, task, args, seed, episode_id, logger)
                finally:
                    task.close()

            print(
                "completed controller=%s seed=%s output=%s"
                % (controller_name, seed, run_dir),
                flush=True,
            )

    print(f"batch run completed: {run_root}", flush=True)
    return run_root


def main():
    parser = argparse.ArgumentParser(description="Unified batch runner for baseline / diffusion / SAGE controllers")
    parser.add_argument("--controller", choices=["baseline", "diffusion", "sage", "all"], required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--output_dir", default=_repo_path("examples", "sim_gazebo", "batch_runs"))
    parser.add_argument("--headless", dest="gazebo", action="store_false", default=False)
    parser.add_argument("--gazebo", dest="gazebo", action="store_true")
    parser.add_argument("--scene", choices=["default", "tall"], default="default")
    parser.add_argument("--task-file", default=None)
    parser.add_argument("--robot-file", default=None)
    parser.add_argument("--world-file", default=None)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--goal", type=float, nargs=6, default=None)
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--stop-on-success", action="store_true", default=False)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--no-cuda", dest="cuda", action="store_false")
    args = parser.parse_args()

    run_root = run_controller_batch(args)
    print(run_root, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
