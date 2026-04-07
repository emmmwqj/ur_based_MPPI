#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR7e SAGE-MPPI Reach Static - Gazebo high-wall scene.

This mirrors the original tall-scene sim_gazebo project:
- Same tall task/world scene files
- Same Gazebo/ROS2 control loop and RViz visualization topics
- Same dynamic goal update flow via /target_pose

The only intended behavioral difference is the controller/task assembly:
- baseline project uses MPPI via GazeboReacherTask
- this project uses SAGE_MPPI via SageReacherTask
"""

import argparse
import os
import signal
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import torch
import yaml

STORM_ROOT = os.path.expanduser("~/storm")
if STORM_ROOT not in sys.path:
    sys.path.insert(0, STORM_ROOT)

torch.multiprocessing.set_start_method("spawn", force=True)

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
except ImportError:
    print("=" * 60)
    print("错误: 未找到 ROS2 Python 包")
    print("请先 source ROS2 环境:")
    print("  source /opt/ros/humble/setup.bash")
    print("=" * 60)
    sys.exit(1)

from examples.sim_gazebo.gazebo_obstacle_utils import (
    count_primitive_obstacles,
    spawn_gazebo_obstacles,
)
from examples.sim_gazebo.reach_static_ur7e import inv_transform_point, transform_point
from examples.sim_gazebo.reach_static_ur7e_tall import (
    _compute_link_poses_robot_frame,
    _log,
    _recover_command,
    _reset_control_process_timing,
    _shutdown_control_process,
    CollisionSphereVisualizer,
    TallGazeboRobotInterface,
)
from storm_kit.mpc.task.sage_reacher_task import SageReacherTask

np.set_printoptions(precision=3, suppress=True)

FORWARD_POSITION_COMMAND_TOPIC = "/forward_position_controller/commands"


def _reset_sage_distribution(mpc) -> None:
    controller = getattr(mpc, "controller", None)
    if controller is not None and hasattr(controller, "reset"):
        controller.reset()


def _get_top_ee_trajs_world(
    mpc,
    robot_pos_world: np.ndarray,
    robot_quat_xyzw: np.ndarray,
    current_ee_pos_world: np.ndarray,
    max_trajs: int = 5,
):
    """
    Build world-frame top trajectories for RViz and explicitly prepend the
    current end-effector position.

    The rollout stores future ee positions starting at the first simulated
    horizon step, so if we visualize those points directly the line appears to
    start "in front of" the live robot. For RViz we want a continuous line from
    the current ee marker to the predicted future path.
    """
    controller = getattr(mpc, "controller", None)
    trajectories = getattr(controller, "trajectories", None)
    total_costs = getattr(controller, "total_costs", None)
    if trajectories is None or total_costs is None:
        return None

    ee_pos_seq = trajectories.get("ee_pos_seq", None)
    if ee_pos_seq is None:
        return None

    if isinstance(ee_pos_seq, torch.Tensor):
        ee_pos_seq_np = ee_pos_seq.detach().cpu().numpy()
    else:
        ee_pos_seq_np = np.asarray(ee_pos_seq)

    if isinstance(total_costs, torch.Tensor):
        total_costs_np = total_costs.detach().cpu().numpy()
    else:
        total_costs_np = np.asarray(total_costs)

    if ee_pos_seq_np.ndim != 3 or ee_pos_seq_np.shape[-1] != 3 or total_costs_np.ndim != 1:
        return None

    top_count = min(max_trajs, ee_pos_seq_np.shape[0], total_costs_np.shape[0])
    if top_count <= 0:
        return None

    top_indices = np.argsort(total_costs_np)[:top_count]
    top_trajs_np = ee_pos_seq_np[top_indices]
    current_ee_pos_world = np.asarray(current_ee_pos_world, dtype=np.float64).reshape(1, 3)

    top_trajs_world = []
    for traj_points in top_trajs_np:
        traj_world = transform_point(robot_pos_world, robot_quat_xyzw, traj_points)
        traj_world = np.asarray(traj_world, dtype=np.float64)
        if traj_world.ndim != 2 or traj_world.shape[-1] != 3:
            continue
        traj_world = np.concatenate((current_ee_pos_world, traj_world), axis=0)
        top_trajs_world.append(traj_world)

    if not top_trajs_world:
        return None
    return np.asarray(top_trajs_world, dtype=np.float64)


class _GoalHoldController:
    """
    Latch a local position hold once the end-effector has stably reached the goal.

    The tall scene can transiently cross the success radius and then drift back
    out because the controller keeps replanning around a nearly solved target.
    This helper adds a small hysteresis loop:
    - enter hold only after several consecutive low-error, low-velocity steps
    - keep publishing a latched joint-position target while the robot remains close
    - leave hold only if the error grows well above the success radius
    """

    def __init__(
        self,
        success_threshold: float,
        enter_threshold: float = None,
        exit_threshold: float = None,
        enter_count: int = 5,
        exit_count: int = 6,
        velocity_threshold: float = 0.08,
    ) -> None:
        self.success_threshold = float(success_threshold)
        self.enter_threshold = float(
            self.success_threshold if enter_threshold is None else enter_threshold
        )
        self.exit_threshold = float(
            max(self.enter_threshold * 1.5, self.success_threshold * 1.5)
            if exit_threshold is None
            else exit_threshold
        )
        self.enter_count = int(enter_count)
        self.exit_count = int(exit_count)
        self.velocity_threshold = float(velocity_threshold)
        self.active = False
        self.hold_positions = None
        self._enter_streak = 0
        self._exit_streak = 0

    def reset(self) -> None:
        self.active = False
        self.hold_positions = None
        self._enter_streak = 0
        self._exit_streak = 0

    def force_activate(self, q: np.ndarray) -> None:
        self.active = True
        self.hold_positions = np.asarray(q, dtype=np.float64).copy()
        self._enter_streak = self.enter_count
        self._exit_streak = 0

    def update(self, error: float, q: np.ndarray, dq: np.ndarray):
        just_entered = False
        just_released = False

        if self.active:
            if error > self.exit_threshold:
                self._exit_streak += 1
                if self._exit_streak >= self.exit_count:
                    self.reset()
                    just_released = True
            else:
                self._exit_streak = 0
            hold_positions = None if self.hold_positions is None else self.hold_positions.copy()
            return self.active, hold_positions, just_entered, just_released

        velocity_norm = float(np.linalg.norm(dq))
        if error <= self.enter_threshold and velocity_norm <= self.velocity_threshold:
            self._enter_streak += 1
            if self._enter_streak >= self.enter_count:
                self.active = True
                self.hold_positions = np.asarray(q, dtype=np.float64).copy()
                self._exit_streak = 0
                just_entered = True
        else:
            self._enter_streak = 0

        hold_positions = None if self.hold_positions is None else self.hold_positions.copy()
        return self.active, hold_positions, just_entered, just_released


class _NearGoalRefinementController:
    """
    Shrink exploration and bias the controller toward fine convergence near goal.

    SAGE's larger proposal is helpful in the tall scene before the arm reaches the
    basin of attraction, but it becomes counterproductive once the remaining
    position error is only a few centimeters. This helper temporarily:
    - reduces sigma_0
    - disables large stagnation amplification
    - optionally increases the positional goal weight
    - keeps a hysteresis band so it does not chatter on threshold crossings
    """

    def __init__(
        self,
        mpc,
        controller,
        rollout_fn,
        enter_threshold: float = 0.08,
        exit_threshold: float = 0.11,
        sigma_scale: float = 0.2,
        stagnation_alpha: float = 0.0,
        goal_weight_scale: float = 1.5,
        tau_p: float = None,
        step_size_mean: float = None,
    ) -> None:
        self.mpc = mpc
        self.controller = controller
        self.rollout_fn = rollout_fn
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.sigma_scale = float(sigma_scale)
        self.refine_stagnation_alpha = float(stagnation_alpha)
        self.goal_weight_scale = float(goal_weight_scale)
        self.refine_tau_p = None if tau_p is None else float(tau_p)
        self.refine_step_size_mean = None if step_size_mean is None else float(step_size_mean)
        self.active = False

        self.base_sigma_0 = float(controller.sigma_0)
        self.base_stagnation_alpha = float(controller.stagnation_alpha)
        self.base_tau_p = float(controller.tau_p)
        self.base_step_size_mean = float(controller.step_size_mean)
        self.base_goal_weight = self._get_goal_position_weight()
        self.base_retract_state = np.asarray(
            self.mpc.exp_params["cost"]["retract_state"],
            dtype=np.float64,
        ).copy()

    def _get_goal_position_weight(self) -> Optional[float]:
        goal_cost = getattr(self.rollout_fn, "goal_cost", None)
        if goal_cost is None or not hasattr(goal_cost, "weight"):
            return None
        weight = goal_cost.weight
        if isinstance(weight, torch.Tensor):
            if weight.numel() < 2:
                return None
            return float(weight.detach().reshape(-1)[1].item())
        if len(weight) < 2:
            return None
        return float(weight[1])

    def _set_goal_position_weight(self, value: float) -> None:
        goal_cost = getattr(self.rollout_fn, "goal_cost", None)
        if goal_cost is None or not hasattr(goal_cost, "weight"):
            return
        weight = goal_cost.weight
        if isinstance(weight, torch.Tensor):
            weight = weight.clone()
            weight.reshape(-1)[1] = float(value)
            goal_cost.weight = weight
            return
        weight = list(weight)
        if len(weight) >= 2:
            weight[1] = float(value)
            goal_cost.weight = weight

    def _apply_refine_params(self) -> None:
        self.controller.sigma_0 = self.base_sigma_0 * self.sigma_scale
        self.controller.stagnation_alpha = self.refine_stagnation_alpha
        if self.refine_tau_p is not None:
            self.controller.tau_p = self.refine_tau_p
        if self.refine_step_size_mean is not None:
            self.controller.step_size_mean = self.refine_step_size_mean
        if self.base_goal_weight is not None:
            self._set_goal_position_weight(self.base_goal_weight * self.goal_weight_scale)

    def _restore_nominal_params(self) -> None:
        self.controller.sigma_0 = self.base_sigma_0
        self.controller.stagnation_alpha = self.base_stagnation_alpha
        self.controller.tau_p = self.base_tau_p
        self.controller.step_size_mean = self.base_step_size_mean
        self.mpc.update_params(retract_state=self.base_retract_state)
        if self.base_goal_weight is not None:
            self._set_goal_position_weight(self.base_goal_weight)

    def reset(self) -> None:
        self.active = False
        self._restore_nominal_params()

    def update(self, error: float, current_q: np.ndarray):
        just_entered = False
        just_exited = False

        if self.active:
            if error > self.exit_threshold:
                self.active = False
                self._restore_nominal_params()
                just_exited = True
            return self.active, just_entered, just_exited

        if error <= self.enter_threshold:
            self.active = True
            self._apply_refine_params()
            self.mpc.update_params(retract_state=np.asarray(current_q, dtype=np.float64).copy())
            just_entered = True
        return self.active, just_entered, just_exited


class _CartesianGoalRefiner:
    """
    Final centimeters refinement using the current end-effector Jacobian.

    Once SAGE has already driven the arm into the success basin, continuing to
    rely on sampled proposals is inefficient. This local damped least-squares
    step directly reduces the residual Cartesian position error before we latch
    the final hold.
    """

    def __init__(
        self,
        rollout_fn,
        tensor_args,
        enter_threshold: float = 0.05,
        exit_threshold: float = 0.07,
        damping: float = 0.05,
        gain: float = 0.7,
        max_joint_step: float = 0.02,
    ) -> None:
        self.rollout_fn = rollout_fn
        self.tensor_args = tensor_args
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.damping = float(damping)
        self.gain = float(gain)
        self.max_joint_step = float(max_joint_step)
        self.active = False

        self.robot_model = rollout_fn.dynamics_model.robot_model
        self.ee_link_name = rollout_fn.exp_params["model"]["ee_link_name"]
        dyn_model = rollout_fn.dynamics_model
        self.q_lower = dyn_model.state_lower_bounds[: dyn_model.n_dofs].detach().cpu().numpy()
        self.q_upper = dyn_model.state_upper_bounds[: dyn_model.n_dofs].detach().cpu().numpy()

    def reset(self) -> None:
        self.active = False

    def update(self, error: float):
        just_entered = False
        just_exited = False
        if self.active:
            if error > self.exit_threshold:
                self.active = False
                just_exited = True
            return self.active, just_entered, just_exited
        if error <= self.enter_threshold:
            self.active = True
            just_entered = True
        return self.active, just_entered, just_exited

    def compute_command(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        goal_ee_pos_robot: np.ndarray,
    ) -> np.ndarray:
        q_t = torch.as_tensor(q, **self.tensor_args).unsqueeze(0)
        qd_t = torch.as_tensor(dq, **self.tensor_args).unsqueeze(0)
        goal_t = torch.as_tensor(goal_ee_pos_robot, **self.tensor_args).reshape(1, 3)

        ee_pos, _, lin_jac, _ = self.robot_model.compute_fk_and_jacobian(
            q_t,
            qd_t,
            link_name=self.ee_link_name,
        )
        pos_err = (goal_t - ee_pos).reshape(3)
        jac = lin_jac.reshape(3, -1)
        ident = torch.eye(3, **self.tensor_args)
        dls_step = jac.transpose(-2, -1) @ torch.linalg.solve(
            jac @ jac.transpose(-2, -1) + (self.damping ** 2) * ident,
            pos_err.unsqueeze(-1),
        )
        joint_step = self.gain * dls_step.squeeze(-1)
        joint_step = torch.clamp(joint_step, -self.max_joint_step, self.max_joint_step)
        q_cmd = q_t.reshape(-1) + joint_step
        q_cmd = torch.max(torch.min(q_cmd, torch.as_tensor(self.q_upper, **self.tensor_args)),
                          torch.as_tensor(self.q_lower, **self.tensor_args))
        return q_cmd.detach().cpu().numpy()


class _StallMonitor:
    """
    Minimal stall detector for the tall scene.

    This mirrors the previously validated tall-scene debug criterion, but
    keeps only the logic needed for recovery:
    - large goal error
    - little end-effector motion over a short history window
    - low joint velocity magnitude
    """

    def __init__(
        self,
        history_len: int = 50,
        min_runtime: float = 8.0,
        error_threshold: float = 0.12,
        motion_threshold: float = 0.01,
        velocity_threshold: float = 0.08,
        cooldown: float = 8.0,
    ) -> None:
        self.history = deque(maxlen=history_len)
        self.min_runtime = float(min_runtime)
        self.error_threshold = float(error_threshold)
        self.motion_threshold = float(motion_threshold)
        self.velocity_threshold = float(velocity_threshold)
        self.cooldown = float(cooldown)
        self.last_recovery_t = -1e9

    def _history_motion(self) -> float:
        if len(self.history) < 2:
            return np.inf
        points = np.asarray(self.history, dtype=np.float64)
        ref = points[0]
        disp = np.linalg.norm(points - ref.reshape(1, 3), axis=1)
        return float(np.max(disp))

    def update(self, ee_pos_world: np.ndarray) -> None:
        self.history.append(np.asarray(ee_pos_world, dtype=np.float64).copy())

    def should_recover(
        self,
        t_step: float,
        ee_pos_world: np.ndarray,
        goal_world: np.ndarray,
        joint_velocity: np.ndarray,
    ) -> bool:
        self.update(ee_pos_world)
        if t_step < self.min_runtime:
            return False
        if len(self.history) < self.history.maxlen:
            return False
        if (t_step - self.last_recovery_t) < self.cooldown:
            return False

        ee_error = float(np.linalg.norm(ee_pos_world - goal_world))
        history_motion = self._history_motion()
        velocity_norm = float(np.linalg.norm(joint_velocity))

        if ee_error <= self.error_threshold:
            return False
        if history_motion >= self.motion_threshold:
            return False
        if velocity_norm >= self.velocity_threshold:
            return False

        self.last_recovery_t = float(t_step)
        return True


def mpc_control_main(args):
    shutdown_event = threading.Event()
    executor = None
    spin_thread = None
    robot = None
    mpc = None
    exit_code = 0

    def request_shutdown(signum=None, _frame=None):
        if shutdown_event.is_set():
            return
        if signum is not None:
            _log(f"\n收到 {signal.Signals(signum).name}，准备退出...")
        shutdown_event.set()

    try:
        _log("=" * 60)
        _log("UR7e SAGE-MPPI Reach Static - Gazebo 高墙场景")
        _log("=" * 60)

        sim_gazebo_config_dir = os.path.join(STORM_ROOT, "examples", "sim_gazebo", "config")
        robot_file = os.path.join(sim_gazebo_config_dir, "ur7e_robot_gazebo.yml")
        task_file = os.path.join(sim_gazebo_config_dir, "ur7e_reacher_gazebo_tall_sage.yml")
        world_file = os.path.join(sim_gazebo_config_dir, "collision_world_gazebo_tall.yml")

        _log("\n加载配置文件...")
        _log(f"  Robot: {robot_file}")
        _log(f"  Task:  {task_file}")
        _log(f"  World: {world_file}")

        with open(robot_file) as f:
            robot_params = yaml.safe_load(f)
        with open(world_file) as f:
            world_params = yaml.safe_load(f)

        sim_params = robot_params.get("sim_params", {})
        robot_pose = sim_params.get("robot_pose", [0, 0, 0, 0, 0, 0, 1])
        robot_pos = np.array(robot_pose[:3], dtype=np.float64)
        robot_quat_xyzw = np.array(robot_pose[3:], dtype=np.float64)

        joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        n_dof = len(joint_names)

        _log("\n初始化 ROS2...")
        rclpy.init(args=None)
        signal.signal(signal.SIGINT, request_shutdown)
        signal.signal(signal.SIGTERM, request_shutdown)

        control_rate = args.rate
        robot = TallGazeboRobotInterface(joint_names, control_rate=control_rate)
        command_topic = getattr(getattr(robot, "pub_position_cmd", None), "topic_name", None)
        if command_topic != FORWARD_POSITION_COMMAND_TOPIC:
            _log(
                "错误: SAGE 控制器命令话题不是 ros2_control forward_position_controller: "
                f"{command_topic}"
            )
            return 1
        _log(
            "控制命令已显式绑定到 ros2_control forward_position_controller: "
            f"{FORWARD_POSITION_COMMAND_TOPIC}"
        )

        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(robot)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        _log("\n等待 Gazebo 关节状态...")
        timeout = 10.0
        start = time.time()
        while not robot.is_connected():
            if shutdown_event.is_set():
                return 130
            if time.time() - start > timeout:
                _log("错误: 无法接收关节状态，请确保 Gazebo 仿真正在运行")
                return 1
            time.sleep(0.1)

        _log("已连接到 Gazebo 机器人!")
        n_world_spheres, n_world_cubes = count_primitive_obstacles(world_params, include_ground=False)
        if spawn_gazebo_obstacles(robot, world_params, model_prefix="sim_tall_sage", include_ground=False):
            _log(
                "Gazebo 真实障碍物已生成: spheres=%d cubes=%d"
                % (n_world_spheres, n_world_cubes)
            )
        else:
            _log("警告: Gazebo 真实障碍物未完整生成，请检查 /spawn_entity 与 /delete_entity 服务")

        _log("\n初始化 SAGE-MPPI 控制器...")
        device = "cuda" if args.cuda else "cpu"
        _log(f"计算设备: {device}")

        tensor_args = {
            "device": torch.device(device, 0) if device == "cuda" else torch.device("cpu"),
            "dtype": torch.float32,
        }

        mpc = SageReacherTask(task_file, robot_file, world_file, tensor_args)
        control_dt = mpc.exp_params.get("control_dt", 0.02)
        _log(f"SAGE-MPPI 控制周期: {control_dt} s ({1.0 / control_dt:.1f} Hz)")

        default_goal_seed_state = np.array([0.5, -1.2, 1.2, -1.57, -1.57, 0.0, 0, 0, 0, 0, 0, 0])
        mpc.update_params(goal_state=default_goal_seed_state)

        goal_ee_quat = np.ravel(mpc.controller.rollout_fn.goal_ee_quat.cpu().numpy())
        goal_ee_world = np.array([0.5, -0.45, 0.4], dtype=np.float64)
        goal_ee_pos_robot = inv_transform_point(robot_pos, robot_quat_xyzw, goal_ee_world)
        mpc.update_params(goal_ee_pos=goal_ee_pos_robot, goal_ee_quat=goal_ee_quat)

        _log(f"\n默认目标末端位置 (机器人坐标系): {goal_ee_pos_robot}")
        _log(f"默认目标末端位置 (世界坐标系): {goal_ee_world}")

        rollout_fn = mpc.controller.rollout_fn
        collision_sphere_visualizer = CollisionSphereVisualizer(
            mpc.exp_params["model"]["robot_collision_params"]
        )
        sage_params = mpc.exp_params.get("sage", {})
        success_threshold = float(sage_params.get("success_threshold", 0.05))
        goal_hold = _GoalHoldController(
            success_threshold=success_threshold,
            enter_threshold=sage_params.get("hold_enter_threshold"),
            exit_threshold=sage_params.get("hold_exit_threshold"),
            enter_count=sage_params.get("hold_enter_count", 5),
            exit_count=sage_params.get("hold_exit_count", 6),
            velocity_threshold=sage_params.get("hold_velocity_threshold", 0.08),
        )
        refinement = _NearGoalRefinementController(
            mpc=mpc,
            controller=mpc.controller,
            rollout_fn=rollout_fn,
            enter_threshold=sage_params.get("refine_enter_threshold", 0.08),
            exit_threshold=sage_params.get("refine_exit_threshold", 0.11),
            sigma_scale=sage_params.get("refine_sigma_scale", 0.2),
            stagnation_alpha=sage_params.get("refine_stagnation_alpha", 0.0),
            goal_weight_scale=sage_params.get("refine_goal_weight_scale", 1.5),
            tau_p=sage_params.get("refine_tau_p"),
            step_size_mean=sage_params.get("refine_step_size_mean"),
        )
        cart_refiner = _CartesianGoalRefiner(
            rollout_fn=rollout_fn,
            tensor_args=tensor_args,
            enter_threshold=sage_params.get("cart_refine_enter_threshold", success_threshold),
            exit_threshold=sage_params.get("cart_refine_exit_threshold", 0.07),
            damping=sage_params.get("cart_refine_damping", 0.05),
            gain=sage_params.get("cart_refine_gain", 0.7),
            max_joint_step=sage_params.get("cart_refine_max_joint_step", 0.02),
        )
        n_collision_spheres = sum(
            len(collision_sphere_visualizer.spheres_by_link.get(link_name, []))
            for link_name in collision_sphere_visualizer.link_names
        )
        current_goal_ee = goal_ee_pos_robot.copy()
        current_goal_world = goal_ee_world.copy()
        stall_monitor = _StallMonitor()
        cart_hold_threshold = float(sage_params.get("cart_hold_threshold", 0.01))
        cart_hold_count = int(sage_params.get("cart_hold_count", 10))
        cart_hold_streak = 0
        _log(
            "[SAGE保持] success_threshold=%.3f, enter=%.3f, exit=%.3f, enter_count=%d, exit_count=%d"
            % (
                success_threshold,
                goal_hold.enter_threshold,
                goal_hold.exit_threshold,
                goal_hold.enter_count,
                goal_hold.exit_count,
            )
        )
        _log(
            "[SAGE精修] enter=%.3f, exit=%.3f, sigma_scale=%.3f, stagnation_alpha=%.3f, goal_weight_scale=%.2f"
            % (
                refinement.enter_threshold,
                refinement.exit_threshold,
                refinement.sigma_scale,
                refinement.refine_stagnation_alpha,
                refinement.goal_weight_scale,
            )
        )
        _log(
            "[SAGE末端精修] enter=%.3f, exit=%.3f, damping=%.3f, gain=%.2f, max_joint_step=%.3f"
            % (
                cart_refiner.enter_threshold,
                cart_refiner.exit_threshold,
                cart_refiner.damping,
                cart_refiner.gain,
                cart_refiner.max_joint_step,
            )
        )
        _log(
            "[SAGE末端锁定] cart_hold_threshold=%.4f, cart_hold_count=%d"
            % (cart_hold_threshold, cart_hold_count)
        )

        _log("\n" + "=" * 60)
        _log("开始 SAGE-MPPI 控制循环... (Ctrl+C 退出)")
        _log("=" * 60)
        _log("\n提示:")
        _log("  - 当前使用高墙场景 primitive world 避障")
        _log("  - 发布 PoseStamped 到 /target_pose 可动态更新目标")
        _log("  - 在 RViz 中查看 /visualization_marker_array")
        _log("  - 在 RViz 中查看 /mppi_top_traj_markers (SAGE top-5 预测轨迹)")
        _log("  - Gazebo 中已真实生成高墙/球体障碍物，可直接观察物理碰撞")
        _log("  - 红球=目标, 绿球=末端, 蓝色障碍物=高墙场景")
        _log(f"  - 黄球=机械臂碰撞球模型 ({n_collision_spheres} 个)")
        _log("  - 红线=SAGE top-5 末端预测轨迹")
        _log("  - 黄色碰撞球与预测轨迹按控制周期实时刷新")
        _log("  - 控制器使用同步求解，避免 async 命令时域耗尽")
        _log("")

        _log("预热 SAGE-MPPI 控制器...")
        current_state = robot.get_state()
        for warm_idx in range(5):
            if shutdown_event.is_set():
                return 130
            if current_state is None:
                time.sleep(0.01)
                current_state = robot.get_state()
                continue
            try:
                mpc.get_command(warm_idx * control_dt, current_state, control_dt=control_dt, WAIT=True)
            except Exception as exc:
                _log(f"预热异常 (可忽略): {exc}")
            time.sleep(0.01)

        _log("SAGE-MPPI 预热完成，开始控制!\n")

        loop_count = 0
        viz_update_every = 1
        loop_start = time.time()

        while rclpy.ok() and not shutdown_event.is_set():
            iter_start = time.time()
            t_step = time.time() - loop_start
            cmd = None

            state = robot.get_state()
            if state is None:
                time.sleep(control_dt)
                continue

            q = state["position"]
            dq = state["velocity"]
            ddq = state["acceleration"]
            curr = np.hstack([q, dq, ddq])
            ee_pose = rollout_fn.get_ee_pose(torch.as_tensor(curr, **tensor_args).unsqueeze(0))
            ee_pos_robot = np.ravel(ee_pose["ee_pos_seq"].cpu().numpy())
            ee_pos_world = transform_point(robot_pos, robot_quat_xyzw, ee_pos_robot)
            error = float(np.linalg.norm(ee_pos_world - current_goal_world))

            new_target = robot.get_target_position()
            if new_target is not None:
                target_robot = inv_transform_point(robot_pos, robot_quat_xyzw, new_target)
                if np.linalg.norm(target_robot - current_goal_ee) > 0.005:
                    current_goal_ee = target_robot.copy()
                    current_goal_world = new_target.copy()
                    goal_hold.reset()
                    refinement.reset()
                    cart_refiner.reset()
                    cart_hold_streak = 0
                    mpc.update_params(goal_ee_pos=current_goal_ee, goal_ee_quat=goal_ee_quat)
                    _log(
                        "[目标更新] 世界: %s, 机器人: %s"
                        % (np.round(current_goal_world, 3), np.round(current_goal_ee, 3))
                    )
                    try:
                        _reset_sage_distribution(mpc)
                        _reset_control_process_timing(mpc.control_process, t_step, control_dt)
                        cmd = mpc.get_command(t_step, state, control_dt=control_dt, WAIT=True)
                        _log("[目标更新] 已重置 SAGE 分布与时间基准，并同步重规划")
                    except Exception as sync_exc:
                        _log(f"[SAGE异常] 目标更新后的同步重规划失败: {sync_exc}")
                        time.sleep(control_dt)
                        continue

            refine_active, just_entered_refine, just_exited_refine = refinement.update(error, q)
            if just_entered_refine:
                _reset_sage_distribution(mpc)
                _reset_control_process_timing(mpc.control_process, t_step, control_dt)
                _log(
                    "[SAGE精修] 进入近目标精修 (ee_error=%.4f)，收缩 proposal 并提高目标位置权重"
                    % error
                )
            elif just_exited_refine:
                _reset_sage_distribution(mpc)
                _reset_control_process_timing(mpc.control_process, t_step, control_dt)
                _log(
                    "[SAGE精修] 误差回升到 %.4f，退出精修并恢复常规探索"
                    % error
                )

            cart_refine_active, just_entered_cart, just_exited_cart = cart_refiner.update(error)
            if just_entered_cart:
                _reset_sage_distribution(mpc)
                _reset_control_process_timing(mpc.control_process, t_step, control_dt)
                _log(
                    "[SAGE末端精修] 进入 Jacobian 末端精修 (ee_error=%.4f)"
                    % error
                )
            elif just_exited_cart:
                _reset_sage_distribution(mpc)
                _reset_control_process_timing(mpc.control_process, t_step, control_dt)
                cart_hold_streak = 0
                _log(
                    "[SAGE末端精修] 误差回升到 %.4f，退出 Jacobian 末端精修"
                    % error
                )

            if cart_refiner.active and (not goal_hold.active):
                if error <= cart_hold_threshold:
                    cart_hold_streak += 1
                    if cart_hold_streak >= cart_hold_count:
                        goal_hold.force_activate(q)
                        cart_refiner.reset()
                        _reset_sage_distribution(mpc)
                        _reset_control_process_timing(mpc.control_process, t_step, control_dt)
                        _log(
                            "[SAGE保持] Jacobian 精修已稳定收敛 (ee_error=%.4f)，锁定当前位置保持"
                            % error
                        )
                else:
                    cart_hold_streak = 0
            else:
                cart_hold_streak = 0

            hold_active = False
            if cmd is None:
                hold_active, hold_q, just_entered_hold, just_released_hold = goal_hold.update(
                    error=error,
                    q=q,
                    dq=dq,
                )

                if just_entered_hold:
                    cart_refiner.reset()
                    _reset_sage_distribution(mpc)
                    _reset_control_process_timing(mpc.control_process, t_step, control_dt)
                    _log(
                        "[SAGE保持] 末端稳定进入目标半径 (ee_error=%.4f)，锁定当前位置保持"
                        % error
                    )

                if just_released_hold:
                    _reset_sage_distribution(mpc)
                    _reset_control_process_timing(mpc.control_process, t_step, control_dt)
                    _log(
                        "[SAGE保持] 误差回升到 %.4f，退出保持并恢复重规划"
                        % error
                    )

                if hold_active and hold_q is not None:
                    cmd = {"position": hold_q}

            if cmd is None and cart_refiner.active:
                try:
                    cmd = {
                        "position": cart_refiner.compute_command(
                            q=q,
                            dq=dq,
                            goal_ee_pos_robot=current_goal_ee,
                        )
                    }
                except Exception as cart_exc:
                    _log(f"[SAGE末端精修] Jacobian 精修失败，回退到 MPC: {cart_exc}")

            if cmd is None:
                try:
                    cmd = mpc.get_command(t_step, state, control_dt=control_dt, WAIT=True)
                except (IndexError, RuntimeError, ValueError) as exc:
                    _log(
                        "[SAGE恢复] 同步取命令失败 (%s)，重置控制进程时间基准后重规划"
                        % exc
                    )
                    try:
                        cmd = _recover_command(mpc, t_step, state, control_dt)
                    except Exception as recover_exc:
                        _log(f"[SAGE异常] 同步重规划失败: {recover_exc}")
                        time.sleep(control_dt)
                        continue

            if cmd is None or "position" not in cmd:
                time.sleep(control_dt)
                continue

            target_positions = cmd["position"]
            if isinstance(target_positions, torch.Tensor):
                target_positions = target_positions.cpu().numpy()
            target_positions = np.array(target_positions).flatten()[:n_dof]
            robot.send_position_command(target_positions)
            robot.publish_ee_pose(ee_pos_world)

            if (not goal_hold.active) and stall_monitor.should_recover(t_step, ee_pos_world, current_goal_world, dq):
                _reset_sage_distribution(mpc)
                _reset_control_process_timing(mpc.control_process, t_step, control_dt)
                _log("[SAGE恢复] 检测到停滞，已重置 SAGE 分布与时间基准，下一轮将以放大 proposal 重新探索")

            if (loop_count % viz_update_every) == 0:
                link_pos_robot, link_rot_robot = _compute_link_poses_robot_frame(
                    rollout_fn,
                    q,
                    dq,
                    tensor_args,
                )
                collision_spheres_world = collision_sphere_visualizer.get_world_spheres(
                    link_pos_robot,
                    link_rot_robot,
                    robot_pos,
                    robot_quat_xyzw,
                )
                robot.publish_markers(
                    world_params,
                    current_goal_world,
                    ee_pos_world,
                    collision_spheres=collision_spheres_world,
                )
                top_trajs_world = _get_top_ee_trajs_world(
                    mpc,
                    robot_pos,
                    robot_quat_xyzw,
                    ee_pos_world,
                    max_trajs=5,
                )
                robot.publish_top_trajectories(top_trajs_world)

            loop_count += 1
            if loop_count % 50 == 0:
                latest_stats = {}
                if hasattr(mpc, "get_latest_stats"):
                    try:
                        latest_stats = mpc.get_latest_stats()
                    except Exception:
                        latest_stats = {}
                _log(
                    f"[{loop_count:5d}] t={t_step:.2f}s | "
                    f"q=[{q[0]:+.2f}, {q[1]:+.2f}, {q[2]:+.2f}, {q[3]:+.2f}, {q[4]:+.2f}, {q[5]:+.2f}] | "
                    f"ee_error={error:.4f} | opt_dt={mpc.opt_dt:.3f}s | "
                    f"refine={int(refinement.active)} | "
                    f"cart={int(cart_refiner.active)} | "
                    f"hold={int(goal_hold.active)} | "
                    f"z_t={int(latest_stats.get('z_t', 0))} | rho_k={float(latest_stats.get('rho_k', 0.0)):.3f}"
                )

            elapsed = time.time() - iter_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if shutdown_event.is_set():
            exit_code = 130

        return exit_code
    finally:
        _log("\n清理资源...")
        shutdown_event.set()

        if mpc is not None:
            try:
                _shutdown_control_process(getattr(mpc, "control_process", None))
            except Exception as exc:
                _log(f"关闭 SAGE-MPPI 资源时出现异常: {exc}")

        if robot is not None:
            try:
                robot.destroy_node()
            except Exception as exc:
                _log(f"销毁 ROS2 节点时出现异常: {exc}")

        if executor is not None:
            try:
                executor.shutdown(timeout_sec=0.0)
            except TypeError:
                pass
            except Exception as exc:
                _log(f"关闭 ROS2 executor 时出现异常: {exc}")

        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as exc:
                _log(f"关闭 ROS2 时出现异常: {exc}")

        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=1.0)
            if spin_thread.is_alive():
                _log("ROS2 spin 线程未在超时内退出")

        _log("程序结束")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UR7e SAGE-MPPI Reach Static Gazebo Tall Scene")
    parser.add_argument("--cuda", action="store_true", default=True, help="使用 CUDA 加速 (默认: True)")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="禁用 CUDA")
    parser.add_argument("--rate", type=float, default=50.0, help="控制频率 Hz (默认: 50)")
    args = parser.parse_args()
    sys.exit(mpc_control_main(args))
