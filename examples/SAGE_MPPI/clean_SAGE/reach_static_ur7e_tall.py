#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR7e SAGE clean pipeline entry for the Gazebo tall scene.

This example is the intended clean end-to-end entry:
- latest canonical controller: ``SAGE_MPPI``
- clean rollout path: ``SageArmReacher``
- clean task assembly: ``SageReacherTask``
- clean config grouping: controller core vs deployment refinement

The example supports two modes:
- ``--offline-smoke``: correctness-only path without ROS2/Gazebo
- default Gazebo mode: real ROS2/Gazebo control loop

Deployment refinement is optional and fully external to the controller core.
"""

from __future__ import annotations

import argparse
import os
import queue
import signal
import sys
import threading
import time
from typing import Optional

import numpy as np
import torch
import yaml

STORM_ROOT = os.path.expanduser("~/storm")
if STORM_ROOT not in sys.path:
    sys.path.insert(0, STORM_ROOT)

torch.multiprocessing.set_start_method("spawn", force=True)

from examples.SAGE_MPPI.deployment_refinement import DeploymentRefinementStack
from storm_kit.mpc.control.sage_mppi import SAGE_MPPI
from storm_kit.mpc.task.sage_reacher_task import SageReacherTask

np.set_printoptions(precision=3, suppress=True)

EXAMPLE_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.join(EXAMPLE_DIR, "config")
SIM_GAZEBO_CONFIG_DIR = os.path.join(STORM_ROOT, "examples", "sim_gazebo", "config")
OFFICIAL_TASK_FILE = os.path.join(CONFIG_DIR, "ur7e_reacher_gazebo_tall_sage_clean.yml")
TASK_FILE = os.environ.get("SAGE_TASK_FILE", OFFICIAL_TASK_FILE)
ROBOT_FILE = os.path.join(SIM_GAZEBO_CONFIG_DIR, "ur7e_robot_gazebo.yml")
WORLD_FILE = os.path.join(SIM_GAZEBO_CONFIG_DIR, "collision_world_gazebo_tall.yml")
DEFAULT_GOAL_WORLD = np.array([0.4, -0.5, 0.4], dtype=np.float64)


def _build_tensor_args(use_cuda: bool) -> dict:
    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    return {
        "device": torch.device(device, 0) if device == "cuda" else torch.device("cpu"),
        "dtype": torch.float32,
    }


def _load_robot_and_world_params():
    with open(ROBOT_FILE) as f:
        robot_params = yaml.safe_load(f)
    with open(WORLD_FILE) as f:
        world_params = yaml.safe_load(f)
    return robot_params, world_params


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


def _reset_control_process_timing_strict(control_process, t_step: float, control_dt: float) -> None:
    """Reset the ControlProcess timeline after abrupt goal changes.

    The stock ControlProcess assumes the old command horizon remains valid.
    After a large goal jump, that assumption is harmful: stale command timing
    can trigger `find_first_idx()` failures and keep the hot-start distribution
    trapped around the previous solution.
    """
    control_process.command = None
    control_process.command_tstep = control_process.traj_tstep + t_step + control_dt
    control_process.prev_mpc_tstep = max(0.0, t_step - control_dt)
    control_process.mpc_dt = control_dt
    control_process.params = None
    _drain_mp_queue(getattr(control_process, "result_queue", None))
    _drain_mp_queue(getattr(control_process, "opt_queue", None))


def _recover_command_strict(mpc, t_step: float, state: dict, control_dt: float):
    _reset_control_process_timing_strict(mpc.control_process, t_step, control_dt)
    return mpc.get_command(t_step, state, control_dt=control_dt, WAIT=True)


def _get_robot_pose_world(robot_params: dict):
    sim_params = robot_params.get("sim_params", {})
    robot_pose = sim_params.get("robot_pose", [0, 0, 0, 0, 0, 0, 1])
    robot_pos = np.asarray(robot_pose[:3], dtype=np.float64)
    robot_quat_xyzw = np.asarray(robot_pose[3:], dtype=np.float64)
    return robot_pos, robot_quat_xyzw


def _make_clean_task(tensor_args: dict) -> SageReacherTask:
    task = SageReacherTask(
        task_file=TASK_FILE,
        robot_file=ROBOT_FILE,
        world_file=WORLD_FILE,
        tensor_args=tensor_args,
    )
    _apply_execution_mode(task)
    return task


def _get_execution_mode(mpc) -> str:
    mppi_cfg = getattr(mpc, "exp_params", {}).get("mppi", {})
    mode = str(mppi_cfg.get("execution_mode", "mean")).strip().lower()
    if mode not in ("best_sample", "mean"):
        return "mean"
    return mode


def _apply_execution_mode(mpc) -> str:
    mode = _get_execution_mode(mpc)
    controller = getattr(mpc, "controller", None)
    if controller is None:
        return mode
    use_best = mode == "best_sample"
    if hasattr(controller, "execute_best"):
        controller.execute_best = use_best
    return mode


def _configure_default_goal(mpc, robot_pos_world, robot_quat_xyzw, inv_transform_point_fn):
    default_goal_seed_state = np.array(
        [0.5, -1.2, 1.2, -1.57, -1.57, 0.0, 0, 0, 0, 0, 0, 0],
        dtype=np.float64,
    )
    mpc.update_params(goal_state=default_goal_seed_state)
    goal_ee_quat = np.ravel(mpc.controller.rollout_fn.goal_ee_quat.detach().cpu().numpy())
    goal_ee_pos_robot = inv_transform_point_fn(
        robot_pos_world,
        robot_quat_xyzw,
        DEFAULT_GOAL_WORLD,
    )
    mpc.update_params(goal_ee_pos=goal_ee_pos_robot, goal_ee_quat=goal_ee_quat)
    return goal_ee_pos_robot, goal_ee_quat, DEFAULT_GOAL_WORLD.copy()


def _get_clean_top_ee_trajs_world(
    mpc,
    robot_pos_world: np.ndarray,
    robot_quat_xyzw: np.ndarray,
    current_ee_pos_world: np.ndarray,
    transform_point_fn,
    max_trajs: int = 5,
):
    """
    Return world-frame top trajectories with the current EE position prepended.

    The rollout trajectories begin at the first predicted future step, not the
    current EE state. For RViz, prepend the current EE world point so the line
    strip visibly starts from the robot's live end-effector location.
    """
    controller = getattr(mpc, "controller", None)
    top_trajs = getattr(controller, "top_trajs", None)
    if top_trajs is None:
        return None

    if isinstance(top_trajs, torch.Tensor):
        top_trajs_np = top_trajs.detach().cpu().numpy()
    else:
        top_trajs_np = np.asarray(top_trajs)

    if top_trajs_np.ndim == 2:
        top_trajs_np = top_trajs_np[None, ...]
    if top_trajs_np.ndim != 3 or top_trajs_np.shape[-1] != 3:
        return None

    top_count = min(max_trajs, top_trajs_np.shape[0])
    if top_count <= 0:
        return None

    current_ee_pos_world = np.asarray(current_ee_pos_world, dtype=np.float64).reshape(1, 3)
    world_trajs = []
    for traj_points_robot in top_trajs_np[:top_count]:
        traj_points_world = transform_point_fn(
            robot_pos_world,
            robot_quat_xyzw,
            traj_points_robot,
        )
        if traj_points_world.ndim != 2 or traj_points_world.shape[-1] != 3:
            continue
        if len(traj_points_world) == 0:
            continue

        # Avoid duplicate first point when the predicted trajectory already
        # starts extremely close to the live EE position.
        if np.linalg.norm(traj_points_world[0] - current_ee_pos_world[0]) < 1.0e-4:
            stitched = traj_points_world
        else:
            stitched = np.concatenate([current_ee_pos_world, traj_points_world], axis=0)
        world_trajs.append(stitched)

    if not world_trajs:
        return None
    return np.asarray(world_trajs, dtype=np.float64)


def _apply_refinement_overrides(refinement_cfg: dict, args) -> dict:
    refinement_cfg = dict(refinement_cfg or {})
    if "cartesian_refinement" in refinement_cfg:
        refinement_cfg["cartesian_refinement"] = dict(refinement_cfg["cartesian_refinement"])
    if "local_refinement" in refinement_cfg:
        refinement_cfg["local_refinement"] = dict(refinement_cfg["local_refinement"])

    if args.disable_deployment_refinement:
        refinement_cfg["enabled"] = False
        return refinement_cfg

    if args.enable_deployment_refinement:
        refinement_cfg["enabled"] = True

    local_cfg = dict(refinement_cfg.get("local_refinement", refinement_cfg.get("cartesian_refinement", {})))
    if args.enable_cartesian_refinement:
        refinement_cfg["enabled"] = True
        local_cfg["enabled"] = True
    elif args.disable_cartesian_refinement:
        local_cfg["enabled"] = False

    if local_cfg:
        refinement_cfg["local_refinement"] = local_cfg
    return refinement_cfg


def _run_offline_smoke(args) -> int:
    from examples.sim_gazebo.reach_static_ur7e import inv_transform_point

    robot_params, _ = _load_robot_and_world_params()
    robot_pos, robot_quat_xyzw = _get_robot_pose_world(robot_params)
    tensor_args = _build_tensor_args(use_cuda=args.cuda)
    mpc = None
    try:
        mpc = _make_clean_task(tensor_args)
        if not isinstance(mpc.controller, SAGE_MPPI):
            raise RuntimeError("clean example did not instantiate latest canonical SAGE_MPPI")

        goal_ee_pos_robot, goal_ee_quat, goal_world = _configure_default_goal(
            mpc,
            robot_pos,
            robot_quat_xyzw,
            inv_transform_point,
        )

        init_q = np.asarray(mpc.exp_params["model"]["init_state"], dtype=np.float64)
        zero = np.zeros_like(init_q)
        state = {
            "position": init_q.copy(),
            "velocity": zero.copy(),
            "acceleration": zero.copy(),
        }

        refinement_cfg = _apply_refinement_overrides(
            mpc.deployment_refinement_config,
            args,
        )
        refinement = DeploymentRefinementStack(
            mpc=mpc,
            tensor_args=tensor_args,
            refinement_cfg=refinement_cfg,
            reset_timing_fn=lambda control_process, t_step, control_dt: None,
        )

        cmd, stats = mpc.get_command_and_stats(
            0.0,
            state,
            control_dt=mpc.exp_params.get("control_dt", 0.02),
            WAIT=True,
        )
        ee_pose = mpc.controller.rollout_fn.get_ee_pose(
            torch.as_tensor(np.hstack([init_q, zero, zero]), **tensor_args).unsqueeze(0)
        )
        ee_pos_robot = np.ravel(ee_pose["ee_pos_seq"].detach().cpu().numpy())
        ee_error = float(np.linalg.norm(ee_pos_robot - goal_ee_pos_robot))

        print("offline_smoke_ok=True")
        print(f"controller_class={type(mpc.controller).__name__}")
        print(f"uses_canonical_sage_controller={isinstance(mpc.controller, SAGE_MPPI)}")
        print(f"deployment_refinement_enabled={refinement.enabled}")
        print(f"local_refinement_enabled={refinement.local_refinement is not None}")
        print(f"cartesian_refinement_enabled={refinement.cartesian is not None}")
        print(f"enable_stage_scale={stats.get('enable_stage_scale')}")
        print(f"enable_anisotropic_shape_update={stats.get('enable_anisotropic_shape_update')}")
        print(f"enable_stagnation_amplification={stats.get('enable_stagnation_amplification')}")
        print(f"cmd_keys={sorted(cmd.keys())}")
        print(f"initial_goal_error={ee_error:.6f}")
        print(f"goal_world={np.round(goal_world, 4).tolist()}")
        return 0
    finally:
        if mpc is not None:
            try:
                mpc.close()
            except Exception:
                pass


def _run_gazebo_main(args) -> int:
    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from std_msgs.msg import ColorRGBA
        from visualization_msgs.msg import Marker, MarkerArray
    except ImportError:
        print("=" * 60)
        print("错误: 未找到 ROS2 Python 包")
        print("请先 source ROS2 环境:")
        print("  source /opt/ros/humble/setup.bash")
        print("=" * 60)
        return 1

    from examples.sim_gazebo.gazebo_obstacle_utils import (
        count_primitive_obstacles,
        spawn_gazebo_obstacles,
    )
    from examples.sim_gazebo.reach_static_ur7e import (
        GazeboRobotInterface,
        inv_transform_point,
        transform_point,
    )
    from examples.sim_gazebo.reach_static_ur7e_tall import (
        _compute_link_poses_robot_frame,
        _log,
        _shutdown_control_process,
        CollisionSphereVisualizer,
        TallGazeboRobotInterface,
    )

    class CleanTallGazeboRobotInterface(TallGazeboRobotInterface):
        """
        Clean-SAGE visualizer with collision spheres on a dedicated topic.

        This keeps the live EE/goal/world markers separate from the robot
        collision-envelope markers, so RViz can toggle them independently.
        """

        def __init__(self, joint_names: list, control_rate: float = 50.0):
            super().__init__(joint_names, control_rate=control_rate)
            self._latest_sim_time = None

        def _joint_state_callback(self, msg):
            self._latest_sim_time = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1.0e-9
            super()._joint_state_callback(msg)

        def get_latest_sim_time(self):
            return self._latest_sim_time

        def publish_live_goal_ee_markers(self, goal_pos: np.ndarray, ee_pos: np.ndarray):
            marker_array = MarkerArray()
            stamp = self.get_clock().now().to_msg()

            goal_marker = Marker()
            goal_marker.header.frame_id = "world"
            goal_marker.header.stamp = stamp
            goal_marker.ns = "goal"
            goal_marker.id = 0
            goal_marker.type = Marker.SPHERE
            goal_marker.action = Marker.ADD
            goal_marker.pose.position.x = float(goal_pos[0])
            goal_marker.pose.position.y = float(goal_pos[1])
            goal_marker.pose.position.z = float(goal_pos[2])
            goal_marker.pose.orientation.w = 1.0
            goal_marker.scale.x = 0.06
            goal_marker.scale.y = 0.06
            goal_marker.scale.z = 0.06
            goal_marker.color = ColorRGBA(r=0.9, g=0.1, b=0.1, a=0.8)
            marker_array.markers.append(goal_marker)

            ee_marker = Marker()
            ee_marker.header.frame_id = "world"
            ee_marker.header.stamp = stamp
            ee_marker.ns = "ee"
            ee_marker.id = 1
            ee_marker.type = Marker.SPHERE
            ee_marker.action = Marker.ADD
            ee_marker.pose.position.x = float(ee_pos[0])
            ee_marker.pose.position.y = float(ee_pos[1])
            ee_marker.pose.position.z = float(ee_pos[2])
            ee_marker.pose.orientation.w = 1.0
            ee_marker.scale.x = 0.05
            ee_marker.scale.y = 0.05
            ee_marker.scale.z = 0.05
            ee_marker.color = ColorRGBA(r=0.1, g=0.9, b=0.1, a=0.8)
            marker_array.markers.append(ee_marker)

            self.pub_markers.publish(marker_array)

        def publish_markers(
            self,
            obstacles: dict,
            goal_pos: np.ndarray,
            ee_pos: np.ndarray,
            collision_spheres=None,
        ):
            GazeboRobotInterface.publish_markers(self, obstacles, goal_pos, ee_pos)

            marker_array = MarkerArray()
            stamp = self.get_clock().now().to_msg()
            current_count = 0

            if collision_spheres:
                for marker_id, sphere in enumerate(collision_spheres):
                    marker = Marker()
                    marker.header.frame_id = "world"
                    marker.header.stamp = stamp
                    marker.ns = "collision_spheres"
                    marker.id = int(sphere.get("marker_id", marker_id))
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.frame_locked = True
                    marker.pose.position.x = float(sphere["center_world"][0])
                    marker.pose.position.y = float(sphere["center_world"][1])
                    marker.pose.position.z = float(sphere["center_world"][2])
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 2.0 * sphere["radius"]
                    marker.scale.y = 2.0 * sphere["radius"]
                    marker.scale.z = 2.0 * sphere["radius"]
                    marker.color = ColorRGBA(r=1.0, g=0.78, b=0.12, a=0.45)
                    marker_array.markers.append(marker)
                current_count = len(collision_spheres)

            for marker_id in range(current_count, self._prev_collision_marker_count):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = stamp
                marker.ns = "collision_spheres"
                marker.id = marker_id
                marker.action = Marker.DELETE
                marker_array.markers.append(marker)

            self._prev_collision_marker_count = current_count
            self.pub_collision_sphere_markers.publish(marker_array)

    robot_params, world_params = _load_robot_and_world_params()
    robot_pos, robot_quat_xyzw = _get_robot_pose_world(robot_params)
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    n_dof = len(joint_names)
    tensor_args = _build_tensor_args(use_cuda=args.cuda)

    robot = None
    executor = None
    spin_thread = None
    spin_running = [True]
    mpc = None
    exit_code = 0
    shutdown_event = threading.Event()

    try:
        _log("=" * 60)
        _log("UR7e SAGE CLEAN MPC Reach Static - Gazebo Tall Scene")
        _log("=" * 60)
        _log(f"Task:  {TASK_FILE}")
        _log(f"Robot: {ROBOT_FILE}")
        _log(f"World: {WORLD_FILE}")

        rclpy.init(args=None)
        robot = CleanTallGazeboRobotInterface(joint_names, control_rate=args.rate)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(robot)

        def spin_with_check():
            while spin_running[0] and rclpy.ok() and not shutdown_event.is_set():
                try:
                    executor.spin_once(timeout_sec=0.1)
                except RuntimeError as exc:
                    # During shutdown, ROS2 may reject work on objects already
                    # marked for destruction. Treat that as a normal exit path.
                    if "Destroyable" in str(exc):
                        break
                    raise

        spin_thread = threading.Thread(target=spin_with_check, daemon=True)
        spin_thread.start()

        _log("\n等待 Gazebo 关节状态...")
        start = time.time()
        while not robot.is_connected():
            if shutdown_event.is_set():
                return 130
            if time.time() - start > 10.0:
                _log("错误: 无法接收关节状态，请确保 Gazebo 仿真正在运行")
                return 1
            time.sleep(0.1)

        _log("已连接到 Gazebo 机器人!")
        n_world_spheres, n_world_cubes = count_primitive_obstacles(world_params, include_ground=False)
        if spawn_gazebo_obstacles(robot, world_params, model_prefix="sage_clean", include_ground=False):
            _log(
                "Gazebo 真实障碍物已生成: spheres=%d cubes=%d"
                % (n_world_spheres, n_world_cubes)
            )
        else:
            _log("警告: Gazebo 真实障碍物未完整生成，请检查 /spawn_entity 与 /delete_entity 服务")

        mpc = _make_clean_task(tensor_args)
        if not isinstance(mpc.controller, SAGE_MPPI):
            raise RuntimeError("clean Gazebo example did not instantiate latest canonical SAGE_MPPI")

        control_dt = mpc.exp_params.get("control_dt", 0.02)
        refinement_cfg = _apply_refinement_overrides(
            mpc.deployment_refinement_config,
            args,
        )

        refinement = DeploymentRefinementStack(
            mpc=mpc,
            tensor_args=tensor_args,
            refinement_cfg=refinement_cfg,
            reset_timing_fn=_reset_control_process_timing_strict,
            log_fn=_log,
        )
        _log(f"deployment_refinement_enabled={refinement.enabled}")
        _log(f"local_refinement_enabled={refinement.local_refinement is not None}")
        _log(f"cartesian_refinement_enabled={refinement.cartesian is not None}")
        _log(f"execution_mode={_get_execution_mode(mpc)}")
        _log("说明: /target_pose 只读取 position.x/y/z, 发布的 orientation 不参与目标更新")
        _log(
            "说明: deployment refinement / hold / Cartesian refinement 只生成关节目标; "
            "运行时仍统一通过 /forward_position_controller/commands 下发, 不直接写 Gazebo 关节状态"
        )

        goal_ee_pos_robot, goal_ee_quat, goal_world = _configure_default_goal(
            mpc,
            robot_pos,
            robot_quat_xyzw,
            inv_transform_point,
        )
        _log(f"默认目标末端位置 (机器人坐标系): {goal_ee_pos_robot}")
        _log(f"默认目标末端位置 (世界坐标系): {goal_world}")

        rollout_fn = mpc.controller.rollout_fn
        collision_sphere_visualizer = CollisionSphereVisualizer(
            mpc.exp_params["model"]["robot_collision_params"]
        )

        running = [True]

        def shutdown_handler(sig, frame):
            running[0] = False
            shutdown_event.set()

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        current_state = robot.get_state()
        _log("预热 clean MPC 控制器...")
        for warm_idx in range(3):
            if current_state is None:
                time.sleep(0.01)
                current_state = robot.get_state()
                continue
            try:
                mpc.get_command_and_stats(
                    warm_idx * control_dt,
                    current_state,
                    control_dt=control_dt,
                    WAIT=True,
                )
            except Exception as exc:
                _log(f"预热异常 (可忽略): {exc}")
            time.sleep(0.01)

        loop_count = 0
        loop_start = time.time()
        last_wall_time = None
        last_sim_time = None
        current_goal_ee = goal_ee_pos_robot.copy()
        current_goal_world = goal_world.copy()
        max_steps = None if args.max_steps <= 0 else args.max_steps
        viz_update_every = max(1, int(args.viz_update_every))

        while running[0] and rclpy.ok() and not shutdown_event.is_set():
            iter_start = time.time()
            t_step = time.time() - loop_start
            actual_loop_dt_wall = None if last_wall_time is None else (iter_start - last_wall_time)
            current_sim_time = robot.get_latest_sim_time()
            actual_loop_dt_sim = None if last_sim_time is None or current_sim_time is None else (current_sim_time - last_sim_time)
            last_wall_time = iter_start
            last_sim_time = current_sim_time

            state = robot.get_state()
            if state is None:
                time.sleep(control_dt)
                continue

            q = state["position"]
            dq = state["velocity"]
            ddq = state["acceleration"]
            curr = np.hstack([q, dq, ddq])

            new_target = robot.get_target_position()
            if new_target is not None:
                target_robot = inv_transform_point(robot_pos, robot_quat_xyzw, new_target)
                if np.linalg.norm(target_robot - current_goal_ee) > 0.005:
                    current_goal_ee = target_robot.copy()
                    current_goal_world = new_target.copy()
                    mpc.update_params(goal_ee_pos=current_goal_ee, goal_ee_quat=goal_ee_quat)
                    refinement.on_goal_changed(t_step, control_dt)
                    _log(
                        "[目标更新] 世界: %s, 机器人: %s"
                        % (np.round(current_goal_world, 3), np.round(current_goal_ee, 3))
                    )

            ee_pose = rollout_fn.get_ee_pose(torch.as_tensor(curr, **tensor_args).unsqueeze(0))
            ee_pos_robot = np.ravel(ee_pose["ee_pos_seq"].detach().cpu().numpy())
            ee_pos_world = transform_point(robot_pos, robot_quat_xyzw, ee_pos_robot)
            ee_error = float(np.linalg.norm(ee_pos_world - current_goal_world))

            refinement.update_modes(
                error=ee_error,
                q=q,
                dq=dq,
                t_step=t_step,
                control_dt=control_dt,
            )
            stats = {}
            try:
                cmd, stats = mpc.get_command_and_stats(
                    t_step,
                    state,
                    control_dt=control_dt,
                    WAIT=True,
                )
            except Exception as exc:
                _log(f"[CleanMPC] 同步取命令失败，执行恢复重规划: {exc}")
                try:
                    cmd = _recover_command_strict(mpc, t_step, state, control_dt)
                    stats = mpc.get_latest_stats()
                except Exception as recover_exc:
                    _log(f"[CleanMPC] 恢复失败: {recover_exc}")
                    time.sleep(control_dt)
                    continue

            nominal_position_cmd = None
            if cmd is not None and "position" in cmd:
                nominal_position_cmd = cmd["position"]
                if isinstance(nominal_position_cmd, torch.Tensor):
                    nominal_position_cmd = nominal_position_cmd.detach().cpu().numpy()

            override_cmd = refinement.maybe_get_override_command(
                error=ee_error,
                q=q,
                dq=dq,
                goal_ee_pos_robot=current_goal_ee,
                t_step=t_step,
                control_dt=control_dt,
                nominal_position_cmd=nominal_position_cmd,
            )
            if override_cmd is not None:
                cmd = override_cmd

            if cmd is None or "position" not in cmd:
                time.sleep(control_dt)
                continue

            target_positions = cmd["position"]
            if isinstance(target_positions, torch.Tensor):
                target_positions = target_positions.detach().cpu().numpy()
            target_positions = np.asarray(target_positions, dtype=np.float64).flatten()[:n_dof]
            robot.send_position_command(target_positions)
            robot.publish_ee_pose(ee_pos_world)
            robot.publish_live_goal_ee_markers(current_goal_world, ee_pos_world)

            if refinement.enabled:
                refinement.maybe_trigger_recovery(
                    t_step=t_step,
                    ee_pos_world=ee_pos_world,
                    goal_world=current_goal_world,
                    joint_velocity=dq,
                    control_dt=control_dt,
                )

            loop_count += 1
            if loop_count % viz_update_every == 0:
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
                top_trajs_world = _get_clean_top_ee_trajs_world(
                    mpc,
                    robot_pos,
                    robot_quat_xyzw,
                    current_ee_pos_world=ee_pos_world,
                    transform_point_fn=transform_point,
                    max_trajs=5,
                )
                robot.publish_top_trajectories(top_trajs_world)

            if loop_count % 25 == 0:
                local_stats = dict(getattr(refinement, "latest_local_refinement_stats", {}) or {})
                runtime_stats_enabled = bool(stats.get("enable_runtime_stats", False))
                if runtime_stats_enabled:
                    _log(
                        f"[{loop_count:5d}] t={t_step:.2f}s | "
                        f"ee_error={ee_error:.4f} | "
                        f"weight_entropy={float(stats.get('weight_entropy', 0.0)):.4f} | "
                        f"cov_fallbacks={int(stats.get('covariance_fallback_count', 0))} | "
                        f"near_goal={bool(stats.get('near_goal_active', False))} | "
                        f"ng_skip_cnt={int(stats.get('shape_skip_count_near_goal', 0))} | "
                        f"ng_low_ent={int(stats.get('low_entropy_trigger_count_near_goal', 0))} | "
                        f"ng_fb={int(stats.get('fallback_fraction_trigger_count_near_goal', 0))} | "
                        f"ng_prev_shape={bool(stats.get('near_goal_shape_update_used_previous_shape', False))} | "
                        f"ng_scale={float(stats.get('near_goal_proposal_scale', 0.0)):.4f} | "
                        f"shape_temp={float(stats.get('shape_temperature_used', 1.0)):.2f} | "
                        f"ng_floor={bool(stats.get('near_goal_scale_floor_active', False))} | "
                        f"ng_scale_floor={float(stats.get('near_goal_scale_after_floor', 0.0)):.4f} | "
                        f"ng_cond={float(stats.get('near_goal_shape_condition', 1.0)):.2f} | "
                        f"lr_active={bool(local_stats.get('local_refinement_active', False))} | "
                        f"lr_mode={local_stats.get('local_refinement_mode', 'off')} | "
                        f"lr_step={float(local_stats.get('local_refinement_step_norm', 0.0)):.4f} | "
                        f"lr_gain={float(local_stats.get('local_refinement_gain_used', 0.0)):.2f} | "
                        f"lr_blend={float(local_stats.get('local_refinement_blend_ratio', 0.0)):.2f} | "
                        f"actual_loop_dt_wall={float('nan') if actual_loop_dt_wall is None else actual_loop_dt_wall:.3f}s | "
                        f"actual_loop_dt_sim={float('nan') if actual_loop_dt_sim is None else actual_loop_dt_sim:.3f}s | "
                        f"opt_dt={mpc.opt_dt:.3f}s"
                    )
                else:
                    _log(
                        f"[{loop_count:5d}] t={t_step:.2f}s | "
                        f"ee_error={ee_error:.4f} | "
                        f"shape_skip={bool(stats.get('shape_update_skipped', False))} | "
                        f"reason={stats.get('shape_skip_reason', '') or '-'} | "
                        f"near_goal={bool(stats.get('near_goal_active', False))} | "
                        f"ng_skip_cnt={int(stats.get('shape_skip_count_near_goal', 0))} | "
                        f"ng_low_ent={int(stats.get('low_entropy_trigger_count_near_goal', 0))} | "
                        f"ng_fb={int(stats.get('fallback_fraction_trigger_count_near_goal', 0))} | "
                        f"ng_prev_shape={bool(stats.get('near_goal_shape_update_used_previous_shape', False))} | "
                        f"ng_scale={float(stats.get('near_goal_proposal_scale', 0.0)):.4f} | "
                        f"shape_temp={float(stats.get('shape_temperature_used', 1.0)):.2f} | "
                        f"ng_floor={bool(stats.get('near_goal_scale_floor_active', False))} | "
                        f"ng_scale_floor={float(stats.get('near_goal_scale_after_floor', 0.0)):.4f} | "
                        f"ng_cond={float(stats.get('near_goal_shape_condition', 1.0)):.2f} | "
                        f"lr_active={bool(local_stats.get('local_refinement_active', False))} | "
                        f"lr_mode={local_stats.get('local_refinement_mode', 'off')} | "
                        f"lr_step={float(local_stats.get('local_refinement_step_norm', 0.0)):.4f} | "
                        f"lr_gain={float(local_stats.get('local_refinement_gain_used', 0.0)):.2f} | "
                        f"lr_blend={float(local_stats.get('local_refinement_blend_ratio', 0.0)):.2f} | "
                        f"actual_loop_dt_wall={float('nan') if actual_loop_dt_wall is None else actual_loop_dt_wall:.3f}s | "
                        f"actual_loop_dt_sim={float('nan') if actual_loop_dt_sim is None else actual_loop_dt_sim:.3f}s | "
                        f"opt_dt={mpc.opt_dt:.3f}s"
                    )

            if max_steps is not None and loop_count >= max_steps:
                _log(f"达到 max_steps={max_steps}，按配置退出 clean 控制循环")
                break

            elapsed = time.time() - iter_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        return exit_code
    finally:
        shutdown_event.set()
        spin_running[0] = False
        if mpc is not None:
            try:
                _shutdown_control_process(getattr(mpc, "control_process", None))
            except Exception:
                pass
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=0.0)
            except TypeError:
                pass
            except Exception:
                pass
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=1.0)
        if robot is not None:
            try:
                robot.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="UR7e SAGE clean pipeline Gazebo tall entry")
    parser.add_argument("--cuda", action="store_true", default=True, help="使用 CUDA 加速 (默认: True)")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="禁用 CUDA")
    parser.add_argument("--rate", type=float, default=50.0, help="控制频率 Hz (默认: 50)")
    parser.add_argument("--max-steps", type=int, default=0, help="Gazebo 模式下最大控制步数，<=0 表示不限")
    parser.add_argument("--viz-update-every", type=int, default=5, help="可视化刷新步数间隔")
    parser.add_argument("--offline-smoke", action="store_true", help="运行无需 ROS2/Gazebo 的 clean correctness smoke test")
    parser.add_argument("--enable-deployment-refinement", action="store_true", help="显式启用 deployment refinement")
    parser.add_argument("--disable-deployment-refinement", action="store_true", help="显式禁用 deployment refinement")
    parser.add_argument("--enable-cartesian-refinement", action="store_true", help="显式启用 Cartesian refinement")
    parser.add_argument("--disable-cartesian-refinement", action="store_true", help="显式禁用 Cartesian refinement")
    args = parser.parse_args()

    if args.offline_smoke:
        return _run_offline_smoke(args)
    return _run_gazebo_main(args)


if __name__ == "__main__":
    sys.exit(main())
