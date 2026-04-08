#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR7e SAGE clean pipeline entry for the Gazebo tall scene.

This example is the intended clean end-to-end entry:
- clean controller core: ``SAGE_MPPI_CORE``
- clean rollout path: ``SageArmReacher``
- clean task assembly: ``SageReacherTaskV3``
- clean config grouping: controller core vs deployment refinement

The example supports two modes:
- ``--offline-smoke``: correctness-only path without ROS2/Gazebo
- default Gazebo mode: real ROS2/Gazebo control loop

Deployment refinement is optional and fully external to the controller core.
"""

from __future__ import annotations

import argparse
import os
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
from storm_kit.mpc.control.sage_mppi_core import SAGE_MPPI_CORE
from storm_kit.mpc.task.sage_reacher_task_v3 import SageReacherTaskV3

np.set_printoptions(precision=3, suppress=True)

EXAMPLE_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.join(EXAMPLE_DIR, "config")
TASK_FILE = os.path.join(CONFIG_DIR, "ur7e_reacher_gazebo_tall_sage_clean.yml")
ROBOT_FILE = os.path.join(CONFIG_DIR, "ur7e_robot_gazebo.yml")
WORLD_FILE = os.path.join(CONFIG_DIR, "collision_world_gazebo_tall.yml")
DEFAULT_GOAL_WORLD = np.array([0.5, -0.45, 0.4], dtype=np.float64)


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


def _get_robot_pose_world(robot_params: dict):
    sim_params = robot_params.get("sim_params", {})
    robot_pose = sim_params.get("robot_pose", [0, 0, 0, 0, 0, 0, 1])
    robot_pos = np.asarray(robot_pose[:3], dtype=np.float64)
    robot_quat_xyzw = np.asarray(robot_pose[3:], dtype=np.float64)
    return robot_pos, robot_quat_xyzw


def _make_clean_task(tensor_args: dict) -> SageReacherTaskV3:
    return SageReacherTaskV3(
        task_file=TASK_FILE,
        robot_file=ROBOT_FILE,
        world_file=WORLD_FILE,
        tensor_args=tensor_args,
    )


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


def _run_offline_smoke(args) -> int:
    from examples.sim_gazebo.reach_static_ur7e import inv_transform_point

    robot_params, _ = _load_robot_and_world_params()
    robot_pos, robot_quat_xyzw = _get_robot_pose_world(robot_params)
    tensor_args = _build_tensor_args(use_cuda=args.cuda)
    mpc = None
    try:
        mpc = _make_clean_task(tensor_args)
        if not isinstance(mpc.controller, SAGE_MPPI_CORE):
            raise RuntimeError("clean example did not instantiate SAGE_MPPI_CORE")
        if isinstance(mpc.controller, SAGE_MPPI):
            raise RuntimeError("clean controller still inherits legacy SAGE_MPPI")

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

        refinement_cfg = dict(mpc.deployment_refinement_config)
        if args.disable_deployment_refinement:
            refinement_cfg["enabled"] = False
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
        print(f"uses_legacy_inheritance={isinstance(mpc.controller, SAGE_MPPI)}")
        print(f"deployment_refinement_enabled={refinement.enabled}")
        print(f"margin_fallback={stats.get('margin_fallback')}")
        print(f"has_minimum_safety_margin={stats.get('minimum_safety_margin') is not None}")
        print(f"enable_stage_scale={stats.get('enable_stage_scale')}")
        print(f"enable_safe_elite_shape={stats.get('enable_safe_elite_shape')}")
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
    from examples.sim_gazebo.reach_static_ur7e import inv_transform_point, transform_point
    from examples.sim_gazebo.reach_static_ur7e_tall import (
        _compute_link_poses_robot_frame,
        _get_top_ee_trajs_world,
        _log,
        _recover_command,
        _reset_control_process_timing,
        _shutdown_control_process,
        CollisionSphereVisualizer,
        TallGazeboRobotInterface,
    )

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
        robot = TallGazeboRobotInterface(joint_names, control_rate=args.rate)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(robot)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
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
        if not isinstance(mpc.controller, SAGE_MPPI_CORE):
            raise RuntimeError("clean Gazebo example did not instantiate SAGE_MPPI_CORE")
        if isinstance(mpc.controller, SAGE_MPPI):
            raise RuntimeError("clean Gazebo example unexpectedly uses legacy SAGE controller inheritance")

        control_dt = mpc.exp_params.get("control_dt", 0.02)
        refinement_cfg = dict(mpc.deployment_refinement_config)
        if args.disable_deployment_refinement:
            refinement_cfg["enabled"] = False
        elif args.enable_deployment_refinement:
            refinement_cfg["enabled"] = True

        refinement = DeploymentRefinementStack(
            mpc=mpc,
            tensor_args=tensor_args,
            refinement_cfg=refinement_cfg,
            reset_timing_fn=_reset_control_process_timing,
            log_fn=_log,
        )
        _log(f"deployment_refinement_enabled={refinement.enabled}")

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
        current_goal_ee = goal_ee_pos_robot.copy()
        current_goal_world = goal_world.copy()
        max_steps = None if args.max_steps <= 0 else args.max_steps
        viz_update_every = max(1, int(args.viz_update_every))

        while running[0] and rclpy.ok() and not shutdown_event.is_set():
            iter_start = time.time()
            t_step = time.time() - loop_start

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
            override_cmd = refinement.maybe_get_override_command(
                error=ee_error,
                q=q,
                dq=dq,
                goal_ee_pos_robot=current_goal_ee,
                t_step=t_step,
                control_dt=control_dt,
            )

            stats = {}
            if override_cmd is None:
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
                        cmd = _recover_command(mpc, t_step, state, control_dt)
                        stats = mpc.get_latest_stats()
                    except Exception as recover_exc:
                        _log(f"[CleanMPC] 恢复失败: {recover_exc}")
                        time.sleep(control_dt)
                        continue
            else:
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
                top_trajs_world = _get_top_ee_trajs_world(
                    mpc,
                    robot_pos,
                    robot_quat_xyzw,
                    max_trajs=5,
                )
                robot.publish_top_trajectories(top_trajs_world)

            if loop_count % 25 == 0:
                _log(
                    f"[{loop_count:5d}] t={t_step:.2f}s | "
                    f"ee_error={ee_error:.4f} | "
                    f"margin_fallback={stats.get('margin_fallback')} | "
                    f"min_margin={stats.get('minimum_safety_margin')} | "
                    f"opt_dt={mpc.opt_dt:.3f}s"
                )

            if max_steps is not None and loop_count >= max_steps:
                break

            elapsed = time.time() - iter_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        return exit_code
    finally:
        shutdown_event.set()
        if mpc is not None:
            try:
                _shutdown_control_process(getattr(mpc, "control_process", None))
            except Exception:
                pass
        if robot is not None:
            try:
                robot.destroy_node()
            except Exception:
                pass
        if executor is not None:
            try:
                executor.shutdown(timeout_sec=0.0)
            except TypeError:
                pass
            except Exception:
                pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=1.0)


def main():
    parser = argparse.ArgumentParser(description="UR7e SAGE clean pipeline Gazebo tall entry")
    parser.add_argument("--cuda", action="store_true", default=True, help="使用 CUDA 加速 (默认: True)")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="禁用 CUDA")
    parser.add_argument("--rate", type=float, default=50.0, help="控制频率 Hz (默认: 50)")
    parser.add_argument("--max-steps", type=int, default=100, help="Gazebo 模式下最大控制步数，<=0 表示不限")
    parser.add_argument("--viz-update-every", type=int, default=1, help="可视化刷新步数间隔")
    parser.add_argument("--offline-smoke", action="store_true", help="运行无需 ROS2/Gazebo 的 clean correctness smoke test")
    parser.add_argument("--enable-deployment-refinement", action="store_true", help="显式启用 deployment refinement")
    parser.add_argument("--disable-deployment-refinement", action="store_true", help="显式禁用 deployment refinement")
    args = parser.parse_args()

    if args.offline_smoke:
        return _run_offline_smoke(args)
    return _run_gazebo_main(args)


if __name__ == "__main__":
    sys.exit(main())
