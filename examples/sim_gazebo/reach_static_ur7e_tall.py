#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR7e STORM MPC Reach Static Target - Gazebo high-wall scene.

This keeps the original sim_gazebo reach_static flow:
- STORM MPPI-MPC controller
- primitive world avoidance
- dynamic goal update via /target_pose
- RViz visualization for obstacles, goal, and end-effector

Only the task/world configs are swapped to the tall-scene variants.
"""

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time

import numpy as np
import torch
import yaml

STORM_ROOT = os.path.expanduser('~/storm')
if STORM_ROOT not in sys.path:
    sys.path.insert(0, STORM_ROOT)

torch.multiprocessing.set_start_method('spawn', force=True)

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError:
    print("=" * 60)
    print("错误: 未找到 ROS2 Python 包")
    print("请先 source ROS2 环境:")
    print("  source /opt/ros/humble/setup.bash")
    print("=" * 60)
    sys.exit(1)

from examples.sim_gazebo.reach_static_ur7e import (
    GazeboReacherTask,
    GazeboRobotInterface,
    inv_transform_point,
    transform_point,
)
from examples.sim_gazebo.gazebo_obstacle_utils import (
    count_primitive_obstacles,
    spawn_gazebo_obstacles,
)
from storm_kit.mpc.utils.mpc_process_wrapper import ControlProcess
from storm_kit.util_file import get_mpc_configs_path, join_path

np.set_printoptions(precision=3, suppress=True)

OFFICIAL_TASK_FILE = os.path.join(os.path.dirname(__file__), "config", "ur7e_reacher_gazebo_tall.yml")
TASK_FILE = os.environ.get("STORM_TASK_FILE", OFFICIAL_TASK_FILE)
DEFAULT_GOAL_WORLD = np.array(
    json.loads(os.environ.get("STORM_DEFAULT_GOAL_WORLD", "[0.5, -0.45, 0.4]")),
    dtype=np.float64,
)


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
    for method_name in ("close", "cancel_join_thread"):
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
    _drain_mp_queue(getattr(control_process, "result_queue", None))

    done_message = {"state": None, "dt": None, "done": True, "params": None}
    opt_queue = getattr(control_process, "opt_queue", None)
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

    opt_process = getattr(control_process, "opt_process", None)
    if opt_process is not None:
        opt_process.join(timeout=join_timeout)
        if opt_process.is_alive():
            _log("后台 MPC 进程未在超时内退出，强制终止...")
            opt_process.terminate()
            opt_process.join(timeout=join_timeout)

    if opt_queue is not None:
        _close_mp_queue(opt_queue)
    result_queue = getattr(control_process, "result_queue", None)
    if result_queue is not None:
        _close_mp_queue(result_queue)


def _command_horizon_exhausted(control_process, t_step: float) -> bool:
    command = getattr(control_process, "command", None)
    command_tstep = getattr(control_process, "command_tstep", None)
    if command is None or command_tstep is None:
        return False

    try:
        if len(command_tstep) == 0:
            return True
        last_t = float(command_tstep[-1].item() if hasattr(command_tstep[-1], "item") else command_tstep[-1])
    except Exception:
        return False
    return t_step >= last_t


def _reset_control_process_timing(control_process, t_step: float, control_dt: float) -> None:
    control_process.command = None
    control_process.command_tstep = control_process.traj_tstep + t_step
    control_process.prev_mpc_tstep = max(0.0, t_step - control_dt)
    control_process.mpc_dt = control_dt
    _drain_mp_queue(getattr(control_process, "result_queue", None))
    _drain_mp_queue(getattr(control_process, "opt_queue", None))


def _recover_command(mpc, t_step: float, state: dict, control_dt: float):
    control_process = mpc.control_process
    _reset_control_process_timing(control_process, t_step, control_dt)
    return _get_sync_command(mpc, t_step, state, control_dt)


def _get_execution_mode(mpc) -> str:
    mppi_cfg = getattr(mpc, "exp_params", {}).get("mppi", {})
    mode = str(mppi_cfg.get("execution_mode", "best_sample")).strip().lower()
    if mode not in ("best_sample", "mean"):
        return "best_sample"
    return mode


def _get_sync_command(mpc, t_step: float, state: dict, control_dt: float):
    filt_state = mpc.state_filter.filter_joint_state(state)
    state_tensor = mpc._state_to_tensor(filt_state)
    next_command, _, _, _ = mpc.control_process.get_command_debug(
        t_step,
        state_tensor.numpy(),
        control_dt=control_dt,
    )

    qdd_des = np.asarray(next_command, dtype=np.float64)
    execution_mode = _get_execution_mode(mpc)
    if execution_mode == "best_sample" and mpc.exp_params.get("control_space", "acc") == "acc":
        best_traj = getattr(mpc.controller, "best_traj", None)
        if best_traj is not None:
            if isinstance(best_traj, torch.Tensor):
                best_traj_np = best_traj.detach().cpu().numpy()
            else:
                best_traj_np = np.asarray(best_traj)
            if best_traj_np.ndim == 2 and best_traj_np.shape[0] > 0 and best_traj_np.shape[1] == mpc.n_dofs:
                qdd_des = np.asarray(best_traj_np[0], dtype=np.float64)
                if getattr(mpc.control_process, "command", None) is not None:
                    mpc.control_process.command[0] = best_traj_np

    mpc.prev_qdd_des = qdd_des
    cmd_des = mpc.state_filter.integrate_acc(qdd_des)
    return cmd_des


def _restart_control_process(mpc, join_timeout: float = 0.2) -> None:
    old_control_process = getattr(mpc, "control_process", None)
    _shutdown_control_process(old_control_process, join_timeout=join_timeout)
    mpc.control_process = ControlProcess(mpc.controller)


class CollisionSphereVisualizer:
    def __init__(self, robot_collision_params: dict):
        sphere_config = os.path.expanduser(robot_collision_params["collision_spheres"])
        if not os.path.isabs(sphere_config):
            sphere_config = join_path(get_mpc_configs_path(), sphere_config)

        with open(sphere_config) as f:
            sphere_params = yaml.safe_load(f)

        self.link_names = list(robot_collision_params["link_objs"])
        raw_spheres_by_link = sphere_params["collision_spheres"]
        self.spheres_by_link = {}
        marker_id = 0
        for link_name in self.link_names:
            self.spheres_by_link[link_name] = []
            for sphere in raw_spheres_by_link.get(link_name, []):
                sphere_entry = dict(sphere)
                sphere_entry["marker_id"] = marker_id
                self.spheres_by_link[link_name].append(sphere_entry)
                marker_id += 1
        self.total_sphere_count = marker_id

    def get_world_spheres(
        self,
        link_pos_robot: np.ndarray,
        link_rot_robot: np.ndarray,
        robot_pos_world: np.ndarray,
        robot_quat_xyzw: np.ndarray,
    ):
        world_spheres = []
        for link_idx, link_name in enumerate(self.link_names):
            link_pos = link_pos_robot[link_idx]
            link_rot = link_rot_robot[link_idx]
            for sphere in self.spheres_by_link.get(link_name, []):
                center_local = np.asarray(sphere["center"], dtype=np.float64)
                center_robot = link_rot @ center_local + link_pos
                center_world = transform_point(robot_pos_world, robot_quat_xyzw, center_robot)
                world_spheres.append(
                    {
                        "marker_id": int(sphere["marker_id"]),
                        "link_name": link_name,
                        "center_world": center_world,
                        "radius": float(sphere["radius"]),
                    }
                )
        return world_spheres


class TallGazeboRobotInterface(GazeboRobotInterface):
    def __init__(self, joint_names: list, control_rate: float = 50.0):
        super().__init__(joint_names, control_rate=control_rate)

    def publish_markers(
        self,
        obstacles: dict,
        goal_pos: np.ndarray,
        ee_pos: np.ndarray,
        collision_spheres=None,
    ):
        super().publish_markers(obstacles, goal_pos, ee_pos)

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

    def publish_top_trajectories(self, top_trajs_world):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        clear_marker = Marker()
        clear_marker.header.frame_id = "world"
        clear_marker.header.stamp = stamp
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        if top_trajs_world is None or len(top_trajs_world) == 0:
            self.pub_top_traj_markers.publish(marker_array)
            return

        for traj_id, traj_points in enumerate(top_trajs_world[:5]):
            line_marker = Marker()
            line_marker.header.frame_id = "world"
            line_marker.header.stamp = stamp
            line_marker.ns = "mppi_top_trajs"
            line_marker.id = traj_id
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.pose.orientation.w = 1.0
            line_marker.scale.x = 0.002
            line_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.95)
            line_marker.points = []

            for point in np.asarray(traj_points, dtype=np.float64):
                if not np.all(np.isfinite(point)):
                    continue
                point_msg = Point()
                point_msg.x = float(point[0])
                point_msg.y = float(point[1])
                point_msg.z = float(point[2])
                line_marker.points.append(point_msg)

            if len(line_marker.points) >= 2:
                marker_array.markers.append(line_marker)

        self.pub_top_traj_markers.publish(marker_array)


def _compute_link_poses_robot_frame(rollout_fn, q: np.ndarray, dq: np.ndarray, tensor_args: dict):
    q_tensor = torch.as_tensor(q, **tensor_args).unsqueeze(0)
    dq_tensor = torch.as_tensor(dq, **tensor_args).unsqueeze(0)
    robot_model = rollout_fn.dynamics_model.robot_model
    robot_model.compute_fk_and_jacobian(
        q_tensor,
        dq_tensor,
        rollout_fn.exp_params["model"]["ee_link_name"],
    )

    link_pos_robot = []
    link_rot_robot = []
    for link_name in rollout_fn.dynamics_model.link_names:
        link_pos, link_rot = robot_model.get_link_pose(link_name)
        link_pos_robot.append(link_pos[0].detach().cpu().numpy())
        link_rot_robot.append(link_rot[0].detach().cpu().numpy())

    return np.stack(link_pos_robot, axis=0), np.stack(link_rot_robot, axis=0)


def _get_top_ee_trajs_world(
    mpc,
    robot_pos_world: np.ndarray,
    robot_quat_xyzw: np.ndarray,
    current_ee_pos_world: np.ndarray,
    max_trajs: int = 5,
):
    controller = getattr(mpc, "controller", None)
    trajectories = getattr(controller, "trajectories", None)
    total_costs = getattr(controller, "total_costs", None)

    if trajectories is not None and total_costs is not None:
        ee_pos_seq = trajectories.get("ee_pos_seq", None)
        if ee_pos_seq is not None:
            if isinstance(ee_pos_seq, torch.Tensor):
                ee_pos_seq_np = ee_pos_seq.detach().cpu().numpy()
            else:
                ee_pos_seq_np = np.asarray(ee_pos_seq)

            if isinstance(total_costs, torch.Tensor):
                total_costs_np = total_costs.detach().cpu().numpy()
            else:
                total_costs_np = np.asarray(total_costs)

            if ee_pos_seq_np.ndim == 3 and ee_pos_seq_np.shape[-1] == 3 and total_costs_np.ndim == 1:
                top_count = min(max_trajs, ee_pos_seq_np.shape[0], total_costs_np.shape[0])
                if top_count > 0:
                    top_indices = np.argsort(total_costs_np)[:top_count]
                    top_trajs_np = ee_pos_seq_np[top_indices]
                else:
                    return None
            else:
                return None
        else:
            return None
    else:
        top_trajs = getattr(mpc, "top_trajs", None)
        if top_trajs is None and controller is not None:
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
    top_trajs_np = top_trajs_np[:top_count]

    current_ee_pos_world = np.asarray(current_ee_pos_world, dtype=np.float64).reshape(1, 3)
    world_trajs = []
    for traj_points in top_trajs_np:
        traj_points_world = transform_point(robot_pos_world, robot_quat_xyzw, traj_points)
        if traj_points_world.ndim != 2 or traj_points_world.shape[-1] != 3:
            continue
        if len(traj_points_world) == 0:
            continue
        if np.linalg.norm(traj_points_world[0] - current_ee_pos_world[0]) < 1.0e-4:
            stitched = traj_points_world
        else:
            stitched = np.concatenate([current_ee_pos_world, traj_points_world], axis=0)
        world_trajs.append(stitched)

    if not world_trajs:
        return None
    return world_trajs


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
        _log("UR7e STORM MPC Reach Static - Gazebo 高墙场景")
        _log("=" * 60)

        config_dir = os.path.join(os.path.dirname(__file__), "config")
        robot_file = os.path.join(config_dir, "ur7e_robot_gazebo.yml")
        task_file = TASK_FILE
        world_file = os.path.join(config_dir, "collision_world_gazebo_tall.yml")

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
        if spawn_gazebo_obstacles(robot, world_params, model_prefix="sim_tall", include_ground=False):
            _log(
                "Gazebo 真实障碍物已生成: spheres=%d cubes=%d"
                % (n_world_spheres, n_world_cubes)
            )
        else:
            _log("警告: Gazebo 真实障碍物未完整生成，请检查 /spawn_entity 与 /delete_entity 服务")

        _log("\n初始化 STORM MPC 控制器...")
        device = "cuda" if args.cuda else "cpu"
        _log(f"计算设备: {device}")

        tensor_args = {
            "device": torch.device(device, 0) if device == "cuda" else torch.device("cpu"),
            "dtype": torch.float32,
        }

        mpc = GazeboReacherTask(task_file, robot_file, world_file, tensor_args)
        mpc.set_position_only_goal_mode()
        control_dt = mpc.exp_params.get("control_dt", 0.02)
        _log(f"MPC 控制周期: {control_dt} s ({1.0 / control_dt:.1f} Hz)")
        _log("目标模式: position-only (/target_pose 只约束 xyz, 不约束末端姿态)")

        default_goal_seed_state = np.array([0.5, -1.2, 1.2, -1.57, -1.57, 0.0, 0, 0, 0, 0, 0, 0])
        mpc.update_params(goal_state=default_goal_seed_state)

        goal_ee_world = DEFAULT_GOAL_WORLD.copy()
        goal_ee_pos_robot = inv_transform_point(robot_pos, robot_quat_xyzw, goal_ee_world)
        mpc.update_params(goal_ee_pos=goal_ee_pos_robot)

        _log(f"\n默认目标末端位置 (机器人坐标系): {goal_ee_pos_robot}")
        _log(f"默认目标末端位置 (世界坐标系): {goal_ee_world}")

        rollout_fn = mpc.controller.rollout_fn
        collision_sphere_visualizer = CollisionSphereVisualizer(
            mpc.exp_params["model"]["robot_collision_params"]
        )
        n_collision_spheres = sum(
            len(collision_sphere_visualizer.spheres_by_link.get(link_name, []))
            for link_name in collision_sphere_visualizer.link_names
        )
        current_goal_ee = goal_ee_pos_robot.copy()
        current_goal_world = goal_ee_world.copy()

        _log("\n" + "=" * 60)
        _log("开始 MPC 控制循环... (Ctrl+C 退出)")
        _log("=" * 60)
        _log("\n提示:")
        _log("  - 当前使用高墙场景 primitive world 避障")
        _log("  - 发布 PoseStamped 到 /target_pose 可动态更新目标")
        _log("  - /target_pose 的 orientation 不参与目标更新")
        _log("  - 在 RViz 中查看 /visualization_marker_array")
        _log("  - 在 RViz 中查看 /mppi_top_traj_markers (MPPI 前若干条预测轨迹)")
        _log("  - Gazebo 中已真实生成高墙/球体障碍物，可直接观察物理碰撞")
        _log("  - 红球=目标, 绿球=末端, 蓝色障碍物=高墙场景")
        _log(f"  - 黄球=机械臂碰撞球模型 ({n_collision_spheres} 个)")
        _log("  - 红线=MPPI 前若干条末端预测轨迹")
        _log(f"  - 控制器使用同步求解，执行模式={_get_execution_mode(mpc)}")
        _log("")

        _log("预热 MPC 控制器...")
        current_state = robot.get_state()
        for warm_idx in range(5):
            if shutdown_event.is_set():
                return 130
            if current_state is None:
                time.sleep(0.01)
                current_state = robot.get_state()
                continue
            try:
                _get_sync_command(mpc, warm_idx * control_dt, current_state, control_dt)
            except Exception as exc:
                _log(f"预热异常 (可忽略): {exc}")
            time.sleep(0.01)

        _log("MPC 预热完成，开始控制!\n")

        loop_count = 0
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

            new_target = robot.get_target_position()
            if new_target is not None:
                target_robot = inv_transform_point(robot_pos, robot_quat_xyzw, new_target)
                if np.linalg.norm(target_robot - current_goal_ee) > 0.005:
                    current_goal_ee = target_robot.copy()
                    current_goal_world = new_target.copy()
                    mpc.update_params(goal_ee_pos=current_goal_ee)
                    _log(
                        "[目标更新] 世界: %s, 机器人: %s"
                        % (np.round(current_goal_world, 3), np.round(current_goal_ee, 3))
                    )
                    try:
                        _reset_control_process_timing(mpc.control_process, t_step, control_dt)
                        cmd = _get_sync_command(mpc, t_step, state, control_dt)
                        _log("[目标更新] 已同步重规划并重置 MPC 时间基准")
                    except Exception as sync_exc:
                        _log(f"[MPC异常] 目标更新后的同步重规划失败: {sync_exc}")
                        time.sleep(control_dt)
                        continue

            if cmd is None:
                try:
                    cmd = _get_sync_command(mpc, t_step, state, control_dt)
                except (IndexError, RuntimeError, ValueError) as exc:
                    _log(
                        "[MPC恢复] 同步取命令失败 (%s)，重置控制进程时间基准后重规划"
                        % exc
                    )
                    try:
                        cmd = _recover_command(mpc, t_step, state, control_dt)
                    except Exception as recover_exc:
                        _log(f"[MPC异常] 同步重规划失败: {recover_exc}")
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

            curr = np.hstack([q, dq, ddq])
            ee_pose = rollout_fn.get_ee_pose(torch.as_tensor(curr, **tensor_args).unsqueeze(0))
            ee_pos_robot = np.ravel(ee_pose["ee_pos_seq"].cpu().numpy())
            ee_pos_world = transform_point(robot_pos, robot_quat_xyzw, ee_pos_robot)
            robot.publish_ee_pose(ee_pos_world)

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
                error = np.linalg.norm(ee_pos_world - current_goal_world)
                _log(
                    f"[{loop_count:5d}] t={t_step:.2f}s | "
                    f"q=[{q[0]:+.2f}, {q[1]:+.2f}, {q[2]:+.2f}, {q[3]:+.2f}, {q[4]:+.2f}, {q[5]:+.2f}] | "
                    f"ee_error={error:.4f} | opt_dt={mpc.opt_dt:.3f}s"
                )

            if args.max_steps > 0 and loop_count >= args.max_steps:
                _log(f"达到 max_steps={args.max_steps}，结束本轮运行")
                break

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
                _log(f"关闭 MPC 资源时出现异常: {exc}")

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
    parser = argparse.ArgumentParser(description="UR7e STORM MPC Reach Static Gazebo Tall Scene")
    parser.add_argument("--cuda", action="store_true", default=True, help="使用 CUDA 加速 (默认: True)")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="禁用 CUDA")
    parser.add_argument("--rate", type=float, default=50.0, help="控制频率 Hz (默认: 50)")
    parser.add_argument("--max-steps", type=int, default=0, help="最大控制步数，0 表示不限")
    args = parser.parse_args()
    sys.exit(mpc_control_main(args))
