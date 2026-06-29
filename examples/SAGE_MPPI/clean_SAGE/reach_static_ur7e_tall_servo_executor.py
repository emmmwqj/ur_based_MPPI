#!/usr/bin/env python3
"""Clean SAGE tall Gazebo entry with a switchable joint execution layer.

The planner remains unchanged. The only behavioral difference from the base
clean tall entry is the output path after MPPI produces q_next:

    q_next -> JointServoExecutor -> GazeboBackend -> ros2_control topic
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

STORM_ROOT = os.path.expanduser("~/storm")
if STORM_ROOT not in sys.path:
    sys.path.insert(0, STORM_ROOT)

torch.multiprocessing.set_start_method("spawn", force=True)

from examples.SAGE_MPPI.clean_SAGE.joint_servo_executor import (  # noqa: E402
    GazeboBackend,
    JointServoExecutor,
    ServoExecutionLogger,
    parse_joint_limit,
)
from examples.SAGE_MPPI.clean_SAGE.reach_static_ur7e_tall import (  # noqa: E402
    TASK_FILE,
    WORLD_FILE,
    ROBOT_FILE,
    _apply_refinement_overrides,
    _build_tensor_args,
    _configure_default_goal,
    _get_clean_top_ee_trajs_world,
    _get_execution_mode,
    _get_robot_pose_world,
    _load_robot_and_world_params,
    _make_clean_task,
    _recover_command_strict,
    _reset_control_process_timing_strict,
)
from examples.SAGE_MPPI.deployment_refinement import DeploymentRefinementStack  # noqa: E402
from storm_kit.mpc.control.sage_mppi import SAGE_MPPI  # noqa: E402

np.set_printoptions(precision=3, suppress=True)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONTROLLER_TOPIC = "/forward_position_controller/commands"
DEFAULT_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _log(message: str) -> None:
    print(message, flush=True)


def _parse_joint_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("joint name list cannot be empty")
    return names


def _default_log_dir() -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return str(SCRIPT_DIR / "servo_executor_logs" / timestamp)


def _run_gazebo_main(args) -> int:
    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from std_msgs.msg import ColorRGBA
        from visualization_msgs.msg import Marker, MarkerArray
    except ImportError:
        print("=" * 60)
        print("error: ROS2 Python packages were not found")
        print("source ROS2 first, for example:")
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
        CollisionSphereVisualizer,
        TallGazeboRobotInterface,
        _compute_link_poses_robot_frame,
        _shutdown_control_process,
    )

    class ServoTallGazeboRobotInterface(TallGazeboRobotInterface):
        """Tall-scene robot interface with the same visualization behavior as clean SAGE."""

        def __init__(self, joint_names: list[str], control_rate: float = 50.0):
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

            for marker_id, (ns, pos, color, scale) in enumerate(
                (
                    ("goal", goal_pos, ColorRGBA(r=0.9, g=0.1, b=0.1, a=0.8), 0.06),
                    ("ee", ee_pos, ColorRGBA(r=0.1, g=0.9, b=0.1, a=0.8), 0.05),
                )
            ):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = stamp
                marker.ns = ns
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(pos[0])
                marker.pose.position.y = float(pos[1])
                marker.pose.position.z = float(pos[2])
                marker.pose.orientation.w = 1.0
                marker.scale.x = scale
                marker.scale.y = scale
                marker.scale.z = scale
                marker.color = color
                marker_array.markers.append(marker)

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
    joint_names = args.joint_names
    n_dof = len(joint_names)
    tensor_args = _build_tensor_args(use_cuda=args.cuda)

    max_joint_speed = parse_joint_limit(args.max_joint_speed, n_dof, "max_joint_speed")
    parsed_joint_acceleration = parse_joint_limit(
        args.max_joint_acceleration,
        n_dof,
        "max_joint_acceleration",
    )
    max_joint_acceleration = (
        parsed_joint_acceleration
        if np.any(parsed_joint_acceleration > 0.0)
        else None
    )

    robot = None
    executor = None
    spin_thread = None
    spin_running = [True]
    mpc = None
    exit_code = 0
    shutdown_event = threading.Event()
    backend = None
    servo_executor = None
    servo_logger = None

    try:
        _log("=" * 60)
        _log("UR7e SAGE CLEAN MPC Reach Static - Gazebo Tall Scene + Servo Executor")
        _log("=" * 60)
        _log(f"Task:  {TASK_FILE}")
        _log(f"Robot: {ROBOT_FILE}")
        _log(f"World: {WORLD_FILE}")
        _log(f"execution_layer={args.execution_layer}")
        _log(f"executor_frequency={args.executor_frequency:.1f} Hz")
        _log(f"max_joint_speed={np.round(max_joint_speed, 4).tolist()}")
        _log(f"max_joint_acceleration={None if max_joint_acceleration is None else np.round(max_joint_acceleration, 4).tolist()}")
        _log(f"controller_topic={args.controller_topic}")
        _log(f"executor_log_dir={args.executor_log_dir}")

        rclpy.init(args=None)
        robot = ServoTallGazeboRobotInterface(joint_names, control_rate=args.rate)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(robot)

        def spin_with_check():
            while spin_running[0] and rclpy.ok() and not shutdown_event.is_set():
                try:
                    executor.spin_once(timeout_sec=0.1)
                except RuntimeError as exc:
                    if "Destroyable" in str(exc):
                        break
                    raise

        spin_thread = threading.Thread(target=spin_with_check, daemon=True)
        spin_thread.start()

        _log("\nwaiting for Gazebo joint states...")
        start = time.time()
        while not robot.is_connected():
            if shutdown_event.is_set():
                return 130
            if time.time() - start > 10.0:
                _log("error: did not receive /joint_states; make sure Gazebo is running")
                return 1
            time.sleep(0.1)

        _log("connected to Gazebo robot")
        backend = GazeboBackend(robot, args.controller_topic, joint_names)
        servo_logger = ServoExecutionLogger(args.executor_log_dir, joint_names)

        if args.execution_layer == "servo":
            servo_executor = JointServoExecutor(
                backend=backend,
                joint_names=joint_names,
                executor_frequency=args.executor_frequency,
                max_joint_speed=max_joint_speed,
                max_joint_acceleration=max_joint_acceleration,
                actual_position_fn=robot.get_joint_positions,
                actual_velocity_fn=robot.get_joint_velocities,
                logger=servo_logger,
                log_every_n_ticks=args.executor_log_every,
            )
            servo_executor.start(robot.get_joint_positions())
            _log("JointServoExecutor started")
        else:
            _log("direct execution layer selected; q_next will be published without servo smoothing")

        n_world_spheres, n_world_cubes = count_primitive_obstacles(world_params, include_ground=False)
        if spawn_gazebo_obstacles(robot, world_params, model_prefix="sage_clean_servo", include_ground=False):
            _log("Gazebo obstacles spawned: spheres=%d cubes=%d" % (n_world_spheres, n_world_cubes))
        else:
            _log("warning: Gazebo obstacles were not fully spawned")

        mpc = _make_clean_task(tensor_args)
        if not isinstance(mpc.controller, SAGE_MPPI):
            raise RuntimeError("clean Gazebo example did not instantiate SAGE_MPPI")

        control_dt = float(mpc.exp_params.get("control_dt", 0.02))
        refinement_cfg = _apply_refinement_overrides(mpc.deployment_refinement_config, args)
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

        goal_ee_pos_robot, goal_ee_quat, goal_world = _configure_default_goal(
            mpc,
            robot_pos,
            robot_quat_xyzw,
            inv_transform_point,
        )
        _log(f"default goal robot frame: {goal_ee_pos_robot}")
        _log(f"default goal world frame: {goal_world}")

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
        _log("warming clean SAGE-MPPI controller...")
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
                _log(f"warmup warning: {exc}")
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
            actual_loop_dt_wall = None if last_wall_time is None else iter_start - last_wall_time
            current_sim_time = robot.get_latest_sim_time()
            actual_loop_dt_sim = None if last_sim_time is None or current_sim_time is None else current_sim_time - last_sim_time
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
                    _log("[target update] world=%s robot=%s" % (np.round(current_goal_world, 3), np.round(current_goal_ee, 3)))

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
                _log(f"[CleanMPC] command failed; recovering: {exc}")
                try:
                    cmd = _recover_command_strict(mpc, t_step, state, control_dt)
                    stats = mpc.get_latest_stats()
                except Exception as recover_exc:
                    _log(f"[CleanMPC] recovery failed: {recover_exc}")
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

            q_next = cmd["position"]
            if isinstance(q_next, torch.Tensor):
                q_next = q_next.detach().cpu().numpy()
            q_next = np.asarray(q_next, dtype=np.float64).flatten()[:n_dof]

            if args.execution_layer == "servo":
                servo_executor.set_target(q_next, source="mppi", q_actual=q, dq_actual=dq)
            else:
                backend.publish(q_next)
                servo_logger.log(
                    event="direct_publish",
                    sequence=loop_count,
                    source="mppi",
                    q_next=q_next,
                    q_cmd=q_next,
                    q_actual=q,
                    qd_cmd=None,
                    dq_actual=dq,
                    target_age_s=0.0,
                    max_abs_cmd_step=None,
                )

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
                current_q_cmd = servo_executor.get_current_command() if servo_executor is not None else q_next
                servo_error = float(np.max(np.abs(q_next - current_q_cmd))) if current_q_cmd is not None else float("nan")
                _log(
                    f"[{loop_count:5d}] t={t_step:.2f}s | "
                    f"ee_error={ee_error:.4f} | "
                    f"layer={args.execution_layer} | "
                    f"servo_target_err={servo_error:.4f} | "
                    f"near_goal={bool(stats.get('near_goal_active', False))} | "
                    f"lr_active={bool(local_stats.get('local_refinement_active', False))} | "
                    f"lr_mode={local_stats.get('local_refinement_mode', 'off')} | "
                    f"actual_loop_dt_wall={float('nan') if actual_loop_dt_wall is None else actual_loop_dt_wall:.3f}s | "
                    f"actual_loop_dt_sim={float('nan') if actual_loop_dt_sim is None else actual_loop_dt_sim:.3f}s | "
                    f"opt_dt={mpc.opt_dt:.3f}s"
                )

            if max_steps is not None and loop_count >= max_steps:
                _log(f"reached max_steps={max_steps}; exiting")
                break

            elapsed = time.time() - iter_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        return exit_code
    finally:
        shutdown_event.set()
        if servo_executor is not None:
            try:
                servo_executor.stop()
            except Exception:
                pass
        if servo_logger is not None:
            try:
                servo_logger.close()
                _log(f"servo execution log: {servo_logger.path}")
            except Exception:
                pass
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


def main() -> int:
    parser = argparse.ArgumentParser(description="UR7e clean SAGE tall entry with JointServoExecutor")
    parser.add_argument("--cuda", action="store_true", default=True, help="use CUDA when available")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="disable CUDA")
    parser.add_argument("--rate", type=float, default=50.0, help="MPPI/control loop rate in Hz")
    parser.add_argument("--max-steps", type=int, default=0, help="maximum MPPI loop steps; <=0 means unlimited")
    parser.add_argument("--viz-update-every", type=int, default=5, help="visualization update interval in MPPI steps")
    parser.add_argument("--enable-deployment-refinement", action="store_true", help="force enable deployment refinement")
    parser.add_argument("--disable-deployment-refinement", action="store_true", help="force disable deployment refinement")
    parser.add_argument("--enable-cartesian-refinement", action="store_true", help="force enable Cartesian refinement")
    parser.add_argument("--disable-cartesian-refinement", action="store_true", help="force disable Cartesian refinement")

    parser.add_argument("--execution-layer", choices=("servo", "direct"), default="servo", help="output execution layer")
    parser.add_argument("--executor-frequency", type=float, default=200.0, help="JointServoExecutor output frequency in Hz")
    parser.add_argument("--max-joint-speed", default="0.6", help="scalar or comma-separated rad/s joint speed limit")
    parser.add_argument("--max-joint-acceleration", default="0.0", help="0 disables acceleration limiting; otherwise scalar or comma-separated rad/s^2")
    parser.add_argument("--controller-topic", default=DEFAULT_CONTROLLER_TOPIC, help="Gazebo ros2_control command topic")
    parser.add_argument("--joint-names", type=_parse_joint_names, default=DEFAULT_JOINT_NAMES, help="comma-separated joint names")
    parser.add_argument("--executor-log-dir", default=_default_log_dir(), help="directory for servo_execution_log.csv")
    parser.add_argument("--executor-log-every", type=int, default=1, help="log every N executor ticks")
    args = parser.parse_args()

    return _run_gazebo_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
