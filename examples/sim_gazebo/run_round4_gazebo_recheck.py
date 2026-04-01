#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

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

from experiment_logging import CsvExperimentLogger, normalize_step_record, summarize_episode
from gazebo_obstacle_utils import _wait_for_future, _wait_for_service, iter_primitive_obstacles, spawn_gazebo_obstacles
from run_controller_batch import (
    _apply_seed,
    _build_task,
    _build_tensor_args,
    _default_paths,
    _init_ros,
    _make_robot_interface,
    _task_command_and_raw_stats,
)


ROBOT_FILE = os.path.join(THIS_DIR, "config", "ur7e_robot_gazebo.yml")
INITIAL_POSITIONS_FILE = os.path.join(THIS_DIR, "config", "initial_positions.yaml")
WORLD_OBSTACLE = os.path.join(THIS_DIR, "config", "collision_world_gazebo_obstacle.yml")
WORLD_NARROW = os.path.join(THIS_DIR, "config", "collision_world_gazebo_tall.yml")

SCENE_SPECS = {
    "obstacle_hard": {"scene": "default", "world_file": WORLD_OBSTACLE},
    "narrow_hard": {"scene": "tall", "world_file": WORLD_NARROW},
}

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
SIM_INIT_Q = np.asarray([0.0, -1.57, 1.10, -1.57, -1.57, 0.0], dtype=np.float64)


def _load_pairs(path):
    with open(path) as f:
        return json.load(f)


def _goal_state(goal_joint_positions):
    goal_q = np.asarray(goal_joint_positions, dtype=np.float64)
    return np.concatenate([goal_q, np.zeros_like(goal_q)]).tolist()


def _reset_filters(task):
    for filter_name in ("state_filter", "command_filter"):
        filter_obj = getattr(task, filter_name, None)
        if filter_obj is not None:
            filter_obj.cmd_joint_state = None
            filter_obj.prev_cmd_qdd = None
    if hasattr(task, "prev_qdd_des"):
        task.prev_qdd_des = None


def _load_world_params(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _top_pairs(pairs_path, pairs_per_scene, scenes):
    data = _load_pairs(pairs_path)
    selected = {}
    for scene_name in scenes:
        ranked = sorted(
            data["scenes"][scene_name],
            key=lambda item: (
                -float(item.get("difficulty_score", 0.0)) / (1.0 + float(np.linalg.norm(np.asarray(item["initial_joint_positions"], dtype=np.float64) - SIM_INIT_Q))),
                -float(item.get("difficulty_score", 0.0)),
            ),
        )
        selected[scene_name] = ranked[:pairs_per_scene]
    return selected


def _launch_gazebo_process(gui):
    source_bits = [
        "source /opt/ros/humble/setup.bash",
        "if [ -f ~/ur_arm/ros_ur_driver/install/setup.bash ]; then source ~/ur_arm/ros_ur_driver/install/setup.bash; fi",
        "if [ -f ~/ur_arm/gazebo_ur_sim/install/setup.bash ]; then source ~/ur_arm/gazebo_ur_sim/install/setup.bash; fi",
    ]
    launch_cmd = (
        f"ros2 launch ur_simulation_gazebo ur_sim_control.launch.py "
        f"ur_type:=ur7e "
        f"initial_joint_controller:=forward_position_controller "
        f"initial_positions_file:={INITIAL_POSITIONS_FILE} "
        f"launch_rviz:=false "
        f"gazebo_gui:={'true' if gui else 'false'}"
    )
    shell_cmd = " && ".join(source_bits + [launch_cmd])
    return subprocess.Popen(
        ["bash", "-lc", shell_cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _delete_prefixed_obstacles(node, all_world_params, model_prefix):
    from gazebo_msgs.srv import DeleteEntity

    delete_client = node.create_client(DeleteEntity, "/delete_entity")
    if not _wait_for_service(delete_client, 8.0):
        return

    delete_names = set()
    for world_params in all_world_params:
        for obstacle in iter_primitive_obstacles(world_params, include_ground=False):
            delete_names.add(f"{model_prefix}_{obstacle['name']}")
    for model_name in sorted(delete_names):
        req = DeleteEntity.Request()
        req.name = model_name
        try:
            _wait_for_future(delete_client.call_async(req), timeout_sec=2.0)
        except Exception:
            pass


def _set_scene_obstacles(robot_node, scene_name, world_params_map):
    model_prefix = "round4_recheck"
    _delete_prefixed_obstacles(
        robot_node,
        [world_params_map["obstacle_hard"], world_params_map["narrow_hard"]],
        model_prefix=model_prefix,
    )
    ok = spawn_gazebo_obstacles(
        robot_node,
        world_params_map[scene_name],
        model_prefix=model_prefix,
        include_ground=False,
        service_timeout_sec=20.0,
    )
    if not ok:
        robot_node.get_logger().warning(
            f"Gazebo obstacle synchronization failed for scene={scene_name}; "
            f"continuing recheck with planner-side world only."
        )
        return False
    time.sleep(1.0)
    return True


def _play_waypoint_segment(robot, start_positions, target_positions, interp_steps=50, sleep_sec=0.08):
    for alpha in np.linspace(0.0, 1.0, interp_steps):
        waypoint = (1.0 - alpha) * start_positions + alpha * target_positions
        robot.send_position_command(waypoint)
        time.sleep(sleep_sec)


def _drive_to_joint_state(robot, target_positions, timeout=20.0, tol=0.06):
    target_positions = np.asarray(target_positions, dtype=np.float64).reshape(-1)
    current_state = robot.get_state()
    current_positions = (
        np.asarray(current_state["position"], dtype=np.float64).reshape(-1)
        if current_state is not None
        else target_positions.copy()
    )
    if np.linalg.norm(current_positions - SIM_INIT_Q) > 0.12:
        _play_waypoint_segment(robot, current_positions, SIM_INIT_Q, interp_steps=45, sleep_sec=0.08)
        current_positions = SIM_INIT_Q.copy()
    _play_waypoint_segment(robot, current_positions, target_positions, interp_steps=55, sleep_sec=0.08)

    start = time.time()
    stable_since = None
    while time.time() - start < timeout:
        state = robot.get_state()
        if state is not None:
            err = np.max(np.abs(np.asarray(state["position"], dtype=np.float64) - target_positions))
            if err <= tol:
                if stable_since is None:
                    stable_since = time.time()
                if time.time() - stable_since >= 0.75:
                    return float(err)
            else:
                stable_since = None
        robot.send_position_command(target_positions)
        time.sleep(0.05)
    state = robot.get_state()
    if state is None:
        raise RuntimeError("Failed to receive Gazebo joint state while moving to initial pose")
    err = np.max(np.abs(np.asarray(state["position"], dtype=np.float64) - target_positions))
    if err > 0.45:
        raise RuntimeError(f"Failed to reach initial joint state, max error={err:.4f}")
    return float(err)


def _run_single_episode(robot, task, pair, controller_name, scene_name, logger, seed, rate, success_threshold):
    _reset_filters(task)
    _drive_to_joint_state(robot, pair["initial_joint_positions"])
    time.sleep(0.3)

    task.update_params(goal_state=_goal_state(pair["goal_joint_positions"]))
    episode_id = f"{pair['pair_id']}_{controller_name}"
    control_dt = task.exp_params["control_dt"]
    t_step = 0.0
    step_records = []

    for step_id in range(int(pair.get("max_steps", 150))):
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
            success_threshold=success_threshold,
        )
        logger.log_step(step_record)
        step_records.append(step_record)
        t_step += control_dt
        if bool(step_record["success"]):
            break
        time.sleep(max(1.0 / rate, 0.02))

    logger.log_episode(summarize_episode(step_records))
    return step_records


def run_recheck(output_root, pairs_path, pairs_per_scene, controller_seed, rate, success_threshold, use_cuda, launch_gazebo, gazebo_gui, scenes):
    selected_pairs = _top_pairs(pairs_path, pairs_per_scene=pairs_per_scene, scenes=scenes)
    tensor_args = _build_tensor_args(use_cuda)
    os.makedirs(output_root, exist_ok=True)

    launch_proc = _launch_gazebo_process(gazebo_gui) if launch_gazebo else None
    rclpy = None
    robot = None
    executor = None
    spin_thread = None
    spin_running = True
    try:
        rclpy, MultiThreadedExecutor, Node, JointState, Float64MultiArray = _init_ros()
        GazeboRobotInterface = _make_robot_interface(Node, JointState, Float64MultiArray)

        rclpy.init(args=None)
        robot = GazeboRobotInterface(JOINT_NAMES, rate)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(robot)

        def _spin():
            while spin_running and rclpy.ok():
                executor.spin_once(timeout_sec=0.1)

        spin_thread = threading.Thread(target=_spin, daemon=True)
        spin_thread.start()

        if not robot.wait_for_state(rclpy, timeout=40.0 if launch_gazebo else 15.0):
            raise RuntimeError("Unable to receive /joint_states from Gazebo during round4 recheck")
        if launch_gazebo:
            time.sleep(8.0)

        world_params_map = {
            scene_name: _load_world_params(scene_spec["world_file"])
            for scene_name, scene_spec in SCENE_SPECS.items()
        }

        meta = {
            "pairs_path": os.path.abspath(pairs_path),
            "pairs_per_scene": int(pairs_per_scene),
            "controller_seed": int(controller_seed),
            "rate": float(rate),
            "success_threshold": float(success_threshold),
            "physical_obstacles_spawned": {},
            "selected_pairs": {
                scene_name: [pair["pair_id"] for pair in pairs]
            for scene_name, pairs in selected_pairs.items()
            },
        }

        for scene_name in scenes:
            meta["physical_obstacles_spawned"][scene_name] = bool(
                _set_scene_obstacles(robot, scene_name, world_params_map)
            )
            scene_root = os.path.join(output_root, f"scene={scene_name}")
            os.makedirs(scene_root, exist_ok=True)
            for controller_name in ("baseline", "sage"):
                task_default, robot_default, world_default = _default_paths(
                    controller_name,
                    SCENE_SPECS[scene_name]["scene"],
                )
                logger = CsvExperimentLogger(os.path.join(scene_root, f"controller={controller_name}"))
                world_file = SCENE_SPECS[scene_name]["world_file"] or world_default
                for pair in selected_pairs[scene_name]:
                    task = _build_task(
                        controller_name,
                        task_default,
                        ROBOT_FILE,
                        world_file,
                        tensor_args,
                    )
                    try:
                        _apply_seed(task, controller_seed)
                        _run_single_episode(
                            robot=robot,
                            task=task,
                            pair=pair,
                            controller_name=controller_name,
                            scene_name=scene_name,
                            logger=logger,
                            seed=controller_seed,
                            rate=rate,
                            success_threshold=success_threshold,
                        )
                    finally:
                        task.close()
                print(f"completed scene={scene_name} controller={controller_name}", flush=True)
        with open(os.path.join(output_root, "round4_gazebo_recheck_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return output_root
    finally:
        spin_running = False
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)
        if robot is not None:
            robot.destroy_node()
        if executor is not None:
            executor.shutdown(timeout_sec=1.0)
        try:
            if rclpy is not None and rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        if launch_proc is not None:
            try:
                os.killpg(os.getpgid(launch_proc.pid), signal.SIGINT)
            except Exception:
                pass
            try:
                launch_proc.wait(timeout=20.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(launch_proc.pid), signal.SIGKILL)
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description="Run small-scale Gazebo recheck on round4 hard scenes")
    parser.add_argument("--pairs-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pairs-per-scene", type=int, default=5)
    parser.add_argument("--controller-seed", type=int, default=0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--launch-gazebo", action="store_true", default=False)
    parser.add_argument("--gazebo-gui", action="store_true", default=False)
    parser.add_argument("--scenes", nargs="+", default=("obstacle_hard", "narrow_hard"))
    args = parser.parse_args()

    run_recheck(
        output_root=args.output_root,
        pairs_path=args.pairs_path,
        pairs_per_scene=args.pairs_per_scene,
        controller_seed=args.controller_seed,
        rate=args.rate,
        success_threshold=args.success_threshold,
        use_cuda=args.cuda,
        launch_gazebo=args.launch_gazebo,
        gazebo_gui=args.gazebo_gui,
        scenes=list(args.scenes),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
