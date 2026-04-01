#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

torch.multiprocessing.set_start_method("spawn", force=True)

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from experiment_logging import CsvExperimentLogger, normalize_step_record, summarize_episode
from generate_round4_pairs import generate_pairs
from run_controller_batch import (
    _apply_seed,
    _build_task,
    _build_tensor_args,
    _default_paths,
    _task_command_and_raw_stats,
)


WORLD_OBSTACLE = os.path.join(THIS_DIR, "config", "collision_world_gazebo_obstacle.yml")

SCENE_WORLD = {
    "obstacle_medium": {"scene": "default", "world_file": WORLD_OBSTACLE},
    "obstacle_hard": {"scene": "default", "world_file": WORLD_OBSTACLE},
    "narrow_medium": {"scene": "tall", "world_file": None},
    "narrow_hard": {"scene": "tall", "world_file": None},
}


def _load_pairs(path):
    with open(path) as f:
        return json.load(f)


def _goal_state(goal_joint_positions):
    goal_q = np.asarray(goal_joint_positions, dtype=np.float64)
    return np.concatenate([goal_q, np.zeros_like(goal_q)]).tolist()


def _initial_state(initial_joint_positions, n_dofs):
    q = np.asarray(initial_joint_positions, dtype=np.float64)
    return {
        "position": q[:n_dofs].copy(),
        "velocity": np.zeros(n_dofs, dtype=np.float64),
        "acceleration": np.zeros(n_dofs, dtype=np.float64),
    }


def _reset_filters(task):
    for filter_name in ("state_filter", "command_filter"):
        filter_obj = getattr(task, filter_name, None)
        if filter_obj is not None:
            filter_obj.cmd_joint_state = None
            filter_obj.prev_cmd_qdd = None
    if hasattr(task, "prev_qdd_des"):
        task.prev_qdd_des = None


def _run_pair_episode(controller_name, task, pair, steps, success_threshold, seed, logger):
    _reset_filters(task)
    current_state = _initial_state(pair["initial_joint_positions"], task.n_dofs)
    control_dt = task.exp_params["control_dt"]
    t_step = 0.0
    episode_id = f"{pair['pair_id']}_{controller_name}"
    step_records = []

    for step_id in range(int(steps)):
        command, raw_info = _task_command_and_raw_stats(task, t_step, current_state, control_dt)
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
        current_state = {
            "position": np.asarray(command["position"], dtype=np.float64).copy(),
            "velocity": np.asarray(command["velocity"], dtype=np.float64).copy(),
            "acceleration": np.asarray(command["acceleration"], dtype=np.float64).copy(),
        }
        t_step += control_dt
        if bool(step_record["success"]):
            break

    logger.log_episode(summarize_episode(step_records))
    return step_records


def run_round4(output_root, pairs_path, success_threshold, controller_seed, use_cuda=False, controllers=None):
    pairs = _load_pairs(pairs_path)
    tensor_args = _build_tensor_args(use_cuda)
    controllers = ["baseline", "sage"] if not controllers else list(controllers)

    run_meta = {
        "runner": "run_baseline_vs_sage_round4.py",
        "created_at": datetime.now().isoformat(),
        "pairs_path": os.path.abspath(pairs_path),
        "success_threshold": float(success_threshold),
        "controller_seed": int(controller_seed),
        "use_cuda": bool(use_cuda),
        "controllers": controllers,
    }
    with open(os.path.join(output_root, "round4_run_metadata.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    for scene_name in ("obstacle_medium", "obstacle_hard", "narrow_medium", "narrow_hard"):
        scene_root = os.path.join(output_root, f"scene={scene_name}")
        os.makedirs(scene_root, exist_ok=True)
        scene_pairs = pairs["scenes"][scene_name]
        scene_meta = pairs["scene_meta"][scene_name]
        for controller_name in controllers:
            task_default, robot_default, world_default = _default_paths(
                controller_name,
                SCENE_WORLD[scene_name]["scene"],
            )
            world_file = SCENE_WORLD[scene_name]["world_file"] or world_default
            logger = CsvExperimentLogger(
                os.path.join(scene_root, f"controller={controller_name}")
            )
            for pair in scene_pairs:
                task = _build_task(
                    controller_name,
                    task_default,
                    robot_default,
                    world_file,
                    tensor_args,
                )
                try:
                    _apply_seed(task, controller_seed)
                    task.update_params(goal_state=_goal_state(pair["goal_joint_positions"]))
                    _run_pair_episode(
                        controller_name=controller_name,
                        task=task,
                        pair=pair,
                        steps=pair.get("max_steps", scene_meta["max_steps"]),
                        success_threshold=success_threshold,
                        seed=controller_seed,
                        logger=logger,
                    )
                finally:
                    task.close()
            print(
                "completed scene=%s controller=%s output=%s"
                % (scene_name, controller_name, os.path.join(scene_root, f"controller={controller_name}")),
                flush=True,
            )
    return output_root


def main():
    parser = argparse.ArgumentParser(description="Run harder round4 baseline vs SAGE benchmark")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pairs-path", default=None)
    parser.add_argument("--pairs-per-scene", type=int, default=50)
    parser.add_argument("--pair-seed", type=int, default=20260404)
    parser.add_argument("--controller-seed", type=int, default=0)
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--controllers", nargs="+", default=("baseline", "sage"))
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    pairs_path = args.pairs_path
    if pairs_path is None:
        pairs_path = os.path.join(args.output_root, "round4_pairs.json")
        generate_pairs(
            output_path=pairs_path,
            pairs_per_scene=args.pairs_per_scene,
            seed=args.pair_seed,
            use_cuda=args.cuda,
        )

    run_round4(
        output_root=args.output_root,
        pairs_path=pairs_path,
        success_threshold=args.success_threshold,
        controller_seed=args.controller_seed,
        use_cuda=args.cuda,
        controllers=args.controllers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
