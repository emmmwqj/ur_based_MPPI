#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
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
from generate_round3_pairs import generate_pairs
from run_controller_batch import (
    _apply_seed,
    _build_task,
    _build_tensor_args,
    _default_paths,
    _task_command_and_raw_stats,
)
from summarize_round3_benchmark import summarize


WORLD_EASY = os.path.join(THIS_DIR, "config", "collision_world_gazebo_easy.yml")
WORLD_OBSTACLE = os.path.join(THIS_DIR, "config", "collision_world_gazebo_obstacle.yml")

SCENE_SPECS = {
    "easy": {"scene": "default", "world_file": WORLD_EASY},
    "obstacle": {"scene": "default", "world_file": WORLD_OBSTACLE},
    "narrow": {"scene": "tall", "world_file": None},
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


def _run_pair_episode(
    controller_name,
    task,
    pair,
    steps,
    success_threshold,
    seed,
    logger,
):
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


def run_round3(output_root, pairs_path, steps, success_threshold, controller_seed, use_cuda=False):
    pairs = _load_pairs(pairs_path)
    tensor_args = _build_tensor_args(use_cuda)
    run_meta = {
        "runner": "run_baseline_vs_sage_round3.py",
        "created_at": datetime.now().isoformat(),
        "pairs_path": os.path.abspath(pairs_path),
        "steps": int(steps),
        "success_threshold": float(success_threshold),
        "controller_seed": int(controller_seed),
        "use_cuda": bool(use_cuda),
    }
    with open(os.path.join(output_root, "round3_run_metadata.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    for scene_name in ("easy", "obstacle", "narrow"):
        scene_root = os.path.join(output_root, f"scene={scene_name}")
        os.makedirs(scene_root, exist_ok=True)
        scene_pairs = pairs["scenes"][scene_name]
        for controller_name in ("baseline", "sage"):
            task_default, robot_default, world_default = _default_paths(
                controller_name,
                SCENE_SPECS[scene_name]["scene"],
            )
            world_file = SCENE_SPECS[scene_name]["world_file"] or world_default
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
                        steps=steps,
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

    summary_dir = os.path.join(output_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    summarize(output_root, summary_dir)
    return summary_dir


def main():
    parser = argparse.ArgumentParser(description="Round3 multi-initial-goal baseline vs SAGE benchmark")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pairs-path", default=None)
    parser.add_argument("--pairs-per-scene", type=int, default=50)
    parser.add_argument("--pair-seed", type=int, default=20260402)
    parser.add_argument("--state-pool-path", default=None)
    parser.add_argument("--controller-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    pairs_path = args.pairs_path
    if pairs_path is None:
        pairs_path = os.path.join(args.output_root, "round3_pairs.json")
        generate_pairs(
            output_path=pairs_path,
            pairs_per_scene=args.pairs_per_scene,
            seed=args.pair_seed,
            use_cuda=args.cuda,
            state_pool_path=args.state_pool_path,
        )

    run_round3(
        output_root=args.output_root,
        pairs_path=pairs_path,
        steps=args.steps,
        success_threshold=args.success_threshold,
        controller_seed=args.controller_seed,
        use_cuda=args.cuda,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
