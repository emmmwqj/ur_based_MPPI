#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import torch

torch.multiprocessing.set_start_method("spawn", force=True)

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from experiment_logging import CsvExperimentLogger
from generate_round2_targets import generate_targets
from run_controller_batch import (
    _apply_seed,
    _build_task,
    _build_tensor_args,
    _default_paths,
    _run_headless_episode,
)
from summarize_round2_benchmark import summarize


WORLD_EASY = os.path.join(THIS_DIR, "config", "collision_world_gazebo_easy.yml")
WORLD_OBSTACLE = os.path.join(THIS_DIR, "config", "collision_world_gazebo_obstacle.yml")

SCENE_SPECS = {
    "easy": {
        "scene": "default",
        "world_file": WORLD_EASY,
    },
    "obstacle": {
        "scene": "default",
        "world_file": WORLD_OBSTACLE,
    },
    "narrow": {
        "scene": "tall",
        "world_file": None,
    },
}


def _load_targets(path):
    with open(path) as f:
        return json.load(f)


def _episode_args(goal_joint_positions, steps, success_threshold):
    return SimpleNamespace(
        goal=list(goal_joint_positions),
        steps=int(steps),
        success_threshold=float(success_threshold),
        stop_on_success=True,
    )


def _run_round2(output_root, targets_path, steps, success_threshold, controller_seed, use_cuda=False):
    targets = _load_targets(targets_path)
    tensor_args = _build_tensor_args(use_cuda)
    run_meta = {
        "runner": "run_baseline_vs_sage_round2.py",
        "created_at": datetime.now().isoformat(),
        "targets_path": os.path.abspath(targets_path),
        "steps": int(steps),
        "success_threshold": float(success_threshold),
        "controller_seed": int(controller_seed),
        "use_cuda": bool(use_cuda),
    }
    with open(os.path.join(output_root, "round2_run_metadata.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    for scene_name in ("easy", "obstacle", "narrow"):
        scene_root = os.path.join(output_root, f"scene={scene_name}")
        os.makedirs(scene_root, exist_ok=True)
        for controller_name in ("baseline", "sage"):
            task_default, robot_default, world_default = _default_paths(
                controller_name,
                SCENE_SPECS[scene_name]["scene"],
            )
            world_file = SCENE_SPECS[scene_name]["world_file"] or world_default
            task_file = task_default
            robot_file = robot_default
            logger = CsvExperimentLogger(
                os.path.join(scene_root, f"controller={controller_name}")
            )

            for target in targets["scenes"][scene_name]:
                episode_id = f"{scene_name}_{controller_name}_{target['target_id']}"
                args = _episode_args(
                    goal_joint_positions=target["goal_joint_positions"],
                    steps=steps,
                    success_threshold=success_threshold,
                )
                task = _build_task(controller_name, task_file, robot_file, world_file, tensor_args)
                try:
                    _apply_seed(task, controller_seed)
                    task.update_params(
                        goal_state=target["goal_joint_positions"] + [0.0] * len(target["goal_joint_positions"])
                    )
                    _run_headless_episode(
                        controller_name=controller_name,
                        task=task,
                        args=args,
                        seed=controller_seed,
                        episode_id=episode_id,
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
    parser = argparse.ArgumentParser(description="Round2 multi-target baseline vs SAGE benchmark")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--targets-path", default=None)
    parser.add_argument("--targets-per-scene", type=int, default=20)
    parser.add_argument("--target-seed", type=int, default=20260401)
    parser.add_argument("--controller-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    targets_path = args.targets_path
    if targets_path is None:
        targets_path = os.path.join(args.output_root, "round2_targets.json")
        generate_targets(
            output_path=targets_path,
            targets_per_scene=args.targets_per_scene,
            seed=args.target_seed,
            use_cuda=args.cuda,
        )

    _run_round2(
        output_root=args.output_root,
        targets_path=targets_path,
        steps=args.steps,
        success_threshold=args.success_threshold,
        controller_seed=args.controller_seed,
        use_cuda=args.cuda,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
