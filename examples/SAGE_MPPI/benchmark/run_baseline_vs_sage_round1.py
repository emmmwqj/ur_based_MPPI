#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BENCHMARK_DIR = os.path.join(REPO_ROOT, "examples", "SAGE_MPPI", "benchmark")
BATCH_RUNNER = os.path.join(BENCHMARK_DIR, "run_controller_batch.py")
SUMMARIZER = os.path.join(BENCHMARK_DIR, "summarize_experiments.py")


SCENE_SPECS = {
    "easy": {
        "scene": "default",
        "world_file": os.path.join(
            REPO_ROOT, "examples", "sim_gazebo", "config", "collision_world_gazebo_easy.yml"
        ),
    },
    "obstacle": {
        "scene": "default",
        "world_file": os.path.join(
            REPO_ROOT, "examples", "sim_gazebo", "config", "collision_world_gazebo_obstacle.yml"
        ),
    },
    "narrow": {
        "scene": "tall",
        "world_file": None,
    },
}


def _run(cmd, env):
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description="Round-1 baseline vs SAGE comparison runner")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=sorted(SCENE_SPECS.keys()),
        default=["easy", "obstacle", "narrow"],
    )
    parser.add_argument("--success-threshold", type=float, default=0.05)
    parser.add_argument("--stop-on-success", action="store_true", default=True)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    env = os.environ.copy()
    python_bin = sys.executable
    os.makedirs(args.output_root, exist_ok=True)

    for scene_name in args.scenes:
        spec = SCENE_SPECS[scene_name]
        scene_root = os.path.join(args.output_root, f"scene={scene_name}")
        os.makedirs(scene_root, exist_ok=True)
        for controller_name in ("baseline", "sage"):
            cmd = [
                python_bin,
                BATCH_RUNNER,
                "--controller",
                controller_name,
                "--episodes",
                str(args.episodes),
                "--steps",
                str(args.steps),
                "--seed",
                *[str(seed) for seed in args.seeds],
                "--output_dir",
                scene_root,
                "--scene",
                spec["scene"],
                "--success-threshold",
                str(args.success_threshold),
            ]
            if args.stop_on_success:
                cmd.append("--stop-on-success")
            if args.cuda:
                cmd.append("--cuda")
            else:
                cmd.append("--no-cuda")
            if spec["world_file"] is not None:
                cmd.extend(["--world-file", spec["world_file"]])
            _run(cmd, env)

    summary_dir = os.path.join(args.output_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    _run(
        [
            python_bin,
            SUMMARIZER,
            "--input-root",
            args.output_root,
            "--output-dir",
            summary_dir,
        ],
        env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
