#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
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

from run_controller_batch import _build_task, _build_tensor_args


ROBOT_FILE = os.path.join(THIS_DIR, "config", "ur7e_robot_gazebo.yml")
BASELINE_TASK_FILE = os.path.join(REPO_ROOT, "content", "configs", "mpc", "ur7e_reacher.yml")
WORLD_EASY = os.path.join(THIS_DIR, "config", "collision_world_gazebo_easy.yml")
WORLD_OBSTACLE = os.path.join(THIS_DIR, "config", "collision_world_gazebo_obstacle.yml")
WORLD_NARROW = os.path.join(THIS_DIR, "config", "collision_world_gazebo_tall.yml")

INIT_Q = np.asarray([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float64)
JOINT_LOW = np.asarray([-1.0, -2.1, 0.7, -2.6, -2.3, -1.3], dtype=np.float64)
JOINT_HIGH = np.asarray([1.0, -0.8, 2.1, -0.6, -0.8, 1.3], dtype=np.float64)

SCENE_SPECS = {
    "easy": {
        "world_file": WORLD_EASY,
        "anchors": [
            [0.12, -1.46, 1.46, -1.57, -1.57, 0.0],
            [0.22, -1.38, 1.38, -1.57, -1.57, 0.0],
            [0.32, -1.30, 1.30, -1.57, -1.57, 0.0],
        ],
        "ee_bounds": {
            "x": (0.48, 0.64),
            "y": (0.16, 0.40),
            "z": (0.40, 0.58),
        },
        "ee_dist": (0.03, 0.14),
        "min_margin": 0.02,
    },
    "obstacle": {
        "world_file": WORLD_OBSTACLE,
        "anchors": [
            [0.22, -1.38, 1.38, -1.57, -1.57, 0.0],
            [0.35, -1.30, 1.30, -1.57, -1.57, 0.0],
            [0.48, -1.22, 1.22, -1.57, -1.57, 0.0],
        ],
        "ee_bounds": None,
        "ee_dist": (0.04, 0.16),
        "min_margin": 0.01,
    },
    "narrow": {
        "world_file": WORLD_NARROW,
        "anchors": [
            [0.00, -1.46, 1.46, -1.57, -1.57, 0.0],
            [0.12, -1.40, 1.40, -1.57, -1.57, 0.0],
            [0.22, -1.34, 1.34, -1.57, -1.57, 0.0],
        ],
        "ee_bounds": None,
        "ee_dist": (0.04, 0.16),
        "min_margin": 0.005,
    },
}


def _state_tensor(q, tensor_args, d_state):
    state = torch.zeros(1, d_state, **tensor_args)
    state[0, : q.shape[0]] = torch.as_tensor(q, **tensor_args)
    return state


def _ee_pos(task, q):
    tensor_args = task.controller.tensor_args
    q_tensor = torch.as_tensor(q, **tensor_args).view(1, -1)
    qd_tensor = torch.zeros_like(q_tensor)
    ee_pos, _ = task.controller.rollout_fn.dynamics_model.robot_model.compute_forward_kinematics(
        q_tensor,
        qd_tensor,
        link_name=task.exp_params["model"]["ee_link_name"],
    )
    return ee_pos.detach().cpu().numpy().reshape(-1)


def _single_state_safety_margin(task, q):
    rollout_fn = task.controller.rollout_fn
    tensor_args = task.controller.tensor_args
    d_state = rollout_fn.dynamics_model.d_state
    start_state = _state_tensor(q, tensor_args, d_state)
    horizon = int(task.controller.horizon)
    zero_act = torch.zeros(1, horizon, rollout_fn.dynamics_model.d_action, **tensor_args)
    state_dict = rollout_fn.dynamics_model.rollout_open_loop(start_state, zero_act)
    margins = []

    if (
        hasattr(rollout_fn, "primitive_collision_cost")
        and "link_pos_seq" in state_dict
        and "link_rot_seq" in state_dict
    ):
        p_cost = rollout_fn.primitive_collision_cost
        batch_size, horizon, n_links = state_dict["link_pos_seq"].shape[:3]
        if p_cost.batch_size != batch_size:
            p_cost.batch_size = batch_size
            p_cost.robot_world_coll.build_batch_features(
                batch_size * horizon,
                clone_pose=True,
                clone_points=True,
            )
        link_pos_batch = state_dict["link_pos_seq"].view(batch_size * horizon, n_links, 3)
        link_rot_batch = state_dict["link_rot_seq"].view(batch_size * horizon, n_links, 3, 3)
        raw_signed_dist = p_cost.robot_world_coll.check_robot_sphere_collisions(
            link_pos_batch,
            link_rot_batch,
        ).view(batch_size, horizon, n_links)
        primitive_margin = -(raw_signed_dist + p_cost.distance_threshold)
        margins.append(float(primitive_margin.amin().item()))

    if hasattr(rollout_fn, "robot_self_collision_cost") and "state_seq" in state_dict:
        self_cost = rollout_fn.robot_self_collision_cost
        n_dofs = rollout_fn.dynamics_model.n_dofs
        q_seq = state_dict["state_seq"][:, :, :n_dofs]
        q_flat = q_seq.reshape(-1, n_dofs)
        raw_signed_dist = self_cost.coll.check_self_collisions_nn(q_flat)
        self_margin = -(raw_signed_dist + self_cost.distance_threshold)
        margins.append(float(self_margin.amin().item()))

    if not margins:
        return math.nan
    return min(margins)


def _candidate_joint_goal(rng, scene_name):
    anchors = np.asarray(SCENE_SPECS[scene_name]["anchors"], dtype=np.float64)
    anchor = anchors[rng.integers(0, len(anchors))]
    jitter = np.asarray([0.07, 0.06, 0.06, 0.03, 0.03, 0.15], dtype=np.float64)
    q = anchor + rng.normal(loc=0.0, scale=jitter, size=6)
    q = np.clip(q, JOINT_LOW, JOINT_HIGH)
    delta_norm = np.linalg.norm(q - INIT_Q)
    if delta_norm < 0.08 or delta_norm > 0.9:
        return None
    return q


def _accepted_target(task, scene_name, q, initial_ee):
    spec = SCENE_SPECS[scene_name]
    ee_pos = _ee_pos(task, q)
    ee_dist = float(np.linalg.norm(ee_pos - initial_ee))
    if not (spec["ee_dist"][0] <= ee_dist <= spec["ee_dist"][1]):
        return None

    if spec["ee_bounds"] is not None:
        for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
            lo, hi = spec["ee_bounds"][axis]
            if not (lo <= ee_pos[idx] <= hi):
                return None

    safety_margin = _single_state_safety_margin(task, q)
    if math.isnan(safety_margin) or safety_margin < spec["min_margin"]:
        return None

    return {
        "goal_joint_positions": [round(float(x), 6) for x in q.tolist()],
        "goal_ee_pos": [round(float(x), 6) for x in ee_pos.tolist()],
        "goal_ee_distance_from_start": round(ee_dist, 6),
        "goal_safety_margin": round(float(safety_margin), 6),
    }


def generate_targets(output_path, targets_per_scene, seed, use_cuda=False):
    rng = np.random.default_rng(seed)
    tensor_args = _build_tensor_args(use_cuda)
    all_targets = {
        "meta": {
            "generator": "generate_round2_targets.py",
            "created_at": datetime.now().isoformat(),
            "targets_per_scene": int(targets_per_scene),
            "seed": int(seed),
            "init_joint_positions": [round(float(x), 6) for x in INIT_Q.tolist()],
        },
        "scenes": {},
    }

    for scene_name, spec in SCENE_SPECS.items():
        task = _build_task(
            "baseline",
            BASELINE_TASK_FILE,
            ROBOT_FILE,
            spec["world_file"],
            tensor_args,
        )
        try:
            initial_ee = _ee_pos(task, INIT_Q)
            accepted = []
            seen = set()
            max_trials = max(10000, targets_per_scene * 2000)
            for _ in range(max_trials):
                q = _candidate_joint_goal(rng, scene_name)
                if q is None:
                    continue
                rounded_key = tuple(np.round(q, 4).tolist())
                if rounded_key in seen:
                    continue
                target = _accepted_target(task, scene_name, q, initial_ee)
                if target is None:
                    continue
                seen.add(rounded_key)
                target["target_id"] = f"{scene_name}_{len(accepted):03d}"
                accepted.append(target)
                if len(accepted) >= targets_per_scene:
                    break

            if len(accepted) < targets_per_scene:
                raise RuntimeError(
                    f"Unable to generate enough targets for scene={scene_name}. "
                    f"Generated {len(accepted)}/{targets_per_scene}."
                )

            all_targets["scenes"][scene_name] = accepted
        finally:
            task.close()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_targets, f, indent=2)
    print(output_path, flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate reproducible round2 target sets")
    parser.add_argument("--output", required=True)
    parser.add_argument("--targets-per-scene", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260401)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()
    generate_targets(
        output_path=args.output,
        targets_per_scene=args.targets_per_scene,
        seed=args.seed,
        use_cuda=args.cuda,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
