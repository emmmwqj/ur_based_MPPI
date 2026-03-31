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
import yaml

torch.multiprocessing.set_start_method("spawn", force=True)

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from generate_round2_targets import (
    BASELINE_TASK_FILE,
    JOINT_HIGH,
    JOINT_LOW,
    WORLD_EASY,
    WORLD_NARROW,
    WORLD_OBSTACLE,
    generate_targets,
)
from run_controller_batch import _build_tensor_args
from storm_kit.mpc.rollout.arm_reacher import ArmReacher


SCENE_SPECS = {
    "easy": {
        "world_file": WORLD_EASY,
        "state_anchors": [
            [-0.10, -1.52, 1.52, -1.57, -1.57, 0.00],
            [0.00, -1.48, 1.48, -1.57, -1.57, 0.00],
            [0.10, -1.42, 1.42, -1.57, -1.57, 0.00],
            [0.22, -1.36, 1.36, -1.57, -1.57, 0.00],
            [0.30, -1.28, 1.28, -1.57, -1.57, 0.00],
        ],
        "start_anchors": [
            [-0.10, -1.52, 1.52, -1.57, -1.57, 0.00],
            [0.00, -1.48, 1.48, -1.57, -1.57, 0.00],
            [0.10, -1.42, 1.42, -1.57, -1.57, 0.00],
        ],
        "goal_anchors": [
            [0.12, -1.44, 1.44, -1.57, -1.57, 0.00],
            [0.22, -1.36, 1.36, -1.57, -1.57, 0.00],
            [0.30, -1.28, 1.28, -1.57, -1.57, 0.00],
        ],
        "ee_pair_dist": (0.03, 1.0),
        "joint_pair_dist": (0.08, 10.0),
        "min_margin": 0.02,
        "min_lateral_gap": 0.0,
        "ee_bounds": {
            "x": (0.45, 0.68),
            "y": (0.10, 0.42),
            "z": (0.36, 0.62),
        },
    },
    "obstacle": {
        "world_file": WORLD_OBSTACLE,
        "state_anchors": [
            [0.12, -1.42, 1.42, -1.57, -1.57, 0.00],
            [0.22, -1.38, 1.38, -1.57, -1.57, 0.00],
            [0.32, -1.32, 1.32, -1.57, -1.57, 0.00],
            [0.40, -1.28, 1.28, -1.57, -1.57, 0.00],
            [0.48, -1.22, 1.22, -1.57, -1.57, 0.00],
        ],
        "start_anchors": [
            [0.10, -1.42, 1.42, -1.57, -1.57, 0.00],
            [0.22, -1.38, 1.38, -1.57, -1.57, 0.00],
            [0.30, -1.32, 1.32, -1.57, -1.57, 0.00],
        ],
        "goal_anchors": [
            [0.26, -1.34, 1.34, -1.57, -1.57, 0.00],
            [0.38, -1.28, 1.28, -1.57, -1.57, 0.00],
            [0.48, -1.22, 1.22, -1.57, -1.57, 0.00],
        ],
        "ee_pair_dist": (0.03, 1.0),
        "joint_pair_dist": (0.08, 10.0),
        "min_margin": 0.005,
        "min_lateral_gap": 0.0,
        "ee_bounds": None,
    },
    "narrow": {
        "world_file": WORLD_NARROW,
        "state_anchors": [
            [0.00, -1.46, 1.46, -1.57, -1.57, 0.00],
            [0.10, -1.40, 1.40, -1.57, -1.57, 0.00],
            [0.18, -1.36, 1.36, -1.57, -1.57, 0.00],
            [0.26, -1.32, 1.32, -1.57, -1.57, 0.00],
            [0.34, -1.26, 1.26, -1.57, -1.57, 0.00],
        ],
        "start_anchors": [
            [0.00, -1.46, 1.46, -1.57, -1.57, 0.00],
            [0.10, -1.40, 1.40, -1.57, -1.57, 0.00],
            [0.18, -1.36, 1.36, -1.57, -1.57, 0.00],
        ],
        "goal_anchors": [
            [0.22, -1.34, 1.34, -1.57, -1.57, 0.00],
            [0.30, -1.30, 1.30, -1.57, -1.57, 0.00],
            [0.38, -1.24, 1.24, -1.57, -1.57, 0.00],
        ],
        "ee_pair_dist": (0.03, 1.0),
        "joint_pair_dist": (0.06, 10.0),
        "min_margin": 0.001,
        "min_lateral_gap": 0.0,
        "ee_bounds": None,
    },
}

JITTER = np.asarray([0.06, 0.05, 0.05, 0.02, 0.02, 0.12], dtype=np.float64)


def _candidate_joint_state(rng, anchors):
    anchor = np.asarray(anchors[rng.integers(0, len(anchors))], dtype=np.float64)
    q = anchor + rng.normal(loc=0.0, scale=JITTER, size=6)
    return np.clip(q, JOINT_LOW, JOINT_HIGH)


def _bounds_ok(ee_pos, bounds):
    if bounds is None:
        return True
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        lo, hi = bounds[axis]
        if not (lo <= ee_pos[idx] <= hi):
            return False
    return True


def _build_rollout_bundle(scene_name, tensor_args):
    with open(SCENE_SPECS[scene_name]["world_file"]) as f:
        world_params = yaml.safe_load(f)
    with open(BASELINE_TASK_FILE) as f:
        exp_params = yaml.safe_load(f)
    exp_params["robot_params"] = exp_params["model"]
    rollout_fn = ArmReacher(
        exp_params=exp_params,
        tensor_args=tensor_args,
        world_params=world_params,
    )
    return {
        "rollout_fn": rollout_fn,
        "tensor_args": tensor_args,
        "ee_link_name": exp_params["model"]["ee_link_name"],
    }


def _ee_pos(bundle, q):
    rollout_fn = bundle["rollout_fn"]
    tensor_args = bundle["tensor_args"]
    q_tensor = torch.as_tensor(q, **tensor_args).view(1, -1)
    qd_tensor = torch.zeros_like(q_tensor)
    ee_pos, _ = rollout_fn.dynamics_model.robot_model.compute_forward_kinematics(
        q_tensor,
        qd_tensor,
        link_name=bundle["ee_link_name"],
    )
    return ee_pos.detach().cpu().numpy().reshape(-1)


def _static_state_safety_margin(bundle, q):
    rollout_fn = bundle["rollout_fn"]
    tensor_args = bundle["tensor_args"]
    dynamics_model = rollout_fn.dynamics_model
    q_tensor = torch.as_tensor(q, **tensor_args).view(1, -1)
    qd_tensor = torch.zeros_like(q_tensor)
    robot_model = dynamics_model.robot_model
    robot_model.compute_fk_and_jacobian(
        q_tensor,
        qd_tensor,
        link_name=dynamics_model.ee_link_name,
    )
    margins = []

    if hasattr(rollout_fn, "primitive_collision_cost"):
        p_cost = rollout_fn.primitive_collision_cost
        n_links = len(dynamics_model.link_names)
        link_pos = torch.empty(1, n_links, 3, **tensor_args)
        link_rot = torch.empty(1, n_links, 3, 3, **tensor_args)
        for idx, link_name in enumerate(dynamics_model.link_names):
            curr_pos, curr_rot = robot_model.get_link_pose(link_name)
            link_pos[:, idx, :] = curr_pos.view(1, 3)
            link_rot[:, idx, :, :] = curr_rot.view(1, 3, 3)
        batch_size = 1
        horizon = 1
        if p_cost.batch_size != batch_size:
            p_cost.batch_size = batch_size
            p_cost.robot_world_coll.build_batch_features(
                batch_size * horizon,
                clone_pose=True,
                clone_points=True,
            )
        link_pos_batch = link_pos.view(batch_size * horizon, n_links, 3)
        link_rot_batch = link_rot.view(batch_size * horizon, n_links, 3, 3)
        raw_signed_dist = p_cost.robot_world_coll.check_robot_sphere_collisions(
            link_pos_batch,
            link_rot_batch,
        ).view(batch_size, horizon, n_links)
        primitive_margin = -(raw_signed_dist + p_cost.distance_threshold)
        margins.append(float(primitive_margin.amin().item()))

    if hasattr(rollout_fn, "robot_self_collision_cost"):
        self_cost = rollout_fn.robot_self_collision_cost
        raw_signed_dist = self_cost.coll.check_self_collisions_nn(q_tensor)
        self_margin = -(raw_signed_dist + self_cost.distance_threshold)
        margins.append(float(self_margin.amin().item()))

    if not margins:
        return math.nan
    return min(margins)


def _accepted_state(task, scene_name, q):
    spec = SCENE_SPECS[scene_name]
    ee_pos = _ee_pos(task, q)
    if not _bounds_ok(ee_pos, spec["ee_bounds"]):
        return None
    safety_margin = _static_state_safety_margin(task, q)
    if math.isnan(safety_margin) or safety_margin < spec["min_margin"]:
        return None
    return {
        "joint_positions": [round(float(x), 6) for x in q.tolist()],
        "ee_pos": [round(float(x), 6) for x in ee_pos.tolist()],
        "safety_margin": round(float(safety_margin), 6),
    }


def _pair_is_valid(scene_name, start_state, goal_state):
    spec = SCENE_SPECS[scene_name]
    start_q = np.asarray(start_state["joint_positions"], dtype=np.float64)
    goal_q = np.asarray(goal_state["joint_positions"], dtype=np.float64)
    start_ee = np.asarray(start_state["ee_pos"], dtype=np.float64)
    goal_ee = np.asarray(goal_state["ee_pos"], dtype=np.float64)

    joint_dist = float(np.linalg.norm(goal_q - start_q))
    if joint_dist < spec["joint_pair_dist"][0]:
        return False

    ee_dist = float(np.linalg.norm(goal_ee - start_ee))
    if ee_dist < spec["ee_pair_dist"][0]:
        return False

    lateral_gap = float(abs(goal_ee[1] - start_ee[1]))
    if spec["min_lateral_gap"] > 0.0 and lateral_gap < spec["min_lateral_gap"]:
        return False

    return True


def generate_pairs(output_path, pairs_per_scene, seed, use_cuda=False, state_pool_path=None):
    rng = np.random.default_rng(seed)
    result = {
        "meta": {
            "generator": "generate_round3_pairs.py",
            "created_at": datetime.now().isoformat(),
            "pairs_per_scene": int(pairs_per_scene),
            "seed": int(seed),
        },
        "scenes": {},
    }

    if state_pool_path is None:
        state_pool_size = max(20, min(24, int(pairs_per_scene)))
        state_pool_path = os.path.join(
            os.path.dirname(os.path.abspath(output_path)),
            ".round3_safe_state_pool.json",
        )
        generate_targets(
            output_path=state_pool_path,
            targets_per_scene=state_pool_size,
            seed=seed,
            use_cuda=use_cuda,
        )

    with open(state_pool_path) as f:
        safe_pool = json.load(f)

    for scene_name, spec in SCENE_SPECS.items():
        states = safe_pool["scenes"][scene_name]
        candidate_pairs = []
        for start_idx, start in enumerate(states):
            start_state = {
                "joint_positions": start["goal_joint_positions"],
                "ee_pos": start["goal_ee_pos"],
                "safety_margin": start["goal_safety_margin"],
            }
            for goal_idx, goal in enumerate(states):
                if start_idx == goal_idx:
                    continue
                goal_state = {
                    "joint_positions": goal["goal_joint_positions"],
                    "ee_pos": goal["goal_ee_pos"],
                    "safety_margin": goal["goal_safety_margin"],
                }
                if not _pair_is_valid(scene_name, start_state, goal_state):
                    continue
                start_ee = np.asarray(start_state["ee_pos"], dtype=np.float64)
                goal_ee = np.asarray(goal_state["ee_pos"], dtype=np.float64)
                candidate_pairs.append(
                    {
                        "initial_joint_positions": start_state["joint_positions"],
                        "initial_ee_pos": start_state["ee_pos"],
                        "initial_safety_margin": start_state["safety_margin"],
                        "goal_joint_positions": goal_state["joint_positions"],
                        "goal_ee_pos": goal_state["ee_pos"],
                        "goal_safety_margin": goal_state["safety_margin"],
                        "pair_ee_distance": round(float(np.linalg.norm(goal_ee - start_ee)), 6),
                        "initial_state_id": start["target_id"],
                        "goal_state_id": goal["target_id"],
                    }
                )

        rng.shuffle(candidate_pairs)
        if len(candidate_pairs) < pairs_per_scene:
            raise RuntimeError(
                f"Unable to generate enough pairs for scene={scene_name}. "
                f"Generated {len(candidate_pairs)}/{pairs_per_scene}."
            )
        selected = []
        for idx, pair in enumerate(candidate_pairs[:pairs_per_scene]):
            pair = dict(pair)
            pair["pair_id"] = f"{scene_name}_{idx:03d}"
            selected.append(pair)
        result["scenes"][scene_name] = selected

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(output_path, flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate reproducible round3 initial-goal pairs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs-per-scene", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument("--state-pool-path", default=None)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()
    generate_pairs(
        output_path=args.output,
        pairs_per_scene=args.pairs_per_scene,
        seed=args.seed,
        use_cuda=args.cuda,
        state_pool_path=args.state_pool_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
