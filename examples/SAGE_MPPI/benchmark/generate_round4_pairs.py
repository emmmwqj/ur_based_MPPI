#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

torch.multiprocessing.set_start_method("spawn", force=True)

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from generate_round3_pairs import _accepted_state, _build_rollout_bundle, JOINT_HIGH, JOINT_LOW
from run_controller_batch import _build_tensor_args


BASE_SCENE_SPECS = {
    "obstacle": {
        "anchors": [
            [0.05, -1.55, 1.55, -1.57, -1.57, 0.00],
            [0.20, -1.45, 1.45, -1.57, -1.57, 0.00],
            [0.38, -1.32, 1.32, -1.57, -1.57, 0.00],
            [0.55, -1.15, 1.15, -1.57, -1.57, 0.00],
            [0.70, -1.05, 1.05, -1.57, -1.57, 0.00],
        ],
        "jitter": [0.12, 0.10, 0.10, 0.06, 0.06, 0.18],
        "target_states": 160,
        "state_usage_cap_medium": 6,
        "state_usage_cap_hard": 4,
    },
    "narrow": {
        "anchors": [
            [-0.15, -1.55, 1.55, -1.57, -1.57, 0.00],
            [0.02, -1.46, 1.46, -1.57, -1.57, 0.00],
            [0.20, -1.34, 1.34, -1.57, -1.57, 0.00],
            [0.40, -1.22, 1.22, -1.57, -1.57, 0.00],
            [0.55, -1.10, 1.10, -1.57, -1.57, 0.00],
        ],
        "jitter": [0.12, 0.10, 0.10, 0.05, 0.05, 0.18],
        "target_states": 120,
        "state_usage_cap_medium": 6,
        "state_usage_cap_hard": 4,
    },
}

ROUND4_SCENE_SPECS = {
    "obstacle_medium": {
        "base_scene": "obstacle",
        "difficulty": "medium",
        "max_steps": 120,
        "ee_quantile": 0.58,
        "lateral_quantile": 0.60,
        "margin_quantile": 0.40,
        "score_quantile_lo": 0.55,
        "score_quantile_hi": 0.82,
        "pair_margin_floor": 0.006,
    },
    "obstacle_hard": {
        "base_scene": "obstacle",
        "difficulty": "hard",
        "max_steps": 150,
        "ee_quantile": 0.76,
        "lateral_quantile": 0.70,
        "margin_quantile": 0.35,
        "score_quantile_lo": 0.78,
        "score_quantile_hi": 1.00,
        "pair_margin_floor": 0.0045,
    },
    "narrow_medium": {
        "base_scene": "narrow",
        "difficulty": "medium",
        "max_steps": 120,
        "ee_quantile": 0.55,
        "lateral_quantile": 0.56,
        "margin_quantile": 0.45,
        "score_quantile_lo": 0.55,
        "score_quantile_hi": 0.84,
        "pair_margin_floor": 0.0035,
    },
    "narrow_hard": {
        "base_scene": "narrow",
        "difficulty": "hard",
        "max_steps": 150,
        "ee_quantile": 0.76,
        "lateral_quantile": 0.68,
        "margin_quantile": 0.32,
        "score_quantile_lo": 0.80,
        "score_quantile_hi": 1.00,
        "pair_margin_floor": 0.0030,
    },
}


def _quantile(values, q):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return math.nan
    return float(np.quantile(values, q))


def _normalize(value, scale):
    if scale <= 1.0e-8:
        return 0.0
    return float(np.clip(value / scale, 0.0, 2.0))


def _sample_safe_states(base_scene, seed, target_states, use_cuda=False):
    tensor_args = _build_tensor_args(use_cuda)
    rollout_bundle = _build_rollout_bundle(base_scene, tensor_args)
    spec = BASE_SCENE_SPECS[base_scene]
    anchors = np.asarray(spec["anchors"], dtype=np.float64)
    jitter = np.asarray(spec["jitter"], dtype=np.float64)
    rng = np.random.default_rng(seed)

    accepted = []
    seen = set()
    max_trials = max(4000, target_states * 40)

    for _ in range(max_trials):
        anchor = anchors[rng.integers(0, len(anchors))]
        q = anchor + rng.normal(loc=0.0, scale=jitter, size=6)
        q = np.clip(q, JOINT_LOW, JOINT_HIGH)
        key = tuple(np.round(q, 4).tolist())
        if key in seen:
            continue
        state = _accepted_state(rollout_bundle, base_scene, q)
        if state is None:
            continue
        seen.add(key)
        state = dict(state)
        state["state_id"] = f"{base_scene}_state_{len(accepted):03d}"
        accepted.append(state)
        if len(accepted) >= target_states:
            break

    if len(accepted) < target_states:
        raise RuntimeError(
            f"Unable to harvest enough safe states for base_scene={base_scene}. "
            f"Generated {len(accepted)}/{target_states}."
        )
    return accepted


def _build_candidate_pairs(base_scene, states):
    ee_distances = []
    joint_distances = []
    lateral_gaps = []
    vertical_gaps = []
    pair_margins = []
    raw_candidates = []

    for start_idx, start in enumerate(states):
        start_q = np.asarray(start["joint_positions"], dtype=np.float64)
        start_ee = np.asarray(start["ee_pos"], dtype=np.float64)
        for goal_idx, goal in enumerate(states):
            if start_idx == goal_idx:
                continue
            goal_q = np.asarray(goal["joint_positions"], dtype=np.float64)
            goal_ee = np.asarray(goal["ee_pos"], dtype=np.float64)

            ee_dist = float(np.linalg.norm(goal_ee - start_ee))
            joint_dist = float(np.linalg.norm(goal_q - start_q))
            lateral_gap = float(abs(goal_ee[1] - start_ee[1]))
            vertical_gap = float(abs(goal_ee[2] - start_ee[2]))
            pair_margin = float(min(start["safety_margin"], goal["safety_margin"]))

            raw_candidates.append(
                {
                    "base_scene": base_scene,
                    "initial_state_id": start["state_id"],
                    "goal_state_id": goal["state_id"],
                    "initial_joint_positions": start["joint_positions"],
                    "goal_joint_positions": goal["joint_positions"],
                    "initial_ee_pos": start["ee_pos"],
                    "goal_ee_pos": goal["ee_pos"],
                    "initial_safety_margin": float(start["safety_margin"]),
                    "goal_safety_margin": float(goal["safety_margin"]),
                    "pair_ee_distance": ee_dist,
                    "pair_joint_distance": joint_dist,
                    "lateral_gap": lateral_gap,
                    "vertical_gap": vertical_gap,
                    "pair_min_safety_margin": pair_margin,
                    "unordered_state_key": "|".join(sorted([start["state_id"], goal["state_id"]])),
                }
            )
            ee_distances.append(ee_dist)
            joint_distances.append(joint_dist)
            lateral_gaps.append(lateral_gap)
            vertical_gaps.append(vertical_gap)
            pair_margins.append(pair_margin)

    ee_scale = _quantile(ee_distances, 0.90)
    joint_scale = _quantile(joint_distances, 0.90)
    lateral_scale = _quantile(lateral_gaps, 0.90)
    vertical_scale = _quantile(vertical_gaps, 0.90)
    margin_scale = _quantile(pair_margins, 0.75)

    for candidate in raw_candidates:
        margin_term = 1.0 - np.clip(candidate["pair_min_safety_margin"] / max(margin_scale, 1.0e-6), 0.0, 1.0)
        if base_scene == "obstacle":
            score = (
                0.40 * _normalize(candidate["pair_ee_distance"], ee_scale)
                + 0.25 * _normalize(candidate["lateral_gap"], lateral_scale)
                + 0.15 * _normalize(candidate["pair_joint_distance"], joint_scale)
                + 0.10 * _normalize(candidate["vertical_gap"], vertical_scale)
                + 0.10 * float(margin_term)
            )
        else:
            score = (
                0.35 * _normalize(candidate["pair_ee_distance"], ee_scale)
                + 0.30 * _normalize(candidate["lateral_gap"], lateral_scale)
                + 0.10 * _normalize(candidate["pair_joint_distance"], joint_scale)
                + 0.10 * _normalize(candidate["vertical_gap"], vertical_scale)
                + 0.15 * float(margin_term)
            )
        candidate["difficulty_score"] = float(score)

    return raw_candidates


def _filter_candidates(candidates, spec):
    ee_min = _quantile([c["pair_ee_distance"] for c in candidates], spec["ee_quantile"])
    lateral_min = _quantile([c["lateral_gap"] for c in candidates], spec["lateral_quantile"])
    margin_max = _quantile([c["pair_min_safety_margin"] for c in candidates], spec["margin_quantile"])

    filtered = [
        c
        for c in candidates
        if c["pair_ee_distance"] >= ee_min
        and c["lateral_gap"] >= lateral_min
        and c["pair_min_safety_margin"] <= margin_max
        and c["pair_min_safety_margin"] >= spec["pair_margin_floor"]
    ]
    if not filtered:
        raise RuntimeError(f"No candidate pairs survived difficulty filter for {spec}")

    score_lo = _quantile([c["difficulty_score"] for c in filtered], spec["score_quantile_lo"])
    score_hi = _quantile([c["difficulty_score"] for c in filtered], spec["score_quantile_hi"])
    selected_pool = [
        c
        for c in filtered
        if c["difficulty_score"] >= score_lo and c["difficulty_score"] <= score_hi + 1.0e-8
    ]
    if not selected_pool:
        raise RuntimeError(f"No candidate pairs survived score-range filter for {spec}")

    return selected_pool, {
        "ee_min": ee_min,
        "lateral_min": lateral_min,
        "margin_max": margin_max,
        "score_lo": score_lo,
        "score_hi": score_hi,
    }


def _greedy_select_pairs(scene_name, candidates, target_count, state_usage_cap):
    selected = []
    usage = defaultdict(int)
    used_unordered = set()

    ranked = sorted(
        candidates,
        key=lambda item: (
            item["difficulty_score"],
            item["pair_ee_distance"],
            item["lateral_gap"],
            -item["pair_min_safety_margin"],
        ),
        reverse=True,
    )

    cap_schedule = [state_usage_cap, state_usage_cap + 2, state_usage_cap + 4, target_count]
    for active_cap in cap_schedule:
        for candidate in ranked:
            if len(selected) >= target_count:
                break
            if candidate["unordered_state_key"] in used_unordered:
                continue
            if usage[candidate["initial_state_id"]] >= active_cap:
                continue
            if usage[candidate["goal_state_id"]] >= active_cap:
                continue
            selected.append(candidate)
            used_unordered.add(candidate["unordered_state_key"])
            usage[candidate["initial_state_id"]] += 1
            usage[candidate["goal_state_id"]] += 1
        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        raise RuntimeError(
            f"Unable to select enough unique harder pairs for {scene_name}. "
            f"Generated {len(selected)}/{target_count}."
        )
    return selected[:target_count]


def generate_pairs(output_path, pairs_per_scene, seed, use_cuda=False):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result = {
        "meta": {
            "generator": "generate_round4_pairs.py",
            "created_at": datetime.now().isoformat(),
            "pairs_per_scene": int(pairs_per_scene),
            "seed": int(seed),
            "round4_note": (
                "Round4 deliberately increases task difficulty by combining larger initial-goal "
                "end-effector displacement, lower but positive safety margins, and larger lateral "
                "motion through obstacle or corridor-constrained regions. Pairs remain filtered "
                "through the existing collision checks so they are safe states, but they sit closer "
                "to failure boundaries than round3."
            ),
        },
        "scene_meta": {},
        "scenes": {},
    }

    base_state_pools = {}
    for base_scene, base_spec in BASE_SCENE_SPECS.items():
        base_state_pools[base_scene] = _sample_safe_states(
            base_scene=base_scene,
            seed=seed + (101 if base_scene == "narrow" else 0),
            target_states=base_spec["target_states"],
            use_cuda=use_cuda,
        )

    base_candidates = {
        base_scene: _build_candidate_pairs(base_scene, states)
        for base_scene, states in base_state_pools.items()
    }

    for scene_name, scene_spec in ROUND4_SCENE_SPECS.items():
        base_scene = scene_spec["base_scene"]
        candidates, thresholds = _filter_candidates(base_candidates[base_scene], scene_spec)
        usage_cap = BASE_SCENE_SPECS[base_scene][f"state_usage_cap_{scene_spec['difficulty']}"]
        chosen = _greedy_select_pairs(
            scene_name=scene_name,
            candidates=candidates,
            target_count=pairs_per_scene,
            state_usage_cap=usage_cap,
        )

        scene_pairs = []
        for idx, pair in enumerate(chosen):
            pair_record = dict(pair)
            pair_record["scene_condition"] = scene_name
            pair_record["difficulty"] = scene_spec["difficulty"]
            pair_record["max_steps"] = int(scene_spec["max_steps"])
            pair_record["pair_id"] = f"{scene_name}_{idx:03d}"
            pair_record["difficulty_reason"] = (
                "farther initial-goal displacement + smaller positive safety margin + larger lateral motion"
            )
            scene_pairs.append(pair_record)

        result["scene_meta"][scene_name] = {
            "base_scene": base_scene,
            "difficulty": scene_spec["difficulty"],
            "max_steps": int(scene_spec["max_steps"]),
            "pairs_per_scene": int(pairs_per_scene),
            "selection_thresholds": thresholds,
            "num_candidate_pairs": len(candidates),
            "num_harvested_states": len(base_state_pools[base_scene]),
        }
        result["scenes"][scene_name] = scene_pairs

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(output_path, flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate harder round4 initial-goal pairs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs-per-scene", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260404)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()
    generate_pairs(
        output_path=args.output,
        pairs_per_scene=args.pairs_per_scene,
        seed=args.seed,
        use_cuda=args.cuda,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
