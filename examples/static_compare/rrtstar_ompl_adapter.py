#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.static_compare.utils.metrics import trajectory_length_joint
from examples.static_compare.utils.static_collision_checker import StaticTallCollisionChecker

try:
    from ompl import base as ob
    from ompl import geometric as og

    rrtstar_available = True
    skipped_reason = ""
except Exception as e:  # pragma: no cover - depends on local OMPL install
    ob = None
    og = None
    rrtstar_available = False
    skipped_reason = f"OMPL Python binding unavailable: {e}"


class _StaticValidityChecker(ob.StateValidityChecker if rrtstar_available else object):
    def __init__(self, si, checker: StaticTallCollisionChecker):
        if rrtstar_available:
            super().__init__(si)
        self.checker = checker
        self.dim = len(checker.joint_lower)

    def isValid(self, state) -> bool:
        q = np.array([float(state[i]) for i in range(self.dim)], dtype=float)
        return self.checker.is_state_valid(q)


def _state_to_array(state, dim: int) -> np.ndarray:
    return np.array([float(state[i]) for i in range(dim)], dtype=float)


def _make_state(space, q: Iterable[float]):
    state = space.allocState()
    for idx, value in enumerate(np.asarray(q, dtype=float)):
        state[idx] = float(value)
    return state


def _extract_path(solution_path, dim: int) -> list[list[float]]:
    return [_state_to_array(solution_path.getState(i), dim).tolist() for i in range(solution_path.getStateCount())]


def plan_joint_space_rrtstar(
    start: Iterable[float],
    goal: Iterable[float],
    checker: StaticTallCollisionChecker,
    planning_time_limit: float = 2.0,
    goal_bias: float = 0.05,
    interpolation_resolution: float = 0.05,
    collision_check_resolution: float = 0.05,
) -> dict:
    """Plan with OMPL RRTstar in UR7e joint space.

    Internal state validity calls static_collision_checker. Gazebo is not used.
    The returned path is revalidated with static_collision_checker.check_motion.
    """
    if not rrtstar_available:
        return {
            "rrtstar_available": False,
            "rrtstar_path_found": False,
            "rrtstar_exact_solution": False,
            "rrtstar_path_valid": False,
            "skipped_reason": skipped_reason,
            "path": [],
            "planning_time": 0.0,
            "path_length_joint": math.nan,
            "minimum_safety_margin": math.nan,
            "number_of_validity_checks": 0,
            "number_of_invalid_states": 0,
        }

    dim = len(checker.joint_lower)
    checker.reset_counters()
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    start_valid = checker.check_state(start)
    goal_valid = checker.check_state(goal)
    if not start_valid.valid or not goal_valid.valid:
        return {
            "rrtstar_available": True,
            "rrtstar_path_found": False,
            "rrtstar_exact_solution": False,
            "rrtstar_path_valid": False,
            "skipped_reason": "RRT* start or goal is invalid under static_collision_checker",
            "path": [],
            "planning_time": 0.0,
            "path_length_joint": math.nan,
            "minimum_safety_margin": min(start_valid.minimum_safety_margin, goal_valid.minimum_safety_margin),
            "number_of_validity_checks": checker.number_of_validity_checks,
            "number_of_invalid_states": checker.number_of_invalid_states,
        }

    space = ob.RealVectorStateSpace(dim)
    bounds = ob.RealVectorBounds(dim)
    for i in range(dim):
        bounds.setLow(i, float(checker.joint_lower[i]))
        bounds.setHigh(i, float(checker.joint_upper[i]))
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(_StaticValidityChecker(si, checker))
    extent = float(np.linalg.norm(checker.joint_upper - checker.joint_lower))
    if extent > 0.0:
        si.setStateValidityCheckingResolution(float(collision_check_resolution) / extent)
    si.setup()

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(_make_state(space, start), _make_state(space, goal), 1.0e-3)

    planner = og.RRTstar(si)
    if hasattr(planner, "setGoalBias"):
        planner.setGoalBias(float(goal_bias))
    planner.setProblemDefinition(pdef)
    planner.setup()

    t0 = time.time()
    solved = planner.solve(float(planning_time_limit))
    planning_time = time.time() - t0

    path: list[list[float]] = []
    has_solution = bool(solved) and bool(pdef.hasSolution())
    exact_solution = bool(pdef.hasExactSolution())
    if has_solution:
        solution_path = pdef.getSolutionPath()
        try:
            solution_path.interpolate(max(2, int(math.ceil(trajectory_length_joint([start, goal]) / interpolation_resolution))))
        except Exception:
            try:
                solution_path.interpolate()
            except Exception:
                pass
        path = _extract_path(solution_path, dim)

    path_metrics = checker.path_metrics(path, motion_resolution=collision_check_resolution) if path else {
        "valid": False,
        "minimum_safety_margin": math.nan,
        "number_of_validity_checks": 0,
        "number_of_invalid_states": 0,
    }
    path_found = bool(exact_solution and path_metrics["valid"])
    if path_found:
        reason = ""
    elif has_solution and not exact_solution:
        reason = "OMPL RRTstar returned only an approximate solution within the planning time limit"
    elif path and not path_metrics["valid"]:
        reason = "OMPL RRTstar path failed static_collision_checker motion validation"
    else:
        reason = "OMPL RRTstar did not find a path within the planning time limit"

    return {
        "rrtstar_available": True,
        "rrtstar_path_found": path_found,
        "rrtstar_exact_solution": exact_solution,
        "rrtstar_path_valid": bool(path_metrics["valid"]),
        "skipped_reason": reason,
        "path": path,
        "planning_time": float(planning_time),
        "path_length_joint": trajectory_length_joint(path) if path else math.nan,
        "minimum_safety_margin": float(path_metrics["minimum_safety_margin"]),
        "number_of_validity_checks": int(checker.number_of_validity_checks),
        "number_of_invalid_states": int(checker.number_of_invalid_states),
    }
