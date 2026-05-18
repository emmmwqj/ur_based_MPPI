from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .io_utils import load_yaml, repo_root, resolve_repo_path

ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storm_kit.differentiable_robot_model.differentiable_robot_model import (  # noqa: E402
    DifferentiableRobotModel,
)
from storm_kit.util_file import get_assets_path  # noqa: E402


DEFAULT_TASK_FILE = "examples/sim_gazebo/config/ur7e_reacher_gazebo_tall.yml"
DEFAULT_ROBOT_FILE = "examples/sim_gazebo/config/ur7e_robot_gazebo.yml"
DEFAULT_WORLD_FILE = "examples/sim_gazebo/config/collision_world_gazebo_tall.yml"
DEFAULT_COLLISION_SPHERES_FILE = "examples/sim_gazebo/config/ur7e_collision_spheres.yml"
DEFAULT_JOINT_LIMIT_FILE = "examples/sim_gazebo/config/ur7e_gazebo.yml"


@dataclass
class StateValidity:
    valid: bool
    minimum_safety_margin: float
    number_of_validity_checks: int
    number_of_invalid_states: int


def _quat_xyzw_to_matrix(quat_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=float)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _signed_distance_to_box(point: np.ndarray, half_extents: np.ndarray) -> float:
    q = np.abs(point) - half_extents
    outside = np.maximum(q, 0.0)
    outside_dist = float(np.linalg.norm(outside))
    inside_dist = float(min(max(q[0], q[1], q[2]), 0.0))
    return outside_dist + inside_dist


class StaticTallCollisionChecker:
    """Analytic UR7e collision-sphere checker for the existing tall scene.

    The checker uses robot link collision spheres from the tall STORM/SAGE
    configs and computes signed margins against primitive world spheres and
    boxes. It intentionally does not call Gazebo.
    """

    def __init__(
        self,
        task_file: str | Path = DEFAULT_TASK_FILE,
        robot_file: str | Path = DEFAULT_ROBOT_FILE,
        world_file: str | Path = DEFAULT_WORLD_FILE,
        collision_spheres_file: str | Path = DEFAULT_COLLISION_SPHERES_FILE,
        joint_limit_file: str | Path = DEFAULT_JOINT_LIMIT_FILE,
        include_ground: bool = True,
        collision_threshold: float = 0.0,
        tensor_args: dict | None = None,
    ) -> None:
        self.task_file = resolve_repo_path(task_file)
        self.robot_file = resolve_repo_path(robot_file)
        self.world_file = resolve_repo_path(world_file)
        self.collision_spheres_file = resolve_repo_path(collision_spheres_file)
        self.joint_limit_file = resolve_repo_path(joint_limit_file)
        self.include_ground = bool(include_ground)
        self.collision_threshold = float(collision_threshold)
        self.tensor_args = tensor_args or {"device": torch.device("cpu"), "dtype": torch.float32}
        self.number_of_validity_checks = 0
        self.number_of_invalid_states = 0

        self.task_params = load_yaml(self.task_file)
        self.robot_params = load_yaml(self.robot_file)
        self.world_params = load_yaml(self.world_file)
        self.sphere_params = load_yaml(self.collision_spheres_file)
        self.gazebo_joint_limits = load_yaml(self.joint_limit_file).get("joint_limits", {})

        model_params = self.task_params["model"]
        urdf_path = Path(get_assets_path()) / model_params["urdf_path"]
        self.ee_link_name = model_params["ee_link_name"]
        self.link_names = list(model_params["robot_collision_params"]["link_objs"])
        self.robot_model = DifferentiableRobotModel(str(urdf_path), None, tensor_args=self.tensor_args)

        sim_params = self.robot_params.get("sim_params", {})
        robot_pose = sim_params.get("robot_pose", [0, 0, 0, 0, 0, 0, 1])
        self.robot_position_world = np.asarray(robot_pose[:3], dtype=float)
        self.robot_rotation_world = _quat_xyzw_to_matrix(robot_pose[3:])
        self.obstacles = self._load_obstacles()
        self.collision_spheres = self._load_collision_spheres()
        self.joint_lower, self.joint_upper = self._load_joint_limits()

    @property
    def warnings(self) -> list[str]:
        return [
            "Uses analytic link-sphere vs primitive-obstacle distances.",
            "Does not query Gazebo contacts or Gazebo physics.",
            "Does not include robot self-collision in this pilot checker.",
        ]

    def reset_counters(self) -> None:
        self.number_of_validity_checks = 0
        self.number_of_invalid_states = 0

    def _load_obstacles(self) -> list[dict]:
        coll_objs = self.world_params.get("world_model", {}).get("coll_objs", {})
        obstacles: list[dict] = []
        for name, spec in coll_objs.get("sphere", {}).items():
            if name == "ground" and not self.include_ground:
                continue
            obstacles.append(
                {
                    "type": "sphere",
                    "name": name,
                    "position": np.asarray(spec["position"], dtype=float),
                    "radius": float(spec["radius"]),
                }
            )
        for name, spec in coll_objs.get("cube", {}).items():
            if name == "ground" and not self.include_ground:
                continue
            pose = np.asarray(spec["pose"], dtype=float)
            obstacles.append(
                {
                    "type": "box",
                    "name": name,
                    "position": pose[:3],
                    "rotation": _quat_xyzw_to_matrix(pose[3:]),
                    "half_extents": 0.5 * np.asarray(spec["dims"], dtype=float),
                }
            )
        return obstacles

    def _load_collision_spheres(self) -> list[dict]:
        by_link = self.sphere_params["collision_spheres"]
        spheres: list[dict] = []
        for link_name in self.link_names:
            for sphere in by_link.get(link_name, []):
                spheres.append(
                    {
                        "link_name": link_name,
                        "center": np.asarray(sphere["center"], dtype=float),
                        "radius": float(sphere["radius"]),
                    }
                )
        return spheres

    def _load_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = []
        upper = []
        for lim in self.robot_model.get_joint_limits():
            lower.append(float(lim["lower"]))
            upper.append(float(lim["upper"]))
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)

        fallback_lower = np.asarray(self.gazebo_joint_limits.get("lower", []), dtype=float)
        fallback_upper = np.asarray(self.gazebo_joint_limits.get("upper", []), dtype=float)
        if lower_arr.shape != (6,) or upper_arr.shape != (6,) or not np.all(np.isfinite(lower_arr + upper_arr)):
            lower_arr = fallback_lower
            upper_arr = fallback_upper
        return lower_arr, upper_arr

    def within_joint_limits(self, q: Iterable[float], tolerance: float = 1.0e-6) -> bool:
        q_arr = np.asarray(q, dtype=float)
        return bool(
            q_arr.shape == self.joint_lower.shape
            and np.all(q_arr >= self.joint_lower - tolerance)
            and np.all(q_arr <= self.joint_upper + tolerance)
        )

    def compute_link_poses(self, q: Iterable[float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        q_tensor = torch.as_tensor(np.asarray(q, dtype=float), **self.tensor_args).reshape(1, -1)
        qd_tensor = torch.zeros_like(q_tensor)
        self.robot_model.compute_forward_kinematics(q_tensor, qd_tensor, self.ee_link_name)
        poses = {}
        for link_name in self.link_names:
            pos, rot = self.robot_model.get_link_pose(link_name)
            poses[link_name] = (
                np.ravel(pos.detach().cpu().numpy()).astype(float),
                np.asarray(rot[0].detach().cpu().numpy(), dtype=float),
            )
        return poses

    def ee_position(self, q: Iterable[float]) -> np.ndarray:
        q_tensor = torch.as_tensor(np.asarray(q, dtype=float), **self.tensor_args).reshape(1, -1)
        qd_tensor = torch.zeros_like(q_tensor)
        pos, _ = self.robot_model.compute_forward_kinematics(q_tensor, qd_tensor, self.ee_link_name)
        pos_robot = np.ravel(pos.detach().cpu().numpy()).astype(float)
        return self.robot_position_world + self.robot_rotation_world @ pos_robot

    def world_collision_spheres(self, q: Iterable[float]) -> list[dict]:
        poses = self.compute_link_poses(q)
        world_spheres: list[dict] = []
        for sphere in self.collision_spheres:
            link_pos, link_rot = poses[sphere["link_name"]]
            center_robot = link_rot @ sphere["center"] + link_pos
            center_world = self.robot_position_world + self.robot_rotation_world @ center_robot
            world_spheres.append(
                {
                    "link_name": sphere["link_name"],
                    "center": center_world,
                    "radius": sphere["radius"],
                }
            )
        return world_spheres

    def minimum_safety_margin(self, q: Iterable[float]) -> float:
        margins: list[float] = []
        for robot_sphere in self.world_collision_spheres(q):
            center = robot_sphere["center"]
            radius = robot_sphere["radius"]
            for obstacle in self.obstacles:
                if obstacle["type"] == "sphere":
                    dist = float(np.linalg.norm(center - obstacle["position"]))
                    margins.append(dist - radius - obstacle["radius"])
                elif obstacle["type"] == "box":
                    local = obstacle["rotation"].T @ (center - obstacle["position"])
                    box_sdf = _signed_distance_to_box(local, obstacle["half_extents"])
                    margins.append(box_sdf - radius)
        if not margins:
            return float("inf")
        return float(np.min(margins))

    def check_state(self, q: Iterable[float]) -> StateValidity:
        self.number_of_validity_checks += 1
        q_arr = np.asarray(q, dtype=float)
        if not self.within_joint_limits(q_arr):
            self.number_of_invalid_states += 1
            return StateValidity(False, float("-inf"), self.number_of_validity_checks, self.number_of_invalid_states)
        margin = self.minimum_safety_margin(q_arr)
        valid = bool(margin > self.collision_threshold)
        if not valid:
            self.number_of_invalid_states += 1
        return StateValidity(valid, margin, self.number_of_validity_checks, self.number_of_invalid_states)

    def is_state_valid(self, q: Iterable[float]) -> bool:
        return self.check_state(q).valid

    def check_motion(
        self,
        q1: Iterable[float],
        q2: Iterable[float],
        resolution: float = 0.05,
    ) -> StateValidity:
        q1_arr = np.asarray(q1, dtype=float)
        q2_arr = np.asarray(q2, dtype=float)
        dist = float(np.linalg.norm(q2_arr - q1_arr))
        steps = max(1, int(math.ceil(dist / max(float(resolution), 1.0e-6))))
        min_margin = float("inf")
        valid = True
        start_checks = self.number_of_validity_checks
        start_invalid = self.number_of_invalid_states
        for idx in range(steps + 1):
            alpha = idx / float(steps)
            q = (1.0 - alpha) * q1_arr + alpha * q2_arr
            state = self.check_state(q)
            min_margin = min(min_margin, state.minimum_safety_margin)
            valid = valid and state.valid
            if not valid:
                break
        return StateValidity(
            valid,
            min_margin,
            self.number_of_validity_checks - start_checks,
            self.number_of_invalid_states - start_invalid,
        )

    def path_metrics(
        self,
        q_path: Iterable[Iterable[float]],
        motion_resolution: float = 0.05,
    ) -> dict:
        q_list = [np.asarray(q, dtype=float) for q in q_path]
        min_margin = float("inf")
        valid = True
        start_checks = self.number_of_validity_checks
        start_invalid = self.number_of_invalid_states
        if not q_list:
            return {
                "valid": False,
                "minimum_safety_margin": float("-inf"),
                "number_of_validity_checks": 0,
                "number_of_invalid_states": 0,
            }
        for q in q_list:
            state = self.check_state(q)
            min_margin = min(min_margin, state.minimum_safety_margin)
            valid = valid and state.valid
        for q1, q2 in zip(q_list[:-1], q_list[1:]):
            motion = self.check_motion(q1, q2, resolution=motion_resolution)
            min_margin = min(min_margin, motion.minimum_safety_margin)
            valid = valid and motion.valid
        return {
            "valid": bool(valid),
            "minimum_safety_margin": float(min_margin),
            "number_of_validity_checks": self.number_of_validity_checks - start_checks,
            "number_of_invalid_states": self.number_of_invalid_states - start_invalid,
        }
