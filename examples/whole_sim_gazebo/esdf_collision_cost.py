import copy

import torch
import torch.nn as nn

from storm_kit.geom.sdf.robot import RobotSphereCollision
from storm_kit.mpc.cost.gaussian_projection import GaussianProjection
from storm_kit.util_file import get_assets_path, join_path

from examples.whole_sim_gazebo.esdf_snapshot import ESDFSnapshot


def _log(message: str) -> None:
    print(f"[ESDFCollisionCost] {message}", flush=True)


def _format_vec(values) -> list[float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    return [round(float(v), 4) for v in values]


class ESDFCollisionCost(nn.Module):
    """Environment collision cost driven by a static ESDF snapshot."""

    def __init__(
        self,
        weight=None,
        world_params=None,
        robot_params=None,
        gaussian_params=None,
        distance_threshold=0.05,
        clamp_max=0.2,
        cost_scale=0.25,
        tensor_args=None,
    ):
        super().__init__()
        if tensor_args is None:
            tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}
        if gaussian_params is None:
            gaussian_params = {}

        self.tensor_args = tensor_args
        self.device = tensor_args["device"]
        self.dtype = tensor_args["dtype"]
        self.weight = torch.as_tensor(weight, **self.tensor_args)
        self.distance_threshold = float(distance_threshold)
        self.clamp_max = float(clamp_max)
        self.cost_scale = float(cost_scale)
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
        self.batch_size = -1
        self.last_valid_ratio = 0.0
        self._has_logged_forward = False
        self.snapshot_path = None

        robot_collision_params = copy.deepcopy(robot_params["robot_collision_params"])
        robot_collision_params["urdf"] = join_path(get_assets_path(), robot_collision_params["urdf"])
        self.robot_coll = RobotSphereCollision(robot_collision_params, batch_size=1, tensor_args=self.tensor_args)
        self.robot_coll.build_batch_features(batch_size=1, clone_pose=True, clone_objs=True)

        world_model = world_params["world_model"]
        self.snapshot_path = world_model["esdf_snapshot_path"]
        self.snapshot = ESDFSnapshot(
            npz_path=self.snapshot_path,
            tensor_args=self.tensor_args,
            interpolation=world_model.get("interpolation", "trilinear"),
            invalid_esdf_value=world_model.get("invalid_esdf_value", None),
            require_valid_neighbor=bool(world_model.get("require_valid_neighbor", True)),
        )
        self.query_frame_translation = torch.as_tensor(
            world_model.get("query_frame_translation_world", [0.0, 0.0, 0.0]),
            **self.tensor_args,
        )
        _log("Environment collision source: ESDF snapshot")
        _log(
            "Using ESDF collision: snapshot_path=%s threshold=%.3f clamp_max=%.3f"
            % (
                self.snapshot_path,
                self.distance_threshold,
                self.clamp_max,
            )
        )
        _log(
            "snapshot bounds_min=%s bounds_max=%s"
            % (
                _format_vec(self.snapshot.bounds_min),
                _format_vec(self.snapshot.bounds_max),
            )
        )
        _log(
            "query_frame_translation_world=%s"
            % (
                _format_vec(self.query_frame_translation),
            )
        )

    def forward(self, link_pos_seq, link_rot_seq):
        inp_device = link_pos_seq.device
        batch_size = link_pos_seq.shape[0]
        horizon = link_pos_seq.shape[1]
        n_links = link_pos_seq.shape[2]
        batch_horizon = batch_size * horizon

        if self.batch_size != batch_horizon:
            self.batch_size = batch_horizon
            self.robot_coll.build_batch_features(
                batch_size=batch_horizon,
                clone_pose=True,
                clone_objs=True,
            )

        link_pos_batch = link_pos_seq.view(batch_horizon, n_links, 3)
        link_rot_batch = link_rot_seq.view(batch_horizon, n_links, 3, 3)
        self.robot_coll.update_batch_robot_collision_objs(link_pos_batch, link_rot_batch)
        w_link_spheres = self.robot_coll.get_batch_robot_link_spheres()

        link_penetration = torch.zeros((batch_horizon, n_links), **self.tensor_args)
        total_queries = 0
        total_valid = 0

        for link_idx, spheres in enumerate(w_link_spheres):
            centers = spheres[:, :, :3].reshape(-1, 3)
            radii = spheres[:, :, 3].reshape(-1)
            esdf_values, valid_mask = self.snapshot.query(centers + self.query_frame_translation.unsqueeze(0))
            esdf_values = esdf_values.view(batch_horizon, -1)
            valid_mask = valid_mask.view(batch_horizon, -1)
            radii = radii.view(batch_horizon, -1)

            # nvblox ESDF: free space positive, obstacle interior negative.
            penetration = radii - esdf_values
            link_penetration[:, link_idx] = torch.max(penetration, dim=-1)[0]

            total_queries += int(valid_mask.numel())
            total_valid += int(valid_mask.count_nonzero().item())

        self.last_valid_ratio = (float(total_valid) / float(total_queries)) if total_queries else 0.0
        if total_queries and not self._has_logged_forward:
            self._has_logged_forward = True
            _log(
                "ESDF collision active: queried_spheres=%d valid=%d esdf_valid_ratio=%.2f%%"
                % (total_queries, total_valid, 100.0 * self.last_valid_ratio)
            )

        link_penetration = link_penetration.view(batch_size, horizon, n_links)
        link_penetration = link_penetration + self.distance_threshold
        link_penetration = torch.clamp(link_penetration, min=0.0, max=self.clamp_max)
        link_penetration = link_penetration / self.cost_scale

        cost = torch.sum(link_penetration, dim=-1)
        cost = self.weight * cost
        return cost.to(inp_device)
