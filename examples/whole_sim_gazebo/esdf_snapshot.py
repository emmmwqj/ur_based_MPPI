import os
from typing import Tuple

import numpy as np
import torch


def _log(message: str) -> None:
    print(f"[ESDFSnapshot] {message}", flush=True)


class ESDFSnapshot:
    """Loads a static ESDF snapshot exported from nvblox and supports point queries."""

    def __init__(
        self,
        npz_path: str,
        tensor_args=None,
        interpolation: str = "trilinear",
        invalid_esdf_value: float | None = None,
        require_valid_neighbor: bool = True,
    ) -> None:
        if tensor_args is None:
            tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}

        self.tensor_args = tensor_args
        self.device = tensor_args["device"]
        self.dtype = tensor_args["dtype"]
        self.npz_path = os.path.expanduser(npz_path)
        self.interpolation = interpolation
        self.require_valid_neighbor = require_valid_neighbor
        self.invalid_esdf_value = invalid_esdf_value

        self.origin_world = None
        self.voxel_size = None
        self.dims = None
        self.esdf = None
        self.valid_mask = None
        self.unknown_distance = None
        self.bounds_min = None
        self.bounds_max = None

        self._has_logged_query_success = False
        self._has_logged_query_failure = False
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.npz_path):
            raise FileNotFoundError(f"ESDF snapshot not found: {self.npz_path}")

        data = np.load(self.npz_path)
        required_keys = ["origin_world", "voxel_size", "dims", "esdf", "valid_mask"]
        missing_keys = [key for key in required_keys if key not in data.files]
        if missing_keys:
            raise KeyError(f"ESDF snapshot missing keys: {missing_keys}")

        dims = np.asarray(data["dims"], dtype=np.int64)
        esdf = np.asarray(data["esdf"], dtype=np.float32)
        valid_mask = np.asarray(data["valid_mask"], dtype=np.uint8).astype(np.bool_)

        if tuple(esdf.shape) != tuple(dims.tolist()):
            raise ValueError(
                "ESDF shape mismatch: dims=%s esdf.shape=%s"
                % (tuple(dims.tolist()), tuple(esdf.shape))
            )
        if tuple(valid_mask.shape) != tuple(dims.tolist()):
            raise ValueError(
                "valid_mask shape mismatch: dims=%s valid_mask.shape=%s"
                % (tuple(dims.tolist()), tuple(valid_mask.shape))
            )

        self.origin_world = torch.as_tensor(data["origin_world"], **self.tensor_args)
        self.voxel_size = float(np.asarray(data["voxel_size"], dtype=np.float32).item())
        self.dims = tuple(int(v) for v in dims.tolist())
        self.esdf = torch.as_tensor(esdf, **self.tensor_args)
        self.valid_mask = torch.as_tensor(valid_mask, device=self.device, dtype=torch.bool)

        if "unknown_distance" in data.files:
            unknown_distance = float(np.asarray(data["unknown_distance"], dtype=np.float32).item())
            if np.isfinite(unknown_distance):
                self.unknown_distance = unknown_distance
            else:
                self.unknown_distance = None
        else:
            self.unknown_distance = None

        if self.invalid_esdf_value is None:
            if self.unknown_distance is not None:
                self.invalid_esdf_value = self.unknown_distance
            else:
                self.invalid_esdf_value = 100.0

        dims_tensor = torch.tensor(self.dims, device=self.device, dtype=self.dtype)
        self.bounds_min = self.origin_world.clone()
        self.bounds_max = self.origin_world + (dims_tensor - 1.0) * self.voxel_size

        valid_voxels = int(self.valid_mask.count_nonzero().item())
        total_voxels = int(self.valid_mask.numel())
        valid_ratio = 100.0 * valid_voxels / total_voxels if total_voxels else 0.0
        _log(f"Snapshot load success: {self.npz_path}")
        _log(
            "dims=%s voxel_size=%.4f valid=%d/%d (%.2f%%)" % (
                self.dims,
                self.voxel_size,
                valid_voxels,
                total_voxels,
                valid_ratio,
            )
        )
        _log(
            "bounds_min=%s bounds_max=%s interpolation=%s"
            % (
                [round(float(x), 4) for x in self.bounds_min.detach().cpu().tolist()],
                [round(float(x), 4) for x in self.bounds_max.detach().cpu().tolist()],
                self.interpolation,
            )
        )

    def query(self, points_world: torch.Tensor | np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        points = torch.as_tensor(points_world, device=self.device, dtype=self.dtype)
        if points.shape[-1] != 3:
            raise ValueError(f"points_world last dimension must be 3, got {tuple(points.shape)}")

        original_shape = points.shape[:-1]
        points = points.reshape(-1, 3)
        if points.numel() == 0:
            empty_dist = torch.empty(original_shape, device=self.device, dtype=self.dtype)
            empty_valid = torch.empty(original_shape, device=self.device, dtype=torch.bool)
            return empty_dist, empty_valid

        coords = (points - self.origin_world.unsqueeze(0)) / self.voxel_size

        if self.interpolation == "nearest":
            distances, valid = self._query_nearest(coords)
        elif self.interpolation == "trilinear":
            distances, valid = self._query_trilinear(coords)
        else:
            raise ValueError(f"Unsupported interpolation mode: {self.interpolation}")

        if valid.any() and not self._has_logged_query_success:
            self._has_logged_query_success = True
            _log(
                "ESDF query success: method=%s points=%d valid=%d"
                % (self.interpolation, int(valid.numel()), int(valid.count_nonzero().item()))
            )
        elif not valid.any() and not self._has_logged_query_failure:
            self._has_logged_query_failure = True
            _log(
                "ESDF query warning: method=%s points=%d valid=0"
                % (self.interpolation, int(valid.numel()))
            )

        return distances.view(original_shape), valid.view(original_shape)

    def _query_nearest(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dims_tensor = torch.tensor(self.dims, device=self.device, dtype=self.dtype)
        in_bounds = torch.logical_and(coords >= 0.0, coords <= (dims_tensor - 1.0)).all(dim=-1)
        nearest = torch.round(coords).to(dtype=torch.long)
        nearest = torch.minimum(torch.maximum(nearest, torch.zeros_like(nearest)), (dims_tensor.long() - 1))

        distances = torch.full(
            (coords.shape[0],),
            float(self.invalid_esdf_value),
            device=self.device,
            dtype=self.dtype,
        )
        valid = torch.zeros((coords.shape[0],), device=self.device, dtype=torch.bool)
        if in_bounds.any():
            ix, iy, iz = nearest[in_bounds, 0], nearest[in_bounds, 1], nearest[in_bounds, 2]
            nearest_valid = self.valid_mask[ix, iy, iz]
            if not self.require_valid_neighbor:
                nearest_valid = torch.ones_like(nearest_valid, dtype=torch.bool)
            values = self.esdf[ix, iy, iz]
            distances[in_bounds] = torch.where(
                nearest_valid,
                values,
                torch.full_like(values, float(self.invalid_esdf_value)),
            )
            valid[in_bounds] = nearest_valid
        return distances, valid

    def _query_trilinear(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dims_tensor = torch.tensor(self.dims, device=self.device, dtype=self.dtype)
        max_coord = dims_tensor - 1.0
        in_bounds = torch.logical_and(coords >= 0.0, coords <= max_coord).all(dim=-1)

        base = torch.floor(coords).to(dtype=torch.long)
        base = torch.minimum(torch.maximum(base, torch.zeros_like(base)), dims_tensor.long() - 2)
        frac = (coords - base.to(dtype=self.dtype)).clamp(0.0, 1.0)
        upper = base + 1

        x0, y0, z0 = base[:, 0], base[:, 1], base[:, 2]
        x1, y1, z1 = upper[:, 0], upper[:, 1], upper[:, 2]
        tx, ty, tz = frac[:, 0], frac[:, 1], frac[:, 2]

        def gather(ix, iy, iz):
            return self.esdf[ix, iy, iz], self.valid_mask[ix, iy, iz]

        c000, v000 = gather(x0, y0, z0)
        c100, v100 = gather(x1, y0, z0)
        c010, v010 = gather(x0, y1, z0)
        c110, v110 = gather(x1, y1, z0)
        c001, v001 = gather(x0, y0, z1)
        c101, v101 = gather(x1, y0, z1)
        c011, v011 = gather(x0, y1, z1)
        c111, v111 = gather(x1, y1, z1)

        weights = torch.stack(
            [
                (1.0 - tx) * (1.0 - ty) * (1.0 - tz),
                tx * (1.0 - ty) * (1.0 - tz),
                (1.0 - tx) * ty * (1.0 - tz),
                tx * ty * (1.0 - tz),
                (1.0 - tx) * (1.0 - ty) * tz,
                tx * (1.0 - ty) * tz,
                (1.0 - tx) * ty * tz,
                tx * ty * tz,
            ],
            dim=1,
        )
        values = torch.stack([c000, c100, c010, c110, c001, c101, c011, c111], dim=1)
        valid = torch.stack([v000, v100, v010, v110, v001, v101, v011, v111], dim=1)

        if self.require_valid_neighbor:
            effective_weights = torch.where(valid, weights, torch.zeros_like(weights))
            weight_sum = effective_weights.sum(dim=1)
            query_valid = torch.logical_and(in_bounds, weight_sum > 0.0)
            weighted_sum = (values * effective_weights).sum(dim=1)
            interp = torch.full(
                (coords.shape[0],),
                float(self.invalid_esdf_value),
                device=self.device,
                dtype=self.dtype,
            )
            interp[query_valid] = weighted_sum[query_valid] / weight_sum[query_valid]
        else:
            query_valid = in_bounds
            interp = (values * weights).sum(dim=1)
            interp[~query_valid] = float(self.invalid_esdf_value)

        return interp, query_valid
