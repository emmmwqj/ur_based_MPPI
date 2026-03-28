#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

STORM_ROOT = os.path.expanduser("~/storm")
if STORM_ROOT not in sys.path:
    sys.path.insert(0, STORM_ROOT)

from storm_kit.geom.sdf.world import WorldPrimitiveCollision
from storm_kit.geom.sdf.primitives import get_pt_primitive_distance


def log(message: str) -> None:
    print(f"[TrueSDF] {message}", flush=True)


def load_reference_grid(snapshot_path: str) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    data = np.load(snapshot_path)
    required = ["origin_world", "voxel_size", "dims", "valid_mask"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"Reference snapshot missing keys: {missing}")

    origin_world = np.asarray(data["origin_world"], dtype=np.float32)
    voxel_size = float(np.asarray(data["voxel_size"], dtype=np.float32).item())
    dims = np.asarray(data["dims"], dtype=np.int32)
    valid_mask = np.asarray(data["valid_mask"], dtype=np.uint8)
    return origin_world, voxel_size, dims, valid_mask


def build_world_points(origin_world: np.ndarray, voxel_size: float, dims: np.ndarray) -> np.ndarray:
    grid = np.stack(
        np.meshgrid(
            np.arange(dims[0], dtype=np.float32),
            np.arange(dims[1], dtype=np.float32),
            np.arange(dims[2], dtype=np.float32),
            indexing="ij",
        ),
        axis=-1,
    )
    points_world = origin_world.reshape(1, 1, 1, 3) + grid * voxel_size
    return points_world.reshape(-1, 3)


def compute_storm_signed_distance(
    world_params: dict,
    points_world: np.ndarray,
    batch_size: int = 65536,
) -> np.ndarray:
    tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}
    # WorldPrimitiveCollision requires bounds in __init__. We only use its
    # analytic primitive-distance query here, so keep the bootstrap grid tiny.
    dummy_bounds = [[0.0, 0.0, 0.0], [0.05, 0.05, 0.05]]
    world = WorldPrimitiveCollision(
        world_params["world_model"],
        batch_size=1,
        tensor_args=tensor_args,
        bounds=dummy_bounds,
        grid_resolution=0.05,
    )

    sdf_chunks = []
    n_points = points_world.shape[0]
    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        pts = torch.as_tensor(points_world[start:end], **tensor_args).view(1, -1, 3)
        dist = torch.zeros((1, world.n_objs, pts.shape[1]), **tensor_args)
        dist = get_pt_primitive_distance(pts, world._world_spheres, world._world_cubes, dist)
        sdf = torch.max(dist, dim=1)[0].reshape(-1).detach().cpu().numpy().astype(np.float32)
        sdf_chunks.append(sdf)
        if start == 0 or end == n_points or ((start // batch_size) + 1) % 5 == 0:
            log(f"computed {end}/{n_points} grid points")

    return np.concatenate(sdf_chunks, axis=0)


def compare_against_reference(
    output_esdf: np.ndarray,
    reference_snapshot_path: str,
) -> dict:
    data = np.load(reference_snapshot_path)
    reference_esdf = np.asarray(data["esdf"], dtype=np.float32)
    reference_valid = np.asarray(data["valid_mask"], dtype=np.uint8).astype(bool)

    diff = output_esdf - reference_esdf
    exterior_mask = output_esdf > 0.0
    compare_mask = np.logical_and(reference_valid, exterior_mask)

    metrics = {
        "reference_valid_ratio": float(reference_valid.mean()),
        "compare_exterior_ratio": float(compare_mask.mean()),
        "compare_exterior_voxels": int(np.count_nonzero(compare_mask)),
    }

    if np.any(compare_mask):
        metrics["mean_abs_error_on_compare_exterior"] = float(np.mean(np.abs(diff[compare_mask])))
        metrics["max_abs_error_on_compare_exterior"] = float(np.max(np.abs(diff[compare_mask])))
    else:
        metrics["mean_abs_error_on_compare_exterior"] = float("nan")
        metrics["max_abs_error_on_compare_exterior"] = float("nan")

    near_surface_exterior = np.logical_and(compare_mask, np.logical_and(reference_esdf > 0.0, reference_esdf <= 0.05))
    if np.any(near_surface_exterior):
        metrics["mean_abs_error_near_surface_exterior"] = float(np.mean(np.abs(diff[near_surface_exterior])))
        metrics["max_abs_error_near_surface_exterior"] = float(np.max(np.abs(diff[near_surface_exterior])))
        metrics["near_surface_exterior_voxels"] = int(np.count_nonzero(near_surface_exterior))
    else:
        metrics["mean_abs_error_near_surface_exterior"] = float("nan")
        metrics["max_abs_error_near_surface_exterior"] = float("nan")
        metrics["near_surface_exterior_voxels"] = 0

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute tall-scene primitive SDF using STORM and save as .npz")
    parser.add_argument(
        "--world-file",
        default="/home/wqj/storm/examples/sim_gazebo/config/collision_world_gazebo_tall.yml",
        help="Primitive world config used by STORM",
    )
    parser.add_argument(
        "--reference-snapshot",
        default="/home/wqj/perception_D435i/src/sim_nvblox/result/tall_esdf_snapshot.npz",
        help="Reference snapshot whose origin_world/voxel_size/dims are reused",
    )
    parser.add_argument(
        "--output",
        default="/home/wqj/storm/examples/True_sdf/results/tall_true_sdf_snapshot.npz",
        help="Output .npz path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Number of query points per batch",
    )
    args = parser.parse_args()

    world_file = os.path.expanduser(args.world_file)
    reference_snapshot = os.path.expanduser(args.reference_snapshot)
    output_path = os.path.expanduser(args.output)

    log(f"world_file={world_file}")
    log(f"reference_snapshot={reference_snapshot}")
    log(f"output={output_path}")

    with open(world_file) as f:
        world_params = yaml.safe_load(f)

    origin_world, voxel_size, dims, reference_valid_mask = load_reference_grid(reference_snapshot)
    log(
        "grid origin=%s voxel_size=%.5f dims=%s reference_valid_ratio=%.2f%%"
        % (
            np.round(origin_world, 4).tolist(),
            voxel_size,
            dims.tolist(),
            100.0 * float(reference_valid_mask.astype(bool).mean()),
        )
    )

    points_world = build_world_points(origin_world, voxel_size, dims)
    storm_signed_distance = compute_storm_signed_distance(
        world_params=world_params,
        points_world=points_world,
        batch_size=args.batch_size,
    ).reshape(tuple(int(v) for v in dims.tolist()))

    # STORM primitive world uses positive-inside / negative-outside for spheres,
    # and cube interior saturates to 0.0. For easier direct comparison with
    # nvblox ESDF, save a flipped version where free space is positive.
    esdf_compare = (-storm_signed_distance).astype(np.float32)
    valid_mask = np.ones(tuple(int(v) for v in dims.tolist()), dtype=np.uint8)

    metrics = compare_against_reference(esdf_compare, reference_snapshot)
    for key, value in metrics.items():
        log(f"{key}={value}")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        origin_world=origin_world.astype(np.float32),
        voxel_size=np.float32(voxel_size),
        dims=dims.astype(np.int32),
        esdf=esdf_compare.astype(np.float32),
        valid_mask=valid_mask,
        storm_signed_distance=storm_signed_distance.astype(np.float32),
        source_world_file=np.asarray(world_file),
        reference_snapshot=np.asarray(reference_snapshot),
        note=np.asarray(
            "esdf uses nvblox-style sign (free positive, obstacle negative); "
            "storm_signed_distance preserves STORM primitive raw sign; "
            "comparison metrics are computed on exterior voxels only."
        ),
        reference_valid_ratio=np.float32(metrics["reference_valid_ratio"]),
        compare_exterior_ratio=np.float32(metrics["compare_exterior_ratio"]),
        compare_exterior_voxels=np.int32(metrics["compare_exterior_voxels"]),
        mean_abs_error_on_compare_exterior=np.float32(metrics["mean_abs_error_on_compare_exterior"]),
        max_abs_error_on_compare_exterior=np.float32(metrics["max_abs_error_on_compare_exterior"]),
        mean_abs_error_near_surface_exterior=np.float32(metrics["mean_abs_error_near_surface_exterior"]),
        max_abs_error_near_surface_exterior=np.float32(metrics["max_abs_error_near_surface_exterior"]),
        near_surface_exterior_voxels=np.int32(metrics["near_surface_exterior_voxels"]),
    )

    log(f"saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
