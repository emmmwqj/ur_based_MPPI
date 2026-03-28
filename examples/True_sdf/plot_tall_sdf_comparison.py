#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def log(message: str) -> None:
    print(f"[TrueSDFPlot] {message}", flush=True)


def world_to_idx(origin: np.ndarray, voxel_size: float, value: float, axis_size: int) -> int:
    idx = int(round((value - float(origin)) / voxel_size))
    return int(np.clip(idx, 0, axis_size - 1))


def masked_slice(data: np.ndarray, mask: np.ndarray, slicer):
    sliced = data[slicer]
    sliced_mask = mask[slicer]
    return np.where(sliced_mask, sliced, np.nan)


def compute_summary_stats(true_esdf: np.ndarray, ref_esdf: np.ndarray, compare_mask: np.ndarray) -> dict:
    abs_err = np.abs(true_esdf - ref_esdf)[compare_mask]
    stats = {
        "n_compare": int(abs_err.size),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(abs_err ** 2))),
        "max": float(np.max(abs_err)),
        "p50": float(np.percentile(abs_err, 50)),
        "p90": float(np.percentile(abs_err, 90)),
        "p95": float(np.percentile(abs_err, 95)),
        "p99": float(np.percentile(abs_err, 99)),
        "lt_1cm": float(np.mean(abs_err <= 0.01)),
        "lt_2cm": float(np.mean(abs_err <= 0.02)),
        "lt_5cm": float(np.mean(abs_err <= 0.05)),
    }

    near_surface = np.logical_and(compare_mask, np.logical_and(ref_esdf > 0.0, ref_esdf <= 0.05))
    if np.any(near_surface):
        near_err = np.abs(true_esdf - ref_esdf)[near_surface]
        stats["near_n"] = int(near_err.size)
        stats["near_mae"] = float(np.mean(near_err))
        stats["near_max"] = float(np.max(near_err))
        stats["near_p95"] = float(np.percentile(near_err, 95))
    else:
        stats["near_n"] = 0
        stats["near_mae"] = float("nan")
        stats["near_max"] = float("nan")
        stats["near_p95"] = float("nan")

    return stats


def plot_xy_slices(
    true_esdf: np.ndarray,
    ref_esdf: np.ndarray,
    compare_mask: np.ndarray,
    origin_world: np.ndarray,
    voxel_size: float,
    output_path: str,
) -> None:
    z_values = [0.10, 0.20, 0.40]
    fig, axes = plt.subplots(len(z_values), 3, figsize=(14, 12), constrained_layout=True)
    extent = [
        float(origin_world[0]),
        float(origin_world[0] + voxel_size * (true_esdf.shape[0] - 1)),
        float(origin_world[1]),
        float(origin_world[1] + voxel_size * (true_esdf.shape[1] - 1)),
    ]

    for row, z_world in enumerate(z_values):
        z_idx = world_to_idx(origin_world[2], voxel_size, z_world, true_esdf.shape[2])
        true_slice = masked_slice(true_esdf, compare_mask, (slice(None), slice(None), z_idx)).T
        ref_slice = masked_slice(ref_esdf, compare_mask, (slice(None), slice(None), z_idx)).T
        err_slice = masked_slice(np.abs(true_esdf - ref_esdf), compare_mask, (slice(None), slice(None), z_idx)).T

        im0 = axes[row, 0].imshow(true_slice, origin="lower", extent=extent, vmin=0.0, vmax=0.30, cmap="viridis")
        axes[row, 0].set_title(f"True SDF (z={z_world:.2f} m)")
        im1 = axes[row, 1].imshow(ref_slice, origin="lower", extent=extent, vmin=0.0, vmax=0.30, cmap="viridis")
        axes[row, 1].set_title(f"nvblox ESDF (z={z_world:.2f} m)")
        im2 = axes[row, 2].imshow(err_slice, origin="lower", extent=extent, vmin=0.0, vmax=0.05, cmap="magma")
        axes[row, 2].set_title(f"|error| (z={z_world:.2f} m)")

        for col in range(3):
            axes[row, col].set_xlabel("x [m]")
            axes[row, col].set_ylabel("y [m]")

    cbar0 = fig.colorbar(im0, ax=axes[:, :2], shrink=0.98, location="right")
    cbar0.set_label("distance [m]")
    cbar1 = fig.colorbar(im2, ax=axes[:, 2], shrink=0.98, location="right")
    cbar1.set_label("abs error [m]")
    fig.suptitle("Tall Scene Exterior Comparison: XY Slices", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_yz_slices(
    true_esdf: np.ndarray,
    ref_esdf: np.ndarray,
    compare_mask: np.ndarray,
    origin_world: np.ndarray,
    voxel_size: float,
    output_path: str,
) -> None:
    x_values = [0.25, 0.40, 0.55]
    fig, axes = plt.subplots(len(x_values), 3, figsize=(14, 12), constrained_layout=True)
    extent = [
        float(origin_world[1]),
        float(origin_world[1] + voxel_size * (true_esdf.shape[1] - 1)),
        float(origin_world[2]),
        float(origin_world[2] + voxel_size * (true_esdf.shape[2] - 1)),
    ]

    for row, x_world in enumerate(x_values):
        x_idx = world_to_idx(origin_world[0], voxel_size, x_world, true_esdf.shape[0])
        true_slice = masked_slice(true_esdf, compare_mask, (x_idx, slice(None), slice(None))).T
        ref_slice = masked_slice(ref_esdf, compare_mask, (x_idx, slice(None), slice(None))).T
        err_slice = masked_slice(np.abs(true_esdf - ref_esdf), compare_mask, (x_idx, slice(None), slice(None))).T

        im0 = axes[row, 0].imshow(true_slice, origin="lower", extent=extent, aspect="auto", vmin=0.0, vmax=0.30, cmap="viridis")
        axes[row, 0].set_title(f"True SDF (x={x_world:.2f} m)")
        im1 = axes[row, 1].imshow(ref_slice, origin="lower", extent=extent, aspect="auto", vmin=0.0, vmax=0.30, cmap="viridis")
        axes[row, 1].set_title(f"nvblox ESDF (x={x_world:.2f} m)")
        im2 = axes[row, 2].imshow(err_slice, origin="lower", extent=extent, aspect="auto", vmin=0.0, vmax=0.05, cmap="magma")
        axes[row, 2].set_title(f"|error| (x={x_world:.2f} m)")

        for col in range(3):
            axes[row, col].set_xlabel("y [m]")
            axes[row, col].set_ylabel("z [m]")

    cbar0 = fig.colorbar(im0, ax=axes[:, :2], shrink=0.98, location="right")
    cbar0.set_label("distance [m]")
    cbar1 = fig.colorbar(im2, ax=axes[:, 2], shrink=0.98, location="right")
    cbar1.set_label("abs error [m]")
    fig.suptitle("Tall Scene Exterior Comparison: YZ Slices", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_distribution(
    true_esdf: np.ndarray,
    ref_esdf: np.ndarray,
    compare_mask: np.ndarray,
    output_path: str,
) -> None:
    abs_err = np.abs(true_esdf - ref_esdf)[compare_mask]
    ref_vals = ref_esdf[compare_mask]
    true_vals = true_esdf[compare_mask]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    axes[0].hist(abs_err, bins=100, range=(0.0, 0.08), color="#3b82f6", alpha=0.9)
    axes[0].set_title("Exterior |error| Histogram")
    axes[0].set_xlabel("abs error [m]")
    axes[0].set_ylabel("count")

    sorted_err = np.sort(abs_err)
    cdf = np.linspace(0.0, 1.0, sorted_err.size, endpoint=False)
    axes[1].plot(sorted_err, cdf, color="#ef4444", linewidth=2.0)
    axes[1].axvline(0.01, color="gray", linestyle="--", linewidth=1.0)
    axes[1].axvline(0.02, color="gray", linestyle="--", linewidth=1.0)
    axes[1].axvline(0.05, color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_xlim(0.0, 0.08)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Exterior |error| CDF")
    axes[1].set_xlabel("abs error [m]")
    axes[1].set_ylabel("fraction")

    if true_vals.size > 30000:
        rng = np.random.default_rng(0)
        idx = rng.choice(true_vals.size, size=30000, replace=False)
        true_plot = true_vals[idx]
        ref_plot = ref_vals[idx]
    else:
        true_plot = true_vals
        ref_plot = ref_vals

    axes[2].scatter(ref_plot, true_plot, s=2, alpha=0.18, color="#10b981", edgecolors="none")
    lim_max = min(0.35, float(max(np.max(ref_plot), np.max(true_plot))))
    axes[2].plot([0.0, lim_max], [0.0, lim_max], color="black", linestyle="--", linewidth=1.0)
    axes[2].set_xlim(0.0, lim_max)
    axes[2].set_ylim(0.0, lim_max)
    axes[2].set_title("Exterior SDF Scatter")
    axes[2].set_xlabel("nvblox ESDF [m]")
    axes[2].set_ylabel("True SDF [m]")

    fig.suptitle("Tall Scene Exterior Error Distribution", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(report_path: str, stats: dict, image_paths: dict) -> None:
    lines = [
        "# Tall SDF Comparison Report",
        "",
        "比较范围：只统计障碍物外部体素，即 `true_sdf esdf > 0` 且 `reference valid_mask == True`。",
        "",
        "## Summary",
        f"- exterior voxels compared: {stats['n_compare']}",
        f"- MAE: {stats['mae']:.4f} m",
        f"- RMSE: {stats['rmse']:.4f} m",
        f"- Max error: {stats['max']:.4f} m",
        f"- P50 / P90 / P95 / P99: {stats['p50']:.4f} / {stats['p90']:.4f} / {stats['p95']:.4f} / {stats['p99']:.4f} m",
        f"- fraction <= 1 cm: {100.0 * stats['lt_1cm']:.2f}%",
        f"- fraction <= 2 cm: {100.0 * stats['lt_2cm']:.2f}%",
        f"- fraction <= 5 cm: {100.0 * stats['lt_5cm']:.2f}%",
        "",
        "## Near Surface Exterior",
        f"- voxels: {stats['near_n']}",
        f"- MAE: {stats['near_mae']:.4f} m",
        f"- P95: {stats['near_p95']:.4f} m",
        f"- Max error: {stats['near_max']:.4f} m",
        "",
        "## Figures",
        f"- XY slices: `{image_paths['xy']}`",
        f"- YZ slices: `{image_paths['yz']}`",
        f"- Error distribution: `{image_paths['dist']}`",
        "",
        "## Reading Guide",
        "- XY 切片图主要看墙和平面球体附近的误差是否局部偏大。",
        "- YZ 切片图主要看高墙在高度方向上的边界是否一致。",
        "- 分布图看整体误差集中在哪个量级，以及近表面误差是否仍可接受。",
    ]
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot tall-scene SDF comparison charts")
    parser.add_argument(
        "--true-sdf",
        default="/home/wqj/storm/examples/True_sdf/results/tall_true_sdf_snapshot.npz",
        help="True SDF npz generated from STORM primitive world",
    )
    parser.add_argument(
        "--reference-snapshot",
        default="/home/wqj/perception_D435i/src/sim_nvblox/result/tall_esdf_snapshot.npz",
        help="Reference nvblox snapshot npz",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/wqj/storm/examples/True_sdf/results",
        help="Directory to save figures/report",
    )
    args = parser.parse_args()

    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    true_data = np.load(os.path.expanduser(args.true_sdf))
    ref_data = np.load(os.path.expanduser(args.reference_snapshot))

    true_esdf = np.asarray(true_data["esdf"], dtype=np.float32)
    ref_esdf = np.asarray(ref_data["esdf"], dtype=np.float32)
    ref_valid = np.asarray(ref_data["valid_mask"], dtype=np.uint8).astype(bool)
    origin_world = np.asarray(true_data["origin_world"], dtype=np.float32)
    voxel_size = float(np.asarray(true_data["voxel_size"], dtype=np.float32).item())

    compare_mask = np.logical_and(ref_valid, true_esdf > 0.0)
    stats = compute_summary_stats(true_esdf, ref_esdf, compare_mask)
    for key, value in stats.items():
        log(f"{key}={value}")

    xy_path = str(output_dir / "tall_sdf_xy_slices.png")
    yz_path = str(output_dir / "tall_sdf_yz_slices.png")
    dist_path = str(output_dir / "tall_sdf_error_distribution.png")
    report_path = str(output_dir / "tall_sdf_comparison_report.md")

    plot_xy_slices(true_esdf, ref_esdf, compare_mask, origin_world, voxel_size, xy_path)
    plot_yz_slices(true_esdf, ref_esdf, compare_mask, origin_world, voxel_size, yz_path)
    plot_distribution(true_esdf, ref_esdf, compare_mask, dist_path)
    write_report(
        report_path,
        stats,
        {"xy": xy_path, "yz": yz_path, "dist": dist_path},
    )

    log(f"saved {xy_path}")
    log(f"saved {yz_path}")
    log(f"saved {dist_path}")
    log(f"saved {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
