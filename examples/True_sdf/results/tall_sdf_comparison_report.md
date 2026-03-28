# Tall SDF Comparison Report

比较范围：只统计障碍物外部体素，即 `true_sdf esdf > 0` 且 `reference valid_mask == True`。

## Summary
- exterior voxels compared: 163255
- MAE: 0.0154 m
- RMSE: 0.0281 m
- Max error: 1.9500 m
- P50 / P90 / P95 / P99: 0.0100 / 0.0382 / 0.0752 / 0.1006 m
- fraction <= 1 cm: 57.10%
- fraction <= 2 cm: 84.44%
- fraction <= 5 cm: 91.96%

## Near Surface Exterior
- voxels: 13212
- MAE: 0.0090 m
- P95: 0.0147 m
- Max error: 0.0390 m

## Figures
- XY slices: `/home/wqj/storm/examples/True_sdf/results/tall_sdf_xy_slices.png`
- YZ slices: `/home/wqj/storm/examples/True_sdf/results/tall_sdf_yz_slices.png`
- Error distribution: `/home/wqj/storm/examples/True_sdf/results/tall_sdf_error_distribution.png`

## Reading Guide
- XY 切片图主要看墙和平面球体附近的误差是否局部偏大。
- YZ 切片图主要看高墙在高度方向上的边界是否一致。
- 分布图看整体误差集中在哪个量级，以及近表面误差是否仍可接受。