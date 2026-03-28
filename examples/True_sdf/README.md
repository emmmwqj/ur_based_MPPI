# True_sdf

这个目录用于用 STORM 自己的 primitive world 实现，给高墙场景离线计算一份可对比的 SDF/ESDF 网格。

默认行为：
- 场景几何来自 `examples/sim_gazebo/config/collision_world_gazebo_tall.yml`
- 网格参数直接复用：
  - `/home/wqj/perception_D435i/src/sim_nvblox/result/tall_esdf_snapshot.npz`
- 输出保存到：
  - `examples/True_sdf/results/tall_true_sdf_snapshot.npz`

运行：

```bash
cd ~/storm/examples/True_sdf
python3 compute_tall_true_sdf.py
```

生成图表：

```bash
cd ~/storm/examples/True_sdf
python3 plot_tall_sdf_comparison.py
```

输出 `.npz` 里的主要字段：
- `origin_world`
- `voxel_size`
- `dims`
- `esdf`
- `valid_mask`
- `storm_signed_distance`

说明：
- `esdf` 已按 nvblox 的符号方向保存，方便直接和 `tall_esdf_snapshot.npz` 对比
- `storm_signed_distance` 保留了 STORM primitive 原始符号
- STORM 的 primitive cube 在盒子内部会返回 `0`，不是严格的欧氏 signed distance；比较时要注意这一点
- 当前脚本的误差统计只在障碍物外部进行，也就是只比较 `true_sdf esdf > 0` 的体素
- 图表和报告会保存在 `examples/True_sdf/results/`
