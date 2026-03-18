# whole_sim_gazebo

一个独立于 `examples/sim_gazebo` 的 UR7e Gazebo + STORM MPPI 示例。

差异点只有一个核心点：
- `sim_gazebo` 的环境碰撞来自 `collision_world_gazebo.yml` 中的 primitive world
- `whole_sim_gazebo` 的环境碰撞来自 nvblox 导出的静态 ESDF snapshot

使用的 snapshot:
- `/home/wqj/perception_D435i/src/sim_nvblox/result/latest_esdf_snapshot.npz`

Tall 场景使用的 snapshot:
- `/home/wqj/perception_D435i/src/sim_nvblox/result/tall_esdf_snapshot.npz`

Diffusion 控制入口默认使用 Tall 场景:
- `config/ur7e_reacher_whole_gazebo_diffusion_tall.yml`
- `config/esdf_world_gazebo_tall.yml`

## 目录说明
- `ur7e_mpc_whole_gazebo.py`: 主控制脚本
- `ur7e_mpc_whole_gazebo_diffusion.py`: Diffusion MPPI 控制脚本
- `esdf_snapshot.py`: ESDF snapshot 读取与查询
- `esdf_collision_cost.py`: 机器人 collision spheres 到 ESDF 的环境碰撞代价
- `arm_base_esdf.py` / `arm_reacher_esdf.py`: 本地 rollout
- `whole_gazebo_diffusion_task.py`: DiffusionMPPI + ArmReacherESDF 任务封装
- `run_rviz.sh`: 单独启动 whole_sim_gazebo 的 RViz 配置
- `inspect_esdf_snapshot.py`: 不依赖 ROS2 的 snapshot 查询自检
- `config/`: 独立配置文件

## 运行
终端 1:
```bash
cd ~/storm/examples/whole_sim_gazebo
./run_gazebo.sh
```

终端 2:
```bash
cd ~/storm/examples/whole_sim_gazebo
./run_whole_mpc.sh
```

Diffusion MPPI 版本:

终端 1:
```bash
cd ~/storm/examples/whole_sim_gazebo
./run_gazebo.sh
```

终端 2:
```bash
cd ~/storm/examples/whole_sim_gazebo
./run_whole_mpc_diffusion.sh
```

Tall 场景:

终端 1:
```bash
cd ~/storm/examples/whole_sim_gazebo
./run_gazebo.sh
```

终端 2:
```bash
cd ~/storm/examples/whole_sim_gazebo
./run_whole_mpc_tall.sh
```

运行 `./run_whole_mpc.sh` 时会自动打开 RViz。
如果只想运行控制器、不自动打开 RViz：
```bash
./run_whole_mpc.sh --no-rviz
```

如果只想运行 Diffusion 控制器、不自动打开 RViz：
```bash
./run_whole_mpc_diffusion.sh --no-rviz
```

只检查 ESDF 读取与查询:
```bash
cd ~/storm/examples/whole_sim_gazebo
python3 inspect_esdf_snapshot.py
```

## 如何确认它用的是 ESDF 而不是 primitive collision
看启动日志，应该出现这些关键信息：
- `[ESDFSnapshot] Snapshot load success: ...latest_esdf_snapshot.npz`
- `[ESDFSnapshot] dims=... voxel_size=... valid=...`
- `[ESDFCollisionCost] Environment collision source: ESDF snapshot`
- `[ESDFCollisionCost] Using ESDF collision: snapshot_path=...latest_esdf_snapshot.npz`
- `[ESDFCollisionCost] snapshot bounds_min=... bounds_max=...`
- `[WholeGazeboReacherTask] primitive_collision.weight = 0.0`
- `[WholeGazeboReacherTask] voxel_collision.weight     = 0.0`
- `[WholeGazeboReacherTask] esdf_collision.weight      = 5000.0`
- `[WholeGazeboReacherTask] environment_collision      = ESDF snapshot`
- `[ESDFCollisionCost] ESDF collision active: queried_spheres=... esdf_valid_ratio=...`
- `RViz scene markers prepared: points=... voxel_size=...`
- `... | esdf_valid_ratio=...%`

Diffusion 入口额外会打印：
- `[WholeGazeboDiffusionReacherTask] controller_type = DiffusionMPPI`
- `[WholeGazeboDiffusionReacherTask] diffusion.beta_1 / beta_2 / sigma_base / n_diffuse / n_diffuse_init`
- `... | diff_sigma=... | diff_best_cost=...`
- `[ESDFSnapshot] Snapshot load success: ...tall_esdf_snapshot.npz`

Tall 场景看启动日志，应该出现：
- `[ESDFSnapshot] Snapshot load success: ...tall_esdf_snapshot.npz`
- `World: ...config/esdf_world_gazebo_tall.yml`
- `[WholeGazeboReacherTask] esdf_snapshot_path         = /home/wqj/perception_D435i/src/sim_nvblox/result/tall_esdf_snapshot.npz`

RViz 中的可视化来源：
- 机械臂: `RobotModel` 读取 `/robot_description`
- 场景: 从 ESDF snapshot 中提取近表面体素，作为一个 `CUBE_LIST` marker 发布到 `/visualization_marker_array`
- 目标: 红色球 marker
- 末端: 绿色球 marker

配置层面也能验证：
- `config/ur7e_reacher_whole_gazebo.yml` 中 `primitive_collision.weight = 0.0`
- `config/ur7e_reacher_whole_gazebo.yml` 中 `voxel_collision.weight = 0.0`
- `config/ur7e_reacher_whole_gazebo.yml` 中 `esdf_collision.weight > 0`
- `config/esdf_world_gazebo.yml` 中 `source: esdf_snapshot`

如果运行后 `esdf_valid_ratio` 长时间保持 `0.0%`：
- 先确认 snapshot 本身 `valid` 不是 `0`
- 再检查日志里的 snapshot `bounds_min / bounds_max`
- 当前示例默认直接在 world frame 下查询，`query_frame_translation_world` 保持 `[0.0, 0.0, 0.0]`
