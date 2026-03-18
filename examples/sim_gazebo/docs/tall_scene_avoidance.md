# sim_gazebo 高墙场景避障分析

## 原 sim_gazebo 的避障是怎么实现的
1. `reach_static_ur7e.py` / `ur7e_mpc_gazebo.py` 在启动时读取 `collision_world_gazebo.yml`。
2. `GazeboReacherTask` 把 `world_file` 传给 `ArmReacher` rollout。
3. `ur7e_reacher_gazebo.yml` 中打开了：
   - `primitive_collision.weight = 5000.0`
   - `robot_self_collision.weight = 5000.0`
4. `ArmBase.cost_fn()` 会把环境 primitive collision 和 self collision 加到每条采样轨迹成本里。
5. 环境碰撞不是来自 Gazebo 物理接触，而是来自 STORM 内部的 primitive world SDF：
   - 机器人几何: `ur7e_collision_spheres.yml`
   - 世界几何: `collision_world_gazebo.yml`
   - 代价: 机器人 collision spheres 查询 primitive world SDF
6. `reach_static_ur7e.py` 里的 RViz 障碍物显示只是 markers，可视化与控制器使用的是同一份 YAML 障碍物配置。

## 高墙场景是如何接入的
1. 新建 `config/collision_world_gazebo_tall.yml`
2. 几何参数来自：
   - `/home/wqj/perception_D435i/src/sim_gazebo_scene/config/collision_world_gazebo_new.yml`
3. 与旧场景相比，仅把两面墙改成更高的盒体：
   - `cube1`: `[0.3, 0.1, 0.6]`
   - `cube2`: `[0.3, 0.1, 0.6]`
4. 保留 `ground`，这样控制器内部的 primitive 避障逻辑和旧场景一致。
5. 新建 `ur7e_reacher_gazebo_tall.yml`，保持与旧场景完全相同的 MPC/碰撞权重。
6. 新建：
   - `reach_static_ur7e_tall.py`
   - `ur7e_mpc_gazebo_tall.py`
   - `run_reach_static_tall.sh`
   - `test_mpc_tall.sh`

## 这套 tall 实验与原场景的一致性
- 控制器: 同一个 `MPPI`
- rollout: 同一个 `ArmReacher`
- 环境碰撞: 同一个 `primitive_collision`
- 自碰撞: 同一个 `robot_self_collision`
- 机器人几何: 同一个 `ur7e_collision_spheres.yml`
- 目标更新: 同一个 `/target_pose`
- RViz marker 发布: 同一套逻辑

## 运行方式
终端 1:
```bash
cd ~/storm/examples/sim_gazebo
./run_gazebo.sh
```

终端 2:
```bash
cd ~/storm/examples/sim_gazebo
./run_reach_static_tall.sh
```

如果只想跑基础 MPC 版本：
```bash
cd ~/storm/examples/sim_gazebo
./test_mpc_tall.sh
```

## 注意
- 这套实现保持与原 `sim_gazebo` 一致：高墙用于控制器内部 primitive-world 避障和 RViz 可视化。
- 它不会自动把高墙真正 spawn 成 Gazebo 物理障碍物。
