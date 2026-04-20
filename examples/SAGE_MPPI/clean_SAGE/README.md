# clean_SAGE quick usage

正式配置文件：
- `examples/SAGE_MPPI/clean_SAGE/config/ur7e_reacher_gazebo_tall_sage_clean.yml`

启动 Gazebo + clean SAGE：
```bash
cd ~/storm/examples/SAGE_MPPI/clean_SAGE
./run_all_reach_static_tall.sh
```

只启动 clean SAGE 控制器：
```bash
cd ~/storm/examples/SAGE_MPPI/clean_SAGE
./run_reach_static_tall.sh
```

通过 `/target_pose` 发布新目标：
```bash
source /opt/ros/humble/setup.bash
ros2 topic pub /target_pose geometry_msgs/PoseStamped "{header: {frame_id: 'world'}, pose: {position: {x: 0.5, y: 0.0, z: 0.45}, orientation: {w: 1.0}}}" -1
```

运行固定回归验证：
```bash
cd ~/storm/examples/SAGE_MPPI/clean_SAGE
python3 run_local_refinement_regression.py
```

回归验证固定内容：
- 目标点：5 个固定 `target_pose`
- 种子：`0, 1, 2`
- 汇总输出：
  - `mean_final_ee_error`
  - `worst_final_ee_error`
  - `<2cm` 成功率
  - `<5mm` 成功率
