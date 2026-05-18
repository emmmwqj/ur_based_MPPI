# sage_sim_dynamic

这个目录下当前保留的是：

- `run_all_sage_reach_dynamic_ball_predict_simple.sh`
- 对应的 `simple` SAGE-MPPI predictive dynamic obstacle demo

## 关键配置文件

主控制器配置文件：

- [config/ur7e_reacher_sage_dynamic_ball_predict_simple.yml](/home/wqj/storm/examples/sage_sim_dynamic/config/ur7e_reacher_sage_dynamic_ball_predict_simple.yml)

动态球场景配置文件：

- [config/collision_world_sage_dynamic_ball.yml](/home/wqj/storm/examples/sage_sim_dynamic/config/collision_world_sage_dynamic_ball.yml)

## `dynamic_ball_safety_margin` 在哪修改

在这里改：

- [config/ur7e_reacher_sage_dynamic_ball_predict_simple.yml](/home/wqj/storm/examples/sage_sim_dynamic/config/ur7e_reacher_sage_dynamic_ball_predict_simple.yml)

字段：

```yaml
task:
  dynamic_ball_safety_margin: 0.03
```

含义：

- 动态球的有效碰撞半径会变成  
  `effective_radius = dynamic_ball_radius + dynamic_ball_safety_margin`
- 当前球半径在 world 配置里是 `0.06`
- 现在有效半径是 `0.06 + 0.03 = 0.09 m`

调大这个值：

- 会更早避障
- 更保守
- 更不容易擦球
- 但更可能绕不开或到目标更慢

## `primitive_collision.weight` 在哪修改

在这里改：

- [config/ur7e_reacher_sage_dynamic_ball_predict_simple.yml](/home/wqj/storm/examples/sage_sim_dynamic/config/ur7e_reacher_sage_dynamic_ball_predict_simple.yml)

字段：

```yaml
cost:
  primitive_collision:
    weight: 5000.0
```

含义：

- 这是环境 primitive collision cost 的权重
- 静态墙和动态球都走这项 cost
- 值越大，控制器越不愿意靠近障碍物

调大这个值：

- 会更保守
- 更早绕开墙和动态球
- 但可能降低收敛速度，甚至更容易停在局部次优位置

## 动态避障效果常用可调参数

优先看这几个：

1. `task.dynamic_ball_safety_margin`
- 位置：
  [config/ur7e_reacher_sage_dynamic_ball_predict_simple.yml](/home/wqj/storm/examples/sage_sim_dynamic/config/ur7e_reacher_sage_dynamic_ball_predict_simple.yml)
- 作用：
  增大动态球有效半径
- 建议范围：
  `0.03 ~ 0.08`

2. `cost.primitive_collision.weight`
- 位置：
  同上
- 作用：
  提高障碍物避让优先级
- 建议范围：
  `5000 ~ 12000`

3. `cost.primitive_collision.distance_threshold`
- 位置：
  同上
- 当前值：
  `0.05`
- 作用：
  提前多久开始产生碰撞代价
- 调大后：
  会更早躲避障碍物

4. `mppi.horizon`
- 位置：
  同上
- 当前值：
  `30`
- 作用：
  预测更远的未来
- 动态障碍场景下，如果经常“看到太晚”，可以适当调大

5. `mppi.num_particles`
- 位置：
  同上
- 当前值：
  `1000`
- 作用：
  提高采样覆盖度
- 调大后：
  更可能找到绕动态球的安全轨迹
- 代价：
  更慢

6. `mppi.execution_mode`
- 位置：
  同上
- 当前值：
  `best_sample`
- 可选：
  `best_sample` / `mean`
- 一般规律：
  - `best_sample` 更激进，动作更果断
  - `mean` 更平滑，通常更保守

## 动态球运动参数在哪改

在这里改：

- [config/collision_world_sage_dynamic_ball.yml](/home/wqj/storm/examples/sage_sim_dynamic/config/collision_world_sage_dynamic_ball.yml)

关键字段：

```yaml
world_model:
  dynamic_obstacles:
    dynamic_ball:
      radius: 0.06
      initial_position: [0.4, -0.6, 0.4]
      y_limits: [-0.6, 0.6]
      speed: 0.1
      update_hz: 20.0
```

这些参数影响：

- 球本身大小
- 初始位置
- 运动范围
- 运动速度
- mover 更新频率

## 如果想让动态避障更强，建议先怎么调

建议按这个顺序试：

1. 先把 `dynamic_ball_safety_margin` 从 `0.03` 调到 `0.05`
2. 再把 `primitive_collision.weight` 从 `5000` 调到 `8000`
3. 如果仍然太晚躲避，再把 `distance_threshold` 从 `0.05` 调到 `0.07`
4. 如果还是容易找不到绕行动作，再增加 `num_particles`
5. 如果动作太激进，可以把 `execution_mode` 从 `best_sample` 改成 `mean`

## 启动命令

```bash
cd /home/wqj/storm/examples/sage_sim_dynamic/bash
./run_all_sage_reach_dynamic_ball_predict_simple.sh
```

## 结论

如果你的目标是先提高动态避障安全性，最先改的是：

1. `task.dynamic_ball_safety_margin`
2. `cost.primitive_collision.weight`
3. `cost.primitive_collision.distance_threshold`

这三个参数最直接。*** End Patch
