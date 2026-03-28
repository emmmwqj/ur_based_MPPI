# ESDF 查询安全裕量

## 这是什么

`safety_margin_world` 表示在查询 ESDF 时，主动把环境障碍物看得比真实更大一点。

当前实现等价于：

```text
effective_esdf = queried_esdf - safety_margin_world
```

如果 `safety_margin_world = 0.03`，那么控制器会把所有障碍物边界向自由空间方向“膨胀” `3 cm`。

## 为什么要加它

`whole_sim_gazebo` 用的是离线 ESDF snapshot，不是 primitive 几何真值。

即使离线对比里平均误差不大，墙附近局部仍然可能有几厘米误差。  
控制器如果直接相信这张 ESDF，就可能认为“还能过”，但 Gazebo 里的真实墙已经碰上了。

安全裕量的作用就是让控制器更保守：

- 更早开始惩罚靠近障碍物的轨迹
- 减少“ESDF 看起来没撞，Gazebo 里实际撞了”的情况

## 当前默认值

高墙场景当前默认使用：

- 配置文件：[esdf_world_gazebo_tall.yml](/home/wqj/storm/examples/whole_sim_gazebo/config/esdf_world_gazebo_tall.yml)
- 参数：`safety_margin_world: 0.03`

也就是默认加了 `3 cm` 的保守裕量。

## 怎么调

如果你发现：

- 机械臂仍然会擦墙或撞墙  
  把它调大，例如：
  - `0.04`
  - `0.05`

- 机械臂明显过于保守，本来能通过的缝隙也不走  
  把它调小，例如：
  - `0.02`
  - `0.01`

## 调参建议

建议按下面顺序调：

1. 先试 `0.03`
2. 还会碰撞就升到 `0.04`
3. 再不够就升到 `0.05`
4. 如果开始明显绕得太远，再回退到 `0.02`

## 副作用

安全裕量不是免费的。它会带来两个直接副作用：

1. 轨迹更保守，可能绕路更多
2. 狭窄通道会变得更难通过

所以它的目标不是“越大越好”，而是：

- 大到足够覆盖 ESDF 的局部误差
- 小到不破坏正常通过能力

## 这次还顺手修了什么

除了加安全裕量，这次还把 `whole_sim_gazebo` 的 tall 入口改成了：

- 目标更新后重置 `ControlProcess`
- 立即同步重规划一次

这样可以减少“目标从墙一侧跳到另一侧时，控制器还沿用旧 hotstart 分布”带来的撞墙风险。
