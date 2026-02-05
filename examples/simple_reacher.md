**配置文件**:content/configs/mpc/simple_reacher.yml

```
use_cuda: True
cuda_device_num: 0

control_dt: 0.1
control_space: 'acc'
float_dtype: 'float32'
state_filter_coeff: 1.0
cmd_filter_coeff: 1.0

model:
  # any link that is not specified as learnable will be initialized from urdf
  #urdf_path: "urdf/franka_description/franka_panda_no_gripper.urdf"
  #learnable_rigid_body_config:
  #  learnable_links: []
  #name: "franka_panda"
  dt: 0.0
  max_action: 0.1 #10.0

  dt_traj_params:
    base_dt: 0.1
    base_ratio: 1.0
    max_dt: 0.3
  init_state: [0.0,0.0]
  position_bounds: [[0.0, 0.5], [0.0,0.5]]
  
cost:
  goal_state:
    vec_weight: [1.0, 1.0]
    weight: 100.0
    gaussian_params: {'n':0, 'c':0.2, 's':0.0, 'r':10.0}
    

  zero_vel:
    weight: 0.0
    hinge_val: 0.2 #0.2
    gaussian_params: {'n':0, 'c':0.2, 's':0, 'r':1.0}
    
  stop_cost:
    weight: 100.0
    max_nlimit: 0.05 #0.2
    gaussian_params: {'n':0, 'c':0.2, 's':0, 'r':10.0}
  stop_cost_acc:
    weight: 100.0
    max_limit: 0.01 #0.2
    gaussian_params: {'n':0, 'c':0.2, 's':0, 'r':10.0}
    
  
  smooth: # on robot acceleration
    weight: 0.0 
    gaussian_params: {'n':0, 'c':0.2, 's':0, 'r':1.0}
    order: 3 # on velocity

    
  image_collision: # on robot acceleration
    weight: 1000.0 
    gaussian_params: {'n':0, 'c':1.0, 's':0, 'r':10.0}
    collision_file: 'collision_maps/collision_map_cem.png'
    dist_thresh: 0.01
  state_bound:
    weight: 100.0 
    gaussian_params: {'n':0, 'c':1.0, 's':0, 'r':10.0}
  terminal:
    weight: 0.0
    gaussian_params: {'n':0, 'c':1.0, 's':0, 'r':10.0}
mppi:
  horizon           : 30 # 100
  init_cov          : 0.01 #.5
  gamma             : 0.98 #
  n_iters           : 1
  step_size_mean    : 0.9
  step_size_cov     : 0.6
  beta              : 1.0
  alpha             : 1
  num_particles     : 500 #10000
  update_cov        : True
  cov_type          : 'diag_AxA' # 
  kappa             : 0.0001
  null_act_frac     : 0.01
  sample_mode       : 'mean'
  base_action       : 'repeat'
  squash_fn         : 'clamp' # [clamp,]
  hotstart          : True
  visual_traj       : 'state_seq'
  sample_params:
    type: 'multiple'
    fixed_samples: True
    sample_ratio: {'halton':0.0, 'halton-knot':1.0, 'random':0.0, 'random-knot':0.0}
    seed: 0
    filter_coeffs: None #[0.5, 0.3, 0.2]
    knot_scale: 5
    #filter_coeffs: [1.0, 0.0, 0.0]
```
**horizon和num_particles的影响**

当horizon=100时，num_particles=10000，效果会更好(可达到论文图三Halton B-Spline从两个障碍物中间钻过去的效果)，但计算开销也更大。

当horizon=30，num_particles=500，计算开销小，但无法实现从障碍物中间钻过去。
**knot_scale的影响**
## knot_scale 的物理意义

`knot_scale` 表示**每个 B 样条数据点"负责"多少个时间步**，即数据点之间的时间间隔（以时间步为单位）。

### 数学关系

$$M = \lfloor H / \text{knot\_scale} \rfloor$$

$$\text{数据点时间间隔} = \text{knot\_scale} \times dt$$

### 物理含义图解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    knot_scale = 4 的物理含义 (H=30, dt=0.02s)                │
└─────────────────────────────────────────────────────────────────────────────┘

  时间步:  0   1   2   3   4   5   6   7   8   9  10  11  12 ... 28  29
           |       |       |       |       |       |       |       |
  数据点:  P0      P1      P2      P3      P4      P5      P6     (M=7)
           |←─────→|
           knot_scale = 4 个时间步
           = 4 × 0.02s = 0.08s

  含义: 每 0.08 秒放置一个数据点，共 7 个点描述 0.6 秒的轨迹
```

### 不同 knot_scale 的物理对比

| knot_scale | 数据点数 M | 数据点时间间隔 | 物理意义 |
|------------|-----------|---------------|----------|
| 2 | 15 | 0.04s (40ms) | 每 40ms 一个控制点，细粒度 |
| 3 | 10 | 0.06s (60ms) | 每 60ms 一个控制点 |
| **4** | **7** | **0.08s (80ms)** | **每 80ms 一个控制点（默认）** |
| 5 | 6 | 0.10s (100ms) | 每 100ms 一个控制点 |
| 6 | 5 | 0.12s (120ms) | 每 120ms 一个控制点，粗粒度 |

### 直观理解

```
knot_scale = "轨迹的时间分辨率"

小 knot_scale (如 2):          大 knot_scale (如 6):
─────────────────────          ─────────────────────
数据点密集                      数据点稀疏
  •  •  •  •  •  •  •            •        •        •
  ↓                              ↓
可以描述快速变化的轨迹          只能描述缓慢变化的轨迹
适合: 快速避障、动态跟踪        适合: 平稳到达、简单任务
```

### 与控制频率的关系

```
控制频率: 50 Hz (dt = 0.02s)
knot_scale = 4

→ 数据点更新频率 = 50 / 4 = 12.5 Hz
→ 即轨迹的"形状"以 12.5 Hz 的分辨率被描述
→ 中间的点由 B 样条平滑插值生成
```

**总结**：`knot_scale` 本质上控制了**轨迹描述的时间分辨率**——值越小，轨迹描述越精细；值越大，轨迹描述越粗糙但更平滑。