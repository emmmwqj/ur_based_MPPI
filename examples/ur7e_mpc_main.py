#!/usr/bin/env python3
"""
UR7e MPC Control - 简洁版本

使用 Isaac Sim Core API 实现 MPPI-MPC 控制
参考 latest_ur7e_reacher.py 的结构

用法:
    cd /path/to/isaac-sim && ./python.sh ~/storm/examples/ur7e_mpc_main.py

架构说明:
    Step 1: 纯仿真 (本脚本)
    Step 2: SIL - 只需替换 get_state() 和 send_command() 为 ROS2 通信
    Step 3: HIL - 同 Step 2，连接真实机器人
"""

import argparse
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

# 全局变量
simulation_app = None

np.set_printoptions(precision=2)


# ============================================================================
# Transform 类
# ============================================================================

class Transform:
    """坐标变换类"""
    def __init__(self, p=None, r=None):
        # p: 位置 [x, y, z]
        # r: 四元数 [x, y, z, w] (xyzw 格式)
        self.p = np.array(p) if p is not None else np.array([0.0, 0.0, 0.0])
        self.r = np.array(r) if r is not None else np.array([0.0, 0.0, 0.0, 1.0])
        
    def transform_point(self, point):
        """将点从局部坐标系变换到世界坐标系"""
        rot = Rotation.from_quat(self.r)
        return rot.apply(point) + self.p


# ============================================================================
# 场景管理类
# ============================================================================

class IsaacSimWorld:
    """Isaac Sim 场景管理 - 生成障碍物和标记"""
    
    def __init__(self, world, world_params, w_T_r=None):
        self.world = world
        self.world_params = world_params
        self.w_T_r = w_T_r  # Transform 对象
        self.objects = {}
        
    def spawn_primitives(self):
        """从配置文件生成碰撞障碍物"""
        from isaacsim.core.api.objects import VisualCuboid, VisualSphere
        
        if self.world_params is None:
            return
            
        world_model = self.world_params.get('world_model', {})
        coll_objs = world_model.get('coll_objs', {})
        
        # 生成球体障碍物
        spheres = coll_objs.get('sphere', {})
        for name, params in spheres.items():
            radius = params.get('radius', 0.1)
            position = np.array(params.get('position', [0, 0, 0]))
            
            # 变换到世界坐标系
            if self.w_T_r is not None:
                position = self.w_T_r.transform_point(position)
                
            sphere = VisualSphere(
                prim_path=f"/World/obstacles/{name}",
                name=name,
                position=position,
                radius=radius,
                color=np.array([0.8, 0.2, 0.2])
            )
            self.world.scene.add(sphere)
            self.objects[name] = sphere
            print(f"  生成球体障碍物: {name} at {position}")
            
        # 生成立方体障碍物
        cubes = coll_objs.get('cube', {})
        for name, params in cubes.items():
            dims = np.array(params.get('dims', [0.1, 0.1, 0.1]))
            pose = params.get('pose', [0, 0, 0, 0, 0, 0, 1])
            position = np.array(pose[:3])
            orientation_xyzw = np.array(pose[3:])
            
            # 变换到世界坐标系
            if self.w_T_r is not None:
                position = self.w_T_r.transform_point(position)
                
            # 转换为 wxyz 格式
            orientation_wxyz = np.array([orientation_xyzw[3], orientation_xyzw[0], 
                                         orientation_xyzw[1], orientation_xyzw[2]])
            
            cube = VisualCuboid(
                prim_path=f"/World/obstacles/{name}",
                name=name,
                position=position,
                orientation=orientation_wxyz,
                size=1.0,
                scale=dims,
                color=np.array([0.5, 0.5, 0.8])
            )
            self.world.scene.add(cube)
            self.objects[name] = cube
            print(f"  生成立方体障碍物: {name} at {position}, dims={dims}")


# ============================================================================
# 辅助函数
# ============================================================================

def transform_point(position, orientation_xyzw, point):
    """将点从机器人坐标系变换到世界坐标系"""
    rot = Rotation.from_quat(orientation_xyzw)
    return rot.apply(point) + position

def inv_transform_point(position, orientation_xyzw, point):
    """将点从世界坐标系变换到机器人坐标系"""
    rot = Rotation.from_quat(orientation_xyzw).inv()
    return rot.apply(point - position)


# ============================================================================
# 主控制函数
# ============================================================================

def mpc_control_main(args):
    """主控制函数 - 必须在 SimulationApp 创建后调用"""
    
    global simulation_app
    
    # ========================================================================
    # 导入（必须在 SimulationApp 之后）
    # ========================================================================
    
    from isaacsim.core.api.world import World
    from isaacsim.core.api.objects import VisualSphere
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import UsdPhysics, UsdGeom, Gf, Vt
    import omni.timeline
    
    from storm_kit.util_file import get_gym_configs_path, join_path, get_assets_path
    from storm_kit.mpc.task.reacher_task import ReacherTask
    from storm_kit.differentiable_robot_model.coordinate_transform import (
        quaternion_to_matrix, CoordinateTransform
    )
    
    # ========================================================================
    # 场景设置
    # ========================================================================
    
    print("=" * 60)
    print("UR7e MPC Control - 简洁版")
    print("=" * 60)
    
    # 加载配置
    robot_file = 'ur7e_isaacsim.yml'
    task_file = 'ur7e_reacher_isaacsim.yml'
    world_file = 'collision_primitives_3d.yml'
    
    with open(join_path(get_gym_configs_path(), robot_file)) as f:
        robot_params = yaml.safe_load(f)
    
    # 加载世界/障碍物配置
    with open(join_path(get_gym_configs_path(), world_file)) as f:
        world_params = yaml.safe_load(f)
    
    sim_params = robot_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    
    # 使用 Isaac Sim Core API 创建 World
    device = 'cuda' if args.cuda else 'cpu'
    world = World(stage_units_in_meters=1.0)
    
    # 添加地面
    world.scene.add_default_ground_plane()
    
    # 机器人位姿
    robot_pose = sim_params.get('robot_pose', [0, 0, 0, 0, 0, 0, 1])
    robot_pos = np.array(robot_pose[:3])
    robot_quat_xyzw = np.array(robot_pose[3:])  # xyzw
    robot_quat_wxyz = np.array([robot_quat_xyzw[3], *robot_quat_xyzw[:3]])  # wxyz
    
    # 创建机器人坐标变换
    w_T_r = Transform(p=robot_pos, r=robot_quat_xyzw)
    
    # 加载机器人
    print(f"加载机器人: {get_assets_path()}/urdf/ur7e/ur7e.usd")
    ur7e_usd_path = get_assets_path() + "/urdf/ur7e/ur7e.usd"
    ur7e_prim_path = "/World/UR7e"
    
    add_reference_to_stage(usd_path=ur7e_usd_path, prim_path=ur7e_prim_path)
    
    n_dof = 6
    init_q = np.array(sim_params.get('init_state', [0.0]*n_dof)[:n_dof])
    
    # 创建 Robot 对象并添加到场景
    robot = world.scene.add(
        Robot(
            prim_path=ur7e_prim_path,
            name="ur7e",
            position=robot_pos,
            orientation=robot_quat_wxyz
        )
    )
    
    # 重置 world 以初始化
    world.reset()
    
    # 获取 articulation controller 用于发送指令
    articulation_controller = robot.get_articulation_controller()
    
    # 设置初始关节位置
    num_robot_dof = robot.num_dof or 6
    init_q_full = np.zeros(num_robot_dof)
    init_q_full[:n_dof] = init_q
    robot.set_joint_positions(init_q_full)
    robot.set_joint_velocities(np.zeros(num_robot_dof))
    
    # 配置关节驱动
    stage = get_current_stage()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.DriveAPI):
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                drive.GetStiffnessAttr().Set(400.0)
                drive.GetDampingAttr().Set(40.0)
    
    print("关节驱动配置完成: stiffness=400, damping=40")
    
    # ========================================================================
    # 创建障碍物场景
    # ========================================================================
    
    print("\n创建障碍物...")
    world_instance = IsaacSimWorld(world, world_params, w_T_r=w_T_r)
    world_instance.spawn_primitives()
    
    # ========================================================================
    # MPC 控制器
    # ========================================================================
    
    print("\n设置 MPC 控制器...")
    tensor_args = {'device': torch.device('cuda', 0) if args.cuda else 'cpu', 'dtype': torch.float32}
    mpc = ReacherTask(task_file, robot_file, world_file, tensor_args)
    control_dt = mpc.exp_params['control_dt']
    
    # 目标状态
    goal_state = np.array([0.5, -1.2, 1.2, -1.57, -1.57, 0.0, 0, 0, 0, 0, 0, 0])
    mpc.update_params(goal_state=goal_state)
    
    goal_ee_pos = np.ravel(mpc.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    goal_ee_quat = np.ravel(mpc.controller.rollout_fn.goal_ee_quat.cpu().numpy())
    print(f"目标末端位置: {goal_ee_pos}")
    
    # 坐标变换（用于轨迹可视化）
    w_T_robot = torch.eye(4)
    w_T_robot[:3, 3] = torch.tensor(robot_pos)
    w_T_robot[:3, :3] = quaternion_to_matrix(torch.tensor(robot_quat_wxyz).unsqueeze(0))[0]
    w_robot_coord = CoordinateTransform(trans=w_T_robot[:3,3].unsqueeze(0), rot=w_T_robot[:3,:3].unsqueeze(0))
    
    # ========================================================================
    # 可视化标记
    # ========================================================================
    
    # 目标标记（红色）
    goal_world = transform_point(robot_pos, robot_quat_xyzw, goal_ee_pos)
    goal_marker = VisualSphere(
        prim_path="/World/Markers/Goal",
        name="goal_marker",
        position=goal_world,
        radius=0.03,
        color=np.array([0.9, 0.1, 0.1])
    )
    world.scene.add(goal_marker)
    
    # EE标记（绿色）
    ee_marker = VisualSphere(
        prim_path="/World/Markers/EE",
        name="ee_marker",
        position=np.array([0, 0, 0]),
        radius=0.025,
        color=np.array([0.1, 0.9, 0.1])
    )
    world.scene.add(ee_marker)
    
    # 轨迹线
    UsdGeom.Scope.Define(stage, "/World/Traj")
    traj_curve = UsdGeom.BasisCurves.Define(stage, "/World/Traj/line")
    traj_curve.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    traj_curve.GetDisplayColorPrimvar().Set(Vt.Vec3fArray([Gf.Vec3f(1, 0, 0)]))
    
    # ========================================================================
    # 主循环
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("开始 MPC 控制循环... (Ctrl+C 退出)")
    print("=" * 60 + "\n")
    
    omni.timeline.get_timeline_interface().play()
    
    # 预热
    for _ in range(10):
        world.step(render=True)
    
    # 加速估计用
    prev_vel = np.zeros(n_dof)
    
    t = 0.0
    i = 0
    
    # 预热 MPC
    print("预热 MPC...")
    for _ in range(5):
        q = robot.get_joint_positions()[:n_dof]
        dq = robot.get_joint_velocities()[:n_dof]
        state = {'position': q, 'velocity': dq, 'acceleration': np.zeros(n_dof)}
        try:
            mpc.get_command(t, state, control_dt=control_dt, WAIT=False)
        except:
            pass
        t += control_dt
        world.step(render=True)
    
    print("MPC 预热完成，进入主循环\n")
    print("提示: 在 Isaac Sim 中选中红色目标球并拖动，机械臂会动态跟踪！\n")
    
    # 用于检测目标位置变化
    current_goal_ee = goal_ee_pos.copy()
    
    while simulation_app.is_running():
        try:
            world.step(render=True)
            t += control_dt
            
            # --- 检测目标是否被拖动 ---
            goal_world_new, _ = goal_marker.get_world_pose()
            goal_robot_new = inv_transform_point(robot_pos, robot_quat_xyzw, goal_world_new)
            
            if np.linalg.norm(goal_robot_new - current_goal_ee) > 0.005:  # 移动超过5mm
                current_goal_ee = goal_robot_new.copy()
                mpc.update_params(goal_ee_pos=current_goal_ee, goal_ee_quat=goal_ee_quat)
                print(f"[目标更新] 新位置: {np.round(current_goal_ee, 3)}")
            
            # --- 获取状态 ---
            q = robot.get_joint_positions()[:n_dof]
            dq = robot.get_joint_velocities()[:n_dof]
            ddq = (dq - prev_vel) / max(control_dt, 0.001)
            prev_vel = dq.copy()
            
            state = {'position': q, 'velocity': dq, 'acceleration': ddq}
            
            # --- MPC 计算 ---
            try:
                cmd = mpc.get_command(t, state, control_dt=control_dt, WAIT=True)
            except IndexError:
                i += 1
                continue
            
            # --- 发送指令 ---
            q_des = np.zeros(num_robot_dof)
            q_des[:n_dof] = cmd['position']
            action = ArticulationAction(joint_positions=q_des)
            articulation_controller.apply_action(action)
            
            # --- 更新可视化 ---
            # EE位置
            curr = np.hstack([q, dq, ddq])
            ee_pose = mpc.controller.rollout_fn.get_ee_pose(
                torch.as_tensor(curr, **tensor_args).unsqueeze(0)
            )
            ee_pos = np.ravel(ee_pose['ee_pos_seq'].cpu().numpy())
            ee_world = transform_point(robot_pos, robot_quat_xyzw, ee_pos)
            ee_marker.set_world_pose(position=ee_world)
            
            # 轨迹
            if mpc.top_trajs is not None:
                traj = mpc.top_trajs[0].cpu().float()  # 取第一条轨迹
                traj_world = w_robot_coord.transform_point(traj).cpu().numpy()
                # 转换为 Python float 类型
                pts = Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in traj_world])
                traj_curve.GetPointsAttr().Set(pts)
                traj_curve.GetCurveVertexCountsAttr().Set(Vt.IntArray([len(traj_world)]))
                traj_curve.GetWidthsAttr().Set(Vt.FloatArray([0.003]*len(traj_world)))
            
            # --- 打印状态 ---
            if i % 50 == 0:
                err = mpc.get_current_error(state)
                print(f"[{i:4d}] 误差: {[f'{x:.3f}' for x in err]}, "
                      f"opt: {mpc.opt_dt:.3f}s, mpc: {mpc.mpc_dt:.3f}s")
            
            i += 1
            
        except KeyboardInterrupt:
            print("\n退出...")
            break
        except Exception as e:
            print(f"错误: {e}")
            i += 1
    
    # ========================================================================
    # 清理
    # ========================================================================
    
    print("\n清理资源...")
    mpc.close()
    simulation_app.close()
    print("完成!")


# ============================================================================
# 入口点 - 必须在 if __name__ == '__main__' 保护下
# ============================================================================

if __name__ == '__main__':
    # 解析参数
    parser = argparse.ArgumentParser(description='UR7e MPC Control')
    parser.add_argument('--headless', action='store_true', help='无头模式')
    parser.add_argument('--cuda', action='store_true', default=True, help='使用CUDA')
    args = parser.parse_args()
    
    # 启动 Isaac Sim - 必须在 __main__ 中
    from isaacsim import SimulationApp
    
    print("启动 Isaac Sim...")
    simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})
    
    # Torch 配置
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 运行主控制函数
    try:
        mpc_control_main(args)
    except Exception as e:
        print(f"致命错误: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
