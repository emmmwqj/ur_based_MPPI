#!/usr/bin/env python3
"""
UR7e Isaac Sim + ROS2 桥接脚本 (OmniGraph 版本)

使用 Isaac Sim 内置的 OmniGraph ROS2 桥接节点发布/订阅话题。
Isaac Sim 使用内置 Python 3.11 ROS2 库，外部 ROS 节点使用系统 ROS2。
DDS 中间件负责跨 Python 版本的通信。

发布话题:
    /joint_states (sensor_msgs/JointState) - 机器人关节状态
    
订阅话题:
    /joint_command (sensor_msgs/JointState) - 关节位置指令

用法:
    # 终端 1: 运行本脚本 (不要 source 系统 ROS2!)
    cd ~/storm/examples
    ./run_ur7e_ros2_sim.sh
    
    # 终端 2: 运行 MPC 控制器 (使用系统 ROS2)
    conda activate storm_py310
    source /opt/ros/humble/setup.bash
    python3 ur7e_mpc_ros2.py

架构说明:
    - Isaac Sim 内置 ROS2 库 (Python 3.11) 通过 OmniGraph 发布关节状态
    - MPC 控制器使用系统 ROS2 (Python 3.10) 订阅状态、发布指令
    - FastDDS 中间件处理不同 Python 版本之间的通信
"""

import sys
import os
import numpy as np
import yaml
import time

# ============================================================================
# Isaac Sim 初始化
# ============================================================================

print("=" * 60)
print("UR7e Isaac Sim + ROS2 桥接 (OmniGraph)")
print("=" * 60)

import isaacsim
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
})

# ============================================================================
# Isaac Sim 导入
# ============================================================================

from isaacsim.core.api.world import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import VisualSphere, VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdGeom, Gf, UsdPhysics, Sdf
import omni.graph.core as og
import omni.kit.app
import omni.timeline

# 启用 ROS2 扩展
print("\n启用 ROS2 扩展...")
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
print("已启用 isaacsim.ros2.bridge")

# STORM 路径
sys.path.insert(0, '/home/wqj/storm')
from storm_kit.util_file import get_gym_configs_path, join_path, get_assets_path


# ============================================================================
# 辅助函数
# ============================================================================

def find_articulation_root(robot_prim_path: str):
    """
    查找机器人的 ArticulationRoot prim 路径
    
    UR7e USD 文件的 ArticulationRoot 可能在子路径 (如 /World/UR7e/ur7e)
    OmniGraph 节点需要正确的 ArticulationRoot 路径才能工作
    """
    stage = get_current_stage()
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    
    if not robot_prim.IsValid():
        print(f"警告: 未找到 prim: {robot_prim_path}")
        return robot_prim_path
    
    # 打印 prim 层级结构用于调试
    print(f"\n调试: {robot_prim_path} 的子 prim 结构:")
    def print_children(prim, indent=0):
        for child in prim.GetChildren():
            has_art = child.HasAPI(UsdPhysics.ArticulationRootAPI)
            marker = " [ArticulationRoot]" if has_art else ""
            print(f"{'  ' * indent}- {child.GetName()}{marker}")
            if indent < 2:  # 限制打印深度
                print_children(child, indent + 1)
    print_children(robot_prim)
    print()
    
    # 检查根 prim 是否是 ArticulationRoot
    if robot_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print(f"  ArticulationRoot: {robot_prim_path}")
        return robot_prim_path
    
    # 递归搜索子 prim
    def search_children(prim, depth=0):
        if depth > 5:  # 增加搜索深度
            return None
        for child in prim.GetChildren():
            if child.HasAPI(UsdPhysics.ArticulationRootAPI):
                return str(child.GetPath())
            result = search_children(child, depth + 1)
            if result:
                return result
        return None
    
    articulation_path = search_children(robot_prim)
    
    if articulation_path:
        print(f"  找到 ArticulationRoot: {articulation_path}")
        return articulation_path
    
    # 如果找不到，尝试常见的子路径模式
    common_subpaths = ["/ur7e", "/robot", "/base_link", "/world"]
    for subpath in common_subpaths:
        test_path = robot_prim_path + subpath
        test_prim = stage.GetPrimAtPath(test_path)
        if test_prim.IsValid() and test_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"  找到 ArticulationRoot: {test_path}")
            return test_path
    
    print(f"  警告: 未找到 ArticulationRoot，使用原路径: {robot_prim_path}")
    return robot_prim_path


# ============================================================================
# OmniGraph ROS2 桥接设置
# ============================================================================

def setup_ros2_omnigraph(robot_prim_path: str, joint_names: list):
    """
    使用 OmniGraph 设置 ROS2 发布和订阅
    
    创建两个独立的图:
    1. 发布器图: 发布关节状态到 /joint_states
    2. 订阅器图: 订阅 /joint_command 并应用到机器人
    """
    
    keys = og.Controller.Keys
    stage = get_current_stage()
    
    success = True
    
    # ========================================================================
    # 1. 创建关节状态发布器图
    # ========================================================================
    try:
        print("\n创建关节状态发布器...")
        
        (pub_graph, pub_nodes, _, _) = og.Controller.edit(
            {"graph_path": "/World/ROS2_JointStatePublisher", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ],
                keys.SET_VALUES: [
                    ("PublishJointState.inputs:topicName", "/joint_states"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                    ("Context.outputs:context", "PublishJointState.inputs:context"),
                ],
            }
        )
        
        # 设置目标机器人 (使用 relationship)
        pub_node_prim = stage.GetPrimAtPath("/World/ROS2_JointStatePublisher/PublishJointState")
        if pub_node_prim.IsValid():
            target_rel = pub_node_prim.CreateRelationship("inputs:targetPrim", False)
            target_rel.SetTargets([Sdf.Path(robot_prim_path)])
            print(f"  发布器目标: {robot_prim_path}")
        
        print("  发布话题: /joint_states")
        
    except Exception as e:
        print(f"  发布器创建失败: {e}")
        success = False
    
    # ========================================================================
    # 2. 创建关节指令订阅器图
    # ========================================================================
    try:
        print("\n创建关节指令订阅器...")
        
        (sub_graph, sub_nodes, _, _) = og.Controller.edit(
            {"graph_path": "/World/ROS2_JointCommandSubscriber", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ],
                keys.SET_VALUES: [
                    ("SubscribeJointState.inputs:topicName", "/joint_command"),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                    ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                    ("SubscribeJointState.outputs:execOut", "ArticulationController.inputs:execIn"),
                    ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ],
            }
        )
        
        # 设置目标机器人 (使用 relationship)
        ctrl_node_prim = stage.GetPrimAtPath("/World/ROS2_JointCommandSubscriber/ArticulationController")
        if ctrl_node_prim.IsValid():
            target_rel = ctrl_node_prim.CreateRelationship("inputs:targetPrim", False)
            target_rel.SetTargets([Sdf.Path(robot_prim_path)])
            print(f"  控制器目标: {robot_prim_path}")
        
        print("  订阅话题: /joint_command")
        
    except Exception as e:
        print(f"  订阅器创建失败: {e}")
        success = False
    
    return success


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主仿真循环"""
    
    # ========================================================================
    # 加载配置
    # ========================================================================
    
    robot_file = 'ur7e_isaacsim.yml'
    world_file = 'collision_primitives_3d.yml'
    
    with open(join_path(get_gym_configs_path(), robot_file)) as f:
        robot_params = yaml.safe_load(f)
    
    with open(join_path(get_gym_configs_path(), world_file)) as f:
        world_params = yaml.safe_load(f)
    
    sim_params = robot_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    
    robot_pose = sim_params.get('robot_pose', [0, 0, 0, 0, 0, 0, 1])
    robot_pos = np.array(robot_pose[:3])
    robot_quat_xyzw = np.array(robot_pose[3:])
    robot_quat_wxyz = np.array([robot_quat_xyzw[3], *robot_quat_xyzw[:3]])
    
    n_dof = 6
    joint_names = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]
    
    # ========================================================================
    # 创建仿真世界
    # ========================================================================
    
    print("\n创建仿真世界...")
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0/60.0,
        rendering_dt=1.0/60.0
    )
    
    world.scene.add_default_ground_plane()
    
    # 添加机器人
    usd_path = get_assets_path() + "/urdf/ur7e/ur7e.usd"
    robot_prim_path = "/World/UR7e"
    print(f"加载机器人: {usd_path}")
    
    add_reference_to_stage(usd_path, robot_prim_path)
    
    robot = world.scene.add(Robot(
        prim_path=robot_prim_path,
        name="ur7e",
        position=robot_pos,
        orientation=robot_quat_wxyz
    ))
    
    # 目标标记
    goal_marker = world.scene.add(VisualSphere(
        prim_path="/World/goal_marker",
        name="goal_marker",
        position=np.array([0.4, 0.0, 0.5]),
        radius=0.03,
        color=np.array([0.1, 0.8, 0.1])
    ))
    
    # ========================================================================
    # 添加障碍物
    # ========================================================================
    
    print("\n创建障碍物...")
    world_model = world_params.get('world_model', {})
    coll_objs = world_model.get('coll_objs', {})
    
    # 球体
    for name, params in coll_objs.get('sphere', {}).items():
        pos = np.array(params.get('position', [0, 0, 0]))
        world.scene.add(VisualSphere(
            prim_path=f"/World/obstacles/{name}",
            name=name,
            position=pos,
            radius=params.get('radius', 0.1),
            color=np.array([0.8, 0.2, 0.2])
        ))
        print(f"  球体: {name} at {pos}")
    
    # 立方体
    for name, params in coll_objs.get('cube', {}).items():
        pose = params.get('pose', [0, 0, 0, 0, 0, 0, 1])
        dims = np.array(params.get('dims', [0.1, 0.1, 0.1]))
        pos = np.array(pose[:3])
        quat_xyzw = np.array(pose[3:])
        quat_wxyz = np.array([quat_xyzw[3], *quat_xyzw[:3]])
        world.scene.add(VisualCuboid(
            prim_path=f"/World/obstacles/{name}",
            name=name,
            position=pos,
            orientation=quat_wxyz,
            size=1.0,
            scale=dims,
            color=np.array([0.5, 0.5, 0.8])
        ))
        print(f"  立方体: {name} at {pos}")
    
    # ========================================================================
    # 初始化仿真
    # ========================================================================
    
    print("\n初始化仿真...")
    world.reset()
    
    # 配置关节驱动
    stage = get_current_stage()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.DriveAPI):
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                drive.GetStiffnessAttr().Set(400.0)
                drive.GetDampingAttr().Set(40.0)
    
    articulation_controller = robot.get_articulation_controller()
    
    # 设置初始姿态
    init_q = np.array(sim_params.get('init_state', [0.0]*6)[:6])
    num_dof = robot.num_dof or 6
    init_q_full = np.zeros(num_dof)
    init_q_full[:6] = init_q
    robot.set_joint_positions(init_q_full)
    robot.set_joint_velocities(np.zeros(num_dof))
    
    for _ in range(10):
        world.step(render=True)
    
    # 启动 timeline (OmniGraph 需要 timeline 运行)
    omni.timeline.get_timeline_interface().play()
    
    # 等待物理引擎初始化
    for _ in range(30):
        world.step(render=True)
    
    # ========================================================================
    # 设置 ROS2 OmniGraph 桥接
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("设置 ROS2 OmniGraph 桥接...")
    print("=" * 60)
    
    # 查找正确的 ArticulationRoot 路径
    articulation_path = find_articulation_root(robot_prim_path)
    
    ros2_ok = setup_ros2_omnigraph(articulation_path, joint_names)
    
    # ========================================================================
    # 主循环
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("仿真已启动!")
    print("=" * 60)
    
    if ros2_ok:
        print("\nROS2 话题 (通过 OmniGraph):")
        print("  发布: /joint_states (sensor_msgs/JointState)")
        print("  订阅: /joint_command (sensor_msgs/JointState)")
        print("\n在另一个终端验证 (使用系统 ROS2):")
        print("  source /opt/ros/humble/setup.bash")
        print("  ros2 topic list")
        print("  ros2 topic echo /joint_states")
    else:
        print("\nROS2 桥接创建失败!")
        print("请检查 isaacsim.ros2.bridge 扩展是否正常加载")
    
    print("\n按 Ctrl+C 或关闭窗口退出")
    print("=" * 60 + "\n")
    
    i = 0
    last_print = time.time()
    
    try:
        while simulation_app.is_running():
            world.step(render=True)
            
            # 每 2 秒打印状态
            if time.time() - last_print > 2.0:
                q = robot.get_joint_positions()[:n_dof]
                print(f"[{i:5d}] q: {np.round(q, 2)}")
                last_print = time.time()
            
            i += 1
            
    except KeyboardInterrupt:
        print("\n收到退出信号...")
    
    print("清理...")
    simulation_app.close()
    print("完成!")


if __name__ == '__main__':
    main()
