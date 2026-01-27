#!/usr/bin/env python3
"""
UR7e Isaac Sim + ROS2 桥接脚本 (OmniGraph 版本)

使用 Isaac Sim 内置的 OmniGraph ROS2 桥接节点发布/订阅话题。
Isaac Sim 使用内置 Python 3.11 ROS2 库，外部 ROS 节点使用系统 ROS2。
DDS 中间件负责跨 Python 版本的通信。

发布话题:
    /joint_states (sensor_msgs/JointState) - 机器人关节状态
    /target_pose (geometry_msgs/PoseStamped) - 目标位置（红球拖动时）
    
订阅话题:
    /joint_command (sensor_msgs/JointState) - 关节位置指令
    /ee_pose (geometry_msgs/PoseStamped) - 末端位置（用于更新绿球）

可视化:
    - 红色球: 目标位置（可拖动）
    - 绿色球: 末端位置（自动更新）

用法:
    # 终端 1: 运行本脚本 (不要 source 系统 ROS2!)
    cd ~/storm/examples/SIL
    ./run_ur7e_ros2_sim.sh
    
    # 终端 2: 运行 MPC 控制器 (使用系统 ROS2)
    cd ~/storm/examples/SIL
    ./run_ur7e_mpc_ros2.sh

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

# ROS2 消息类型 (用于自定义发布)
try:
    from std_msgs.msg import Header
    from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    ROS2_AVAILABLE = True
    print("ROS2 Python 库可用")
except ImportError:
    ROS2_AVAILABLE = False
    print("警告: ROS2 Python 库不可用，目标/末端位置话题将不工作")

# STORM 路径
sys.path.insert(0, '/home/wqj/storm')
from storm_kit.util_file import get_gym_configs_path, join_path, get_assets_path


# ============================================================================
# 辅助函数
# ============================================================================

def transform_point(position, orientation_xyzw, point):
    """将点从机器人坐标系变换到世界坐标系"""
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat(orientation_xyzw)
    return rot.apply(point) + position

def inv_transform_point(position, orientation_xyzw, point):
    """将点从世界坐标系变换到机器人坐标系"""
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat(orientation_xyzw).inv()
    return rot.apply(point - position)


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
# ROS2 目标/末端位置通信节点
# ============================================================================

class MarkerROS2Node:
    """
    处理目标球和末端球位置的 ROS2 通信
    
    发布:
        /target_pose - 当红色目标球被拖动时发布新位置
    
    订阅:
        /ee_pose - 从 MPC 接收末端位置，更新绿球
        /initial_target_pose - 从 MPC 接收初始目标位置，更新红球
    """
    
    def __init__(self):
        self.node = None
        self.target_pub = None
        self.ee_sub = None
        self.initial_target_sub = None
        self.ee_pose_received = None
        self.initial_target_received = None
        self._initialized = False
        
    def init(self):
        """初始化 ROS2 节点"""
        if not ROS2_AVAILABLE:
            print("警告: ROS2 不可用，标记位置话题将不工作")
            return False
            
        try:
            if not rclpy.ok():
                rclpy.init()
            
            self.node = rclpy.create_node('isaac_sim_markers')
            qos = QoSProfile(depth=10)
            
            # 发布目标位置
            self.target_pub = self.node.create_publisher(
                PoseStamped, '/target_pose', qos
            )
            
            # 订阅末端位置
            self.ee_sub = self.node.create_subscription(
                PoseStamped, '/ee_pose', self._ee_callback, qos
            )
            
            # 订阅初始目标位置 (从 MPC 端设置红球位置)
            self.initial_target_sub = self.node.create_subscription(
                PoseStamped, '/initial_target_pose', self._initial_target_callback, qos
            )
            
            self._initialized = True
            print("  ROS2 标记节点已初始化")
            print("    发布: /target_pose (目标位置)")
            print("    订阅: /ee_pose (末端位置)")
            print("    订阅: /initial_target_pose (初始目标位置)")
            return True
            
        except Exception as e:
            print(f"警告: ROS2 标记节点初始化失败: {e}")
            return False
    
    def _ee_callback(self, msg: PoseStamped):
        """末端位置回调"""
        self.ee_pose_received = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
    
    def _initial_target_callback(self, msg: PoseStamped):
        """初始目标位置回调 (从 MPC 设置红球位置)"""
        self.initial_target_received = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
    
    def publish_target(self, position: np.ndarray):
        """发布目标位置"""
        if not self._initialized or self.target_pub is None:
            print(f"警告: ROS2 未初始化，无法发布目标位置")
            return
            
        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.w = 1.0
        
        self.target_pub.publish(msg)
        print(f"  已发布目标到 /target_pose: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
    
    def get_ee_pose(self) -> np.ndarray:
        """获取接收到的末端位置"""
        return self.ee_pose_received
    
    def get_initial_target(self) -> np.ndarray:
        """获取从 MPC 接收的初始目标位置 (只获取一次)"""
        if self.initial_target_received is not None:
            pos = self.initial_target_received.copy()
            self.initial_target_received = None  # 清除，只用一次
            return pos
        return None
    
    def spin_once(self):
        """处理一次 ROS2 回调"""
        if self._initialized and self.node is not None:
            rclpy.spin_once(self.node, timeout_sec=0.001)
    
    def shutdown(self):
        """关闭节点"""
        if self._initialized:
            try:
                self.node.destroy_node()
            except:
                pass


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
    
    # ========================================================================
    # 可视化标记 - 使用 STORM 正运动学计算位置
    # ========================================================================
    
    # 导入 STORM 正运动学
    import torch
    from storm_kit.mpc.task.reacher_task import ReacherTask
    
    print("\n初始化 STORM 正运动学...")
    tensor_args = {'device': torch.device('cuda', 0), 'dtype': torch.float32}
    task_file = 'ur7e_reacher_isaacsim.yml'
    
    # 创建 MPC 实例获取正运动学和目标位置
    mpc_fk = ReacherTask(task_file, robot_file, world_file, tensor_args)
    
    # 获取目标末端位置 (与 MPC 控制器一致)
    goal_state = np.array([0.5, -1.2, 1.2, -1.57, -1.57, 0.0, 0, 0, 0, 0, 0, 0])
    mpc_fk.update_params(goal_state=goal_state)
    goal_ee_pos_robot = np.ravel(mpc_fk.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    goal_ee_world = transform_point(robot_pos, robot_quat_xyzw, goal_ee_pos_robot)
    print(f"  目标末端位置 (机器人): {goal_ee_pos_robot}")
    print(f"  目标末端位置 (世界): {goal_ee_world}")
    
    # 保存 rollout_fn 用于计算末端位置
    rollout_fn = mpc_fk.controller.rollout_fn
    
    # 目标标记（红色 - 可拖动）
    goal_marker = world.scene.add(VisualSphere(
        prim_path="/World/Markers/Goal",
        name="goal_marker",
        position=goal_ee_world,
        radius=0.03,
        color=np.array([0.9, 0.1, 0.1])  # 红色
    ))
    print(f"  目标标记 (红色): {goal_ee_world}")
    
    # 末端标记（绿色 - 显示实际末端位置）
    ee_marker = world.scene.add(VisualSphere(
        prim_path="/World/Markers/EE",
        name="ee_marker",
        position=goal_ee_world,  # 初始放在目标位置附近
        radius=0.025,
        color=np.array([0.1, 0.9, 0.1])  # 绿色
    ))
    print("  末端标记 (绿色): 通过正运动学实时计算")
    
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
    # 初始化 ROS2 标记节点
    # ========================================================================
    
    marker_node = MarkerROS2Node()
    marker_ros2_ok = marker_node.init()
    
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
    
    if marker_ros2_ok:
        print("\nROS2 话题 (标记位置):")
        print("  发布: /target_pose (目标位置 - 拖动红球更新)")
        print("  订阅: /ee_pose (末端位置 - 更新绿球)")
    
    print("\n提示:")
    print("  - 拖动红色球来动态设置目标位置")
    print("  - 绿色球显示机械臂实际末端位置")
    print("\n在另一个终端启动 MPC 控制器:")
    print("  cd ~/storm/examples/SIL && ./run_ur7e_mpc_ros2.sh")
    
    print("\n按 Ctrl+C 或关闭窗口退出")
    print("=" * 60 + "\n")
    
    i = 0
    last_print = time.time()
    
    # 用于检测目标球被拖动
    current_goal_world = goal_ee_world.copy()
    
    try:
        while simulation_app.is_running():
            world.step(render=True)
            
            # 处理 ROS2 回调
            if marker_ros2_ok:
                marker_node.spin_once()
            
            # --- 使用正运动学计算并更新末端位置 (绿球) ---
            q = robot.get_joint_positions()[:n_dof]
            dq = robot.get_joint_velocities()[:n_dof]
            ddq = np.zeros(n_dof)
            state_tensor = torch.as_tensor(
                np.hstack([q, dq, ddq]), **tensor_args
            ).unsqueeze(0)
            
            ee_pose = rollout_fn.get_ee_pose(state_tensor)
            ee_pos_robot = np.ravel(ee_pose['ee_pos_seq'].cpu().numpy())
            ee_pos_world = transform_point(robot_pos, robot_quat_xyzw, ee_pos_robot)
            ee_marker.set_world_pose(position=ee_pos_world)
            
            # --- 检测目标球是否被拖动 ---
            goal_world_new, _ = goal_marker.get_world_pose()
            if np.linalg.norm(goal_world_new - current_goal_world) > 0.003:  # 移动超过3mm
                current_goal_world = goal_world_new.copy()
                # 发布新目标位置
                if marker_ros2_ok:
                    marker_node.publish_target(current_goal_world)
                goal_robot = inv_transform_point(robot_pos, robot_quat_xyzw, current_goal_world)
                print(f"[目标更新] 世界: {np.round(current_goal_world, 3)}, 机器人: {np.round(goal_robot, 3)}")
            
            # 每 2 秒打印状态
            if time.time() - last_print > 2.0:
                print(f"[{i:5d}] q: {np.round(q, 2)}, ee: {np.round(ee_pos_world, 3)}")
                last_print = time.time()
            
            i += 1
            
    except KeyboardInterrupt:
        print("\n收到退出信号...")
    
    print("清理...")
    if marker_ros2_ok:
        marker_node.shutdown()
    mpc_fk.close()
    simulation_app.close()
    print("完成!")


if __name__ == '__main__':
    main()
