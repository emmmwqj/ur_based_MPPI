#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#

import argparse
import sys
import os
import time
import traceback

# ---------------------------------------------------------------------------
# 1. 启动 Isaac Sim
# ---------------------------------------------------------------------------
from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser(description='Isaac Sim MPC Demo for UR7e')
parser.add_argument('--robot', type=str, default='ur7e', help='Robot to spawn')
parser.add_argument('--headless', action='store_true', default=False, help='headless mode')
parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
args, unknown = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

# ---------------------------------------------------------------------------
# 2. 导入依赖
# ---------------------------------------------------------------------------
import torch
import numpy as np
import yaml

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import RigidPrim
import omni.kit.commands
from pxr import UsdPhysics, Usd, PhysxSchema

# STORM Imports
from storm_kit.util_file import get_gym_configs_path, get_mpc_configs_path, get_assets_path, join_path
from storm_kit.mpc.task.reacher_task import ReacherTask

# Torch Settings
torch.multiprocessing.set_start_method('spawn', force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def log(msg):
    """强制刷新打印"""
    print(msg, flush=True)

def load_robot_asset(urdf_relative_path, prim_path):
    assets_root = get_assets_path()
    
    if "urdf/" in urdf_relative_path:
        full_urdf_path = join_path(assets_root, urdf_relative_path)
    else:
        full_urdf_path = urdf_relative_path

    urdf_dir = os.path.dirname(full_urdf_path)
    urdf_filename = os.path.basename(full_urdf_path)
    usd_filename = urdf_filename.replace(".urdf", ".usd")
    usd_path = os.path.join(urdf_dir, usd_filename)

    log(f"🔎 Checking for USD at: {usd_path}")

    if os.path.exists(usd_path):
        log(f"✅ Found compiled USD. Loading directly...")
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        return True
    else:
        log(f"❌ Error: USD file not found! Expected at: {usd_path}")
        return False

def find_robot_root(stage, root_path):
    log(f"🔍 Scanning for robot physics root under {root_path}...")
    root_prim = stage.GetPrimAtPath(root_path)
    
    if not root_prim.IsValid():
        log(f"❌ Prim invalid: {root_path}")
        return None

    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            log(f"🚀 Found Root (API): {prim.GetPath()}")
            return prim.GetPath().pathString
            
    log("⚠️ No API found. Searching for 'base_link' or 'base'...")
    for prim in Usd.PrimRange(root_prim):
        name = prim.GetName()
        if "base" in name or "link0" in name:
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(prim)
            log(f"🚀 Found Base: {prim.GetPath()}. Force applying ArticulationRoot.")
            UsdPhysics.ArticulationRootAPI.Apply(prim)
            return prim.GetPath().pathString

    log("❌ Fatal: No valid robot link found.")
    return None

def main():
    try:
        # ---------------------------------------------------------------------------
        # 初始化
        # ---------------------------------------------------------------------------
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()

        robot_name = args.robot 
        
        # 1. 读取 Gym 配置
        gym_conf_file = robot_name + '.yml'
        gym_conf_path = join_path(get_gym_configs_path(), gym_conf_file)
        log(f"📂 Loading Gym config: {gym_conf_path}")
        
        if not os.path.exists(gym_conf_path):
            log(f"❌ Gym Config not found: {gym_conf_path}")
            return

        with open(gym_conf_path) as f:
            gym_conf = yaml.safe_load(f)

        # 2. 获取 URDF 路径和初始状态
        sim_urdf_path = gym_conf['sim_params']['sim_urdf'] 
        init_state_list = gym_conf['sim_params']['init_state'] 

        # 3. 加载机器人
        robot_prim_path = "/World/" + robot_name
        if not load_robot_asset(sim_urdf_path, robot_prim_path):
            return

        # 4. 物理刷新
        log("⏳ Physics Step (Expanding USD)...")
        world.step(render=False) 

        # 5. 寻找并修正路径
        stage = omni.usd.get_context().get_stage()
        real_robot_path = find_robot_root(stage, robot_prim_path)

        if not real_robot_path:
            return

        # 6. 注册机器人
        log(f"✅ Robot Path Resolved: {real_robot_path}")
        robot = Robot(prim_path=real_robot_path, name=robot_name)
        world.scene.add(robot)

        # 7. 加载目标物体 (Mug)
        target_mug_path = join_path(get_assets_path(), "urdf/mug/movable_mug.urdf")
        try:
            mug_urdf = target_mug_path
            status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
            import_config.merge_fixed_joints = False
            import_config.fix_base = False
            omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=mug_urdf, import_config=import_config, dest_path="/World/target_mug")
            target_mug = RigidPrim(prim_path="/World/target_mug/movable_mug", name="target_mug")
            world.scene.add(target_mug)
        except Exception as e:
            log(f"⚠️ Warning: Could not load Mug. ({e})")
            target_mug = None

        # 8. 初始化相机
        try:
            from isaacsim.core.utils.viewports import set_camera_view
        except ImportError:
            from omni.isaac.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([2.0, 0.0, 1.5]), target=np.array([0, 0, 0.5]))

        # 9. 初始化 STORM MPC 控制器
        device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
        tensor_args = {'device': device, 'dtype': torch.float32}
        log("🧠 Initializing MPC Controller...")
        
        mpc_control = ReacherTask(robot_name + '_reacher.yml', robot_name + '.yml', "collision_primitives_3d.yml", tensor_args)
        
        log("🔄 Resetting World...")
        world.reset()

        # 10. 设置初始关节角度
        if robot.is_valid():
            n_dof = robot.num_dof 
            log(f"🤖 Robot DOF: {n_dof}")
            
            q_init = np.array(init_state_list)
            if len(q_init) > n_dof: 
                q_init = q_init[:n_dof]
            elif len(q_init) < n_dof:
                q_init = np.pad(q_init, (0, n_dof - len(q_init)))
                
            robot.set_joint_positions(q_init)
        
        # 11. 仿真循环
        from isaacsim.util.debug_draw import _debug_draw
        draw = _debug_draw.acquire_debug_draw_interface()
        
        # 设定目标
        x_des = np.zeros(robot.num_dof * 2) 
        x_des[0:6] = np.array([-0.5, -1.0, 1.5, -1.5, 1.5, 0.0])
        
        mpc_control.update_params(goal_state=x_des)

        t_step = 0
        sim_dt = mpc_control.exp_params['control_dt']
        g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        
        # 动态查找正确的自由度属性名
        control_dofs = 6
        if hasattr(mpc_control, 'n_dofs'):
            control_dofs = mpc_control.n_dofs
        elif hasattr(mpc_control, 'n_dof'):
            control_dofs = mpc_control.n_dof
        elif hasattr(mpc_control, 'num_dof'):
             control_dofs = mpc_control.num_dof
        
        log(f"🚀 Simulation Loop Started! (Controller DOF: {control_dofs})")
        
        while simulation_app.is_running():
            world.step(render=True)
            
            if world.is_playing() and robot.is_valid():
                q_curr = robot.get_joint_positions()
                qd_curr = robot.get_joint_velocities()
                
                if len(q_curr) != control_dofs: 
                    continue

                current_robot_state = {
                    'position': np.array(q_curr),
                    'velocity': np.array(qd_curr),
                    'acceleration': np.zeros_like(q_curr)
                }

                # MPC 计算
                command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
                
                # 打印调试信息 (注意缩进一致)
                if t_step % 0.5 < sim_dt:
                    log(f"✅ Simulating t={t_step:.2f}s | Target joint: {command['position'][0]:.2f}")

                # 执行
                try:
                    robot.apply_action(ArticulationAction(joint_positions=command['position'], joint_velocities=command['velocity']))
                except Exception as e:
                    pass

                # 画线
                draw.clear_lines()
                if hasattr(mpc_control, 'top_trajs'):
                    curr_state_tensor = torch.as_tensor(
                        np.hstack((q_curr, qd_curr, np.zeros_like(q_curr))), 
                        **tensor_args
                    ).unsqueeze(0)
                    
                    curr_ee = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
                    curr_pos = curr_ee['ee_pos_seq'].cpu().numpy().ravel()
                    
                    draw.draw_lines([curr_pos.tolist()], [g_pos.tolist()], [(0, 1, 0, 1)], [2.0])

                t_step += sim_dt

    except Exception as e:
        log(f"❌ CRITICAL ERROR IN MAIN LOOP: {e}")
        traceback.print_exc()
    
    finally:
        simulation_app.close()

if __name__ == '__main__':
    main()