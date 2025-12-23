import numpy as np
import torch
import yaml
import os
from omni.isaac.kit import SimulationApp

# 1. 启动仿真
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import VisualSphere
from storm_kit.differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel

def visualize():
    world = World()
    world.scene.add_default_ground_plane()
    
    robot_name = "ur7e"
    storm_root = os.path.expanduser("~/storm")
    urdf_path = os.path.join(storm_root, "content/assets/urdf/ur7e/ur7e.urdf")
    robot_yml = os.path.join(storm_root, "content/configs/robot/ur7e.yml")
    
    with open(robot_yml) as f:
        robot_params = yaml.safe_load(f)
    
    # 2. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    robot_model = DifferentiableRobotModel(urdf_path, name=robot_name)
    robot_model.to(device)
    
    # 3. 预创建 USD 球体
    coll_spheres = robot_params['collision_spheres']
    usd_spheres = []
    sphere_info = []

    print("🏗️ 正在创建 USD 碰撞球模型...")
    counter = 0
    for link_name, spheres in coll_spheres.items():
        for i, s in enumerate(spheres):
            # 创建物理可见的球体 prim
            sp = VisualSphere(
                prim_path=f"/World/SphereVisual/sp_{counter}",
                name=f"sphere_{counter}",
                radius=s['radius'],
                color=np.array([0, 1, 0]) # 绿色
            )
            usd_spheres.append(sp)
            sphere_info.append({'link': link_name, 'center': s['center']})
            counter += 1

    # 4. 设置姿态：完全伸直 [0,0,0,0,0,0]
    q = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    
    print(f"🚀 成功创建 {len(usd_spheres)} 个球体。正在实时更新位置...")

    while simulation_app.is_running():
        # 5. 更新所有球体位置
        with torch.no_grad():
            for i, info in enumerate(sphere_info):
                link_name = info['link']
                # 获取该连杆的位姿
                res = robot_model.compute_forward_kinematics(q, q, link_name)
                pos = res[0].squeeze() 
                rot = res[1].squeeze()
                
                # 局部坐标转世界坐标
                center_local = torch.tensor(info['center'], device=device, dtype=torch.float32)
                center_world = rot.matmul(center_local) + pos
                
                # 更新 USD 属性
                usd_spheres[i].set_world_pose(position=center_world.cpu().numpy())

        world.step(render=True)

if __name__ == '__main__':
    visualize()
