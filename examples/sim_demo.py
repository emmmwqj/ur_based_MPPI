# -*- coding: utf-8 -*-
"""
UR7e MPC Control using Isaac Lab Framework (Final Fixed Version)
Environment: env_isaaclab
"""

import argparse
import torch
import numpy as np
from typing import Dict

# ==============================================================================
# 1. Isaac Lab App Launcher (MUST be the first import)
# ==============================================================================
from isaaclab.app import AppLauncher

# Parse args to configure app
parser = argparse.ArgumentParser(description="UR7e MPC with Isaac Lab")
parser.add_argument("--headless", action="store_true", default=False)
args = parser.parse_args()

# Launch App
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ==============================================================================
# 2. Imports (After App Launch)
# ==============================================================================
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
# VisualizationMarkers 单独导入，不放在 Scene 配置里
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

# Storm Kit Imports
from storm_kit.util_file import get_assets_path, get_gym_configs_path, join_path
from storm_kit.mpc.task.reacher_task import ReacherTask

# [CRITICAL] Fix Multiprocessing for CUDA
try:
    torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass


# ==============================================================================
# 3. Configuration (Declarative)
# ==============================================================================

@configclass
class UR7eSceneCfg(InteractiveSceneCfg):
    """Configuration for the physical scene (Robot only)."""
    
    # Robot Configuration
    robot = ArticulationCfg(
        prim_path="/World/ur7e",
        spawn=sim_utils.UsdFileCfg(
            usd_path=get_assets_path() + "/urdf/ur7e/ur7e.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0), # [w, x, y, z]
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.57,
                "elbow_joint": 1.57,
                "wrist_1_joint": -1.57,
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.0,
            },
        ),
        actuators={
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=400.0,
                damping=40.0,
            ),
        },
    )
    # [FIX] 移除了 target_marker，因为 Scene 不支持它


# ==============================================================================
# 4. MPC Controller Wrapper
# ==============================================================================

class MPCController:
    def __init__(self, device: str):
        self.device = torch.device(device)
        robot_file = "ur7e_isaacsim.yml"
        task_file = "ur7e_reacher_isaacsim.yml"
        world_file = "collision_primitives_3d.yml"
        
        self.task = ReacherTask(
            task_file, 
            robot_file, 
            world_file, 
            {'device': self.device, 'dtype': torch.float32}
        )
        self.opt_dt = self.task.exp_params['control_dt']

    def get_command(self, t: float, position: torch.Tensor, velocity: torch.Tensor, acceleration: torch.Tensor):
        state = {
            'position': position.cpu().numpy(),
            'velocity': velocity.cpu().numpy(),
            'acceleration': acceleration.cpu().numpy()
        }
        return self.task.get_command(t, state, control_dt=self.opt_dt, WAIT=True)

    def update_goal(self, pos: np.ndarray):
        self.task.update_params(goal_ee_pos=pos)
    
    def close(self):
        self.task.close()


# ==============================================================================
# 5. Main Execution Loop
# ==============================================================================

def main():
    # 1. Setup Simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device if hasattr(args, 'device') else "cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 0.0, 1.0], [0.0, 0.0, 0.5])

    # 2. Create Scene (Physical Entities)
    scene_cfg = UR7eSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # 3. [FIX] Create Markers Manually (Visual Entities)
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Target",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            ),
        },
    )
    target_vis = VisualizationMarkers(marker_cfg)

    # Retrieve Robot Handle
    robot: Articulation = scene["robot"]
    
    # 4. Initialize MPC
    print("[INFO] Initializing MPC Controller...")
    mpc = MPCController(sim.device)

    # 5. Reset Simulation
    sim.reset()
    print("[INFO] Simulation started.")

    # 6. Logic Variables
    target_pos = torch.tensor([0.5, 0.0, 0.5], device=sim.device)
    # Visualize initial target (Require shape: [N, 3])
    target_vis.visualize(target_pos.unsqueeze(0)) 

    t_step = 0.0
    prev_vel = torch.zeros((1, 6), device=sim.device)
    
    # 7. Simulation Loop
    while simulation_app.is_running():
        sim.step()
        
        # State Handling
        q = robot.data.joint_pos[0, :6]
        qd = robot.data.joint_vel[0, :6]
        qdd = (qd - prev_vel[0, :6]) / sim.get_physics_dt()
        prev_vel[0, :6] = qd.clone()

        # MPC Control
        try:
            mpc.update_goal(target_pos.cpu().numpy())
            cmd = mpc.get_command(t_step, q, qd, qdd)
            
            # Apply Action
            q_des = torch.tensor(cmd['position'], device=sim.device).unsqueeze(0)
            
            # Handle extra joints if any
            if robot.num_joints > 6:
                padding = torch.zeros((1, robot.num_joints - 6), device=sim.device)
                q_des = torch.cat([q_des, padding], dim=1)
                
            robot.set_joint_position_target(q_des)
            
            # Write buffers
            scene.write_data_to_sim()
            
            t_step += mpc.opt_dt
            
        except IndexError:
            pass
        except Exception as e:
            print(f"[Warning] MPC Error: {e}")

    mpc.close()

if __name__ == "__main__":
    main()
    simulation_app.close()