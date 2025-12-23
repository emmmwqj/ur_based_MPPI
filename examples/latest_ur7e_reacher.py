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
"""
UR7e Robot MPC Control in Isaac Sim 5.1
Based on the Franka example, adapted for UR7e robot

Author: Auto-generated
Date: 2024
"""

import argparse
import copy
import time
import yaml
import numpy as np

# Global variable for simulation app
simulation_app = None

np.set_printoptions(precision=2)


class IsaacSimRobotWrapper:
    """Wrapper class to interface with UR7e robot in Isaac Sim 5.1"""
    
    def __init__(self, world, robot_prim_path, robot_name, n_dof=6, device='cuda'):
        self.world = world
        self.robot_prim_path = robot_prim_path
        self.robot_name = robot_name
        self.device = device
        self.robot = None
        self.n_dof = n_dof  # UR7e has 6 DOF
        self.prev_joint_velocities = None
        self.prev_time = None
        self._articulation_controller = None
        
    def initialize(self):
        """Initialize the robot articulation after world is reset"""
        self.robot = self.world.scene.get_object(self.robot_name)
        if self.robot is None:
            raise RuntimeError(f"Robot '{self.robot_name}' not found in scene")
        
        # Get articulation controller for sending commands
        self._articulation_controller = self.robot.get_articulation_controller()
        
        self.prev_joint_velocities = np.zeros(self.n_dof)
        self.prev_time = time.time()
        
    def get_state(self):
        """Get current robot state similar to Isaac Gym format"""
        joint_positions = self.robot.get_joint_positions()[:self.n_dof]
        joint_velocities = self.robot.get_joint_velocities()[:self.n_dof]
        
        # Estimate acceleration from velocity difference
        current_time = time.time()
        dt = current_time - self.prev_time if self.prev_time else 0.01
        dt = max(dt, 0.001)  # Prevent division by zero
        
        joint_accelerations = (joint_velocities - self.prev_joint_velocities) / dt
        
        self.prev_joint_velocities = joint_velocities.copy()
        self.prev_time = current_time
        
        # UR7e joint names
        joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        return {
            'name': joint_names[:self.n_dof],
            'position': joint_positions,
            'velocity': joint_velocities,
            'acceleration': joint_accelerations
        }
    
    def command_robot_position(self, q_des):
        """Send position command to robot using ArticulationAction"""
        from isaacsim.core.utils.types import ArticulationAction
        
        # UR7e has 6 DOF, no gripper to pad
        num_dof = self.robot.num_dof
        if len(q_des) < num_dof:
            q_des_full = np.zeros(num_dof)
            q_des_full[:len(q_des)] = q_des
        else:
            q_des_full = q_des[:num_dof]
        
        # Use ArticulationAction to apply position targets
        action = ArticulationAction(joint_positions=q_des_full)
        self._articulation_controller.apply_action(action)
        
    def set_robot_state(self, q_des, qd_des):
        """Directly set robot joint state"""
        num_dof = self.robot.num_dof
        
        if len(q_des) < num_dof:
            q_des_full = np.zeros(num_dof)
            q_des_full[:len(q_des)] = q_des
        else:
            q_des_full = q_des[:num_dof]
            
        if len(qd_des) < num_dof:
            qd_des_full = np.zeros(num_dof)
            qd_des_full[:len(qd_des)] = qd_des
        else:
            qd_des_full = qd_des[:num_dof]
            
        self.robot.set_joint_positions(q_des_full)
        self.robot.set_joint_velocities(qd_des_full)


class Transform:
    """Simple transform class to mimic Isaac Gym's Transform"""
    def __init__(self, p=None, r=None):
        # p: position [x, y, z]
        # r: quaternion [x, y, z, w] (xyzw format)
        self.p = np.array(p) if p is not None else np.array([0.0, 0.0, 0.0])
        self.r = np.array(r) if r is not None else np.array([0.0, 0.0, 0.0, 1.0])
        
    def inverse(self):
        """Return inverse transform"""
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(self.r)  # scipy uses xyzw
        rot_inv = rot.inv()
        p_inv = -rot_inv.apply(self.p)
        return Transform(p=p_inv, r=rot_inv.as_quat())
    
    def __mul__(self, other):
        """Compose transforms"""
        from scipy.spatial.transform import Rotation
        rot1 = Rotation.from_quat(self.r)
        rot2 = Rotation.from_quat(other.r)
        
        new_rot = rot1 * rot2
        new_p = rot1.apply(other.p) + self.p
        
        return Transform(p=new_p, r=new_rot.as_quat())
    
    def transform_point(self, point):
        """Transform a point from local to world frame"""
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(self.r)
        return rot.apply(point) + self.p


class IsaacSimWorld:
    """World class for managing obstacles and objects in Isaac Sim"""
    
    def __init__(self, world, world_params, w_T_r=None):
        self.world = world
        self.world_params = world_params
        self.w_T_r = w_T_r
        self.objects = {}
        
    def spawn_primitives(self):
        """Spawn collision primitives from world params"""
        from isaacsim.core.api.objects import VisualCuboid, VisualSphere
        
        if self.world_params is None:
            return
            
        world_model = self.world_params.get('world_model', {})
        coll_objs = world_model.get('coll_objs', {})
        
        # Spawn spheres
        spheres = coll_objs.get('sphere', {})
        for name, params in spheres.items():
            radius = params.get('radius', 0.1)
            position = np.array(params.get('position', [0, 0, 0]))
            
            # Transform position by robot base
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
            
        # Spawn cubes
        cubes = coll_objs.get('cube', {})
        for name, params in cubes.items():
            dims = np.array(params.get('dims', [0.1, 0.1, 0.1]))
            pose = params.get('pose', [0, 0, 0, 0, 0, 0, 1])
            position = np.array(pose[:3])
            orientation_xyzw = np.array(pose[3:])
            
            # Transform position by robot base
            if self.w_T_r is not None:
                position = self.w_T_r.transform_point(position)
                
            # Convert to wxyz for Isaac Sim
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
            
    def spawn_target_marker(self, name, position, orientation=None, color=None):
        """Spawn a visual marker for the target"""
        from isaacsim.core.api.objects import VisualSphere
        
        if color is None:
            color = np.array([0.8, 0.1, 0.1])
        
        position = np.array(position)
            
        marker = VisualSphere(
            prim_path=f"/World/markers/{name}",
            name=name,
            position=position,
            radius=0.03,
            color=color
        )
        self.world.scene.add(marker)
        self.objects[name] = marker
        return marker
        
    def update_marker_pose(self, name, position, orientation=None):
        """Update marker position"""
        if name in self.objects:
            marker = self.objects[name]
            marker.set_world_pose(position=np.array(position))
            
    def get_pose(self, name):
        """Get pose of an object"""
        if name in self.objects:
            obj = self.objects[name]
            pos, rot_wxyz = obj.get_world_pose()
            # Convert wxyz to xyzw
            rot_xyzw = np.array([rot_wxyz[1], rot_wxyz[2], rot_wxyz[3], rot_wxyz[0]])
            return Transform(p=pos, r=rot_xyzw)
        return None


def mpc_robot_interactive(args):
    """Main function for MPC robot control in Isaac Sim 5.1"""
    
    # Import global simulation_app and required modules
    global simulation_app
    import torch
    from isaacsim.core.api.world import World
    from isaacsim.core.api.objects import VisualCuboid, VisualSphere
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.nucleus import get_assets_root_path
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    import omni.timeline
    import carb
    
    from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
    from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
    from storm_kit.mpc.task.reacher_task import ReacherTask
    
    print("=" * 50)
    print("Starting UR7e MPC Robot Interactive")
    print("=" * 50)
    
    vis_ee_target = True
    robot_file = args.robot + '_isaacsim.yml'  # Use Isaac Sim specific config
    task_file = args.robot + '_reacher_isaacsim.yml'  # Use Isaac Sim specific task config
    world_file = 'collision_primitives_3d.yml'
    
    print(f"Using robot config: {robot_file}")
    print(f"Using task config: {task_file}")
    
    # Load configurations
    print("Loading configuration files...")
    world_yml = join_path(get_gym_configs_path(), world_file)
    with open(world_yml) as file:
        world_params = yaml.safe_load(file)
        
    # Use Isaac Sim specific robot config
    robot_yml = join_path(get_gym_configs_path(), robot_file)
    print(f"Loading robot config from: {robot_yml}")
    with open(robot_yml) as file:
        robot_params = yaml.safe_load(file)
        
    sim_params = robot_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Create Isaac Sim World using Core API
    print("Creating Isaac Sim World...")
    world = World(stage_units_in_meters=1.0)
    
    # Add ground plane
    print("Adding ground plane...")
    world.scene.add_default_ground_plane()
    
    # Get robot pose from config
    robot_pose_cfg = sim_params.get('robot_pose', [0, 0, 0, 0, 0, 0, 1])
    robot_position = np.array(robot_pose_cfg[:3])
    robot_orientation_xyzw = np.array(robot_pose_cfg[3:])
    # Convert to wxyz for Isaac Sim
    robot_orientation_wxyz = np.array([
        robot_orientation_xyzw[3],
        robot_orientation_xyzw[0], 
        robot_orientation_xyzw[1],
        robot_orientation_xyzw[2]
    ])
    
    print(f"Robot position: {robot_position}")
    print(f"Robot orientation (wxyz): {robot_orientation_wxyz}")
    
    # Create transform for robot base (xyzw format for internal use)
    w_T_r = Transform(p=robot_position, r=robot_orientation_xyzw)
    
    # Load UR7e robot from local USD file
    print("Loading UR7e robot...")
    
    # Use local USD file from STORM assets
    ur7e_usd_path = get_assets_path() + "/urdf/ur7e/ur7e.usd"
    print(f"Loading robot from: {ur7e_usd_path}")
    
    ur7e_prim_path = "/World/UR7e"
    
    # Add robot to stage from local USD
    print("Adding robot to stage...")
    add_reference_to_stage(usd_path=ur7e_usd_path, prim_path=ur7e_prim_path)
    
    # Create Robot object and add to scene
    print("Creating Robot object...")
    ur7e_robot = world.scene.add(
        Robot(
            prim_path=ur7e_prim_path,
            name="ur7e_robot",
            position=robot_position,
            orientation=robot_orientation_wxyz
        )
    )
    
    # Reset world once to initialize robot articulation
    print("Performing initial world reset to configure robot...")
    world.reset()
    
    # Force set the robot pose after reset
    print("Setting robot pose explicitly...")
    ur7e_robot.set_world_pose(position=robot_position, orientation=robot_orientation_wxyz)
    
    # Step simulation to apply pose
    for _ in range(3):
        world.step(render=True)
    
    # Configure joint drive properties for position control
    print("Configuring joint drive properties...")
    from pxr import UsdPhysics, PhysxSchema
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    
    # Set joint drive stiffness and damping for position control
    joint_drive_stiffness = 400.0
    joint_drive_damping = 40.0
    
    # Iterate over all prims to find joint drives
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.DriveAPI):
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                drive.GetStiffnessAttr().Set(joint_drive_stiffness)
                drive.GetDampingAttr().Set(joint_drive_damping)
                    
    print(f"Joint drive configured: stiffness={joint_drive_stiffness}, damping={joint_drive_damping}")
    
    # UR7e has 6 DOF
    n_dof = 6
    
    # Create robot wrapper
    robot_sim = IsaacSimRobotWrapper(world, ur7e_prim_path, "ur7e_robot", n_dof=n_dof, device=device)
    
    # Setup tensor args
    torch_device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    tensor_args = {'device': torch_device, 'dtype': torch.float32}
    
    # Create world instance for obstacles
    print("Creating obstacles...")
    world_instance = IsaacSimWorld(world, world_params, w_T_r=w_T_r)
    world_instance.spawn_primitives()
    
    # Create MPC controller
    print("Creating MPC controller...")
    mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)
    
    n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
    print(f"Robot DOF: {n_dof}")
    
    # Setup initial state
    mpc_tensor_dtype = {'device': torch_device, 'dtype': torch.float32}
    
    # Goal state for UR7e: first 6 values are TARGET JOINT ANGLES (radians)
    # The system uses forward kinematics to compute the end-effector target position
    # UR7e init_state: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    # Let's set a goal that moves the arm to a reachable position
    # These joint angles should put the end-effector in front of the robot at a reasonable height
    ur7e_goal_state = np.array([
        0.5,      # shoulder_pan: rotate slightly to the side
        -1.2,     # shoulder_lift: raise arm up (less negative = higher)
        1.2,      # elbow: bend elbow
        -1.57,    # wrist_1: keep wrist orientation
        -1.57,    # wrist_2: keep wrist orientation
        0.0,      # wrist_3: keep wrist orientation
        # velocities (6 zeros)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])
    x_des_list = [ur7e_goal_state]
    x_des = x_des_list[0]
    
    mpc_control.update_params(goal_state=x_des)
    
    # Get goal pose
    g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
    print(f"Goal position: {g_pos}")
    print(f"Goal quaternion: {g_q}")
    
    # Setup coordinate transform for trajectory visualization
    w_T_robot = torch.eye(4)
    w_T_robot[0, 3] = robot_position[0]
    w_T_robot[1, 3] = robot_position[1]
    w_T_robot[2, 3] = robot_position[2]
    # quaternion_to_matrix expects [w, x, y, z] format
    quat_wxyz = torch.tensor([robot_orientation_wxyz[0], robot_orientation_wxyz[1], 
                              robot_orientation_wxyz[2], robot_orientation_wxyz[3]]).unsqueeze(0)
    rot = quaternion_to_matrix(quat_wxyz)
    w_T_robot[:3, :3] = rot[0]
    
    w_robot_coord = CoordinateTransform(
        trans=w_T_robot[0:3, 3].unsqueeze(0),
        rot=w_T_robot[0:3, 0:3].unsqueeze(0)
    )
    
    # Control parameters
    sim_dt = mpc_control.exp_params['control_dt']
    print(f"Control dt: {sim_dt}")
    
    # Initialize robot wrapper (world was already reset during robot configuration)
    print("Initializing robot wrapper...")
    robot_sim.initialize()
    
    # Set initial joint configuration
    init_state = sim_params.get('init_state', [0.0] * n_dof)
    init_positions = np.array(init_state[:n_dof])
    print(f"Initial joint positions: {init_positions}")
    
    # Get actual robot DOF
    num_robot_dof = ur7e_robot.num_dof
    print(f"Total robot DOF: {num_robot_dof}")
    
    if num_robot_dof is None:
        print("WARNING: num_dof is None, using default 6 DOF")
        num_robot_dof = 6
        
    if num_robot_dof > n_dof:
        init_positions_full = np.zeros(num_robot_dof)
        init_positions_full[:n_dof] = init_positions
    else:
        init_positions_full = init_positions[:num_robot_dof]
    
    ur7e_robot.set_joint_positions(init_positions_full)
    ur7e_robot.set_joint_velocities(np.zeros(num_robot_dof))
    
    # Force set robot pose again after joint initialization
    print("Re-setting robot pose after joint initialization...")
    ur7e_robot.set_world_pose(position=robot_position, orientation=robot_orientation_wxyz)
    
    # Debug: Print actual robot pose after initialization
    robot_actual_pos, robot_actual_ori = ur7e_robot.get_world_pose()
    print(f"Expected robot position: {robot_position}")
    print(f"Expected robot orientation (wxyz): {robot_orientation_wxyz}")
    print(f"Robot actual position after init: {robot_actual_pos}")
    print(f"Robot actual orientation (wxyz) after init: {robot_actual_ori}")
    
    # Step the world a few times to let the robot settle
    for _ in range(5):
        world.step(render=True)
    
    # Final pose check
    robot_actual_pos, robot_actual_ori = ur7e_robot.get_world_pose()
    print(f"Final robot position: {robot_actual_pos}")
    print(f"Final robot orientation (wxyz): {robot_actual_ori}")
    
    # Create target markers after world reset
    if vis_ee_target:
        print("Creating target markers...")
        # Transform goal to world frame
        goal_pos_world = w_T_r.transform_point(g_pos)
        world_instance.spawn_target_marker(
            "ee_target",
            position=goal_pos_world,
            color=np.array([0.8, 0.1, 0.1])
        )
        world_instance.spawn_target_marker(
            "ee_current",
            position=[0, 0, 0],
            color=np.array([0.1, 0.8, 0.1])
        )
    
    # Main loop variables
    t_step = 0.0
    i = 0
    
    print("=" * 50)
    print("Starting MPC control loop...")
    print("Press Ctrl+C to exit")
    print("=" * 50)
    
    # Start the timeline (physics simulation)
    omni.timeline.get_timeline_interface().play()
    
    # Give the simulation a moment to start
    for _ in range(10):
        world.step(render=True)
    
    # Initialize MPC by getting first state and warming up
    current_robot_state = robot_sim.get_state()
    
    # First call to MPC to initialize - use WAIT=False for first few iterations
    print("Warming up MPC controller...")
    for warmup_i in range(5):
        try:
            _ = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=False)
            t_step += sim_dt
            world.step(render=True)
            current_robot_state = robot_sim.get_state()
        except Exception as e:
            print(f"Warmup iteration {warmup_i}: {e}")
            t_step += sim_dt
            world.step(render=True)
    
    print("MPC warmup complete, starting main control loop...")
    
    while simulation_app.is_running():
        try:
            # Step the simulation
            world.step(render=True)
            
            if not world.is_playing():
                if world.is_stopped():
                    world.reset()
                    robot_sim.initialize()
                world.step(render=True)
                continue
                
            t_step += sim_dt
            
            # Get current robot state
            current_robot_state = robot_sim.get_state()
            
            # Get MPC command
            try:
                command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)
            except IndexError as e:
                # MPC timing issue, skip this iteration
                if i % 50 == 0:
                    print(f"MPC timing skip at t={t_step:.3f}")
                i += 1
                continue
            
            filtered_state_mpc = current_robot_state
            curr_state = np.hstack((
                filtered_state_mpc['position'],
                filtered_state_mpc['velocity'],
                filtered_state_mpc['acceleration']
            ))
            
            curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
            
            # Get position command
            q_des = copy.deepcopy(command['position'])
            qd_des = copy.deepcopy(command['velocity'])
            qdd_des = copy.deepcopy(command['acceleration'])
            
            # Calculate error
            ee_error = mpc_control.get_current_error(filtered_state_mpc)
            
            # Get current end-effector pose
            pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
            e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
            e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
            
            # Update current EE marker
            if vis_ee_target:
                # Transform to world frame
                ee_pos_world = w_T_r.transform_point(e_pos)
                world_instance.update_marker_pose("ee_current", ee_pos_world)
            
            # Print status every 10 iterations
            if i % 10 == 0:
                print(f"[{i}] Error: {['{:.3f}'.format(x) for x in ee_error]}, "
                      f"opt_dt: {mpc_control.opt_dt:.3f}, mpc_dt: {mpc_control.mpc_dt:.3f}")
            
            # Send command to robot
            robot_sim.command_robot_position(q_des)
            
            i += 1
            
        except KeyboardInterrupt:
            print('Closing...')
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            # Don't break on first error, try to continue
            i += 1
            continue
    
    # Cleanup
    print("Cleaning up...")
    mpc_control.close()
    simulation_app.close()
    
    return 1


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='UR7e Reacher MPC Control in Isaac Sim 5.1')
    parser.add_argument('--robot', type=str, default='ur7e', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless mode')
    parser.add_argument('--control_space', type=str, default='acc', help='Control space')
    args = parser.parse_args()
    
    # Launch Isaac Sim - MUST be inside __main__ to prevent multiprocessing issues
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless})
    
    # Now import Isaac Sim modules after SimulationApp is created
    import torch
    # Only set start method if not already set
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Isaac Sim 5.1 Core API imports (based on official docs)
    from isaacsim.core.api.world import World
    from isaacsim.core.api.objects import VisualCuboid, VisualSphere
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.nucleus import get_assets_root_path
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    import omni.timeline
    import carb
    
    # STORM imports
    from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
    from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
    from storm_kit.mpc.task.reacher_task import ReacherTask
    
    try:
        mpc_robot_interactive(args)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
