#!/usr/bin/env python3
"""
URDF to USD Converter for Isaac Sim 5.1

This script converts URDF files to USD format for use in Isaac Sim.
Run this script using Isaac Sim's Python:
    ./python.sh ~/storm/examples/convert_urdf_to_usd.py

Author: Auto-generated
Date: 2024
"""

import argparse
import os


def convert_urdf_to_usd(urdf_path: str, usd_path: str = None, fix_base: bool = False):
    """
    Convert a URDF file to USD format.
    
    Args:
        urdf_path: Path to the input URDF file
        usd_path: Path for the output USD file (optional, defaults to same dir as URDF)
        fix_base: Whether to fix the base link (default: False for movable objects)
    """
    from isaacsim import SimulationApp
    
    # Start Isaac Sim in headless mode for conversion
    simulation_app = SimulationApp({"headless": True})
    
    # Import after SimulationApp is created
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.asset.importer.urdf")
    
    # Wait for extension to load
    import omni.kit.app
    app = omni.kit.app.get_app()
    for _ in range(10):
        app.update()
    
    import omni.kit.commands
    from pxr import Usd, UsdGeom
    
    # Import URDF extension module
    from isaacsim.asset.importer.urdf import _urdf
    
    # Resolve paths
    urdf_path = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(urdf_path)
    urdf_filename = os.path.basename(urdf_path)
    
    if usd_path is None:
        usd_path = urdf_path.replace('.urdf', '.usd')
    else:
        usd_path = os.path.abspath(usd_path)
    
    print(f"Converting URDF: {urdf_path}")
    print(f"Output USD: {usd_path}")
    
    # Create ImportConfig object (Isaac Sim 5.1 API)
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = fix_base
    import_config.import_inertia_tensor = True
    import_config.self_collision = False
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.default_drive_strength = 1e4
    import_config.default_position_drive_damping = 1e3
    import_config.distance_scale = 1.0
    import_config.density = 0.0
    import_config.make_default_prim = True
    import_config.create_physics_scene = False
    
    # Create URDF interface and import
    urdf_interface = _urdf.acquire_urdf_interface()
    
    try:
        # Parse URDF
        parsed_robot = urdf_interface.parse_urdf(urdf_dir, urdf_filename, import_config)
        
        if parsed_robot is None:
            print(f"✗ Failed to parse URDF: {urdf_path}")
            simulation_app.close()
            return False, None
        
        # Import robot to USD
        prim_path = urdf_interface.import_robot(
            urdf_dir,
            urdf_filename, 
            parsed_robot,
            import_config,
            usd_path,
            False  # getArticulationRoot
        )
        
        if prim_path:
            print(f"✓ Successfully converted to: {usd_path}")
            print(f"  Root prim path: {prim_path}")
            result = True
        else:
            print(f"✗ Failed to import robot")
            result = False
            
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        result = False
        prim_path = None
    
    # Close Isaac Sim
    simulation_app.close()
    
    return result, usd_path


def main():
    parser = argparse.ArgumentParser(description='Convert URDF to USD for Isaac Sim')
    parser.add_argument('--urdf', type=str, default=None, help='Path to URDF file')
    parser.add_argument('--usd', type=str, default=None, help='Output USD path (optional)')
    parser.add_argument('--fix-base', action='store_true', default=False, 
                        help='Fix the base link (for static objects)')
    parser.add_argument('--all-mugs', action='store_true', default=False,
                        help='Convert all mug URDF files in the assets folder')
    args = parser.parse_args()
    
    if args.all_mugs:
        # Convert all mug URDF files in a single session
        convert_all_mugs()
    else:
        if args.urdf is None:
            parser.error("--urdf is required when not using --all-mugs")
        convert_urdf_to_usd(args.urdf, args.usd, args.fix_base)


def convert_all_mugs():
    """Convert all mug URDF files in a single Isaac Sim session"""
    from isaacsim import SimulationApp
    
    # Start Isaac Sim in headless mode
    simulation_app = SimulationApp({"headless": True})
    
    # Import after SimulationApp is created
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.asset.importer.urdf")
    
    # Wait for extension to load
    import omni.kit.app
    app = omni.kit.app.get_app()
    for _ in range(10):
        app.update()
    
    from isaacsim.asset.importer.urdf import _urdf
    
    # Files to convert
    mug_dir = "/home/wqj/storm/content/assets/urdf/mug"
    urdf_files = [
        (os.path.join(mug_dir, "mug.urdf"), True),        # fix_base=True for static ee marker
        (os.path.join(mug_dir, "movable_mug.urdf"), False), # fix_base=False for draggable target
    ]
    
    for urdf_path, fix_base in urdf_files:
        if not os.path.exists(urdf_path):
            print(f"File not found: {urdf_path}")
            continue
            
        print(f"\n{'='*50}")
        urdf_dir = os.path.dirname(urdf_path)
        urdf_filename = os.path.basename(urdf_path)
        usd_path = urdf_path.replace('.urdf', '.usd')
        
        print(f"Converting URDF: {urdf_path}")
        print(f"Output USD: {usd_path}")
        print(f"Fix base: {fix_base}")
        
        # Create ImportConfig object
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.fix_base = fix_base
        import_config.import_inertia_tensor = True
        import_config.self_collision = False
        import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        import_config.default_drive_strength = 1e4
        import_config.default_position_drive_damping = 1e3
        import_config.distance_scale = 1.0
        import_config.density = 0.0
        import_config.make_default_prim = True
        import_config.create_physics_scene = False
        
        # Create URDF interface and import
        urdf_interface = _urdf.acquire_urdf_interface()
        
        try:
            # Parse URDF
            parsed_robot = urdf_interface.parse_urdf(urdf_dir, urdf_filename, import_config)
            
            if parsed_robot is None:
                print(f"✗ Failed to parse URDF")
                continue
            
            # Import robot to USD
            prim_path = urdf_interface.import_robot(
                urdf_dir,
                urdf_filename, 
                parsed_robot,
                import_config,
                usd_path,
                False
            )
            
            if prim_path:
                print(f"✓ Successfully converted to: {usd_path}")
                print(f"  Root prim path: {prim_path}")
            else:
                print(f"✗ Failed to import robot")
                
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Close Isaac Sim
    simulation_app.close()


if __name__ == '__main__':
    main()
