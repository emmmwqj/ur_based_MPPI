import time

from gazebo_msgs.srv import DeleteEntity, SpawnEntity


def iter_primitive_obstacles(world_params: dict, include_ground: bool = False):
    coll_objs = world_params.get("world_model", {}).get("coll_objs", {})

    for name, params in coll_objs.get("sphere", {}).items():
        if not include_ground and name == "ground":
            continue
        yield {
            "kind": "sphere",
            "name": name,
            "radius": float(params.get("radius", 0.1)),
            "position": [float(v) for v in params.get("position", [0.0, 0.0, 0.0])],
        }

    for name, params in coll_objs.get("cube", {}).items():
        if not include_ground and name == "ground":
            continue
        yield {
            "kind": "cube",
            "name": name,
            "dims": [float(v) for v in params.get("dims", [0.1, 0.1, 0.1])],
            "pose": [float(v) for v in params.get("pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
        }


def count_primitive_obstacles(world_params: dict, include_ground: bool = False) -> tuple[int, int]:
    n_spheres = 0
    n_cubes = 0
    for obstacle in iter_primitive_obstacles(world_params, include_ground=include_ground):
        if obstacle["kind"] == "sphere":
            n_spheres += 1
        elif obstacle["kind"] == "cube":
            n_cubes += 1
    return n_spheres, n_cubes


def _wait_for_service(client, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if client.wait_for_service(timeout_sec=0.5):
            return True
    return False


def _wait_for_future(future, timeout_sec: float):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if future.done():
            break
        time.sleep(0.05)

    if not future.done():
        return None

    if future.exception() is not None:
        raise future.exception()

    return future.result()


def _build_box_sdf(model_name: str, dims: list[float]) -> str:
    return f"""<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='{model_name}'>
    <static>true</static>
    <link name='link'>
      <collision name='collision'>
        <geometry>
          <box>
            <size>{dims[0]} {dims[1]} {dims[2]}</size>
          </box>
        </geometry>
      </collision>
      <visual name='visual'>
        <geometry>
          <box>
            <size>{dims[0]} {dims[1]} {dims[2]}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.8 1.0</ambient>
          <diffuse>0.5 0.5 0.8 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


def _build_sphere_sdf(model_name: str, radius: float) -> str:
    return f"""<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='{model_name}'>
    <static>true</static>
    <link name='link'>
      <collision name='collision'>
        <geometry>
          <sphere>
            <radius>{radius}</radius>
          </sphere>
        </geometry>
      </collision>
      <visual name='visual'>
        <geometry>
          <sphere>
            <radius>{radius}</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>0.8 0.2 0.2 1.0</ambient>
          <diffuse>0.8 0.2 0.2 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


def spawn_gazebo_obstacles(
    node,
    world_params: dict,
    model_prefix: str = "sim_tall",
    include_ground: bool = False,
    service_timeout_sec: float = 8.0,
) -> bool:
    spawn_client = node.create_client(SpawnEntity, "/spawn_entity")
    delete_client = node.create_client(DeleteEntity, "/delete_entity")

    if not _wait_for_service(spawn_client, service_timeout_sec):
        node.get_logger().error("Gazebo /spawn_entity 服务不可用，未能生成真实障碍物")
        return False
    if not _wait_for_service(delete_client, service_timeout_sec):
        node.get_logger().error("Gazebo /delete_entity 服务不可用，未能刷新真实障碍物")
        return False

    all_ok = True
    for obstacle in iter_primitive_obstacles(world_params, include_ground=include_ground):
        model_name = f"{model_prefix}_{obstacle['name']}"

        delete_req = DeleteEntity.Request()
        delete_req.name = model_name
        try:
            _wait_for_future(delete_client.call_async(delete_req), timeout_sec=2.0)
        except Exception:
            pass

        spawn_req = SpawnEntity.Request()
        spawn_req.name = model_name
        spawn_req.robot_namespace = model_name
        spawn_req.reference_frame = "world"

        if obstacle["kind"] == "sphere":
            spawn_req.xml = _build_sphere_sdf(model_name, obstacle["radius"])
            pos = obstacle["position"]
            spawn_req.initial_pose.position.x = pos[0]
            spawn_req.initial_pose.position.y = pos[1]
            spawn_req.initial_pose.position.z = pos[2]
            spawn_req.initial_pose.orientation.w = 1.0
        else:
            spawn_req.xml = _build_box_sdf(model_name, obstacle["dims"])
            pose = obstacle["pose"]
            spawn_req.initial_pose.position.x = pose[0]
            spawn_req.initial_pose.position.y = pose[1]
            spawn_req.initial_pose.position.z = pose[2]
            spawn_req.initial_pose.orientation.x = pose[3]
            spawn_req.initial_pose.orientation.y = pose[4]
            spawn_req.initial_pose.orientation.z = pose[5]
            spawn_req.initial_pose.orientation.w = pose[6]

        try:
            result = _wait_for_future(spawn_client.call_async(spawn_req), timeout_sec=service_timeout_sec)
        except Exception as exc:
            node.get_logger().error(f"生成 Gazebo 障碍物失败: {model_name}: {exc}")
            all_ok = False
            continue

        if result is None or not result.success:
            status = "timeout" if result is None else result.status_message
            node.get_logger().error(f"生成 Gazebo 障碍物失败: {model_name}: {status}")
            all_ok = False
            continue

        node.get_logger().info(f"已在 Gazebo 中生成障碍物: {model_name}")

    return all_ok
