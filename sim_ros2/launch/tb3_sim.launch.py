"""
TurtleBot3 (burger) closed-loop sim in CLASSIC Gazebo (gazebo11 / gazebo_ros).

Open world with CYLINDER obstacles generated from a planner case (green = goal,
red = avoid) -- no walls.  TurtleBot3 senses the cylinders online with its 360
LiDAR (/scan) and the planner routes around them; the robot drives via /cmd_vel
(diff-drive) and publishes /odom, so RViz shows it moving for real.

    ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=cond_reach_either
    ros2 launch sim_ros2/launch/tb3_sim.launch.py case:=cond_reach_either gui:=true rviz:=true

Then drive it with the closed-loop follower:
    ros2 run ... (see tb3_follower.py)  OR  python3 sim_ros2/tb3_follower.py --case cond_reach_either
"""
import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))


def _setup(context, *args, **kwargs):
    from sim_ros2.tb3_world_gen import cylinder_world
    from planner import get_case

    case_id = LaunchConfiguration("case").perform(context)
    gui     = LaunchConfiguration("gui").perform(context) == "true"
    rviz    = LaunchConfiguration("rviz").perform(context) == "true"
    model   = LaunchConfiguration("model").perform(context)

    # env the TB3 stock launches + spawn read (set here so the launch is
    # self-contained -- no need to `export TURTLEBOT3_MODEL` first).
    os.environ["TURTLEBOT3_MODEL"] = model
    pkg_tb3_models = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"), "models")
    os.environ["GAZEBO_MODEL_PATH"] = (
        pkg_tb3_models + ":" + os.environ.get("GAZEBO_MODEL_PATH", ""))

    # A snap-packaged terminal/IDE (e.g. VS Code installed as a snap) prepends
    # /snap/.../lib to LD_LIBRARY_PATH; rviz2 then links the wrong libpthread and
    # dies with "undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE"
    # (exit 127).  Strip the snap entries so rviz2/gzserver use the system libs.
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        p for p in ld.split(":") if p and "/snap/" not in p)
    # rviz2 on a Wayland session: use XWayland (xcb) to avoid the Qt-Wayland path.
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    case = get_case(case_id)
    sx, sy = case["map_hint"]["start"]
    # EXTRA obstacle cylinders -- only the robot's LiDAR knows about these (they
    # are NOT in the STL spec); it senses them online and routes around them.
    if case_id == "closed_loop_multi":
        # the shared UNIFORM cylinder grid (sim_ros2/scenario.py), same field the
        # 2D demo + the paper figure use.
        from sim_ros2.scenario import CYLINDERS
        extra = list(CYLINDERS)
    else:
        # a couple of cylinders in the upper field, leaving a corridor open.
        extra = [(-0.6, 1.0, 0.5), (0.9, 1.6, 0.45)]
    world_path = f"/tmp/{case_id}_tb3_cyl.world"
    Path(world_path).write_text(cylinder_world(case_id, extra_obstacles=extra))
    print(f"[tb3_sim] case={case_id} model={model} spawn=({sx},{sy}) "
          f"-> open cylinder world {world_path}")

    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    pkg_tb3        = get_package_share_directory("turtlebot3_gazebo")
    urdf_sdf = os.path.join(pkg_tb3, "models", f"turtlebot3_{model}", "model.sdf")

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")),
        launch_arguments={"world": world_path}.items())
    actions = [gzserver]
    if gui:
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py"))))

    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3, "launch", "robot_state_publisher.launch.py")),
        launch_arguments={"use_sim_time": "true"}.items()))

    actions.append(Node(
        package="gazebo_ros", executable="spawn_entity.py", output="screen",
        arguments=["-entity", f"tb3_{model}", "-file", urdf_sdf,
                   "-x", str(sx), "-y", str(sy), "-z", "0.01"]))

    if rviz:
        # TB3 /odom is already in WORLD coords, so its odom frame == world.
        # map markers are in world coords too -> map->odom is identity.
        actions.append(Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="map_to_odom_static", output="log",
            arguments=["--x", "0", "--y", "0", "--z", "0",
                       "--frame-id", "map", "--child-frame-id", "odom"]))
        actions.append(Node(
            package="rviz2", executable="rviz2", name="rviz2_markers",
            output="screen",
            arguments=["-d", str(CODE_ROOT / "sim_ros2" / "gui" / "markers.rviz")]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("case", default_value="cond_reach_either"),
        DeclareLaunchArgument("model", default_value="burger",
                              choices=["burger", "waffle", "waffle_pi"]),
        DeclareLaunchArgument("gui", default_value="false",
                              choices=["true", "false"],
                              description="Gazebo 3D client (heavy); default off "
                                          "-- watch in RViz markers instead."),
        DeclareLaunchArgument("rviz", default_value="true",
                              choices=["true", "false"],
                              description="RViz with /ubicomp/markers "
                                          "(start, all goals, travelled path)."),
        OpaqueFunction(function=_setup),
    ])
