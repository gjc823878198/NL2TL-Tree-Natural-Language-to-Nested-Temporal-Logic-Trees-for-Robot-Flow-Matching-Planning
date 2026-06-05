"""
NL2TL-Tree demo — ONE launch file.

Brings up the closed-loop demonstration Gazebo world (the SAME uniform cylinder
field + start as code/sim_ros2/scenario.py) with a TurtleBot3, and pops up the
natural-language input box (demo/run_ros.py). You then:

    type a task  ->  frozen-LLM pipeline -> nested STL tree -> grounded case
                 ->  TeLoGraF plans the trajectory
                 ->  the TurtleBot3 executes it, with RViz opening to watch.

    ros2 launch "Demo supplement/demo/launch/demo.launch.py"
    # headless Gazebo (no 3D client, lighter):
    ros2 launch "Demo supplement/demo/launch/demo.launch.py" gui:=false
    # offline parse (no GROQ_API_KEY):
    ros2 launch "Demo supplement/demo/launch/demo.launch.py" llm:=false

The Gazebo world + robot + map->odom TF + RViz are started here at launch (RViz
shows the same markers.rviz as the closed-loop sim); the TeLoGraF follower is
started by the GUI when you submit a task. The map+task are identical to the
closed-loop simulation.
"""
import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DEMO = Path(__file__).resolve().parents[1]                 # the demo/ directory


def _find_code_root(start):                                # robust to demo location
    for p in [start, *start.parents]:
        if (p / "planner").is_dir() and (p / "nl_to_tree_groq.py").exists():
            return p
        if (p / "code" / "planner").is_dir():
            return p / "code"
    return start


CODE = _find_code_root(DEMO)                               # paper code/ root
sys.path.insert(0, str(CODE))


def _setup(context, *args, **kwargs):
    from sim_ros2.tb3_world_gen import cylinder_world
    from sim_ros2.scenario import CYLINDERS, START

    gui = LaunchConfiguration("gui").perform(context) == "true"
    llm = LaunchConfiguration("llm").perform(context) == "true"
    model = LaunchConfiguration("model").perform(context)

    os.environ["TURTLEBOT3_MODEL"] = model
    pkg_tb3_models = os.path.join(
        get_package_share_directory("turtlebot3_gazebo"), "models")
    os.environ["GAZEBO_MODEL_PATH"] = (
        pkg_tb3_models + ":" + os.environ.get("GAZEBO_MODEL_PATH", ""))
    # strip snap libs (rviz2/gzserver link the wrong libpthread otherwise)
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        p for p in ld.split(":") if p and "/snap/" not in p)
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    sx, sy = START
    world_path = "/tmp/nl_demo_tb3_cyl.world"
    Path(world_path).write_text(
        cylinder_world("closed_loop_multi", extra_obstacles=list(CYLINDERS)))
    print(f"[demo] open closed-loop world {world_path}; spawn=({sx},{sy})")

    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    pkg_tb3 = get_package_share_directory("turtlebot3_gazebo")
    urdf_sdf = os.path.join(pkg_tb3, "models", f"turtlebot3_{model}", "model.sdf")

    actions = [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")),
        launch_arguments={"world": world_path}.items())]
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
    # map == odom (TB3 /odom is already world coords)
    actions.append(Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="map_to_odom_static", output="log",
        arguments=["--x", "0", "--y", "0", "--z", "0",
                   "--frame-id", "map", "--child-frame-id", "odom"]))
    # RViz up FROM STARTUP (same markers.rviz as the closed-loop sim): it shows
    # /ubicomp/markers -- start, goals, planned path, travelled history, sense
    # ring, sim clock -- as soon as the follower runs.  Launched here (not by the
    # GUI) so it appears immediately on `ros2 launch`, exactly like tb3_sim.
    actions.append(Node(
        package="rviz2", executable="rviz2", name="rviz2_markers", output="screen",
        arguments=["-d", str(CODE / "sim_ros2" / "gui" / "markers.rviz")]))
    # the natural-language GUI (--no-rviz: RViz is already open above; the GUI
    # starts only the TeLoGraF follower when a task is submitted)
    gui_cmd = [sys.executable, str(DEMO / "run_ros.py"), "--no-rviz"]
    if not llm:
        gui_cmd.append("--no-llm")
    actions.append(ExecuteProcess(cmd=gui_cmd, output="screen"))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("model", default_value="burger",
                              choices=["burger", "waffle", "waffle_pi"]),
        DeclareLaunchArgument("gui", default_value="true",
                              choices=["true", "false"],
                              description="show the Gazebo 3D client (the world)"),
        DeclareLaunchArgument("llm", default_value="true",
                              choices=["true", "false"],
                              description="use the Groq LLM parse (false = offline "
                                          "keyword parser)"),
        OpaqueFunction(function=_setup),
    ])
