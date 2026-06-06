"""
Generate a Classic Gazebo (gazebo11) `.world` from a planner case:
flat ground + sun + CYLINDER obstacles (green = goal, red = avoid), no walls.

Used by the TurtleBot3 closed-loop demo (`tb3_sim.launch.py`).  The robot senses
the red cylinders online with its 360-degree LiDAR (`/scan`) and the planner
routes around them in open space.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from planner import get_case                                      # noqa: E402
from sim2d.env_2d import build_scene, Disk                        # noqa: E402

_CYL = """\
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">
        <collision name="col">
          <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>
        </collision>
        <visual name="vis">
          <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
        </visual>
      </link>
    </model>
"""
# goal regions: a FLAT, VISUAL-ONLY green disc on the ground (no <collision>, so
# the LiDAR -- which senses collision geometry -- ignores it and the robot can
# still drive in; it just MARKS the target area in green, as in the 2D view).
_DISC = """\
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} 0.015 0 0 0</pose>
      <link name="link">
        <visual name="vis">
          <geometry><cylinder><radius>{r}</radius><length>0.02</length></cylinder></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
        </visual>
      </link>
    </model>
"""
_COLOR = {"reach": "0.2 0.85 0.4 1", "avoid": "0.9 0.2 0.2 1",
          "obstacle": "0.55 0.55 0.55 1"}


def cylinder_disks(case_id: str, extra_obstacles=()):
    """The case's reach/avoid disks (as cylinders) + any extra obstacle disks."""
    scene = build_scene(get_case(case_id), seed=0)
    scene.rects = []                                # cylinders only, no walls
    disks = [d for d in scene.disks]
    for (ex, ey, er) in extra_obstacles:
        disks.append(Disk(ex, ey, er, kind="avoid", name="extra"))
    return disks


def cylinder_world(case_id: str, extra_obstacles=(), h: float = 0.6) -> str:
    # OBSTACLES become physical (collidable) cylinders.  Goal (reach) regions are
    # drawn as FLAT, VISUAL-ONLY green discs (no collision) so they MARK the target
    # area in Gazebo without the LiDAR treating them as obstacles -- the robot can
    # still drive in and arrive.
    models = ""
    for i, d in enumerate(cylinder_disks(case_id, extra_obstacles)):
        if d.kind == "reach":
            models += _DISC.format(name=f"goal_{i}", x=d.x, y=d.y, r=d.r,
                                   rgba=_COLOR["reach"])
            continue
        models += _CYL.format(name=f"{d.kind}_{i}", x=d.x, y=d.y, z=h / 2,
                              r=d.r, h=h,
                              rgba=_COLOR.get(d.kind, _COLOR["obstacle"]))
    # ground + sun are embedded INLINE (not model://...) so gzserver never
    # touches the online model database -- it starts fast and the
    # /spawn_entity service comes up well within the spawner's timeout.
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="default">
    <scene><shadows>false</shadows></scene>
    <physics type="ode">
      <real_time_update_rate>1000.0</real_time_update_rate>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <surface><friction><ode><mu>100</mu><mu2>50</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material><ambient>0.9 0.9 0.9 1</ambient><diffuse>0.9 0.9 0.9 1</diffuse></material>
        </visual>
      </link>
    </model>
{models}  </world>
</sdf>
"""


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="cond_reach_either")
    ap.add_argument("--out", default="/tmp/tb3_cyl.world")
    args = ap.parse_args()
    Path(args.out).write_text(cylinder_world(args.case))
    print("wrote", args.out)
