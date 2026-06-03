"""
Convert a 2-D LiDAR scan into planner-format obstacle disks, in WORLD frame.

This is the *real-time sensor* obstacle source for the closed loop.  The
TurtleBot3's 360 LiDAR publishes sensor_msgs/LaserScan; each return that is
closer than `range_max` is a hit.  We project hits to world
coordinates using the robot pose, snap them to an occupancy grid so a wall
becomes a handful of disks rather than hundreds of points, and hand those disks
to plan_waypoints(obstacles=...).  The planner (STLCG guidance + A*) then routes
around whatever the robot can actually see -- so it avoids walls the STL spec
never mentioned, world-agnostically (works in depot/warehouse where the geometry
is Fuel models, not inline SDF boxes).

Pure functions (no ROS dependency) so they unit-test without a simulator.
"""
from __future__ import annotations

import math
from typing import List, Tuple


def scan_to_points(ranges: List[float], angle_min: float, angle_inc: float,
                   pose: Tuple[float, float, float],
                   range_min: float = 0.05,
                   range_max: float = 12.0) -> List[Tuple[float, float]]:
    """Project valid LiDAR returns to world (x, y) points.

    pose = (rx, ry, ryaw) of the laser in the world frame."""
    rx, ry, ryaw = pose
    pts: List[Tuple[float, float]] = []
    for i, r in enumerate(ranges):
        if r is None or not math.isfinite(r) or r < range_min or r > range_max:
            continue
        a = ryaw + angle_min + i * angle_inc
        pts.append((rx + r * math.cos(a), ry + r * math.sin(a)))
    return pts


def points_to_disks(points: List[Tuple[float, float]],
                    cell: float = 0.30, radius: float = 0.22,
                    min_hits: int = 1) -> List[dict]:
    """Snap world points to a grid; each occupied cell -> one obstacle disk.

    cell:     grid resolution (m).  radius: disk radius placed at each cell
    centre.   min_hits: ignore cells with fewer than this many returns
    (rejects lidar speckle)."""
    counts: dict = {}
    for (x, y) in points:
        key = (round(x / cell), round(y / cell))
        counts[key] = counts.get(key, 0) + 1
    disks: List[dict] = []
    for (cx, cy), n in counts.items():
        if n < min_hits:
            continue
        disks.append({"kind": "circle", "x": cx * cell, "y": cy * cell,
                      "r": radius})
    return disks


def scan_to_obstacles(ranges, angle_min, angle_inc, pose,
                      *, cell: float = 0.30, radius: float = 0.22,
                      range_min: float = 0.05, range_max: float = 12.0,
                      min_hits: int = 1) -> List[dict]:
    """LaserScan -> world-frame obstacle disks (scan_to_points + points_to_disks)."""
    pts = scan_to_points(ranges, angle_min, angle_inc, pose,
                         range_min=range_min, range_max=range_max)
    return points_to_disks(pts, cell=cell, radius=radius, min_hits=min_hits)


def merge_obstacles(existing: List[dict], new: List[dict],
                    dedup_cell: float = 0.30) -> List[dict]:
    """Accumulate sensed disks across scans, deduped on a grid."""
    seen = {(round(o["x"] / dedup_cell), round(o["y"] / dedup_cell))
            for o in existing}
    out = list(existing)
    for o in new:
        key = (round(o["x"] / dedup_cell), round(o["y"] / dedup_cell))
        if key not in seen:
            seen.add(key)
            out.append(o)
    return out
