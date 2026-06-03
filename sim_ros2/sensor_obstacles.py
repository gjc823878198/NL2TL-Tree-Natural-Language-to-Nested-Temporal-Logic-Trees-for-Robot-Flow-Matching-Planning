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
    """Accumulate sensed disks across scans, deduped on a grid.

    NOTE: this only ever *adds* and never forgets, so as the robot orbits an
    object and sweeps more of its surface the union of per-cell disks paints an
    ever-growing ring.  Prefer ``cluster_points_to_disks`` + ``fuse_sensed`` for
    the live closed loop; this is kept for the offline/known-map path and tests.
    """
    seen = {(round(o["x"] / dedup_cell), round(o["y"] / dedup_cell))
            for o in existing}
    out = list(existing)
    for o in new:
        key = (round(o["x"] / dedup_cell), round(o["y"] / dedup_cell))
        if key not in seen:
            seen.add(key)
            out.append(o)
    return out


def cluster_points_to_disks(points: List[Tuple[float, float]],
                            viewpoint: Tuple[float, float] | None = None,
                            *, link: float = 0.45, min_pts: int = 2,
                            base_r: float = 0.16, margin: float = 0.06,
                            max_r: float = 0.60, push: float = 0.12) -> List[dict]:
    """Connected-component cluster of world points -> ONE bounding circle per object.

    A LiDAR sees only the *near face* of an object, so snapping each return to
    its own grid cell paints a ring of disks that GROWS as the robot orbits the
    object and sweeps more surface (the bug behind "obstacles keep getting
    bigger").  Instead we link returns that are within ``link`` of each other
    into one cluster and emit a single disk at the cluster centroid, nudged
    ``push`` m AWAY from ``viewpoint`` (the robot) to re-centre it from the
    near face onto the object body.  Radius = covering radius + ``margin``,
    clamped to [``base_r``, ``max_r``].  Clusters with < ``min_pts`` returns are
    dropped as speckle.  Pure function (grid-bucketed union-find, no ROS)."""
    pts = [(float(x), float(y)) for (x, y) in points]
    n = len(pts)
    if n == 0:
        return []
    buckets: dict = {}
    for i, (x, y) in enumerate(pts):
        buckets.setdefault((round(x / link), round(y / link)), []).append(i)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    link2 = link * link
    for (gx, gy), idxs in buckets.items():
        neigh = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neigh += buckets.get((gx + dx, gy + dy), [])
        for a in idxs:
            ax, ay = pts[a]
            for b in neigh:
                if b <= a:
                    continue
                bx, by = pts[b]
                if (ax - bx) ** 2 + (ay - by) ** 2 <= link2:
                    union(a, b)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    disks: List[dict] = []
    for idxs in groups.values():
        if len(idxs) < min_pts:
            continue
        xs = [pts[i][0] for i in idxs]
        ys = [pts[i][1] for i in idxs]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        if viewpoint is not None and push:
            vx, vy = viewpoint
            d = math.hypot(cx - vx, cy - vy)
            if d > 1e-6:
                cx += push * (cx - vx) / d
                cy += push * (cy - vy) / d
        cover = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys))
        r = min(max_r, max(base_r, cover + margin))
        disks.append({"kind": "circle", "x": cx, "y": cy, "r": r})
    return disks


def fuse_sensed(existing: List[dict], new: List[dict], *,
                link: float = 0.45, ttl: int = 4, ema: float = 0.4) -> List[dict]:
    """Fuse freshly-clustered bounding circles into a SHORT-memory sensed set.

    The set tracks only what is CURRENTLY around the robot -- it does not keep a
    long-term map.  A small ``ttl`` (a few control ticks, ~0.5 s) just smooths
    single-frame LiDAR dropout; an object the robot has driven past is forgotten
    almost immediately and simply re-detected the next time it is in view:
      * a new circle within ``link`` of an existing one refreshes it (EMA
        position, radius relaxes toward the new measurement, age reset);
      * a new circle far from all existing ones is appended;
      * every call ages all circles by 1; any not re-seen within ``ttl`` calls
        are dropped.
    Each disk carries an internal ``_age`` (calls since last seen)."""
    out = []
    for o in existing:
        o = dict(o)
        o["_age"] = int(o.get("_age", 0)) + 1
        out.append(o)
    for nd in new:
        best, bestd = None, link
        for o in out:
            d = math.hypot(o["x"] - nd["x"], o["y"] - nd["y"])
            if d < bestd:
                best, bestd = o, d
        if best is None:
            o = dict(nd)
            o["_age"] = 0
            out.append(o)
        else:
            best["x"] = (1 - ema) * best["x"] + ema * nd["x"]
            best["y"] = (1 - ema) * best["y"] + ema * nd["y"]
            # relax the radius toward the new measurement so an over-estimate
            # decays instead of sticking (never below the fresh reading).
            best["r"] = max(nd["r"], best["r"] * 0.9)
            best["_age"] = 0
    return [o for o in out if o["_age"] <= ttl]
