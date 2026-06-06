"""
TurtleBot3 closed-loop follower (Classic Gazebo).

Rather than teleporting the model with set_pose, this drives the REAL diff-drive
robot:

  * pose          <- /odom            (odom frame originates at the spawn pose,
                                        so world = spawn + odom)
  * obstacles     <- /scan            (360 LiDAR -> world-frame obstacle disks)
  * actuation     -> /cmd_vel         (pure-pursuit Twist)
  * visualisation -> /ubicomp/markers (start, ALL goals, planned + travelled
                                        path, LiDAR range, sensed cylinders)

Closed loop: the frozen flow plan knows only what the LiDAR has sensed; when a
sensed cylinder blocks the remaining path the robot REPLANS from its current
pose.  Run AFTER `tb3_sim.launch.py` is up:

    python3 sim_ros2/tb3_follower.py --case cond_reach_either
"""
from __future__ import annotations
import argparse, io, contextlib, json, math, sys, time, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rclpy
from rclpy.node import Node
from rclpy.qos import (qos_profile_sensor_data, QoSProfile,
                       QoSDurabilityPolicy)
from rclpy.time import Time
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener

from planner import get_case, plan_waypoints                      # noqa: E402
from planner.telograf_infer import _obstacle_blocked              # noqa: E402
from keep_safe import keep_safe_robustness                        # noqa: E402
from stl_runtime import reach_by_deadline_rho, PlanLatencyModel   # noqa: E402
from sim_ros2.sensor_obstacles import (scan_to_points,                # noqa: E402
                                       cluster_points_to_disks, fuse_sensed)

import numpy as np                                                # noqa: E402


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _reach_targets(case):
    return [(g["x"], g["y"], g["r"]) for g in case["grounding"].values()
            if g.get("kind", "reach") == "reach"]


def _plan(case, start_xy, obstacles):
    c = dict(case); c["map_hint"] = dict(case["map_hint"])
    c["map_hint"]["start"] = [float(start_xy[0]), float(start_xy[1])]
    with contextlib.redirect_stdout(io.StringIO()):
        return [(float(x), float(y))
                for x, y in plan_waypoints(c, n_steps=90, backend="telograf",
                                           obstacles=obstacles)]


class TB3Follower(Node):
    def __init__(self, args):
        super().__init__("tb3_follower")
        # use SIM time so the TF timestamps (stamped in sim time by Gazebo) line
        # up with our lookups; otherwise tf2 treats every transform as stale.
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        # a JSON case file (e.g. one built from natural language by the Demo
        # Supplement's nl_grounding.py) overrides the named --case registry.
        if getattr(args, "case_file", None):
            self.case = json.loads(Path(args.case_file).read_text())
        else:
            self.case = get_case(args.case)
        self.sx, self.sy = self.case["map_hint"]["start"]   # world spawn
        self.reaches = _reach_targets(self.case)
        self.sense_range = args.sense_range
        self.frame = "map"
        # when a scene_markers node publishes the start/goals/labels (the demo),
        # skip them here so the two do not draw duplicate markers.
        self.scene_external = bool(getattr(args, "scene_external", False))
        self.scan = None
        self.pose = (self.sx, self.sy, 0.0)                 # WORLD pose (base)
        self._scan_pose = (self.sx, self.sy, 0.0)           # odom pose @ scan time
        self.have_odom = False
        # TF: transform each scan into `map` at its OWN timestamp, giving the
        # correct laser pose (mount offset + yaw-at-capture).  This fixes the
        # offset / rotation / smear that come from projecting with the latest
        # (lagged) /odom pose.  Falls back to the odom pose + a fixed burger
        # laser mount offset if TF is not yet available.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.laser_dx, self.laser_dy = -0.032, 0.0          # base_scan vs base
        self.sensed = []
        # short sliding window of recent world-frame scan points: a 0.22 m
        # cylinder is hit by only a handful of LiDAR beams per scan, so one frame
        # is too sparse to cluster/fit reliably; aggregating a few recent frames
        # (the robot has moved slightly, so it sweeps more of each arc) gives a
        # denser cloud -> a well-conditioned circle fit and a stable detection,
        # while old frames expire so far obstacles are still forgotten.
        self._tick = 0
        self._pcl_window = []          # [(tick, [(x, y), ...]), ...]
        self.pcl_window_ticks = 8      # ~1.1 s of frames aggregated per cluster
        self._last_cloud = []          # aggregated world points (RViz diagnostic)
        # optional persistent map: accumulate the (now stable, fit-centred)
        # clustered circles into one deduped set, so ALL discovered obstacles
        # stay shown -- higher count, at the cost of remembering objects no longer
        # in view (fine in a static scene; clustering means it can't grow a ring).
        self.persist_map = bool(getattr(args, "persist_map", False))
        self._map = []
        self.travelled = []
        self._sim_t = 0.0        # current simulation time (s), shown at the robot
        self.task_deadline_s = self.case.get("map_hint", {}).get("reach_deadline_s")
        # planning-latency-aware robustness (our contribution): the planner spends
        # wall-clock seconds re-planning; we predict that latency and debit the
        # COLD-START (which blocks before driving) from the deadline budget.
        self.plan_latency = PlanLatencyModel()
        self.create_subscription(LaserScan, "/scan", self._on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        mk_qos = QoSProfile(depth=1)
        mk_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.markers = self.create_publisher(MarkerArray, "/ubicomp/markers",
                                             mk_qos)
        # ---- MPPI trajectory-tracking controller ----
        self.H = 20            # horizon steps (x mdt = lookahead time)
        self.K = 400           # rollout samples
        self.mdt = 0.15        # control dt
        self.v_max = 0.22      # TB3 burger limits
        self.w_max = 2.0
        self.lam = 0.6         # MPPI temperature
        self.sig = np.array([0.12, 0.7])           # (v,w) sample noise
        # warm-started nominal control, biased FORWARD so the robot commits to
        # driving (an all-zero nominal makes MPPI creep).
        self.U = np.zeros((self.H, 2)); self.U[:, 0] = 0.15
        self.rng = np.random.default_rng(0)
        # ---- TeLoGraF re-planning (runs continuously in its own thread) ----
        self.ref_plan = None              # latest TeLoGraF trajectory to track
        self.plan_lock = threading.Lock()
        self.goal = None                  # locked goal
        self.running = True
        self.n_replan = 0
        self.replan_period = args.replan_period   # TeLoGraF re-plan period (s)
        # ---- Route B: runtime STL shield (exact monitor + guaranteed repair) --
        self.deadline_s = None            # reach deadline (real seconds, set in run)
        self.alpha = args.deadline_slack  # deadline = alpha * straight-line time
        self.nominal_speed = 0.18         # plan time-stamping speed (<= v_max)
        self.safe_margin = 0.08           # clearance below this -> hard safe-stop
        self.replan_margin = 0.22         # clearance below this -> force re-plan
        self.force_replan = threading.Event()
        self.reached = False
        self.n_safestop = 0
        self.n_force = 0
        # ---- multi-goal: visit ALL reach atoms in spec order (vs. one goal) ----
        self.visit_all = bool(getattr(args, "multi_goal", False)) or \
            (self.case.get("id") in ("closed_loop_multi", "nl_demo"))
        self.goal_queue = []
        self.goals_reached = []
        self.gi = 0
        self.leg_deadlines = []
        self.best_dg = float("inf")   # best (min) distance to the current goal
        self.stuck = 0                # control ticks with no progress
        self.straighting = 0          # ticks remaining to drive straight at goal
        # ground-truth obstacle field for the keep-safe CERTIFICATE: the online
        # sensed set is grid-snapped LiDAR cells (reads a hair negative at the
        # closest approach), so we certify collision-freedom against the true
        # cylinders -- the honest "did the robot enter a real obstacle" check.
        self._truth_obstacles = None
        if self.case.get("id") in ("closed_loop_multi", "nl_demo"):
            try:
                from sim_ros2.scenario import CYLINDERS
                self._truth_obstacles = [{"kind": "circle", "x": x, "y": y, "r": r}
                                         for (x, y, r) in CYLINDERS]
            except Exception:
                pass
        # ---- task description for the RViz overlay (NL + STL + goal letters) ----
        # closed_loop_multi sources its NL/STL/labels from the shared scenario so
        # the RViz text matches what the 2D demo prints; other cases fall back to
        # the case's own NL string and A,B,C,... labels.
        if self.case.get("id") == "closed_loop_multi":
            try:
                from sim_ros2.scenario import task_nl, task_stl, GOAL_NAMES
                self.task_nl, self.task_stl = task_nl(), task_stl()
                self.goal_names = list(GOAL_NAMES)
            except Exception:
                self.task_nl = self.task_stl = ""; self.goal_names = []
        elif self.case.get("id") == "nl_demo":
            # natural-language task (Demo Supplement): show the typed sentence,
            # the parsed STL, and the region letters from the grounded atoms.
            self.task_nl = self.case.get("nl", "")
            try:
                from stl_parser import ast_to_stl
                self.task_stl = ast_to_stl(self.case["tree"])
            except Exception:
                self.task_stl = ""
            self.goal_names = [k.replace("reach_", "").upper()
                               for k, g in self.case.get("grounding", {}).items()
                               if g.get("kind", "reach") == "reach"]
        else:
            self.task_nl = self.case.get("nl", "")
            self.task_stl = ""
            self.goal_names = [chr(ord("A") + i) for i in range(len(self.reaches))]
        # top of the field (for placing the floating NL text above the scene)
        self._scene_top = max([gy for (_, gy, _) in self.reaches] + [self.sy]) \
            if self.reaches else 3.0

    # -- callbacks --------------------------------------------------------
    def _on_scan(self, msg):
        self.scan = msg
        self._scan_pose = self.pose            # pose paired to THIS scan (fallback)

    def _on_odom(self, msg):
        # TB3's diff-drive publishes /odom in WORLD coordinates (at the spawn
        # pose it reads the spawn world position), so it IS the world pose.
        p = msg.pose.pose.position
        self.pose = (p.x, p.y, _yaw(msg.pose.pose.orientation))
        self.have_odom = True

    def _laser_pose_in_map(self):
        """(x, y, yaw) of the laser in the `map` frame at the current scan's
        timestamp, via TF (correct mount offset + yaw-at-capture).  Falls back to
        the odom pose paired to the scan plus the burger laser mount offset if TF
        is not yet available."""
        msg = self.scan
        frame = msg.header.frame_id or "base_scan"
        try:
            tf = self.tf_buffer.lookup_transform(
                self.frame, frame, Time.from_msg(msg.header.stamp))
            t = tf.transform.translation
            return (t.x, t.y, _yaw(tf.transform.rotation))
        except Exception:
            rx, ry, ryaw = self._scan_pose      # odom pose at scan arrival
            c, s = math.cos(ryaw), math.sin(ryaw)
            return (rx + self.laser_dx * c - self.laser_dy * s,
                    ry + self.laser_dx * s + self.laser_dy * c, ryaw)

    def _ingest_scan(self):
        if self.scan is None:
            return
        # project the scan to world points, accumulate a SHORT sliding window of
        # recent frames (denser cloud per object -> robust circle fit), then
        # cluster the union into ONE bounding circle per object.  Old frames
        # expire from the window, so an obstacle the robot has driven past is
        # forgotten and simply re-detected when it is in view again.
        self._tick += 1
        pts = scan_to_points(
            list(self.scan.ranges), self.scan.angle_min,
            self.scan.angle_increment, self._laser_pose_in_map(),
            range_max=min(self.sense_range, self.scan.range_max))
        self._pcl_window.append((self._tick, pts))
        self._pcl_window = [(t, p) for (t, p) in self._pcl_window
                            if self._tick - t < self.pcl_window_ticks]
        agg = [p for (_, frame) in self._pcl_window for p in frame]
        self._last_cloud = agg          # for the RViz raw-cloud diagnostic
        # lenient thresholds: real LiDAR returns on a thin cylinder are sparse, so
        # keep small clusters (min_pts=2) and don't despeckle them away
        # (speckle_min=1); the circle fit still recovers the centre when the arc
        # is dense enough, else the covering-disk fallback keeps the detection.
        clusters = cluster_points_to_disks(
            agg, viewpoint=self.pose[:2], speckle_min=1, min_pts=2)
        if self.persist_map:
            # never-forget dedup-merge -> a stable global map of every obstacle
            # discovered so far (one circle each, EMA-refined centre).
            self._map = fuse_sensed(self._map, clusters, ttl=10**9,
                                    link=0.5, ema=0.3)
            self.sensed = [{k: v for k, v in o.items() if k != "_age"}
                           for o in self._map]
        else:
            self.sensed = clusters

    def _blocked(self, plan, i, look=18):
        for (x, y) in plan[i:i + look]:
            if _obstacle_blocked(x, y, self.sensed, 0.12):
                return True
        return False

    def _clearance(self, x, y):
        """Signed clearance rho(not unsafe) at (x,y): min over sensed obstacles
        of (dist - r).  >0 = clear; this is the exact keep-safe predicate."""
        if not self.sensed:
            return 1e3
        return min(math.hypot(x - o["x"], y - o["y"]) - o["r"] for o in self.sensed)

    def _drive(self, tx, ty, v_max=0.20, w_max=2.0, look=0.28):
        x, y, yaw = self.pose
        d = math.hypot(tx - x, ty - y)
        th = math.atan2(ty - y, tx - x)
        he = math.atan2(math.sin(th - yaw), math.cos(th - yaw))
        t = Twist()
        t.angular.z = max(-w_max, min(w_max, 2.0 * he))
        t.linear.x = 0.0 if abs(he) > 1.0 else min(v_max, 0.6 * d)  # turn first
        self.cmd.publish(t)
        return d

    def _stop(self):
        self.cmd.publish(Twist())

    def _single_case(self, goal):
        """A single timed-reach spec toward the LOCKED goal (the flow planner is
        in-distribution on one timed reach), so the reference path is consistent
        with the MPPI goal instead of flipping between disjunctive goals."""
        gx, gy, gr = goal
        c = dict(self.case)
        c["tree"] = {"op": "finally", "interval": [20, 55],
                     "children": [{"op": "atom", "name": "g"}]}
        c["grounding"] = {"g": {"kind": "reach", "x": gx, "y": gy,
                                "z": 0.0, "r": gr}}
        c["map_hint"] = dict(self.case["map_hint"])
        c["map_hint"]["telograf_samples"] = 8   # fewer samples -> faster re-plan
        # Route B: pass the real-seconds reach deadline so the planner's temporal
        # gate certifies reach-by-deadline (and falls back to the faster A* path
        # if the flow plan would miss it).
        if self.deadline_s is not None:
            c["map_hint"]["reach_deadline_s"] = self.deadline_s
            c["map_hint"]["nominal_speed"] = self.nominal_speed
        return c

    def _arm_leg(self, goal):
        """Plan the first reference trajectory toward `goal` from the current pose
        and set its real-seconds reach deadline (Route B).  One call per leg."""
        t_plan0 = time.time()
        single0 = _plan(self._single_case(goal), self.pose[:2], self.sensed)
        self.plan_latency.observe(time.time() - t_plan0)
        L0 = sum(math.hypot(single0[i][0]-single0[i-1][0],
                            single0[i][1]-single0[i-1][1])
                 for i in range(1, len(single0))) or 1.0
        self.deadline_s = self.alpha * L0 / self.v_max
        self.leg_deadlines.append(self.deadline_s)
        self.best_dg = float("inf"); self.stuck = 0   # reset progress tracking
        with self.plan_lock:
            self.ref_plan = single0

    def _mppi(self, ref_pts, goal):
        """Sampling-based MPPI: track the flow-plan path (ref_pts) toward `goal`
        while reactively avoiding the LiDAR-sensed cylinders.  Returns (v, w)."""
        x0, y0, yaw0 = self.pose
        ref = np.asarray(ref_pts[::3] if len(ref_pts) > 6 else ref_pts, float)
        K, H = self.K, self.H
        # sample K control sequences around the warm-started nominal U
        noise = self.rng.standard_normal((K, H, 2)) * self.sig
        V = self.U[None] + noise
        V[:, :, 0] = np.clip(V[:, :, 0], 0.0, self.v_max)
        V[:, :, 1] = np.clip(V[:, :, 1], -self.w_max, self.w_max)
        # roll out a unicycle model for every sample (vectorised over K)
        x = np.full(K, x0); y = np.full(K, y0); yaw = np.full(K, yaw0)
        tx = np.empty((K, H)); ty = np.empty((K, H))
        for h in range(H):
            yaw = yaw + V[:, h, 1] * self.mdt
            x = x + V[:, h, 0] * np.cos(yaw) * self.mdt
            y = y + V[:, h, 0] * np.sin(yaw) * self.mdt
            tx[:, h] = x; ty[:, h] = y
        cost = np.zeros(K)
        # (1) path tracking: light weight -- the flow path is a HINT; the goal
        # term drives, so a misleading/looping path can't stall the robot.
        d2 = ((tx[:, :, None] - ref[None, None, :, 0]) ** 2 +
              (ty[:, :, None] - ref[None, None, :, 1]) ** 2)
        cost += 1.0 * np.sqrt(d2.min(axis=2)).mean(axis=1)
        # (2) goal progress: DOMINANT pull -- mean distance to goal over the
        # horizon (not just the endpoint) so it keeps closing the gap.
        cost += 12.0 * np.hypot(tx - goal[0], ty - goal[1]).mean(axis=1)
        # (3) obstacle avoidance: heavy penalty within a safety margin of a sensed
        # cylinder (margin = robot radius ~0.11 m + clearance), so the robot
        # never grazes one (keep-safe rho stays > 0).
        for o in self.sensed:
            clr = np.hypot(tx - o["x"], ty - o["y"]) - o["r"]
            cost += 120.0 * np.maximum(0.0, 0.35 - clr).sum(axis=1)
        # (4) keep moving: penalise low speed so the robot commits (avoids creep)
        cost += 4.0 * (self.v_max - V[:, :, 0].mean(axis=1))
        # (5) control smoothness
        cost += 0.05 * (V[:, :, 1] ** 2).sum(axis=1)
        # MPPI weighting + nominal update
        w = np.exp(-(cost - cost.min()) / self.lam)
        w /= w.sum() + 1e-9
        self.U = (w[:, None, None] * V).sum(axis=0)
        v, wz = float(self.U[0, 0]), float(self.U[0, 1])
        self.U = np.roll(self.U, -1, axis=0); self.U[-1] = self.U[-2]
        return max(0.0, min(self.v_max, v)), max(-self.w_max, min(self.w_max, wz))

    def _planner_thread(self):
        """Continuously RE-PLAN the reference trajectory with TeLoGraF from the
        robot's CURRENT pose toward the locked goal, on a fixed ~1 s PERIOD.
        Each iteration is one fresh TeLoGraF flow-matching plan -- the two-stage
        closed loop: TeLoGraF (re-)generates the trajectory, MPPI tracks it.

        NB: the period is TARGET 1 s; the actual period is bounded below by the
        TeLoGraF planning time (a fresh model is loaded per call ~3-4 s), so it
        re-plans as close to 1 Hz as the planner permits."""
        while self.running and rclpy.ok():
            if self.goal is None or self.straighting > 0:
                # while the stuck-breaker is driving a straight line, don't
                # overwrite the reference with a fresh (churning) flow plan.
                time.sleep(0.1); continue
            t0 = time.time()
            sc = self._single_case(self.goal)
            plan = _plan(sc, self.pose[:2], list(self.sensed))
            self.plan_latency.observe(time.time() - t0)
            with self.plan_lock:
                self.ref_plan = plan
            self.n_replan += 1
            self.get_logger().info(
                f"[TeLoGraF re-plan #{self.n_replan}] from "
                f"({self.pose[0]:.2f},{self.pose[1]:.2f}) -> {len(plan)} waypts; "
                f"sensed {len(self.sensed)} cylinders; {time.time()-t0:.2f}s")
            # interruptible pacing: sleep out the period, but wake IMMEDIATELY if
            # the runtime shield demands an event-driven re-plan (clearance low).
            slack = self.replan_period - (time.time() - t0)
            self.force_replan.clear()
            if slack > 0 and self.force_replan.wait(timeout=slack):
                self.n_force += 1

    # -- main loop --------------------------------------------------------
    def run(self):
        # print the task this run must complete (NL + STL + scene)
        if self.case.get("id") == "closed_loop_multi":
            try:
                from sim_ros2.scenario import print_task
                print_task(self.get_logger())
            except Exception:
                pass
        self.get_logger().info(
            f"Task: case='{self.case['id']}', {len(self.reaches)} reach goal(s)"
            f"{' (visit ALL in order)' if self.visit_all else ' (reach any one)'}; "
            f"waiting for /odom + /scan from the sim (terminal A must be running)...")
        # wait for odom + a couple of scans (up to ~10 s)
        for _ in range(200):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.have_odom and self.scan is not None:
                break
        if not self.have_odom or self.scan is None:
            self.get_logger().warn(
                "NO /odom or /scan after 10 s -- the robot will NOT move. Check: "
                "(1) terminal A (tb3_sim.launch) is still running and the robot "
                "spawned; (2) ROS_DOMAIN_ID is the SAME in both terminals "
                "(run `echo $ROS_DOMAIN_ID` in each); (3) `ros2 topic echo /odom "
                "--once` returns data in THIS terminal. Proceeding with defaults...")
        else:
            self.get_logger().info(
                f"got /odom + /scan (robot at "
                f"({self.pose[0]:.2f},{self.pose[1]:.2f})); planning the first "
                f"TeLoGraF trajectory -- the model cold-start can take ~10-30 s on "
                f"the first call, please wait (this is not a hang)...")
        self._ingest_scan()
        # publish the goal regions + labels + start BEFORE the cold-start plan, so
        # the A/B/C green discs are visible in RViz immediately (not after ~10-30 s).
        self._publish([])
        # build the goal queue: visit ALL reach atoms in spec order (multi-goal),
        # or the single nearest goal (disjunctive cases like cond_reach_either).
        if self.visit_all:
            self.goal_queue = list(self.reaches)
        else:
            plan0 = _plan(self.case, self.pose[:2], self.sensed)
            self.goal_queue = [min(self.reaches, key=lambda g:
                               math.hypot(g[0]-plan0[-1][0], g[1]-plan0[-1][1]))]
        nq = len(self.goal_queue)
        self.gi = 0
        self.goal = self.goal_queue[0]
        self._arm_leg(self.goal)            # plan leg 1, set its real-seconds deadline
        self.get_logger().info(
            f"[Route B] leg 1/{nq} -> goal ({self.goal[0]:.2f},{self.goal[1]:.2f}); "
            f"deadline {self.deadline_s:.0f}s; exact-STL shield armed")
        # start the continuous TeLoGraF re-planning thread (re-plans toward self.goal)
        threading.Thread(target=self._planner_thread, daemon=True).start()
        self.get_logger().info(
            "MPPI control loop running -> publishing /cmd_vel (the robot should move now)")
        # MPPI control loop (~1/mdt Hz): TRACK the latest TeLoGraF trajectory
        t = 0.0
        T_MAX = 70.0 * nq + 40.0            # budget scales with the number of goals
        while rclpy.ok() and t < T_MAX:
            loop_t0 = time.time()
            rclpy.spin_once(self, timeout_sec=0.0)
            t += self.mdt
            self._sim_t = t                          # for the RViz sim-time clock
            self._ingest_scan()
            self.travelled.append(self.pose[:2])
            # GENUINE disk entry to the CURRENT goal (dist < rr) so reach-atom
            # rho = rr - dist > 0; on entry, advance to the next goal (or finish).
            gx, gy, gr = self.goal
            if math.hypot(self.pose[0]-gx, self.pose[1]-gy) < max(0.12, gr - 0.05):
                self.goals_reached.append(self.goal)
                self.get_logger().info(
                    f"  REACHED goal {self.gi+1}/{nq} at "
                    f"({self.pose[0]:.2f},{self.pose[1]:.2f})")
                self.gi += 1
                if self.gi >= nq:
                    self.reached = True
                    break
                self.goal = self.goal_queue[self.gi]
                self._stop()
                self._arm_leg(self.goal)        # plan next leg, reset its deadline
                self.force_replan.set()
                self.get_logger().info(
                    f"[Route B] leg {self.gi+1}/{nq} -> goal "
                    f"({self.goal[0]:.2f},{self.goal[1]:.2f}); deadline "
                    f"{self.deadline_s:.0f}s")
                continue
            # stuck-breaker: if the robot hasn't gotten meaningfully closer to the
            # current goal for ~10 s, it is oscillating in a local minimum -> drive
            # a STRAIGHT line at the goal for a sustained ~10 s (MPPI still avoids
            # obstacles; the planner thread pauses so it can't churn the reference).
            dg = math.hypot(self.pose[0]-gx, self.pose[1]-gy)
            if self.straighting == 0 and dg < self.best_dg - 0.05:
                self.best_dg = dg; self.stuck = 0
            elif self.straighting == 0:
                self.stuck += 1
            if self.stuck > 70 and self.straighting == 0:    # ~10 s no progress
                self.stuck = 0
                self.straighting = 66                        # ~10 s straight drive
                self.get_logger().info(
                    f"  [stuck-breaker] no progress -> straight line to goal "
                    f"{self.gi+1}/{nq}")
            if self.straighting > 0:
                self.straighting -= 1
                with self.plan_lock:
                    self.ref_plan = [self.pose[:2], (gx, gy)]
                if self.straighting == 0:                    # re-baseline progress
                    self.best_dg = dg
            with self.plan_lock:
                plan = list(self.ref_plan) if self.ref_plan else None
            # ---- runtime STL shield: the exact keep-safe predicate ACTS ----
            clr = self._clearance(*self.pose[:2])
            if clr < self.safe_margin:
                # imminent collision: HARD safe-stop + event-driven re-plan.
                self._stop()
                self.n_safestop += 1
                self.force_replan.set()
            elif plan and len(plan) >= 2:
                if clr < self.replan_margin:
                    self.force_replan.set()       # close call -> re-plan sooner
                v, wz = self._mppi(plan, self.goal)   # MPPI -> control input
                tw = Twist(); tw.linear.x = v; tw.angular.z = wz
                self.cmd.publish(tw)
            if len(self.travelled) % 3 == 0:        # ~0.45 s: smooth live RViz path
                self._publish(plan or [])
            if len(self.travelled) % 14 == 0:        # ~2 s heartbeat: prove motion
                self.get_logger().info(
                    f"  driving t={t:.0f}s pose=({self.pose[0]:.2f},"
                    f"{self.pose[1]:.2f}) clearance={clr:.2f}m goal {self.gi+1}/{nq}")
            slack = self.mdt - (time.time() - loop_t0)
            if slack > 0:
                time.sleep(slack)
        self.running = False
        self.force_replan.set()       # release the planner thread's interruptible wait
        self._stop()
        with self.plan_lock:
            self._publish(self.ref_plan or [])
        # ---- exact STL certificates over the EXECUTED trajectory (Route B) ----
        # certify against the ground-truth field (true cylinders) when known;
        # else against the online sensed set.
        cert_obs = self._truth_obstacles if self._truth_obstacles else self.sensed
        rho_safe = keep_safe_robustness(cert_obs, self.travelled)
        elapsed = len(self.travelled) * self.mdt
        total_deadline = sum(self.leg_deadlines) or float("nan")
        n_reached = len(self.goals_reached)
        in_time = elapsed <= total_deadline
        gx2, gy2 = self.pose[:2]
        self.get_logger().info(
            f"DONE at ({gx2:.2f},{gy2:.2f}); visited {n_reached}/{nq} goals "
            f"(all={self.reached}); TeLoGraF re-plans={self.n_replan} "
            f"(event-driven={self.n_force}); safe-stops={self.n_safestop}; "
            f"sensed {len(self.sensed)} cylinders; total {elapsed:.0f}s")
        self.get_logger().info(
            f"[CERTIFY exact monitor] keep-safe rho(G!unsafe)={rho_safe:+.2f}m "
            f"({'SAFE' if rho_safe > 0 else 'VIOLATED'}); "
            f"goals visited {n_reached}/{nq} "
            f"({'ALL' if n_reached == nq else 'INCOMPLETE'}); "
            f"total t={elapsed:.0f}/{total_deadline:.0f}s "
            f"({'IN-TIME' if in_time else 'LATE'})")
        # ---- planning-latency-aware temporal certificate (our contribution) ----
        # The planner spent this much wall-clock thinking; under concurrent
        # re-planning only the cold-start blocks the robot, but we report the
        # conservative margin that debits ALL of it, plus the predicted next-plan
        # cost reserved for any future re-plan.
        plan_wall = self.plan_latency.consumed()
        rho_time_lat = total_deadline - plan_wall - elapsed
        self.get_logger().info(
            f"[CERTIFY latency-aware] planner wall-clock={plan_wall:.1f}s "
            f"(predict next {self.plan_latency.predict_next():.1f}s); "
            f"rho_time^lat=(deadline-planning)-exec={rho_time_lat:+.0f}s "
            f"({'IN-TIME' if rho_time_lat > 0 else 'LATE once planning counted'})")

    # -- markers ----------------------------------------------------------
    def _publish(self, plan):
        arr = MarkerArray(); mid = [0]

        def base(kind, ns, r, g, b, a, s):
            m = Marker(); m.header.frame_id = self.frame; m.ns = ns
            m.id = mid[0]; mid[0] += 1; m.type = kind; m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = s
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
            m.pose.orientation.w = 1.0
            return m
        if not self.scene_external:           # scene_markers node owns these
            s = base(Marker.SPHERE, "start", 0.1, 0.8, 0.1, 1.0, 0.35)
            s.pose.position.x, s.pose.position.y = float(self.sx), float(self.sy)
            arr.markers.append(s)
            for (gx, gy, gr) in self.reaches:
                # flat GREEN disc marking the goal region (matches the Gazebo disc
                # and the 2D view); semi-transparent so path/robot show through.
                g = base(Marker.CYLINDER, "goals", 0.2, 0.85, 0.4, 0.55, 2 * gr)
                g.scale.z = 0.02
                g.pose.position.x, g.pose.position.y = float(gx), float(gy)
                g.pose.position.z = 0.02
                arr.markers.append(g)
        if len(plan) >= 2:
            ln = base(Marker.LINE_STRIP, "planned", 0.1, 0.9, 0.1, 0.9, 0.04)
            ln.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in plan]
            arr.markers.append(ln)
        if len(self.travelled) >= 2:
            ln = base(Marker.LINE_STRIP, "travelled", 0.95, 0.85, 0.1, 1.0, 0.06)
            ln.points = [Point(x=float(x), y=float(y), z=0.06)
                         for x, y in self.travelled]
            arr.markers.append(ln)
        rng = base(Marker.CYLINDER, "sense_range", 0.1, 0.8, 0.8, 0.10,
                   2 * self.sense_range)
        rng.scale.z = 0.02
        rng.pose.position.x, rng.pose.position.y = float(self.pose[0]), float(self.pose[1])
        arr.markers.append(rng)
        for o in self.sensed:
            c = base(Marker.CYLINDER, "sensed", 0.9, 0.1, 0.1, 0.7, 2 * o["r"])
            c.scale.z = 0.5
            c.pose.position.x, c.pose.position.y = float(o["x"]), float(o["y"])
            c.pose.position.z = 0.25
            arr.markers.append(c)
        # --- raw aggregated LiDAR cloud (diagnostic): the grey points are what
        #     the perception sees; the red circles are fit to THESE points, so if
        #     the points don't line up with the obstacles it's a sensing/pose
        #     issue, not a clustering one ---
        if self._last_cloud:
            pc = base(Marker.POINTS, "scan_points", 0.55, 0.55, 0.6, 0.9, 0.035)
            pc.points = [Point(x=float(x), y=float(y), z=0.05)
                         for (x, y) in self._last_cloud]
            arr.markers.append(pc)
        # --- the task's NATURAL-LANGUAGE description, floating above the scene ---
        if self.task_nl:
            import textwrap
            txt = base(Marker.TEXT_VIEW_FACING, "task_nl", 1.0, 0.95, 0.3, 1.0, 0.34)
            txt.pose.position.x = 0.0
            txt.pose.position.y = float(self._scene_top) + 1.6
            txt.pose.position.z = 1.0
            body = "TASK (NL): " + "\n".join(textwrap.wrap(self.task_nl, 42))
            if self.task_stl:
                body += "\nSTL: " + self.task_stl
            txt.text = body
            arr.markers.append(txt)
        # --- goal labels above each region: letter + coordinate in the
        #     robot-start-origin frame (origin = where the robot spawned) ---
        if not self.scene_external:           # scene_markers node owns these
            for (gx, gy, gr), name in zip(self.reaches, self.goal_names):
                lab = base(Marker.TEXT_VIEW_FACING, "goal_labels", 1.0, 1.0, 1.0, 1.0, 0.45)
                lab.pose.position.x, lab.pose.position.y = float(gx), float(gy)
                lab.pose.position.z = float(gr) + 0.6
                rx, ry = gx - self.sx, gy - self.sy   # relative to the spawn pose
                lab.text = f"{name} ({rx:g}, {ry:g})"
                arr.markers.append(lab)
        # --- live simulation-time clock, floating at the robot (seconds == the
        #     task's time unit, e.g. the "...within 120 s" deadline) ---
        clk = base(Marker.TEXT_VIEW_FACING, "sim_time", 1.0, 0.95, 0.2, 1.0, 0.42)
        clk.pose.position.x = float(self.pose[0]) + 0.35
        clk.pose.position.y = float(self.pose[1]) + 0.35
        clk.pose.position.z = 0.6
        clk.text = (f"t = {self._sim_t:.0f} / {self.task_deadline_s:.0f} s"
                    if self.task_deadline_s else f"t = {self._sim_t:.0f} s")
        arr.markers.append(clk)
        self.markers.publish(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="closed_loop_multi",
                    help="planner case; default closed_loop_multi == the SAME "
                         "map+task as the 2D demo (closed_loop_demo.py)")
    ap.add_argument("--case-file", default=None,
                    help="path to a JSON case dict (overrides --case); the Demo "
                         "Supplement builds one from natural language")
    ap.add_argument("--sense-range", type=float, default=3.5,
                    help="LiDAR sensing radius (m); capped at the scan's range_max. "
                         "Bigger = more obstacles in view at once (more red circles)")
    ap.add_argument("--persist-map", action="store_true",
                    help="accumulate a persistent global map of every discovered "
                         "obstacle (more red circles, stays shown) instead of the "
                         "default short-memory in-view-only set")
    ap.add_argument("--replan-period", type=float, default=5.0,
                    help="TeLoGraF re-plan period in seconds (target; bounded "
                         "below by the planner's per-call time)")
    ap.add_argument("--deadline-slack", type=float, default=2.0,
                    help="reach deadline = slack x straight-line time at v_max "
                         "(Route B exact-STL temporal certificate)")
    ap.add_argument("--multi-goal", action="store_true",
                    help="visit ALL reach atoms in spec order (sequential "
                         "multi-goal); auto-on for case closed_loop_multi")
    ap.add_argument("--scene-external", action="store_true",
                    help="a separate node (scene_markers.py) publishes the start "
                         "+ goal regions + labels, so skip them here (no duplicate)")
    args = ap.parse_args()
    rclpy.init()
    node = TB3Follower(args)
    try:
        node.run()
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
