#!/usr/bin/env python3
"""Static demo-scene markers, published from LAUNCH.

Publishes the fixed A/B/C goal regions (flat green discs), their letter $+$
robot-start-origin coordinate labels, and the start marker to ``/ubicomp/markers``
continuously, so RViz shows the target regions AS SOON AS IT OPENS -- before any
task is submitted and before the TeLoGraF follower runs.

The follower owns the DYNAMIC markers (planned path, travelled history, robot,
sim-clock, sensed obstacles); when it is started with ``--scene-external`` it skips
the start/goals/labels so the two never duplicate.  (For the standalone
closed-loop sim there is no scene node, so the follower publishes them itself.)
"""
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim_ros2.scenario import GOALS, GOAL_NAMES, START          # noqa: E402


class SceneMarkers(Node):
    def __init__(self):
        super().__init__("scene_markers")
        self.pub = self.create_publisher(MarkerArray, "/ubicomp/markers", 10)
        self.frame = "map"
        self.create_timer(0.5, self._publish)                   # 2 Hz, persistent

    def _publish(self):
        arr = MarkerArray()
        nid = [0]

        def base(kind, ns, r, g, b, a, s):
            m = Marker()
            m.header.frame_id = self.frame
            m.ns = ns
            m.id = nid[0]; nid[0] += 1
            m.type = kind; m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = s
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
            m.pose.orientation.w = 1.0
            return m

        sx, sy = float(START[0]), float(START[1])
        s = base(Marker.SPHERE, "start", 0.1, 0.8, 0.1, 1.0, 0.35)
        s.pose.position.x, s.pose.position.y = sx, sy
        arr.markers.append(s)
        sl = base(Marker.TEXT_VIEW_FACING, "start_label", 1.0, 1.0, 1.0, 1.0, 0.4)
        sl.pose.position.x, sl.pose.position.y, sl.pose.position.z = sx, sy - 0.55, 0.4
        sl.text = "start (0, 0)"
        arr.markers.append(sl)

        for (gx, gy, gr), name in zip(GOALS, GOAL_NAMES):
            gx, gy, gr = float(gx), float(gy), float(gr)
            g = base(Marker.CYLINDER, "goals", 0.2, 0.85, 0.4, 0.55, 2 * gr)
            g.scale.z = 0.02
            g.pose.position.x, g.pose.position.y, g.pose.position.z = gx, gy, 0.02
            arr.markers.append(g)
            lab = base(Marker.TEXT_VIEW_FACING, "goal_labels", 1.0, 1.0, 1.0, 1.0, 0.45)
            lab.pose.position.x, lab.pose.position.y, lab.pose.position.z = gx, gy, gr + 0.6
            rx, ry = gx - sx, gy - sy            # robot-start-origin coordinate
            lab.text = f"{name} ({rx:g}, {ry:g})"
            arr.markers.append(lab)

        self.pub.publish(arr)


def main():
    rclpy.init()
    node = SceneMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
