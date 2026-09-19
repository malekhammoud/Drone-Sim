"""Unit tests for Vehicle.pose_at interpolation (no radio needed)."""
import math
import unittest
from collections import deque

from arcticlib.config import AssetSpec, Config
from arcticlib.types import Pose
from arcticlib.vehicle import Vehicle


def make_vehicle() -> Vehicle:
    spec = AssetSpec("quadcopter", "copter", 1, "localhost", 14550, None)
    return Vehicle(spec, Config(), auto_connect=False)


def pose(t, lat, lon, alt=0.0, yaw=0.0):
    return Pose("quadcopter", t_sim=t, t_wall=t * 10, lat=lat, lon=lon,
                alt_rel=alt, yaw=yaw)


class TestPoseAt(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(make_vehicle().pose_at(5.0))

    def test_linear_interpolation(self):
        v = make_vehicle()
        v._poses = deque([pose(0, 72.0, -95.0, 0), pose(10, 72.01, -95.02, 100)])
        p = v.pose_at(5.0, clock="sim")
        self.assertAlmostEqual(p.lat, 72.005, places=6)
        self.assertAlmostEqual(p.lon, -95.01, places=6)
        self.assertAlmostEqual(p.alt_rel, 50.0, places=6)

    def test_nearest_when_outside(self):
        v = make_vehicle()
        v._poses = deque([pose(10, 72.0, -95.0), pose(20, 73.0, -96.0)])
        self.assertAlmostEqual(v.pose_at(0.0).lat, 72.0)
        self.assertAlmostEqual(v.pose_at(99.0).lat, 73.0)

    def test_yaw_wraps_short_way(self):
        v = make_vehicle()
        # 170 deg -> -170 deg is a +20 deg step across the seam, not -340.
        v._poses = deque([pose(0, 72.0, -95.0, yaw=math.radians(170)),
                          pose(10, 72.0, -95.0, yaw=math.radians(-170))])
        p = v.pose_at(5.0)
        self.assertAlmostEqual(math.degrees(p.yaw), 180.0, delta=1e-6)

    def test_wall_clock(self):
        v = make_vehicle()
        v._poses = deque([pose(0, 72.0, -95.0), pose(10, 72.1, -95.0)])
        p = v.pose_at(50.0, clock="wall")   # t_wall = t_sim*10
        self.assertAlmostEqual(p.lat, 72.05, places=6)

    def test_exact_sample(self):
        v = make_vehicle()
        v._poses = deque([pose(0, 72.0, -95.0), pose(10, 72.1, -95.0),
                          pose(20, 72.2, -95.0)])
        self.assertAlmostEqual(v.pose_at(10.0).lat, 72.1, places=6)


if __name__ == "__main__":
    unittest.main()
