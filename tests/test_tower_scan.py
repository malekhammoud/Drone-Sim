"""Unit tests for the tower watch geometry (no sim, no CV model)."""
import math
import unittest

from arcticlib.config import load_config
from arcticlib.geo import Georef, bearing_deg, distance_m
from tools.tower_scan import (SweepPlan, TowerContact, TowerScanner, TowerWatch,
                              line_intersection, triangulation_sigma_m,
                              _angle_between)

CFG = load_config()
CONV = CFG.convergence_deg
T1 = (71.996396, -94.891696)
T2 = (71.986899, -94.778238)


class FakeTower:
    connected = True
    mode = "MANUAL"

    def __init__(self):
        self.commands = []

    def point(self, az_deg, el_deg):
        self.commands.append((az_deg, el_deg))
        return True

    def set_mode(self, mode, timeout=0.0):
        self.mode = mode
        return True


class FakeCam:
    def grab(self):
        return None


class FakeDetector:
    def detect(self, *a, **k):
        return []


class FakeFleet:
    def __init__(self):
        self.config = CFG
        self._v = {"tower-1": FakeTower(), "tower-2": FakeTower()}
        self.cams = {"tower-1": FakeCam(), "tower-2": FakeCam()}

    def vehicle(self, name):
        return self._v.get(name)


class TestBearingMapping(unittest.TestCase):
    def test_base_yaw_and_round_trip(self):
        sc = TowerScanner(FakeFleet(), "tower-1", detector=FakeDetector())
        self.assertAlmostEqual(sc.base_yaw_deg, (CONV + 90.0) % 360.0, places=6)
        for brg in (0.0, 40.0, 180.0, 351.2666, 290.0):
            az = sc.az_for_bearing(brg)
            self.assertTrue(-180.0 <= az <= 180.0)
            self.assertAlmostEqual(sc.camera_bearing_deg(az), brg % 360.0, places=6)

    def test_point_bearing_sends_az(self):
        fleet = FakeFleet()
        sc = TowerScanner(fleet, "tower-1", detector=FakeDetector())
        sc.point_bearing(351.2666, -7.8)
        az, el = fleet.vehicle("tower-1").commands[-1]
        self.assertAlmostEqual(az, 48.93, places=1)     # base(40.195) - 351.267
        self.assertAlmostEqual(el, -7.8, places=6)

    def test_effective_pose_bakes_camera_direction(self):
        sc = TowerScanner(FakeFleet(), "tower-2", detector=FakeDetector())
        p = sc.effective_pose(200.0, -5.0, t_sim=3.0)
        self.assertAlmostEqual(math.degrees(p.yaw), 200.0, places=6)
        self.assertAlmostEqual(math.degrees(p.pitch), -5.0, places=6)
        self.assertAlmostEqual(p.lat, T2[0], places=6)
        self.assertAlmostEqual(p.alt_amsl, sc.ground_elev_m + 2.70, places=6)


class TestSweep(unittest.TestCase):
    def test_serpentine_and_bounds(self):
        plan = SweepPlan(center_bearing_deg=100.0, half_width_deg=60.0,
                         pan_step_deg=20.0, tilts_deg=(-3.0, -8.0))
        steps = plan.steps()
        self.assertEqual(len(steps), 2 * (2 * 60 // 20 + 1))     # 2 tilts x 7 pans
        first = [b for b, e in steps if e == -3.0]
        second = [b for b, e in steps if e == -8.0]
        self.assertEqual(first, sorted(first))                    # left -> right
        self.assertEqual(second, sorted(second, reverse=True))    # right -> left
        for b, _ in steps:
            self.assertTrue(-180.0 <= b <= 180.0)
        self.assertTrue(all(40.0 <= b <= 160.0 for b, _ in steps))

    def test_scanner_cycles_through_all_steps(self):
        sc = TowerScanner(FakeFleet(), "tower-1", detector=FakeDetector(),
                          sweep=SweepPlan(0.0, half_width_deg=30.0, pan_step_deg=30.0,
                                          tilts_deg=(-3.0,)))
        seen = [sc.next_step() for _ in range(3)]
        self.assertEqual(len(set(seen)), 3)
        self.assertEqual(sc.next_step(), seen[0])                 # wraps


class TestTriangulation(unittest.TestCase):
    def test_crossing_rays_meet_at_target(self):
        # Two sensors due south and due west of a target, looking at it.
        target = (72.0, -94.8)
        s1 = (71.99, -94.8)      # south of target
        s2 = (72.0, -94.81)      # west of target
        b1 = bearing_deg(*s1, *target)
        b2 = bearing_deg(*s2, *target)
        hit = line_intersection(*s1, b1, *s2, b2)
        self.assertIsNotNone(hit)
        lat, lon, t1, t2 = hit
        self.assertLess(distance_m(lat, lon, *target), 5.0)
        self.assertGreater(t1, 0)
        self.assertGreater(t2, 0)

    def test_parallel_and_behind_rejected(self):
        self.assertIsNone(line_intersection(72.0, -94.8, 90.0, 72.0, -94.7, 90.0))
        # Both look away from each other -> intersection is behind both.
        self.assertIsNone(line_intersection(72.0, -94.8, 270.0, 72.0, -94.7, 90.0))

    def test_sigma_grows_with_range_and_shrinks_with_cross(self):
        near = triangulation_sigma_m(500.0, 500.0, 60.0)
        far = triangulation_sigma_m(4000.0, 4000.0, 60.0)
        self.assertGreater(far, near)
        shallow = triangulation_sigma_m(1000.0, 1000.0, 10.0)
        steep = triangulation_sigma_m(1000.0, 1000.0, 70.0)
        self.assertGreater(shallow, steep)

    def test_angle_between(self):
        self.assertAlmostEqual(_angle_between(350.0, 10.0), 20.0)
        self.assertAlmostEqual(_angle_between(90.0, 270.0), 180.0)


class TestTips(unittest.TestCase):
    def _watch_with(self, contacts, names=("tower-1", "tower-2")):
        w = TowerWatch(FakeFleet(), CFG, names=names, detector_factory=FakeDetector)
        w._contacts = contacts
        return w

    _tick = 0.0

    def _contact(self, tower, lat, lon, brg, score=0.8):
        TestTips._tick += 1.0
        return TowerContact(tower=tower, t_sim=1.0, t_wall=TestTips._tick,
                            bearing_deg=brg, elevation_deg=-5.0, lat=lat, lon=lon,
                            error_radius_m=100.0, ground_range_m=800.0,
                            score=score, confirmed=True, hits=6)

    def test_two_tower_contact_triangulates(self):
        # Two well-separated masts looking north converge to an apex.
        w = self._watch_with([
            self._contact("tower-1", 72.00, -94.78, 30.0),
            self._contact("tower-2", 72.00, -94.72, 330.0)])
        tip = w.latest_tip()
        self.assertIsNotNone(tip)
        self.assertEqual(tip.source, "triangulated")
        self.assertEqual(set(tip.towers), {"tower-1", "tower-2"})
        self.assertGreater(tip.lat, 72.005)     # apex is north of both masts

    def test_single_tower_contact_is_a_line(self):
        cs = [self._contact("tower-1", *T1, 101.7) for _ in range(2)]
        tip = self._watch_with(cs).latest_tip()
        self.assertIsNotNone(tip)
        self.assertEqual(tip.source, "single")
        self.assertEqual(tip.bearing_tower, "tower-1")
        self.assertAlmostEqual(tip.bearing_deg, 101.7)

    def test_one_sighting_is_not_enough(self):
        # A single dwell can be a glint; persistence needs min_confirm dwells.
        self.assertIsNone(self._watch_with([self._contact("tower-1", *T1, 101.7)]).latest_tip())

    def test_unconfirmed_contacts_ignored(self):
        cs = [self._contact("tower-1", *T1, 101.7) for _ in range(2)]
        for c in cs:
            c.confirmed = False
        self.assertIsNone(self._watch_with(cs).latest_tip())

    def test_single_line_search_point_is_range_clamped(self):
        cs = [self._contact("tower-1", *T1, 90.0) for _ in range(2)]
        cs[0].ground_range_m = 40000.0       # grazing ray meets water far away
        w = self._watch_with(cs)
        tip = w.latest_tip()
        lat, lon = w.search_point(tip, max_range_m=2500.0)
        # Should be 2500 m east of the tower, not 40 km.
        self.assertAlmostEqual(distance_m(T1[0], T1[1], lat, lon), 2500.0, delta=30.0)
        self.assertAlmostEqual(bearing_deg(T1[0], T1[1], lat, lon), 90.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
