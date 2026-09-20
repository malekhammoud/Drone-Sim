"""Tests for track posting: derived heading/speed (no network)."""
import unittest

from arcticlib.geo import destination
from arcticlib.geolocate import fuse_estimates, track_course_speed
from arcticlib.tracks import TrackClient


class TestPostFix(unittest.TestCase):
    def setUp(self):
        self.client = TrackClient("http://127.0.0.1:9", min_interval=0.0, retries=0)
        self.calls = []
        self.client.post = lambda *a, **k: (self.calls.append((a, k)), {"ok": True})[1]

    def test_first_fix_has_no_motion(self):
        self.client.post_fix("X", 72.0, -95.0, t=100.0)
        _, kw = self.calls[0]
        self.assertIsNone(kw["heading"])
        self.assertIsNone(kw["speed"])

    def test_derives_heading_and_speed(self):
        self.client.post_fix("X", 72.0, -95.0, t=100.0)
        lat2, lon2 = destination(72.0, -95.0, 0.0, 100.0)   # 100 m due north
        self.client.post_fix("X", lat2, lon2, t=110.0)      # over 10 s
        _, kw = self.calls[1]
        self.assertAlmostEqual(kw["heading"], 0.0, delta=1.0)
        self.assertAlmostEqual(kw["speed"], 10.0, delta=0.5)

    def test_no_time_means_no_heading(self):
        self.client.post_fix("X", 72.0, -95.0, t=1.0)
        self.client.post_fix("X", 72.001, -95.0, t=None)
        _, kw = self.calls[1]
        self.assertIsNone(kw["heading"])


class TestThrottle(unittest.TestCase):
    def setUp(self):
        # gentle defaults, no global spacing so the test is fast
        self.client = TrackClient("http://127.0.0.1:9", min_interval=0.0,
                                  post_every_s=5.0, min_move_m=5.0, retries=0)
        self.calls = []
        self.client.post = lambda *a, **k: (self.calls.append((a, k)), {"ok": True})[1]

    def test_skips_soon_and_unmoved(self):
        self.client.post_fix("X", 72.0, -95.0, t=100.0)          # create -> posted
        skipped = self.client.post_fix("X", 72.00001, -95.0, t=101.0)  # 1 s, ~1 m
        self.assertIsNone(skipped)
        self.assertEqual(len(self.calls), 1)

    def test_posts_after_interval(self):
        self.client.post_fix("X", 72.0, -95.0, t=100.0)
        self.client.post_fix("X", 72.0, -95.0, t=106.0)          # 6 s later
        self.assertEqual(len(self.calls), 2)

    def test_posts_after_meaningful_move(self):
        self.client.post_fix("X", 72.0, -95.0, t=100.0)
        lat2, lon2 = destination(72.0, -95.0, 0.0, 50.0)         # moved 50 m
        self.client.post_fix("X", lat2, lon2, t=100.5)           # but only 0.5 s
        self.assertEqual(len(self.calls), 2)


class TestTrackCourseSpeed(unittest.TestCase):
    def _member(self, lat, lon, t, idx):
        return {"lat": lat, "lon": lon, "error_radius_m": 10.0, "index": idx,
                "t_sim": t, "score": 0.5, "depression_deg": 20.0}

    def test_from_members(self):
        lat2, lon2 = destination(72.0, -95.0, 0.0, 100.0)
        track = fuse_estimates([self._member(72.0, -95.0, 100.0, 0),
                                self._member(lat2, lon2, 110.0, 1)])
        hdg, spd = track_course_speed(track)
        self.assertAlmostEqual(hdg, 0.0, delta=1.0)
        self.assertAlmostEqual(spd, 10.0, delta=0.5)

    def test_single_member_returns_none(self):
        track = fuse_estimates([self._member(72.0, -95.0, 100.0, 0)])
        self.assertEqual(track_course_speed(track), (None, None))


if __name__ == "__main__":
    unittest.main()
