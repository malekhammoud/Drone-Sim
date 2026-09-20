"""Unit tests for the two-step helpers (standoff geometry, track handoff)."""
import json
import math
import os
import tempfile
import unittest

from arcticlib.geo import bearing_deg, destination, distance_m
from tools.quad_follow_ship import load_target_from_tracks, standoff_point

LAT0, LON0 = 72.0, -95.0


class TestStandoff(unittest.TestCase):
    def test_hold_point_is_standoff_from_ship_on_quad_side(self):
        # Quad 200 m north of the ship -> hold 100 m north of the ship.
        quad = destination(LAT0, LON0, 0.0, 200.0)
        hl, ho, yaw = standoff_point(LAT0, LON0, quad[0], quad[1], 100.0)
        self.assertAlmostEqual(distance_m(LAT0, LON0, hl, ho), 100.0, delta=1.0)
        # Hold point should be north of the ship (same side as the quad).
        self.assertGreater(hl, LAT0)
        # Yaw points from the hold point back at the ship: due south.
        self.assertAlmostEqual(yaw, 180.0, delta=1.0)

    def test_hold_point_follows_side(self):
        # Quad to the east -> hold east of the ship, yaw west.
        quad = destination(LAT0, LON0, 90.0, 300.0)
        hl, ho, yaw = standoff_point(LAT0, LON0, quad[0], quad[1], 50.0)
        self.assertAlmostEqual(distance_m(LAT0, LON0, hl, ho), 50.0, delta=1.0)
        self.assertGreater(ho, LON0)
        self.assertAlmostEqual(yaw, 270.0, delta=1.0)

    def test_zero_distance_defaults_north(self):
        hl, ho, yaw = standoff_point(LAT0, LON0, LAT0, LON0, 100.0)
        self.assertAlmostEqual(distance_m(LAT0, LON0, hl, ho), 100.0, delta=1.0)
        self.assertAlmostEqual(yaw, 180.0, delta=1.0)


class TestTrackHandoff(unittest.TestCase):
    def test_picks_most_observed_track(self):
        tracks = [
            {"lat": 71.0, "lon": -95.0, "n": 3, "n_frames": 2},
            {"lat": 72.0, "lon": -94.0, "n": 50, "n_frames": 20},
            {"lat": 70.0, "lon": -96.0, "n": 40, "n_frames": 19},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            for t in tracks:
                fh.write(json.dumps(t) + "\n")
            path = fh.name
        try:
            self.assertEqual(load_target_from_tracks(path), (72.0, -94.0))
        finally:
            os.unlink(path)

    def test_empty_file_returns_none(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            path = fh.name
        try:
            self.assertIsNone(load_target_from_tracks(path))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
