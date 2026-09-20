#!/usr/bin/env python3
"""Tests for tactical geodesy helpers: closed-zone rerouting & search spiral."""
import math
import unittest

from arcticlib.geo import (
    distance_m,
    generate_figure8_pattern,
    generate_racetrack_pattern,
    generate_search_spiral,
    reroute_around_closed_zone,
    segment_circle_dist_m,
)


class TestTacticalPlanning(unittest.TestCase):

    def test_segment_circle_dist(self):
        # Center at (71.990, -94.850)
        c_lat, c_lon = 71.990, -94.850
        # Line from West to East passing right through the center
        p1 = (71.990, -94.860)
        p2 = (71.990, -94.840)
        dist = segment_circle_dist_m(p1[0], p1[1], p2[0], p2[1], c_lat, c_lon)
        self.assertLess(dist, 1.0)  # Should be ~0 metres

        # Line offset to the north by ~0.005 deg (~555 metres)
        p3 = (71.995, -94.860)
        p4 = (71.995, -94.840)
        dist_offset = segment_circle_dist_m(p3[0], p3[1], p4[0], p4[1], c_lat, c_lon)
        self.assertAlmostEqual(dist_offset, 555.0, delta=20.0)

    def test_generate_search_spiral(self):
        c_lat, c_lon = 71.985, -94.750
        spiral = generate_search_spiral(c_lat, c_lon, alt=75.0, r0=100.0, dr=130.0, r_max=600.0)
        self.assertGreater(len(spiral), 15)

        # Distances from center must monotonically expand
        distances = [distance_m(w[0], w[1], c_lat, c_lon) for w in spiral]
        self.assertAlmostEqual(distances[0], 100.0, delta=5.0)
        self.assertLessEqual(distances[-1], 650.0)

        for i in range(len(distances) - 1):
            self.assertGreater(distances[i+1], distances[i])

    def test_generate_figure8_pattern(self):
        c_lat, c_lon = 71.985, -94.750
        fig8 = generate_figure8_pattern(c_lat, c_lon, bearing_deg=85.0, length_m=400.0, width_m=160.0, alt=75.0, num_cycles=2)
        # Each cycle has 10 waypoints (2 overflight passes + 2 reversal loops)
        self.assertEqual(len(fig8), 20)

        # Verify overflight waypoints pass directly over center
        overflights = [w for w in fig8 if "Overflight" in w[3]]
        self.assertEqual(len(overflights), 4)
        for ow in overflights:
            dist = distance_m(ow[0], ow[1], c_lat, c_lon)
            self.assertLess(dist, 1.0)  # Exactly at center

        # Verify extension waypoints reach ~400m
        extensions = [w for w in fig8 if "Extension" in w[3]]
        for ew in extensions:
            dist = distance_m(ew[0], ew[1], c_lat, c_lon)
            self.assertAlmostEqual(dist, 400.0, delta=10.0)

    def test_generate_racetrack_pattern(self):
        c_lat, c_lon = 71.985, -94.750
        race = generate_racetrack_pattern(c_lat, c_lon, bearing_deg=85.0, length_m=400.0, width_m=160.0, alt=75.0, num_cycles=2)
        self.assertEqual(len(race), 12)
        overflights = [w for w in race if "Overflight" in w[3]]
        self.assertEqual(len(overflights), 2)
        for ow in overflights:
            dist = distance_m(ow[0], ow[1], c_lat, c_lon)
            self.assertLess(dist, 1.0)

    def test_reroute_around_closed_zone(self):
        # Create a simple synthetic line of waypoints
        c_lat, c_lon = 71.990, -94.850
        cz_radius = 500.0

        raw_wps = [
            (71.990, -94.870, 75.0, "W1"),
            (71.990, -94.855, 75.0, "W2_inside"),
            (71.990, -94.850, 75.0, "W3_center"),
            (71.990, -94.845, 75.0, "W4_inside"),
            (71.990, -94.830, 75.0, "W5"),
        ]

        rerouted, invalidated = reroute_around_closed_zone(
            raw_wps, c_lat, c_lon, cz_radius, safe_buffer_m=50.0
        )

        # W2, W3, W4 should be invalidated
        self.assertEqual(invalidated, [1, 2, 3])

        # All points in rerouted must be > (cz_radius + buffer)
        for wlat, wlon, walt, wname in rerouted:
            dist = distance_m(wlat, wlon, c_lat, c_lon)
            self.assertGreaterEqual(dist, cz_radius + 45.0)

        # There should be detour waypoints inserted between W1 and W5
        detour_wps = [w for w in rerouted if "Detour" in w[3]]
        self.assertGreaterEqual(len(detour_wps), 1)


if __name__ == "__main__":
    unittest.main()
