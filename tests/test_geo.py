"""Unit tests for arcticlib.geo. Run: python -m unittest discover -s tests"""
import math
import unittest

from arcticlib.geo import (Georef, bearing_deg, destination, distance_m,
                           latlon_to_ps, ps_to_latlon)

# Site centre and EPSG:3413 centre for Fort Ross (from /api/site).
LAT0, LON0 = 71.991960, -94.822428
PSX, PSY = -1502373.733, -1268596.501


class TestDistanceBearing(unittest.TestCase):
    def test_one_degree_latitude(self):
        d = distance_m(72.0, -95.0, 73.0, -95.0)
        self.assertAlmostEqual(d, 111_195, delta=500)   # ~111.2 km

    def test_longitude_shrinks_with_latitude(self):
        # At 72N a degree of longitude is ~0.31 of a degree of latitude.
        dlon = distance_m(72.0, -95.0, 72.0, -94.0)
        dlat = distance_m(72.0, -95.0, 73.0, -95.0)
        self.assertLess(abs(dlon / dlat - 0.309), 0.01)

    def test_bearing_cardinal(self):
        self.assertAlmostEqual(bearing_deg(72.0, -95.0, 73.0, -95.0), 0.0, delta=0.5)
        self.assertAlmostEqual(bearing_deg(72.0, -95.0, 71.0, -95.0), 180.0, delta=0.5)
        self.assertAlmostEqual(bearing_deg(72.0, -95.0, 72.0, -94.0), 90.0, delta=1.0)
        self.assertAlmostEqual(bearing_deg(72.0, -95.0, 72.0, -96.0), 270.0, delta=1.0)

    def test_destination_round_trip(self):
        lat, lon = destination(72.0, -95.0, 37.0, 2500.0)
        self.assertAlmostEqual(bearing_deg(72.0, -95.0, lat, lon), 37.0, delta=0.1)
        self.assertAlmostEqual(distance_m(72.0, -95.0, lat, lon), 2500.0, delta=1.0)


class TestGeoref(unittest.TestCase):
    def setUp(self):
        self.g = Georef(LAT0, LON0, ps_centre_x=PSX, ps_centre_y=PSY)

    def test_origin_is_zero(self):
        e, n, u = self.g.to_enu(LAT0, LON0, 0.0)
        self.assertAlmostEqual(e, 0.0, places=6)
        self.assertAlmostEqual(n, 0.0, places=6)
        self.assertAlmostEqual(u, 0.0, places=6)

    def test_round_trip(self):
        for east, north in [(1000, 0), (0, 1000), (-2500, 1800), (500, -900)]:
            lat, lon, _ = self.g.to_latlon(east, north)
            e2, n2, _ = self.g.to_enu(lat, lon)
            self.assertAlmostEqual(east, e2, delta=0.5)
            self.assertAlmostEqual(north, n2, delta=0.5)

    def test_enu_scale_matches_distance(self):
        lat, lon, _ = self.g.to_latlon(1500.0, 1500.0)
        # ENU is a local tangent plane while distance_m is great-circle, so over
        # 2.1 km they differ by ~0.3% by construction. Assert to that tolerance.
        self.assertAlmostEqual(distance_m(LAT0, LON0, lat, lon),
                               math.hypot(1500, 1500), delta=15.0)

    def test_world_centre_maps_to_origin(self):
        lat, lon = self.g.world_to_latlon(0.0, 0.0)
        self.assertAlmostEqual(lat, LAT0, delta=1e-4)
        self.assertAlmostEqual(lon, LON0, delta=1e-4)


class TestPolarStereographic(unittest.TestCase):
    def test_ps_centre_round_trip(self):
        lat, lon = ps_to_latlon(PSX, PSY)
        self.assertAlmostEqual(lat, LAT0, delta=1e-4)
        self.assertAlmostEqual(lon, LON0, delta=1e-4)

    def test_forward_inverse_round_trip(self):
        for lat, lon in [(72.0, -95.0), (71.5, -94.0), (72.5, -95.5)]:
            x, y = latlon_to_ps(lat, lon)
            lat2, lon2 = ps_to_latlon(x, y)
            self.assertAlmostEqual(lat, lat2, places=7)
            self.assertAlmostEqual(lon, lon2, places=7)


if __name__ == "__main__":
    unittest.main()
