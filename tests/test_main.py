"""Tests for the full-mission orchestrator wiring (no sim needed)."""
import unittest

from main import _patrol_ns, _quad_ns, build_parser


class TestMissionWiring(unittest.TestCase):
    def test_parser_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.patrol_duration, 180.0)
        self.assertEqual(args.quad_alt, 25.0)
        self.assertEqual(args.view_depression, 45.0)
        self.assertFalse(args.follow_ship)
        self.assertFalse(args.skip_patrol)

    def test_patrol_namespace_fields(self):
        args = build_parser().parse_args(
            ["--patrol-duration", "120", "--patrol-alt", "80",
             "--follow-ship", "--margin", "90"])
        ns = _patrol_ns(args)
        self.assertEqual(ns.duration, 120.0)
        self.assertEqual(ns.alt, 80.0)
        self.assertTrue(ns.follow_ship)
        self.assertEqual(ns.margin, 90.0)
        for f in ("spacing_lon", "hz", "out", "model", "track_gate",
                  "min_track_hits", "ground_elevation", "alt_ref", "no_geolocate"):
            self.assertTrue(hasattr(ns, f), f)

    def test_quad_namespace_fields(self):
        args = build_parser().parse_args(
            ["--quad-alt", "30", "--quad-duration", "60",
             "--view-depression", "50", "--follow-ship"])
        ns = _quad_ns(args)
        self.assertEqual(ns.alt, 30.0)
        self.assertEqual(ns.duration, 60.0)
        self.assertEqual(ns.view_depression, 50.0)
        self.assertTrue(ns.from_ship)   # --follow-ship implies DEV truth steering
        for f in ("speed", "update_period", "out", "model", "publish_tracks",
                  "track_name", "takeoff_timeout", "no_geolocate"):
            self.assertTrue(hasattr(ns, f), f)


if __name__ == "__main__":
    unittest.main()
