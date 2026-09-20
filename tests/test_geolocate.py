"""Unit tests for arcticlib.geolocate — hand-checkable geometry.

Run: python -m unittest discover -s tests
"""
import math
import unittest

from arcticlib.geo import bearing_deg, distance_m
from arcticlib.geolocate import (GeoEstimate, camera_mount_pitch_deg,
                                 intrinsics_from_fov, intersect_ground,
                                 pixel_to_gps, pixel_to_gps_deg,
                                 project_to_pixel)

# A camera similar to the fixed-wing FPV sensor.
INTR = intrinsics_from_fov(1280, 720, 68.98, 42.19)
LAT0, LON0 = 72.0, -95.0
H = 50.0


def _assert_offset(est: GeoEstimate, north: float, east: float, tol: float = 1.0):
    if est is None:
        raise AssertionError("expected an estimate, got None")
    # Compare via geodesy so a degree/lat-lon mix-up cannot pass.
    dn = distance_m(LAT0, LON0, est.lat, est.lon)
    br = bearing_deg(LAT0, LON0, est.lat, est.lon) if dn > 1e-6 else 0.0
    want_d = math.hypot(north, east)
    want_b = math.degrees(math.atan2(east, north)) % 360.0 if want_d > 1e-6 else 0.0
    if want_d < 1e-6:
        assert abs(est.north_m) < 1e-6 and abs(est.east_m) < 1e-6
    else:
        assert abs(dn - want_d) < tol, f"distance {dn:.3f} != {want_d:.3f}"
        db = abs((br - want_b + 180.0) % 360.0 - 180.0)
        assert db < 0.5, f"bearing {br:.3f} != {want_b:.3f}"


class TestHandCheckedCases(unittest.TestCase):
    def test_straight_down_center_is_directly_below(self):
        # Camera pointing straight down (pitch -90 deg), center pixel.
        est = pixel_to_gps_deg(INTR["cx"], INTR["cy"], LAT0, LON0, H,
                               yaw_deg=0, pitch_deg=-90, roll_deg=0,
                               gimbal_pitch_deg=0, intrinsics=INTR)
        self.assertIsNotNone(est)
        _assert_offset(est, 0.0, 0.0)
        self.assertAlmostEqual(est.depression_deg, 90.0, places=6)
        self.assertAlmostEqual(est.ground_range_m, 0.0, places=6)

    def test_45deg_depression_yaw0_is_50m_north(self):
        est = pixel_to_gps_deg(INTR["cx"], INTR["cy"], LAT0, LON0, H,
                               yaw_deg=0, pitch_deg=-45, roll_deg=0,
                               gimbal_pitch_deg=0, intrinsics=INTR)
        self.assertIsNotNone(est)
        _assert_offset(est, north=50.0, east=0.0)
        self.assertAlmostEqual(est.depression_deg, 45.0, places=6)

    def test_45deg_depression_yaw90_is_50m_east(self):
        est = pixel_to_gps_deg(INTR["cx"], INTR["cy"], LAT0, LON0, H,
                               yaw_deg=90, pitch_deg=-45, roll_deg=0,
                               gimbal_pitch_deg=0, intrinsics=INTR)
        self.assertIsNotNone(est)
        _assert_offset(est, north=0.0, east=50.0)

    def test_pixel_above_horizon_returns_none(self):
        # Level camera, pixel above center -> ray points at/above the horizon.
        est = pixel_to_gps_deg(INTR["cx"], INTR["cy"] - 100, LAT0, LON0, H,
                               yaw_deg=0, pitch_deg=0, roll_deg=0,
                               gimbal_pitch_deg=0, intrinsics=INTR)
        self.assertIsNone(est)

    def test_zero_altitude_returns_none(self):
        est = pixel_to_gps_deg(INTR["cx"], INTR["cy"], LAT0, LON0, 0.0,
                               yaw_deg=0, pitch_deg=-45, roll_deg=0,
                               gimbal_pitch_deg=0, intrinsics=INTR)
        self.assertIsNone(est)

    def test_roll_sign_hand_worked(self):
        # Nose straight down (pitch -90), rolled +90 deg (right wing down).
        # A pixel to the image right maps to body right -> body down -> south.
        est = pixel_to_gps_deg(INTR["cx"] + 100, INTR["cy"], LAT0, LON0, H,
                               yaw_deg=0, pitch_deg=-90, roll_deg=90,
                               gimbal_pitch_deg=0, intrinsics=INTR)
        self.assertIsNotNone(est)
        # north is negative (south of the drone), east ~ 0.
        self.assertLess(est.north_m, -0.1)
        self.assertAlmostEqual(est.east_m, 0.0, places=6)

    def test_pitch_sign_hand_worked(self):
        # Nose down 30 deg with the camera body-aligned: depression is 30 deg, so
        # the target is north at ground range h / tan(30) = 86.60 m at h = 50.
        est = pixel_to_gps_deg(INTR["cx"], INTR["cy"], LAT0, LON0, H,
                               yaw_deg=0, pitch_deg=-30, roll_deg=0,
                               gimbal_pitch_deg=0, intrinsics=INTR)
        self.assertIsNotNone(est)
        _assert_offset(est, north=H / math.tan(math.radians(30)), east=0.0)


class TestMountAndGimbal(unittest.TestCase):
    def test_mount_pitch_table(self):
        self.assertAlmostEqual(camera_mount_pitch_deg("fixed-wing"), -8.021)
        self.assertAlmostEqual(camera_mount_pitch_deg("quadcopter"), -20.002)
        self.assertEqual(camera_mount_pitch_deg("unknown-asset"), 0.0)

    def test_gimbal_down_tilt_shifts_center_pixel_below_forward(self):
        # Body level; camera tilted 8 deg down -> center pixel hits ground ahead.
        est = pixel_to_gps_deg(INTR["cx"], INTR["cy"], LAT0, LON0, H,
                               yaw_deg=0, pitch_deg=0, roll_deg=0,
                               gimbal_pitch_deg=-8.021, intrinsics=INTR)
        self.assertIsNotNone(est)
        self.assertAlmostEqual(est.depression_deg, 8.021, places=2)
        self.assertAlmostEqual(est.north_m, H / math.tan(math.radians(8.021)),
                               delta=0.5)

    def test_project_to_pixel_is_inverse(self):
        # Geolocate a pixel, then forward-project the result: should round-trip.
        yaw, pitch, roll, gp = (math.radians(30), math.radians(-40),
                                math.radians(10), math.radians(-8.021))
        for u, v in [(INTR["cx"], INTR["cy"]),
                     (INTR["cx"] + 200, INTR["cy"] + 100),
                     (INTR["cx"] - 150, INTR["cy"] + 50)]:
            est = pixel_to_gps(u, v, LAT0, LON0, H, yaw, pitch, roll,
                               gimbal_pitch=gp, intrinsics=INTR)
            self.assertIsNotNone(est)
            uv = project_to_pixel(est.lat, est.lon, LAT0, LON0, H, yaw, pitch, roll,
                                  gimbal_pitch=gp, intrinsics=INTR)
            self.assertIsNotNone(uv)
            self.assertAlmostEqual(uv[0], u, delta=1.0)
            self.assertAlmostEqual(uv[1], v, delta=1.0)

    def test_camera_pitch_offset_radians_matches_degrees(self):
        a = pixel_to_gps(INTR["cx"], INTR["cy"], LAT0, LON0, H,
                         yaw=0.0, pitch=0.0, roll=0.0,
                         gimbal_pitch=math.radians(-8.021), intrinsics=INTR)
        b = pixel_to_gps_deg(INTR["cx"], INTR["cy"], LAT0, LON0, H,
                             yaw_deg=0, pitch_deg=0, roll_deg=0,
                             gimbal_pitch_deg=-8.021, intrinsics=INTR)
        self.assertAlmostEqual(a.lat, b.lat, places=9)
        self.assertAlmostEqual(a.lon, b.lon, places=9)


class TestGrazingAndUncertainty(unittest.TestCase):
    def test_grazing_flagged_and_rejected(self):
        # 5 deg depression is below the 10 deg default threshold.
        est = pixel_to_gps_deg(INTR["cx"], INTR["cy"], LAT0, LON0, H,
                               yaw_deg=0, pitch_deg=-5, roll_deg=0,
                               gimbal_pitch_deg=0, intrinsics=INTR)
        self.assertIsNotNone(est)
        self.assertTrue(est.grazing)
        rej = pixel_to_gps_deg(INTR["cx"], INTR["cy"], LAT0, LON0, H,
                               yaw_deg=0, pitch_deg=-5, roll_deg=0,
                               gimbal_pitch_deg=0, intrinsics=INTR,
                               reject_grazing=True)
        self.assertIsNone(rej)

    def test_error_grows_toward_grazing(self):
        steep = pixel_to_gps_deg(INTR["cx"], INTR["cy"], LAT0, LON0, H,
                                 yaw_deg=0, pitch_deg=-60, roll_deg=0,
                                 gimbal_pitch_deg=0, intrinsics=INTR)
        shallow = pixel_to_gps_deg(INTR["cx"], INTR["cy"], LAT0, LON0, H,
                                   yaw_deg=0, pitch_deg=-12, roll_deg=0,
                                   gimbal_pitch_deg=0, intrinsics=INTR)
        self.assertGreater(shallow.error_radius_m, steep.error_radius_m)

    def test_tuple_unpacking(self):
        est = pixel_to_gps_deg(INTR["cx"], INTR["cy"], LAT0, LON0, H,
                               yaw_deg=0, pitch_deg=-45, roll_deg=0,
                               gimbal_pitch_deg=0, intrinsics=INTR)
        lat, lon, err = est
        self.assertEqual((lat, lon), (est.lat, est.lon))
        self.assertEqual(err, est.error_radius_m)


class TestFusion(unittest.TestCase):
    def _rec(self, lat, lon, sigma, index, t=0.0, score=0.5, dep=20.0):
        return {"lat": lat, "lon": lon, "error_radius_m": sigma, "index": index,
                "t_sim": t, "score": score, "depression_deg": dep}

    def test_inverse_variance_favours_precise_fix(self):
        from arcticlib.geo import Georef
        from arcticlib.geolocate import fuse_estimates
        g = Georef(LAT0, LON0)
        # precise at origin, sloppy 500 m north: weighted mean stays near origin.
        members = [self._rec(LAT0, LON0, 10.0, 0),
                   self._rec(LAT0 + 500.0 / 111195.0, LON0, 1000.0, 1)]
        t = fuse_estimates(members, g)
        self.assertLess(distance_m(LAT0, LON0, t.lat, t.lon), 60.0)
        self.assertLess(t.error_radius_m, 10.0)

    def test_same_location_reduces_error(self):
        from arcticlib.geo import Georef
        from arcticlib.geolocate import fuse_estimates
        g = Georef(LAT0, LON0)
        members = [self._rec(LAT0, LON0, 100.0, i, t=float(i)) for i in range(4)]
        t = fuse_estimates(members, g)
        self.assertLess(distance_m(LAT0, LON0, t.lat, t.lon), 1.0)
        self.assertAlmostEqual(t.error_radius_m, 50.0, delta=1.0)  # 100/sqrt(4)

    def test_track_needs_multiple_frames(self):
        from arcticlib.geolocate import refine_tracks
        same_frame = [self._rec(LAT0, LON0, 50.0, 0), self._rec(LAT0, LON0, 50.0, 0)]
        self.assertEqual(refine_tracks(same_frame, min_frames=2), [])
        two_frames = [self._rec(LAT0, LON0, 50.0, 0), self._rec(LAT0, LON0, 50.0, 1)]
        tracks = refine_tracks(two_frames, min_frames=2)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].n_frames, 2)

    def test_far_apart_detections_stay_separate(self):
        from arcticlib.geolocate import refine_tracks
        dets = [self._rec(LAT0, LON0, 20.0, 0), self._rec(LAT0, LON0, 20.0, 1),
                self._rec(LAT0 + 5000.0 / 111195.0, LON0, 20.0, 0),
                self._rec(LAT0 + 5000.0 / 111195.0, LON0, 20.0, 1)]
        tracks = refine_tracks(dets, track_gate_m=2000.0, min_frames=2)
        self.assertEqual(len(tracks), 2)


class TestIntersectGround(unittest.TestCase):
    def test_upward_ray_returns_none(self):
        self.assertIsNone(intersect_ground([0.0, 0.0, -1.0], 50.0))
        self.assertIsNone(intersect_ground([0.0, 0.0, 0.0], 50.0))

    def test_nadir_hits_directly_below(self):
        n, e, dep = intersect_ground([0.0, 0.0, 1.0], 50.0)
        self.assertAlmostEqual(n, 0.0)
        self.assertAlmostEqual(e, 0.0)
        self.assertAlmostEqual(dep, 90.0)


if __name__ == "__main__":
    unittest.main()
