#!/usr/bin/env python3
"""Pixel -> GPS geolocation for a downward-tilted body-fixed camera.

A pixel is a *direction*, not a point. We cast a ray from the camera through the
pixel, rotate it into the local NED frame, intersect it with flat ground and turn
the horizontal offset into a lat/lon with a WGS84 forward geodesic.

Frame conventions
-----------------
Body frame is **(forward, right, down)** and NED is **(north, east, down)**, so
both have ``+z`` down. Angles follow the rest of ``arcticlib`` (see
``types.py``): **radians**, ``yaw`` clockwise from true north, ``pitch`` positive
nose-up, ``roll`` positive right-wing-down. MAVLink's ``ATTITUDE`` already uses
exactly these signs, so ``Pose.yaw/pitch/roll`` can be passed straight in.

The camera mount is a fixed rotation of the camera relative to the airframe,
reported in the "0 = horizon, -90 = straight down" convention that gimbals
usually use. This sim exposes no gimbal telemetry, but the mount angle is in the
sensor SDF, read live from gzweb ``~/scene``:

======================  =========================  ==========================
asset                   sensor                     mount pitch (down)
======================  =========================  ==========================
``fixed-wing``          ``skywalker_x8/fpv_camera``   -8.021 deg
``quadcopter``          ``gimbal_small_2d/webcam``    -20.002 deg
``tower-1`` / ``tower-2``  ``eo_camera``             derived from servos, not here
======================  =========================  ==========================

The fixed-wing value comes from the SDF quaternion
``(x=0, y=0.069942847, z=0, w=0.997551)`` on ``base_link``: a pure rotation about
body +Y of ``2*atan2(0.069942847, 0.997551) = 8.021 deg``, which tips the camera
``+X`` optical axis downward. The quadcopter value is the composed optical axis of
its (fixed) two-axis gimbal, measured from the same scene dump.

Altitude / AGL
--------------
The maths needs height above the ground **at the target**. We take
``agl = alt_amsl - ground_elevation_m`` by default (the target vessel sits at sea
level, ``z = 0``); ``alt_rel`` is available but only correct if the launch point is
at the target's ground level, so it is treated as a known bias source.

Terrain
-------
Flat-ground only for now. :func:`intersect_ground` is the single seam where a DEM
ray-march would replace the plane later; nothing else assumes flatness.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geo import Georef, bearing_deg, destination, distance_m

log = logging.getLogger("arcticlib.geolocate")

# Fixed camera pitch relative to the airframe, degrees, 0 = horizon, - = down.
# Source: live gzweb ~/scene SDF dump (see module docstring). Override per call.
DEFAULT_CAMERA_PITCH_DEG: dict[str, float] = {
    "fixed-wing": -8.021,
    "quadcopter": -20.002,
    "tower-1": 0.0,   # pointing comes from the pan/tilt servos, not a fixed mount
    "tower-2": 0.0,
    "rover": 0.0,
}


# Empirically calibrated mount, from the 2026-09-19 follow-ship run (300 frames
# with true vessel lat/lon): a grid search minimised median position error on a
# held-out half. The fixed-wing optimum is -8.50 deg, ~0.5 deg steeper than the
# SDF value, likely because the detected red-hull centroid is offset from the
# vessel's model origin. Use when absolute accuracy matters; the SDF value above
# is the physically-grounded default.
CALIBRATED_CAMERA_PITCH_DEG: dict[str, float] = {
    "fixed-wing": -8.50,
    "quadcopter": -20.002,
}


def camera_mount_pitch_deg(asset: str) -> float:
    """Default camera pitch offset for an asset, degrees (0 = horizon)."""
    return DEFAULT_CAMERA_PITCH_DEG.get(asset, 0.0)


def intrinsics_from_fov(width: int, height: int, hfov_deg: float,
                        vfov_deg: float) -> dict:
    """Pinhole intrinsics in pixels from resolution + field of view."""
    return {
        "fx": (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0),
        "fy": (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0),
        "cx": width / 2.0,
        "cy": height / 2.0,
        "width": width,
        "height": height,
    }


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeoEstimate:
    """A geolocated pixel.

    Iterating the object yields ``(lat, lon, error_radius_m)`` so callers can
    write ``lat, lon, err = estimate``.
    """

    lat: float
    lon: float
    error_radius_m: float
    depression_deg: float
    ground_range_m: float
    north_m: float
    east_m: float
    grazing: bool = False

    def __iter__(self):
        yield self.lat
        yield self.lon
        yield self.error_radius_m


# --------------------------------------------------------------------------- #
# Rotation helpers (body/NED share the fwd/right/down, N/E/D convention)
# --------------------------------------------------------------------------- #
def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# Camera (x right, y down, z forward) -> body (forward, right, down).
R_CAM_TO_BODY = np.array([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]], dtype=np.float64)


def camera_ray_ned(u: float, v: float,
                   yaw: float, pitch: float, roll: float,
                   gimbal_pitch: float = 0.0,
                   gimbal_yaw: float = 0.0,
                   gimbal_roll: float = 0.0,
                   intrinsics: Optional[dict] = None) -> np.ndarray:
    """Unit ray through pixel ``(u, v)`` in NED. Angles in radians.

    ``gimbal_pitch`` is the camera's pitch relative to the airframe in the
    "0 = horizon, -90 = straight down" convention. ``yaw/pitch/roll`` are the
    airframe attitude (MAVLink signs). The returned vector is normalised and
    points *down* (``+z``) when the pixel is below the horizon.
    """
    if intrinsics is None:
        raise ValueError("intrinsics are required")
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])

    # 1. pixel -> camera ray (z forward)
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)

    # 2. camera -> body (fixed axis remap)
    d_body = R_CAM_TO_BODY @ d_cam

    # 3. gimbal/mount -> body, then body -> NED
    r_gimbal = _rot_z(gimbal_yaw) @ _rot_y(gimbal_pitch) @ _rot_x(gimbal_roll)
    d_body = r_gimbal @ d_body
    r_ned = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
    d_ned = r_ned @ d_body

    n = float(np.linalg.norm(d_ned))
    return d_ned / n if n > 0 else d_ned


def intersect_ground(d_ned: np.ndarray, agl_m: float
                     ) -> Optional[tuple[float, float, float]]:
    """Intersect a unit NED ray with flat ground ``agl_m`` below the camera.

    Returns ``(north_m, east_m, depression_deg)`` or ``None`` when the ray never
    reaches the ground (points at/above the horizon). This is the seam where a
    terrain ray-march would go.
    """
    if agl_m <= 0.0:
        return None
    dz = float(d_ned[2])
    if dz <= 1e-9:
        return None                      # at or above the horizon: no ground hit
    horiz = math.hypot(float(d_ned[0]), float(d_ned[1]))
    t = agl_m / dz
    north = t * float(d_ned[0])
    east = t * float(d_ned[1])
    depression = math.degrees(math.atan2(dz, horiz))
    return north, east, depression


# --------------------------------------------------------------------------- #
# Core public API
# --------------------------------------------------------------------------- #
def pixel_to_gps(u: float, v: float,
                 drone_lat: float, drone_lon: float, agl_m: float,
                 yaw: float, pitch: float, roll: float,
                 gimbal_pitch: float = 0.0,
                 intrinsics: Optional[dict] = None,
                 gimbal_yaw: float = 0.0, gimbal_roll: float = 0.0,
                 attitude_sigma_deg: float = 0.5,
                 position_sigma_m: float = 3.0,
                 altitude_sigma_m: float = 2.0,
                 min_depression_deg: float = 10.0,
                 reject_grazing: bool = False) -> Optional[GeoEstimate]:
    """Geolocate pixel ``(u, v)`` on flat ground. Angles are in **radians**.

    Parameters
    ----------
    u, v
        Pixel coordinates (``+u`` right, ``+v`` down).
    drone_lat, drone_lon, agl_m
        Camera position and height above the ground **at the target**.
    yaw, pitch, roll
        Airframe attitude, MAVLink signs (radians).
    gimbal_pitch
        Camera pitch relative to the airframe, 0 = horizon, -90 = straight down
        (radians). Use :func:`camera_mount_pitch_deg` for the per-asset default.
    intrinsics
        ``{fx, fy, cx, cy}`` in pixels (see :func:`intrinsics_from_fov`).
    attitude_sigma_deg, position_sigma_m, altitude_sigma_m
        1-sigma input uncertainties used for the error radius.
    min_depression_deg
        Estimates whose ray is shallower than this are flagged ``grazing``.
    reject_grazing
        When True, grazing estimates return ``None`` instead of being flagged.

    Returns
    -------
    :class:`GeoEstimate` or ``None`` when the ray never meets the ground (or is
    rejected as grazing). ``lat, lon, error_radius_m = estimate`` also works.
    """
    if intrinsics is None:
        raise ValueError("intrinsics are required")
    if agl_m is None or agl_m <= 0.0:
        return None

    d_ned = camera_ray_ned(u, v, yaw, pitch, roll,
                           gimbal_pitch=gimbal_pitch, gimbal_yaw=gimbal_yaw,
                           gimbal_roll=gimbal_roll, intrinsics=intrinsics)
    hit = intersect_ground(d_ned, agl_m)
    if hit is None:
        return None
    north, east, depression = hit

    grazing = depression < min_depression_deg
    if grazing and reject_grazing:
        return None

    ground_range = math.hypot(north, east)
    bearing = math.degrees(math.atan2(east, north)) % 360.0
    lat, lon = destination(drone_lat, drone_lon, bearing, ground_range)

    error_radius = _error_radius(
        agl_m=agl_m, depression_deg=depression, ground_range_m=ground_range,
        attitude_sigma_deg=attitude_sigma_deg, position_sigma_m=position_sigma_m,
        altitude_sigma_m=altitude_sigma_m)

    return GeoEstimate(lat=lat, lon=lon, error_radius_m=error_radius,
                       depression_deg=depression, ground_range_m=ground_range,
                       north_m=north, east_m=east, grazing=grazing)


def pixel_to_gps_deg(u: float, v: float,
                     drone_lat: float, drone_lon: float, agl_m: float,
                     yaw_deg: float, pitch_deg: float, roll_deg: float,
                     gimbal_pitch_deg: float = 0.0,
                     intrinsics: Optional[dict] = None,
                     **kwargs) -> Optional[GeoEstimate]:
    """Degree-valued convenience wrapper around :func:`pixel_to_gps`."""
    return pixel_to_gps(
        u, v, drone_lat, drone_lon, agl_m,
        math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg),
        gimbal_pitch=math.radians(gimbal_pitch_deg),
        intrinsics=intrinsics, **kwargs)


def project_to_pixel(lat: float, lon: float,
                     drone_lat: float, drone_lon: float, agl_m: float,
                     yaw: float, pitch: float, roll: float,
                     gimbal_pitch: float = 0.0,
                     intrinsics: Optional[dict] = None,
                     gimbal_yaw: float = 0.0, gimbal_roll: float = 0.0
                     ) -> Optional[tuple[float, float]]:
    """Forward-project a ground lat/lon to a pixel — the inverse of :func:`pixel_to_gps`.

    Angles are radians, same conventions as :func:`pixel_to_gps`. Returns
    ``(u, v)`` or ``None`` when the point is behind the camera. Used to check a
    detection against the true ship position and to aim.
    """
    if intrinsics is None or agl_m is None or agl_m <= 0.0:
        return None
    bearing = math.radians(bearing_deg(drone_lat, drone_lon, lat, lon))
    dist = distance_m(drone_lat, drone_lon, lat, lon)
    north = dist * math.cos(bearing)
    east = dist * math.sin(bearing)
    d_ned = np.array([north, east, agl_m], dtype=np.float64)
    n = float(np.linalg.norm(d_ned))
    if n <= 0:
        return None
    d_ned /= n

    r_ned = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
    r_gimbal = _rot_z(gimbal_yaw) @ _rot_y(gimbal_pitch) @ _rot_x(gimbal_roll)
    d_body = r_ned.T @ d_ned
    d_cam = R_CAM_TO_BODY.T @ r_gimbal.T @ d_body
    if d_cam[2] <= 1e-9:
        return None
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    return (fx * d_cam[0] / d_cam[2] + cx, fy * d_cam[1] / d_cam[2] + cy)


def _error_radius(agl_m: float, depression_deg: float, ground_range_m: float,
                  attitude_sigma_deg: float, position_sigma_m: float,
                  altitude_sigma_m: float) -> float:
    """Approximate 1-sigma radial position error, metres.

    Uses the analytic sensitivity of ground range to depression angle
    (``dr/d(delta) = -h / sin^2(delta)`` — the term that blows up at grazing
    angles), plus altitude sensitivity (``cot(delta)``), lateral yaw error
    (``r * sigma_yaw``) and the input position error, all in quadrature.
    """
    delta = math.radians(max(1e-6, depression_deg))
    sin_d = math.sin(delta)
    sigma_att = math.radians(attitude_sigma_deg)

    range_sigma = agl_m / (sin_d * sin_d) * sigma_att
    altitude_term = (math.cos(delta) / sin_d) * altitude_sigma_m
    lateral_sigma = ground_range_m * sigma_att

    return math.sqrt(range_sigma ** 2 + altitude_term ** 2 +
                     lateral_sigma ** 2 + position_sigma_m ** 2)


def _pget(pose, name: str, default: float = 0.0) -> float:
    """Read a field from a Pose object or a plain dict."""
    if pose is None:
        return default
    if isinstance(pose, dict):
        return pose.get(name, default)
    return getattr(pose, name, default)


@dataclass
class GeoConfig:
    """Shared geolocation settings for the recording/detection tools."""

    alt_ref: str = "amsl"                      # "amsl" or "rel"
    ground_elevation_m: float = 0.0
    camera_pitch_deg: Optional[float] = None   # None -> per-asset SDF default
    min_depression_deg: float = 10.0
    reject_grazing: bool = False
    attitude_sigma_deg: float = 0.5
    position_sigma_m: float = 3.0
    altitude_sigma_m: float = 2.0
    enabled: bool = True

    def agl(self, pose) -> float:
        alt = _pget(pose, "alt_amsl" if self.alt_ref == "amsl" else "alt_rel")
        return float(alt) - self.ground_elevation_m

    def mount_pitch_rad(self, asset: str) -> float:
        if self.camera_pitch_deg is not None:
            return math.radians(float(self.camera_pitch_deg))
        return math.radians(camera_mount_pitch_deg(asset))

    def locate(self, u: float, v: float, pose, asset: str, intrinsics: dict,
               **kwargs) -> Optional[GeoEstimate]:
        """Geolocate one pixel for ``asset`` using this configuration."""
        if not self.enabled or pose is None:
            return None
        return pixel_to_gps(
            u, v, _pget(pose, "lat"), _pget(pose, "lon"), self.agl(pose),
            _pget(pose, "yaw"), _pget(pose, "pitch"), _pget(pose, "roll"),
            gimbal_pitch=self.mount_pitch_rad(asset),
            intrinsics=intrinsics,
            attitude_sigma_deg=self.attitude_sigma_deg,
            position_sigma_m=self.position_sigma_m,
            altitude_sigma_m=self.altitude_sigma_m,
            min_depression_deg=self.min_depression_deg,
            reject_grazing=self.reject_grazing,
            **kwargs)


def geolocate_pose(u: float, v: float, pose, intrinsics: dict, *,
                   asset: Optional[str] = None,
                   camera_pitch_deg: Optional[float] = None,
                   alt_ref: str = "amsl",
                   ground_elevation_m: float = 0.0,
                   gimbal_pitch: Optional[float] = None,
                   gimbal_yaw: Optional[float] = None,
                   **kwargs) -> Optional[GeoEstimate]:
    """Geolocate a pixel using a :class:`~arcticlib.types.Pose`.

    Picks the camera mount from ``asset`` (or an explicit ``camera_pitch_deg`` /
    ``gimbal_pitch``) and the height from ``alt_ref``:

    * ``"amsl"`` (default): ``agl = pose.alt_amsl - ground_elevation_m``
    * ``"rel"``: ``agl = pose.alt_rel - ground_elevation_m``
    """
    if pose is None:
        return None

    if gimbal_pitch is not None:
        g_pitch = gimbal_pitch
    elif camera_pitch_deg is not None:
        g_pitch = math.radians(camera_pitch_deg)
    else:
        g_pitch = math.radians(camera_mount_pitch_deg(asset or pose.asset))

    if alt_ref == "amsl":
        agl = pose.alt_amsl - ground_elevation_m
    elif alt_ref == "rel":
        agl = pose.alt_rel - ground_elevation_m
    else:
        raise ValueError("alt_ref must be 'amsl' or 'rel'")

    return pixel_to_gps(
        u, v, pose.lat, pose.lon, agl,
        pose.yaw, pose.pitch, pose.roll,
        gimbal_pitch=g_pitch,
        gimbal_yaw=0.0 if gimbal_yaw is None else gimbal_yaw,
        intrinsics=intrinsics, **kwargs)


# --------------------------------------------------------------------------- #
# Multi-frame refinement
# --------------------------------------------------------------------------- #
def _dget(d, key: str, default=0.0):
    """Read a field from a dict-like detection record."""
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


@dataclass(frozen=True)
class FusedTrack:
    """A single physical target fused from several per-frame estimates.

    Positions are combined with an **inverse-variance weighted mean**
    (``w_i = 1 / error_radius_i^2``), so a grazing-angle frame with a large
    error radius contributes little. The fused error radius is the 1-sigma of
    that weighted mean, ``1 / sqrt(sum w_i)``.
    """

    lat: float
    lon: float
    error_radius_m: float
    n: int                    # total member detections
    n_frames: int             # distinct frames
    t_first: float
    t_last: float
    mean_score: float
    min_depression_deg: float
    max_depression_deg: float
    members: list             # the per-frame detection records that formed it


def fuse_estimates(members: list, georef: Optional[Georef] = None) -> FusedTrack:
    """Inverse-variance fuse one cluster of per-frame detections."""
    members = [m for m in members if _dget(m, "lat", None) is not None]
    if not members:
        raise ValueError("no members to fuse")
    if georef is None:
        lat0 = sum(float(_dget(m, "lat")) for m in members) / len(members)
        lon0 = sum(float(_dget(m, "lon")) for m in members) / len(members)
        georef = Georef(lat0, lon0)

    wsum = e_sum = n_sum = 0.0
    for m in members:
        sigma = max(1e-3, float(_dget(m, "error_radius_m", 1.0)))
        w = 1.0 / (sigma * sigma)
        e, n, _ = georef.to_enu(float(_dget(m, "lat")), float(_dget(m, "lon")))
        e_sum += w * e
        n_sum += w * n
        wsum += w

    e_bar, n_bar = e_sum / wsum, n_sum / wsum
    lat, lon, _ = georef.to_latlon(e_bar, n_bar)
    fused_sigma = math.sqrt(1.0 / wsum)

    frames = {_dget(m, "index", None) for m in members}
    t_vals = [float(_dget(m, "t_sim", 0.0)) for m in members]
    dep = [float(_dget(m, "depression_deg", 0.0)) for m in members]
    scores = [float(_dget(m, "score", 0.0)) for m in members]
    return FusedTrack(
        lat=lat, lon=lon, error_radius_m=fused_sigma,
        n=len(members), n_frames=len(frames),
        t_first=min(t_vals), t_last=max(t_vals),
        mean_score=sum(scores) / len(scores),
        min_depression_deg=min(dep), max_depression_deg=max(dep),
        members=list(members))


def refine_tracks(detections: list, georef: Optional[Georef] = None,
                  track_gate_m: float = 2000.0, gate_min_m: float = 50.0,
                  gate_sigma: float = 2.0, min_frames: int = 2
                  ) -> list[FusedTrack]:
    """Group per-frame geolocated detections into targets and fuse each group.

    Greedy association in local ENU, processed most-certain-first. Two detections
    associate when their separation is within
    ``clamp(gate_sigma * sqrt(sigma_i^2 + sigma_j^2), gate_min_m, track_gate_m)``
    — i.e. the gate follows the (large) grazing-angle uncertainty but is capped
    so distinct targets do not merge. Only clusters spanning at least
    ``min_frames`` distinct frames are returned as tracks.
    """
    dets = [d for d in detections if _dget(d, "lat", None) is not None]
    if not dets:
        return []
    if georef is None:
        lat0 = sum(float(_dget(d, "lat")) for d in dets) / len(dets)
        lon0 = sum(float(_dget(d, "lon")) for d in dets) / len(dets)
        georef = Georef(lat0, lon0)

    pts = []
    for d in dets:
        e, n, _ = georef.to_enu(float(_dget(d, "lat")), float(_dget(d, "lon")))
        pts.append({"e": e, "n": n, "sigma": max(1e-3, float(_dget(d, "error_radius_m", 1.0))),
                    "d": d})
    order = sorted(range(len(pts)), key=lambda k: pts[k]["sigma"])

    clusters: list[dict] = []
    for k in order:
        p = pts[k]
        best, best_dist = None, float("inf")
        for c in clusters:
            dist = math.hypot(p["e"] - c["e"], p["n"] - c["n"])
            gate = gate_sigma * math.sqrt(p["sigma"] ** 2 + c["sigma"] ** 2)
            gate = max(gate_min_m, min(track_gate_m, gate))
            if dist <= gate and dist < best_dist:
                best, best_dist = c, dist
        if best is None:
            clusters.append({"e": p["e"], "n": p["n"], "sigma": p["sigma"], "members": [p]})
        else:
            best["members"].append(p)
            best["e"] = sum(m["e"] for m in best["members"]) / len(best["members"])
            best["n"] = sum(m["n"] for m in best["members"]) / len(best["members"])
            wsum = sum(1.0 / m["sigma"] ** 2 for m in best["members"])
            best["sigma"] = math.sqrt(1.0 / wsum)

    tracks = []
    for c in clusters:
        frames = {_dget(m["d"], "index", None) for m in c["members"]}
        if len(frames) < min_frames:
            continue
        tracks.append(fuse_estimates([m["d"] for m in c["members"]], georef))
    tracks.sort(key=lambda t: (t.n_frames, t.n), reverse=True)
    return tracks
