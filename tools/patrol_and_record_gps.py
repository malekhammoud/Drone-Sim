#!/usr/bin/env python3
"""Autonomous Fixed-Wing Lawnmower Patrol, Recorder, Video Generator + GPS.

Merges the remote patrol rework (safe 3-tier strait waypoints, 3-stage
Color+CNN+Temporal detector) with the local trig-GPS additions:

1. **GPS + altitude logging.** Every recorded frame carries a ``gps`` block and a
   ``gps_track.csv`` is written next to the frames.
2. **Geolocated detections.** Each detector hit is turned into lat/lon with
   :mod:`arcticlib.geolocate`, annotated on the video HUD and written to
   ``detections.jsonl`` (with error radius + grazing flag). ``tracks.jsonl``
   holds inverse-variance fused tracks.

``run_patrol()`` is the reusable entry point (used by the root ``main.py``); the
CLI below runs the patrol on its own.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import math
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from arcticlib.config import load_config
from arcticlib.fleet import Fleet
from arcticlib.geo import Georef, distance_m
from arcticlib.geolocate import GeoConfig, refine_tracks, track_course_speed
from tools.detect_verified import VerifiedDetector
from tools.detect_verified_gps import draw_geolocated

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("patrol_gps")

DEV = os.environ.get("ARCTICSIM_DEV") == "1"


def _geo_label(est) -> Optional[str]:
    if est is None:
        return None
    return (f"{est.lat:.5f},{est.lon:.5f} +/-{est.error_radius_m:.0f}m "
            f"d{est.depression_deg:.0f}{' GRAZ' if est.grazing else ''}")


def generate_safe_strait_waypoints(step_lon: float = 0.01,
                                   safe_margin_m: float = 120.0
                                   ) -> list[tuple[float, float, float, str]]:
    """N-S zig-zag waypoints with a safe margin and a 3-tier altitude profile."""
    coast_profile = [
        (-94.920, 71.99317, 71.97693, None), (-94.910, 71.99508, 71.97635, None),
        (-94.900, 71.99359, 71.97688, None), (-94.890, 71.99428, 71.97863, None),
        (-94.880, 71.99497, 71.97826, None), (-94.870, 71.99625, 71.98017, None),
        (-94.860, 71.99970, 71.97943, None), (-94.850, 72.00506, 71.98065, 72.00506),
        (-94.840, 72.00450, 71.98176, 72.00450), (-94.830, 71.99933, 71.98197, 72.00200),
        (-94.820, 71.99832, 71.98309, None), (-94.810, 71.99906, 71.98420, None),
        (-94.800, 72.00453, 71.98452, None), (-94.790, 72.00490, 71.98463, None),
        (-94.780, 72.00612, 71.98436, None), (-94.770, 72.00824, 71.98415, None),
        (-94.760, 72.00819, 71.98473, None), (-94.750, 72.00861, 71.99041, None),
        (-94.740, 72.01137, 71.99009, None), (-94.730, 72.01201, 71.99030, None),
        (-94.720, 72.01222, 71.99280, None), (-94.710, 72.01164, 71.99296, None),
        (-94.700, 72.01328, 71.99370, None), (-94.690, 72.01434, 71.99704, None),
    ]
    c_lons = [p[0] for p in coast_profile]
    c_north = [p[1] for p in coast_profile]
    c_south = [p[2] for p in coast_profile]
    margin_deg = safe_margin_m / 111320.0
    col_lons = np.round(np.arange(-94.92, -94.69 + 0.0001, step_lon), 4)
    waypoints: list[tuple[float, float, float, str]] = []

    for i, l in enumerate(col_lons):
        n_lat = float(np.interp(l, c_lons, c_north))
        s_lat = float(np.interp(l, c_lons, c_south))
        is_island = (-94.855 <= l <= -94.830)
        w_top_lat = n_lat - margin_deg
        w_bot_lat = s_lat + margin_deg
        w_mid_lat = (w_top_lat + w_bot_lat) / 2.0
        if is_island:
            alt_top = alt_mid = alt_bot = 125.0
            col_tag = f"Col {i+1} (Island 125m)"
        else:
            alt_top, alt_mid, alt_bot = 100.0, 75.0, 100.0
            col_tag = f"Col {i+1}"
        if i % 2 == 0:
            waypoints += [(w_top_lat, float(l), alt_top, f"{col_tag} North"),
                          (w_mid_lat, float(l), alt_mid, f"{col_tag} Center"),
                          (w_bot_lat, float(l), alt_bot, f"{col_tag} South")]
        else:
            waypoints += [(w_bot_lat, float(l), alt_bot, f"{col_tag} South"),
                          (w_mid_lat, float(l), alt_mid, f"{col_tag} Center"),
                          (w_top_lat, float(l), alt_top, f"{col_tag} North")]

    user_adjustments = {1: -0.0025, 3: +0.0025, 4: +0.0030, 6: -0.0025, 9: +0.0025,
                        21: +0.0025, 24: -0.0025, 25: -0.0025, 51: +0.0022,
                        57: +0.0025, 58: +0.0026}
    return [(wlat + user_adjustments.get(idx, 0.0), wlon, walt, wname)
            for idx, (wlat, wlon, walt, wname) in enumerate(waypoints, start=1)]


def render_patrol_video(frames_dir: str, sidecar_path: str, output_video_path: str,
                        detector: Optional[VerifiedDetector] = None,
                        geo: Optional[GeoConfig] = None,
                        asset: str = "fixed-wing", fps: float = 4.0,
                        detections_path: Optional[str] = None,
                        track_gate_m: float = 2000.0,
                        min_track_hits: int = 2) -> list:
    """3-stage detect + geolocate + HUD; write MP4, detections.jsonl, tracks.jsonl.

    Returns the list of fused :class:`~arcticlib.geolocate.FusedTrack`.
    """
    if not os.path.exists(sidecar_path):
        log.error("Sidecar not found: %s", sidecar_path)
        return []
    if detector is None:
        detector = VerifiedDetector(model_path="models/patch_verifier.pt",
                                    min_color_score=0.25, min_verify_prob=0.50,
                                    enable_temporal=True, min_hits=8)
    if geo is None:
        geo = GeoConfig()

    log.info("Processing frames with 3-stage detector + GPS: %s", output_video_path)
    entries = []
    with open(sidecar_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    if not entries:
        log.warning("No entries in sidecar file.")
        return []

    first_img = cv2.imread(os.path.join(frames_dir, os.path.basename(entries[0]["frame"])))
    if first_img is None:
        log.error("Cannot read first frame.")
        return []
    h, w = first_img.shape[:2]
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    det_file = open(detections_path, "w") if detections_path else None
    geo_records: list[dict] = []
    total_confirmed = 0
    total_frames = len(entries)

    try:
        for idx, entry in enumerate(entries):
            img = cv2.imread(os.path.join(frames_dir, os.path.basename(entry["frame"])))
            if img is None:
                continue
            pose = entry.get("pose") or {}
            intr = entry.get("camera") or {}
            t_sim = entry.get("t_sim", 0.0)

            candidates = detector.detect(img, frame_idx=idx + 1, t_sim=t_sim, pose=pose)
            confirmed = [c for c in candidates if getattr(c, "is_confirmed", False)]
            if confirmed:
                total_confirmed += 1

            estimates = [geo.locate(c.cx, c.cy, pose, asset, intr) for c in candidates]
            labels = [_geo_label(e) for e in estimates]
            for c, e in zip(candidates, estimates):
                if e is None:
                    continue
                rec = {"frame": entry.get("frame"), "index": entry.get("index"),
                       "t_sim": t_sim, "asset": asset, "u": c.cx, "v": c.cy,
                       "score": c.score, "confirmed": bool(getattr(c, "is_confirmed", False)),
                       "lat": e.lat, "lon": e.lon, "error_radius_m": e.error_radius_m,
                       "depression_deg": e.depression_deg, "ground_range_m": e.ground_range_m,
                       "grazing": e.grazing}
                geo_records.append(rec)
                if det_file is not None:
                    det_file.write(json.dumps(rec) + "\n")

            gt = entry.get("groundtruth", {})
            gt_pt = gt.get("point_px_approx")
            vis = draw_geolocated(detector, img, candidates, labels=labels,
                                  gt_pt=gt_pt if (gt_pt and 0 <= gt_pt[0] < w and 0 <= gt_pt[1] < h) else None)

            alt = pose.get("alt_rel", 0.0)
            roll = math.degrees(pose.get("roll", 0.0))
            pitch = math.degrees(pose.get("pitch", 0.0))
            yaw = math.degrees(pose.get("yaw", 0.0)) % 360.0
            speed = math.hypot(pose.get("vx", 0.0), pose.get("vy", 0.0))
            gps = entry.get("gps") or {}

            cv2.rectangle(vis, (10, 10), (470, 170), (20, 20, 20), -1)
            cv2.rectangle(vis, (10, 10), (470, 170), (80, 80, 80), 1)
            cv2.putText(vis, "ARCTIC PATROL - FIXED WING + GPS", (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(vis, f"Sim Time: {t_sim:.1f}s | Frame: {idx+1}/{total_frames}", (20, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(vis, f"Alt: {alt:.1f}m | Speed: {speed:.1f} m/s | Yaw: {yaw:.0f} deg", (20, 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(vis, f"Roll: {roll:+.1f} deg | Pitch: {pitch:+.1f} deg", (20, 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(vis, f"GPS: {gps.get('lat', 0.0):.5f}, {gps.get('lon', 0.0):.5f}"
                             f" | AGL: {gps.get('agl_m', 0.0):.0f}m", (20, 114),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 255), 1)
            if confirmed:
                c0 = confirmed[0]
                e0 = geo.locate(c0.cx, c0.cy, pose, asset, intr)
                status = (f"CONFIRMED #{c0.track_id} ({c0.hits}h {c0.score:.2f}) -> "
                          f"{e0.lat:.5f},{e0.lon:.5f} +/-{e0.error_radius_m:.0f}m"
                          if e0 is not None else
                          f"CONFIRMED #{c0.track_id} ({c0.hits}h {c0.score:.2f}) - no fix")
                cv2.putText(vis, status, (20, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
            elif candidates:
                c0 = candidates[0]
                cv2.putText(vis, f"TENTATIVE #{getattr(c0, 'track_id', '?')} "
                                 f"[{getattr(c0, 'hits', 1)}/8 hits]", (20, 148),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            else:
                cv2.putText(vis, "SEARCHING... (No anomalies)", (20, 148),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            writer.write(vis)
    finally:
        writer.release()
        if det_file is not None:
            det_file.close()

    tracks = []
    if geo_records:
        tracks = refine_tracks(geo_records, track_gate_m=track_gate_m, min_frames=min_track_hits)
        if detections_path:
            tracks_path = os.path.join(os.path.dirname(detections_path), "tracks.jsonl")
            with open(tracks_path, "w") as fh:
                for t in tracks:
                    fh.write(json.dumps({
                        "lat": t.lat, "lon": t.lon, "error_radius_m": t.error_radius_m,
                        "n": t.n, "n_frames": t.n_frames, "t_first": t.t_first,
                        "t_last": t.t_last, "mean_score": t.mean_score,
                        "min_depression_deg": t.min_depression_deg,
                        "max_depression_deg": t.max_depression_deg}) + "\n")
            log.info("Fused %d detections into %d track(s) -> %s",
                     len(geo_records), len(tracks), tracks_path)
    log.info("Finished rendering: %s (confirmed frames: %d/%d, geolocated: %d)",
             output_video_path, total_confirmed, total_frames, len(geo_records))
    return tracks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-wing patrol, GPS recorder & geolocated video.")
    parser.add_argument("--alt", type=float, default=90.0)
    parser.add_argument("--margin", type=float, default=120.0)
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--spacing-lon", type=float, default=0.01)
    parser.add_argument("--hz", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--out", default="patrol_output")
    parser.add_argument("--no-fly", action="store_true")
    parser.add_argument("--follow-ship", action="store_true")
    parser.add_argument("--ground-elevation", type=float, default=0.0)
    parser.add_argument("--alt-ref", choices=["amsl", "rel"], default="amsl")
    parser.add_argument("--camera-pitch-deg", type=float, default=None)
    parser.add_argument("--min-depression", type=float, default=10.0)
    parser.add_argument("--reject-grazing", action="store_true")
    parser.add_argument("--attitude-sigma", type=float, default=0.5)
    parser.add_argument("--position-sigma", type=float, default=3.0)
    parser.add_argument("--altitude-sigma", type=float, default=2.0)
    parser.add_argument("--no-geolocate", action="store_true")
    parser.add_argument("--track-gate", type=float, default=2000.0)
    parser.add_argument("--min-track-hits", type=int, default=2)
    parser.add_argument("--model", default="models/patch_verifier.pt")
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--min-hits", type=int, default=8)
    parser.add_argument("--publish-tracks", action="store_true",
                        help="POST the best fused track to /api/tracks")
    parser.add_argument("--track-name", default="Sierra One")
    return parser


def run_patrol(fleet: Fleet, args, geo: Optional[GeoConfig] = None, gt=None) -> dict:
    """Fly the wing patrol (or chase the ship), record GPS + frames, render.

    Reuses the caller's :class:`Fleet` and optional ``GroundTruth``; does not
    shut them down. Returns paths plus the fused tracks and best handoff target.
    """
    cfg = fleet.config
    georef = Georef(cfg.origin_lat, cfg.origin_lon, ps_centre_x=cfg.ps_centre_x,
                    ps_centre_y=cfg.ps_centre_y)
    if geo is None:
        geo = GeoConfig(alt_ref=args.alt_ref, ground_elevation_m=args.ground_elevation,
                        camera_pitch_deg=args.camera_pitch_deg,
                        min_depression_deg=args.min_depression, reject_grazing=args.reject_grazing,
                        attitude_sigma_deg=args.attitude_sigma, position_sigma_m=args.position_sigma,
                        altitude_sigma_m=args.altitude_sigma, enabled=not args.no_geolocate)

    plane = fleet.plane
    if plane is None or "fixed-wing" not in fleet.cams:
        raise RuntimeError("fixed-wing asset/camera not available")
    cam = fleet.cams["fixed-wing"]
    asset = "fixed-wing"
    mount_pitch = math.degrees(geo.mount_pitch_rad(asset))

    if args.follow_ship:
        waypoints: list[tuple[float, float, float, str]] = []
        log.info("Follow-ship mode: the plane will chase the live target vessel")
    else:
        waypoints = generate_safe_strait_waypoints(step_lon=args.spacing_lon,
                                                   safe_margin_m=args.margin)
        log.info("Generated %d safe waypoints (3-tier altitudes)", len(waypoints))

    stamp = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join(args.out, stamp)
    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    sidecar_path = os.path.join(run_dir, "sidecar.jsonl")
    gps_track_path = os.path.join(run_dir, "gps_track.csv")
    detections_path = os.path.join(run_dir, "detections.jsonl")
    tracks_path = os.path.join(run_dir, "tracks.jsonl")
    video_path = os.path.join(run_dir, f"patrol_{stamp}.mp4")

    if args.follow_ship and gt is None:
        raise RuntimeError("--follow-ship requires ARCTICSIM_DEV=1")

    project_point = None
    if gt is not None:
        from arcticlib.groundtruth import project_point  # noqa: F401

    if not args.no_fly:
        if not plane.armed or plane.alt_rel < 30.0:
            log.info("Fixed-wing on ground; taking off to %.1f m...", args.alt)
            if not plane.takeoff(alt=args.alt, timeout=120.0):
                raise RuntimeError("fixed-wing takeoff failed")
            climb_deadline = time.monotonic() + 60.0
            while time.monotonic() < climb_deadline and plane.alt_rel < 50.0:
                time.sleep(1.0)
        else:
            log.info("Fixed-wing already airborne at alt=%.1fm", plane.alt_rel)
        plane.set_airspeed(args.speed)
        for _ in range(5):
            if plane.set_mode("GUIDED", timeout=3.0):
                break
            time.sleep(1.0)

    sidecar_file = open(sidecar_path, "w")
    gps_file = open(gps_track_path, "w", newline="")
    gps_writer = csv.writer(gps_file)
    gps_writer.writerow(["t_sim", "lat", "lon", "alt_amsl_m", "alt_rel_m", "agl_m",
                         "roll_rad", "pitch_rad", "yaw_rad"])

    log.info("Starting patrol & recording at %.1f Hz for %.0f s -> %s",
             args.hz, args.duration, run_dir)
    period = 1.0 / max(args.hz, 0.1)
    end_time = time.monotonic() + args.duration if args.duration > 0 else float("inf")
    wpt_idx = 0
    frame_idx = 0
    last_goto_time = 0.0

    try:
        if not args.no_fly and waypoints:
            w_lat, w_lon, w_alt, w_name = waypoints[wpt_idx]
            plane.goto(w_lat, w_lon, w_alt)
            last_goto_time = time.monotonic()
            log.info("Dispatched to Waypoint #%d [%s]", wpt_idx + 1, w_name)

        next_tick = time.monotonic()
        while time.monotonic() < end_time:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue
            next_tick += period
            if next_tick < now:
                next_tick = now + period

            frame = cam.grab()
            if frame is None:
                continue
            img_path = os.path.join(frames_dir, f"{frame_idx:05d}.jpg")
            cv2.imwrite(img_path, frame.image)

            pose = fleet.pose_at(asset, frame.t_sim, clock="sim") or fleet.pose(asset)
            ship = gt.ship_pose() if gt else None
            pose_dict = None if pose is None else {
                "lat": pose.lat, "lon": pose.lon, "alt_rel": pose.alt_rel,
                "alt_amsl": pose.alt_amsl, "roll": pose.roll, "pitch": pose.pitch,
                "yaw": pose.yaw, "vx": pose.vx, "vy": pose.vy, "vz": pose.vz}
            entry = {"frame": os.path.relpath(img_path, run_dir), "index": frame_idx,
                     "t_sim": frame.t_sim, "t_wall": frame.t_wall,
                     "width": frame.width, "height": frame.height,
                     "camera": cfg.assets[asset].camera.intrinsics(),
                     "camera_mount_pitch_deg": mount_pitch, "pose": pose_dict}
            if pose is not None:
                agl = geo.agl(pose)
                entry["gps"] = {"lat": pose.lat, "lon": pose.lon, "alt_amsl": pose.alt_amsl,
                                "alt_rel": pose.alt_rel, "agl_m": agl,
                                "ground_elevation_m": geo.ground_elevation_m, "alt_ref": geo.alt_ref}
                gps_writer.writerow([f"{frame.t_sim:.3f}", f"{pose.lat:.7f}", f"{pose.lon:.7f}",
                                     f"{pose.alt_amsl:.2f}", f"{pose.alt_rel:.2f}", f"{agl:.2f}",
                                     f"{pose.roll:.5f}", f"{pose.pitch:.5f}", f"{pose.yaw:.5f}"])
                gps_file.flush()
            if ship is not None and pose is not None and project_point is not None:
                entry["groundtruth"] = ship
                cam_world = georef.latlon_to_world(pose.lat, pose.lon) + (pose.alt_amsl,)
                grid_yaw_deg = pose.yaw * 57.29577951308232 - cfg.convergence_deg
                uv = project_point(ship["world"], cam_world, grid_yaw_deg, entry["camera"],
                                   cam_pitch_deg=math.degrees(pose.pitch),
                                   cam_roll_deg=math.degrees(pose.roll))
                if uv is not None:
                    entry["groundtruth"]["point_px_approx"] = [uv[0], uv[1]]

            sidecar_file.write(json.dumps(entry) + "\n")
            sidecar_file.flush()
            frame_idx += 1

            if not args.no_fly and args.follow_ship and gt is not None:
                sp = gt.ship_latlon()
                if sp is not None and now - last_goto_time > 5.0:
                    if plane.mode.upper() != "GUIDED":
                        plane.set_mode("GUIDED", timeout=2.0)
                    plane.goto(sp[0], sp[1], args.alt)
                    last_goto_time = now
            elif not args.no_fly and pose and wpt_idx < len(waypoints):
                w_lat, w_lon, w_alt, w_name = waypoints[wpt_idx]
                dist_to_wpt = distance_m(pose.lat, pose.lon, w_lat, w_lon)
                if now - last_goto_time > 6.0:
                    if plane.mode.upper() != "GUIDED":
                        plane.set_mode("GUIDED", timeout=2.0)
                    plane.goto(w_lat, w_lon, w_alt)
                    last_goto_time = now
                if frame_idx % int(args.hz * 5) == 0:
                    log.info("Patrol: Alt=%.1fm Spd=%.1fm/s Mode=%s | Wpt #%d [%s] dist=%.0fm | Frames=%d",
                             pose.alt_rel, pose.speed, plane.mode, wpt_idx + 1, w_name, dist_to_wpt, frame_idx)
                if dist_to_wpt < 180.0:
                    wpt_idx += 1
                    if wpt_idx < len(waypoints):
                        w_lat, w_lon, w_alt, w_name = waypoints[wpt_idx]
                        plane.goto(w_lat, w_lon, w_alt)
                        last_goto_time = now
                        log.info("--> Advancing to Waypoint #%d [%s]", wpt_idx + 1, w_name)
                    else:
                        log.info("All waypoints completed! Entering loiter...")
                        break
    except KeyboardInterrupt:
        log.info("Patrol interrupted by user.")
    finally:
        sidecar_file.close()
        gps_file.close()

    log.info("Flight recording finished: %d frames in %s", frame_idx, frames_dir)
    detector = VerifiedDetector(model_path=args.model, min_color_score=0.25,
                                min_verify_prob=0.50, enable_temporal=not args.no_temporal,
                                min_hits=args.min_hits, max_misses=4)
    tracks = render_patrol_video(frames_dir, sidecar_path, video_path, detector, geo,
                                 asset=asset, fps=args.hz, detections_path=detections_path,
                                 track_gate_m=args.track_gate, min_track_hits=args.min_track_hits)
    best_target = None
    if tracks:
        best_target = (tracks[0].lat, tracks[0].lon)
        if getattr(args, "publish_tracks", False):
            from arcticlib.tracks import TrackClient
            hdg, spd = track_course_speed(tracks[0])
            res = TrackClient(cfg.url(cfg.tracks_port)).post(
                getattr(args, "track_name", "Sierra One"),
                tracks[0].lat, tracks[0].lon, heading=hdg, speed=spd)
            log.info("Published track '%s' -> %.6f, %.6f (hdg=%s, spd=%s) ok=%s",
                     getattr(args, "track_name", "Sierra One"), tracks[0].lat, tracks[0].lon,
                     None if hdg is None else round(hdg, 1),
                     None if spd is None else round(spd, 2), res is not None)
    return {"run_dir": run_dir, "video_path": video_path, "sidecar_path": sidecar_path,
            "gps_track_path": gps_track_path, "detections_path": detections_path,
            "tracks_path": tracks_path, "tracks": tracks, "n_frames": frame_idx,
            "best_target": best_target}


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config()
    fleet = Fleet.from_config(cfg)
    fleet.wait_ready(15)
    gt = None
    if DEV:
        from arcticlib.groundtruth import GroundTruth
        gt = GroundTruth(cfg)
    try:
        res = run_patrol(fleet, args, gt=gt)
    finally:
        if gt is not None:
            gt.stop()
        fleet.shutdown()
    print("\n=======================================================")
    print("Patrol complete! Outputs:")
    for k in ("video_path", "sidecar_path", "gps_track_path", "detections_path", "tracks_path"):
        print(f"  {k:<15}: {res[k]}")
    if res.get("best_target"):
        print(f"  best target    : {res['best_target'][0]:.6f}, {res['best_target'][1]:.6f}")
    print("=======================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
