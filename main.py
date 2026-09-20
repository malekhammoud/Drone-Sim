#!/usr/bin/env python3
"""ArcticSim full mission: synchronized wing + quad + GPS + 3-stage CV.

Mission Flow:
  1. **Glider (Fixed-Wing) Search**: Glider launches and flies safe 3-tier strait
     waypoints. It continuously searches and records, and does NOT end recording
     until first sight of the boat is confirmed by the 3-stage detector.
  2. **Quadcopter Standby**: Concurrently, the quadcopter is filming the ground
     on the pad from t=0, so the two video streams are chronologically aligned.
  3. **Handoff**: As soon as the glider confirms the boat sighting, the quadcopter
     takes off and transits to that GPS coordinate. The glider enters a high-altitude
     loiter above the area to maintain situational awareness.
  4. **Quad Acquisition & Follow / Hover Idle**:
     - If the quadcopter finds the boat in its own field of view, it uses its own
       camera (top-down / nadir) to track and follow the vessel.
     - If the quadcopter does not find the boat (false positive or boat moved),
       it hovers idle at the position awaiting the next signal from the glider.
     - When the glider signals an updated sighting, the quad transits to the new location.
  5. **Export All Videos**:
     - Fixed-wing patrol MP4
     - Quadcopter follow MP4
     - Merged top-down MP4 (both feeds vertically stacked, perfectly chronologically aligned).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import math
import os
import shutil
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from arcticlib.config import load_config
from arcticlib.fleet import Fleet
from arcticlib.geo import bearing_deg, destination, distance_m, generate_figure8_pattern
from arcticlib.geolocate import GeoConfig
from tools.detect_verified import VerifiedDetector
from tools.patrol_and_record_gps import generate_safe_strait_waypoints, render_patrol_video
from tools.quad_follow_ship import render_quad_video, standoff_point
from tools.video_merge import merge_videos_top_down

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mission")

DEV = os.environ.get("ARCTICSIM_DEV") == "1"


from tools.patrol_and_record_gps import build_parser as patrol_parser
from tools.quad_follow_ship import build_parser as quad_parser


def _patrol_ns(args) -> argparse.Namespace:
    """Patrol options for phase 1, derived from the mission args."""
    ns = patrol_parser().parse_args([])
    ns.alt = getattr(args, "patrol_alt", 90.0)
    ns.duration = getattr(args, "patrol_duration", 180.0)
    ns.speed = getattr(args, "speed", 20.0)
    ns.margin = getattr(args, "margin", 120.0)
    ns.spacing_lon = getattr(args, "spacing_lon", 0.01)
    ns.hz = getattr(args, "hz", 3.0)
    ns.out = os.path.join(getattr(args, "out", "mission_output"), "wing")
    ns.follow_ship = getattr(args, "follow_ship", DEV)
    ns.no_fly = getattr(args, "no_fly", False)
    ns.model = getattr(args, "model", "models/patch_verifier.pt")
    ns.no_geolocate = getattr(args, "no_geolocate", False)
    ns.ground_elevation = getattr(args, "ground_elevation", 0.0)
    ns.alt_ref = getattr(args, "alt_ref", "amsl")
    ns.camera_pitch_deg = getattr(args, "camera_pitch_deg", None)
    ns.min_depression = getattr(args, "min_depression", 10.0)
    ns.reject_grazing = getattr(args, "reject_grazing", False)
    ns.attitude_sigma = getattr(args, "attitude_sigma", 0.5)
    ns.position_sigma = getattr(args, "position_sigma", 3.0)
    ns.altitude_sigma = getattr(args, "altitude_sigma", 2.0)
    ns.track_gate = getattr(args, "track_gate", 2000.0)
    ns.min_track_hits = getattr(args, "min_track_hits", 2)
    ns.no_temporal = getattr(args, "no_temporal", False)
    ns.min_hits = getattr(args, "min_hits", 6)
    ns.publish_tracks = getattr(args, "publish_tracks", False)
    ns.track_name = getattr(args, "track_name", "Sierra One")
    return ns


def _quad_ns(args) -> argparse.Namespace:
    """Quad options for phase 2, derived from the mission args."""
    ns = quad_parser().parse_args([])
    ns.alt = getattr(args, "quad_alt", 25.0)
    ns.duration = getattr(args, "quad_duration", 120.0)
    ns.speed = getattr(args, "quad_speed", 16.0)
    ns.view_depression = getattr(args, "view_depression", 45.0)
    ns.update_period = getattr(args, "update_period", 3.0)
    ns.takeoff_timeout = getattr(args, "takeoff_timeout", 150.0)
    ns.out = os.path.join(getattr(args, "out", "mission_output"), "quad")
    ns.model = getattr(args, "model", "models/patch_verifier.pt")
    ns.publish_tracks = getattr(args, "publish_tracks", False)
    ns.track_name = getattr(args, "track_name", "Sierra One")
    ns.from_ship = getattr(args, "follow_ship", DEV)
    ns.no_geolocate = getattr(args, "no_geolocate", False)
    ns.ground_elevation = getattr(args, "ground_elevation", 0.0)
    ns.alt_ref = getattr(args, "alt_ref", "amsl")
    ns.camera_pitch_deg = -90.0 if getattr(args, "nadir", False) else getattr(args, "camera_pitch_deg", None)
    ns.min_depression = getattr(args, "min_depression", 10.0)
    ns.reject_grazing = getattr(args, "reject_grazing", False)
    ns.attitude_sigma = getattr(args, "attitude_sigma", 0.5)
    ns.position_sigma = getattr(args, "position_sigma", 3.0)
    ns.altitude_sigma = getattr(args, "altitude_sigma", 2.0)
    ns.nadir = getattr(args, "nadir", False)
    return ns


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Synchronized wing + quad maritime mission.")
    # Glider / search options
    ap.add_argument("--patrol-duration", type=float, default=180.0, help="glider patrol duration, s (default 180)")
    ap.add_argument("--patrol-alt", type=float, default=90.0, help="glider altitude, m (default 90)")
    ap.add_argument("--speed", type=float, default=20.0, help="glider airspeed m/s (default 20)")
    ap.add_argument("--margin", type=float, default=120.0, help="glider safe coast margin, m (default 120)")
    ap.add_argument("--spacing-lon", type=float, default=0.01, help="glider lane spacing, deg (default 0.01)")
    ap.add_argument("--follow-ship", dest="follow_ship", action="store_true", default=None,
                    help="glider chases live vessel in DEV")
    ap.add_argument("--no-follow-ship", dest="follow_ship", action="store_false", default=None,
                    help="force normal safe waypoint search even in DEV")
    ap.add_argument("--search-timeout", type=float, default=600.0,
                    help="max seconds glider searches before timeout (default 600)")
    ap.add_argument("--fig8-repeats", type=int, default=2,
                    help="figure-8 pattern repeats to attempt if boat is lost before resuming waypoint search (default 2)")
    ap.add_argument("--no-fly", action="store_true", help="record only, no flight commands")
    ap.add_argument("--skip-patrol", action="store_true", help="skip glider search")
    # Tower watch (phase 0, runs alongside the glider search)
    ap.add_argument("--no-tower-scan", action="store_true",
                    help="disable the pan/tilt tower watch")
    ap.add_argument("--tower-investigate-duration", type=float, default=45.0,
                    help="seconds the glider spends checking a tower tip")
    ap.add_argument("--tower-contact-ttl", type=float, default=20.0,
                    help="seconds a tower contact stays fresh")
    ap.add_argument("--tower-min-confirm", type=int, default=1,
                    help="confirmed hits before a tower contact counts")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--tracks", default=None, help="existing tracks.jsonl")

    # Quadcopter options
    ap.add_argument("--quad-alt", type=float, default=25.0, help="quad hover altitude, m (default 25)")
    ap.add_argument("--quad-speed", type=float, default=16.0, help="quad cruise speed m/s (default 16)")
    ap.add_argument("--quad-duration", type=float, default=120.0,
                    help="quad tracking seconds once boat is acquired (default 120)")
    ap.add_argument("--view-depression", type=float, default=45.0,
                    help="desired ship depression angle at quad if not nadir, deg (default 45)")
    ap.add_argument("--update-period", type=float, default=3.0, help="quad re-aim period, s (default 3)")
    ap.add_argument("--takeoff-timeout", type=float, default=150.0)
    ap.add_argument("--nadir", action="store_true",
                    help="assume straight-down nadir quad camera (-90 deg); matches iris model edits")

    # Geolocation & Detection
    ap.add_argument("--no-geolocate", action="store_true")
    ap.add_argument("--alt-ref", choices=["amsl", "rel"], default="amsl")
    ap.add_argument("--min-depression", type=float, default=10.0)
    ap.add_argument("--reject-grazing", action="store_true")
    ap.add_argument("--attitude-sigma", type=float, default=0.5)
    ap.add_argument("--position-sigma", type=float, default=3.0)
    ap.add_argument("--altitude-sigma", type=float, default=2.0)
    ap.add_argument("--track-gate", type=float, default=2000.0)
    ap.add_argument("--min-track-hits", type=int, default=2)

    # Shared & outputs
    ap.add_argument("--out", default="mission_output", help="output directory root")
    ap.add_argument("--hz", type=float, default=3.0, help="camera recording rate, Hz (default 3.0)")
    ap.add_argument("--model", default="models/patch_verifier.pt", help="CNN verifier model path")
    ap.add_argument("--no-temporal", action="store_true", help="disable Stage-3 temporal filter")
    ap.add_argument("--min-hits", type=int, default=6, help="temporal hits to confirm boat sighting (default 6)")
    ap.add_argument("--ground-elevation", type=float, default=0.0)
    ap.add_argument("--camera-pitch-deg", type=float, default=None)
    ap.add_argument("--publish-tracks", action="store_true", help="post tracks to track API")
    ap.add_argument("--track-name", default="Sierra One")
    return ap


def resolve_defaults(args) -> argparse.Namespace:
    if args.follow_ship is None:
        args.follow_ship = DEV
    return args



def run_synchronized_mission(fleet: Fleet, args, gt=None, tip_provider=None) -> int:
    """Run synchronized glider search + quad follow with aligned video feeds.

    ``tip_provider`` (a :class:`tools.tower_scan.TowerWatch`) lets the glider
    divert to a boat the tower masts spotted; if the lead is dry it resumes the
    normal waypoint search.
    """
    cfg = fleet.config
    plane = fleet.vehicle("fixed-wing")
    quad = fleet.vehicle("quadcopter")

    if plane is None or "fixed-wing" not in fleet.cams:
        log.error("Fixed-wing vehicle or camera missing from fleet.")
        return 1
    if quad is None or "quadcopter" not in fleet.cams:
        log.error("Quadcopter vehicle or camera missing from fleet.")
        return 1

    cam_w = fleet.cams["fixed-wing"]
    cam_q = fleet.cams["quadcopter"]
    intr_w = cfg.assets["fixed-wing"].camera.intrinsics()
    intr_q = cfg.assets["quadcopter"].camera.intrinsics()

    # Output directories
    stamp = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join(args.out, stamp)
    wing_dir = os.path.join(run_dir, "wing")
    quad_dir = os.path.join(run_dir, "quad")
    wing_frames_dir = os.path.join(wing_dir, "frames")
    quad_frames_dir = os.path.join(quad_dir, "frames")
    os.makedirs(wing_frames_dir, exist_ok=True)
    os.makedirs(quad_frames_dir, exist_ok=True)

    wing_sidecar_path = os.path.join(wing_dir, "sidecar.jsonl")
    quad_sidecar_path = os.path.join(quad_dir, "sidecar.jsonl")
    wing_det_path = os.path.join(wing_dir, "detections.jsonl")
    quad_det_path = os.path.join(quad_dir, "detections.jsonl")
    wing_gps_path = os.path.join(wing_dir, "gps_track.csv")
    quad_gps_path = os.path.join(quad_dir, "gps_track.csv")

    wing_video_path = os.path.join(wing_dir, f"patrol_{stamp}.mp4")
    quad_video_path = os.path.join(quad_dir, f"quad_{stamp}.mp4")
    merged_video_path = os.path.join(run_dir, f"mission_merged_top_down_{stamp}.mp4")

    wing_sidecar_f = open(wing_sidecar_path, "w")
    quad_sidecar_f = open(quad_sidecar_path, "w")
    wing_det_f = open(wing_det_path, "w")
    quad_det_f = open(quad_det_path, "w")

    wing_gps_f = open(wing_gps_path, "w", newline="")
    wing_gps_w = csv.writer(wing_gps_f)
    wing_gps_w.writerow(["t_sim", "lat", "lon", "alt_amsl_m", "alt_rel_m", "agl_m", "roll_rad", "pitch_rad", "yaw_rad"])

    quad_gps_f = open(quad_gps_path, "w", newline="")
    quad_gps_w = csv.writer(quad_gps_f)
    quad_gps_w.writerow(["t_sim", "lat", "lon", "alt_amsl_m", "agl_m", "roll_rad", "pitch_rad", "yaw_rad"])

    # Georeferencing
    geo_w = GeoConfig(camera_pitch_deg=args.camera_pitch_deg, ground_elevation_m=args.ground_elevation)
    quad_pitch = -90.0 if args.nadir else args.camera_pitch_deg
    geo_q = GeoConfig(camera_pitch_deg=quad_pitch, ground_elevation_m=args.ground_elevation)

    # Separate detector instances for glider and quad so temporal trackers stay clean
    detector_w = VerifiedDetector(model_path=args.model, min_color_score=0.25, min_verify_prob=0.50,
                                  enable_temporal=not args.no_temporal, min_hits=args.min_hits)
    detector_q = VerifiedDetector(model_path=args.model, min_color_score=0.25, min_verify_prob=0.40,
                                  enable_temporal=not args.no_temporal, min_hits=3)

    # Launch glider
    glider_state = "SEARCHING"
    if not args.no_fly:
        if not plane.armed or plane.alt_rel < 30.0:
            log.info("Fixed-wing initiating takeoff to %.1f m...", args.patrol_alt)
            if not plane.takeoff(alt=args.patrol_alt, timeout=120.0):
                log.error("Fixed-wing takeoff failed.")
                return 1
            log.info("Fixed-wing in TAKEOFF mode (propeller spinning, climbing out).")
            glider_state = "CLIMBING"
        else:
            log.info("Fixed-wing already airborne at alt=%.1fm", plane.alt_rel)
            plane.set_airspeed(args.speed)
            for _ in range(5):
                if plane.set_mode("GUIDED", timeout=3.0):
                    break
                time.sleep(1.0)
            glider_state = "SEARCHING"

    waypoints = generate_safe_strait_waypoints(step_lon=args.spacing_lon, safe_margin_m=args.margin)
    wpt_idx = 0

    # Mission state:
    # glider_state: CLIMBING -> SEARCHING -> TRACKING
    # quad_state:   ON_GROUND -> TAKEOFF -> TRANSITING -> TRACKING / HOVER_IDLE
    quad_state = "ON_GROUND"
    target_latlon: Optional[tuple[float, float]] = None
    latest_glider_sighting: Optional[tuple[float, float]] = None
    quad_target: Optional[tuple[float, float]] = None
    quad_tracking_start_time: Optional[float] = None
    quad_last_seen_boat = 0.0
    quad_arrived_time = 0.0
    glider_last_seen_boat = 0.0
    fig8_search_wps: list[tuple[float, float, float, str]] = []
    fig8_idx = 0

    # Tower tip diversion state
    tip_seen: set[str] = set()
    tip_active = False
    tip_point: Optional[tuple[float, float]] = None
    tip_deadline = 0.0

    last_plane_cmd = 0.0
    last_quad_cmd = 0.0
    frame_idx = 0
    last_frame_w = None
    last_frame_q = None
    period = 1.0 / max(args.hz, 0.1)
    mission_start = time.monotonic()
    search_deadline = mission_start + args.search_timeout

    log.info("==========================================================================")
    log.info("MISSION STARTED: Glider searching until first confirmed boat sighting.")
    log.info("Quad filming ground on pad. Both feeds recording synchronously from t=0.")
    log.info("==========================================================================")

    try:
        while True:
            t_loop_start = time.monotonic()
            t_sim = fleet.sim.sim_time()

            frame_w = cam_w.grab()
            frame_q = cam_q.grab()
            pose_w = plane.pose()
            pose_q = quad.pose()

            if frame_w is None and last_frame_w is not None:
                frame_w = last_frame_w
            if frame_q is None and last_frame_q is not None:
                frame_q = last_frame_q

            if frame_w is None or frame_q is None:
                time.sleep(0.05)
                continue

            last_frame_w = frame_w
            last_frame_q = frame_q

            # Save synchronized images
            cv2.imwrite(os.path.join(wing_frames_dir, f"{frame_idx:05d}.jpg"), frame_w.image)
            cv2.imwrite(os.path.join(quad_frames_dir, f"{frame_idx:05d}.jpg"), frame_q.image)

            # --- GLIDER PROCESSING ---
            cands_w = detector_w.detect(frame_w.image, frame_idx=frame_idx + 1, t_sim=t_sim, pose=pose_w)
            conf_w = [c for c in cands_w if getattr(c, "is_confirmed", False)]
            estimates_w = [geo_w.locate(c.cx, c.cy, pose_w, "fixed-wing", intr_w) for c in cands_w]

            # Write wing sidecar & telemetry
            if pose_w:
                wing_gps_w.writerow([f"{t_sim:.3f}", f"{pose_w.lat:.7f}", f"{pose_w.lon:.7f}",
                                     f"{pose_w.alt_amsl:.2f}", f"{pose_w.alt_rel:.2f}", f"{geo_w.agl(pose_w):.2f}",
                                     f"{pose_w.roll:.5f}", f"{pose_w.pitch:.5f}", f"{pose_w.yaw:.5f}"])
                wing_gps_f.flush()

            wing_sidecar_f.write(json.dumps({
                "frame": f"frames/{frame_idx:05d}.jpg", "index": frame_idx, "t_sim": t_sim,
                "camera": intr_w, "pose": (None if not pose_w else {
                    "lat": pose_w.lat, "lon": pose_w.lon, "alt_rel": pose_w.alt_rel,
                    "alt_amsl": pose_w.alt_amsl, "roll": pose_w.roll, "pitch": pose_w.pitch,
                    "yaw": pose_w.yaw}),
                "gps": (None if not pose_w else {"lat": pose_w.lat, "lon": pose_w.lon, "agl_m": geo_w.agl(pose_w)}),
            }) + "\n")
            wing_sidecar_f.flush()

            for c, e in zip(cands_w, estimates_w):
                if e is not None:
                    wing_det_f.write(json.dumps({
                        "frame": f"frames/{frame_idx:05d}.jpg", "index": frame_idx, "t_sim": t_sim,
                        "u": c.cx, "v": c.cy, "score": c.score, "confirmed": bool(getattr(c, "is_confirmed", False)),
                        "lat": e.lat, "lon": e.lon, "error_radius_m": e.error_radius_m}) + "\n")
            wing_det_f.flush()

            # --- GLIDER STATE MACHINE ---
            if glider_state == "CLIMBING":
                # Plane is in TAKEOFF mode climbing autonomously. Do not send GUIDED or goto!
                cur_alt = pose_w.alt_rel if pose_w else plane.alt_rel
                if cur_alt >= 45.0:
                    log.info(">>> Fixed-wing reached safe altitude (%.1fm). Transitioning to GUIDED mode & searching! <<<", cur_alt)
                    plane.set_airspeed(args.speed)
                    for _ in range(5):
                        if plane.set_mode("GUIDED", timeout=3.0):
                            break
                        time.sleep(0.5)
                    glider_state = "SEARCHING"

            elif glider_state == "SEARCHING":
                if conf_w:
                    c0 = conf_w[0]
                    e0 = geo_w.locate(c0.cx, c0.cy, pose_w, "fixed-wing", intr_w)
                    if e0 is not None:
                        target_latlon = (e0.lat, e0.lon)
                        latest_glider_sighting = target_latlon
                        glider_state = "TRACKING"
                        glider_last_seen_boat = time.monotonic()
                        log.info(">>> FIRST SIGHT OF BOAT CONFIRMED by Glider: %.6f, %.6f (err=%.1fm)! <<<",
                                 *target_latlon, e0.error_radius_m)
                        log.info(">>> Glider actively TRACKING boat to maintain consistent contact & refine GPS fix. <<<")
                        log.info(">>> DISPATCHING QUADCOPTER TO SIGHTING LOCATION! <<<")
                        quad_state = "TAKEOFF"
                        quad_target = target_latlon

                if glider_state == "SEARCHING" and not args.no_fly and pose_w:
                    if args.follow_ship and DEV and gt is not None:
                        # In DEV follow-ship mode, steer glider toward live vessel to acquire camera view
                        sp = gt.ship_latlon()
                        if sp is not None and time.monotonic() - last_plane_cmd > 5.0:
                            if plane.mode.upper() != "GUIDED":
                                plane.set_mode("GUIDED", timeout=2.0)
                            plane.goto(sp[0], sp[1], args.patrol_alt)
                            last_plane_cmd = time.monotonic()
                    else:
                        # Tower tip: divert to a boat the masts reported, then
                        # resume the waypoint search if the lead turns out dry.
                        now = time.monotonic()
                        tip_handled = False
                        if tip_provider is not None:
                            if tip_active:
                                tip_handled = True
                                if now >= tip_deadline:
                                    log.info("Tower tip dry -> resuming waypoint search")
                                    tip_active = False
                                    tip_handled = False
                                elif now - last_plane_cmd > 5.0:
                                    if plane.mode.upper() != "GUIDED":
                                        plane.set_mode("GUIDED", timeout=2.0)
                                    plane.goto(tip_point[0], tip_point[1], args.patrol_alt)
                                    last_plane_cmd = now
                            else:
                                tip = tip_provider.latest_tip()
                                if tip is not None and tip.id not in tip_seen:
                                    tip_seen.add(tip.id)
                                    tip_point = tip_provider.search_point(tip)
                                    tip_active = True
                                    tip_deadline = now + args.tower_investigate_duration
                                    tip_handled = True
                                    if plane.mode.upper() != "GUIDED":
                                        plane.set_mode("GUIDED", timeout=2.0)
                                    plane.goto(tip_point[0], tip_point[1], args.patrol_alt)
                                    last_plane_cmd = now
                                    log.info(">>> TOWER TIP [%s] %.6f, %.6f (+/-%.0fm) -> "
                                             "diverting glider from waypoint search <<<",
                                             tip.source, tip_point[0], tip_point[1], tip.sigma_m)
                        if not tip_handled:
                            # Normal safe strait waypoints
                            w_lat, w_lon, w_alt, w_name = waypoints[wpt_idx]
                            dist_wpt = distance_m(pose_w.lat, pose_w.lon, w_lat, w_lon)
                            if time.monotonic() - last_plane_cmd > 6.0:
                                if plane.mode.upper() != "GUIDED":
                                    plane.set_mode("GUIDED", timeout=2.0)
                                plane.goto(w_lat, w_lon, w_alt)
                                last_plane_cmd = time.monotonic()
                            if dist_wpt < 180.0:
                                wpt_idx = (wpt_idx + 1) % len(waypoints)
                                w_lat, w_lon, w_alt, w_name = waypoints[wpt_idx]
                                plane.goto(w_lat, w_lon, w_alt)
                                last_plane_cmd = time.monotonic()
                                log.info("Glider advancing -> Waypoint #%d [%s]", wpt_idx + 1, w_name)

            elif glider_state == "TRACKING":
                # Glider continuously tracks the boat to refine location and maintain consistent contact
                best_fix = None
                active_detections = conf_w if conf_w else [c for c in cands_w if c.score >= 0.30]
                for c in active_detections:
                    e = geo_w.locate(c.cx, c.cy, pose_w, "fixed-wing", intr_w)
                    if e is not None and not e.grazing:
                        if best_fix is None or e.error_radius_m < best_fix.error_radius_m:
                            best_fix = e

                if best_fix is not None:
                    target_latlon = (best_fix.lat, best_fix.lon)
                    latest_glider_sighting = target_latlon
                    glider_last_seen_boat = time.monotonic()
                    # Visual contact restored: clear any active search pattern
                    fig8_search_wps = []
                    fig8_idx = 0
                    if frame_idx % int(args.hz * 3) == 0:
                        log.info("Glider refined boat fix: (%.6f, %.6f) +/-%.1fm, dep=%.1f deg",
                                 best_fix.lat, best_fix.lon, best_fix.error_radius_m, best_fix.depression_deg)

                # Determine steering destination
                fly_target = target_latlon
                if args.follow_ship and DEV and gt is not None:
                    sp = gt.ship_latlon()
                    if sp is not None:
                        fly_target = sp
                        if best_fix is None:
                            target_latlon = sp
                            latest_glider_sighting = sp

                strait_axis = 85.0  # Bellot Strait navigable axis (East-West)
                now = time.monotonic()

                # Check if boat has been lost from optical contact
                is_lost = (now - glider_last_seen_boat > 8.0) and not (args.follow_ship and DEV)

                if is_lost and fly_target is not None:
                    # Execute Figure-8 search pattern around last known position
                    if not fig8_search_wps:
                        fig8_search_wps = generate_figure8_pattern(
                            fly_target[0], fly_target[1],
                            bearing_deg=strait_axis,
                            length_m=400.0,
                            width_m=160.0,
                            alt=args.patrol_alt,
                            num_cycles=args.fig8_repeats,
                        )
                        fig8_idx = 0
                        log.warning("Glider lost visual contact (%.0fs). Initiating Figure-8 search (%d repeats) around last fix %.6f, %.6f...",
                                    now - glider_last_seen_boat, args.fig8_repeats, fly_target[0], fly_target[1])

                    if fig8_idx < len(fig8_search_wps):
                        sw_lat, sw_lon, sw_alt, sw_name = fig8_search_wps[fig8_idx]
                        dist_sw = distance_m(pose_w.lat, pose_w.lon, sw_lat, sw_lon)
                        if now - last_plane_cmd > 3.0:
                            if plane.mode.upper() != "GUIDED":
                                plane.set_mode("GUIDED", timeout=2.0)
                            plane.goto(sw_lat, sw_lon, sw_alt)
                            last_plane_cmd = now
                        if dist_sw < 150.0:
                            fig8_idx += 1
                            if fig8_idx < len(fig8_search_wps):
                                log.info("Glider advancing Figure-8 search -> %s (%d/%d)",
                                         fig8_search_wps[fig8_idx][3], fig8_idx + 1, len(fig8_search_wps))
                    else:
                        # Completed all Figure-8 pattern repeats without re-finding boat!
                        # Resume planned waypoint journey along the strait
                        wpt_idx = min(range(len(waypoints)),
                                      key=lambda i: distance_m(pose_w.lat, pose_w.lon, waypoints[i][0], waypoints[i][1]))
                        glider_state = "SEARCHING"
                        fig8_search_wps = []
                        fig8_idx = 0
                        w_lat, w_lon, w_alt, w_name = waypoints[wpt_idx]
                        log.warning("Glider completed %d Figure-8 pattern repeats without re-finding boat (lost for %.0fs). Resuming waypoint journey -> Waypoint #%d [%s]",
                                    args.fig8_repeats, now - glider_last_seen_boat, wpt_idx + 1, w_name)
                        if not args.no_fly:
                            if plane.mode.upper() != "GUIDED":
                                plane.set_mode("GUIDED", timeout=2.0)
                            plane.goto(w_lat, w_lon, w_alt)
                            last_plane_cmd = now

                elif not args.no_fly and pose_w and fly_target is not None:
                    # Active Overflight Re-attack tracking controller (while target is in sight or DEV mode)
                    dist_to_tgt = distance_m(pose_w.lat, pose_w.lon, fly_target[0], fly_target[1])
                    b_to_tgt = bearing_deg(pose_w.lat, pose_w.lon, fly_target[0], fly_target[1])

                    if dist_to_tgt > 160.0:
                        # Inbound pass: Target 250m PAST the boat along approach bearing so plane
                        # flies straight OVER the boat with wings level (roll=0) and camera locked on target
                        dest_lat, dest_lon = destination(fly_target[0], fly_target[1], b_to_tgt, 250.0)
                    else:
                        # Overhead / Extension: Plane is right over or crossing the vessel.
                        # Extend 350m along channel axis to set up next straight pass
                        cur_heading = math.degrees(pose_w.yaw) % 360.0
                        fwd_axis = strait_axis if math.cos(math.radians(cur_heading - strait_axis)) > 0 else (strait_axis + 180.0) % 360.0
                        dest_lat, dest_lon = destination(fly_target[0], fly_target[1], fwd_axis, 350.0)

                    if now - last_plane_cmd > 3.0:
                        if plane.mode.upper() != "GUIDED":
                            plane.set_mode("GUIDED", timeout=2.0)
                        plane.goto(dest_lat, dest_lon, args.patrol_alt)
                        last_plane_cmd = now

            # --- QUADCOPTER PROCESSING ---
            cands_q = detector_q.detect(frame_q.image, frame_idx=frame_idx + 1, t_sim=t_sim, pose=pose_q)
            high_conf_q = [c for c in cands_q if c.score >= 0.35]
            estimates_q = [geo_q.locate(c.cx, c.cy, pose_q, "quadcopter", intr_q) for c in cands_q]

            if pose_q:
                quad_gps_w.writerow([f"{t_sim:.3f}", f"{pose_q.lat:.7f}", f"{pose_q.lon:.7f}",
                                     f"{pose_q.alt_amsl:.2f}", f"{geo_q.agl(pose_q):.2f}",
                                     f"{pose_q.roll:.5f}", f"{pose_q.pitch:.5f}", f"{pose_q.yaw:.5f}"])
                quad_gps_f.flush()

            quad_sidecar_f.write(json.dumps({
                "frame": f"frames/{frame_idx:05d}.jpg", "index": frame_idx, "t_sim": t_sim,
                "camera": intr_q, "camera_mount_pitch_deg": math.degrees(geo_q.mount_pitch_rad("quadcopter")),
                "pose": (None if not pose_q else {
                    "lat": pose_q.lat, "lon": pose_q.lon, "alt_rel": pose_q.alt_rel,
                    "alt_amsl": pose_q.alt_amsl, "roll": pose_q.roll, "pitch": pose_q.pitch,
                    "yaw": pose_q.yaw}),
                "gps": (None if not pose_q else {"lat": pose_q.lat, "lon": pose_q.lon, "agl_m": geo_q.agl(pose_q)}),
            }) + "\n")
            quad_sidecar_f.flush()

            for c, e in zip(cands_q, estimates_q):
                if e is not None:
                    quad_det_f.write(json.dumps({
                        "frame": f"frames/{frame_idx:05d}.jpg", "index": frame_idx, "t_sim": t_sim,
                        "u": c.cx, "v": c.cy, "score": c.score, "confirmed": bool(getattr(c, "is_confirmed", False)),
                        "lat": e.lat, "lon": e.lon, "error_radius_m": e.error_radius_m}) + "\n")
            quad_det_f.flush()


            # Quadcopter State Logic
            if quad_state == "ON_GROUND":
                # Filming ground pad at Fort Ross. Link alive.
                pass

            elif quad_state == "TAKEOFF":
                if not args.no_fly:
                    log.info("Quadcopter taking off to %.1fm...", args.quad_alt)
                    quad.takeoff(alt=args.quad_alt, timeout=args.takeoff_timeout)
                    quad.set_speed(args.quad_speed)
                quad_state = "TRANSITING"
                if latest_glider_sighting is not None:
                    quad_target = latest_glider_sighting
                log.info("Quadcopter transiting to target: %.6f, %.6f", *quad_target)
                if not args.no_fly and quad_target:
                    quad.goto(quad_target[0], quad_target[1], args.quad_alt)
                    last_quad_cmd = time.monotonic()

            elif quad_state == "TRANSITING":
                if not args.no_fly and pose_q and quad_target:
                    # If glider refines the target location, ALWAYS adopt the new target location immediately
                    if latest_glider_sighting is not None and latest_glider_sighting != quad_target:
                        shift = distance_m(latest_glider_sighting[0], latest_glider_sighting[1],
                                           quad_target[0], quad_target[1])
                        if shift > 1.0:  # Any real refinement
                            quad_target = latest_glider_sighting
                            quad.goto(quad_target[0], quad_target[1], args.quad_alt)
                            last_quad_cmd = time.monotonic()
                            log.info("Quad transit target updated from glider fix -> %.6f, %.6f (shifted %.1fm)",
                                     *quad_target, shift)

                    dist_to_tgt = distance_m(pose_q.lat, pose_q.lon, quad_target[0], quad_target[1])
                    if time.monotonic() - last_quad_cmd > 2.0:
                        quad.goto(quad_target[0], quad_target[1], args.quad_alt)
                        last_quad_cmd = time.monotonic()

                    # Arrived in proximity
                    if dist_to_tgt < 45.0 or (high_conf_q and dist_to_tgt < 120.0):
                        log.info("Quadcopter arrived at target area (dist=%.1fm). Searching FOV...", dist_to_tgt)
                        quad_state = "SEARCHING_FOV"
                        quad_arrived_time = time.monotonic()

            elif quad_state in ("SEARCHING_FOV", "TRACKING", "HOVER_IDLE"):
                if high_conf_q:
                    c_q = high_conf_q[0]
                    e_q = geo_q.locate(c_q.cx, c_q.cy, pose_q, "quadcopter", intr_q)
                    if e_q is not None:
                        if quad_state != "TRACKING":
                            log.info(">>> QUADCOPTER ACQUIRED BOAT IN FOV! Tracking with own camera -> %.6f, %.6f <<<",
                                     e_q.lat, e_q.lon)
                            quad_state = "TRACKING"
                            if quad_tracking_start_time is None:
                                quad_tracking_start_time = time.monotonic()

                        quad_last_seen_boat = time.monotonic()
                        if not args.no_fly and pose_q and time.monotonic() - last_quad_cmd > args.update_period:
                            if args.nadir:
                                # Top-down nadir camera: fly directly above the boat
                                quad.goto(e_q.lat, e_q.lon, args.quad_alt)
                            else:
                                # Forward-down camera: hold standoff
                                standoff = max(5.0, args.quad_alt / math.tan(math.radians(args.view_depression)))
                                hl, ho, _ = standoff_point(e_q.lat, e_q.lon, pose_q.lat, pose_q.lon, standoff)
                                quad.goto(hl, ho, args.quad_alt)
                            last_quad_cmd = time.monotonic()
                else:
                    now = time.monotonic()
                    if quad_state == "SEARCHING_FOV" and now - quad_arrived_time > 8.0:
                        quad_state = "HOVER_IDLE"
                        log.warning("Boat not in quad FOV (false positive or moved). Hovering idle awaiting glider signal...")
                    elif quad_state == "TRACKING" and now - quad_last_seen_boat > 12.0:
                        quad_state = "HOVER_IDLE"
                        log.warning("Lost boat in quad FOV. Hovering idle awaiting glider signal...")

                    if quad_state == "HOVER_IDLE":
                        # Check if glider sends an updated sighting
                        if latest_glider_sighting and quad_target:
                            dist_to_new_sighting = distance_m(latest_glider_sighting[0], latest_glider_sighting[1],
                                                              quad_target[0], quad_target[1])
                            if dist_to_new_sighting > 10.0:
                                log.info("Glider sent updated boat fix: %.6f, %.6f (moved %.0fm). Quad resuming transit...",
                                         *latest_glider_sighting, dist_to_new_sighting)
                                quad_target = latest_glider_sighting
                                quad_state = "TRANSITING"
                                if not args.no_fly:
                                    quad.goto(quad_target[0], quad_target[1], args.quad_alt)
                                    last_quad_cmd = time.monotonic()

            # Check tracking completion
            if quad_tracking_start_time and time.monotonic() - quad_tracking_start_time > args.quad_duration:
                log.info("Quadcopter tracking duration (%.0fs) reached! Mission complete.", args.quad_duration)
                break

            if glider_state == "SEARCHING" and time.monotonic() > search_deadline:
                log.warning("Glider search timeout reached (%.0fs) without confirmed sighting.", args.search_timeout)
                break

            # Periodic status logging
            if frame_idx % int(args.hz * 5) == 0:
                alt_w = pose_w.alt_rel if pose_w else 0.0
                alt_q = pose_q.alt_rel if pose_q else 0.0
                log.info("[Frame %04d | t=%.1fs] Glider: %s (Alt=%.0fm) | Quad: %s (Alt=%.0fm)",
                         frame_idx, t_sim, glider_state, alt_w, quad_state, alt_q)

            frame_idx += 1
            dt = time.monotonic() - t_loop_start
            if dt < period:
                time.sleep(period - dt)

    except KeyboardInterrupt:
        log.info("Mission interrupted by user (Ctrl+C). Finalizing and rendering videos...")
    finally:
        wing_sidecar_f.close()
        quad_sidecar_f.close()
        wing_det_f.close()
        quad_det_f.close()
        wing_gps_f.close()
        quad_gps_f.close()

    # --- RENDER VIDEOS ---
    log.info("==========================================================================")
    has_wing_video = False
    has_quad_video = False

    try:
        log.info("Rendering Fixed-Wing MP4: %s", wing_video_path)
        render_patrol_video(wing_frames_dir, wing_sidecar_path, wing_video_path,
                            detector=detector_w, geo=geo_w, asset="fixed-wing", fps=args.hz,
                            detections_path=wing_det_path)
        has_wing_video = os.path.exists(wing_video_path) and os.path.getsize(wing_video_path) > 0
    except Exception as exc:
        log.error("Failed to render fixed-wing MP4: %s", exc)

    try:
        log.info("Rendering Quadcopter MP4: %s", quad_video_path)
        render_quad_video(quad_frames_dir, quad_sidecar_path, quad_video_path,
                          detector=detector_q, geo=geo_q, asset="quadcopter", fps=args.hz,
                          detections_path=quad_det_path)
        has_quad_video = os.path.exists(quad_video_path) and os.path.getsize(quad_video_path) > 0
    except Exception as exc:
        log.error("Failed to render quadcopter MP4: %s", exc)

    has_merged_video = False
    if has_wing_video and has_quad_video:
        try:
            log.info("Rendering Merged Top-Down MP4 (Chronologically Aligned): %s", merged_video_path)
            merge_videos_top_down(
                top_video_path=wing_video_path,
                bottom_video_path=quad_video_path,
                output_path=merged_video_path,
                mode="parallel",
                top_label="FIXED-WING RECONNAISSANCE (GLIDER)",
                bottom_label="QUADCOPTER SURVEILLANCE (TOP-DOWN)",
            )
            has_merged_video = os.path.exists(merged_video_path) and os.path.getsize(merged_video_path) > 0
        except Exception as exc:
            log.error("Failed to merge videos top-down: %s", exc)

    # Convenience copies in mission_output/
    try:
        if has_wing_video:
            shutil.copyfile(wing_video_path, os.path.join(args.out, "fixed_wing_patrol.mp4"))
        if has_quad_video:
            shutil.copyfile(quad_video_path, os.path.join(args.out, "quadcopter_follow.mp4"))
        if has_merged_video:
            shutil.copyfile(merged_video_path, os.path.join(args.out, "mission_merged_top_down.mp4"))
    except Exception as cp_err:
        log.warning("Could not copy convenience files: %s", cp_err)

    print("\n=======================================================")
    print("MISSION COMPLETE! Exported Videos:")
    if has_wing_video:
        print(f"  Fixed-Wing Patrol : {wing_video_path}")
        print(f"                      -> {os.path.join(args.out, 'fixed_wing_patrol.mp4')}")
    if has_quad_video:
        print(f"  Quadcopter Follow : {quad_video_path}")
        print(f"                      -> {os.path.join(args.out, 'quadcopter_follow.mp4')}")
    if has_merged_video:
        print(f"  Merged Top-Down   : {merged_video_path}")
        print(f"                      -> {os.path.join(args.out, 'mission_merged_top_down.mp4')}")
    print("=======================================================\n")
    return 0
    print("MISSION COMPLETE! Exported Videos:")
    print(f"  Fixed-Wing Patrol : {wing_video_path}")
    print(f"                      -> {os.path.join(args.out, 'fixed_wing_patrol.mp4')}")
    print(f"  Quadcopter Follow : {quad_video_path}")
    print(f"                      -> {os.path.join(args.out, 'quadcopter_follow.mp4')}")
    print(f"  Merged Top-Down   : {merged_video_path}")
    print(f"                      -> {os.path.join(args.out, 'mission_merged_top_down.mp4')}")
    print("=======================================================\n")
    return 0


def main() -> int:
    args = resolve_defaults(build_parser().parse_args())

    cfg = load_config()
    fleet = Fleet.from_config(cfg)
    fleet.wait_ready(15)

    gt = None
    if DEV:
        try:
            from arcticlib.groundtruth import GroundTruth
            gt = GroundTruth(cfg)
        except Exception:
            gt = None

    # Phase 0 — tower watch, in the background. It sweeps both masts for boats
    # and the wing diverts to any confirmed contact it reports.
    watch = None
    if not args.no_tower_scan and not args.skip_patrol and not args.follow_ship:
        try:
            from tools.tower_scan import TowerWatch
            watch = TowerWatch(fleet, cfg, contact_ttl_s=args.tower_contact_ttl,
                               min_confirm=args.tower_min_confirm)
            watch.start()
        except Exception as exc:
            log.warning("Tower watch disabled: %s", exc)
            watch = None

    try:
        return run_synchronized_mission(fleet, args, gt=gt, tip_provider=watch)
    finally:
        if watch is not None:
            watch.stop()
        if gt is not None:
            gt.stop()
        fleet.shutdown()


if __name__ == "__main__":
    sys.exit(main())
