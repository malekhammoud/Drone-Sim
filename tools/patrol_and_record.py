#!/usr/bin/env python3
"""Autonomous Fixed-Wing Lawnmower Patrol, Recorder, and Video Generator.

1. Generates a zig-zag / lawnmower search pattern covering the strait.
2. Commands the fixed-wing plane (ArduPlane) to take off (alt ~90m) and fly waypoints.
3. Concurrently records 1280x720 frames, telemetry, and ground truth (if ARCTICSIM_DEV=1).
4. Processes the flight with the Step 1 Color-Anomaly Detector and renders an
   annotated MP4 video showing live boat detections, candidate boxes, and telemetry.

Usage:
    # Run a 3-minute patrol and produce annotated video:
    ARCTICSIM_DEV=1 python tools/patrol_and_record.py --duration 180 --out patrol_run

    # Quick test without takeoff (only camera & video processing):
    python tools/patrol_and_record.py --no-fly --duration 30

    # Custom altitude, lane spacing, and speed:
    python tools/patrol_and_record.py --alt 90 --speed 22 --spacing 500 --duration 300
"""
from __future__ import annotations

import argparse
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
from arcticlib.geo import (
    Georef,
    distance_m,
    generate_figure8_pattern,
    generate_search_spiral,
    reroute_around_closed_zone,
)
from arcticlib.tracks import TrackClient
from tools.detect_color import ColorAnomalyDetector
from tools.detect_verified import VerifiedDetector, pixel_to_latlon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("patrol")

DEV = os.environ.get("ARCTICSIM_DEV") == "1"


def generate_safe_strait_waypoints(step_lon: float = 0.01,
                                    safe_margin_m: float = 120.0) -> list[tuple[float, float, float, str]]:
    """Generate N-S zig-zag waypoints following image-analysis-derived Bellot Strait coastlines.

    Features:
      1. Safe Margin: Waypoints are offset ~120m inward from the physical coast to eliminate
         the risk of collision or stalls when banking near coastal hills.
      2. 3-Tier Altitude Profile:
         - Coast waypoints: 100m (obstacle/hill clearance while camera covers shoreline)
         - Center waypoints: 50m (low-altitude high-resolution vessel inspection)
         - Island region (lon -94.855 to -94.830): 150m (safely clears island elevation while
           crossing over to inspect both the northern stream and southern main channel).
      3. 3 Waypoints per column: North Coast <-> Center <-> South Coast for smooth climbing/descent.
    """
    # Exact coastline profile extracted from satellite mosaic image analysis:
    # (lon, north_coast_lat, south_coast_lat, north_stream_lat_if_island)
    coast_profile = [
        (-94.920, 71.99317, 71.97693, None),
        (-94.910, 71.99508, 71.97635, None),
        (-94.900, 71.99359, 71.97688, None),
        (-94.890, 71.99428, 71.97863, None),
        (-94.880, 71.99497, 71.97826, None),
        (-94.870, 71.99625, 71.98017, None),
        (-94.860, 71.99970, 71.97943, None),
        # Island region (-94.855 to -94.830):
        # Northern stream flows up to 72.0045, island at ~71.995, south channel down to 71.981
        (-94.850, 72.00506, 71.98065, 72.00506),
        (-94.840, 72.00450, 71.98176, 72.00450),
        (-94.830, 71.99933, 71.98197, 72.00200),
        (-94.820, 71.99832, 71.98309, None),
        (-94.810, 71.99906, 71.98420, None),
        (-94.800, 72.00453, 71.98452, None),
        (-94.790, 72.00490, 71.98463, None),
        (-94.780, 72.00612, 71.98436, None),
        (-94.770, 72.00824, 71.98415, None),
        (-94.760, 72.00819, 71.98473, None),
        (-94.750, 72.00861, 71.99041, None),
        (-94.740, 72.01137, 71.99009, None),
        (-94.730, 72.01201, 71.99030, None),
        (-94.720, 72.01222, 71.99280, None),
        (-94.710, 72.01164, 71.99296, None),
        (-94.700, 72.01328, 71.99370, None),
        (-94.690, 72.01434, 71.99704, None),
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

        # Inward safe margins
        w_top_lat = n_lat - margin_deg
        w_bot_lat = s_lat + margin_deg
        w_mid_lat = (w_top_lat + w_bot_lat) / 2.0

        if is_island:
            # Over island & northern stream: fly at 125m across all waypoints
            alt_top = 125.0
            alt_mid = 125.0
            alt_bot = 125.0
            col_tag = f"Col {i+1} (Island 125m)"
        else:
            # Normal column: 100m coast clearance, 75m center channel
            alt_top = 100.0
            alt_mid = 75.0
            alt_bot = 100.0
            col_tag = f"Col {i+1}"

        if i % 2 == 0:
            # Downward: North Coast -> Center -> South Coast
            waypoints.append((w_top_lat, float(l), alt_top, f"{col_tag} North"))
            waypoints.append((w_mid_lat, float(l), alt_mid, f"{col_tag} Center"))
            waypoints.append((w_bot_lat, float(l), alt_bot, f"{col_tag} South"))
        else:
            # Upward: South Coast -> Center -> North Coast
            waypoints.append((w_bot_lat, float(l), alt_bot, f"{col_tag} South"))
            waypoints.append((w_mid_lat, float(l), alt_mid, f"{col_tag} Center"))
            waypoints.append((w_top_lat, float(l), alt_top, f"{col_tag} North"))

    # User-requested coastal safety adjustments (move inward away from cliffs/headlands)
    # 1: -lat, 3: +lat, 4: +lat, 6: -lat, 9: +lat, 21: +lat, 24: -lat, 25: -lat, 51: +lat, 57: +lat, 58: +lat
    user_adjustments = {
        1: -0.0025,
        3: +0.0025,
        4: +0.0030,
        6: -0.0025,
        9: +0.0025,
        21: +0.0025,
        24: -0.0025,
        25: -0.0025,
        51: +0.0022,
        57: +0.0025,
        58: +0.0026,
    }

    adjusted_waypoints = []
    for idx, (wlat, wlon, walt, wname) in enumerate(waypoints, start=1):
        d_lat = user_adjustments.get(idx, 0.0)
        adjusted_waypoints.append((wlat + d_lat, wlon, walt, wname))

    return adjusted_waypoints


def render_patrol_video(frames_dir: str,
                        sidecar_path: str,
                        output_video_path: str,
                        detector: Optional[VerifiedDetector] = None,
                        fps: float = 4.0) -> None:
    """Read recorded frames, run 3-stage detector, draw HUD & detections, write MP4."""
    if not os.path.exists(sidecar_path):
        log.error("Sidecar not found: %s", sidecar_path)
        return

    if detector is None:
        detector = VerifiedDetector(model_path="models/patch_verifier.pt",
                                    min_color_score=0.25, min_verify_prob=0.50,
                                    enable_temporal=True, min_hits=8)

    log.info("Processing frames with 3-Stage Detector to produce video: %s", output_video_path)

    # Read sidecar entries
    entries = []
    with open(sidecar_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        log.warning("No entries in sidecar file.")
        return

    first_frame_path = os.path.join(frames_dir, os.path.basename(entries[0]["frame"]))
    first_img = cv2.imread(first_frame_path)
    if first_img is None:
        log.error("Cannot read first frame: %s", first_frame_path)
        return

    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_dir = os.path.dirname(os.path.abspath(output_video_path))
    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    total_confirmed_frames = 0
    total_frames = len(entries)

    for idx, entry in enumerate(entries):
        frame_name = os.path.basename(entry["frame"])
        frame_path = os.path.join(frames_dir, frame_name)
        img = cv2.imread(frame_path)
        if img is None:
            continue

        pose = entry.get("pose") or {}
        t_sim = entry.get("t_sim", 0.0)

        # Run 3-stage detector (Color + CNN + Temporal)
        candidates = detector.detect(img, frame_idx=idx + 1, t_sim=t_sim, pose=pose)
        confirmed_candidates = [c for c in candidates if getattr(c, "is_confirmed", False)]
        if confirmed_candidates:
            total_confirmed_frames += 1

        # Check ground truth
        gt = entry.get("groundtruth", {})
        gt_pt = gt.get("point_px_approx")  # [u, v]

        # Draw detections and ground truth
        vis = detector.draw_detections(img, candidates, gt_pt=gt_pt if (gt_pt and 0 <= gt_pt[0] < w and 0 <= gt_pt[1] < h) else None)

        # Draw Telemetry HUD overlay
        alt = pose.get("alt_rel", 0.0)
        roll = math.degrees(pose.get("roll", 0.0))
        pitch = math.degrees(pose.get("pitch", 0.0))
        yaw = math.degrees(pose.get("yaw", 0.0)) % 360.0
        vx = pose.get("vx", 0.0)
        vy = pose.get("vy", 0.0)
        speed = math.hypot(vx, vy)

        # HUD background box
        cv2.rectangle(vis, (10, 10), (420, 140), (20, 20, 20), -1)
        cv2.rectangle(vis, (10, 10), (420, 140), (80, 80, 80), 1)

        cv2.putText(vis, f"ARCTIC PATROL - FIXED WING", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(vis, f"Sim Time: {t_sim:.1f}s | Frame: {idx+1}/{total_frames}", (20, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(vis, f"Alt: {alt:.1f}m | Speed: {speed:.1f} m/s | Yaw: {yaw:.0f} deg", (20, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(vis, f"Roll: {roll:+.1f} deg | Pitch: {pitch:+.1f} deg", (20, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Detection Status Banner
        if confirmed_candidates:
            c0 = confirmed_candidates[0]
            status_text = f"CONFIRMED BOAT! Track #{c0.track_id} ({c0.hits} hits, {c0.score:.2f})"
            cv2.putText(vis, status_text, (20, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
        elif candidates:
            c0 = candidates[0]
            hits = getattr(c0, "hits", 1)
            tid = getattr(c0, "track_id", "?")
            status_text = f"TENTATIVE: Track #{tid} [{hits}/8 hits]"
            cv2.putText(vis, status_text, (20, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        else:
            cv2.putText(vis, "SEARCHING... (No anomalies)", (20, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        writer.write(vis)

    writer.release()
    log.info("Finished rendering video: %s (Confirmed boat frames: %d/%d)",
             output_video_path, total_confirmed_frames, total_frames)


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous fixed-wing patrol, recorder & video generator.")
    parser.add_argument("--alt", type=float, default=90.0, help="Patrol altitude in metres (default: 90)")
    parser.add_argument("--margin", type=float, default=120.0, help="Safe distance from coast in metres (default: 120)")
    parser.add_argument("--speed", type=float, default=20.0, help="Cruise airspeed m/s (default: 20)")
    parser.add_argument("--spacing-lon", type=float, default=0.01, help="Longitude step between passes (default: 0.01 deg ~343m)")
    parser.add_argument("--hz", type=float, default=3.0, help="Camera recording frame rate (default: 3.0)")
    parser.add_argument("--duration", type=float, default=300.0, help="Patrol duration in seconds (0 = full grid)")
    parser.add_argument("--out", default="patrol_output", help="Output directory root")
    parser.add_argument("--no-fly", action="store_true", help="Record only without commanding takeoff/waypoints")
    parser.add_argument("--closed-zone", type=str, default=None,
                        help="Closed zone overlay formatted as 'lat,lon,radius_m' (e.g. '71.995,-94.810,800')")
    parser.add_argument("--intercept", type=str, default=None,
                        help="Dynamic intercept coordinate target as 'lat,lon' (e.g. '71.985,-94.750')")
    parser.add_argument("--loop", action="store_true", default=True,
                        help="Continuously loop through waypoints until boat found or duration ends (default: True)")
    parser.add_argument("--no-loop", dest="loop", action="store_false",
                        help="Do not loop waypoints after completing one pass")
    parser.add_argument("--publish-tracks", action="store_true",
                        help="Post confirmed vessel tracks to /api/tracks")
    args = parser.parse_args()

    cfg = load_config()
    georef = Georef(cfg.origin_lat, cfg.origin_lon, ps_centre_x=cfg.ps_centre_x, ps_centre_y=cfg.ps_centre_y)
    track_client = TrackClient(cfg.url(cfg.tracks_port)) if args.publish_tracks else None
    fleet = Fleet.from_config(cfg)
    fleet.wait_ready(15)

    plane = fleet.plane
    if plane is None:
        log.error("Fixed-wing asset ('fixed-wing') is not available in fleet roster.")
        return 1

    if "fixed-wing" not in fleet.cams:
        log.error("Fixed-wing camera is not available.")
        return 1

    cam = fleet.cams["fixed-wing"]

    # Generate safe waypoints along the strait curve with 3-tier altitude
    safe_margin = getattr(args, "margin", 120.0)
    waypoints = generate_safe_strait_waypoints(step_lon=args.spacing_lon, safe_margin_m=safe_margin)
    log.info("Generated %d safe baseline waypoints with 3-tier altitudes (75m/100m/125m)", len(waypoints))

    # Apply Closed-Zone Rerouting if requested
    if args.closed_zone:
        try:
            cz_lat_str, cz_lon_str, cz_rad_str = args.closed_zone.split(",")
            cz_lat, cz_lon, cz_radius_m = float(cz_lat_str), float(cz_lon_str), float(cz_rad_str)
            waypoints, inv = reroute_around_closed_zone(waypoints, cz_lat, cz_lon, cz_radius_m, safe_buffer_m=80.0)
            log.info("Closed Zone Active: %d waypoints pruned. Active route has %d waypoints avoiding zone.",
                     len(inv), len(waypoints))
        except Exception as err:
            log.error("Error parsing --closed-zone parameter: %s", err)

    # Mission States
    STATE_PATROL = "PATROL"
    STATE_INTERCEPT = "INTERCEPT"
    STATE_SPIRAL = "SPIRAL"

    mission_state = STATE_PATROL
    target_vessel_pos: Optional[tuple[float, float]] = None
    target_track_id: Optional[int] = None
    spiral_waypoints: list[tuple[float, float, float, str]] = []
    spiral_idx = 0
    last_vessel_seen_time = 0.0

    if args.intercept:
        try:
            it_lat_str, it_lon_str = args.intercept.split(",")
            target_vessel_pos = (float(it_lat_str), float(it_lon_str))
            mission_state = STATE_INTERCEPT
            log.info("Starting in dynamic INTERCEPT mode to target (%.5f, %.5f)",
                     target_vessel_pos[0], target_vessel_pos[1])
        except Exception as err:
            log.error("Error parsing --intercept parameter: %s", err)

    # Live detector for real-time target confirmation
    cam_intrinsics = cfg.assets["fixed-wing"].camera.intrinsics()
    live_detector = VerifiedDetector(model_path="models/patch_verifier.pt",
                                     min_color_score=0.25, min_verify_prob=0.50,
                                     enable_temporal=True, min_hits=8, max_misses=4)

    # Prepare output directories
    stamp = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join(args.out, stamp)
    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    sidecar_path = os.path.join(run_dir, "sidecar.jsonl")
    video_path = os.path.join(run_dir, f"patrol_{stamp}.mp4")

    gt = None
    if DEV:
        from arcticlib.groundtruth import GroundTruth, project_point
        gt = GroundTruth(cfg, georef)
        log.info("Ground truth active (DEV mode) - will project ship location into video")

    # Step 1: Takeoff and transition to GUIDED
    if not args.no_fly:
        if plane.alt_rel < 10.0:
            log.info("Commanding fixed-wing takeoff to %dm...", int(args.alt))
            if not plane.takeoff(alt=args.alt):
                log.error("Fixed-wing takeoff command failed.")
                return 1
            log.info("Takeoff initiated. Waiting to reach safe altitude (>50m)...")
            climb_deadline = time.monotonic() + 60.0
            while time.monotonic() < climb_deadline and plane.alt_rel < 50.0:
                time.sleep(1.0)
        else:
            log.info("Fixed-wing already airborne at alt=%.1fm", plane.alt_rel)

        plane.set_airspeed(args.speed)
        for _ in range(5):
            if plane.set_mode("GUIDED", timeout=3.0):
                log.info("Plane successfully transitioned to GUIDED mode.")
                break
            time.sleep(1.0)

    # Open sidecar
    sidecar_file = open(sidecar_path, "w")

    log.info("Starting patrol & recording at %.1f Hz for %.0f seconds -> %s", args.hz, args.duration, run_dir)
    period = 1.0 / max(args.hz, 0.1)
    end_time = time.monotonic() + args.duration if args.duration > 0 else float("inf")
    wpt_idx = 0
    frame_idx = 0
    last_goto_time = 0.0

    try:
        # Dispatch initial target
        if not args.no_fly:
            if mission_state == STATE_INTERCEPT and target_vessel_pos:
                plane.goto(target_vessel_pos[0], target_vessel_pos[1], 75.0)
                last_goto_time = time.monotonic()
                log.info("Dispatched straight to INTERCEPT target: (%.5f, %.5f)",
                         target_vessel_pos[0], target_vessel_pos[1])
            elif waypoints:
                w_lat, w_lon, w_alt, w_name = waypoints[wpt_idx]
                plane.goto(w_lat, w_lon, w_alt)
                last_goto_time = time.monotonic()
                log.info("Dispatched to Waypoint #%d [%s]: (%.5f, %.5f) at %.0fm",
                         wpt_idx + 1, w_name, w_lat, w_lon, w_alt)

        next_tick = time.monotonic()

        while time.monotonic() < end_time:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue
            next_tick += period
            if next_tick < now:
                next_tick = now + period

            # 1. Grab camera frame
            frame = cam.grab()
            if frame is None:
                continue

            frame_filename = f"{frame_idx:05d}.jpg"
            img_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(img_path, frame.image)

            # 2. Get Pose & Ground Truth
            pose = fleet.pose_at("fixed-wing", frame.t_sim, clock="sim") or fleet.pose("fixed-wing")
            ship = gt.ship_pose() if gt else None

            # 3. Live Detection & Dynamic Target Trigger
            candidates = live_detector.detect(
                frame.image,
                frame_idx=frame_idx,
                t_sim=frame.t_sim,
                pose=pose,
                cam_intrinsics=cam_intrinsics if pose else None,
                georef=georef
            )
            confirmed_targets = [c for c in candidates if getattr(c, "is_confirmed", False)]

            if confirmed_targets and pose:
                best_c = confirmed_targets[0]
                coords = pixel_to_latlon(best_c.cx, best_c.cy, pose, cam_intrinsics, georef)
                if coords:
                    v_lat, v_lon = coords
                    target_vessel_pos = (v_lat, v_lon)
                    target_track_id = getattr(best_c, "track_id", 1)
                    last_vessel_seen_time = now

                    # Post confirmed vessel track to competition API
                    if track_client:
                        res = track_client.post("Sierra One", v_lat, v_lon)
                        if res:
                            log.info("📡 Confirmed track posted to /api/tracks: (%.5f, %.5f)", v_lat, v_lon)

                    if mission_state == STATE_PATROL:
                        log.info("🎯 TARGET CONFIRMED! (Track #%d, %d hits). Breaking patrol to INTERCEPT at (%.5f, %.5f)...",
                                 target_track_id, best_c.hits, v_lat, v_lon)
                        mission_state = STATE_INTERCEPT
                        if not args.no_fly:
                            plane.goto(v_lat, v_lon, 75.0)
                            last_goto_time = now
                    elif mission_state == STATE_SPIRAL:
                        log.info("🎯 TARGET RE-SIGHTED! Re-centering Figure-8 overflight pattern at (%.5f, %.5f)...",
                                 v_lat, v_lon)
                        spiral_waypoints = generate_figure8_pattern(v_lat, v_lon, bearing_deg=85.0, length_m=400.0, width_m=160.0, alt=75.0)
                        spiral_idx = 0
                        if not args.no_fly:
                            plane.goto(spiral_waypoints[0][0], spiral_waypoints[0][1], 75.0)
                            last_goto_time = now

            entry = {
                "frame": os.path.relpath(img_path, run_dir),
                "index": frame_idx,
                "t_sim": frame.t_sim,
                "t_wall": frame.t_wall,
                "width": frame.width,
                "height": frame.height,
                "camera": cam_intrinsics,
                "mission_state": mission_state,
                "target_track_id": target_track_id,
                "pose": None if pose is None else {
                    "lat": pose.lat, "lon": pose.lon, "alt_rel": pose.alt_rel,
                    "alt_amsl": pose.alt_amsl, "roll": pose.roll,
                    "pitch": pose.pitch, "yaw": pose.yaw,
                    "vx": pose.vx, "vy": pose.vy, "vz": pose.vz,
                }
            }

            if ship is not None and pose is not None:
                entry["groundtruth"] = ship
                cam_world = georef.latlon_to_world(pose.lat, pose.lon) + (pose.alt_amsl,)
                grid_yaw_deg = (pose.yaw * 57.29577951308232 - cfg.convergence_deg)
                uv = project_point(ship["world"], cam_world, grid_yaw_deg, entry["camera"],
                                   cam_pitch_deg=math.degrees(pose.pitch),
                                   cam_roll_deg=math.degrees(pose.roll))
                if uv is not None:
                    entry["groundtruth"]["point_px_approx"] = [uv[0], uv[1]]

            sidecar_file.write(json.dumps(entry) + "\n")
            sidecar_file.flush()
            frame_idx += 1

            # 4. Dynamic Flight Control & State Machine Dispatch
            if not args.no_fly and pose:
                if mission_state == STATE_PATROL and wpt_idx < len(waypoints):
                    w_lat, w_lon, w_alt, w_name = waypoints[wpt_idx]
                    dist_to_wpt = distance_m(pose.lat, pose.lon, w_lat, w_lon)

                    # Periodic setpoint refresh and GUIDED enforcement (every 6 seconds)
                    if now - last_goto_time > 6.0:
                        if plane.mode.upper() != "GUIDED":
                            plane.set_mode("GUIDED", timeout=2.0)
                        plane.goto(w_lat, w_lon, w_alt)
                        last_goto_time = now

                    if frame_idx % int(args.hz * 5) == 0:
                        log.info("[PATROL] Alt=%.1fm Spd=%.1fm/s Mode=%s | Wpt #%d [%s] dist=%.0fm | Frames=%d",
                                 pose.alt_rel, pose.speed, plane.mode, wpt_idx + 1, w_name, dist_to_wpt, frame_idx)

                    # Advance waypoint when within 180m
                    if dist_to_wpt < 180.0:
                        wpt_idx += 1
                        if wpt_idx < len(waypoints):
                            w_lat, w_lon, w_alt, w_name = waypoints[wpt_idx]
                            plane.goto(w_lat, w_lon, w_alt)
                            last_goto_time = now
                            log.info("--> Reached! Advancing to Waypoint #%d [%s]: (%.5f, %.5f)",
                                     wpt_idx + 1, w_name, w_lat, w_lon)
                        else:
                            if args.loop:
                                log.info("Completed full patrol pass (%d waypoints)! Looping back to Waypoint #1 to continue patrol...",
                                         len(waypoints))
                                wpt_idx = 0
                                w_lat, w_lon, w_alt, w_name = waypoints[wpt_idx]
                                plane.goto(w_lat, w_lon, w_alt)
                                last_goto_time = now
                            else:
                                log.info("All waypoints completed! Entering loiter...")
                                break

                elif mission_state == STATE_INTERCEPT and target_vessel_pos:
                    dist_to_target = distance_m(pose.lat, pose.lon, target_vessel_pos[0], target_vessel_pos[1])
                    if now - last_goto_time > 5.0:
                        plane.goto(target_vessel_pos[0], target_vessel_pos[1], 75.0)
                        last_goto_time = now

                    if frame_idx % int(args.hz * 3) == 0:
                        log.info("[INTERCEPT] Rushing to vessel Track #%s at (%.5f, %.5f) | dist=%.0fm",
                                 target_track_id, target_vessel_pos[0], target_vessel_pos[1], dist_to_target)

                    if dist_to_target < 160.0:
                        log.info("📍 Arrived at intercept location (dist=%.0fm). Beginning Figure-8 overflight search around (%.5f, %.5f)...",
                                 dist_to_target, target_vessel_pos[0], target_vessel_pos[1])
                        mission_state = STATE_SPIRAL
                        spiral_waypoints = generate_figure8_pattern(target_vessel_pos[0], target_vessel_pos[1],
                                                                    bearing_deg=85.0, length_m=400.0, width_m=160.0, alt=75.0)
                        spiral_idx = 0
                        plane.goto(spiral_waypoints[0][0], spiral_waypoints[0][1], 75.0)
                        last_goto_time = now

                elif mission_state == STATE_SPIRAL:
                    if spiral_idx < len(spiral_waypoints):
                        sw_lat, sw_lon, sw_alt, sw_name = spiral_waypoints[spiral_idx]
                        dist_to_sw = distance_m(pose.lat, pose.lon, sw_lat, sw_lon)

                        if now - last_goto_time > 5.0:
                            plane.goto(sw_lat, sw_lon, sw_alt)
                            last_goto_time = now

                        if frame_idx % int(args.hz * 3) == 0:
                            log.info("[SPIRAL] %s (Wpt %d/%d) | dist=%.0fm",
                                     sw_name, spiral_idx + 1, len(spiral_waypoints), dist_to_sw)

                        if dist_to_sw < 150.0:
                            spiral_idx += 1
                            if spiral_idx < len(spiral_waypoints):
                                sw_lat, sw_lon, sw_alt, sw_name = spiral_waypoints[spiral_idx]
                                plane.goto(sw_lat, sw_lon, sw_alt)
                                last_goto_time = now
                                log.info("--> Advancing spiral search: %s", sw_name)
                            else:
                                log.info("Completed full search spiral (r_max reached). No target seen for %.0fs. Resuming baseline patrol...",
                                         now - last_vessel_seen_time)
                                # Find closest remaining strait waypoint
                                closest_wpt = min(range(len(waypoints)),
                                                  key=lambda i: distance_m(pose.lat, pose.lon, waypoints[i][0], waypoints[i][1]))
                                wpt_idx = closest_wpt
                                mission_state = STATE_PATROL
                                plane.goto(waypoints[wpt_idx][0], waypoints[wpt_idx][1], waypoints[wpt_idx][2])
                                last_goto_time = now

    except KeyboardInterrupt:
        log.info("Patrol interrupted by user.")
    finally:
        sidecar_file.close()
        if gt:
            gt.stop()

    log.info("Flight recording finished. Total frames: %d in %s", frame_idx, frames_dir)

    # Step 4: Video Generation with 3-Stage Detector
    detector = VerifiedDetector(model_path="models/patch_verifier.pt",
                                min_color_score=0.25, min_verify_prob=0.50,
                                enable_temporal=True, min_hits=8, max_misses=4)
    render_patrol_video(frames_dir, sidecar_path, video_path, detector, fps=args.hz)

    print(f"\n=======================================================")
    print(f"Patrol complete! Output video saved to:")
    print(f"  {video_path}")
    print(f"=======================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
