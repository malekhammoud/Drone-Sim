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
from arcticlib.geo import Georef, distance_m
from tools.detect_color import ColorAnomalyDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("patrol")

DEV = os.environ.get("ARCTICSIM_DEV") == "1"


def generate_user_strait_waypoints(step_lon: float = 0.01,
                                    alt: float = 90.0) -> list[tuple[float, float, float, str]]:
    """Generate N-S zig-zag waypoints following the user-defined strait boundary curve.

    Control points:
      - lon -94.91: top=71.990, bottom=71.980
      - lon -94.85: top=72.000, bottom=71.980
      - lon -94.80: top=72.005, bottom=71.985
      - lon -94.76: top=72.010, bottom=71.985
      - lon -94.75: top=72.010, bottom=71.995
      - lon -94.71: top=72.015, bottom=72.000
      - lon -94.70: top=72.010, bottom=71.995
    """
    ctrl_pts = [
        (-94.91, 71.990, 71.980),
        (-94.85, 72.000, 71.980),
        (-94.80, 72.005, 71.985),
        (-94.76, 72.010, 71.985),
        (-94.75, 72.010, 71.995),
        (-94.71, 72.015, 72.000),
        (-94.70, 72.010, 71.995),
    ]
    ctrl_pts.sort(key=lambda p: p[0])
    c_lons = [p[0] for p in ctrl_pts]
    c_tops = [p[1] for p in ctrl_pts]
    c_bots = [p[2] for p in ctrl_pts]

    col_lons = np.round(np.arange(-94.91, -94.70 + 0.0001, step_lon), 4)
    waypoints: list[tuple[float, float, float, str]] = []

    for i, l in enumerate(col_lons):
        top_lat = float(np.interp(l, c_lons, c_tops))
        bot_lat = float(np.interp(l, c_lons, c_bots))

        if i % 2 == 0:
            # Downward: Top -> Bottom
            waypoints.append((top_lat, float(l), alt, f"Col {i+1} Top"))
            waypoints.append((bot_lat, float(l), alt, f"Col {i+1} Bot"))
        else:
            # Upward: Bottom -> Top
            waypoints.append((bot_lat, float(l), alt, f"Col {i+1} Bot"))
            waypoints.append((top_lat, float(l), alt, f"Col {i+1} Top"))

    return waypoints


def render_patrol_video(frames_dir: str,
                        sidecar_path: str,
                        output_video_path: str,
                        detector: ColorAnomalyDetector,
                        fps: float = 4.0) -> None:
    """Read recorded frames, run color detector, draw HUD & detections, write MP4."""
    if not os.path.exists(sidecar_path):
        log.error("Sidecar not found: %s", sidecar_path)
        return

    log.info("Processing frames with Color-Anomaly Detector to produce video: %s", output_video_path)

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

    total_detections = 0
    total_frames = len(entries)

    for idx, entry in enumerate(entries):
        frame_name = os.path.basename(entry["frame"])
        frame_path = os.path.join(frames_dir, frame_name)
        img = cv2.imread(frame_path)
        if img is None:
            continue

        # Run color detector
        candidates = detector.detect(img)
        if candidates:
            total_detections += 1

        # Check ground truth
        gt = entry.get("groundtruth", {})
        gt_pt = gt.get("point_px_approx")  # [u, v]

        # Draw detections and ground truth
        vis = detector.draw_detections(img, candidates, gt_pt=gt_pt if (gt_pt and 0 <= gt_pt[0] < w and 0 <= gt_pt[1] < h) else None)

        # Draw Telemetry HUD overlay
        pose = entry.get("pose") or {}
        alt = pose.get("alt_rel", 0.0)
        roll = math.degrees(pose.get("roll", 0.0))
        pitch = math.degrees(pose.get("pitch", 0.0))
        yaw = math.degrees(pose.get("yaw", 0.0)) % 360.0
        vx = pose.get("vx", 0.0)
        vy = pose.get("vy", 0.0)
        speed = math.hypot(vx, vy)
        t_sim = entry.get("t_sim", 0.0)

        # HUD background box
        cv2.rectangle(vis, (10, 10), (380, 140), (20, 20, 20), -1)
        cv2.rectangle(vis, (10, 10), (380, 140), (80, 80, 80), 1)

        cv2.putText(vis, f"ARCTIC PATROL - FIXED WING", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(vis, f"Sim Time: {t_sim:.1f}s | Frame: {idx+1}/{total_frames}", (20, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(vis, f"Alt: {alt:.1f}m | Speed: {speed:.1f} m/s | Yaw: {yaw:.0f} deg", (20, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(vis, f"Roll: {roll:+.1f} deg | Pitch: {pitch:+.1f} deg", (20, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Detection Status Banner
        if candidates:
            c0 = candidates[0]
            status_text = f"BOAT DETECTED! Conf: {c0.score:.2f} ({len(candidates)} hits)"
            status_color = (0, 0, 255) if c0.score > 0.5 else (0, 165, 255)
            cv2.putText(vis, status_text, (20, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 2)
        else:
            cv2.putText(vis, "SEARCHING... (No anomalies)", (20, 122),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        writer.write(vis)

    writer.release()
    log.info("Finished rendering video: %s (Total boat sightings: %d/%d frames)",
             output_video_path, total_detections, total_frames)


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous fixed-wing patrol, recorder & video generator.")
    parser.add_argument("--alt", type=float, default=90.0, help="Patrol altitude in metres (default: 90)")
    parser.add_argument("--speed", type=float, default=20.0, help="Cruise airspeed m/s (default: 20)")
    parser.add_argument("--spacing-lon", type=float, default=0.01, help="Longitude step between passes (default: 0.01 deg ~343m)")
    parser.add_argument("--hz", type=float, default=3.0, help="Camera recording frame rate (default: 3.0)")
    parser.add_argument("--duration", type=float, default=300.0, help="Patrol duration in seconds (0 = full grid)")
    parser.add_argument("--out", default="patrol_output", help="Output directory root")
    parser.add_argument("--no-fly", action="store_true", help="Record only without commanding takeoff/waypoints")
    args = parser.parse_args()

    cfg = load_config()
    georef = Georef(cfg.origin_lat, cfg.origin_lon, ps_centre_x=cfg.ps_centre_x, ps_centre_y=cfg.ps_centre_y)
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

    # Generate custom waypoints along the strait curve
    waypoints = generate_user_strait_waypoints(step_lon=args.spacing_lon, alt=args.alt)
    log.info("Generated %d custom zig-zag waypoints following strait curve", len(waypoints))

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
        if not plane.armed or plane.alt_rel < 30.0:
            log.info("Fixed-wing is on ground. Initiating automated takeoff to %.1f m...", args.alt)
            ok = plane.takeoff(alt=args.alt, timeout=120.0)
            if not ok:
                log.error("Takeoff failed or timed out. Aborting.")
                fleet.shutdown()
                return 1
            log.info("Takeoff initiated. Climbing to transition altitude (>50m)...")
            climb_deadline = time.monotonic() + 60.0
            while time.monotonic() < climb_deadline and plane.alt_rel < 50.0:
                time.sleep(1.0)
        else:
            log.info("Fixed-wing already airborne at alt=%.1fm", plane.alt_rel)

        plane.set_airspeed(args.speed)
        # Ensure plane is in GUIDED mode
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
        # Send first waypoint
        if not args.no_fly and waypoints:
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

            entry = {
                "frame": os.path.relpath(img_path, run_dir),
                "index": frame_idx,
                "t_sim": frame.t_sim,
                "t_wall": frame.t_wall,
                "width": frame.width,
                "height": frame.height,
                "camera": cfg.assets["fixed-wing"].camera.intrinsics(),
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

            # 3. Check waypoint progress
            if not args.no_fly and pose and wpt_idx < len(waypoints):
                w_lat, w_lon, w_alt, w_name = waypoints[wpt_idx]
                dist_to_wpt = distance_m(pose.lat, pose.lon, w_lat, w_lon)

                # Periodic setpoint refresh and GUIDED enforcement (every 6 seconds)
                if now - last_goto_time > 6.0:
                    if plane.mode.upper() != "GUIDED":
                        plane.set_mode("GUIDED", timeout=2.0)
                    plane.goto(w_lat, w_lon, w_alt)
                    last_goto_time = now

                if frame_idx % int(args.hz * 5) == 0:
                    log.info("Patrol status: Alt=%.1fm Spd=%.1fm/s Mode=%s | Wpt #%d [%s] dist=%.0fm | Frames=%d",
                             pose.alt_rel, pose.speed, plane.mode, wpt_idx + 1, w_name, dist_to_wpt, frame_idx)

                # If within 180m of waypoint, advance to next waypoint
                if dist_to_wpt < 180.0:
                    wpt_idx += 1
                    if wpt_idx < len(waypoints):
                        w_lat, w_lon, w_alt, w_name = waypoints[wpt_idx]
                        plane.goto(w_lat, w_lon, w_alt)
                        last_goto_time = now
                        log.info("--> Reached! Advancing to Waypoint #%d [%s]: (%.5f, %.5f)",
                                 wpt_idx + 1, w_name, w_lat, w_lon)
                    else:
                        log.info("All waypoints completed! Entering loiter...")
                        break

    except KeyboardInterrupt:
        log.info("Patrol interrupted by user.")
    finally:
        sidecar_file.close()
        if gt:
            gt.stop()

    log.info("Flight recording finished. Total frames: %d in %s", frame_idx, frames_dir)

    # Step 4: Video Generation with Step 1 Color-Anomaly Detector
    detector = ColorAnomalyDetector(min_area=2, min_score=0.32)
    render_patrol_video(frames_dir, sidecar_path, video_path, detector, fps=args.hz)

    print(f"\n=======================================================")
    print(f"Patrol complete! Output video saved to:")
    print(f"  {video_path}")
    print(f"=======================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
