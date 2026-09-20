#!/usr/bin/env python3
"""Step 2: send the quadcopter to the glider's target and follow the ship.

Two-step system
---------------
1. The fixed-wing glider searches the strait and geolocates the vessel
   (``patrol_and_record_gps.py`` / ``detect_verified_gps.py``) -> a lat/lon.
2. **This tool** flies the quadcopter to that lat/lon, keeps the ship in view and
   refines its position, and can post the fix to the track API.

``tools/two_step_mission.py`` chains the two automatically in one command; this
module exposes :func:`run_quad_follow` and :func:`build_parser` for that.

Gimbal reality
--------------
The sim's quadcopter camera is a *fixed* 20.0 deg-down mount: the SDF joint is
type 8 (fixed) with no controller plugin, ``MNT1_TYPE=0``, and neither MAVLink
mount commands nor gzweb link edits move it. So "point it straight down" is not
achievable here. The tool therefore geolocates with the real mount
(``-20.002 deg``); if the model is ever changed to nadir, pass
``--camera-pitch-deg -90`` (or ``--nadir``).

Because the mount looks forward-and-down, a hovering quad would have the ship
*under* it and out of frame. The tool instead holds a **standoff** so the ship
sits at a chosen depression angle (default 45 deg), where a single frame is much
more accurate than the glider's grazing geometry, and yaws toward the ship.

Usage:
    # From the glider's track file:
    python tools/quad_follow_ship.py --tracks tools/gt_runs/<stamp>/eval/tracks.jsonl

    # Explicit coordinate:
    python tools/quad_follow_ship.py --lat 71.9865 --lon -94.8983 --alt 25

    # DEV test: chase the true vessel and score each fix against it:
    ARCTICSIM_DEV=1 python tools/quad_follow_ship.py --from-ship --duration 120
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
from arcticlib.geo import bearing_deg, destination, distance_m
from arcticlib.geolocate import GeoConfig
from arcticlib.tracks import TrackClient
from tools.detect_verified import VerifiedDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("quad_follow")

DEV = os.environ.get("ARCTICSIM_DEV") == "1"
ASSET = "quadcopter"


def load_target_from_tracks(path: str) -> Optional[tuple[float, float]]:
    """Best (most-observed) fused track from a step-1 ``tracks.jsonl``."""
    best = None
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            t = json.loads(line)
            if best is None or (t.get("n_frames", 0), t.get("n", 0)) > \
                               (best.get("n_frames", 0), best.get("n", 0)):
                best = t
    if best is None:
        return None
    return float(best["lat"]), float(best["lon"])


def standoff_point(ship_lat: float, ship_lon: float, quad_lat: float,
                   quad_lon: float, standoff_m: float) -> tuple[float, float, float]:
    """A hold point ``standoff_m`` from the ship, on the quad's current side.

    Returns ``(lat, lon, yaw_deg)`` where the yaw points from the hold point back
    at the ship, so the forward camera faces the target.
    """
    if distance_m(quad_lat, quad_lon, ship_lat, ship_lon) > 1.0:
        away = bearing_deg(ship_lat, ship_lon, quad_lat, quad_lon)
    else:
        away = 0.0
    hold_lat, hold_lon = destination(ship_lat, ship_lon, away, standoff_m)
    yaw = bearing_deg(hold_lat, hold_lon, ship_lat, ship_lon)
    return hold_lat, hold_lon, yaw


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Step 2: quadcopter flies to and follows the ship.")
    # Target
    ap.add_argument("--lat", type=float, default=None, help="target latitude")
    ap.add_argument("--lon", type=float, default=None, help="target longitude")
    ap.add_argument("--tracks", default=None, help="step-1 tracks.jsonl (uses best track)")
    ap.add_argument("--from-ship", action="store_true",
                    help="DEV test: use the live true vessel pose as the target")
    # Flight
    ap.add_argument("--asset", default=ASSET)
    ap.add_argument("--alt", type=float, default=25.0, help="hover altitude, m (default 25)")
    ap.add_argument("--hz", type=float, default=2.0, help="camera/detection rate")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds (0 = forever)")
    ap.add_argument("--takeoff-timeout", type=float, default=150.0)
    ap.add_argument("--speed", type=float, default=12.0, help="cruise speed m/s (WPNAV_SPEED)")
    ap.add_argument("--no-fly", action="store_true", help="detect/geolocate only, no commands")
    # Following
    ap.add_argument("--view-depression", type=float, default=45.0,
                    help="desired ship depression angle, deg (sets the standoff)")
    ap.add_argument("--update-period", type=float, default=4.0,
                    help="how often to re-command the hold point, s")
    ap.add_argument("--no-follow", action="store_true", help="fly to target once, do not track")
    # Detection / geolocation
    ap.add_argument("--model", default="models/patch_verifier.pt")
    ap.add_argument("--min-color-score", type=float, default=0.30)
    ap.add_argument("--min-verify-prob", type=float, default=0.50)
    ap.add_argument("--ground-elevation", type=float, default=0.0)
    ap.add_argument("--alt-ref", choices=["amsl", "rel"], default="amsl")
    ap.add_argument("--camera-pitch-deg", type=float, default=None,
                    help="camera mount pitch; default = quad SDF (-20.002)")
    ap.add_argument("--nadir", action="store_true",
                    help="assume a straight-down camera (-90 deg); only if the model is changed")
    ap.add_argument("--min-depression", type=float, default=10.0)
    ap.add_argument("--reject-grazing", action="store_true")
    ap.add_argument("--attitude-sigma", type=float, default=0.5)
    ap.add_argument("--position-sigma", type=float, default=3.0)
    ap.add_argument("--altitude-sigma", type=float, default=2.0)
    # Output
    ap.add_argument("--out", default="quad_follow_output")
    ap.add_argument("--publish-tracks", action="store_true")
    ap.add_argument("--track-name", default="Sierra One")
    return ap


def run_quad_follow(fleet: Fleet, target: tuple[float, float], args,
                    gt=None) -> dict:
    """Fly the quad to ``target`` and follow the ship. Returns run stats.

    Reuses the caller's :class:`Fleet` (so an orchestrator can keep the glider
    flying); does not shut the fleet down.
    """
    cfg = fleet.config
    q = fleet.vehicle(args.asset)
    if q is None or args.asset not in fleet.cams:
        raise RuntimeError(f"asset {args.asset} has no vehicle/camera")
    cam = fleet.cams[args.asset]
    intr = cfg.assets[args.asset].camera.intrinsics()

    geo = GeoConfig(
        alt_ref=args.alt_ref, ground_elevation_m=args.ground_elevation,
        camera_pitch_deg=args.camera_pitch_deg, min_depression_deg=args.min_depression,
        reject_grazing=args.reject_grazing, attitude_sigma_deg=args.attitude_sigma,
        position_sigma_m=args.position_sigma, altitude_sigma_m=args.altitude_sigma)
    mount_deg = math.degrees(geo.mount_pitch_rad(args.asset))

    detector = VerifiedDetector(model_path=args.model,
                                min_color_score=args.min_color_score,
                                min_verify_prob=args.min_verify_prob)
    track_client = TrackClient(cfg.url(cfg.tracks_port)) if args.publish_tracks else None

    stamp = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join(args.out, stamp)
    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    det_f = open(os.path.join(run_dir, "detections.jsonl"), "w")
    gps_f = open(os.path.join(run_dir, "gps_track.csv"), "w", newline="")
    gps_w = csv.writer(gps_f)
    gps_w.writerow(["t_sim", "lat", "lon", "alt_amsl_m", "agl_m", "roll_rad", "pitch_rad", "yaw_rad"])
    side_f = open(os.path.join(run_dir, "sidecar.jsonl"), "w")

    if not args.no_fly:
        if not q.armed or q.alt_rel < args.alt * 0.8:
            log.info("Quadcopter on ground; taking off to %.0f m...", args.alt)
            if not q.takeoff(alt=args.alt, timeout=args.takeoff_timeout):
                log.error("quadcopter takeoff failed")
                det_f.close(); gps_f.close(); side_f.close()
                return {"run_dir": run_dir, "error": "takeoff_failed", "n_geo": 0}
        q.set_mode("GUIDED", timeout=5.0)
        q.set_speed(args.speed)
        log.info("Flying to target %.6f, %.6f at %.0f m (%.0f m/s)",
                 target[0], target[1], args.alt, args.speed)
        q.goto(target[0], target[1], args.alt)

    log.info("Following at %.1f Hz for %.0fs (mount %.2f deg, desired depression %.0f deg)",
             args.hz, args.duration, mount_deg, args.view_depression)

    period = 1.0 / max(args.hz, 0.1)
    end = time.monotonic() + args.duration if args.duration > 0 else float("inf")
    ship_est: Optional[tuple[float, float]] = target
    truth_errs: list[float] = []
    n_frames = n_dets = n_geo = 0
    last_cmd = 0.0
    frame_idx = 0

    try:
        while time.monotonic() < end:
            t0 = time.monotonic()
            frame = cam.grab()
            if frame is None:
                time.sleep(0.05)
                continue
            n_frames += 1
            pose = fleet.pose_at(args.asset, frame.t_sim, clock="sim") or fleet.pose(args.asset)

            truth = gt.ship_latlon() if gt is not None else None

            dets = detector.detect(frame.image, frame_idx=frame_idx, t_sim=frame.t_sim,
                                   pose=pose, cam_intrinsics=intr)
            estimates = [geo.locate(c.cx, c.cy, pose, args.asset, intr) for c in dets]
            best_i = None
            for i, e in enumerate(estimates):
                if e is None:
                    continue
                if best_i is None or dets[i].score > dets[best_i].score:
                    best_i = i
            n_dets += len(dets)

            status = f"t={frame.t_sim:7.1f}s dets={len(dets)}"
            if best_i is not None:
                e = estimates[best_i]
                n_geo += 1
                status += (f" | fix {e.lat:.6f},{e.lon:.6f} +/-{e.error_radius_m:.0f}m "
                           f"dep={e.depression_deg:.1f}{' GRAZING' if e.grazing else ''}")
                if ship_est is None:
                    ship_est = (e.lat, e.lon)
                else:
                    a = 0.5
                    ship_est = (a * e.lat + (1 - a) * ship_est[0],
                                a * e.lon + (1 - a) * ship_est[1])
                if track_client is not None:
                    track_client.post_fix(args.track_name, e.lat, e.lon, frame.t_sim)
                if truth is not None:
                    te = distance_m(e.lat, e.lon, truth[0], truth[1])
                    truth_errs.append(te)
                    status += f" | truth_err={te:.0f}m"
                det_f.write(json.dumps({
                    "frame": os.path.relpath(os.path.join(frames_dir, f"{frame_idx:05d}.jpg"), run_dir),
                    "index": frame_idx, "t_sim": frame.t_sim, "asset": args.asset,
                    "u": dets[best_i].cx, "v": dets[best_i].cy, "score": dets[best_i].score,
                    "lat": e.lat, "lon": e.lon, "error_radius_m": e.error_radius_m,
                    "depression_deg": e.depression_deg, "ground_range_m": e.ground_range_m,
                    "grazing": e.grazing}) + "\n")
            elif truth is not None and args.from_ship:
                ship_est = truth
                status += " | no detection (steering on truth)"

            if pose is not None:
                gps_w.writerow([f"{frame.t_sim:.3f}", f"{pose.lat:.7f}", f"{pose.lon:.7f}",
                                f"{pose.alt_amsl:.2f}", f"{geo.agl(pose):.2f}",
                                f"{pose.roll:.5f}", f"{pose.pitch:.5f}", f"{pose.yaw:.5f}"])
                side_f.write(json.dumps({
                    "frame": os.path.relpath(os.path.join(frames_dir, f"{frame_idx:05d}.jpg"), run_dir),
                    "index": frame_idx, "t_sim": frame.t_sim, "width": frame.width,
                    "height": frame.height, "camera": intr, "camera_mount_pitch_deg": mount_deg,
                    "pose": {"lat": pose.lat, "lon": pose.lon, "alt_rel": pose.alt_rel,
                             "alt_amsl": pose.alt_amsl, "roll": pose.roll, "pitch": pose.pitch,
                             "yaw": pose.yaw},
                    "gps": {"lat": pose.lat, "lon": pose.lon, "alt_amsl": pose.alt_amsl,
                            "agl_m": geo.agl(pose)},
                    "groundtruth": (None if truth is None else
                                    {"lat": truth[0], "lon": truth[1]}),
                }) + "\n")

            cv2.imwrite(os.path.join(frames_dir, f"{frame_idx:05d}.jpg"), frame.image)
            frame_idx += 1
            print(status, flush=True)

            if (not args.no_fly) and (not args.no_follow) and ship_est is not None \
                    and time.monotonic() - last_cmd > args.update_period:
                agl = geo.agl(pose) if pose is not None else args.alt
                standoff = max(5.0, agl / math.tan(math.radians(args.view_depression)))
                if pose is not None:
                    hl, ho, _yaw = standoff_point(ship_est[0], ship_est[1],
                                                  pose.lat, pose.lon, standoff)
                    q.goto(hl, ho, args.alt)
                    last_cmd = time.monotonic()

            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        log.info("stopped by user")
    finally:
        det_f.close(); gps_f.close(); side_f.close()

    arr = np.asarray(truth_errs) if truth_errs else np.array([])
    result = {
        "run_dir": run_dir, "n_frames": n_frames, "n_dets": n_dets, "n_geo": n_geo,
        "truth_errors": truth_errs,
        "median_error_m": float(np.median(arr)) if arr.size else None,
    }
    print("\n=======================================================")
    print(f"Step-2 follow complete: frames={n_frames} detections={n_dets} geolocated={n_geo}")
    if arr.size:
        print(f"Fix vs GROUND TRUTH: n={arr.size} median={np.median(arr):.1f} m "
              f"mean={arr.mean():.1f} m RMSE={np.sqrt((arr**2).mean()):.1f} m "
              f"within50m={100*np.mean(arr < 50):.0f}%")
    print(f"Outputs: {run_dir}")
    print("=======================================================\n")
    return result


def main() -> int:
    args = build_parser().parse_args()

    if args.nadir:
        args.camera_pitch_deg = -90.0
        log.warning("Assuming a NADIR camera (-90 deg). The sim's gimbal is fixed at "
                    "-20 deg; this is only correct if the model was changed.")
    else:
        log.warning("Quad camera is a FIXED %.3f deg-down mount (SDF); it cannot be "
                    "commanded to nadir in this sim.", 20.002)

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

    target = None
    if args.lat is not None and args.lon is not None:
        target = (args.lat, args.lon)
    elif args.tracks:
        target = load_target_from_tracks(args.tracks)
        if target is None:
            log.error("no usable track in %s", args.tracks)
            return 1
        log.info("Step-1 target from %s: %.6f, %.6f", args.tracks, *target)
    elif args.from_ship and DEV:
        gt = gt or __import__("arcticlib.groundtruth", fromlist=["GroundTruth"]).GroundTruth(cfg)
        for _ in range(20):
            sp = gt.ship_latlon()
            if sp:
                target = sp
                break
            time.sleep(1.0)
        if target is None:
            log.error("could not read the live ship pose")
            return 1
        log.info("DEV target = live ship %.6f, %.6f", *target)
    else:
        build_parser().error("give --lat/--lon, --tracks, or --from-ship (DEV)")

    try:
        result = run_quad_follow(fleet, target, args, gt=gt)
    finally:
        if gt is not None:
            gt.stop()
        fleet.shutdown()
    return 0 if result.get("n_geo", 0) >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
