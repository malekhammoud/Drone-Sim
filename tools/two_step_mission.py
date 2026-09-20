#!/usr/bin/env python3
"""Single command for the two-step search-and-refine mission.

    Phase 1 (glider)  fixed-wing searches / points at the vessel, detects it and
                      geolocates it -> a target lat/lon.
    Phase 2 (quad)    hand the target straight to ``quad_follow_ship``: the
                      quadcopter flies there, holds a standoff and refines the
                      fix to ~10 m.

Both phases share one :class:`Fleet` connection, so the glider keeps flying while
the quad works.

Usage:
    # Fully automatic, DEV-assisted (glider steers at the live vessel to acquire):
    ARCTICSIM_DEV=1 python tools/two_step_mission.py --from-ship \
        --search-duration 120 --duration 180

    # Skip the search and hand off an existing step-1 track:
    python tools/two_step_mission.py --tracks tools/gt_runs/<stamp>/eval/tracks.jsonl

    # Manual target:
    python tools/two_step_mission.py --lat 71.9865 --lon -94.8983
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from arcticlib.config import load_config
from arcticlib.fleet import Fleet
from arcticlib.geo import distance_m
from arcticlib.geolocate import GeoConfig, refine_tracks
from tools.detect_verified import VerifiedDetector
from tools.quad_follow_ship import build_parser as quad_parser
from tools.quad_follow_ship import load_target_from_tracks, run_quad_follow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("two_step")

DEV = os.environ.get("ARCTICSIM_DEV") == "1"


def search_for_ship(fleet: Fleet, args, gt=None) -> dict:
    """Phase 1: fly the glider, detect + geolocate the vessel, return a target.

    Returns ``{"target": (lat, lon), "fixes": [...], "source": ...}``.
    """
    asset = args.glider
    v = fleet.vehicle(asset)
    cam = fleet.cams.get(asset)
    if v is None or cam is None:
        raise RuntimeError(f"glider {asset} has no vehicle/camera")
    intr = fleet.config.assets[asset].camera.intrinsics()
    geo = GeoConfig(camera_pitch_deg=args.glider_camera_pitch_deg,
                    ground_elevation_m=args.glider_ground_elevation)
    detector = VerifiedDetector(model_path=args.model,
                                min_color_score=args.min_color_score,
                                min_verify_prob=args.min_verify_prob)

    # Get the glider airborne and pointed at the search area.
    if not args.no_fly:
        if not v.armed or v.alt_rel < args.search_alt * 0.6:
            log.info("Glider on ground; taking off to %.0f m...", args.search_alt)
            if not v.takeoff(alt=args.search_alt, timeout=args.takeoff_timeout):
                log.error("glider takeoff failed")
                return {"target": None, "fixes": [], "source": "takeoff_failed"}
        v.set_mode("GUIDED", timeout=5.0)
        v.set_airspeed(args.search_speed)

    run_dir = os.path.join(args.out, _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    fixes_path = os.path.join(run_dir, "step1_fixes.jsonl")
    fixes_f = open(fixes_path, "w")

    log.info("Phase 1: searching with %s for up to %.0fs...", asset, args.search_duration)
    period = 1.0 / max(args.hz, 0.1)
    end = time.monotonic() + args.search_duration
    fixes: list[dict] = []
    last_cmd = 0.0
    idx = 0
    target = None

    try:
        while time.monotonic() < end:
            t0 = time.monotonic()
            # Steer: DEV chases the true ship; otherwise just keep it flying.
            if not args.no_fly and time.monotonic() - last_cmd > 5.0:
                if args.from_ship and gt is not None:
                    sp = gt.ship_latlon()
                    if sp:
                        v.goto(sp[0], sp[1], args.search_alt)
                last_cmd = time.monotonic()

            frame = cam.grab()
            if frame is None:
                time.sleep(0.05)
                continue
            pose = fleet.pose_at(asset, frame.t_sim, clock="sim") or fleet.pose(asset)
            dets = detector.detect(frame.image, frame_idx=idx, t_sim=frame.t_sim,
                                   pose=pose, cam_intrinsics=intr)
            for c in dets:
                e = geo.locate(c.cx, c.cy, pose, asset, intr)
                if e is None:
                    continue
                rec = {"index": idx, "t_sim": frame.t_sim, "score": c.score,
                       "lat": e.lat, "lon": e.lon, "error_radius_m": e.error_radius_m,
                       "depression_deg": e.depression_deg, "grazing": e.grazing}
                fixes.append(rec)
                fixes_f.write(json.dumps(rec) + "\n")
            idx += 1

            # Acquire once we have a few geolocated hits.
            if len(fixes) >= args.min_hits:
                tracks = refine_tracks(fixes, min_frames=max(2, args.min_hits // 2))
                if tracks:
                    target = (tracks[0].lat, tracks[0].lon)
                    log.info("Phase 1 acquired a target from %d fixes: %.6f, %.6f "
                             "(hits=%d, %.0f m err bar)",
                             len(fixes), target[0], target[1],
                             tracks[0].n, tracks[0].error_radius_m)
                    break
                best = max(fixes, key=lambda r: r["score"])
                target = (best["lat"], best["lon"])
                log.info("Phase 1 acquired a single-fix target: %.6f, %.6f (score %.2f)",
                         target[0], target[1], best["score"])
                break

            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        fixes_f.close()

    result = {"target": target, "fixes": fixes, "fixes_path": fixes_path}
    if target is None:
        log.warning("Phase 1 found no target in %.0fs", args.search_duration)
    return result


def main() -> int:
    ap = quad_parser()  # reuse every step-2 option
    # Phase-1 options
    ap.add_argument("--glider", default="fixed-wing", help="search aircraft")
    ap.add_argument("--search-duration", type=float, default=120.0)
    ap.add_argument("--search-alt", type=float, default=90.0)
    ap.add_argument("--search-speed", type=float, default=20.0)
    ap.add_argument("--glider-camera-pitch-deg", type=float, default=None)
    ap.add_argument("--glider-ground-elevation", type=float, default=0.0)
    ap.add_argument("--min-hits", type=int, default=3,
                    help="geolocated glider fixes needed to hand off")
    ap.add_argument("--skip-search", action="store_true",
                    help="use --lat/--lon or --tracks directly, no glider search")
    args = ap.parse_args()

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

    try:
        # -- Resolve the target (skip search if given) --------------------- #
        target = None
        if args.lat is not None and args.lon is not None:
            target = (args.lat, args.lon)
            log.info("Using manual target %.6f, %.6f", *target)
        elif args.tracks:
            target = load_target_from_tracks(args.tracks)
            log.info("Using step-1 track %s -> %.6f, %.6f", args.tracks, *target)
        elif args.from_ship and args.skip_search and gt is not None:
            # Handoff test without the glider: use the live vessel as the step-1
            # result (DEV only). GroundTruth needs a moment for its first pose.
            for _ in range(20):
                sp = gt.ship_latlon()
                if sp:
                    target = sp
                    log.info("DEV handoff target = live ship %.6f, %.6f", *target)
                    break
                time.sleep(1.0)

        if target is None and not args.skip_search:
            phase1 = search_for_ship(fleet, args, gt=gt)
            target = phase1["target"]
            if target is not None:
                with open(os.path.join(os.path.dirname(phase1["fixes_path"]),
                                       "step1_target.json"), "w") as fh:
                    json.dump({"lat": target[0], "lon": target[1],
                               "n_fixes": len(phase1["fixes"])}, fh, indent=2)

        if target is None:
            log.error("No target to hand off; aborting before step 2.")
            return 1

        log.info("========== HANDOFF: quad -> %.6f, %.6f ==========", target[0], target[1])
        result = run_quad_follow(fleet, target, args, gt=gt)
        return 0 if result.get("n_geo", 0) > 0 else 1
    finally:
        if gt is not None:
            gt.stop()
        fleet.shutdown()


if __name__ == "__main__":
    sys.exit(main())
