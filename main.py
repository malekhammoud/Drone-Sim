#!/usr/bin/env python3
"""ArcticSim full mission: wing + quad + GPS + 3-stage CV, in one command.

This is the single entry point. It runs, in order:

  1. **Wing (glider) circling patrol** — safe 3-tier strait waypoints (or, with
     ``--follow-ship``, chase the live vessel in DEV).
  2. **Recording + GPS logging** — frames, ``sidecar.jsonl`` (with a ``gps``
     block), ``gps_track.csv``.
  3. **3-stage CV** — color anomaly -> CNN verifier (``models/patch_verifier.pt``)
     -> temporal persistence.
  4. **Geolocated tracks** — every hit is turned into lat/lon with the trig
     geolocator; inverse-variance fused tracks go to ``tracks.jsonl``.
  5. **Automatic handoff** — the best fused track becomes the quad's target.
  6. **Quadcopter fly-to + standoff follow** — the quad flies to the target,
     holds a standoff so the ship sits at a steep, accurate depression, keeps
     following it, and records its own GPS + geolocated detections.

It stays modular: this file only orchestrates ``tools/patrol_and_record_gps``,
``tools/quad_follow_ship`` and ``arcticlib``. Each phase is still runnable and
testable on its own.

Usage:
    # Bare run — full mission with defaults. In DEV the wing auto-chases the
    # vessel and the handoff can fall back to it; otherwise the wing runs the
    # normal safe patrol and hands off whatever the detector confirms:
    python main.py
    ARCTICSIM_DEV=1 python main.py

    # Full DEV-assisted mission (wing chases the vessel, then quad follows):
    ARCTICSIM_DEV=1 python main.py --follow-ship \
        --patrol-duration 150 --quad-duration 180

    # Normal patrol search, hand off whatever the wing finds:
    python main.py --patrol-duration 300 --quad-duration 180

    # Skip the wing, hand off an existing track / coordinate:
    python main.py --skip-patrol --tracks tools/gt_runs/<stamp>/eval/tracks.jsonl
    python main.py --skip-patrol --lat 71.9865 --lon -94.8983
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arcticlib.config import load_config
from arcticlib.fleet import Fleet
from tools.patrol_and_record_gps import build_parser as patrol_parser
from tools.patrol_and_record_gps import run_patrol
from tools.quad_follow_ship import build_parser as quad_parser
from tools.quad_follow_ship import load_target_from_tracks, run_quad_follow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mission")

DEV = os.environ.get("ARCTICSIM_DEV") == "1"


def _patrol_ns(args) -> argparse.Namespace:
    """Patrol options for phase 1, derived from the mission args."""
    ns = patrol_parser().parse_args([])
    ns.alt = args.patrol_alt
    ns.duration = args.patrol_duration
    ns.speed = args.speed
    ns.margin = args.margin
    ns.spacing_lon = args.spacing_lon
    ns.hz = args.hz
    ns.out = os.path.join(args.out, "wing")
    ns.follow_ship = args.follow_ship
    ns.no_fly = args.no_fly
    ns.model = args.model
    ns.no_geolocate = args.no_geolocate
    ns.ground_elevation = args.ground_elevation
    ns.alt_ref = args.alt_ref
    ns.camera_pitch_deg = args.camera_pitch_deg
    ns.min_depression = args.min_depression
    ns.reject_grazing = args.reject_grazing
    ns.attitude_sigma = args.attitude_sigma
    ns.position_sigma = args.position_sigma
    ns.altitude_sigma = args.altitude_sigma
    ns.track_gate = args.track_gate
    ns.min_track_hits = args.min_track_hits
    ns.no_temporal = args.no_temporal
    ns.min_hits = args.min_hits
    return ns


def _quad_ns(args) -> argparse.Namespace:
    """Quad options for phase 2, derived from the mission args."""
    ns = quad_parser().parse_args([])
    ns.alt = args.quad_alt
    ns.duration = args.quad_duration
    ns.speed = args.quad_speed
    ns.view_depression = args.view_depression
    ns.update_period = args.update_period
    ns.takeoff_timeout = args.takeoff_timeout
    ns.out = os.path.join(args.out, "quad")
    ns.model = args.model
    ns.publish_tracks = args.publish_tracks
    ns.track_name = args.track_name
    ns.from_ship = args.follow_ship          # DEV: steer on truth if no detection
    ns.no_geolocate = args.no_geolocate
    ns.ground_elevation = args.ground_elevation
    ns.alt_ref = args.alt_ref
    ns.camera_pitch_deg = args.camera_pitch_deg
    ns.min_depression = args.min_depression
    ns.reject_grazing = args.reject_grazing
    ns.attitude_sigma = args.attitude_sigma
    ns.position_sigma = args.position_sigma
    ns.altitude_sigma = args.altitude_sigma
    return ns


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Full wing+quad GPS mission (one command).")
    # Phase 1 — wing patrol
    ap.add_argument("--patrol-duration", type=float, default=180.0, help="wing patrol seconds")
    ap.add_argument("--patrol-alt", type=float, default=90.0, help="wing takeoff altitude, m")
    ap.add_argument("--speed", type=float, default=20.0, help="wing cruise airspeed m/s")
    ap.add_argument("--margin", type=float, default=120.0, help="wing safe coast margin, m")
    ap.add_argument("--spacing-lon", type=float, default=0.01, help="wing lane spacing, deg")
    ap.add_argument("--follow-ship", dest="follow_ship", action="store_true", default=None,
                    help="wing chases the live vessel (auto-on with ARCTICSIM_DEV=1)")
    ap.add_argument("--no-follow-ship", dest="follow_ship", action="store_false", default=None,
                    help="force the normal safe patrol even in DEV")
    ap.add_argument("--no-fly", action="store_true", help="record only, no commands")
    # Phase 2 — quad
    ap.add_argument("--quad-alt", type=float, default=25.0, help="quad hover altitude (rel), m")
    ap.add_argument("--quad-duration", type=float, default=180.0, help="quad follow seconds")
    ap.add_argument("--quad-speed", type=float, default=18.0, help="quad cruise speed m/s")
    ap.add_argument("--view-depression", type=float, default=45.0,
                    help="desired ship depression at the quad, deg")
    ap.add_argument("--update-period", type=float, default=4.0, help="quad re-aim period, s")
    ap.add_argument("--takeoff-timeout", type=float, default=150.0)
    # Handoff
    ap.add_argument("--skip-patrol", action="store_true", help="skip phase 1")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--tracks", default=None, help="existing step-1 tracks.jsonl")
    # Shared
    ap.add_argument("--out", default="mission_output", help="output root")
    ap.add_argument("--hz", type=float, default=3.0, help="camera rate")
    ap.add_argument("--model", default="models/patch_verifier.pt")
    ap.add_argument("--no-temporal", action="store_true", help="disable Stage-3 temporal filter")
    ap.add_argument("--min-hits", type=int, default=8, help="temporal hits to confirm")
    ap.add_argument("--track-gate", type=float, default=2000.0)
    ap.add_argument("--min-track-hits", type=int, default=2)
    ap.add_argument("--ground-elevation", type=float, default=0.0)
    ap.add_argument("--alt-ref", choices=["amsl", "rel"], default="amsl")
    ap.add_argument("--camera-pitch-deg", type=float, default=None)
    ap.add_argument("--min-depression", type=float, default=10.0)
    ap.add_argument("--reject-grazing", action="store_true")
    ap.add_argument("--attitude-sigma", type=float, default=0.5)
    ap.add_argument("--position-sigma", type=float, default=3.0)
    ap.add_argument("--altitude-sigma", type=float, default=2.0)
    ap.add_argument("--no-geolocate", action="store_true")
    ap.add_argument("--publish-tracks", action="store_true")
    ap.add_argument("--track-name", default="Sierra One")
    return ap


def resolve_defaults(args) -> argparse.Namespace:
    """Fill in no-argument defaults so a bare ``python main.py`` runs the mission.

    In DEV the wing auto-chases the live vessel (and the handoff can fall back to
    it); otherwise the wing flies the normal safe patrol and hands off whatever
    the 3-stage detector confirms. ``--no-follow-ship`` forces the patrol in DEV.
    """
    if args.follow_ship is None:
        args.follow_ship = DEV
    return args


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

    try:
        # -- Resolve the handoff target ----------------------------------- #
        target = None
        if args.lat is not None and args.lon is not None:
            target = (args.lat, args.lon)
            log.info("Manual target: %.6f, %.6f", *target)
        elif args.tracks:
            target = load_target_from_tracks(args.tracks)
            if target is None:
                log.error("no usable track in %s", args.tracks)
                return 1
            log.info("Step-1 track %s -> %.6f, %.6f", args.tracks, *target)

        if target is None and not args.skip_patrol:
            log.info("========== PHASE 1: WING PATROL + CV + GPS ==========")
            try:
                res = run_patrol(fleet, _patrol_ns(args), gt=gt)
            except RuntimeError as exc:
                log.error("Wing patrol failed: %s", exc)
                return 1
            target = res.get("best_target")
            if target is not None:
                log.info("Wing produced a handoff target: %.6f, %.6f", *target)
            elif DEV and gt is not None:
                target = gt.ship_latlon()
                if target:
                    log.warning("No wing track; DEV fallback to live vessel %.6f, %.6f", *target)

        if target is None:
            log.error("No target to hand off. Aborting before phase 2.")
            return 1

        # -- Phase 2: quad fly-to + follow -------------------------------- #
        log.info("========== HANDOFF -> PHASE 2: QUAD -> %.6f, %.6f ==========", *target)
        result = run_quad_follow(fleet, target, _quad_ns(args), gt=gt)
        if result.get("error"):
            log.error("Quad phase failed: %s", result["error"])
            return 1
        log.info("Mission complete: wing + quad + GPS + 3-stage CV.")
        return 0
    finally:
        if gt is not None:
            gt.stop()
        fleet.shutdown()


if __name__ == "__main__":
    sys.exit(main())
