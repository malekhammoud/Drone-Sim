#!/usr/bin/env python3
"""End-to-end 2-Stage Boat Detector (Step 1 + Step 2) with GPS geolocation.

Same two-stage pipeline as ``tools/detect_verified.py``:

Stage 1: Fast color-anomaly candidate detection on the full-resolution frame.
Stage 2: Learned verification with ``PatchVerifierCNN``.

New here: every verified detection is geolocated with
:mod:`arcticlib.geolocate` — cast a ray from the camera through the pixel,
rotate it into NED, intersect flat ground, and convert the offset to lat/lon with
a WGS84 geodesic. The fixed camera mount is taken from the sensor SDF
(fixed-wing ``fpv_camera`` = 8.021 deg down, quadcopter gimbal = 20.002 deg down)
and each result carries a 1-sigma error radius plus a grazing-angle flag.

Usage:
    # On a single image (give it a pose so it can geolocate):
    python tools/detect_verified_gps.py --image frame.jpg --model models/patch_verifier.pt \
        --asset fixed-wing --lat 71.99 --lon -94.82 --alt 90 --yaw 180 --pitch 2 --roll 0

    # On a recorded dataset (geolocates using each frame's sidecar pose):
    python tools/detect_verified_gps.py --dataset data/2026-09-19T18-00-00 \
        --asset fixed-wing --model models/patch_verifier.pt

    # Live from a sim camera, posting geolocated fixes to the track API:
    python tools/detect_verified_gps.py --live --asset fixed-wing \
        --model models/patch_verifier.pt --publish-tracks
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from arcticlib.geo import distance_m
from arcticlib.geolocate import (GeoConfig, intrinsics_from_fov, project_to_pixel,
                                 refine_tracks)
from tools.detect_color import Candidate
from tools.detect_verified import VerifiedDetector

log = logging.getLogger("detect_verified_gps")


def draw_geolocated(detector: VerifiedDetector, bgr: np.ndarray,
                    candidates: list[Candidate],
                    labels: Optional[list[Optional[str]]] = None,
                    gt_pt: Optional[tuple[float, float]] = None) -> np.ndarray:
    """Draw the 3-stage detector's boxes, then overlay geolocation labels."""
    vis = detector.draw_detections(bgr, candidates, gt_pt=gt_pt)
    if labels:
        for idx, c in enumerate(candidates):
            if idx < len(labels) and labels[idx]:
                cv2.putText(vis, labels[idx],
                            (c.x, min(vis.shape[0] - 4, c.y + c.h + 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
    return vis


# --------------------------------------------------------------------------- #
# Geolocation: pixel ray -> flat ground -> lat/lon (arcticlib.geolocate)
# --------------------------------------------------------------------------- #
def pixel_to_latlon(u: float, v: float, pose, camera_intrinsics: dict,
                    georef=None) -> Optional[Tuple[float, float]]:
    """Backwards-compatible wrapper: geolocate a pixel for ``pose.asset``.

    Kept so existing callers of ``tools.detect_verified.pixel_to_latlon`` keep
    working; ``georef`` is no longer needed (the geodesic is global).
    """
    if pose is None:
        return None
    asset = getattr(pose, "asset", "fixed-wing")
    geo = GeoConfig()
    est = geo.locate(u, v, pose, asset, camera_intrinsics)
    if est is None:
        return None
    return est.lat, est.lon


def _label(est) -> Optional[str]:
    if est is None:
        return None
    return f"{est.lat:.5f},{est.lon:.5f} +/-{est.error_radius_m:.0f}m d{est.depression_deg:.0f}"


def _geolocate_all(detector: VerifiedDetector, geo: GeoConfig, image: np.ndarray,
                   pose, asset: str, intrinsics: dict,
                   frame_idx: Optional[int] = None, t_sim: Optional[float] = None):
    """Run the 3-stage detector with temporal context, then geolocate each hit.

    Stage 3 (temporal persistence) associates candidates across frames in pixel
    space; every surviving candidate is then geolocated with the trig pipeline.
    """
    dets = detector.detect(image, frame_idx=frame_idx, t_sim=t_sim,
                           pose=pose, cam_intrinsics=intrinsics)
    estimates = [geo.locate(c.cx, c.cy, pose, asset, intrinsics) for c in dets]
    labels = [_label(e) for e in estimates]
    return dets, estimates, labels


# --------------------------------------------------------------------------- #
# Dataset evaluation with geolocation
# --------------------------------------------------------------------------- #
def evaluate_dataset_gps(dataset_dir: str, asset: str, detector: VerifiedDetector,
                         geo: GeoConfig, out_dir: Optional[str] = None,
                         max_frames: int = 0, fuse: bool = True,
                         track_gate_m: float = 2000.0,
                         min_track_hits: int = 2,
                         score_truth: bool = True,
                         match_px: float = 60.0) -> dict:
    """Run the verified detector over a recorded dataset and geolocate hits."""
    asset_dir = os.path.join(dataset_dir, asset)
    sidecar_path = os.path.join(asset_dir, "sidecar.jsonl")
    if not os.path.exists(sidecar_path):
        sidecar_path = os.path.join(dataset_dir, "sidecar.jsonl")
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(f"sidecar.jsonl not found under {dataset_dir}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    frames = 0
    frames_with_gt = 0
    gt_hits = 0
    total_dets = 0
    geolocated = 0
    grazing = 0
    errs: list[float] = []
    geo_records: list[dict] = []
    truth_errors: list[float] = []      # |fix - true ship| in metres
    truth_radii: list[float] = []       # predicted error radius for those fixes
    truth_resid: list[float] = []       # pixel gap between projected truth and hit
    truth_frames = 0
    truth_matched = 0

    det_path = os.path.join(out_dir, "detections.jsonl") if out_dir else None
    det_file = open(det_path, "w") if det_path else None
    try:
        with open(sidecar_path) as f:
            for line in f:
                if max_frames > 0 and frames >= max_frames:
                    break
                data = json.loads(line)
                frame_rel = data.get("frame")
                frame_path = os.path.join(dataset_dir, frame_rel) if frame_rel else None
                if not frame_path or not os.path.exists(frame_path):
                    continue
                img = cv2.imread(frame_path)
                if img is None:
                    continue
                pose = data.get("pose")
                intr = data.get("camera") or {}
                dets, estimates, labels = _geolocate_all(
                    detector, geo, img, pose, asset, intr,
                    frame_idx=data.get("index"), t_sim=data.get("t_sim"))
                frames += 1
                total_dets += len(dets)
                for c, e in zip(dets, estimates):
                    if e is not None:
                        geolocated += 1
                        errs.append(e.error_radius_m)
                        if e.grazing:
                            grazing += 1
                        rec = {
                            "frame": frame_rel, "index": data.get("index"),
                            "t_sim": data.get("t_sim"), "asset": asset,
                            "u": c.cx, "v": c.cy, "score": c.score,
                            "lat": e.lat, "lon": e.lon,
                            "error_radius_m": e.error_radius_m,
                            "depression_deg": e.depression_deg,
                            "ground_range_m": e.ground_range_m,
                            "grazing": e.grazing,
                        }
                        geo_records.append(rec)
                        if det_file is not None:
                            det_file.write(json.dumps(rec) + "\n")

                gt = data.get("groundtruth", {})
                gt_pt = gt.get("point_px_approx")
                h, w = img.shape[:2]
                if gt_pt is not None and 0 <= gt_pt[0] < w and 0 <= gt_pt[1] < h:
                    frames_with_gt += 1
                    if any(c.dist_to(gt_pt[0], gt_pt[1]) <= 25.0 or
                           c.contains(gt_pt[0], gt_pt[1], margin=8.0) for c in dets):
                        gt_hits += 1

                # -- Score geolocation against the true ship lat/lon --------- #
                tlat, tlon = gt.get("lat"), gt.get("lon")
                if score_truth and tlat is not None and tlon is not None and pose is not None:
                    truth_frames += 1
                    uv = project_to_pixel(
                        tlat, tlon, pose.get("lat"), pose.get("lon"), geo.agl(pose),
                        pose.get("yaw"), pose.get("pitch"), pose.get("roll"),
                        gimbal_pitch=geo.mount_pitch_rad(asset), intrinsics=intr)
                    if uv is not None and dets:
                        k = min(range(len(dets)), key=lambda j: dets[j].dist_to(uv[0], uv[1]))
                        resid = dets[k].dist_to(uv[0], uv[1])
                        if resid <= match_px and estimates[k] is not None:
                            truth_matched += 1
                            truth_resid.append(resid)
                            truth_errors.append(distance_m(estimates[k].lat, estimates[k].lon, tlat, tlon))
                            truth_radii.append(estimates[k].error_radius_m)

                if out_dir:
                    vis = draw_geolocated(
                        detector, img, dets, labels=labels,
                        gt_pt=(gt_pt[0], gt_pt[1]) if gt_pt else None)
                    cv2.imwrite(os.path.join(out_dir, f"det_{os.path.basename(frame_path)}"), vis)
    finally:
        if det_file is not None:
            det_file.close()

    # -- Multi-frame refinement: fuse per-frame fixes into targets ---------- #
    tracks = []
    if fuse and geo_records:
        tracks = refine_tracks(geo_records, track_gate_m=track_gate_m,
                               min_frames=min_track_hits)
        if out_dir:
            with open(os.path.join(out_dir, "tracks.jsonl"), "w") as fh:
                for t in tracks:
                    fh.write(json.dumps({
                        "lat": t.lat, "lon": t.lon,
                        "error_radius_m": t.error_radius_m,
                        "n": t.n, "n_frames": t.n_frames,
                        "t_first": t.t_first, "t_last": t.t_last,
                        "mean_score": t.mean_score,
                        "min_depression_deg": t.min_depression_deg,
                        "max_depression_deg": t.max_depression_deg,
                    }) + "\n")

    recall = (gt_hits / frames_with_gt) if frames_with_gt else 0.0
    truth_stats = {}
    if truth_errors:
        te = np.asarray(truth_errors, dtype=float)
        truth_stats = {
            "truth_frames": truth_frames,
            "truth_matched": truth_matched,
            "truth_match_rate": (truth_matched / truth_frames) if truth_frames else 0.0,
            "truth_median_error_m": float(np.median(te)),
            "truth_mean_error_m": float(te.mean()),
            "truth_rmse_m": float(np.sqrt((te ** 2).mean())),
            "truth_median_predicted_radius_m": float(np.median(truth_radii)),
            "truth_median_px_residual": float(np.median(truth_resid)),
        }
    metrics = {
        "frames_evaluated": frames,
        "frames_with_gt_in_view": frames_with_gt,
        "gt_detected": gt_hits,
        "recall": recall,
        "total_detections": total_dets,
        "geolocated": geolocated,
        "grazing": grazing,
        "median_error_radius_m": float(np.median(errs)) if errs else 0.0,
        "tracks": len(tracks),
    }
    metrics.update(truth_stats)
    print("\n--- Verified + GPS Evaluation ---")
    print(f"Frames evaluated       : {frames}")
    print(f"GT detected (Recall)   : {gt_hits}/{frames_with_gt} ({recall*100:.1f}%)")
    print(f"Total detections       : {total_dets}  (geolocated {geolocated}, grazing {grazing})")
    if errs:
        print(f"Median error radius    : {np.median(errs):.1f} m")
    if det_path:
        print(f"Geolocated detections  : {det_path}")
    if fuse:
        print(f"Fused tracks (>={min_track_hits} frames): {len(tracks)}")
        for i, t in enumerate(tracks[:10]):
            print(f"  track {i+1}: {t.lat:.5f}, {t.lon:.5f}  +/-{t.error_radius_m:.0f}m  "
                  f"hits={t.n} frames={t.n_frames} t={t.t_first:.0f}..{t.t_last:.0f} "
                  f"score={t.mean_score:.2f} dep={t.min_depression_deg:.1f}..{t.max_depression_deg:.1f}")
        if out_dir:
            print(f"Fused tracks           : {os.path.join(out_dir, 'tracks.jsonl')}")
    if truth_stats:
        print(f"\n--- Geolocation vs GROUND TRUTH ---")
        print(f"Truth frames / matched : {truth_frames} / {truth_matched} "
              f"({truth_stats['truth_match_rate']*100:.1f}%)")
        print(f"Position error (matched): median {truth_stats['truth_median_error_m']:.1f} m, "
              f"mean {truth_stats['truth_mean_error_m']:.1f} m, "
              f"RMSE {truth_stats['truth_rmse_m']:.1f} m")
        print(f"Predicted radius       : median {truth_stats['truth_median_predicted_radius_m']:.1f} m "
              f"(vs actual median {truth_stats['truth_median_error_m']:.1f} m)")
        print(f"Projected-truth pixel residual: median {truth_stats['truth_median_px_residual']:.1f} px")
    elif truth_frames:
        print(f"\n--- Geolocation vs GROUND TRUTH ---")
        print(f"Truth frames present: {truth_frames}, but none matched a detection within {match_px:.0f} px.")
    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_geo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ground-elevation", type=float, default=0.0,
                        help="Terrain elevation at the target, m (default 0 = sea level)")
    parser.add_argument("--alt-ref", choices=["amsl", "rel"], default="amsl",
                        help="Height reference for AGL (default amsl)")
    parser.add_argument("--camera-pitch-deg", type=float, default=None,
                        help="Override camera mount pitch, deg (0=horizon, -90=down)")
    parser.add_argument("--min-depression", type=float, default=10.0,
                        help="Flag/reject rays shallower than this (deg, default 10)")
    parser.add_argument("--reject-grazing", action="store_true",
                        help="Drop grazing detections instead of flagging them")
    parser.add_argument("--attitude-sigma", type=float, default=0.5, help="1-sigma attitude error (deg)")
    parser.add_argument("--position-sigma", type=float, default=3.0, help="1-sigma GPS position error (m)")
    parser.add_argument("--altitude-sigma", type=float, default=2.0, help="1-sigma altitude error (m)")
    parser.add_argument("--no-geolocate", action="store_true", help="Disable geolocation")


def _geo_from_args(args) -> GeoConfig:
    return GeoConfig(
        alt_ref=args.alt_ref,
        ground_elevation_m=args.ground_elevation,
        camera_pitch_deg=args.camera_pitch_deg,
        min_depression_deg=args.min_depression,
        reject_grazing=args.reject_grazing,
        attitude_sigma_deg=args.attitude_sigma,
        position_sigma_m=args.position_sigma,
        altitude_sigma_m=args.altitude_sigma,
        enabled=not args.no_geolocate,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-stage boat detector with GPS geolocation.")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--dataset", help="Recorded dataset directory")
    parser.add_argument("--asset", default="quadcopter", help="Asset name")
    parser.add_argument("--model", default="models/patch_verifier.pt", help="Patch verifier model weights")
    parser.add_argument("--live", action="store_true", help="Run live on camera feed")
    parser.add_argument("--hz", type=float, default=2.0, help="Live frame rate")
    parser.add_argument("--out-dir", default="verified_output", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process in --dataset mode (0 = all)")
    parser.add_argument("--publish-tracks", action="store_true", help="Publish tracks to /api/tracks")
    parser.add_argument("--track-gate", type=float, default=2000.0,
                        help="Max association distance for multi-frame tracks, m (default 2000)")
    parser.add_argument("--min-track-hits", type=int, default=2,
                        help="Min distinct frames to call a fused track (default 2)")
    parser.add_argument("--no-fuse", action="store_true",
                        help="Disable multi-frame refinement (per-frame fixes only)")
    parser.add_argument("--no-temporal", action="store_true",
                        help="Disable the Stage 3 temporal persistence filter")
    parser.add_argument("--min-hits", type=int, default=8,
                        help="Temporal hits needed to confirm a track (default 8)")
    parser.add_argument("--max-misses", type=int, default=4,
                        help="Temporal misses before a track is dropped (default 4)")
    parser.add_argument("--track-mode", choices=["pixel", "geo"], default="pixel",
                        help="Stage 3 association space (default pixel)")
    # Single-image pose (all angles in degrees).
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--alt", type=float, default=None, help="Altitude MSL, m")
    parser.add_argument("--yaw", type=float, default=0.0, help="Heading deg from true north")
    parser.add_argument("--pitch", type=float, default=0.0, help="Body pitch deg (+ = nose up)")
    parser.add_argument("--roll", type=float, default=0.0, help="Body roll deg (+ = right wing down)")
    _add_geo_args(parser)
    args = parser.parse_args()

    geo = _geo_from_args(args)
    detector = VerifiedDetector(
        model_path=args.model,
        enable_temporal=not args.no_temporal,
        min_hits=args.min_hits,
        max_misses=args.max_misses,
        track_mode=args.track_mode)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: Cannot read {args.image}")
            return 1

        pose = None
        if args.lat is not None and args.lon is not None and args.alt is not None:
            pose = {"lat": args.lat, "lon": args.lon, "alt_amsl": args.alt,
                    "alt_rel": args.alt, "yaw": math.radians(args.yaw),
                    "pitch": math.radians(args.pitch), "roll": math.radians(args.roll)}
        elif not args.no_geolocate:
            print("Note: no --lat/--lon/--alt given, so detections are not geolocated.")

        from arcticlib.config import load_config
        cfg = load_config()
        spec = cfg.assets.get(args.asset)
        h, w = img.shape[:2]
        if spec is not None and spec.camera is not None:
            intr = spec.camera.intrinsics()
            if (intr["width"], intr["height"]) != (w, h):
                intr = intrinsics_from_fov(w, h, spec.camera.hfov_deg, spec.camera.vfov_deg)
        else:
            intr = intrinsics_from_fov(w, h, 60.0, 36.0)

        dets, estimates, labels = _geolocate_all(detector, geo, img, pose, args.asset, intr)
        print(f"Detected {len(dets)} verified boat candidates:")
        for i, d in enumerate(dets):
            line = f"  #{i+1}: score={d.score:.3f}, center=({d.cx:.1f}, {d.cy:.1f}), bbox={d.bbox()}"
            if estimates[i] is not None:
                e = estimates[i]
                line += (f"  ->  {e.lat:.6f}, {e.lon:.6f} (+/-{e.error_radius_m:.1f} m, "
                         f"dep={e.depression_deg:.1f} deg{', GRAZING' if e.grazing else ''})")
            print(line)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            vis = draw_geolocated(detector, img, dets, labels=labels)
            out_path = os.path.join(args.out_dir, f"verified_{os.path.basename(args.image)}")
            cv2.imwrite(out_path, vis)
            print(f"Saved visualization to {out_path}")
        return 0

    if args.dataset:
        evaluate_dataset_gps(args.dataset, args.asset, detector, geo, out_dir=args.out_dir,
                             max_frames=args.max_frames, fuse=not args.no_fuse,
                             track_gate_m=args.track_gate,
                             min_track_hits=args.min_track_hits)
        return 0

    if args.live:
        from arcticlib.config import load_config
        from arcticlib.fleet import Fleet
        from arcticlib.tracks import TrackClient

        cfg = load_config()
        fleet = Fleet.from_config(cfg, connect=True)
        track_client = TrackClient(cfg.url(cfg.tracks_port)) if args.publish_tracks else None

        if args.asset not in fleet.cams:
            print(f"Error: Asset '{args.asset}' has no camera.")
            return 1

        cam = fleet.cams[args.asset]
        intr = cfg.assets[args.asset].camera.intrinsics()
        mount = math.degrees(geo.mount_pitch_rad(args.asset))
        period = 1.0 / max(args.hz, 0.1)
        print(f"Live verified+GEO detector on {args.asset} at {args.hz} Hz "
              f"(camera mount {mount:.2f} deg, alt_ref={geo.alt_ref})...")
        frame_idx = 0
        try:
            while True:
                t0 = time.monotonic()
                frame = cam.grab()
                if frame is not None:
                    pose = fleet.pose_at(args.asset, frame.t_sim, clock="sim") or fleet.pose(args.asset)
                    dets, estimates, labels = _geolocate_all(
                        detector, geo, frame.image, pose, args.asset, intr,
                        frame_idx=frame_idx, t_sim=frame.t_sim)
                    frame_idx += 1

                    status = f"[{time.strftime('%H:%M:%S')}] Detections: {len(dets)}"
                    if dets:
                        best = dets[0]
                        est = estimates[0]
                        status += f" | Top: score={best.score:.2f} at ({best.cx:.1f}, {best.cy:.1f})"
                        if est is not None:
                            status += (f" -> ({est.lat:.6f}, {est.lon:.6f}) "
                                       f"+/-{est.error_radius_m:.0f}m dep={est.depression_deg:.1f}"
                                       f"{' GRAZING' if est.grazing else ''}")
                            if track_client is not None:
                                res = track_client.post("Sierra One", est.lat, est.lon)
                                status += f" | track={res is not None}"
                        else:
                            status += " -> no ground fix (above horizon / rejected)"
                    print(status)

                    if args.out_dir:
                        os.makedirs(args.out_dir, exist_ok=True)
                        vis = draw_geolocated(detector, frame.image, dets, labels=labels)
                        cv2.imwrite(os.path.join(args.out_dir, "latest_live_detection.jpg"), vis)

                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            fleet.shutdown()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
