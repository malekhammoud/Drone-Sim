#!/usr/bin/env python3
"""End-to-end 2-Stage Boat Detector (Step 1 + Step 2).

Stage 1: Fast color-anomaly candidate detection on the full-resolution frame
         (LAB/HSV/Mahalanobis color distance + connected components).
Stage 2: Learned verification with PatchVerifierCNN (evaluates 48x48 patches
         around candidates to prune false positives).

Optionally geolocates detections using vehicle pose and posts them to the
competition Track API (/api/tracks).

Usage:
    # On a single image:
    python tools/detect_verified.py --image path/to/frame.jpg --model models/patch_verifier.pt

    # On a recorded dataset:
    python tools/detect_verified.py --dataset data/2026-09-19T18-00-00 --model models/patch_verifier.pt

    # Live from sim camera:
    python tools/detect_verified.py --live --asset quadcopter --model models/patch_verifier.pt
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
import torch

from tools.detect_color import Candidate, ColorAnomalyDetector
from tools.train_patch_verifier import PatchVerifierCNN

log = logging.getLogger("detect_verified")


# --------------------------------------------------------------------------- #
# Stage 3: Temporal Persistence & Multi-Frame Tracking
# --------------------------------------------------------------------------- #
class Tracklet:
    """Maintains state for a candidate tracked across video frames."""

    def __init__(self,
                 track_id: int,
                 cx: float,
                 cy: float,
                 w: int,
                 h: int,
                 frame_idx: int = 0,
                 t_sim: Optional[float] = None,
                 lat: Optional[float] = None,
                 lon: Optional[float] = None):
        self.track_id = track_id
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.frame_idx = frame_idx
        self.t_sim = t_sim
        self.lat = lat
        self.lon = lon
        self.hits = 1
        self.misses = 0
        self.confirmed = False

    def update_pos(self,
                   cx: float,
                   cy: float,
                   w: int,
                   h: int,
                   frame_idx: int,
                   t_sim: Optional[float] = None,
                   lat: Optional[float] = None,
                   lon: Optional[float] = None,
                   alpha: float = 0.4):
        """Smooth position update using exponential moving average."""
        self.cx = (1.0 - alpha) * self.cx + alpha * cx
        self.cy = (1.0 - alpha) * self.cy + alpha * cy
        self.w = w
        self.h = h
        self.frame_idx = frame_idx
        if t_sim is not None:
            self.t_sim = t_sim
        if lat is not None and lon is not None:
            if self.lat is not None and self.lon is not None:
                self.lat = (1.0 - alpha) * self.lat + alpha * lat
                self.lon = (1.0 - alpha) * self.lon + alpha * lon
            else:
                self.lat = lat
                self.lon = lon
        self.hits += 1
        self.misses = 0


class TemporalTrackFilter:
    """Stage 3: Temporal persistence and multi-frame association filter.

    Tracks candidate detections across consecutive frames to confirm persistent
    vessel targets and reject transient false alarms (glints, specular wave reflections).
    """

    def __init__(self,
                 min_hits: int = 8,
                 max_misses: int = 4,
                 max_dist_px: float = 65.0,
                 max_dist_geo_m: float = 120.0,
                 mode: str = "pixel"):
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.max_dist_px = max_dist_px
        self.max_dist_geo_m = max_dist_geo_m
        self.mode = mode  # 'pixel' or 'geo'
        self.next_id = 1
        self.tracks: list[Tracklet] = []
        self.current_frame = 0

    def update(self,
               candidates: list[Candidate],
               frame_idx: Optional[int] = None,
               t_sim: Optional[float] = None,
               pose=None,
               cam_intrinsics: Optional[dict] = None,
               georef=None) -> list[Candidate]:
        if frame_idx is None:
            self.current_frame += 1
            frame_idx = self.current_frame
        else:
            self.current_frame = frame_idx

        geo_mode = (self.mode == "geo" and pose is not None and
                    cam_intrinsics is not None and georef is not None)

        # Compute lat/lon coordinates if in geo mode
        c_coords = []
        for c in candidates:
            if geo_mode:
                coords = pixel_to_latlon(c.cx, c.cy, pose, cam_intrinsics, georef)
                c_coords.append(coords)
            else:
                c_coords.append(None)

        matched_track_ids = set()
        for i, c in enumerate(candidates):
            best_track = None
            best_dist = self.max_dist_geo_m if geo_mode else self.max_dist_px

            for t in self.tracks:
                if t.track_id in matched_track_ids or t.frame_idx == frame_idx:
                    continue
                if (frame_idx - t.frame_idx) > self.max_misses:
                    continue

                if geo_mode and c_coords[i] is not None and t.lat is not None and t.lon is not None:
                    from arcticlib.geo import distance_m
                    d = distance_m(c_coords[i][0], c_coords[i][1], t.lat, t.lon)
                else:
                    d = math.hypot(c.cx - t.cx, c.cy - t.cy)

                if d < best_dist:
                    best_dist = d
                    best_track = t

            if best_track is not None:
                matched_track_ids.add(best_track.track_id)
                c_lat, c_lon = c_coords[i] if c_coords[i] is not None else (None, None)
                best_track.update_pos(c.cx, c.cy, c.w, c.h, frame_idx,
                                      t_sim=t_sim, lat=c_lat, lon=c_lon)
                if best_track.hits >= self.min_hits:
                    best_track.confirmed = True

                c.track_id = best_track.track_id
                c.hits = best_track.hits
                c.is_confirmed = best_track.confirmed
            else:
                c_lat, c_lon = c_coords[i] if c_coords[i] is not None else (None, None)
                new_track = Tracklet(self.next_id, c.cx, c.cy, c.w, c.h,
                                     frame_idx=frame_idx, t_sim=t_sim, lat=c_lat, lon=c_lon)
                matched_track_ids.add(new_track.track_id)
                self.next_id += 1
                self.tracks.append(new_track)
                c.track_id = new_track.track_id
                c.hits = 1
                c.is_confirmed = False

        # Prune old tracks exceeding max_misses
        self.tracks = [t for t in self.tracks if (frame_idx - t.frame_idx) <= self.max_misses]

        # Sort candidates: confirmed first, then by score
        candidates.sort(key=lambda c: (getattr(c, "is_confirmed", False), c.score), reverse=True)
        return candidates


class VerifiedDetector:
    """Combines Step 1 (Color Anomaly), Step 2 (CNN Verifier), and Step 3 (Temporal Tracking)."""

    def __init__(self,
                 model_path: Optional[str] = "models/patch_verifier.pt",
                 min_color_score: float = 0.30,
                 min_verify_prob: float = 0.50,
                 patch_size: int = 48,
                 enable_temporal: bool = True,
                 min_hits: int = 8,
                 max_misses: int = 4,
                 track_mode: str = "pixel",
                 device: Optional[str] = None):
        self.color_detector = ColorAnomalyDetector(
            min_area=2, max_area=800, min_score=min_color_score,
            patch_size=patch_size
        )
        self.min_verify_prob = min_verify_prob
        self.patch_size = patch_size

        # Device selection
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[PatchVerifierCNN] = None
        if model_path and os.path.exists(model_path):
            self.model = PatchVerifierCNN(patch_size=patch_size, num_classes=2)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
            print(f"Loaded patch verifier model from {model_path} onto {self.device}")
        else:
            print("Warning: No model loaded; running in Stage 1 only mode.")

        # Stage 3 Temporal Filter
        self.temporal_filter: Optional[TemporalTrackFilter] = None
        if enable_temporal:
            self.temporal_filter = TemporalTrackFilter(
                min_hits=min_hits, max_misses=max_misses, mode=track_mode
            )

    def detect(self,
               bgr: np.ndarray,
               frame_idx: Optional[int] = None,
               t_sim: Optional[float] = None,
               pose=None,
               cam_intrinsics: Optional[dict] = None,
               georef=None) -> list[Candidate]:
        """Run three-stage detection on a BGR image."""
        # Stage 1: Color anomaly candidates
        candidates = self.color_detector.detect(bgr, extract_patches=True)
        if not candidates:
            return candidates

        # Stage 2: Learned patch verification
        if self.model is not None:
            valid_indices = []
            batch_tensors = []
            for i, c in enumerate(candidates):
                if c.patch is not None and c.patch.shape == (self.patch_size, self.patch_size, 3):
                    tensor = torch.from_numpy(c.patch.transpose(2, 0, 1)).float() / 255.0
                    tensor = (tensor - 0.5) / 0.5
                    batch_tensors.append(tensor)
                    valid_indices.append(i)

            if batch_tensors:
                batch = torch.stack(batch_tensors).to(self.device)
                with torch.no_grad():
                    probs = self.model.predict_prob(batch).cpu().numpy()

                verified: list[Candidate] = []
                for idx, prob in zip(valid_indices, probs):
                    c = candidates[idx]
                    p = float(prob)
                    if p >= self.min_verify_prob:
                        c.score = float(0.40 * c.score + 0.60 * p)
                        verified.append(c)
                candidates = verified

        # Stage 3: Temporal Persistence Tracking
        if self.temporal_filter is not None:
            candidates = self.temporal_filter.update(
                candidates,
                frame_idx=frame_idx,
                t_sim=t_sim,
                pose=pose,
                cam_intrinsics=cam_intrinsics,
                georef=georef
            )

        return candidates

    def draw_detections(self, bgr: np.ndarray, candidates: list[Candidate],
                        gt_pt: Optional[tuple[float, float]] = None) -> np.ndarray:
        vis = bgr.copy()

        # Ground truth
        if gt_pt is not None:
            gx, gy = int(round(gt_pt[0])), int(round(gt_pt[1]))
            cv2.circle(vis, (gx, gy), 14, (0, 255, 0), 2)
            cv2.putText(vis, "TRUE BOAT", (gx + 16, gy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Candidates (Differentiate tentative vs confirmed)
        for idx, c in enumerate(candidates):
            is_conf = getattr(c, "is_confirmed", False)
            hits = getattr(c, "hits", 1)
            tid = getattr(c, "track_id", idx + 1)
            min_h = self.temporal_filter.min_hits if self.temporal_filter else 8

            if is_conf:
                # Confirmed vessel target: Vibrant green, thick box + crosshair
                color = (0, 255, 0)
                thickness = 2
                cv2.rectangle(vis, (c.x, c.y), (c.x + c.w, c.y + c.h), color, thickness)
                cx_i, cy_i = int(round(c.cx)), int(round(c.cy))
                cv2.drawMarker(vis, (cx_i, cy_i), color, cv2.MARKER_CROSS, markerSize=18, thickness=2)
                label = f"CONFIRMED BOAT #{tid} [{hits} hits]"
            else:
                # Tentative candidate: Orange/yellow, thinner box
                color = (0, 165, 255)
                thickness = 1
                cv2.rectangle(vis, (c.x, c.y), (c.x + c.w, c.y + c.h), color, thickness)
                label = f"CANDIDATE #{tid} [{hits}/{min_h}]"

            cv2.putText(vis, label, (c.x, max(14, c.y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2 if is_conf else 1)

        return vis


# --------------------------------------------------------------------------- #
# Geolocation Helper: Camera Ray -> Flat Earth Intersection
# --------------------------------------------------------------------------- #
def pixel_to_latlon(u: float, v: float, pose, camera_intrinsics: dict,
                    georef=None) -> Optional[Tuple[float, float]]:
    """Estimate lat/lon with the trig geolocator (:mod:`arcticlib.geolocate`).

    Casts a ray through the pixel, rotates it into NED using the airframe
    attitude and the asset's true camera mount (from the sensor SDF), intersects
    flat ground and converts to lat/lon. ``georef`` is accepted for backwards
    compatibility and ignored (the geodesic is global).
    """
    if pose is None:
        return None
    from arcticlib.geolocate import GeoConfig
    asset = getattr(pose, "asset", "fixed-wing")
    est = GeoConfig().locate(u, v, pose, asset, camera_intrinsics)
    if est is None:
        return None
    return est.lat, est.lon


def main() -> int:
    parser = argparse.ArgumentParser(description="Three-stage boat detector (Color + CNN + Temporal).")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--dataset", help="Recorded dataset directory")
    parser.add_argument("--asset", default="quadcopter", help="Asset name")
    parser.add_argument("--model", default="models/patch_verifier.pt", help="Patch verifier model weights")
    parser.add_argument("--min-hits", type=int, default=8, help="Minimum hits to confirm track (default: 8)")
    parser.add_argument("--max-misses", type=int, default=4, help="Max missed frames before track expires (default: 4)")
    parser.add_argument("--track-mode", choices=["pixel", "geo"], default="pixel", help="Tracking mode (default: pixel)")
    parser.add_argument("--no-temporal", action="store_true", help="Disable Stage 3 temporal tracking")
    parser.add_argument("--live", action="store_true", help="Run live on camera feed")
    parser.add_argument("--hz", type=float, default=2.0, help="Live frame rate")
    parser.add_argument("--out-dir", default="verified_output", help="Output directory")
    parser.add_argument("--publish-tracks", action="store_true", help="Publish tracks to /api/tracks")
    args = parser.parse_args()

    detector = VerifiedDetector(
        model_path=args.model,
        enable_temporal=not args.no_temporal,
        min_hits=args.min_hits,
        max_misses=args.max_misses,
        track_mode=args.track_mode
    )

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: Cannot read {args.image}")
            return 1
        dets = detector.detect(img)
        print(f"Detected {len(dets)} verified boat candidates:")
        for i, d in enumerate(dets):
            print(f"  #{i+1}: score={d.score:.3f}, center=({d.cx:.1f}, {d.cy:.1f}), bbox={d.bbox()}")
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            vis = detector.draw_detections(img, dets)
            out_path = os.path.join(args.out_dir, f"verified_{os.path.basename(args.image)}")
            cv2.imwrite(out_path, vis)
            print(f"Saved visualization to {out_path}")
        return 0

    if args.dataset:
        from tools.detect_color import evaluate_dataset
        # Evaluate using verified detector's detect method
        class Adapter:
            def detect(self, img, extract_patches=True):
                return detector.detect(img)
            def draw_detections(self, img, candidates, gt_pt=None):
                return detector.draw_detections(img, candidates, gt_pt=gt_pt)
        evaluate_dataset(args.dataset, args.asset, Adapter(), out_dir=args.out_dir)
        return 0

    if args.live:
        from arcticlib.config import load_config
        from arcticlib.fleet import Fleet
        from arcticlib.geo import Georef
        from arcticlib.tracks import TrackClient

        cfg = load_config()
        fleet = Fleet.from_config(cfg, connect=True)
        georef = Georef(cfg.origin_lat, cfg.origin_lon, ps_centre_x=cfg.ps_centre_x, ps_centre_y=cfg.ps_centre_y)
        track_client = TrackClient(cfg.url(cfg.tracks_port)) if args.publish_tracks else None

        if args.asset not in fleet.cams:
            print(f"Error: Asset '{args.asset}' has no camera.")
            return 1

        cam = fleet.cams[args.asset]
        cam_intrinsics = cfg.assets[args.asset].camera.intrinsics()
        period = 1.0 / max(args.hz, 0.1)
        print(f"Running live verified detector on {args.asset} at {args.hz} Hz...")
        frame_idx = 0
        try:
            while True:
                t0 = time.monotonic()
                frame = cam.grab()
                if frame is not None:
                    frame_idx += 1
                    pose = fleet.pose_at(args.asset, frame.t_sim, clock="sim") or fleet.pose(args.asset)
                    dets = detector.detect(
                        frame.image,
                        frame_idx=frame_idx,
                        t_sim=frame.t_sim,
                        pose=pose,
                        cam_intrinsics=cam_intrinsics if pose else None,
                        georef=georef
                    )

                    confirmed_dets = [d for d in dets if getattr(d, "is_confirmed", False)]
                    status_str = f"[{time.strftime('%H:%M:%S')}] Frame #{frame_idx} | Candidates: {len(dets)} | Confirmed: {len(confirmed_dets)}"
                    if confirmed_dets:
                        best = confirmed_dets[0]
                        status_str += f" | *** CONFIRMED BOAT *** Track #{best.track_id} ({best.hits} hits, score={best.score:.2f}) at ({best.cx:.1f}, {best.cy:.1f})"

                        # Geolocation and track posting (only for confirmed targets)
                        if track_client and pose:
                            coords = pixel_to_latlon(best.cx, best.cy, pose, cam_intrinsics, georef)
                            if coords:
                                lat, lon = coords
                                res = track_client.post_fix("Sierra One", lat, lon, frame.t_sim)
                                status_str += f" -> Track posted: ({lat:.5f}, {lon:.5f}) res={res is not None}"
                    elif dets:
                        best = dets[0]
                        status_str += f" | Tentative: Track #{getattr(best, 'track_id', '?')} ({getattr(best, 'hits', 1)}/{args.min_hits} hits)"

                    print(status_str)
                    if args.out_dir and dets:
                        os.makedirs(args.out_dir, exist_ok=True)
                        vis = detector.draw_detections(frame.image, dets)
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
