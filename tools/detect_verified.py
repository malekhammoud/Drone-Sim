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


class VerifiedDetector:
    """Combines Step 1 (Color Anomaly) and Step 2 (CNN Verifier)."""

    def __init__(self,
                 model_path: Optional[str] = "models/patch_verifier.pt",
                 min_color_score: float = 0.30,
                 min_verify_prob: float = 0.50,
                 patch_size: int = 48,
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

    def detect(self, bgr: np.ndarray) -> list[Candidate]:
        """Run two-stage detection on a BGR image."""
        # Stage 1: Color anomaly candidates
        candidates = self.color_detector.detect(bgr, extract_patches=True)
        if not candidates or self.model is None:
            return candidates

        # Prepare batch of candidate patches
        valid_indices = []
        batch_tensors = []
        for i, c in enumerate(candidates):
            if c.patch is not None and c.patch.shape == (self.patch_size, self.patch_size, 3):
                # HWC -> CHW float32 normalized
                tensor = torch.from_numpy(c.patch.transpose(2, 0, 1)).float() / 255.0
                tensor = (tensor - 0.5) / 0.5
                batch_tensors.append(tensor)
                valid_indices.append(i)

        if not batch_tensors:
            return candidates

        batch = torch.stack(batch_tensors).to(self.device)
        with torch.no_grad():
            probs = self.model.predict_prob(batch).cpu().numpy()

        # Stage 2: Filter and rescore
        verified: list[Candidate] = []
        for idx, prob in zip(valid_indices, probs):
            c = candidates[idx]
            p = float(prob)
            if p >= self.min_verify_prob:
                # Combined score: 40% color anomaly + 60% CNN verifier confidence
                c.score = float(0.40 * c.score + 0.60 * p)
                verified.append(c)

        verified.sort(key=lambda c: c.score, reverse=True)
        return verified

    def draw_detections(self, bgr: np.ndarray, candidates: list[Candidate],
                        gt_pt: Optional[tuple[float, float]] = None) -> np.ndarray:
        vis = bgr.copy()

        # Ground truth
        if gt_pt is not None:
            gx, gy = int(round(gt_pt[0])), int(round(gt_pt[1]))
            cv2.circle(vis, (gx, gy), 12, (0, 255, 0), 2)
            cv2.putText(vis, "TRUE BOAT", (gx + 15, gy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Candidates
        for idx, c in enumerate(candidates):
            color = (0, 255, 0) if idx == 0 else (0, 165, 255)
            cv2.rectangle(vis, (c.x, c.y), (c.x + c.w, c.y + c.h), color, 2)
            label = f"BOAT #{idx+1} {c.score:.2f}"
            cv2.putText(vis, label, (c.x, max(12, c.y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        return vis


# --------------------------------------------------------------------------- #
# Geolocation Helper: Camera Ray -> Flat Earth Intersection
# --------------------------------------------------------------------------- #
def pixel_to_latlon(u: float, v: float, pose, camera_intrinsics: dict, georef) -> Optional[Tuple[float, float]]:
    """Estimate lat/lon by intersecting camera ray with sea level (z=0)."""
    if pose is None or pose.alt_amsl <= 0.5:
        return None

    # Camera ray in camera frame (+Z forward, +X right, +Y down)
    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]

    ray_c = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    ray_c /= np.linalg.norm(ray_c)

    # Attitude: roll, pitch, yaw (yaw is true heading, convert to grid yaw)
    # Camera fixed looking down/forward
    # For quadcopter with fixed camera: looks down (approx pitch -90 deg or along body)
    # If ray points down (dz < 0), intersect with z = 0
    # Approximate using flat earth distance:
    alt = pose.alt_amsl
    # Pitch down angle
    pitch = pose.pitch
    yaw = pose.yaw
    # Approximate horizontal distance:
    # theta is angle from nadir
    theta = math.atan2(np.hypot(ray_c[0], ray_c[1]), ray_c[2])
    dist_horiz = alt * math.tan(theta)

    # Bearing from vehicle
    bearing_rel = math.atan2(ray_c[0], ray_c[2])
    bearing_true = (math.degrees(yaw + bearing_rel) + 360.0) % 360.0

    from arcticlib.geo import destination
    return destination(pose.lat, pose.lon, bearing_true, dist_horiz)


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-stage boat detector.")
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--dataset", help="Recorded dataset directory")
    parser.add_argument("--asset", default="quadcopter", help="Asset name")
    parser.add_argument("--model", default="models/patch_verifier.pt", help="Patch verifier model weights")
    parser.add_argument("--live", action="store_true", help="Run live on camera feed")
    parser.add_argument("--hz", type=float, default=2.0, help="Live frame rate")
    parser.add_argument("--out-dir", default="verified_output", help="Output directory")
    parser.add_argument("--publish-tracks", action="store_true", help="Publish tracks to /api/tracks")
    args = parser.parse_args()

    detector = VerifiedDetector(model_path=args.model)

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
        period = 1.0 / max(args.hz, 0.1)
        print(f"Running live verified detector on {args.asset} at {args.hz} Hz...")
        try:
            while True:
                t0 = time.monotonic()
                frame = cam.grab()
                if frame is not None:
                    dets = detector.detect(frame.image)
                    pose = fleet.pose_at(args.asset, frame.t_sim, clock="sim") or fleet.pose(args.asset)

                    status_str = f"[{time.strftime('%H:%M:%S')}] Detections: {len(dets)}"
                    if dets:
                        best = dets[0]
                        status_str += f" | Top Boat: score={best.score:.2f} at ({best.cx:.1f}, {best.cy:.1f})"

                        # Geolocation and track posting
                        if track_client and pose:
                            cam_intrinsics = cfg.assets[args.asset].camera.intrinsics()
                            coords = pixel_to_latlon(best.cx, best.cy, pose, cam_intrinsics, georef)
                            if coords:
                                lat, lon = coords
                                res = track_client.post("Sierra One", lat, lon)
                                status_str += f" -> Track posted: ({lat:.5f}, {lon:.5f}) res={res is not None}"

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
