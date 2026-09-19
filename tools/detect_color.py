#!/usr/bin/env python3
"""Step 1: Color-anomaly candidate detector for the red vessel.

Detects red/anomalous boat pixels against dark blue water and grey-white ice
using a combination of:
1. LAB color space analysis (elevated a* channel for red/magenta).
2. HSV red hue thresholding (dual-band [0, 12] and [168, 180]).
3. Mahalanobis distance / background modeling against water and ice clusters.
4. Connected components analysis with morphological filtering.

Usage:
    # On a single image:
    python tools/detect_color.py --image path/to/frame.jpg --out-dir debug_out

    # On a dataset recorded by tools/record.py (evaluates recall against ground truth):
    python tools/detect_color.py --dataset data/2026-09-19T18-00-00 --asset quadcopter

    # Live from sim camera:
    python tools/detect_color.py --live --asset quadcopter --hz 2
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

log = logging.getLogger("detect_color")


@dataclasses.dataclass
class Candidate:
    """A detected candidate region in the image."""
    x: int
    y: int
    w: int
    h: int
    cx: float
    cy: float
    area: int
    score: float
    patch: Optional[np.ndarray] = None

    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def contains(self, px: float, py: float, margin: float = 5.0) -> bool:
        return (self.x - margin <= px <= self.x + self.w + margin and
                self.y - margin <= py <= self.y + self.h + margin)

    def dist_to(self, px: float, py: float) -> float:
        return float(np.hypot(self.cx - px, self.cy - py))


class ColorAnomalyDetector:
    """Color-anomaly candidate detector.

    Exploits the strong prior: the Arctic scene consists of dark blue/cyan water
    and grey/white ice, while the target vessel is red. Even at long range where
    subpixel blending turns a 10px boat into pink or purple, the a* channel in LAB
    space or red-difference (R - (G+B)/2) strongly separates it from the background.
    """

    def __init__(self,
                 min_area: int = 2,
                 max_area: int = 800,
                 min_score: float = 0.35,
                 patch_size: int = 48,
                 use_mahalanobis: bool = True):
        self.min_area = min_area
        self.max_area = max_area
        self.min_score = min_score
        self.patch_size = patch_size
        self.use_mahalanobis = use_mahalanobis

    def compute_anomaly_map(self, bgr: np.ndarray) -> np.ndarray:
        """Compute a normalized [0, 1] float32 anomaly map highlighting red/boat pixels."""
        # 1. LAB Space: In OpenCV LAB, L in [0, 255], a* in [0, 255] (128 is neutral, >128 is red/magenta)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        a_chan = lab[:, :, 1].astype(np.float32)
        b_chan = lab[:, :, 2].astype(np.float32)

        # Red elevation score in LAB: a* > 132 indicates red bias.
        # Water: a* ~ 120-128, b* ~ 110-126 (blue-ish)
        # Ice: a* ~ 126-130, b* ~ 124-132 (neutral)
        # Boat: a* ~ 140-200 (high red)
        lab_red = np.clip((a_chan - 130.0) / 30.0, 0.0, 1.0)

        # 2. HSV Space: Hue wrap-around for red
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        v = hsv[:, :, 2].astype(np.float32) / 255.0

        # Red hue is around 0..10 or 170..180 (out of 180)
        hue_dist_from_red = np.minimum(h, 180.0 - h)
        hue_red = np.clip(1.0 - (hue_dist_from_red / 14.0), 0.0, 1.0)
        # Saturation boost: boat is saturated red, ice is low-sat, water is low-to-mid sat
        hsv_red = hue_red * np.clip(s / 0.25, 0.0, 1.0) * np.clip(v / 0.20, 0.0, 1.0)

        # 3. Direct Color Contrast: R - max(G, B)
        b = bgr[:, :, 0].astype(np.float32)
        g = bgr[:, :, 1].astype(np.float32)
        r = bgr[:, :, 2].astype(np.float32)
        r_excess = np.clip((r - np.maximum(g, b) - 10.0) / 35.0, 0.0, 1.0)

        # 4. Optional Mahalanobis distance against background
        if self.use_mahalanobis:
            # Downsample to estimate background mean and covariance quickly
            small = cv2.resize(lab, (160, 120))
            pts = small.reshape(-1, 3).astype(np.float32)
            # Exclude extreme top 2% brightest / anomalous for robust background fit
            mean = np.mean(pts, axis=0)
            cov = np.cov(pts, rowvar=False) + np.eye(3, dtype=np.float32) * 1e-3
            try:
                inv_cov = np.linalg.inv(cov)
                # Compute distance on full image in LAB
                diff = lab.astype(np.float32) - mean
                # (H, W, 3) @ (3, 3) -> (H, W, 3)
                d_m_sq = np.sum((diff @ inv_cov) * diff, axis=2)
                d_m = np.sqrt(np.maximum(d_m_sq, 0.0))
                # Normalize Mahalanobis distance (deviations > 3 sigma are anomalous)
                mahal_score = np.clip((d_m - 2.5) / 4.0, 0.0, 1.0)
                # Only keep Mahalanobis anomaly if it leans red (a_chan > 128)
                mahal_red = mahal_score * np.clip((a_chan - 128.0) / 15.0, 0.0, 1.0)
            except Exception:
                mahal_red = np.zeros_like(lab_red)
        else:
            mahal_red = np.zeros_like(lab_red)

        # Combine cues: LAB red + HSV red + direct contrast + Mahalanobis
        combined = (0.35 * lab_red +
                    0.25 * hsv_red +
                    0.25 * r_excess +
                    0.15 * mahal_red)
        return combined.astype(np.float32)

    def detect(self, bgr: np.ndarray, extract_patches: bool = True) -> list[Candidate]:
        """Detect boat candidate bounding boxes and patches in a BGR frame."""
        h, w = bgr.shape[:2]
        score_map = self.compute_anomaly_map(bgr)

        # Binary thresholding on the anomaly score map
        binary = (score_map >= self.min_score).astype(np.uint8) * 255

        # Morphological filtering:
        # 1. Close to bridge small disconnected pixels of a boat
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        # 2. Connected components with stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            closed, connectivity=8
        )

        candidates: list[Candidate] = []
        half_p = self.patch_size // 2

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area or area > self.max_area:
                continue

            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = float(centroids[i][0]), float(centroids[i][1])

            # Aspect ratio check: boats are roughly 1:1 to 5:1, not 20:1 lines
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 7.0:
                continue

            # Confidence score: combination of max score in region and area
            mask_i = (labels[by:by+bh, bx:bx+bw] == i)
            patch_scores = score_map[by:by+bh, bx:bx+bw][mask_i]
            if len(patch_scores) == 0:
                continue
            max_s = float(np.max(patch_scores))
            mean_s = float(np.mean(patch_scores))
            score = 0.6 * max_s + 0.4 * mean_s

            # Extract square patch centered at (cx, cy)
            patch = None
            if extract_patches:
                px1 = max(0, int(round(cx)) - half_p)
                py1 = max(0, int(round(cy)) - half_p)
                px2 = min(w, px1 + self.patch_size)
                py2 = min(h, py1 + self.patch_size)
                # Adjust if against borders
                px1 = max(0, px2 - self.patch_size)
                py1 = max(0, py2 - self.patch_size)
                cropped = bgr[py1:py2, px1:px2]
                if cropped.shape[0] == self.patch_size and cropped.shape[1] == self.patch_size:
                    patch = cropped.copy()
                else:
                    patch = cv2.resize(cropped, (self.patch_size, self.patch_size))

            candidates.append(Candidate(
                x=bx, y=by, w=bw, h=bh,
                cx=cx, cy=cy, area=area,
                score=score, patch=patch
            ))

        # Sort candidates descending by confidence score
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def draw_detections(self, bgr: np.ndarray, candidates: list[Candidate],
                        gt_pt: Optional[tuple[float, float]] = None) -> np.ndarray:
        """Annotate frame with detected candidate boxes and optional ground truth."""
        vis = bgr.copy()

        # Draw ground truth if available (Green circle)
        if gt_pt is not None:
            gx, gy = int(round(gt_pt[0])), int(round(gt_pt[1]))
            cv2.circle(vis, (gx, gy), 12, (0, 255, 0), 2)
            cv2.putText(vis, "GROUND TRUTH", (gx + 15, gy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw candidates (Yellow/Red boxes)
        for idx, c in enumerate(candidates):
            # Highlight top candidates in Bright Red/Orange, others in Yellow
            color = (0, 0, 255) if idx < 3 else (0, 255, 255)
            cv2.rectangle(vis, (c.x, c.y), (c.x + c.w, c.y + c.h), color, 2)
            label = f"#{idx+1} s={c.score:.2f} a={c.area}"
            cv2.putText(vis, label, (c.x, max(12, c.y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return vis


def evaluate_dataset(dataset_dir: str, asset: str, detector: ColorAnomalyDetector,
                     out_dir: Optional[str] = None, max_frames: int = 0) -> dict:
    """Evaluate detector on a dataset recorded by tools/record.py."""
    asset_dir = os.path.join(dataset_dir, asset)
    sidecar_path = os.path.join(asset_dir, "sidecar.jsonl")
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(f"Sidecar not found: {sidecar_path}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    frames_evaluated = 0
    frames_with_gt = 0
    gt_detected_count = 0
    total_candidates = 0
    gt_distances: list[float] = []

    with open(sidecar_path) as f:
        for line in f:
            if max_frames > 0 and frames_evaluated >= max_frames:
                break
            data = json.loads(line)
            frame_rel = data.get("frame")
            frame_path = os.path.join(dataset_dir, frame_rel) if frame_rel else None
            if not frame_path or not os.path.exists(frame_path):
                continue

            img = cv2.imread(frame_path)
            if img is None:
                continue

            candidates = detector.detect(img)
            total_candidates += len(candidates)
            frames_evaluated += 1

            # Check ground truth
            gt_data = data.get("groundtruth", {})
            gt_pt = gt_data.get("point_px_approx")  # [u, v]
            hit = False
            min_dist = float("inf")

            if gt_pt is not None:
                u, v = float(gt_pt[0]), float(gt_pt[1])
                # Check if GT is within frame boundaries
                h, w = img.shape[:2]
                if 0 <= u < w and 0 <= v < h:
                    frames_with_gt += 1
                    for c in candidates:
                        d = c.dist_to(u, v)
                        if d < min_dist:
                            min_dist = d
                        # Within 20 px or inside bbox + margin counts as a hit
                        if d <= 25.0 or c.contains(u, v, margin=8.0):
                            hit = True

                    if hit:
                        gt_detected_count += 1
                    if min_dist < float("inf"):
                        gt_distances.append(min_dist)

            if out_dir:
                vis = detector.draw_detections(img, candidates, gt_pt=(u, v) if (gt_pt and 0 <= u < w and 0 <= v < h) else None)
                vis_path = os.path.join(out_dir, f"det_{os.path.basename(frame_path)}")
                cv2.imwrite(vis_path, vis)

    recall = (gt_detected_count / frames_with_gt) if frames_with_gt > 0 else 0.0
    avg_cands = (total_candidates / frames_evaluated) if frames_evaluated > 0 else 0.0
    median_dist = float(np.median(gt_distances)) if gt_distances else 0.0

    metrics = {
        "frames_evaluated": frames_evaluated,
        "frames_with_gt_in_view": frames_with_gt,
        "gt_detected": gt_detected_count,
        "recall": recall,
        "total_candidates": total_candidates,
        "avg_candidates_per_frame": avg_cands,
        "median_gt_distance_px": median_dist
    }

    print("\n--- Evaluation Results ---")
    print(f"Frames evaluated       : {frames_evaluated}")
    print(f"Frames with GT in view : {frames_with_gt}")
    print(f"GT Detected (Recall)   : {gt_detected_count}/{frames_with_gt} ({recall*100:.1f}%)")
    print(f"Avg candidates/frame   : {avg_cands:.2f}")
    if gt_distances:
        print(f"Median dist to GT (px) : {median_dist:.1f}")
    return metrics


def run_live(asset: str, detector: ColorAnomalyDetector, hz: float = 2.0,
             out_dir: Optional[str] = None):
    """Run detector on live camera snapshots from the sim."""
    from arcticlib.config import load_config
    from arcticlib.fleet import Fleet

    cfg = load_config()
    fleet = Fleet.from_config(cfg, connect=True)
    if asset not in fleet.cams:
        print(f"Error: Asset '{asset}' has no camera.")
        return

    print(f"Listening to live camera feed from '{asset}' at {hz} Hz... Press Ctrl+C to stop.")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    period = 1.0 / max(hz, 0.1)
    cam = fleet.cams[asset]
    idx = 0

    try:
        while True:
            t0 = time.monotonic()
            frame = cam.grab()
            if frame is not None:
                candidates = detector.detect(frame.image)
                print(f"[{idx:04d}] t_sim={frame.t_sim:.1f}s candidates={len(candidates)}" +
                      (f" top_score={candidates[0].score:.2f} at ({candidates[0].cx:.1f}, {candidates[0].cy:.1f})" if candidates else ""))

                if out_dir:
                    vis = detector.draw_detections(frame.image, candidates)
                    cv2.imwrite(os.path.join(out_dir, f"live_{idx:04d}.jpg"), vis)
                idx += 1

            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        fleet.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 1: Color-anomaly candidate detector.")
    parser.add_argument("--image", help="Path to a single image file")
    parser.add_argument("--dataset", help="Path to a dataset directory recorded by tools/record.py")
    parser.add_argument("--asset", default="quadcopter", help="Asset name (quadcopter, fixed-wing, etc.)")
    parser.add_argument("--live", action="store_true", help="Run live on camera feed")
    parser.add_argument("--hz", type=float, default=2.0, help="Live frame rate")
    parser.add_argument("--out-dir", default="detection_output", help="Directory to save annotated images")
    parser.add_argument("--min-score", type=float, default=0.35, help="Minimum anomaly score [0, 1]")
    parser.add_argument("--min-area", type=int, default=2, help="Minimum candidate area in pixels")
    parser.add_argument("--max-area", type=int, default=800, help="Maximum candidate area in pixels")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    args = parser.parse_args()

    detector = ColorAnomalyDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        min_score=args.min_score
    )

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: Could not read image {args.image}")
            return 1
        candidates = detector.detect(img)
        print(f"Found {len(candidates)} candidates in {args.image}:")
        for i, c in enumerate(candidates):
            print(f"  #{i+1}: score={c.score:.3f}, pos=({c.cx:.1f}, {c.cy:.1f}), bbox={c.bbox()}, area={c.area}")
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            vis = detector.draw_detections(img, candidates)
            out_path = os.path.join(args.out_dir, f"det_{os.path.basename(args.image)}")
            cv2.imwrite(out_path, vis)
            print(f"Saved annotated image to {out_path}")
        return 0

    if args.dataset:
        evaluate_dataset(args.dataset, args.asset, detector,
                         out_dir=args.out_dir, max_frames=args.max_frames)
        return 0

    if args.live:
        run_live(args.asset, detector, hz=args.hz, out_dir=args.out_dir)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
