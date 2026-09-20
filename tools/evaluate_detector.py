#!/usr/bin/env python3
"""Comprehensive diagnostic evaluation of the 2-stage boat detector.

Evaluates Stage 1 (Color Anomaly) and Stage 2 (Patch Verifier CNN) across
all recorded flights with ground truth to determine:
1. Detection Recall & Precision.
2. False Positive rate (FPs per 100 frames).
3. Failure mode analysis (missed boats vs false alarms).
4. Data adequacy assessment: Does the model need more data, and what kind?
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from tools.detect_color import ColorAnomalyDetector
from tools.detect_verified import VerifiedDetector


def evaluate_run(
    run_dir: str,
    detector: VerifiedDetector,
    dist_threshold_px: float = 25.0,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate detector on a single recorded flight run."""
    sidecar_path = os.path.join(run_dir, "sidecar.jsonl")
    if not os.path.exists(sidecar_path):
        return {}

    total_frames = 0
    gt_in_view_frames = 0

    # Stage 1 metrics
    s1_tp = 0
    s1_fn = 0
    s1_fp = 0

    # Stage 2 metrics
    s2_tp = 0
    s2_fn = 0
    s2_fp = 0

    # Detailed logs
    false_negatives = []
    false_positives = []
    detection_records = []

    with open(sidecar_path) as f:
        lines = f.readlines()

    if max_frames:
        lines = lines[:max_frames]

    for idx, line in enumerate(lines):
        data = json.loads(line)
        frame_rel = data.get("frame")
        if not frame_rel:
            continue
        frame_path = os.path.join(run_dir, frame_rel)
        if not os.path.exists(frame_path):
            continue

        img = cv2.imread(frame_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        total_frames += 1

        gt = data.get("groundtruth", {})
        gt_pt = gt.get("point_px_approx")
        gt_in_frame = False
        gt_u, gt_v = None, None

        if gt_pt is not None:
            gt_u, gt_v = float(gt_pt[0]), float(gt_pt[1])
            if 0 <= gt_u < w and 0 <= gt_v < h:
                gt_in_frame = True
                gt_in_view_frames += 1

        # Stage 1: Color anomaly candidates
        s1_candidates = detector.color_detector.detect(img, extract_patches=True)
        s1_hit = False
        s1_fps_in_frame = 0

        for c in s1_candidates:
            if gt_in_frame:
                dist = c.dist_to(gt_u, gt_v)
                if dist <= dist_threshold_px and not s1_hit:
                    s1_hit = True
                elif dist > dist_threshold_px * 1.5:
                    s1_fps_in_frame += 1
            else:
                s1_fps_in_frame += 1

        if gt_in_frame:
            if s1_hit:
                s1_tp += 1
            else:
                s1_fn += 1
        s1_fp += s1_fps_in_frame

        # Stage 2: Verified detections
        s2_candidates = detector.detect(img)
        s2_hit = False
        s2_fps_in_frame = 0

        for c in s2_candidates:
            if gt_in_frame:
                dist = c.dist_to(gt_u, gt_v)
                if dist <= dist_threshold_px and not s2_hit:
                    s2_hit = True
                elif dist > dist_threshold_px * 1.5:
                    s2_fps_in_frame += 1
            else:
                s2_fps_in_frame += 1

        if gt_in_frame:
            if s2_hit:
                s2_tp += 1
            else:
                s2_fn += 1
                false_negatives.append({
                    "frame_idx": idx,
                    "frame_path": frame_path,
                    "gt_pixel": [gt_u, gt_v],
                    "s1_candidates_count": len(s1_candidates),
                    "s1_hit": s1_hit,
                    "pose": data.get("pose", {}),
                })
        else:
            if len(s2_candidates) > 0:
                false_positives.append({
                    "frame_idx": idx,
                    "frame_path": frame_path,
                    "candidates": [{"cx": c.cx, "cy": c.cy, "score": c.score} for c in s2_candidates],
                    "pose": data.get("pose", {}),
                })

        s2_fp += s2_fps_in_frame

    s1_prec = s1_tp / (s1_tp + s1_fp) if (s1_tp + s1_fp) > 0 else 0.0
    s1_rec = s1_tp / (s1_tp + s1_fn) if (s1_tp + s1_fn) > 0 else 0.0
    s1_f1 = (2 * s1_prec * s1_rec) / (s1_prec + s1_rec) if (s1_prec + s1_rec) > 0 else 0.0

    s2_prec = s2_tp / (s2_tp + s2_fp) if (s2_tp + s2_fp) > 0 else 0.0
    s2_rec = s2_tp / (s2_tp + s2_fn) if (s2_tp + s2_fn) > 0 else 0.0
    s2_f1 = (2 * s2_prec * s2_rec) / (s2_prec + s2_rec) if (s2_prec + s2_rec) > 0 else 0.0

    return {
        "run_dir": run_dir,
        "total_frames": total_frames,
        "gt_in_view_frames": gt_in_view_frames,
        "stage1": {
            "tp": s1_tp, "fn": s1_fn, "fp": s1_fp,
            "precision": s1_prec, "recall": s1_rec, "f1": s1_f1,
            "fp_per_frame": s1_fp / total_frames if total_frames > 0 else 0.0,
        },
        "stage2": {
            "tp": s2_tp, "fn": s2_fn, "fp": s2_fp,
            "precision": s2_prec, "recall": s2_rec, "f1": s2_f1,
            "fp_per_frame": s2_fp / total_frames if total_frames > 0 else 0.0,
        },
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate 2-stage boat detector.")
    parser.add_argument("--runs", nargs="*", default=None, help="Specific run directories to evaluate")
    parser.add_argument("--model", default="models/patch_verifier.pt", help="Path to patch verifier weights")
    parser.add_argument("--min-prob", type=float, default=0.50, help="Minimum CNN verification probability")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per run to evaluate")
    args = parser.parse_args()

    run_dirs = args.runs
    if not run_dirs:
        run_dirs = sorted(glob.glob("patrol_run/*"))
        run_dirs = [d for d in run_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, "sidecar.jsonl"))]

    if not run_dirs:
        print("No recorded runs found in patrol_run/.")
        return 1

    detector = VerifiedDetector(model_path=args.model, min_verify_prob=args.min_prob)

    print(f"\n=======================================================")
    print(f"EVALUATING DETECTOR (Model: {args.model}, min_prob={args.min_prob})")
    print(f"Found {len(run_dirs)} runs to evaluate:")
    for d in run_dirs:
        print(f"  - {d}")
    print(f"=======================================================\n")

    total_frames_all = 0
    gt_frames_all = 0
    s1_tp_all, s1_fn_all, s1_fp_all = 0, 0, 0
    s2_tp_all, s2_fn_all, s2_fp_all = 0, 0, 0
    all_fn = []
    all_fp = []

    for r in run_dirs:
        res = evaluate_run(r, detector, max_frames=args.max_frames)
        if not res:
            continue

        tf = res["total_frames"]
        gtf = res["gt_in_view_frames"]
        s1 = res["stage1"]
        s2 = res["stage2"]

        total_frames_all += tf
        gt_frames_all += gtf
        s1_tp_all += s1["tp"]
        s1_fn_all += s1["fn"]
        s1_fp_all += s1["fp"]
        s2_tp_all += s2["tp"]
        s2_fn_all += s2["fn"]
        s2_fp_all += s2["fp"]

        all_fn.extend(res["false_negatives"])
        all_fp.extend(res["false_positives"])

        print(f"Run: {os.path.basename(r)}")
        print(f"  Frames: {tf} | Boat in FOV: {gtf} ({gtf/tf*100:.1f}%)")
        print(f"  Stage 1 (Color Anomaly):")
        print(f"    Recall: {s1['recall']*100:.1f}% ({s1['tp']}/{gtf}) | Prec: {s1['precision']*100:.1f}% | FP/frame: {s1['fp_per_frame']:.2f} ({s1['fp']} total)")
        print(f"  Stage 2 (CNN Verified):")
        print(f"    Recall: {s2['recall']*100:.1f}% ({s2['tp']}/{gtf}) | Prec: {s2['precision']*100:.1f}% | FP/frame: {s2['fp_per_frame']:.3f} ({s2['fp']} total)")
        print(f"    False Negatives: {len(res['false_negatives'])}, False Positives: {len(res['false_positives'])}\n")

    # Overall Summary
    print(f"=======================================================")
    print(f"OVERALL SUMMARY ACROSS ALL RUNS")
    print(f"=======================================================")
    print(f"Total Frames Analyzed: {total_frames_all}")
    print(f"Ground Truth Boat Frames: {gt_frames_all}")

    s1_overall_prec = s1_tp_all / (s1_tp_all + s1_fp_all) if (s1_tp_all + s1_fp_all) > 0 else 0.0
    s1_overall_rec = s1_tp_all / (s1_tp_all + s1_fn_all) if (s1_tp_all + s1_fn_all) > 0 else 0.0

    s2_overall_prec = s2_tp_all / (s2_tp_all + s2_fp_all) if (s2_tp_all + s2_fp_all) > 0 else 0.0
    s2_overall_rec = s2_tp_all / (s2_tp_all + s2_fn_all) if (s2_tp_all + s2_fn_all) > 0 else 0.0

    print(f"Stage 1 (Color Anomaly):")
    print(f"  Recall:    {s1_overall_rec*100:.2f}% ({s1_tp_all}/{gt_frames_all})")
    print(f"  Precision: {s1_overall_prec*100:.2f}%")
    print(f"  Total FPs: {s1_fp_all} ({s1_fp_all/total_frames_all:.2f} per frame)")

    print(f"Stage 2 (CNN Verified):")
    print(f"  Recall:    {s2_overall_rec*100:.2f}% ({s2_tp_all}/{gt_frames_all})")
    print(f"  Precision: {s2_overall_prec*100:.2f}%")
    print(f"  Total FPs: {s2_fp_all} ({s2_fp_all/total_frames_all:.4f} per frame)")
    print(f"  FP Reduction: {(1.0 - s2_fp_all / max(1, s1_fp_all))*100:.1f}% reduction in false positives!")

    print(f"\nDiagnostic Findings:")
    print(f"  Total Missed Boat Frames (FN): {len(all_fn)}")
    print(f"  Total False Alarm Frames (FP): {len(all_fp)}")

    if all_fn:
        print("\n  Sample False Negatives (Missed Boat):")
        for fn in all_fn[:5]:
            print(f"    - Frame {fn['frame_idx']} ({os.path.basename(fn['frame_path'])}): GT px {fn['gt_pixel']}, S1 candidates: {fn['s1_candidates_count']}, S1 hit: {fn['s1_hit']}")

    if all_fp:
        print("\n  Sample False Positives (False Alarms):")
        for fp in all_fp[:5]:
            print(f"    - Frame {fp['frame_idx']} ({os.path.basename(fp['frame_path'])}): {fp['candidates']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
