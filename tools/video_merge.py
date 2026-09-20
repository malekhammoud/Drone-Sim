#!/usr/bin/env python3
"""Video merge utility: combine two video feeds into a unified top-down display.

Layout:
  ┌────────────────────────────────────────────────────────┐
  │              TOP: FIXED-WING RECONNAISSANCE            │
  │                    (1280 x 720)                        │
  ├────────────────────────────────────────────────────────┤
  │    DOMINION DYNAMICS - MULTI-ASSET MARITIME MISSION    │
  ├────────────────────────────────────────────────────────┤
  │              BOTTOM: QUADCOPTER SURVEILLANCE           │
  │                    (1280 x 720)                        │
  └────────────────────────────────────────────────────────┘

Supports both parallel (simultaneous) and sequential playback modes.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("video_merge")


def _fit_frame(frame: np.ndarray | None, target_w: int, target_h: int,
               placeholder_title: str = "", placeholder_subtitle: str = "") -> np.ndarray:
    """Resize/letterbox frame to (target_w, target_h) preserving aspect ratio."""
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 18)

    if frame is None:
        # Draw placeholder card
        cv2.rectangle(canvas, (40, 40), (target_w - 40, target_h - 40), (45, 45, 45), 2)
        if placeholder_title:
            cv2.putText(canvas, placeholder_title, (70, target_h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2)
        if placeholder_subtitle:
            cv2.putText(canvas, placeholder_subtitle, (70, target_h // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (140, 140, 140), 1)
        return canvas

    h, w = frame.shape[:2]
    if w == target_w and h == target_h:
        return frame.copy()

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _create_separator(width: int, height: int = 44,
                      title: str = "DOMINION DYNAMICS - MULTI-ASSET MARITIME RECONNAISSANCE") -> np.ndarray:
    """Create a sleek separator bar between the top and bottom video panes."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:] = (26, 26, 26)  # Dark charcoal
    cv2.line(bar, (0, 0), (width, 0), (70, 70, 70), 1)
    cv2.line(bar, (0, height - 1), (width, height - 1), (70, 70, 70), 1)

    # Centered title
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (t_w, t_h), _ = cv2.getTextSize(title, font, scale, thickness)
    x = (width - t_w) // 2
    y = (height + t_h) // 2 - 2
    cv2.putText(bar, title, (x, y), font, scale, (200, 200, 200), thickness)
    return bar


def merge_videos_top_down(
    top_video_path: str,
    bottom_video_path: str,
    output_path: str,
    target_width: int = 1280,
    pane_height: int = 720,
    fps: float | None = None,
    mode: str = "parallel",
    top_label: str = "FIXED-WING RECONNAISSANCE (GLIDER)",
    bottom_label: str = "QUADCOPTER SURVEILLANCE (TOP-DOWN)",
) -> str:
    """Merge two videos into a top-down stacked video.

    Args:
        top_video_path: Path to the top video (e.g. fixed-wing patrol).
        bottom_video_path: Path to the bottom video (e.g. quadcopter follow).
        output_path: Output MP4 path.
        target_width: Width of the combined video (default 1280).
        pane_height: Height of each pane (default 720).
        fps: Target FPS. If None, uses max of the two input videos.
        mode: 'parallel' (both play together) or 'sequential' (phase 1 then phase 2).
        top_label: Title label for the top pane.
        bottom_label: Title label for the bottom pane.

    Returns:
        The output_path string.
    """
    if not os.path.exists(top_video_path):
        raise FileNotFoundError(f"Top video not found: {top_video_path}")
    if not os.path.exists(bottom_video_path):
        raise FileNotFoundError(f"Bottom video not found: {bottom_video_path}")

    cap_top = cv2.VideoCapture(top_video_path)
    cap_bottom = cv2.VideoCapture(bottom_video_path)

    fps_top = cap_top.get(cv2.CAP_PROP_FPS) or 3.0
    fps_bottom = cap_bottom.get(cv2.CAP_PROP_FPS) or 2.0
    out_fps = fps or max(fps_top, fps_bottom, 3.0)

    n_top = int(cap_top.get(cv2.CAP_PROP_FRAME_COUNT))
    n_bottom = int(cap_bottom.get(cv2.CAP_PROP_FRAME_COUNT))

    sep_h = 44
    total_h = 2 * pane_height + sep_h
    sep_bar = _create_separator(target_width, sep_h)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (target_width, total_h))

    log.info("Merging videos top-down: top=%s (%d frames), bottom=%s (%d frames) -> %s (mode=%s)",
             os.path.basename(top_video_path), n_top,
             os.path.basename(bottom_video_path), n_bottom,
             output_path, mode)

    try:
        if mode == "sequential":
            # --- PHASE 1: Top active, bottom on standby ---
            log.info("Sequential merge: rendering Phase 1 (Top)...")
            standby_bottom = _fit_frame(
                None, target_width, pane_height,
                placeholder_title="PHASE 1: FIXED-WING WIDE-AREA STRAIT SEARCH",
                placeholder_subtitle="Quadcopter standing by on pad at Fort Ross awaiting target handoff...")
            while True:
                ret_t, frame_t = cap_top.read()
                if not ret_t:
                    break
                fitted_t = _fit_frame(frame_t, target_width, pane_height)
                # Label badge on top pane
                cv2.rectangle(fitted_t, (target_width - 340, 10), (target_width - 10, 42), (20, 20, 20), -1)
                cv2.putText(fitted_t, top_label, (target_width - 330, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                stacked = np.vstack([fitted_t, sep_bar, standby_bottom])
                writer.write(stacked)

            # Rewind top for holding last frame
            cap_top.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_top - 1))
            ret_last, last_top_frame = cap_top.read()
            frozen_top = _fit_frame(last_top_frame, target_width, pane_height)
            # Add overlay tag on frozen top
            cv2.rectangle(frozen_top, (target_width - 380, 10), (target_width - 10, 68), (20, 20, 20), -1)
            cv2.putText(frozen_top, top_label, (target_width - 370, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frozen_top, "[TARGET HANDED OFF - LOITERING]", (target_width - 370, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

            # --- PHASE 2: Top loitering/frozen, bottom active ---
            log.info("Sequential merge: rendering Phase 2 (Bottom)...")
            while True:
                ret_b, frame_b = cap_bottom.read()
                if not ret_b:
                    break
                fitted_b = _fit_frame(frame_b, target_width, pane_height)
                cv2.rectangle(fitted_b, (target_width - 360, 10), (target_width - 10, 42), (20, 20, 20), -1)
                cv2.putText(fitted_b, bottom_label, (target_width - 350, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                stacked = np.vstack([frozen_top, sep_bar, fitted_b])
                writer.write(stacked)

        else:
            # --- PARALLEL MODE (Default): Both play in parallel ---
            log.info("Parallel merge: rendering both streams simultaneously...")
            last_t = None
            last_b = None
            total_steps = max(n_top, n_bottom)
            step = 0

            while step < total_steps:
                ret_t, frame_t = cap_top.read()
                ret_b, frame_b = cap_bottom.read()

                if ret_t:
                    last_t = frame_t
                if ret_b:
                    last_b = frame_b

                if not ret_t and not ret_b and last_t is None and last_b is None:
                    break

                fitted_t = _fit_frame(last_t, target_width, pane_height)
                fitted_b = _fit_frame(last_b, target_width, pane_height)

                # Top pane label
                cv2.rectangle(fitted_t, (target_width - 340, 10), (target_width - 10, 42), (20, 20, 20), -1)
                cv2.putText(fitted_t, top_label, (target_width - 330, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                if not ret_t and n_top > 0:
                    cv2.putText(fitted_t, "[PATROL COMPLETE]", (target_width - 330, 56),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)

                # Bottom pane label
                cv2.rectangle(fitted_b, (target_width - 360, 10), (target_width - 10, 42), (20, 20, 20), -1)
                cv2.putText(fitted_b, bottom_label, (target_width - 350, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                if not ret_b and n_bottom > 0:
                    cv2.putText(fitted_b, "[MISSION COMPLETE]", (target_width - 350, 56),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)

                stacked = np.vstack([fitted_t, sep_bar, fitted_b])
                writer.write(stacked)
                step += 1

    finally:
        cap_top.release()
        cap_bottom.release()
        writer.release()

    log.info("Successfully exported merged top-down video: %s", output_path)
    return output_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge two videos into a top-down stacked video.")
    ap.add_argument("--top", required=True, help="Path to top video (fixed-wing)")
    ap.add_argument("--bottom", required=True, help="Path to bottom video (quadcopter)")
    ap.add_argument("--out", default="merged_top_down.mp4", help="Output MP4 path")
    ap.add_argument("--mode", choices=["parallel", "sequential"], default="parallel",
                    help="Playback mode: parallel (default) or sequential")
    ap.add_argument("--width", type=int, default=1280, help="Output width (default 1280)")
    ap.add_argument("--pane-height", type=int, default=720, help="Height per pane (default 720)")
    ap.add_argument("--fps", type=float, default=None, help="Output FPS (default max of inputs)")
    args = ap.parse_args()

    merge_videos_top_down(
        top_video_path=args.top,
        bottom_video_path=args.bottom,
        output_path=args.out,
        target_width=args.width,
        pane_height=args.pane_height,
        fps=args.fps,
        mode=args.mode,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
