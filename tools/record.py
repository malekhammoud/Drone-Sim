#!/usr/bin/env python3
"""Dataset recorder for the CV teammate.

Grabs frames from chosen assets at N Hz and writes, next to each JPEG, a JSONL
line with the pose, camera intrinsics, sim time and — **only when
``ARCTICSIM_DEV=1``** — the target vessel's ground-truth position and an
approximate pixel box. That makes a labelled dataset for free.

    python tools/record.py --assets quadcopter fixed-wing --hz 2 --duration 60
    # -> data/2026-09-19T18-00-00/{meta.json, quadcopter/{00000.jpg,...}, ...}

Never enable ground truth for the judged run; it is cheating there.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from arcticlib.config import load_config  # noqa: E402
from arcticlib.fleet import Fleet  # noqa: E402
from arcticlib.geo import Georef  # noqa: E402

DEV = os.environ.get("ARCTICSIM_DEV") == "1"


def _camera_world(pose, georef):
    """Approximate camera world (x, y, z) for a body-fixed camera."""
    x, y = georef.latlon_to_world(pose.lat, pose.lon)
    return x, y, pose.alt_amsl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=None,
                    help="assets to record (default: all with a camera)")
    ap.add_argument("--hz", type=float, default=2.0, help="frames per second per asset")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds (0 = forever)")
    ap.add_argument("--out", default="data", help="output root")
    args = ap.parse_args()

    cfg = load_config()
    georef = Georef(cfg.origin_lat, cfg.origin_lon, ps_centre_x=cfg.ps_centre_x,
                    ps_centre_y=cfg.ps_centre_y)
    fleet = Fleet.from_config(cfg)
    fleet.wait_ready(20)

    assets = args.assets or [n for n, s in cfg.assets.items()
                             if s.camera is not None and n in fleet.cams]
    stamp = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root = os.path.join(args.out, stamp)
    os.makedirs(root, exist_ok=True)

    gt = None
    if DEV:
        from arcticlib.groundtruth import GroundTruth
        gt = GroundTruth(cfg, georef)

    handles = {}
    for name in assets:
        if name not in fleet.cams:
            print(f"  skip {name}: no camera source")
            continue
        os.makedirs(os.path.join(root, name), exist_ok=True)
        handles[name] = open(os.path.join(root, name, "sidecar.jsonl"), "w")

    meta = {
        "started": stamp, "assets": list(handles), "hz": args.hz,
        "site": {"name": cfg.site_name, "lat": cfg.origin_lat, "lon": cfg.origin_lon,
                 "convergence_deg": cfg.convergence_deg,
                 "ps_centre_x": cfg.ps_centre_x, "ps_centre_y": cfg.ps_centre_y},
        "dev_ground_truth": DEV,
        "cameras": {n: cfg.assets[n].camera.intrinsics() for n in handles},
    }
    with open(os.path.join(root, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"recording {list(handles)} at {args.hz} Hz -> {root}")
    if DEV:
        print("  (ground truth ON — dev only)")

    period = 1.0 / max(args.hz, 0.05)
    counts = {n: 0 for n in handles}
    end = time.monotonic() + args.duration if args.duration > 0 else float("inf")
    next_t = time.monotonic()
    try:
        while time.monotonic() < end:
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.05, next_t - now))
                continue
            next_t += period
            if next_t < now:
                next_t = now + period
            ship = gt.ship_pose() if gt else None
            for name in handles:
                frame = fleet.cams[name].grab()
                if frame is None:
                    continue
                idx = counts[name]
                path = os.path.join(root, name, f"{idx:05d}.jpg")
                cv2.imwrite(path, frame.image)
                pose = fleet.pose_at(name, frame.t_sim, clock="sim") or fleet.pose(name)
                row = {
                    "asset": name, "frame": os.path.relpath(path, root), "index": idx,
                    "t_sim": frame.t_sim, "t_wall": frame.t_wall,
                    "width": frame.width, "height": frame.height,
                    "camera": cfg.assets[name].camera.intrinsics(),
                    "pose": None if pose is None else {
                        "lat": pose.lat, "lon": pose.lon, "alt_rel": pose.alt_rel,
                        "alt_amsl": pose.alt_amsl, "roll": pose.roll,
                        "pitch": pose.pitch, "yaw": pose.yaw,
                        "vx": pose.vx, "vy": pose.vy, "vz": pose.vz},
                }
                if ship is not None:
                    row["groundtruth"] = ship
                    if pose is not None:
                        from arcticlib.groundtruth import project_point
                        cam_world = _camera_world(pose, georef)
                        # Vehicle yaw is a true heading; project_point wants the
                        # camera bearing from GRID north, so subtract convergence.
                        grid_yaw_deg = (pose.yaw * 57.29577951308232
                                        - cfg.convergence_deg)
                        uv = project_point(ship["world"], cam_world, grid_yaw_deg,
                                           row["camera"])
                        # A point, not a box: ground truth is a position, not an
                        # extent. Approximate (body-fixed camera, level attitude).
                        if uv is not None:
                            row["groundtruth"]["point_px_approx"] = [uv[0], uv[1]]
                handles[name].write(json.dumps(row) + "\n")
                counts[name] = idx + 1
    finally:
        for fh in handles.values():
            fh.close()
        if gt is not None:
            gt.stop()
        fleet.shutdown()

    print("wrote:", ", ".join(f"{n}={c}" for n, c in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
