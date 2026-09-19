#!/usr/bin/env python3
"""Tower calibration writer / verifier.

The pan/tilt PWM<->angle mapping is **known exactly** from the model: the sim's
ArduPilotPlugin maps a 0..1 servo command to a joint angle as
``multiplier * (offset + cmd)`` (terrain/tower.py), and the tracker's servos run
over **1100..1900 us** (SERVO1/2_MIN/MAX), so pan is 1500 us = 0 deg at
2.2222 us/deg and tilt is 1420 us = level at 10.667 us/deg. This tool verifies
the loop tracks the commanded PWM (via SERVO_OUTPUT_RAW) and writes
``calib/tower_<name>.json``.

It cannot derive ``base_yaw_deg`` (the world yaw of pan pwm 1500): the tower is a
static model, so its joint angles are not published. Set it by hand if you need
absolute aiming: aim at a landmark you can see, compare with its true bearing,
and write the difference.

    python tools/calibrate_tower.py --tower tower-1
    python tools/calibrate_tower.py --tower tower-2 --grab-frames calib/frames
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from arcticlib.config import load_config  # noqa: E402
from arcticlib.vehicle import Tower  # noqa: E402

# Effective servo range is 1100..1900 us (SERVO1/2_MIN/MAX), not 1000..2000.
PAN_SWEEP = [1100, 1300, 1500, 1700, 1900]
TILT_SWEEP = [1100, 1300, 1420, 1700, 1900]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", default="tower-1")
    ap.add_argument("--out-dir", default="calib")
    ap.add_argument("--grab-frames", default=None,
                    help="directory to save a JPEG at each sweep step")
    args = ap.parse_args()

    cfg = load_config()
    spec = cfg.assets.get(args.tower)
    if spec is None:
        print(f"unknown tower {args.tower}")
        return 1
    tower = Tower(spec, cfg)
    tower.load_calibration()
    if not tower.connected:
        tower.connect()
    if not tower.connected:
        print(f"{args.tower}: not connected")
        return 1

    tower.stop_scan()
    time.sleep(0.5)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.grab_frames:
        os.makedirs(args.grab_frames, exist_ok=True)
    # camera index for this asset comes from config
    cam = spec.camera
    failures = []

    def step(kind: str, pwm: int) -> None:
        setter = tower.set_pan_pwm if kind == "pan" else tower.set_tilt_pwm
        ch = 1 if kind == "pan" else 2
        ok = setter(pwm)
        time.sleep(1.0)
        observed = tower.servo.get(ch)
        tracked = observed == pwm
        if not (ok and tracked):
            failures.append(f"{kind} {pwm}: ack={ok} servo_out={observed}")
        print(f"  {kind:4} cmd={pwm}  ack={ok}  SERVO_OUTPUT_RAW={observed}  "
              f"{'OK' if tracked else 'MISMATCH'}")
        if args.grab_frames and cam is not None:
            import requests
            try:
                r = requests.get(cam.snapshot_url, timeout=5)
                path = os.path.join(args.grab_frames, f"{kind}_{pwm}.jpg")
                with open(path, "wb") as fh:
                    fh.write(r.content)
            except Exception as exc:
                print(f"    (frame grab failed: {exc})")

    print(f"=== {args.tower}: pan sweep ===")
    for pwm in PAN_SWEEP:
        step("pan", pwm)
    print(f"=== {args.tower}: tilt sweep ===")
    for pwm in TILT_SWEEP:
        step("tilt", pwm)
    tower.center()
    time.sleep(0.5)

    # Force the model-derived defaults (ignore any stale file) so we write the
    # fit this run verified, not a previous run's leftovers.
    calib = tower.load_calibration(
        path=os.path.join(args.out_dir, "_model_defaults.json"))
    path = os.path.join(args.out_dir, f"tower_{spec.name}.json")
    with open(path, "w") as fh:
        json.dump(calib, fh, indent=2)
    print(f"\nwrote {path}")
    print(json.dumps(calib, indent=2))
    print(f"\nservo tracking: {len(failures)} failures")
    for f in failures:
        print("  ", f)
    tower.shutdown()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
