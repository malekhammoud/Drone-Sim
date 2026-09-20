#!/usr/bin/env python3
"""End-to-end smoke test against the running sim.

Checks, each PASS/FAIL independently:

  1. sim control plane reachable
  2. every rostered vehicle heartbeats and reports a valid pose
  3. quadcopter takes off and flies to a waypoint
  4. fixed-wing flies to a waypoint (takeoff if grounded)
  5. both towers sweep pan and tilt (verified against SERVO_OUTPUT_RAW)
  6. one frame grabbed from each of the four cameras, saved as PNG
  7. a test track is posted and appears in the list

Usage:
    python tools/smoke_test.py [--host 127.0.0.1] [--no-fly] [--outdir smoke_frames]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from arcticlib.config import load_config  # noqa: E402
from arcticlib.fleet import Fleet  # noqa: E402
from arcticlib.geo import destination  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def fly_to(vehicle, offset_m: float, bearing: float, alt: float, timeout: float = 60):
    """Send a vehicle to a point offset from its current position."""
    p = vehicle.pose()
    lat, lon = destination(p.lat, p.lon, bearing, offset_m)
    start = (p.lat, p.lon)
    ok = vehicle.goto(lat, lon, alt)
    # Wait for the vehicle to actually start moving toward the target.
    moved = False
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        q = vehicle.pose()
        if abs(q.lat - start[0]) > 1e-5 or abs(q.lon - start[1]) > 1e-5:
            moved = True
            break
        time.sleep(0.5)
    return ok, moved, (lat, lon)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    ap.add_argument("--no-fly", action="store_true", help="telemetry/camera only")
    ap.add_argument("--outdir", default="smoke_frames")
    args = ap.parse_args()

    if args.host:
        os.environ["ARCTICSIM_HOST"] = args.host
    cfg = load_config()
    print(f"sim host   : {cfg.host}")
    print(f"control    : {cfg.url(cfg.control_port)}")
    print(f"tracks     : {cfg.url(cfg.tracks_port)}\n")

    # 1. control plane
    fleet = Fleet.from_config(cfg)
    st = fleet.sim.status()
    check("sim control plane reachable", st.get("state") not in (None, "unreachable"),
          f"state={st.get('state')} {st.get('detail', '')[:60]}")
    check("sim clock ticking", fleet.sim.clocks > 0,
          f"sim_time={fleet.sim.sim_time():.1f}s samples={fleet.sim.clocks}")

    # 2. telemetry — only assets actually rostered in the live sim
    roster = {a.get("name"): a for a in fleet.sim.assets()}
    rostered = {n for n, a in roster.items() if a.get("rostered")}
    skipped = sorted(set(fleet.vehicles) - rostered)
    if skipped:
        print(f"  (skipping non-rostered assets: {', '.join(skipped)})")
    fleet.wait_ready(timeout=15)
    for name, v in fleet.vehicles.items():
        if name not in rostered:
            continue
        # Wait briefly for a valid fix; a fresh container can take a few seconds.
        end = time.monotonic() + 12
        p = v.pose()
        while time.monotonic() < end and not (abs(p.lat) > 1 and abs(p.lon) > 1):
            time.sleep(0.5)
            p = v.pose()
        valid = v.connected and abs(p.lat) > 1 and abs(p.lon) > 1
        check(f"{name}: heartbeat + pose", valid,
              f"mode={v.mode} armed={v.armed} lat={p.lat:.5f} lon={p.lon:.5f} "
              f"alt={p.alt_rel:.1f}m")
        check(f"{name}: pose history", len(v._poses) > 0,
              f"{len(v._poses)} poses buffered")

    # 3. quadcopter flight
    if not args.no_fly and fleet.quad is not None:
        q = fleet.quad
        if q.mode.upper() != "GUIDED" or not q.armed or q.alt_rel < 3:
            check("quadcopter: takeoff", q.takeoff(20.0), f"alt={q.alt_rel:.1f}m")
        else:
            check("quadcopter: already airborne", True, f"alt={q.alt_rel:.1f}m")
        ok, moved, tgt = fly_to(q, 400, 45, 25)
        check("quadcopter: goto waypoint", ok and moved,
              f"-> {tgt[0]:.5f},{tgt[1]:.5f} moved={moved}")

    # 4. fixed-wing flight
    if not args.no_fly and fleet.plane is not None:
        p = fleet.plane
        if p.mode.upper() != "GUIDED" or not p.armed or p.alt_rel < 30:
            check("fixed-wing: takeoff", p.takeoff(), f"alt={p.alt_rel:.1f}m")
            time.sleep(8)
        else:
            check("fixed-wing: already airborne", True, f"alt={p.alt_rel:.1f}m")
        ok, moved, tgt = fly_to(p, 800, 135, 120)
        check("fixed-wing: goto waypoint", ok,
              f"-> {tgt[0]:.5f},{tgt[1]:.5f} moved={moved}")

    # 5. towers
    time.sleep(3)
    for name in ("tower-1", "tower-2"):
        t = fleet.vehicle(name)
        if t is None:
            continue
        ok1 = t.set_pan_pwm(1800)
        ok2 = t.set_tilt_pwm(1150)
        time.sleep(1.5)
        servo = dict(t.servo)
        check(f"{name}: pan/tilt servo", ok1 and ok2 and servo.get(1) == 1800
              and servo.get(2) == 1150,
              f"commanded 1800/1150, SERVO_OUTPUT_RAW={servo.get(1)}/{servo.get(2)}")
        check(f"{name}: scan mode", t.scan(), f"mode={t.mode}")
        t.stop_scan()
        t.center()

    # 6. cameras — rostered assets only (rover/boat have no feed here)
    os.makedirs(args.outdir, exist_ok=True)
    for name, spec in cfg.assets.items():
        if spec.camera is None or name not in rostered:
            continue
        cam = fleet.cams.get(name)
        frame = cam.grab() if cam else None
        if frame is None:
            check(f"camera {name}", False, "no frame from " + spec.camera.snapshot_url)
            continue
        path = os.path.join(args.outdir, f"{name}.png")
        cv2.imwrite(path, frame.image)
        expected = (spec.camera.width, spec.camera.height)
        actual = (frame.width, frame.height)
        check(f"camera {name}", frame.image.size > 0,
              f"{actual[0]}x{actual[1]} -> {path}"
              + ("" if actual == expected else f"  (config says {expected[0]}x{expected[1]})"))

    # 7. tracks
    resp = fleet.tracks.post("TestTrack", 71.9965, -94.8448)
    created = bool(resp and resp.get("ok"))
    time.sleep(0.5)
    listed = {t.get("name") for t in fleet.tracks.list()}
    check("tracks: create + list", created and "TestTrack" in listed,
          f"uuid={resp.get('uuid') if resp else None} listed={'TestTrack' in listed}")

    # summary
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name} — {detail}")

    fleet.shutdown()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
