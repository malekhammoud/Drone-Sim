#!/usr/bin/env python3
"""Sim health check: is each asset's SITL actually getting sensor data?

The Arctic sim can get into a state where a vehicle's SITL is alive and its
MAVLink answers, but the Gazebo->SITL FDM/IMU feed is dead. The autopilot then
invents an attitude from a zero accelerometer and refuses to arm ("EKF3
Roll/Pitch inconsistent", "Check mag field", "3D Accel calibration needed").
`/api/reset` and `/api/rebuild` do NOT always clear it.

This tool flags that condition directly: a level, stationary IMU must read about
-1000 mg on Z. All-zero raw IMU = the FDM link is dead.

    python tools/sim_health.py
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcticlib.config import load_config  # noqa: E402
from arcticlib.simctl import SimClient  # noqa: E402


def check_asset(cfg, name: str) -> dict:
    from pymavlink import mavutil
    M = mavutil.mavlink
    spec = cfg.assets[name]
    m = mavutil.mavlink_connection(spec.master, source_system=255, source_component=190)
    hb = None
    t0 = time.time()
    while hb is None and time.time() - t0 < 10:
        m.mav.heartbeat_send(6, 8, 0, 0, 0)
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
    out = {"name": name, "connected": hb is not None}
    if hb is None:
        m.close()
        return out
    for nm in ("RAW_IMU", "SIMSTATE", "ATTITUDE", "STATUSTEXT"):
        mid = getattr(M, f"MAVLINK_MSG_ID_{nm}", None)
        if mid:
            m.mav.command_long_send(m.target_system, m.target_component,
                                    M.MAV_CMD_SET_MESSAGE_INTERVAL, 0, mid, 200000,
                                    0, 0, 0, 0, 0)
    end = time.time() + 5
    imu = sim = att = None
    text = ""
    while time.time() < end:
        msg = m.recv_match(blocking=True, timeout=0.5)
        if not msg:
            continue
        t = msg.get_type()
        if t == "RAW_IMU":
            imu = (msg.xacc, msg.yacc, msg.zacc)
        elif t == "SIMSTATE":
            sim = (math.degrees(msg.roll), math.degrees(msg.pitch), math.degrees(msg.yaw))
        elif t == "ATTITUDE":
            att = (math.degrees(msg.roll), math.degrees(msg.pitch), math.degrees(msg.yaw))
        elif t == "STATUSTEXT":
            text = msg.text
    m.close()
    out.update(imu=imu, sim=sim, att=att, last_statustext=text)
    if imu is not None:
        mag = math.sqrt(sum(v * v for v in imu))
        out["imu_magnitude"] = mag
        # A real IMU at rest reads ~1000 mg (1 g) on some axis.
        out["healthy"] = mag > 500.0
    else:
        out["healthy"] = False
    return out


def main() -> int:
    cfg = load_config()
    sim = SimClient(cfg)
    time.sleep(2)
    print("sim:", sim.status())
    assets = sim.assets()
    if not assets:
        print("  control API unreachable or no assets — is the local stack running?")
        print(f"  (tried {cfg.url(cfg.control_port)}/api/assets)")
        sim.stop()
        return 1
    bad = []
    for a in assets:
        if not a.get("mavlink"):
            print(f"  {a['name']:<12} MAVLINK DOWN")
            continue
        r = check_asset(cfg, a["name"])
        if not r.get("connected"):
            print(f"  {r['name']:<12} NO HEARTBEAT")
            bad.append(r["name"])
            continue
        imu = r.get("imu")
        flag = "OK" if r.get("healthy") else "**BAD**"
        print(f"  {r['name']:<12} {flag:<8} IMU={imu} |mag|={r.get('imu_magnitude', 0):.0f} "
              f"sim={tuple(round(v,1) for v in (r.get('sim') or ()))} "
              f"att={tuple(round(v,1) for v in (r.get('att') or ()))}")
        if r.get("last_statustext"):
            print(f"               last: {r['last_statustext']}")
        if not r.get("healthy"):
            bad.append(r["name"])
    sim.stop()
    print()
    if bad:
        print(f"UNHEALTHY (dead FDM/IMU): {', '.join(bad)}")
        print("Fix: restart the sim containers on the host (docker compose down/up),")
        print("or give each vehicle its own model dir so FDM ports do not collide")
        print("(the sim logs warn: '5 assets share model skywalker_x8'). /api/reset")
        print("and /api/rebuild do NOT reliably clear this.")
        return 1
    print("All assets healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
