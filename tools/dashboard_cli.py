#!/usr/bin/env python3
"""Live terminal dashboard: asset positions in local ENU metres.

A rough north-up ASCII map plus a status table. The fancy dashboard is someone
else's job; this is for a quick eyeball during integration.

    python tools/dashboard_cli.py            # refresh every second
    python tools/dashboard_cli.py --once
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcticlib.config import load_config  # noqa: E402
from arcticlib.fleet import Fleet  # noqa: E402
from arcticlib.geo import Georef  # noqa: E402

MARK = {"quadcopter": "Q", "fixed-wing": "P", "tower-1": "1", "tower-2": "2",
        "rover": "R"}
DEV = os.environ.get("ARCTICSIM_DEV") == "1"


def draw_map(points: dict, ship, width: int = 47, height: int = 15) -> str:
    """North-up ASCII map of ENU points (east->right, north->up)."""
    allpts = list(points.values()) + ([ship] if ship else [])
    if not allpts:
        return "(no positions)"
    min_e = min(p[0] for p in allpts)
    max_e = max(p[0] for p in allpts)
    min_n = min(p[1] for p in allpts)
    max_n = max(p[1] for p in allpts)
    span = max(max_e - min_e, max_n - min_n, 1.0)
    grid = [[" "] * width for _ in range(height)]

    def place(e, n, ch):
        cx = int((e - min_e) / span * (width - 1))
        cy = height - 1 - int((n - min_n) / span * (height - 1))
        grid[max(0, min(height - 1, cy))][max(0, min(width - 1, cx))] = ch

    for name, (e, n) in points.items():
        place(e, n, MARK.get(name, "?"))
    if ship:
        place(ship[0], ship[1], "S")
    return "\n".join("".join(row) for row in grid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config()
    georef = Georef(cfg.origin_lat, cfg.origin_lon, ps_centre_x=cfg.ps_centre_x,
                    ps_centre_y=cfg.ps_centre_y)
    fleet = Fleet.from_config(cfg)
    fleet.wait_ready(20)

    gt = None
    if DEV:
        from arcticlib.groundtruth import GroundTruth
        gt = GroundTruth(cfg, georef)

    try:
        while True:
            points = {}
            for name, v in fleet.vehicles.items():
                p = v.pose()
                if abs(p.lat) < 1:
                    continue
                e, n, _ = georef.to_enu(p.lat, p.lon)
                points[name] = (e, n)

            ship = None
            if gt:
                w = gt.ship_world()
                if w:
                    ship = (w[0], w[1])

            st = fleet.sim.status()
            lines = [f"sim {cfg.host}  state={st.get('state')}  "
                     f"sim_time={fleet.sim.sim_time():.0f}s  "
                     f"clock={'ok' if fleet.sim.clocks else 'waiting'}", ""]
            lines.append("      asset        link mode        armed    alt    E(m)    N(m)   spd")
            for name, v in sorted(fleet.vehicles.items()):
                p = v.pose()
                e, n = points.get(name, (float("nan"), float("nan")))
                lines.append(
                    f"  {MARK.get(name,'?')}  {name:<12} "
                    f"{'up' if v.connected else '--':>4} {v.mode:<11} "
                    f"{'Y' if v.armed else 'n':>5} {p.alt_rel:7.1f} "
                    f"{e:7.0f} {n:7.0f} {p.speed:5.1f}")
            if ship:
                lines.append(f"  S  target_vessel  (ground truth dev)  E={ship[0]:.0f} "
                             f"N={ship[1]:.0f}")
            lines += ["", draw_map(points, ship, width=min(60, 47), height=14), "",
                      "north up, east right; S = target vessel (dev)"]

            if args.once:
                print("\n".join(lines))
                break
            sys.stdout.write("\033[2J\033[H" + "\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if gt:
            gt.stop()
        fleet.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
