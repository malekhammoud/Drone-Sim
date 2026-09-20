#!/usr/bin/env python3
"""Track API CLI — the curl examples, as a tool.

    # Create a track (first POST with a new name):
    python tools/tracks_cli.py post --name "Sierra One" --lat 71.9965 --lon -94.8448

    # Update it (same name -> created:false, bumps fixes), with motion:
    python tools/tracks_cli.py post --name "Sierra One" --lat 71.9975 --lon -94.8450 \
        --heading 315 --speed 6.5

    # List current tracks:
    python tools/tracks_cli.py list

The endpoint comes from config (``ARCTICSIM_HOST`` / ``ARCTICSIM_TRACKS_PORT``);
override with ``--url http://<sim-ip>:8010``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcticlib.config import load_config
from arcticlib.tracks import TrackClient


def main() -> int:
    ap = argparse.ArgumentParser(description="Track API CLI (create/update/list).")
    ap.add_argument("--url", default=None, help="track API base URL (default: config)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("post", help="create or update a track by name")
    p.add_argument("--name", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--heading", type=float, default=None, help="deg true")
    p.add_argument("--speed", type=float, default=None, help="m/s")

    sub.add_parser("list", help="list current tracks")
    args = ap.parse_args()

    cfg = load_config()
    client = TrackClient(args.url or cfg.url(cfg.tracks_port))

    if args.cmd == "post":
        res = client.post(args.name, args.lat, args.lon,
                          heading=args.heading, speed=args.speed)
        if res is None:
            print("POST failed (endpoint unreachable?)")
            return 1
        print(json.dumps(res))
        return 0 if res.get("ok", True) else 1

    tracks = client.list()
    print(json.dumps({"ok": True, "count": len(tracks), "tracks": tracks}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
