#!/usr/bin/env python3
"""Plot the Fixed-Wing Hot-Dog Lawnmower Patrol Plan over real satellite terrain.

Fetches satellite imagery of Bellot Strait / Fort Ross, plots the planned hot-dog
waypoints, starting positions of all assets, and any previously observed ship tracks.

Usage:
    python tools/plot_patrol_plan.py --out patrol_plan.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arcticlib.config import load_config
from arcticlib.geo import Georef, destination
from tools.patrol_and_record import generate_hotdog_waypoints


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile: int, ytile: int, zoom: int) -> tuple[float, float]:
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    return (math.degrees(lat_rad), lon_deg)


def fetch_satellite_mosaic(center_lat: float, center_lon: float, zoom: int = 12
                           ) -> tuple[np.ndarray, float, float, float, float]:
    """Download and stitch a 3x3 tile mosaic around center coordinates."""
    cx, cy = deg2num(center_lat, center_lon, zoom)
    tiles = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            tx, ty = cx + dx, cy + dy
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}"
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=8) as resp:
                    arr = np.frombuffer(resp.read(), dtype=np.uint8)
                    tiles[(dx, dy)] = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception as exc:
                print(f"Tile {tx},{ty} failed: {exc}")

    rows = []
    for dy in [-1, 0, 1]:
        row = []
        for dx in [-1, 0, 1]:
            row.append(tiles.get((dx, dy), np.zeros((256, 256, 3), dtype=np.uint8)))
        rows.append(np.hstack(row))
    mosaic = np.vstack(rows)
    mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)

    top_lat, left_lon = num2deg(cx - 1, cy - 1, zoom)
    bottom_lat, right_lon = num2deg(cx + 2, cy + 2, zoom)
    return mosaic_rgb, top_lat, bottom_lat, left_lon, right_lon


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="patrol_plan.png", help="Output plot filename")
    parser.add_argument("--sidecar", default="patrol_run/2026-09-19T16-22-21/sidecar.jsonl",
                        help="Optional sidecar with observed ship positions")
    args = parser.parse_args()

    cfg = load_config()
    georef = Georef(cfg.origin_lat, cfg.origin_lon, ps_centre_x=cfg.ps_centre_x, ps_centre_y=cfg.ps_centre_y)

    print("Fetching satellite terrain mosaic for Bellot Strait...")
    mosaic_rgb, top_lat, bottom_lat, left_lon, right_lon = fetch_satellite_mosaic(
        cfg.origin_lat, cfg.origin_lon, zoom=12
    )

    waypoints = generate_hotdog_waypoints(georef, leg_length_m=4800.0, lane_spacing_m=350.0, num_lanes=4)

    fig, ax = plt.subplots(figsize=(13, 10), dpi=150)
    ax.imshow(mosaic_rgb, extent=[left_lon, right_lon, bottom_lat, top_lat])

    # Key Landmark Coordinates
    t1 = (71.980671, -94.853711)
    t2 = (72.011778, -94.804721)
    plane_start = (71.998195, -94.841967)

    ax.plot(t1[1], t1[0], "r^", markersize=12, label="Tower 1 (SW)", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(t2[1], t2[0], "m^", markersize=12, label="Tower 2 (NE)", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(plane_start[1], plane_start[0], "go", markersize=12, label="Plane Start (Middle)", markeredgecolor="white", markeredgewidth=2.0)

    # Flight path
    w_lats = [w[0] for w in waypoints]
    w_lons = [w[1] for w in waypoints]
    full_lats = [plane_start[0]] + w_lats
    full_lons = [plane_start[1]] + w_lons

    ax.plot(full_lons, full_lats, color="yellow", linestyle="--", linewidth=2.2, alpha=0.95, label="Hot-Dog Flight Path")

    for idx, (wlat, wlon, _, name) in enumerate(waypoints):
        ax.plot(wlon, wlat, "yo", markersize=8, markeredgecolor="black")
        ax.annotate(f"#{idx+1} {name}", (wlon, wlat), textcoords="offset points", xytext=(5, 5),
                    color="yellow", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.65))

    # Observed ship positions if sidecar provided
    if os.path.exists(args.sidecar):
        ship_pts = []
        with open(args.sidecar) as f:
            for line in f:
                d = json.loads(line)
                if d.get("groundtruth"):
                    ship_pts.append((d["groundtruth"]["lat"], d["groundtruth"]["lon"]))
        if ship_pts:
            s_lats, s_lons = zip(*ship_pts)
            ax.plot(s_lons, s_lats, color="cyan", linewidth=3.0, label="Observed Ship Track")
            ax.plot(s_lons[-1], s_lats[-1], "cs", markersize=10, label="Ship Current Area")

    ax.set_title("Fixed-Wing 'Hot-Dog' Patrol Plan (Bellot Strait)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_xlim([left_lon, right_lon])
    ax.set_ylim([bottom_lat, top_lat])
    ax.legend(loc="lower right", framealpha=0.85)
    ax.grid(True, linestyle=":", alpha=0.5, color="white")

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Patrol plan plot saved to: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
