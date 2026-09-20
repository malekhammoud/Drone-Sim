#!/usr/bin/env python3
"""Plot the Alternative Fixed-Wing Patrol Plan over satellite imagery of Bellot Strait.

Overlays image-analysis-derived safe waypoints with 3-tier altitude profile:
  - 100m: Coast Clearance
  - 50m: Center Channel
  - 150m: Island & Northern Stream (-94.855 to -94.830)

Usage:
    python tools/plot_patrol_plan.py --out patrol_plan_alternative.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arcticlib.config import load_config
from tools.patrol_and_record import generate_safe_strait_waypoints


def num2deg(xtile: int, ytile: int, zoom: int) -> tuple[float, float]:
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    return (math.degrees(lat_rad), lon_deg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="patrol_plan_alternative.png", help="Output plot filename")
    parser.add_argument("--margin", type=float, default=120.0, help="Safe margin from coast in metres")
    parser.add_argument("--spacing", type=float, default=0.01, help="Longitude spacing")
    parser.add_argument("--sidecar", default="patrol_run/2026-09-19T18-51-51/sidecar.jsonl",
                        help="Optional sidecar with observed ship positions")
    args = parser.parse_args()

    mosaic_path = "strait_full_mosaic.jpg"
    if not os.path.exists(mosaic_path):
        print(f"Error: {mosaic_path} not found. Run image fetch first.")
        return 1

    mosaic_bgr = cv2.imread(mosaic_path)
    mosaic_rgb = cv2.cvtColor(mosaic_bgr, cv2.COLOR_BGR2RGB)

    zoom = 13
    x_min, x_max = 1935, 1941
    y_min, y_max = 1691, 1696
    top_lat, left_lon = num2deg(x_min, y_min, zoom)
    bot_lat, right_lon = num2deg(x_max + 1, y_max + 1, zoom)

    waypoints = generate_safe_strait_waypoints(step_lon=args.spacing, safe_margin_m=args.margin)
    print(f"Loaded {len(waypoints)} safe alternative waypoints.")

    fig, ax = plt.subplots(figsize=(16, 11), dpi=160)
    ax.imshow(mosaic_rgb, extent=[left_lon, right_lon, bot_lat, top_lat])

    # Key Landmark Coordinates
    t1 = (71.980671, -94.853711)
    t2 = (72.011778, -94.804721)
    plane_start = (71.998195, -94.841967)

    ax.plot(t1[1], t1[0], "r^", markersize=13, label="Tower 1 (SW)", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(t2[1], t2[0], "m^", markersize=13, label="Tower 2 (NE)", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(plane_start[1], plane_start[0], "wo", markersize=12, label="Plane Takeoff (71.998, -94.842)", markeredgecolor="black", markeredgewidth=2.0)

    # Highlight Island & Northern Stream Region
    isl_lon_min, isl_lon_max = -94.855, -94.830
    ax.axvspan(isl_lon_min, isl_lon_max, color="cyan", alpha=0.15, label="Island & Stream Zone (Fly 150m)")

    # Draw Flight Path
    w_lats = [w[0] for w in waypoints]
    w_lons = [w[1] for w in waypoints]
    w_alts = [w[2] for w in waypoints]

    full_lats = [plane_start[0]] + w_lats
    full_lons = [plane_start[1]] + w_lons

    ax.plot(full_lons, full_lats, color="white", linestyle="--", linewidth=1.6, alpha=0.85, label="Flight Route")

    # Plot waypoints by altitude tier
    wp_75_lat, wp_75_lon, wp_75_lbl = [], [], []
    wp_100_lat, wp_100_lon, wp_100_lbl = [], [], []
    wp_125_lat, wp_125_lon, wp_125_lbl = [], [], []

    for idx, (wlat, wlon, walt, wname) in enumerate(waypoints):
        num = idx + 1
        if walt <= 80.0:
            wp_75_lat.append(wlat)
            wp_75_lon.append(wlon)
            wp_75_lbl.append((num, wname))
        elif walt <= 110.0:
            wp_100_lat.append(wlat)
            wp_100_lon.append(wlon)
            wp_100_lbl.append((num, wname))
        else:
            wp_125_lat.append(wlat)
            wp_125_lon.append(wlon)
            wp_125_lbl.append((num, wname))

    # Scatter by altitude
    if wp_75_lon:
        ax.scatter(wp_75_lon, wp_75_lat, c="#00FF66", s=55, edgecolors="black", linewidths=1.0, zorder=5, label="Channel Center (75m alt)")
    if wp_100_lon:
        ax.scatter(wp_100_lon, wp_100_lat, c="#FFDD00", s=65, edgecolors="black", linewidths=1.0, zorder=5, label="Coast Clearance (100m alt)")
    if wp_125_lon:
        ax.scatter(wp_125_lon, wp_125_lat, c="#FF00FF", s=85, edgecolors="white", linewidths=1.5, zorder=5, label="Island & Stream (125m alt)")

    # Annotate sample waypoints
    for idx, (wlat, wlon, walt, wname) in enumerate(waypoints):
        num = idx + 1
        if num in [1, 2, 3, 10, 11, 12, 19, 20, 21, 22, 23, 24, 35, 36, 45, 50, len(waypoints)]:
            col_txt = "#00FF66" if walt == 50 else ("#FFDD00" if walt == 100 else "#FF00FF")
            ax.annotate(f"#{num} ({walt:.0f}m)", (wlon, wlat), textcoords="offset points", xytext=(4, 4),
                        color="white", fontsize=7.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", ec=col_txt, alpha=0.75))

    # Observed ship positions if sidecar provided
    if os.path.exists(args.sidecar):
        ship_pts = []
        with open(args.sidecar) as f:
            for line in f:
                d = json.loads(line)
                if d.get("groundtruth"):
                    ship_pts.append((d["groundtruth"]["lat"], d["groundtruth"]["lon"]))
        if ship_pts:
            s_lats, s_lons = zip(*ship_pts[::10])
            ax.plot(s_lons, s_lats, color="cyan", linewidth=2.5, alpha=0.8, label="Observed Ship Track")
            ax.plot(s_lons[-1], s_lats[-1], "s", color="cyan", markersize=9, markeredgecolor="black", label="Ship Observed Location")

    # Zoom in to the Bellot Strait channel
    ax.set_xlim([-94.945, -94.675])
    ax.set_ylim([71.968, 72.028])

    ax.set_title("Alternative Fixed-Wing Patrol Plan (Image Analysis + 3-Tier Altitude Profile)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.4, color="white")

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Alternative patrol plan plot saved to: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

