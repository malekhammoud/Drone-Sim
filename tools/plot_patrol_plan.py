#!/usr/bin/env python3
"""Plot the Fixed-Wing Patrol Plan over satellite imagery of Bellot Strait.

Supports both:
1. Safe 3-tier altitude profile (75m Channel Center, 100m Coast Clearance, 125m Island/Stream)
   derived from high-resolution satellite image analysis.
2. Tactical edge-case overlays from field modifications:
   - Restricted / Closed Exclusion Zones (--closed-zone 'lat,lon,radius_m')
   - Dynamic Emergency Intercept Targets (--intercept 'lat,lon')

Usage:
    # Standard 3-tier safe patrol plan:
    python tools/plot_patrol_plan.py --out patrol_plan_alternative.png

    # Tactical overlay with closed zone:
    python tools/plot_patrol_plan.py --closed-zone 71.995,-94.810,800

    # Tactical overlay with emergency intercept target:
    python tools/plot_patrol_plan.py --intercept 71.985,-94.750
    python tools/plot_patrol_plan.py --ai-event "Attack reported near 71.995, -94.810! Close off the zone with an 800m radius."
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
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from arcticlib.config import load_config
from arcticlib.geo import (
    distance_m,
    generate_search_spiral,
    reroute_around_closed_zone,
)
from tools.patrol_and_record import generate_safe_strait_waypoints
from ollama_control import query_tether  # NEW


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


# NEW: translates a natural-language --ai-event into the existing --closed-zone
# / --intercept args, so nothing downstream needs to know the AI was involved.
def resolve_ai_event(args: argparse.Namespace) -> None:
    print(f"Querying AI supervisor: {args.ai_event!r}")
    decision = query_tether(args.ai_event)

    if decision is None:
        print("AI supervisor returned no valid decision — proceeding without an overlay.")
        return

    action = decision.get("action")
    lat = decision.get("lat")
    lon = decision.get("lon")
    reason = decision.get("reason", "")

    valid_coords = (
        isinstance(lat, (int, float)) and isinstance(lon, (int, float))
        and -90 <= lat <= 90 and -180 <= lon <= 180
    )

    print(f"AI decision: {action} at ({lat}, {lon}) — {reason}")

    if not valid_coords and action in ("CLOSED_ZONE", "INTERCEPT"):
        print(f"AI returned invalid or missing coordinates for action '{action}' — skipping overlay.")
        return

    if action == "CLOSED_ZONE":
        radius = decision.get("radius_m", 500)
        if not isinstance(radius, (int, float)) or radius <= 0:
            print(f"AI returned invalid radius_m ({radius!r}) — defaulting to 500m.")
            radius = 500
        args.closed_zone = f"{lat},{lon},{radius}"
    elif action == "INTERCEPT":
        args.intercept = f"{lat},{lon}"
    elif action == "ABORT":
        print("AI supervisor returned ABORT — no overlay will be drawn.")
    else:
        print(f"Unrecognized action '{action}' — skipping overlay.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Fixed-Wing Patrol Plan with optional tactical overlays.")
    parser.add_argument("--out", default="patrol_plan_alternative.png", help="Output plot filename")
    parser.add_argument("--margin", type=float, default=120.0, help="Safe margin from coast in metres (default: 120)")
    parser.add_argument("--spacing", type=float, default=0.01, help="Longitude spacing (default: 0.01 deg)")
    parser.add_argument("--sidecar", default="patrol_run/2026-09-19T18-51-51/sidecar.jsonl",
                        help="Optional sidecar with observed ship positions")
    # Tactical Edge Case Overlays
    parser.add_argument("--closed-zone", type=str, default=None,
                        help="Closed zone overlay formatted as 'lat,lon,radius_m' (e.g. '71.995,-94.810,800')")
    parser.add_argument("--intercept", type=str, default=None,
                        help="Dynamic intercept coordinate target as 'lat,lon' (e.g. '71.985,-94.750')")
    parser.add_argument("--ai-event", type=str, default=None,  # NEW
                        help="Natural-language tactical event description, routed through the AI supervisor "
                             "to decide the overlay (e.g., 'Unknown attack reported near tower 1')")
    args = parser.parse_args()

    if args.ai_event:  # NEW — only runs if the flag is passed; otherwise behaves exactly as before
        resolve_ai_event(args)

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

    base_waypoints = generate_safe_strait_waypoints(step_lon=args.spacing, safe_margin_m=args.margin)
    print(f"Loaded {len(base_waypoints)} safe baseline waypoints.")

    # Apply Closed-Zone Rerouting if requested
    cz_lat, cz_lon, cz_radius_m = None, None, None
    invalidated_indices = []
    if args.closed_zone:
        try:
            cz_lat_str, cz_lon_str, cz_rad_str = args.closed_zone.split(",")
            cz_lat, cz_lon, cz_radius_m = float(cz_lat_str), float(cz_lon_str), float(cz_rad_str)
            active_waypoints, invalidated_indices = reroute_around_closed_zone(
                base_waypoints, cz_lat, cz_lon, cz_radius_m, safe_buffer_m=80.0
            )
            print(f"Closed Zone: {len(invalidated_indices)} waypoints pruned. "
                  f"Active path rerouted to {len(active_waypoints)} waypoints avoiding exclusion zone.")
        except Exception as err:
            print(f"Error parsing --closed-zone parameter: {err}")
            active_waypoints = base_waypoints
    else:
        active_waypoints = base_waypoints

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
    ax.axvspan(isl_lon_min, isl_lon_max, color="cyan", alpha=0.15, label="Island & Stream Zone (Fly 125m)")

    # Draw Active Flight Path
    w_lats = [w[0] for w in active_waypoints]
    w_lons = [w[1] for w in active_waypoints]

    full_lats = [plane_start[0]] + w_lats
    full_lons = [plane_start[1]] + w_lons
    route_label = "Rerouted Flight Route (Avoids Closed Zone)" if args.closed_zone else "Flight Route"
    ax.plot(full_lons, full_lats, color="white", linestyle="--", linewidth=1.8, alpha=0.9, label=route_label, zorder=4)

    # Group active waypoints by altitude tier
    wp_75_lat, wp_75_lon = [], []
    wp_100_lat, wp_100_lon = [], []
    wp_125_lat, wp_125_lon = [], []

    for idx, (wlat, wlon, walt, wname) in enumerate(active_waypoints):
        if "Detour" in wname:
            continue
        if walt <= 80.0:
            wp_75_lat.append(wlat)
            wp_75_lon.append(wlon)
        elif walt <= 110.0:
            wp_100_lat.append(wlat)
            wp_100_lon.append(wlon)
        else:
            wp_125_lat.append(wlat)
            wp_125_lon.append(wlon)

    # Scatter points by altitude tier
    if wp_75_lon:
        ax.scatter(wp_75_lon, wp_75_lat, c="#00FF66", s=55, edgecolors="black", linewidths=1.0, zorder=5, label="Channel Center (75m alt)")
    if wp_100_lon:
        ax.scatter(wp_100_lon, wp_100_lat, c="#FFDD00", s=65, edgecolors="black", linewidths=1.0, zorder=5, label="Coast Clearance (100m alt)")
    if wp_125_lon:
        ax.scatter(wp_125_lon, wp_125_lat, c="#FF00FF", s=85, edgecolors="white", linewidths=1.5, zorder=5, label="Island & Stream (125m alt)")

    # Plot Detour Waypoints if present
    detour_pts = [w for w in active_waypoints if "Detour" in w[3]]
    if detour_pts:
        d_lons = [w[1] for w in detour_pts]
        d_lats = [w[0] for w in detour_pts]
        ax.scatter(d_lons, d_lats, c="#00FFFF", marker="D", s=90, edgecolors="black", linewidths=1.5, zorder=7, label="Bypass Detour Waypoint")
        for dw in detour_pts:
            ax.annotate("DETOUR", (dw[1], dw[0]), textcoords="offset points", xytext=(6, 6),
                        color="#00FFFF", fontsize=8, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="#00FFFF", alpha=0.8), zorder=8)

    # Annotate representative waypoints
    for idx, (wlat, wlon, walt, wname) in enumerate(active_waypoints):
        num = idx + 1
        if "Detour" not in wname and num in [1, 2, 3, 10, 11, 12, 19, 20, 21, 22, 23, 24, 35, 36, 45, 50, len(active_waypoints)]:
            col_txt = "#00FF66" if walt <= 80.0 else ("#FFDD00" if walt <= 110.0 else "#FF00FF")
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

    # --- TACTICAL EDGE CASE OVERLAYS ---

    # Edge Case 1: Closed-off / Restricted Exclusion Zone
    if cz_lat is not None and cz_lon is not None and cz_radius_m is not None:
        deg_radius_lat = cz_radius_m / 111000.0
        deg_radius_lon = cz_radius_m / (111000.0 * math.cos(math.radians(cz_lat)))

        circle = patches.Ellipse(
            (cz_lon, cz_lat), width=2 * deg_radius_lon, height=2 * deg_radius_lat,
            color="red", alpha=0.35, zorder=6, label=f"Closed Zone ({cz_radius_m:.0f}m)"
        )
        ax.add_patch(circle)
        ax.plot(cz_lon, cz_lat, "rx", markersize=14, markeredgewidth=2.5, zorder=7)

        # Plot invalidated waypoints in faded red
        for inv_i in invalidated_indices:
            inv_wp = base_waypoints[inv_i]
            ax.plot(inv_wp[1], inv_wp[0], "ro", markersize=8, markeredgecolor="white", alpha=0.7, zorder=6)
            ax.annotate(f"#[CLOSED]", (inv_wp[1], inv_wp[0]), textcoords="offset points", xytext=(-15, -15),
                        color="red", fontsize=8, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.85), zorder=7)

    # Edge Case 2: Dynamic Intercept Target & Expanding Search Spiral
    if args.intercept:
        try:
            it_lat_str, it_lon_str = args.intercept.split(",")
            it_lat, it_lon = float(it_lat_str), float(it_lon_str)

            ax.plot(it_lon, it_lat, "m*", markersize=18, label="Dynamic Intercept Target", markeredgecolor="white", zorder=10)
            ax.annotate("EMERGENCY TARGET", (it_lon, it_lat), textcoords="offset points", xytext=(10, -10),
                        color="magenta", fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="magenta", alpha=0.85), zorder=11)

            # Dynamic transit line from aircraft start to intercept point
            ax.plot([plane_start[1], it_lon], [plane_start[0], it_lat], color="magenta", linestyle=":", linewidth=2.5, zorder=9, label="Transit to Target")

            # Generate and draw expanding Archimedean search spiral around target
            spiral_wps = generate_search_spiral(it_lat, it_lon, alt=75.0, r0=100.0, dr=130.0, r_max=650.0)
            sp_lats = [it_lat] + [w[0] for w in spiral_wps]
            sp_lons = [it_lon] + [w[1] for w in spiral_wps]
            ax.plot(sp_lons, sp_lats, color="#FF00AA", linestyle="-", linewidth=2.0, alpha=0.85, zorder=10, label="Expanding Search Spiral (100m-650m)")
            ax.scatter([w[1] for w in spiral_wps], [w[0] for w in spiral_wps], c="#FF00AA", s=25, edgecolors="white", linewidths=0.8, zorder=11)

            print(f"Dynamic Intercept: Target at ({it_lat:.5f}, {it_lon:.5f}) with {len(spiral_wps)} spiral search waypoints (r=100m..650m).")
        except Exception as err:
            print(f"Error parsing --intercept parameter: {err}")

    # Set map bounds (expand dynamically if tactical overlays lie outside channel center)
    min_lon, max_lon = -94.945, -94.675
    min_lat, max_lat = 71.968, 72.028

    if cz_lat is not None and cz_lon is not None:
        deg_r_lon = cz_radius_m / (111000.0 * math.cos(math.radians(cz_lat)))
        deg_r_lat = cz_radius_m / 111000.0
        min_lon = min(min_lon, cz_lon - deg_r_lon * 1.15)
        max_lon = max(max_lon, cz_lon + deg_r_lon * 1.15)
        min_lat = min(min_lat, cz_lat - deg_r_lat * 1.15)
        max_lat = max(max_lat, cz_lat + deg_r_lat * 1.15)

    if args.intercept:
        try:
            it_lat, it_lon = (float(x) for x in args.intercept.split(","))
            min_lon = min(min_lon, it_lon - 0.015)
            max_lon = max(max_lon, it_lon + 0.015)
            min_lat = min(min_lat, it_lat - 0.008)
            max_lat = max(max_lat, it_lat + 0.008)
        except Exception:
            pass

    ax.set_xlim([min_lon, max_lon])
    ax.set_ylim([min_lat, max_lat])

    # Title based on mode
    if args.closed_zone or args.intercept:
        ax.set_title("Fixed-Wing Patrol Plan with Tactical Edge-Cases (Bellot Strait)", fontsize=14, fontweight="bold")
    else:
        ax.set_title("Alternative Fixed-Wing Patrol Plan (Image Analysis + 3-Tier Altitude Profile)", fontsize=14, fontweight="bold")

    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.4, color="white")

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Patrol plan plot successfully saved to: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())