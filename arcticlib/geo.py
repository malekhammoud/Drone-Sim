"""Geodesy helpers: local ENU metres, bearings, and the sim's world frame.

Two coordinate systems appear in this project and they are easy to confuse:

* **lat/lon** — what MAVLink, detections and the track API speak.
* **world metres** — Gazebo's ``x y z``, which the sim UI shows on right-click
  and which ``~/pose/info`` carries for the target vessel. World axes are
  EPSG:3413 polar-stereographic grid metres offset by the site centre, so they
  are **not** true ENU: at Fort Ross grid north is ~49.8 deg off true north.

:class:`Georef` covers the local ENU maths (filters, dashboards, spacing) and
:func:`ps_to_latlon` / :func:`latlon_to_ps` convert the sim's world frame.

At this latitude a degree of longitude is ~0.31 of a degree of latitude, so
never treat lat/lon as a flat grid by hand — go through here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# WGS84
A = 6378137.0
F = 1.0 / 298.257223563
E2 = F * (2.0 - F)
E = math.sqrt(E2)

# EPSG:3413 / ArcticDEM projection parameters (matches the sim UI's own maths).
LAT_TS = 70.0
LON_0 = -45.0


# --------------------------------------------------------------------------- #
# Spherical helpers
# --------------------------------------------------------------------------- #
def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (haversine)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * A * math.asin(min(1.0, math.sqrt(h)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial true bearing from point 1 to point 2, degrees in [0, 360)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def destination(lat: float, lon: float, bearing: float,
                dist_m: float) -> tuple[float, float]:
    """Point ``dist_m`` metres from (lat, lon) along a true bearing."""
    p1 = math.radians(lat)
    l1 = math.radians(lon)
    br = math.radians(bearing)
    d = dist_m / A
    p2 = math.asin(math.sin(p1) * math.cos(d) +
                   math.cos(p1) * math.sin(d) * math.cos(br))
    l2 = l1 + math.atan2(math.sin(br) * math.sin(d) * math.cos(p1),
                         math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
# Polar stereographic (the sim's world frame)
# --------------------------------------------------------------------------- #
def _ps_constants() -> tuple[float, float]:
    lat_ts = math.radians(LAT_TS)
    tc = (math.tan(math.pi / 4 - lat_ts / 2) /
          ((1 - E * math.sin(lat_ts)) / (1 + E * math.sin(lat_ts))) ** (E / 2))
    mc = math.cos(lat_ts) / math.sqrt(1 - E2 * math.sin(lat_ts) ** 2)
    return tc, mc


_TC, _MC = _ps_constants()


def ps_to_latlon(x: float, y: float) -> tuple[float, float]:
    """EPSG:3413 metres -> (lat, lon) degrees. Same maths as the sim UI."""
    rho = math.hypot(x, y)
    if rho < 1e-9:
        return LAT_TS, LON_0
    t = rho * _TC / (A * _MC)
    chi = math.pi / 2 - 2 * math.atan(t)
    e2, e4, e6, e8 = E2, E2 ** 2, E2 ** 3, E2 ** 4
    lat = (chi
           + (e2 / 2 + 5 * e4 / 24 + e6 / 12 + 13 * e8 / 360) * math.sin(2 * chi)
           + (7 * e4 / 48 + 29 * e6 / 240 + 811 * e8 / 11520) * math.sin(4 * chi)
           + (7 * e6 / 120 + 81 * e8 / 1120) * math.sin(6 * chi)
           + (4279 * e8 / 161280) * math.sin(8 * chi))
    lon = math.radians(LON_0) + math.atan2(x, -y)
    return math.degrees(lat), (math.degrees(lon) + 540.0) % 360.0 - 180.0


def latlon_to_ps(lat: float, lon: float) -> tuple[float, float]:
    """(lat, lon) degrees -> EPSG:3413 metres (inverse of :func:`ps_to_latlon`)."""
    p = math.radians(lat)
    t = (math.tan(math.pi / 4 - p / 2) /
         ((1 - E * math.sin(p)) / (1 + E * math.sin(p))) ** (E / 2))
    rho = A * _MC * t / _TC
    dl = math.radians(lon - LON_0)
    return rho * math.sin(dl), -rho * math.cos(dl)


# --------------------------------------------------------------------------- #
# Local ENU around an origin
# --------------------------------------------------------------------------- #
@dataclass
class Georef:
    """Local East/North/Up metres around a configurable origin.

    Uses the WGS84 meridian and prime-vertical radii at the origin, which is
    accurate to well under a metre across a few kilometres. For high-precision
    or long-baseline work swap in pyproj; nothing here needs it.

    Also maps the sim's Gazebo world frame when ``ps_centre_x/y`` are given
    (the EPSG:3413 coordinates of world ``(0, 0)``).
    """

    lat0: float
    lon0: float
    alt0: float = 0.0
    ps_centre_x: float = 0.0
    ps_centre_y: float = 0.0

    def __post_init__(self) -> None:
        p = math.radians(self.lat0)
        self._sin = math.sin(p)
        self._m = A * (1 - E2) / (1 - E2 * self._sin ** 2) ** 1.5  # meridian radius
        self._n = A / math.sqrt(1 - E2 * self._sin ** 2)           # prime vertical

    # -- lat/lon <-> ENU --------------------------------------------------- #
    def to_enu(self, lat: float, lon: float, alt: float = 0.0
               ) -> tuple[float, float, float]:
        """(lat, lon, alt) -> (east, north, up) metres from the origin."""
        east = math.radians(lon - self.lon0) * self._n * math.cos(math.radians(self.lat0))
        north = math.radians(lat - self.lat0) * self._m
        return east, north, alt - self.alt0

    def to_latlon(self, east: float, north: float, up: float = 0.0
                  ) -> tuple[float, float, float]:
        """(east, north, up) metres -> (lat, lon, alt). Inverse of :meth:`to_enu`."""
        lat = self.lat0 + math.degrees(north / self._m)
        lon = self.lon0 + math.degrees(east / (self._n * math.cos(math.radians(self.lat0))))
        return lat, lon, self.alt0 + up

    # -- world (Gazebo) <-> lat/lon -------------------------------------- #
    def world_to_latlon(self, x: float, y: float) -> tuple[float, float]:
        """Gazebo world metres -> (lat, lon). Needs ``ps_centre_x/y``."""
        return ps_to_latlon(self.ps_centre_x + x, self.ps_centre_y + y)

    def latlon_to_world(self, lat: float, lon: float) -> tuple[float, float]:
        """(lat, lon) -> Gazebo world metres. Needs ``ps_centre_x/y``."""
        px, py = latlon_to_ps(lat, lon)
        return px - self.ps_centre_x, py - self.ps_centre_y

    # -- bearings ---------------------------------------------------------- #
    @staticmethod
    def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        return bearing_deg(lat1, lon1, lat2, lon2)

    @staticmethod
    def distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        return distance_m(lat1, lon1, lat2, lon2)


# --------------------------------------------------------------------------- #
# Tactical Geodesy: Intercept Spirals & Closed-Zone Rerouting
# --------------------------------------------------------------------------- #
def segment_circle_dist_m(lat1: float, lon1: float,
                          lat2: float, lon2: float,
                          c_lat: float, c_lon: float) -> float:
    """Compute the minimum distance in metres from circle center to segment (P1, P2)."""
    georef = Georef(c_lat, c_lon)
    x1, y1, _ = georef.to_enu(lat1, lon1)
    x2, y2, _ = georef.to_enu(lat2, lon2)

    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-9:
        return math.hypot(x1, y1)

    # Project origin (0, 0) onto line segment: P(t) = P1 + t * (P2 - P1)
    t = -(x1 * dx + y1 * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.hypot(closest_x, closest_y)


def generate_search_spiral(center_lat: float,
                           center_lon: float,
                           alt: float = 75.0,
                           r0: float = 100.0,
                           dr: float = 130.0,
                           r_max: float = 650.0,
                           n_points_per_turn: int = 8,
                           start_bearing: float = 0.0) -> list[tuple[float, float, float, str]]:
    """Generate an expanding Archimedean search spiral around (center_lat, center_lon).

    Parameters:
        center_lat, center_lon: Center coordinates (last known boat sighting).
        alt: Altitude in metres (default: 75m).
        r0: Initial spiral radius in metres (default: 100m, > plane turning radius).
        dr: Radial expansion per 360-degree turn in metres (default: 130m).
        r_max: Maximum search radius in metres (default: 650m).
        n_points_per_turn: Number of waypoints per revolution (default: 8).
        start_bearing: Initial angle in degrees (default: 0 deg / North).

    Returns:
        List of (lat, lon, alt, name) waypoints.
    """
    waypoints = []
    d_theta = 360.0 / n_points_per_turn
    dr_step = dr / n_points_per_turn

    k = 0
    while True:
        r_k = r0 + k * dr_step
        if r_k > r_max:
            break
        bearing = (start_bearing + k * d_theta) % 360.0
        wlat, wlon = destination(center_lat, center_lon, bearing, r_k)
        wname = f"Spiral_{k+1}_r{r_k:.0f}m"
        waypoints.append((wlat, wlon, alt, wname))
        k += 1

    return waypoints


def generate_figure8_pattern(center_lat: float,
                             center_lon: float,
                             bearing_deg: float = 85.0,
                             length_m: float = 400.0,
                             width_m: float = 160.0,
                             alt: float = 75.0,
                             num_cycles: int = 3) -> list[tuple[float, float, float, str]]:
    """Generate a Bowtie / Figure-8 maritime search & overflight tracking pattern.

    Replaces the spiral pattern with alternating straight overflight passes directly
    over (center_lat, center_lon) aligned with the waterway axis (bearing_deg).

    Advantages over the spiral:
    1. Straight overflight runs: wings are completely level (roll=0), nose is pointed
       directly at the vessel, keeping it dead-center in the camera field of view
       for 15-25 seconds per pass.
    2. Continuous re-attack: crosses the vessel from both directions (e.g. East & West),
       maximizing detection probability and depression angle while never getting stuck
       banking in a blind loiter circle.
    3. Channel-aligned: stays along the navigable water channel, avoiding surrounding terrain.

    Parameters:
        center_lat, center_lon: Center coordinates (vessel position or sighting).
        bearing_deg: Primary axis of the strait / vessel track (default: 85.0 deg).
        length_m: Half-length of the overflight run (default: 400m).
        width_m: Lateral turn offset for the 180-degree reversals (default: 160m).
        alt: Flight altitude in metres (default: 75m).
        num_cycles: Number of figure-8 cycles to generate (default: 3).

    Returns:
        List of (lat, lon, alt, name) waypoints.
    """
    waypoints = []
    fwd_bearing = bearing_deg % 360.0
    rev_bearing = (bearing_deg + 180.0) % 360.0
    right_bearing = (bearing_deg + 90.0) % 360.0
    left_bearing = (bearing_deg - 90.0) % 360.0

    p_center = (center_lat, center_lon)
    p_fwd = destination(center_lat, center_lon, fwd_bearing, length_m)
    p_aft = destination(center_lat, center_lon, rev_bearing, length_m)

    p_fwd_right = destination(p_fwd[0], p_fwd[1], right_bearing, width_m)
    p_mid_right = destination(center_lat, center_lon, right_bearing, width_m)
    p_aft_left = destination(p_aft[0], p_aft[1], left_bearing, width_m)
    p_mid_left = destination(center_lat, center_lon, left_bearing, width_m)

    for cycle in range(num_cycles):
        c_num = cycle + 1
        # Pass 1: Aft -> Center (Overflight) -> Fwd
        waypoints.append((p_aft[0], p_aft[1], alt, f"Fig8_C{c_num}_InboundAft"))
        waypoints.append((p_center[0], p_center[1], alt, f"Fig8_C{c_num}_Overflight_Fwd"))
        waypoints.append((p_fwd[0], p_fwd[1], alt, f"Fig8_C{c_num}_ExtensionFwd"))

        # Right reversal loop (smooth 180 degree turn outside viewing area)
        waypoints.append((p_fwd_right[0], p_fwd_right[1], alt, f"Fig8_C{c_num}_TurnRightApex"))
        waypoints.append((p_mid_right[0], p_mid_right[1], alt, f"Fig8_C{c_num}_TurnRightBase"))

        # Pass 2: Fwd -> Center (Overflight) -> Aft
        waypoints.append((p_fwd[0], p_fwd[1], alt, f"Fig8_C{c_num}_InboundFwd"))
        waypoints.append((p_center[0], p_center[1], alt, f"Fig8_C{c_num}_Overflight_Rev"))
        waypoints.append((p_aft[0], p_aft[1], alt, f"Fig8_C{c_num}_ExtensionAft"))

        # Left reversal loop (smooth 180 degree turn outside viewing area)
        waypoints.append((p_aft_left[0], p_aft_left[1], alt, f"Fig8_C{c_num}_TurnLeftApex"))
        waypoints.append((p_mid_left[0], p_mid_left[1], alt, f"Fig8_C{c_num}_TurnLeftBase"))

    return waypoints


def generate_racetrack_pattern(center_lat: float,
                               center_lon: float,
                               bearing_deg: float = 85.0,
                               length_m: float = 400.0,
                               width_m: float = 160.0,
                               alt: float = 75.0,
                               num_cycles: int = 3) -> list[tuple[float, float, float, str]]:
    """Generate a Racetrack overflight pattern oriented along bearing_deg."""
    waypoints = []
    fwd_bearing = bearing_deg % 360.0
    rev_bearing = (bearing_deg + 180.0) % 360.0
    offset_bearing = (bearing_deg + 90.0) % 360.0

    p_center = (center_lat, center_lon)
    p_fwd = destination(center_lat, center_lon, fwd_bearing, length_m)
    p_aft = destination(center_lat, center_lon, rev_bearing, length_m)

    p_fwd_out = destination(p_fwd[0], p_fwd[1], offset_bearing, width_m)
    p_mid_out = destination(center_lat, center_lon, offset_bearing, width_m)
    p_aft_out = destination(p_aft[0], p_aft[1], offset_bearing, width_m)

    for cycle in range(num_cycles):
        c_num = cycle + 1
        # Straight overflight leg directly over vessel
        waypoints.append((p_aft[0], p_aft[1], alt, f"Race_C{c_num}_Inbound"))
        waypoints.append((p_center[0], p_center[1], alt, f"Race_C{c_num}_Overflight"))
        waypoints.append((p_fwd[0], p_fwd[1], alt, f"Race_C{c_num}_Outbound"))
        # Racetrack return leg
        waypoints.append((p_fwd_out[0], p_fwd_out[1], alt, f"Race_C{c_num}_TurnFwd"))
        waypoints.append((p_mid_out[0], p_mid_out[1], alt, f"Race_C{c_num}_Downwind"))
        waypoints.append((p_aft_out[0], p_aft_out[1], alt, f"Race_C{c_num}_TurnAft"))

    return waypoints


def reroute_around_closed_zone(waypoints: list[tuple[float, float, float, str]],
                               c_lat: float,
                               c_lon: float,
                               radius_m: float,
                               safe_buffer_m: float = 80.0,
                               channel_center_lat: float = 71.988) -> tuple[list[tuple[float, float, float, str]], list[int]]:
    """Reroute waypoints around a circular exclusion zone.

    1. Removes waypoints located inside (radius_m + safe_buffer_m).
    2. Identifies flight segments that penetrate the exclusion zone and inserts
       tangent bypass detour waypoints skirting the navigable side of the channel.

    Returns:
        (rerouted_waypoints, invalidated_original_indices)
    """
    r_avoid = radius_m + safe_buffer_m
    invalidated_indices = []

    # Step 1: Filter out waypoints inside the avoidance zone
    surviving: list[tuple[int, tuple[float, float, float, str]]] = []
    for idx, wp in enumerate(waypoints):
        wlat, wlon, walt, wname = wp
        dist = distance_m(wlat, wlon, c_lat, c_lon)
        if dist <= r_avoid:
            invalidated_indices.append(idx)
        else:
            surviving.append((idx, wp))

    if not surviving:
        return [], invalidated_indices

    # Determine preferred detour side (skirt towards channel centerline)
    # If closed zone is North of channel center, bypass to the South (bearing ~180)
    # If closed zone is South of channel center, bypass to the North (bearing ~0)
    detour_bearing = 180.0 if c_lat >= channel_center_lat else 0.0

    # Step 2: Build rerouted sequence, checking for penetrating legs
    rerouted: list[tuple[float, float, float, str]] = [surviving[0][1]]

    for i in range(len(surviving) - 1):
        prev_idx, (lat1, lon1, alt1, name1) = surviving[i]
        next_idx, (lat2, lon2, alt2, name2) = surviving[i + 1]

        # Check if segment penetrates the exclusion zone
        min_dist = segment_circle_dist_m(lat1, lon1, lat2, lon2, c_lat, c_lon)
        if min_dist < r_avoid:
            # Segment intersects! Insert detour waypoint around perimeter
            # Compute detour waypoint along the safe perimeter
            # Angle pointing from center toward the midpoint of the segment or towards channel
            georef = Georef(c_lat, c_lon)
            x1, y1, _ = georef.to_enu(lat1, lon1)
            x2, y2, _ = georef.to_enu(lat2, lon2)
            mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            # Detour offset direction: choose vector pointing away from center towards channel
            mid_bearing = (math.degrees(math.atan2(mid_x, mid_y)) + 360.0) % 360.0
            # Blend segment midpoint bearing with channel center preference
            chosen_bearing = 0.5 * mid_bearing + 0.5 * detour_bearing
            d_lat, d_lon = destination(c_lat, c_lon, chosen_bearing, r_avoid)
            d_alt = (alt1 + alt2) / 2.0
            d_name = f"Detour_CZ_{prev_idx+1}_{next_idx+1}"
            rerouted.append((d_lat, d_lon, d_alt, d_name))

        rerouted.append((lat2, lon2, alt2, name2))

    return rerouted, invalidated_indices
