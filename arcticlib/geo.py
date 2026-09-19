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
