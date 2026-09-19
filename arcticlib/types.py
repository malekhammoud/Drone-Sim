"""The data contract other teammates code against.

Keep these stable. Fields may gain defaults, but existing names and meanings
must not change without a version bump and a shout in the group chat.

Conventions
-----------
* ``t_sim``  — Gazebo sim seconds (from ``~/world_stats``). Use this for
  anything that feeds a filter.
* ``t_wall`` — ``time.monotonic()`` seconds at receipt. Always available, and
  the only clock that cannot jump backwards.
* Altitudes are metres. ``alt_rel`` is above the launch/home point,
  ``alt_amsl`` is above mean sea level.
* Angles are radians unless the name ends in ``_deg``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Pose:
    """One asset's state at one instant."""

    asset: str
    t_sim: float
    t_wall: float
    lat: float
    lon: float
    alt_rel: float = 0.0
    alt_amsl: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0          # NED north, m/s
    vy: float = 0.0          # NED east, m/s
    vz: float = 0.0          # NED down, m/s
    gimbal_pitch: Optional[float] = None
    gimbal_yaw: Optional[float] = None

    @property
    def speed(self) -> float:
        """Horizontal ground speed, m/s."""
        return float(np.hypot(self.vx, self.vy))


@dataclass
class Frame:
    """A camera frame with the metadata needed to geolocate detections."""

    asset: str
    t_sim: float
    t_wall: float
    image: np.ndarray                 # BGR, HxWx3, uint8
    width: int
    height: int
    hfov_deg: float
    vfov_deg: float

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.image.shape)


@dataclass
class Detection:
    """A geolocated detection, ready to post to the track API.

    ``bearing_only`` marks a detection whose range is unknown (single camera,
    no intersection): it still carries a lat/lon guess but the planner should
    treat it as a bearing. ``sigma_m`` is the 1-sigma position uncertainty in
    metres; ``conf`` is in [0, 1].
    """

    asset: str
    t_sim: float
    lat: float
    lon: float
    sigma_m: float
    conf: float
    bearing_only: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Battery:
    """Battery state. ``remaining_pct`` is None when the autopilot omits it."""

    voltage_v: Optional[float] = None
    current_a: Optional[float] = None
    remaining_pct: Optional[float] = None


@dataclass
class AssetStatus:
    """A lightweight liveness/health summary for dashboards and smoke tests."""

    asset: str
    connected: bool
    mode: str = "?"
    armed: bool = False
    battery: Optional[Battery] = None
    last_heartbeat_age: float = float("inf")
