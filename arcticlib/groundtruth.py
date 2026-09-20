"""Ground truth for the target vessel — **DEVELOPMENT ONLY**.

The sim publishes every moving model's true world pose on ``~/pose/info``. The
target vessel is the model named ``target_vessel`` (RECON.md §3), so we can read
its exact position. That is invaluable for auto-labelling CV data and scoring our
tracker offline — and it is **cheating in the judged run**, because the whole
task is to detect the ship without it.

Guard rails: importing is fine, but *constructing* :class:`GroundTruth` raises
unless ``ARCTICSIM_DEV=1`` is set, and it logs a loud warning when it does run.
The autonomous run must never import this module.
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from typing import Optional

from .config import Config, load_config
from .geo import Georef
from .simctl import SimClient

log = logging.getLogger("arcticlib.groundtruth")

DEV_ENV = "ARCTICSIM_DEV"
DEFAULT_SHIP = "target_vessel"


class GroundTruth:
    """Reads the target vessel's true pose. Refuses to run outside dev."""

    def __init__(self, config: Optional[Config] = None,
                 georef: Optional[Georef] = None,
                 ship_name: str = DEFAULT_SHIP) -> None:
        if os.environ.get(DEV_ENV) != "1":
            raise RuntimeError(
                "GroundTruth is DEV ONLY and must not be used in the judged run. "
                f"Set {DEV_ENV}=1 to override (e.g. for auto-labelling).")
        log.warning("GROUND TRUTH ENABLED (DEV ONLY) — never use in the judged run")

        self.config = config or load_config()
        self.georef = georef or Georef(self.config.origin_lat, self.config.origin_lon,
                                       ps_centre_x=self.config.ps_centre_x,
                                       ps_centre_y=self.config.ps_centre_y)
        self.ship_name = ship_name

        self._world: Optional[tuple[float, float, float]] = None
        self._hist: deque[tuple[float, float, float]] = deque(maxlen=400)  # x,y,t
        self._heading_deg: Optional[float] = None
        self._speed: Optional[float] = None
        self._sim = SimClient(self.config, pose_callback=self._on_pose)

    # ------------------------------------------------------------------ #
    def _on_pose(self, body: dict) -> None:
        name = (body.get("name") or "").split("::")[0]
        if name != self.ship_name:
            return
        pos = body.get("position") or {}
        x, y, z = float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))
        now = time.monotonic()
        self._hist.append((x, y, now))
        # pose/info arrives in bursts, so a per-message finite difference is
        # noisy (dt can be ~1 ms). Measure over a ~1 s baseline instead.
        window_start = now - 1.0
        while len(self._hist) > 2 and self._hist[1][2] < window_start:
            self._hist.popleft()
        x0, y0, t0 = self._hist[0]
        dt = now - t0
        if dt >= 0.2:
            dx, dy = x - x0, y - y0
            dist = math.hypot(dx, dy)
            self._speed = dist / dt
            if dist > 0.1:
                grid = math.degrees(math.atan2(dx, dy))
                self._heading_deg = (grid + self.config.convergence_deg + 360.0) % 360.0
        self._world = (x, y, z)

    # ------------------------------------------------------------------ #
    def ship_world(self) -> Optional[tuple[float, float, float]]:
        """True world (x, y, z) metres, or None until the first sample."""
        return self._world

    def ship_latlon(self) -> Optional[tuple[float, float]]:
        if self._world is None:
            return None
        return self.georef.world_to_latlon(self._world[0], self._world[1])

    def ship_pose(self) -> Optional[dict]:
        """``{lat, lon, heading, speed, world}`` or None."""
        if self._world is None:
            return None
        lat, lon = self.georef.world_to_latlon(self._world[0], self._world[1])
        return {"lat": lat, "lon": lon, "heading": self._heading_deg,
                "speed": self._speed, "world": self._world}

    def stop(self) -> None:
        self._sim.stop()


# --------------------------------------------------------------------------- #
# Projection helper (pure geometry, useful for auto-labelling)
# --------------------------------------------------------------------------- #
def project_point(point_world: tuple[float, float, float],
                  cam_world: tuple[float, float, float],
                  cam_yaw_deg: float, intrinsics: dict,
                  cam_pitch_deg: float = 0.0, cam_roll_deg: float = 0.0
                  ) -> Optional[tuple[float, float]]:
    """Project a world point into a camera image. Returns (u, v) or None.

    World frame: Gazebo (X East, Y North, Z Up).
    NED frame: North = Y, East = X, Down = -Z.
    Body frame: X forward, Y right, Z down.
    Camera optical frame: X right (u), Y down (v), Z forward (depth).
    """
    dx = point_world[0] - cam_world[0]
    dy = point_world[1] - cam_world[1]
    dz = point_world[2] - cam_world[2]

    # Convert to NED:
    n = dy
    e = dx
    d = -dz

    # Rotate NED -> body frame by yaw, pitch, roll
    psi = math.radians(cam_yaw_deg)
    theta = math.radians(cam_pitch_deg)
    phi = math.radians(cam_roll_deg)

    # 1. Yaw around D:
    x1 =  n * math.cos(psi) + e * math.sin(psi)
    y1 = -n * math.sin(psi) + e * math.cos(psi)
    z1 =  d
    # 2. Pitch around Y1:
    x2 =  x1 * math.cos(theta) + z1 * math.sin(theta)
    y2 =  y1
    z2 = -x1 * math.sin(theta) + z1 * math.cos(theta)
    # 3. Roll around X2:
    x3 =  x2
    y3 =  y2 * math.cos(phi) + z2 * math.sin(phi)
    z3 = -y2 * math.sin(phi) + z2 * math.cos(phi)

    # Body to camera optical frame:
    # Camera looks along body +X (forward), image +u is body +Y (right), image +v is body +Z (down).
    x_cam = y3
    y_cam = z3
    z_cam = x3

    if z_cam <= 0.5:
        return None  # Behind or too close to camera

    u = intrinsics["fx"] * (x_cam / z_cam) + intrinsics["cx"]
    v = intrinsics["fy"] * (y_cam / z_cam) + intrinsics["cy"]
    return u, v
