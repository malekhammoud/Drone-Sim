"""Fleet: connect every asset, camera, track client and the sim clock.

    from arcticlib import Fleet
    fleet = Fleet.from_config()
    fleet.quad.takeoff(20)
    fleet.quad.goto(lat, lon, 20)
    f = fleet.frame("quadcopter")
    fleet.tracks.post("Sierra One", lat, lon)

Only the assets present in the roster are connected; missing ones are ``None``.
The reader threads auto-reconnect, and :meth:`on_reset` re-establishes
everything after the sim's Reset button recreates the containers.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from .camera import CameraSource
from .config import Config, load_config
from .simctl import SimClient
from .tracks import TrackClient
from .vehicle import Copter, Plane, Tower, Vehicle, connect_vehicle

log = logging.getLogger("arcticlib.fleet")

_CANON = {
    "quad": "quadcopter",
    "plane": "fixed-wing",
    "tower1": "tower-1",
    "tower2": "tower-2",
}


class Fleet:
    """Everything the rest of the system talks to, in one object."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.sim = SimClient(config)
        self.tracks = TrackClient(config.url(config.tracks_port))

        self.vehicles: dict[str, Vehicle] = {}
        self.cams: dict[str, CameraSource] = {}

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: Optional[Config] = None,
                    connect: bool = True) -> "Fleet":
        fleet = cls(config or load_config())
        if connect:
            fleet.connect_all()
        return fleet

    def _make(self, name: str) -> Optional[Vehicle]:
        spec = self.config.assets.get(name)
        if spec is None:
            return None
        return connect_vehicle(spec, self.config, sim_time_fn=self.sim.sim_time)

    def connect_all(self, warm_frames: float = 0.0) -> None:
        """Connect every asset and camera (idempotent)."""
        names = []
        for attr, name in _CANON.items():
            if name not in self.config.assets:
                setattr(self, attr, None)
                continue
            names.append(name)
            if name not in self.vehicles:
                self.vehicles[name] = self._make(name)
            spec = self.config.assets[name]
            if spec.camera is not None and name not in self.cams:
                self.cams[name] = CameraSource(spec.camera, self.sim.sim_time)
            setattr(self, attr, self.vehicles.get(name))

        # Warm the cameras so latest() is not empty on the first call.
        if warm_frames > 0:
            for cam in self.cams.values():
                cam.poll(rate_hz=warm_frames)

    # ------------------------------------------------------------------ #
    def wait_ready(self, timeout: float = 20.0) -> bool:
        """Wait until every connected vehicle has a live heartbeat."""
        import time
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.vehicles and all(v.connected for v in self.vehicles.values()):
                return True
            time.sleep(0.25)
        return bool(self.vehicles) and all(v.connected for v in self.vehicles.values())

    # -- accessors ----------------------------------------------------- #
    def vehicle(self, asset: str) -> Optional[Vehicle]:
        return self.vehicles.get(asset)

    def pose(self, asset: str):
        v = self.vehicles.get(asset)
        return v.pose() if v else None

    def pose_at(self, asset: str, t: float, clock: str = "sim"):
        v = self.vehicles.get(asset)
        return v.pose_at(t, clock=clock) if v else None

    def frame(self, asset: str, poll: bool = False, rate_hz: float = 2.0):
        """Latest camera frame for an asset, or None.

        ``poll=True`` lazily starts the background poller on first use.
        """
        cam = self.cams.get(asset)
        if cam is None:
            return None
        if poll:
            cam.poll(rate_hz)
        return cam.latest()

    def status(self) -> dict:
        return {
            "sim": self.sim.status(),
            "sim_time": self.sim.sim_time(),
            "assets": {n: v.status() for n, v in self.vehicles.items()},
            "cameras": {n: (c.latest() is not None) for n, c in self.cams.items()},
        }

    # -- lifecycle ----------------------------------------------------- #
    def on_reset(self, wait: float = 240.0) -> bool:
        """Wait for the fleet to come back after a Reset, then reconnect.

        The reader threads reconnect on their own; this just blocks until the
        control plane reports every rostered asset answering again.
        """
        log.info("fleet: waiting for the sim to come back after reset")
        self.sim.stop()
        self.sim.start()
        if not self.sim.wait_until_ready(timeout=wait, wait_for_reset=True):
            log.warning("fleet: sim did not become ready within %.0fs", wait)
            return False
        for name, vehicle in self.vehicles.items():
            if not vehicle.connected:
                vehicle.connect()
        self.wait_ready(timeout=30.0)
        for cam in self.cams.values():
            cam.poll(rate_hz=2.0)
        return True

    def shutdown(self) -> None:
        for v in self.vehicles.values():
            v.shutdown()
        for c in self.cams.values():
            c.stop()
        self.sim.stop()
