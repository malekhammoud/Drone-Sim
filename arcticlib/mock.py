"""A drop-in fake :class:`~arcticlib.fleet.Fleet` for algorithm iteration.

Same public surface — ``fleet.quad``, ``fleet.plane``, ``fleet.tower1/2``,
``fleet.cams``, ``fleet.tracks``, ``fleet.sim``, ``fleet.pose``,
``fleet.pose_at``, ``fleet.frame``, ``fleet.status``, ``fleet.shutdown`` — but
everything is synthetic, local and runs at whatever speed you like. Swap:

    from arcticlib import Fleet
    fleet = Fleet.from_config()

for

    from arcticlib.mock import MockFleet
    fleet = MockFleet()

and the CV/planner/tracker code above it does not change.

The world is a simple 2D strait: three aircraft/towers around a coastal origin
and one ship steaming up the channel. Cameras render a flat "sea" with the ship
drawn when it is inside the sensor FOV, so a detector has something to find.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np

from .config import Config, load_config
from .geo import Georef
from .types import AssetStatus, Battery, Frame, Pose

# Handy for tests: freeze the clock and step it by hand.
DEFAULT_TIMESTEP = 0.1


class _Clock:
    def __init__(self, speedup: float = 100.0) -> None:
        self.t0 = time.monotonic()
        self.speedup = speedup

    def sim_time(self) -> float:
        return (time.monotonic() - self.t0) * self.speedup


class MockVehicle:
    """Kinematic stand-in for :class:`~arcticlib.vehicle.Vehicle`."""

    kind = "generic"

    def __init__(self, name: str, kind: str, lat: float, lon: float,
                 georef: Georef, clock: _Clock, alt: float = 0.0) -> None:
        self.spec = type("S", (), {"name": name, "kind": kind, "sysid": 1})()
        self.name, self.kind = name, kind
        self.georef, self.clock = georef, clock
        self.lat, self.lon = lat, lon
        self.alt_rel = alt
        self.yaw = 0.0
        self.roll = self.pitch = 0.0
        self.vn = self.ve = self.vd = 0.0
        self.mode = "GUIDED"
        self.armed = kind == "copter"
        self.connected = True
        self.servo: dict[int, int] = {}
        self._target: tuple[float, float, float] | None = None
        self._poses: deque[Pose] = deque(maxlen=4096)
        self._max_speed = 20.0 if kind == "plane" else 8.0
        self._calib = None

    # -- telemetry ------------------------------------------------------ #
    def sim_time(self) -> float:
        return self.clock.sim_time()

    def pose(self) -> Pose:
        return Pose(self.name, self.sim_time(), time.monotonic(), self.lat, self.lon,
                    self.alt_rel, self.alt_rel + 100.0, self.roll, self.pitch,
                    self.yaw, self.vn, self.ve, self.vd)

    def pose_at(self, t: float, clock: str = "sim"):
        if not self._poses:
            return self.pose()
        key = (lambda p: p.t_sim) if clock == "sim" else (lambda p: p.t_wall)
        ps = list(self._poses)
        if t <= key(ps[0]):
            return ps[0]
        if t >= key(ps[-1]):
            return ps[-1]
        for a, b in zip(ps, ps[1:]):
            ta, tb = key(a), key(b)
            if ta <= t <= tb:
                f = 0.0 if tb == ta else (t - ta) / (tb - ta)
                return Pose(a.asset, a.t_sim + f * (b.t_sim - a.t_sim),
                            a.t_wall + f * (b.t_wall - a.t_wall),
                            a.lat + f * (b.lat - a.lat), a.lon + f * (b.lon - a.lon),
                            a.alt_rel + f * (b.alt_rel - a.alt_rel),
                            a.alt_amsl + f * (b.alt_amsl - a.alt_amsl),
                            a.roll, a.pitch, a.yaw, a.vx, a.vy, a.vz)
        return ps[-1]

    # -- commands ------------------------------------------------------- #
    def set_mode(self, name: str, timeout: float = 6.0) -> bool:
        self.mode = name.upper()
        return True

    def arm(self, timeout: float = 15.0) -> bool:
        self.armed = True
        return True

    def disarm(self, timeout: float = 5.0) -> bool:
        self.armed = False
        return True

    def takeoff(self, *a, **k) -> bool:
        self.armed = True
        self.mode = "GUIDED"
        self.alt_rel = float(a[0]) if a else 15.0
        return True

    def goto(self, lat: float, lon: float, alt: float, **k) -> bool:
        self._target = (lat, lon, alt)
        return True

    def land(self) -> bool:
        self.mode = "LAND"
        self.alt_rel = 0.0
        self.armed = False
        return True

    def rtl(self) -> bool:
        self.mode = "RTL"
        return True

    def set_speed(self, mps: float) -> bool:
        self._max_speed = mps
        return True

    def wait_alt(self, alt: float, timeout: float = 45.0) -> bool:
        return self.alt_rel >= alt

    def status(self) -> AssetStatus:
        return AssetStatus(self.name, True, self.mode, self.armed,
                           Battery(16.0, 5.0, 80.0), 0.0)

    def shutdown(self) -> None:
        self.connected = False

    # -- towers --------------------------------------------------------- #
    def set_pan_pwm(self, pwm: float, timeout: float = 4.0) -> bool:
        self.servo[1] = int(pwm)
        return True

    def set_tilt_pwm(self, pwm: float, timeout: float = 4.0) -> bool:
        self.servo[2] = int(pwm)
        return True

    def point(self, az_deg: float, el_deg: float) -> bool:
        self.set_pan_pwm(1500 + az_deg * (800 / 360))
        self.set_tilt_pwm(1420 + el_deg * (800 / 75))
        return True

    def aim_at(self, lat: float, lon: float, alt: float = 0.0) -> bool:
        from .geo import bearing_deg
        return self.point(bearing_deg(self.lat, self.lon, lat, lon), 0.0)

    def scan(self) -> bool:
        self.mode = "SCAN"
        return True

    def stop_scan(self) -> bool:
        self.mode = "MANUAL"
        return True

    def center(self) -> bool:
        return self.point(0.0, 0.0)

    def load_calibration(self, path=None) -> dict:
        self._calib = {"pan": {"center_pwm": 1500, "pwm_per_deg": 800 / 360,
                               "base_yaw_deg": 0.0},
                       "tilt": {"zero_deg_pwm": 1420, "pwm_per_deg": 800 / 75}}
        return self._calib

    # -- simulation ----------------------------------------------------- #
    def update(self, dt: float) -> None:
        if self._target is not None:
            tlat, tlon, talt = self._target
            e, n, _ = self.georef.to_enu(self.lat, self.lon)
            te, tn, _ = self.georef.to_enu(tlat, tlon)
            de, dn = te - e, tn - n
            dist = math.hypot(de, dn)
            step = self._max_speed * dt
            if dist <= max(step, 1.0):
                self.lat, self.lon = tlat, tlon
                self._target = None
            else:
                self.lat, self.lon, _ = self.georef.to_latlon(
                    e + de / dist * step, n + dn / dist * step)
                self.yaw = math.atan2(de, dn)
            self.alt_rel += max(-2.0, min(2.0, talt - self.alt_rel)) * 0.5
            self.vn, self.ve = (self._max_speed * dn / dist,
                                self._max_speed * de / dist)
        else:
            self.vn = self.ve = 0.0
        if self.kind == "tower" and self.mode == "SCAN":
            # gentle scan so frames change over time
            self.servo[1] = 1500 + int(400 * math.sin(self.clock.sim_time() * 0.5))
        self._poses.append(self.pose())


class MockCamera:
    """Renders a flat sea and, when in view, the ship."""

    def __init__(self, spec, fleet) -> None:
        self.spec = spec
        self.fleet = fleet
        self._latest: Frame | None = None

    def _render(self) -> Frame:
        w, h = self.spec.width, self.spec.height
        img = np.zeros((h, w, 3), np.uint8)
        img[:, :] = (120, 90, 40)                      # BGR sea
        img[: h // 3, :] = (170, 140, 90)              # sky
        v = self.fleet.vehicle(self.spec.asset)
        if v is not None:
            ship = self.fleet._ship_latlon()
            if ship:
                from .geo import bearing_deg, distance_m
                brg = bearing_deg(v.lat, v.lon, ship[0], ship[1])
                rel = (brg - math.degrees(v.yaw) + 540) % 360 - 180
                if abs(rel) < self.spec.hfov_deg / 2:
                    u = int(w / 2 + math.tan(math.radians(rel)) /
                            math.tan(math.radians(self.spec.hfov_deg / 2)) * w / 2)
                    size = max(3, int(20000 / max(distance_m(v.lat, v.lon, *ship), 50)))
                    img[max(0, h // 2 - size):h // 2 + size,
                        max(0, u - size):u + size] = (40, 40, 40)
        return Frame(self.spec.asset, self.fleet.sim.sim_time(), time.monotonic(),
                     img, w, h, self.spec.hfov_deg, self.spec.vfov_deg)

    def grab(self) -> Frame:
        self._latest = self._render()
        return self._latest

    def latest(self) -> Frame | None:
        if self._latest is None:
            self.grab()
        return self._latest

    def poll(self, rate_hz: float = 2.0) -> None:
        self.grab()

    def stream(self):
        while True:
            yield self.grab()

    def stop(self) -> None:
        pass


class MockTrackClient:
    def __init__(self) -> None:
        self._tracks: dict[str, dict] = {}

    def post(self, name, lat, lon, heading=None, speed=None):
        created = name not in self._tracks
        t = self._tracks.setdefault(name, {"name": name, "uuid": f"entity-{abs(hash(name))%10**10:x}",
                                           "fixes": 0})
        t.update({"lat": lat, "lon": lon, "heading": heading, "speed": speed})
        t["fixes"] += 1
        return {"ok": True, "created": created, **t}

    def list(self):
        return list(self._tracks.values())


class MockSim:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.clocks = 1

    def sim_time(self) -> float:
        return self.clock.sim_time()

    def status(self) -> dict:
        return {"state": "idle", "detail": "mock"}

    def assets(self):
        return []

    def reset(self) -> bool:
        self.clock.t0 = time.monotonic()
        return True

    def wait_until_ready(self, timeout: float = 5.0, poll: float = 0.1,
                         wait_for_reset: bool = False) -> bool:
        return True

    def stop(self) -> None:
        pass

    def world_stats(self) -> dict:
        return {"sim_time": {"sec": int(self.sim_time()), "nsec": 0}}


class MockFleet:
    """Drop-in replacement for :class:`~arcticlib.fleet.Fleet`."""

    def __init__(self, config: Config | None = None, speedup: float = 100.0) -> None:
        self.config = config or load_config()
        self.georef = Georef(self.config.origin_lat, self.config.origin_lon,
                             ps_centre_x=self.config.ps_centre_x,
                             ps_centre_y=self.config.ps_centre_y)
        self.clock = _Clock(speedup)
        self.sim = MockSim(self.clock)
        self.tracks = MockTrackClient()
        self._lock = threading.Lock()
        self._stop = threading.Event()

        lat0, lon0 = self.config.origin_lat, self.config.origin_lon
        self.vehicles = {
            "quadcopter": MockVehicle("quadcopter", "copter", lat0 + 0.004,
                                      lon0 - 0.017, self.georef, self.clock, 40.0),
            "fixed-wing": MockVehicle("fixed-wing", "plane", lat0 + 0.006,
                                      lon0 - 0.020, self.georef, self.clock, 120.0),
            "tower-1": MockVehicle("tower-1", "tower", lat0 - 0.011,
                                   lon0 - 0.031, self.georef, self.clock),
            "tower-2": MockVehicle("tower-2", "tower", lat0 + 0.020,
                                   lon0 + 0.018, self.georef, self.clock),
        }
        self.quad = self.vehicles["quadcopter"]
        self.plane = self.vehicles["fixed-wing"]
        self.tower1 = self.vehicles["tower-1"]
        self.tower2 = self.vehicles["tower-2"]

        for t in (self.tower1, self.tower2):
            t.load_calibration()

        self.cams = {name: MockCamera(self.config.assets[name].camera, self)
                     for name in self.vehicles if self.config.assets.get(name)
                     and self.config.assets[name].camera}

        # the ship steams north-east across the strait
        self._ship = [lat0 + 0.010, lon0 - 0.055]
        self._ship_dir = 65.0
        self._ship_speed = 0.0006     # deg/s-ish, scaled by speedup
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # -- simulation loop ------------------------------------------------ #
    def _loop(self) -> None:
        dt = DEFAULT_TIMESTEP
        while not self._stop.is_set():
            with self._lock:
                for v in self.vehicles.values():
                    v.update(dt)
                rad = math.radians(self._ship_dir)
                # move in the local tangent plane, then back to lat/lon
                e, n, _ = self.georef.to_enu(self._ship[0], self._ship[1])
                self._ship = self.georef.to_latlon(
                    e + math.sin(rad) * self._ship_speed * 200 * dt,
                    n + math.cos(rad) * self._ship_speed * 200 * dt)[:2]
            time.sleep(dt / max(self.clock.speedup / 10.0, 1.0))

    def _ship_latlon(self):
        return tuple(self._ship)

    # -- Fleet API ------------------------------------------------------ #
    def vehicle(self, asset: str):
        return self.vehicles.get(asset)

    def pose(self, asset: str):
        v = self.vehicles.get(asset)
        return v.pose() if v else None

    def pose_at(self, asset: str, t: float, clock: str = "sim"):
        v = self.vehicles.get(asset)
        return v.pose_at(t, clock=clock) if v else None

    def frame(self, asset: str, poll: bool = False, rate_hz: float = 2.0):
        cam = self.cams.get(asset)
        return cam.latest() if cam else None

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return True

    def on_reset(self, wait: float = 5.0) -> bool:
        self.sim.reset()
        return True

    def status(self) -> dict:
        return {"sim": self.sim.status(), "sim_time": self.sim.sim_time(),
                "assets": {n: v.status() for n, v in self.vehicles.items()},
                "cameras": {n: True for n in self.cams}}

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
