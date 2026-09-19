"""MAVLink vehicles: connection, telemetry, pose history and commands.

Design notes
------------
* **Threads, not asyncio.** ``pymavlink`` is blocking, so each vehicle owns one
  background reader thread. That thread is the only reader; it keeps a
  thread-safe latest snapshot and a ring buffer of the last ``pose_history_s``
  seconds of :class:`~arcticlib.types.Pose`, which is what lets a camera frame
  be matched to the vehicle's pose at the frame's instant.
* **Never raise into the caller.** Every command catches, logs and returns
  ``False`` on failure, and waits for a ``COMMAND_ACK`` with a timeout.
* **Auto-reconnect.** The sim's Reset recreates every container, so the reader
  watchdog notices a stale heartbeat, reopens the link and re-requests streams.
* Endpoints are ``udpout`` because the sim's are ``udpin`` listeners: they stay
  mute until we transmit (see RECON.md §4).
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Callable, List, Optional

from pymavlink import mavutil

from .config import AssetSpec, Config, load_config
from .types import AssetStatus, Battery, Pose

log = logging.getLogger("arcticlib.vehicle")

M = mavutil.mavlink

# ArduPilot custom mode numbers (RECON.md §4). Kept explicit rather than asked
# of mavutil.mode_mapping(), which needs a heartbeat first and differs per type.
COPTER_MODES = {"STABILIZE": 0, "ALT_HOLD": 2, "AUTO": 3, "GUIDED": 4,
                "LOITER": 5, "RTL": 6, "CIRCLE": 7, "LAND": 9, "POSHOLD": 16,
                "BRAKE": 17, "SMART_RTL": 21}
PLANE_MODES = {"MANUAL": 0, "CIRCLE": 1, "STABILIZE": 2, "FBWA": 5, "FBWB": 6,
               "CRUISE": 7, "AUTO": 10, "RTL": 11, "LOITER": 12, "TAKEOFF": 13,
               "GUIDED": 15}
TRACKER_MODES = {"MANUAL": 0, "STOP": 1, "SCAN": 2, "SERVO_TEST": 3, "AUTO": 4}

MODE_MAPS = {"copter": COPTER_MODES, "plane": PLANE_MODES, "tower": TRACKER_MODES,
             "rover": {"MANUAL": 0, "HOLD": 4, "AUTO": 3, "GUIDED": 15, "RTL": 11}}


def _wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class Vehicle:
    """Base class: link management, telemetry, pose history, command plumbing."""

    kind = "generic"

    def __init__(self, spec: AssetSpec, config: Optional[Config] = None,
                 sim_time_fn: Optional[Callable[[], float]] = None,
                 auto_connect: bool = True) -> None:
        self.spec = spec
        self.config = config or load_config()
        self._sim_time_fn = sim_time_fn

        self._mav = None
        self._lock = threading.RLock()
        self._ack_cond = threading.Condition()
        self._acks: dict[int, int] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # latest telemetry
        self.mode = "?"
        self.armed = False
        self.lat = 0.0
        self.lon = 0.0
        self.alt_rel = 0.0
        self.alt_amsl = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self.vn = self.ve = self.vd = 0.0
        self._last_hb = 0.0
        self._sys_time_s: Optional[float] = None
        self._battery = Battery()
        self.servo: dict[int, int] = {}          # 1-based channel -> raw PWM
        self.params: dict[str, float] = {}        # PARAM_VALUE cache
        self.statustexts: deque[tuple[float, str]] = deque(maxlen=50)

        self._poses: deque[Pose] = deque(maxlen=4096)

        if auto_connect:
            self.connect()

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    @property
    def connected(self) -> bool:
        with self._lock:
            return (self._mav is not None and
                    time.monotonic() - self._last_hb < self.config.heartbeat_timeout)

    def connect(self) -> bool:
        """Open the link and start the reader thread. Idempotent, never raises."""
        self._stop.clear()
        if not self._open():
            log.warning("%s: initial connect failed; reader will keep retrying",
                        self.spec.name)
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._reader_loop,
                                            name=f"mav-{self.spec.name}", daemon=True)
            self._thread.start()
        return self.connected

    def _open(self) -> bool:
        try:
            mav = mavutil.mavlink_connection(self.spec.master, source_system=255,
                                             source_component=190)
        except Exception as exc:                       # pragma: no cover - env
            log.warning("%s: connect error: %s", self.spec.name, exc)
            return False
        deadline = time.monotonic() + 10.0
        hb = None
        while hb is None and time.monotonic() < deadline:
            try:
                mav.mav.heartbeat_send(M.MAV_TYPE_GCS, M.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                hb = mav.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            except Exception:
                time.sleep(0.2)
        if hb is None:
            return False
        mav.target_system = hb.get_srcSystem()
        mav.target_component = hb.get_srcComponent()
        with self._lock:
            self._mav = mav
            self._last_hb = time.monotonic()
        self._on_message(hb)
        self._request_rates()
        log.info("%s: connected (sysid %d, mode %s)", self.spec.name,
                 mav.target_system, self.mode)
        return True

    def _on_reconnect(self) -> None:
        """Hook for subclasses that must re-send state after a link drop."""

    def _request_rates(self) -> None:
        """Ask for the streams we actually use via SET_MESSAGE_INTERVAL."""
        wanted = [
            ("ATTITUDE", 20),
            ("GLOBAL_POSITION_INT", 10),
            ("LOCAL_POSITION_NED", 10),
            ("SYS_STATUS", 2),
            ("BATTERY_STATUS", 1),
            ("SERVO_OUTPUT_RAW", 5),
            ("SYSTEM_TIME", 1),
        ]
        for name, hz in wanted:
            msg_id = getattr(M, f"MAVLINK_MSG_ID_{name}", None)
            if msg_id is None:
                continue
            self._send_command(M.MAV_CMD_SET_MESSAGE_INTERVAL,
                               [msg_id, int(1e6 / hz), 0, 0, 0, 0, 0], wait=False)

    # ------------------------------------------------------------------ #
    # Reader thread
    # ------------------------------------------------------------------ #
    def _reader_loop(self) -> None:
        last_hb_sent = 0.0
        while not self._stop.is_set():
            mav = self._mav
            if mav is None:
                time.sleep(1.0)
                self._open()
                self._on_reconnect()
                continue
            try:
                msg = mav.recv_match(blocking=True, timeout=1)
            except Exception as exc:                   # pragma: no cover - env
                log.warning("%s: recv error: %s", self.spec.name, exc)
                self._drop()
                continue

            now = time.monotonic()
            if msg is not None:
                try:
                    self._on_message(msg)
                except Exception as exc:               # pragma: no cover
                    log.debug("%s: message handling error: %s", self.spec.name, exc)

            if now - last_hb_sent >= 1.0:
                self._send_heartbeat()
                last_hb_sent = now

            if now - self._last_hb > self.config.heartbeat_timeout:
                log.warning("%s: heartbeat stale, reconnecting", self.spec.name)
                self._drop()

    def _drop(self) -> None:
        with self._lock:
            mav, self._mav = self._mav, None
            self._last_hb = 0.0
        try:
            if mav is not None:
                mav.close()
        except Exception:
            pass
        time.sleep(0.5)
        if self._open():
            self._on_reconnect()

    def _on_message(self, msg) -> None:
        t = msg.get_type()
        if t == "BAD_DATA":
            return
        if msg.get_srcSystem() != self._mav.target_system if self._mav else False:
            # Only trust our vehicle's messages.
            return
        now = time.monotonic()
        with self._lock:
            if t == "HEARTBEAT":
                self._last_hb = now
                self.armed = bool(msg.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)
                # Prefer our explicit per-type map: mavutil.mode_string_v10 is
                # copter-centric and prints "Mode(3)" for tracker modes.
                name = None
                for candidate, number in MODE_MAPS.get(self.kind, {}).items():
                    if number == msg.custom_mode:
                        name = candidate
                        break
                self.mode = name or mavutil.mode_string_v10(msg)
            elif t == "ATTITUDE":
                self.roll, self.pitch, self.yaw = msg.roll, msg.pitch, msg.yaw
            elif t == "GLOBAL_POSITION_INT":
                self.lat = msg.lat / 1e7
                self.lon = msg.lon / 1e7
                self.alt_amsl = msg.alt / 1000.0
                self.alt_rel = msg.relative_alt / 1000.0
                self.vn, self.ve, self.vd = msg.vx / 100.0, msg.vy / 100.0, msg.vz / 100.0
                self._append_pose(now)
            elif t == "SYSTEM_TIME":
                self._sys_time_s = msg.time_boot_ms / 1000.0
            elif t == "SYS_STATUS":
                self._battery.voltage_v = msg.voltage_battery / 1000.0
                self._battery.current_a = msg.current_battery / 100.0
                if msg.battery_remaining >= 0:
                    self._battery.remaining_pct = float(msg.battery_remaining)
            elif t == "BATTERY_STATUS":
                if msg.battery_remaining >= 0:
                    self._battery.remaining_pct = float(msg.battery_remaining)
            elif t == "SERVO_OUTPUT_RAW":
                self.servo = {
                    i: getattr(msg, f"servo{i}_raw")
                    for i in range(1, 17)
                    if getattr(msg, f"servo{i}_raw", 0)
                }
            elif t == "PARAM_VALUE":
                self.params[msg.param_id] = msg.param_value
            elif t == "STATUSTEXT":
                self.statustexts.append((now, msg.text))
        if t == "COMMAND_ACK":
            with self._ack_cond:
                self._acks[msg.command] = msg.result
                self._ack_cond.notify_all()

    # ------------------------------------------------------------------ #
    # Pose history
    # ------------------------------------------------------------------ #
    def sim_time(self) -> float:
        """Best available sim clock: injected global clock, else SYSTEM_TIME."""
        if self._sim_time_fn is not None:
            try:
                return self._sim_time_fn()
            except Exception:
                pass
        with self._lock:
            return self._sys_time_s if self._sys_time_s is not None else time.monotonic()

    def _append_pose(self, now: float) -> None:
        pose = Pose(
            asset=self.spec.name, t_sim=self.sim_time(), t_wall=now,
            lat=self.lat, lon=self.lon, alt_rel=self.alt_rel, alt_amsl=self.alt_amsl,
            roll=self.roll, pitch=self.pitch, yaw=self.yaw,
            vx=self.vn, vy=self.ve, vz=self.vd,
        )
        self._poses.append(pose)
        cutoff = now - self.config.pose_history_s
        while self._poses and self._poses[0].t_wall < cutoff:
            self._poses.popleft()

    def pose(self) -> Pose:
        """Latest pose snapshot."""
        with self._lock:
            if self._poses:
                return self._poses[-1]
            return Pose(self.spec.name, self.sim_time(), time.monotonic(),
                        self.lat, self.lon, self.alt_rel, self.alt_amsl,
                        self.roll, self.pitch, self.yaw, self.vn, self.ve, self.vd)

    def pose_at(self, t: float, clock: str = "sim") -> Optional[Pose]:
        """Linearly interpolated pose at time ``t`` on ``clock`` ('sim'|'wall').

        Returns the nearest pose when ``t`` is outside the buffer instead of
        None, so callers do not have to special-case the very first frames.
        """
        with self._lock:
            poses: List[Pose] = list(self._poses)
        if not poses:
            return None
        key = (lambda p: p.t_sim) if clock == "sim" else (lambda p: p.t_wall)
        poses.sort(key=key)
        if t <= key(poses[0]):
            return poses[0]
        if t >= key(poses[-1]):
            return poses[-1]
        for a, b in zip(poses, poses[1:]):
            ta, tb = key(a), key(b)
            if ta <= t <= tb:
                f = 0.0 if tb == ta else (t - ta) / (tb - ta)
                return Pose(
                    asset=a.asset, t_sim=a.t_sim + f * (b.t_sim - a.t_sim),
                    t_wall=a.t_wall + f * (b.t_wall - a.t_wall),
                    lat=a.lat + f * (b.lat - a.lat),
                    lon=a.lon + f * (b.lon - a.lon),
                    alt_rel=a.alt_rel + f * (b.alt_rel - a.alt_rel),
                    alt_amsl=a.alt_amsl + f * (b.alt_amsl - a.alt_amsl),
                    roll=a.roll + f * _wrap_pi(b.roll - a.roll),
                    pitch=a.pitch + f * (b.pitch - a.pitch),
                    yaw=a.yaw + f * _wrap_pi(b.yaw - a.yaw),
                    vx=a.vx + f * (b.vx - a.vx),
                    vy=a.vy + f * (b.vy - a.vy),
                    vz=a.vz + f * (b.vz - a.vz),
                )
        return poses[-1]

    # ------------------------------------------------------------------ #
    # Command plumbing
    # ------------------------------------------------------------------ #
    def _send_heartbeat(self) -> None:
        mav = self._mav
        if mav is None:
            return
        try:
            mav.mav.heartbeat_send(M.MAV_TYPE_GCS, M.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        except Exception:
            pass

    def _send_command(self, command: int, params: list, wait: bool = True,
                      timeout: Optional[float] = None) -> bool:
        """COMMAND_LONG with optional ACK wait. Returns success, never raises."""
        mav = self._mav
        if mav is None:
            return False
        timeout = timeout if timeout is not None else self.config.command_timeout
        try:
            with self._ack_cond:
                self._acks.pop(command, None)
            mav.mav.command_long_send(mav.target_system, mav.target_component,
                                      command, 0, *params)
        except Exception as exc:
            log.warning("%s: send cmd %d failed: %s", self.spec.name, command, exc)
            return False
        if not wait:
            return True
        end = time.monotonic() + timeout
        with self._ack_cond:
            while time.monotonic() < end:
                result = self._acks.get(command)
                if result is not None:
                    if result == M.MAV_RESULT_IN_PROGRESS:
                        self._acks.pop(command, None)
                        self._ack_cond.wait(min(0.5, max(0.0, end - time.monotonic())))
                        continue
                    return result in (M.MAV_RESULT_ACCEPTED,)
                self._ack_cond.wait(min(0.2, max(0.0, end - time.monotonic())))
        log.info("%s: no ACK for cmd %d within %.1fs", self.spec.name, command, timeout)
        return False

    def _set_param(self, name: str, value: float, timeout: float = 3.0) -> bool:
        """PARAM_SET. Fire-and-forget; ArduPilot confirms via PARAM_VALUE echo."""
        mav = self._mav
        if mav is None:
            return False
        try:
            mav.mav.param_set_send(mav.target_system, mav.target_component,
                                   name.encode(), float(value),
                                   M.MAV_PARAM_TYPE_REAL32)
        except Exception as exc:
            log.warning("%s: param set %s failed: %s", self.spec.name, name, exc)
            return False
        return True

    def send_global_target(self, lat: float, lon: float, alt: float,
                           yaw: Optional[float] = None, repeats: int = 5) -> bool:
        """Stream a GUIDED position target (relative altitude) and return sent.

        This is what MAVProxy's ``guided <lat> <lon> <alt>`` uses, and it is the
        only thing this sim accepts: ``MAV_CMD_DO_REPOSITION`` answers
        ``MAV_RESULT_UNSUPPORTED`` on both copter and plane (verified). MAVLink
        gives no ACK for a setpoint, so "success" means the message was sent
        without error — callers that need proof should watch the pose.
        """
        mav = self._mav
        if mav is None:
            return False
        mask = (M.POSITION_TARGET_TYPEMASK_VX_IGNORE
                | M.POSITION_TARGET_TYPEMASK_VY_IGNORE
                | M.POSITION_TARGET_TYPEMASK_VZ_IGNORE
                | M.POSITION_TARGET_TYPEMASK_AX_IGNORE
                | M.POSITION_TARGET_TYPEMASK_AY_IGNORE
                | M.POSITION_TARGET_TYPEMASK_AZ_IGNORE
                | M.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)
        if yaw is None:
            mask |= M.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        try:
            for _ in range(max(1, repeats)):
                mav.mav.set_position_target_global_int_send(
                    int((time.monotonic() * 1000)) & 0xFFFFFFFF,
                    mav.target_system, mav.target_component,
                    M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, mask,
                    int(lat * 1e7), int(lon * 1e7), float(alt),
                    0, 0, 0, 0, 0, 0,
                    float(yaw or 0.0), 0)
                time.sleep(0.1)
        except Exception as exc:
            log.warning("%s: global target send failed: %s", self.spec.name, exc)
            return False
        return True

    def get_param(self, name: str, timeout: float = 3.0) -> Optional[float]:
        """Fetch a parameter. The reader thread caches PARAM_VALUE replies."""
        name = name.upper()
        if name in self.params:
            return self.params[name]
        mav = self._mav
        if mav is None:
            return None
        try:
            mav.mav.param_request_read_send(mav.target_system, mav.target_component,
                                            name.encode(), -1)
        except Exception:
            return None
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if name in self.params:
                return self.params[name]
            time.sleep(0.05)
        return None

    def mode_number(self, name: str) -> Optional[int]:
        return MODE_MAPS.get(self.kind, {}).get(name.upper())

    def set_mode(self, name: str, timeout: float = 6.0) -> bool:
        """Switch flight mode and confirm via HEARTBEAT. Returns success."""
        target = self.mode_number(name)
        if target is None:
            log.warning("%s: unknown mode %s", self.spec.name, name)
            return False
        if self.mode.upper() == name.upper():
            return True
        ok = self._send_command(M.MAV_CMD_DO_SET_MODE,
                                [M.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, target,
                                 0, 0, 0, 0, 0], timeout=timeout)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.mode.upper() == name.upper():
                return True
            time.sleep(0.1)
        return ok and self.mode.upper() == name.upper()

    def arm(self, timeout: float = 60.0) -> bool:
        """Arm, retrying while pre-arm checks settle. Returns armed state.

        The default is generous on purpose: straight after a sim Reset the EKF
        can take a minute to become armable, and a planner that gives up early
        will simply never take off.
        """
        if self.armed:
            return True
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self._send_command(M.MAV_CMD_COMPONENT_ARM_DISARM, [1, 0, 0, 0, 0, 0, 0],
                               timeout=2.0)
            if self.armed:
                return True
            time.sleep(0.5)
        return self.armed

    def disarm(self, timeout: float = 5.0) -> bool:
        self._send_command(M.MAV_CMD_COMPONENT_ARM_DISARM, [0, 0, 0, 0, 0, 0, 0],
                           timeout=timeout)
        return not self.armed

    def status(self) -> AssetStatus:
        return AssetStatus(asset=self.spec.name, connected=self.connected,
                           mode=self.mode, armed=self.armed,
                           battery=self._battery,
                           last_heartbeat_age=time.monotonic() - self._last_hb)

    def shutdown(self) -> None:
        """Stop the reader and close the link."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        with self._lock:
            mav, self._mav = self._mav, None
        if mav is not None:
            try:
                mav.close()
            except Exception:
                pass


class Copter(Vehicle):
    """ArduCopter: hover, GUIDED goto, takeoff/land."""

    kind = "copter"

    def takeoff(self, alt: float = 15.0, timeout: float = 120.0) -> bool:
        """GUIDED -> arm -> TAKEOFF -> climb, retrying until it works.

        Straight after a sim Reset the EKF needs ~1-2 min to converge. The
        autopilot will happily accept `arm` and `NAV_TAKEOFF` during that window
        but hold the motors at idle and then **auto-disarm** ("Disarming
        motors"), so a single attempt looks like a rejected takeoff. We therefore
        loop: if the vehicle disarms without climbing, re-arm and try again until
        the whole ``timeout`` is spent. The arm state only lasts ~3 s, so the
        takeoff command follows the arm immediately.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.set_mode("GUIDED", timeout=4.0):
                time.sleep(1.0)
                continue
            if not self.armed:
                remaining = max(5.0, deadline - time.monotonic())
                if not self.arm(timeout=min(60.0, remaining)):
                    time.sleep(1.0)
                    continue
            if not self._send_command(M.MAV_CMD_NAV_TAKEOFF,
                                      [0, 0, 0, 0, 0, 0, alt], timeout=6.0):
                time.sleep(1.0)
                continue
            climb_until = min(deadline, time.monotonic() + 30.0)
            while time.monotonic() < climb_until:
                if self.alt_rel >= alt * 0.8:
                    return True
                if not self.armed:
                    log.info("%s: auto-disarmed during takeoff; retrying "
                             "(EKF still settling?)", self.spec.name)
                    break
                time.sleep(0.2)
            if self.alt_rel >= alt * 0.8:
                return True
        log.warning("%s: takeoff did not climb within %.0fs", self.spec.name, timeout)
        return False

    def wait_alt(self, alt_rel: float, timeout: float = 45.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.alt_rel >= alt_rel:
                return True
            time.sleep(0.2)
        return self.alt_rel >= alt_rel

    def goto(self, lat: float, lon: float, alt: float, timeout: float = 6.0) -> bool:
        """Fly to a point and hold. Uses the unacknowledged global setpoint."""
        if self.mode.upper() != "GUIDED" and not self.set_mode("GUIDED"):
            return False
        return self.send_global_target(lat, lon, alt)

    def land(self) -> bool:
        return self.set_mode("LAND")

    def rtl(self) -> bool:
        return self.set_mode("RTL")

    def set_speed(self, mps: float) -> bool:
        """Set horizontal cruise speed via WPNAV_SPEED (cm/s)."""
        return self._set_param("WPNAV_SPEED", max(0.0, mps) * 100.0)

    def has_gimbal(self) -> bool:
        """False on this sim: copter.parm sets MNT1_TYPE 0 (RECON.md §4)."""
        return False

    def gimbal(self, pitch_deg: Optional[float] = None,
               yaw_deg: Optional[float] = None) -> bool:
        log.info("%s: no gimbal configured (MNT1_TYPE 0)", self.spec.name)
        return False


class Plane(Vehicle):
    """ArduPlane: cannot hover; GUIDED goto loiters around the point."""

    kind = "plane"

    def takeoff(self, alt: float = 80.0, timeout: float = 120.0) -> bool:
        """GUIDED -> arm -> mode TAKEOFF, retrying while the EKF settles.

        Like the copter, a fresh reset leaves pre-arm checks unhappy for a
        while; the plane will not arm until they pass, so retry until timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.set_mode("GUIDED", timeout=4.0):
                time.sleep(1.0)
                continue
            if not self.armed:
                remaining = max(5.0, deadline - time.monotonic())
                if not self.arm(timeout=min(60.0, remaining)):
                    time.sleep(1.0)
                    continue
            if self.set_mode("TAKEOFF", timeout=6.0):
                return True
            time.sleep(1.0)
        return False

    def goto(self, lat: float, lon: float, alt: float, timeout: float = 6.0) -> bool:
        """Fly to / loiter around (lat, lon) at ``alt``.

        ArduPlane SITL requires MAV_CMD_DO_REPOSITION via command_int_send
        (it ignores set_position_target_global_int in GUIDED mode).
        """
        if self.mode.upper() not in ("GUIDED", "AUTO") and not self.set_mode("GUIDED"):
            return False
        mav = self._mav
        if mav is None:
            return False
        try:
            mav.mav.command_int_send(
                self.spec.sysid, 1,
                M.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                M.MAV_CMD_DO_REPOSITION,
                0, 0,
                -1,  # p1: ground speed (-1 is use-default)
                M.MAV_DO_REPOSITION_FLAGS_CHANGE_MODE,  # p2: flags
                0,   # p3: loiter radius (0 is default)
                0,   # p4: yaw
                int(lat * 1e7),
                int(lon * 1e7),
                float(alt)
            )
            return True
        except Exception as exc:
            log.warning("%s: DO_REPOSITION failed: %s", self.spec.name, exc)
            return False

    def loiter(self, lat: float, lon: float, alt: float,
               radius: Optional[float] = None) -> bool:
        """GUIDED goto plus an optional loiter radius (WP_LOITER_RAD, metres)."""
        if radius is not None:
            self._set_param("WP_LOITER_RAD", abs(radius))
        return self.goto(lat, lon, alt)

    def set_airspeed(self, mps: float) -> bool:
        """Set cruise airspeed, m/s. DO_CHANGE_SPEED then AIRSPEED_CRUISE."""
        ok = self._send_command(M.MAV_CMD_DO_CHANGE_SPEED,
                                [0, max(0.0, mps), -1, 0, 0, 0, 0])
        self._set_param("AIRSPEED_CRUISE", max(0.0, mps))
        return ok

    def rtl(self) -> bool:
        return self.set_mode("RTL")


class Tower(Vehicle):
    """ArduPilot AntennaTracker mast: servo 1 = pan, servo 2 = tilt.

    The sim's plugin maps a 0..1 servo command to a joint angle as
    ``multiplier * (offset + cmd)`` (RECON.md §4 / terrain/tower.py). ArduPilot
    normalises a PWM to 0..1 over the **channel's own min..max**, and the
    tracker's servos are pinned to **1100..1900 us**, not 1000..2000 (verified:
    commanding 1000/2000 yields SERVO_OUTPUT_RAW 1100/1900). With
    ``SERVO1/2_MIN=1100, MAX=1900`` that gives, exactly:

    * pan:  1500 us = 0 deg, +2.2222 us/deg  (1100 = -180, 1900 = +180)
    * tilt: 1420 us = 0 deg elevation, +10.667 us/deg
            (1100 = -30 deg, 1900 = +45 deg)

    Those are the defaults; ``calib/tower_<name>.json`` overrides them, and
    ``tools/calibrate_tower.py`` writes that file after verifying the loop.
    """

    kind = "tower"

    def __init__(self, *args, calib_dir: str = "calib", **kwargs) -> None:
        self._calib: dict = {}
        self._calib_dir = calib_dir
        super().__init__(*args, **kwargs)

    def _on_reconnect(self) -> None:
        pass

    # -- calibration ---------------------------------------------------- #
    def load_calibration(self, path: Optional[str] = None) -> dict:
        import json
        import os
        if path is None:
            path = os.path.join(self._calib_dir, f"tower_{self.spec.name}.json")
        default = {
            "pan": {"center_pwm": 1500.0, "pwm_per_deg": 800.0 / 360.0, "sign": 1,
                    "min_pwm": 1100, "max_pwm": 1900, "base_yaw_deg": 0.0},
            "tilt": {"zero_deg_pwm": 1420.0, "pwm_per_deg": 800.0 / 75.0, "sign": 1,
                     "min_pwm": 1100, "max_pwm": 1900},
        }
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    loaded = json.load(fh)
                default["pan"].update(loaded.get("pan", {}))
                default["tilt"].update(loaded.get("tilt", {}))
            except Exception as exc:
                log.warning("%s: bad calibration %s: %s", self.spec.name, path, exc)
        self._calib = default
        return default

    # -- raw servo control ---------------------------------------------- #
    def set_pan_pwm(self, pwm: float, timeout: float = 4.0) -> bool:
        return self._send_command(M.MAV_CMD_DO_SET_SERVO,
                                  [1, int(round(pwm)), 0, 0, 0, 0, 0], timeout=timeout)

    def set_tilt_pwm(self, pwm: float, timeout: float = 4.0) -> bool:
        return self._send_command(M.MAV_CMD_DO_SET_SERVO,
                                  [2, int(round(pwm)), 0, 0, 0, 0, 0], timeout=timeout)

    # -- angle control -------------------------------------------------- #
    def point(self, az_deg: float, el_deg: float) -> bool:
        """Aim the camera: azimuth and elevation in degrees, local frame.

        ``az_deg`` is relative to the tower's zero pan (pwm 1500); ``el_deg``
        is elevation above horizontal, positive up.
        """
        if not self._calib:
            self.load_calibration()
        pan = self._calib["pan"]
        tilt = self._calib["tilt"]
        pan_pwm = pan["center_pwm"] + pan["sign"] * pan["pwm_per_deg"] * az_deg
        tilt_pwm = tilt["zero_deg_pwm"] + tilt["sign"] * tilt["pwm_per_deg"] * el_deg
        pan_pwm = max(pan["min_pwm"], min(pan["max_pwm"], pan_pwm))
        tilt_pwm = max(tilt["min_pwm"], min(tilt["max_pwm"], tilt_pwm))
        return self.set_pan_pwm(pan_pwm) and self.set_tilt_pwm(tilt_pwm)

    def aim_at(self, lat: float, lon: float, alt: float = 0.0) -> bool:
        """Point at a lat/lon using this tower's own position.

        Requires the base yaw (world orientation of pan pwm 1500) to be known;
        it comes from ``calib``'s ``base_yaw_deg`` and defaults to 0, which is
        the model's spawn heading. Calibrate it with tools/calibrate_tower.py.
        """
        from .geo import bearing_deg, distance_m
        calib = self._calib or self.load_calibration()
        bearing = bearing_deg(self.lat, self.lon, lat, lon)
        dist = distance_m(self.lat, self.lon, lat, lon)
        az = bearing - calib["pan"].get("base_yaw_deg", 0.0)
        az = (az + 180.0) % 360.0 - 180.0
        el = math.degrees(math.atan2(alt - self.alt_amsl, max(dist, 1.0)))
        el = max(-30.0, min(45.0, el))
        return self.point(az, el)

    # -- scan ----------------------------------------------------------- #
    def scan(self) -> bool:
        return self.set_mode("SCAN")

    def stop_scan(self) -> bool:
        """Leave SCAN. MANUAL lets us set servos directly again."""
        return self.set_mode("MANUAL")

    def center(self) -> bool:
        return self.point(0.0, 0.0)


def connect_vehicle(spec: AssetSpec, config: Optional[Config] = None,
                    sim_time_fn: Optional[Callable[[], float]] = None) -> Vehicle:
    """Factory: build the right Vehicle subclass from the asset's kind."""
    cls = {"copter": Copter, "plane": Plane, "tower": Tower}.get(spec.kind, Vehicle)
    return cls(spec, config=config, sim_time_fn=sim_time_fn)
