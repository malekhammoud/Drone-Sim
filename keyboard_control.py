#!/usr/bin/env python3
"""
Keyboard flight control for an ArduCopter flying in GUIDED mode.

Arrow keys fly the aircraft, WASD handle altitude and yaw:

    Up / Down      forward / backward
    Left / Right   strafe left / right
    W / S          climb / descend
    A / D          yaw left / right
    Space          stop now (zero velocity, clear held keys)
    Q / Ctrl-C     quit (aircraft stops and hovers)

How it works
------------
The aircraft is in GUIDED mode. We continuously send
SET_POSITION_TARGET_LOCAL_NED messages containing a velocity vector in the
MAV_FRAME_BODY_OFFSET_NED frame plus a yaw rate. That frame is relative to
the aircraft's current heading, so "forward" always means "the way the nose
is pointing" and we do not have to do any frame math ourselves.

Setpoints are streamed at ~30 Hz, which ArduCopter needs to keep its GUIDED
velocity controller engaged. Telemetry is drained every iteration so the
UDP receive buffer never overflows.

The commanded velocity is slew-limited (ramped), so the aircraft does not
jerk when a key is pressed or released. If no keys are held, the target is
zero and the aircraft brakes to a hover.

Because a terminal has no key-up events, a key is considered held until it
has not been seen for KEY_HOLD seconds (key auto-repeat keeps refreshing
it). Press Space to stop immediately.
"""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import termios
import time
import tty

from pymavlink import mavutil

M = mavutil.mavlink

# Body-relative velocity + yaw rate, ignoring position and acceleration.
FRAME = M.MAV_FRAME_BODY_OFFSET_NED
TYPEMASK_VEL_YAWRATE = (
    M.POSITION_TARGET_TYPEMASK_X_IGNORE
    | M.POSITION_TARGET_TYPEMASK_Y_IGNORE
    | M.POSITION_TARGET_TYPEMASK_Z_IGNORE
    | M.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | M.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | M.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | M.POSITION_TARGET_TYPEMASK_YAW_IGNORE
)

# A key is "held" if seen within this many seconds. Must be longer than the
# OS key-repeat delay (~0.5 s) or holding a key would briefly look released.
KEY_HOLD = 0.65

ARROWS = {
    b"\x1b[A": "up",
    b"\x1b[B": "down",
    b"\x1b[C": "right",
    b"\x1b[D": "left",
    b"\x1bOA": "up",
    b"\x1bOB": "down",
    b"\x1bOC": "right",
    b"\x1bOD": "left",
}

HELP = """
  Arrow keys      fly: up/down = forward/back, left/right = strafe
  W / S           climb / descend
  A / D           yaw left / right
  Space           stop (hover)
  T               take off (arms if needed, climbs to takeoff altitude)
  L               land
  Q / Ctrl-C      quit
"""


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def approach(current: float, target: float, max_delta: float) -> float:
    """Move `current` toward `target` by at most `max_delta`."""
    if target > current:
        return min(current + max_delta, target)
    return max(current - max_delta, target)


class KeyReader:
    """Non-blocking reader for arrow keys and single characters in a TTY."""

    def __init__(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "keyboard control needs an interactive terminal "
                "(stdin is not a TTY)"
            )
        self.fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self.fd)
        self._buf = b""
        self._held: dict[str, float] = {}

    def __enter__(self) -> "KeyReader":
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)

    def _pump(self) -> None:
        """Read whatever bytes are available into the buffer."""
        while select.select([self.fd], [], [], 0)[0]:
            chunk = os.read(self.fd, 64)
            if not chunk:
                break
            self._buf += chunk

    def poll(self) -> list[str]:
        """Return key names seen since the last call (refreshes held keys)."""
        self._pump()
        seen: list[str] = []
        while self._buf:
            # A lone ESC (or a partial escape sequence) means the rest has not
            # arrived yet; keep it buffered and try again next poll.
            if self._buf[0] == 0x1B:
                match = next(
                    (seq for seq in ARROWS if self._buf.startswith(seq)), None
                )
                if match is None:
                    if len(self._buf) < 3:
                        break
                    # Unknown escape sequence: drop the ESC and resync.
                    self._buf = self._buf[1:]
                    continue
                self._buf = self._buf[len(match):]
                seen.append(ARROWS[match])
                continue

            ch = self._buf[:1]
            self._buf = self._buf[1:]
            if ch in (b"q", b"Q"):
                seen.append("quit")
            elif ch == b" ":
                seen.append("stop")
            elif ch in (b"w", b"W"):
                seen.append("climb")
            elif ch in (b"s", b"S"):
                seen.append("descend")
            elif ch in (b"a", b"A"):
                seen.append("yaw_left")
            elif ch in (b"d", b"D"):
                seen.append("yaw_right")
            elif ch in (b"t", b"T"):
                seen.append("takeoff")
            elif ch in (b"l", b"L"):
                seen.append("land")

        now = time.monotonic()
        for key in seen:
            self._held[key] = now
        return seen

    def held(self) -> set[str]:
        now = time.monotonic()
        return {k for k, t in self._held.items() if now - t < KEY_HOLD}

    def clear(self) -> None:
        self._held.clear()


class DroneController:
    """Connects to the vehicle and streams GUIDED body-velocity setpoints."""

    def __init__(
        self,
        master: str,
        max_speed: float = 3.0,
        max_climb: float = 1.5,
        max_yaw_rate: float = math.radians(60),
        accel: float = 4.0,
        climb_accel: float = 2.0,
        yaw_accel: float = math.radians(180),
        verbose: bool = True,
    ) -> None:
        self.master = master
        self.max_speed = max_speed
        self.max_climb = max_climb
        self.max_yaw_rate = max_yaw_rate
        self.accel = accel
        self.climb_accel = climb_accel
        self.yaw_accel = yaw_accel
        self.verbose = verbose

        # current (ramped) body-frame state: forward, right, up, yaw_rate
        self.cur = [0.0, 0.0, 0.0, 0.0]
        # desired body-frame state
        self.tgt = [0.0, 0.0, 0.0, 0.0]

        self.yaw = 0.0
        self.alt = 0.0
        self.armed = False
        self.mode = "?"
        self.vn = self.ve = self.vd = 0.0
        self.x = self.y = 0.0
        self.connected = False

        self.mav = mavutil.mavlink_connection(
            master, source_system=255, source_component=190
        )
        self._t0 = time.monotonic()

    # -- connection -------------------------------------------------------
    def connect(self, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        hb = None
        while time.monotonic() < deadline and hb is None:
            self.send_heartbeat()
            hb = self.mav.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb is None:
            raise RuntimeError(f"no heartbeat from {self.master}")
        self.mav.target_system = hb.get_srcSystem()
        self.mav.target_component = hb.get_srcComponent()

        # Only ask for the streams we actually use; the link may be a slow
        # radio/network path and the default firehose is a lot of traffic.
        for stream, rate in (
            (M.MAV_DATA_STREAM_POSITION, 5),
            (M.MAV_DATA_STREAM_EXTRA1, 10),   # ATTITUDE
            (M.MAV_DATA_STREAM_EXTENDED_STATUS, 2),
        ):
            self.mav.mav.request_data_stream_send(
                self.mav.target_system, self.mav.target_component, stream, rate, 1
            )
        self._update_from(hb)
        self.connected = True
        if self.verbose:
            print(f"Connected: system {self.mav.target_system}, mode {self.mode}, "
                  f"armed={self.armed}")

    def send_heartbeat(self) -> None:
        self.mav.mav.heartbeat_send(
            M.MAV_TYPE_GCS, M.MAV_AUTOPILOT_INVALID, 0, 0, 0
        )

    def set_guided(self) -> None:
        self.mav.set_mode("GUIDED")
        time.sleep(0.2)

    def ensure_guided(self) -> bool:
        if self.mode == "GUIDED":
            return True
        print(f"Switching {self.mode} -> GUIDED ...")
        self.set_guided()
        end = time.monotonic() + 5
        while time.monotonic() < end and self.mode != "GUIDED":
            self.poll_telemetry()
            self.send_heartbeat()
            time.sleep(0.1)
        return self.mode == "GUIDED"

    def arm(self, timeout: float = 20.0) -> bool:
        """Arm the vehicle, retrying because pre-arm checks can take a moment."""
        if self.armed:
            return True
        deadline = time.monotonic() + timeout
        next_try = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_try:
                self.mav.arducopter_arm()
                next_try = now + 2.0
            self.send_heartbeat()
            self.poll_telemetry()
            if self.armed:
                return True
            time.sleep(0.1)
        return False

    def takeoff(self, altitude: float = 15.0, timeout: float = 120.0) -> bool:
        """Arm if needed and climb to `altitude` in GUIDED mode.

        Straight after a sim reset the EKF needs ~1-2 min to settle. During that
        window the autopilot accepts `arm` and `NAV_TAKEOFF` but holds the motors
        at idle and then auto-disarms ("Disarming motors" ~10 s after arming), so
        a single attempt looks like a rejected takeoff — the aircraft appears
        stuck on the ground. Loop instead: if it disarms without climbing, re-arm
        and re-command until `timeout` is spent.
        """
        if self.armed and self.alt > 2.0:
            print("Already flying; ignoring takeoff.")
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.ensure_guided():
                print(f"GUIDED unavailable (mode {self.mode}); retrying ...")
                time.sleep(1.0)
                continue
            if not self.armed:
                print("Arming ...")
                remaining = max(1.0, deadline - time.monotonic())
                if not self.arm(timeout=min(20.0, remaining)):
                    time.sleep(1.0)
                    continue
            print(f"Taking off to {altitude:.0f} m ...")
            self.mav.mav.command_long_send(
                self.mav.target_system, self.mav.target_component,
                M.MAV_CMD_NAV_TAKEOFF, 0,
                0, 0, 0, 0, 0, 0, altitude,
            )
            climb_until = min(deadline, time.monotonic() + 30.0)
            while time.monotonic() < climb_until:
                self.send_heartbeat()
                self.poll_telemetry()
                if self.alt >= altitude - 1.5:
                    print(f"Reached {self.alt:.1f} m.")
                    return True
                if not self.armed:
                    print("Auto-disarmed before climbing; retrying "
                          "(EKF still settling?).")
                    break
                time.sleep(0.1)
            if self.alt >= altitude - 1.5:
                print(f"Reached {self.alt:.1f} m.")
                return True
        print(f"Takeoff failed after {timeout:.0f} s at {self.alt:.1f} m "
              f"(mode {self.mode}, armed={self.armed}).")
        return False

    def land(self, timeout: float = 120.0) -> bool:
        """Switch to LAND and wait until the vehicle disarms."""
        print("Landing ...")
        self.mav.set_mode("LAND")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.send_heartbeat()
            self.poll_telemetry()
            if not self.armed and self.alt < 0.5:
                print("Landed and disarmed.")
                return True
            time.sleep(0.2)
        print(f"Land timed out (alt {self.alt:.1f} m, armed={self.armed}).")
        return False

    # -- state ------------------------------------------------------------
    def _update_from(self, msg) -> None:
        t = msg.get_type()
        if t == "HEARTBEAT" and msg.get_srcSystem() == self.mav.target_system:
            self.mode = mavutil.mode_string_v10(msg)
            self.armed = bool(msg.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)
        elif t == "ATTITUDE":
            self.yaw = msg.yaw
        elif t == "GLOBAL_POSITION_INT":
            self.alt = msg.relative_alt / 1000.0
        elif t == "LOCAL_POSITION_NED":
            self.x, self.y = msg.x, msg.y
            self.vn, self.ve, self.vd = msg.vx, msg.vy, msg.vz

    def poll_telemetry(self, max_msgs: int = 500) -> int:
        """Drain pending MAVLink messages. Returns how many were handled."""
        n = 0
        while n < max_msgs:
            msg = self.mav.recv_match(blocking=False)
            if msg is None:
                break
            self._update_from(msg)
            n += 1
        return n

    # -- control ----------------------------------------------------------
    def command(self, forward=0.0, right=0.0, up=0.0, yaw_rate=0.0) -> None:
        """Set the target body-frame velocity and yaw rate."""
        self.tgt[0] = clamp(forward, self.max_speed)
        self.tgt[1] = clamp(right, self.max_speed)
        self.tgt[2] = clamp(up, self.max_climb)
        self.tgt[3] = clamp(yaw_rate, self.max_yaw_rate)

    def stop(self) -> None:
        self.tgt = [0.0, 0.0, 0.0, 0.0]

    def step(self, dt: float) -> None:
        """Ramp current toward target and send one setpoint."""
        self.cur[0] = approach(self.cur[0], self.tgt[0], self.accel * dt)
        self.cur[1] = approach(self.cur[1], self.tgt[1], self.accel * dt)
        self.cur[2] = approach(self.cur[2], self.tgt[2], self.climb_accel * dt)
        self.cur[3] = approach(self.cur[3], self.tgt[3], self.yaw_accel * dt)

        fwd, right, up, yaw_rate = self.cur
        self.mav.mav.set_position_target_local_ned_send(
            int((time.monotonic() - self._t0) * 1000) & 0xFFFFFFFF,
            self.mav.target_system,
            self.mav.target_component,
            FRAME,
            TYPEMASK_VEL_YAWRATE,
            0.0, 0.0, 0.0,        # position (ignored)
            fwd, right, -up,      # velocity in body frame, NED (down positive)
            0.0, 0.0, 0.0,        # acceleration (ignored)
            0.0,                  # yaw (ignored)
            yaw_rate,
        )

    def close(self, brake_seconds: float = 1.0) -> None:
        """Bring the aircraft to a hover, then release the link."""
        if not self.connected:
            return
        self.stop()
        end = time.monotonic() + brake_seconds
        last = time.monotonic()
        while time.monotonic() < end:
            now = time.monotonic()
            self.step(now - last)
            last = now
            self.poll_telemetry()
            self.send_heartbeat()
            time.sleep(0.02)

    # -- input ------------------------------------------------------------
    def apply_held(self, held: set[str]) -> None:
        forward = right = up = yaw_rate = 0.0
        if "up" in held:
            forward += self.max_speed
        if "down" in held:
            forward -= self.max_speed
        if "right" in held:
            right += self.max_speed
        if "left" in held:
            right -= self.max_speed
        if "climb" in held:
            up += self.max_climb
        if "descend" in held:
            up -= self.max_climb
        if "yaw_left" in held:
            yaw_rate -= self.max_yaw_rate
        if "yaw_right" in held:
            yaw_rate += self.max_yaw_rate
        self.command(forward, right, up, yaw_rate)


def run_interactive(ctrl: DroneController, hz: float = 30.0,
                    takeoff_alt: float = 15.0) -> None:
    period = 1.0 / hz
    next_tick = time.monotonic()
    last_hb = 0.0
    last_hud = 0.0

    print(HELP)
    print("Ctrl-C or Q to quit. Space stops immediately.\n")

    with KeyReader() as keys:
        try:
            while True:
                events = keys.poll()
                if "quit" in events:
                    break
                if "stop" in events:
                    keys.clear()
                if "takeoff" in events:
                    keys.clear()
                    ctrl.stop()
                    ctrl.cur = [0.0, 0.0, 0.0, 0.0]
                    ctrl.takeoff(takeoff_alt)
                    next_tick = time.monotonic() + period
                    continue
                if "land" in events:
                    keys.clear()
                    ctrl.stop()
                    ctrl.cur = [0.0, 0.0, 0.0, 0.0]
                    ctrl.land()
                    next_tick = time.monotonic() + period
                    continue
                ctrl.apply_held(keys.held())

                now = time.monotonic()
                if now >= next_tick:
                    dt = min(now - (next_tick - period), 0.5)
                    ctrl.step(dt)
                    next_tick += period
                    if next_tick < now:          # fell behind; resync
                        next_tick = now + period

                if now - last_hb >= 1.0:
                    ctrl.send_heartbeat()
                    last_hb = now

                ctrl.poll_telemetry()

                if now - last_hud >= 0.25:
                    speed = math.hypot(ctrl.cur[0], ctrl.cur[1])
                    sys.stdout.write(
                        f"\r{ctrl.mode:>9} {'ARMED' if ctrl.armed else 'disarmed':>8} "
                        f"alt {ctrl.alt:6.1f}m  spd {speed:4.1f}m/s  "
                        f"fwd {ctrl.cur[0]:+4.1f} right {ctrl.cur[1]:+4.1f} "
                        f"up {ctrl.cur[2]:+4.1f} yaw {math.degrees(ctrl.cur[3]):+5.1f}d/s  "
                    )
                    sys.stdout.flush()
                    last_hud = now
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write("\nStopping (hover)...\n")
            ctrl.close()
            print("Done.")


def self_test(ctrl: DroneController, seconds: float = 3.0,
              takeoff_alt: float = 10.0) -> int:
    """Fly a short scripted sequence and verify the aircraft actually moves."""
    ctrl.connect()
    took_off = False
    if not ctrl.armed or ctrl.alt < 2.0:
        print("Vehicle is not flying; taking off for the test ...")
        if not ctrl.takeoff(takeoff_alt):
            print("SELF-TEST FAIL (could not take off)")
            return 1
        took_off = True

    def position():
        """Drain for a moment and return the latest local NED position."""
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            ctrl.poll_telemetry()
            time.sleep(0.01)
        return (ctrl.x, ctrl.y)

    # settle / hover
    for _ in range(20):
        ctrl.step(0.1)
        ctrl.poll_telemetry()
        time.sleep(0.05)
    start = position()
    print(f"start : pos=({start[0]:+.1f},{start[1]:+.1f}) yaw={math.degrees(ctrl.yaw):+.1f}")

    print(f"forward {seconds:.0f}s ...")
    ctrl.command(forward=2.0)
    end = time.monotonic() + seconds
    peak = 0.0
    while time.monotonic() < end:
        ctrl.step(0.05)
        ctrl.send_heartbeat()
        ctrl.poll_telemetry()
        peak = max(peak, math.hypot(ctrl.vn, ctrl.ve))
        time.sleep(0.02)
    mid = position()
    moved = math.hypot(mid[0] - start[0], mid[1] - start[1]) if mid else 0.0
    print(f"moving: pos=({mid[0]:+.1f},{mid[1]:+.1f}) moved={moved:.1f}m peak={peak:.2f}m/s")

    print("braking ...")
    ctrl.close(brake_seconds=2.0)
    final = position()
    residual = math.hypot(ctrl.vn, ctrl.ve)
    print(f"final : speed={residual:.2f} m/s")

    ok = moved > 1.0 and residual < 0.5
    if took_off:
        print("landing ...")
        ok = ctrl.land() and ok
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default="udpout:127.0.0.1:14550",
                    help="MAVLink connection string")
    ap.add_argument("--max-speed", type=float, default=3.0,
                    help="max horizontal speed, m/s (default 3)")
    ap.add_argument("--max-climb", type=float, default=1.5,
                    help="max vertical speed, m/s (default 1.5)")
    ap.add_argument("--max-yaw-rate", type=float, default=60.0,
                    help="max yaw rate, deg/s (default 60)")
    ap.add_argument("--takeoff-alt", type=float, default=15.0,
                    help="altitude in metres for the T (takeoff) key (default 15)")
    ap.add_argument("--no-guided", action="store_true",
                    help="do not switch flight mode; just warn if not GUIDED")
    ap.add_argument("--self-test", action="store_true",
                    help="run a scripted takeoff/forward/land test and exit")
    args = ap.parse_args()

    ctrl = DroneController(
        args.master,
        max_speed=args.max_speed,
        max_climb=args.max_climb,
        max_yaw_rate=math.radians(args.max_yaw_rate),
    )

    if args.self_test:
        try:
            return self_test(ctrl, takeoff_alt=args.takeoff_alt)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            print("Is the vehicle/SITL running and reachable at that master?")
            return 1
        finally:
            ctrl.close(brake_seconds=0.5)

    try:
        ctrl.connect()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        print("Is the vehicle/SITL running and reachable at that master?")
        return 1
    if ctrl.mode != "GUIDED":
        if args.no_guided:
            print(f"WARNING: vehicle is in {ctrl.mode}, not GUIDED; "
                  "velocity setpoints will be ignored.")
        else:
            print(f"Vehicle is in {ctrl.mode}; switching to GUIDED...")
            ctrl.set_guided()
            ctrl.poll_telemetry()
            if ctrl.mode != "GUIDED":
                print(f"WARNING: still in {ctrl.mode}; velocity control may be ignored.")
    if not ctrl.armed:
        print("Vehicle is disarmed. Press T to arm and take off, or arm it yourself.")

    run_interactive(ctrl, takeoff_alt=args.takeoff_alt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
