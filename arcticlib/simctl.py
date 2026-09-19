"""Simulation control and the global sim clock.

Two things live here:

* :class:`SimClient` — ``sim_time()``, ``reset()``, ``wait_until_ready()`` and
  the ``/api/status`` + ``/api/assets`` probes.
* A minimal Gazebo-transport WebSocket client. gzweb relays ``~/world_stats``
  (the authoritative sim clock) and ``~/pose/info`` (every moving model's true
  world pose) to any WebSocket client. No extra dependency is used, so the
  package installs with just the four declared deps.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import select
import socket
import struct
import threading
import time
from typing import Callable, Optional

import requests

from .config import Config, load_config

log = logging.getLogger("arcticlib.simctl")


# --------------------------------------------------------------------------- #
# Minimal RFC6455 client (text frames from the server only)
# --------------------------------------------------------------------------- #
class _WSClient:
    """Just enough WebSocket to read gzweb's JSON relay."""

    def __init__(self, host: str, port: int, path: str = "/",
                 timeout: float = 10.0) -> None:
        self.host, self.port, self.path = host, port, path
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._buf = b""

    def connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {self.path} HTTP/1.1\r\nHost: {self.host}:{self.port}\r\n"
               "Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
               f"Origin: http://{self.host}:{self.port}\r\n\r\n")
        sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("gzweb closed during handshake")
            buf += chunk
        head, _, rest = buf.partition(b"\r\n\r\n")
        if b"101" not in head.split(b"\r\n")[0]:
            raise ConnectionError(f"websocket handshake failed: {head[:80]!r}")
        self._buf = rest
        sock.settimeout(self.timeout)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._buf = b""

    def _read_exact(self, n: int) -> bytes:
        assert self._sock is not None
        while len(self._buf) < n:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("gzweb closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _send_pong(self, payload: bytes) -> None:
        assert self._sock is not None
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        header = bytes([0x8A, 0x80 | len(payload)]) + mask
        self._sock.sendall(header + masked)

    def recv(self) -> Optional[str]:
        """Read one text message, or None on a control frame we ignored."""
        for _ in range(8):
            h = self._read_exact(2)
            opcode = h[0] & 0x0F
            ln = h[1] & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", self._read_exact(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._read_exact(8))[0]
            payload = self._read_exact(ln)
            if opcode == 0x1:
                return payload.decode("utf-8", "replace")
            if opcode == 0x9:
                self._send_pong(payload)
            elif opcode == 0x8:
                raise ConnectionError("gzweb closed the connection")
        return None


# --------------------------------------------------------------------------- #
# Sim client
# --------------------------------------------------------------------------- #
class SimClient:
    """Control-plane access plus the authoritative sim clock.

    ``sim_time()`` returns the last ``~/world_stats`` value (seconds); until the
    first sample arrives it falls back to wall time so callers always get a
    usable number.
    """

    def __init__(self, config: Optional[Config] = None,
                 pose_callback: Optional[Callable[[dict], None]] = None,
                 autostart: bool = True) -> None:
        self.config = config or load_config()
        self._pose_callback = pose_callback
        self._sim_time: Optional[float] = None
        self._last_stats: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.clocks = 0                    # world_stats samples seen
        if autostart:
            self.start()

    # -- clock --------------------------------------------------------- #
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._clock_loop, name="simclock",
                                        daemon=True)
        self._thread.start()

    def _clock_loop(self) -> None:
        while not self._stop.is_set():
            ws = _WSClient(self.config.host, self.config.gzweb_port)
            try:
                ws.connect()
                log.info("simclock: connected to gzweb ws://%s:%d",
                         self.config.host, self.config.gzweb_port)
                while not self._stop.is_set():
                    if not select.select([ws._sock], [], [], 1.0)[0]:
                        continue
                    raw = ws.recv()
                    if raw is None:
                        continue
                    try:
                        msg = json.loads(raw)
                    except ValueError:
                        continue
                    topic = msg.get("topic")
                    body = msg.get("msg") or {}
                    if topic == "~/world_stats":
                        st = body.get("sim_time") or {}
                        with self._lock:
                            self._sim_time = float(st.get("sec", 0)) + \
                                float(st.get("nsec", 0)) / 1e9
                            self._last_stats = body
                            self.clocks += 1
                    elif topic == "~/pose/info" and self._pose_callback is not None:
                        try:
                            self._pose_callback(body)
                        except Exception as exc:
                            log.debug("pose callback error: %s", exc)
            except Exception as exc:
                log.debug("simclock: %s; reconnecting", exc)
                ws.close()
                self._stop.wait(1.0)

    def sim_time(self) -> float:
        """Latest Gazebo sim seconds, else wall time (monotonic)."""
        with self._lock:
            if self._sim_time is not None:
                return self._sim_time
        return time.monotonic()

    def world_stats(self) -> dict:
        with self._lock:
            return dict(self._last_stats)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    # -- control API --------------------------------------------------- #
    def status(self) -> dict:
        try:
            r = requests.get(self.config.url(self.config.control_port) + "/api/status",
                             timeout=5)
            return r.json()
        except Exception as exc:
            return {"state": "unreachable", "detail": str(exc)}

    def assets(self) -> list[dict]:
        try:
            r = requests.get(self.config.url(self.config.control_port) + "/api/assets",
                             timeout=6)
            return r.json().get("assets", [])
        except Exception:
            return []

    def ready(self) -> bool:
        """True when every rostered asset answers MAVLink."""
        assets = [a for a in self.assets() if a.get("rostered")]
        return bool(assets) and all(a.get("mavlink") for a in assets)

    def reset(self) -> bool:
        """Restart the sim and all assets. Non-blocking server-side."""
        try:
            r = requests.post(self.config.url(self.config.control_port) + "/api/reset",
                              timeout=10)
            return r.status_code in (200, 202)
        except Exception as exc:
            log.warning("reset failed: %s", exc)
            return False

    def wait_until_ready(self, timeout: float = 240.0, poll: float = 5.0,
                         wait_for_reset: bool = False) -> bool:
        """Poll until the sim is running and every roster asset answers.

        With ``wait_for_reset`` we first wait for the control plane to leave
        ``idle`` (a reset was just requested), so an immediate check cannot
        return the pre-reset ``ready`` state.
        """
        end = time.monotonic() + timeout
        seen_working = not wait_for_reset
        while time.monotonic() < end:
            st = self.status()
            state = st.get("state")
            if state == "working":
                seen_working = True
            if seen_working and state != "working" and self.ready():
                return True
            time.sleep(poll)
        return False
