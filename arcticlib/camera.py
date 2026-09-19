"""Camera access.

The sim serves each Gazebo camera as MJPEG over HTTP (``RECON.md §2``):

* ``/snapshot.jpg`` — one JPEG. Polling this is the cheapest way to feed CV.
* ``/stream``       — ``multipart/x-mixed-replace``; what ``cv2.VideoCapture``
  wants.

:class:`CameraSource` supports both: :meth:`poll` runs a background thread that
grabs snapshots into :meth:`latest` (non-blocking), and :meth:`stream` is a
blocking generator over the MJPEG feed.

Frames are timestamped with sim time when a clock is supplied, else wall time.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Iterator, Optional

import cv2
import numpy as np
import requests

from .config import CameraSpec
from .types import Frame

log = logging.getLogger("arcticlib.camera")


class CameraSource:
    """One camera stream, with a background snapshot poller."""

    def __init__(self, spec: CameraSpec,
                 sim_time_fn: Optional[Callable[[], float]] = None,
                 timeout: float = 5.0) -> None:
        self.spec = spec
        self._sim_time_fn = sim_time_fn
        self._timeout = timeout

        self._latest: Optional[Frame] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    def _stamp(self) -> float:
        if self._sim_time_fn is not None:
            try:
                return self._sim_time_fn()
            except Exception:
                pass
        return time.monotonic()

    def grab(self) -> Optional[Frame]:
        """Fetch and decode one snapshot. Returns None on any failure."""
        try:
            resp = self._session.get(self.spec.snapshot_url, timeout=self._timeout)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            log.debug("%s: snapshot failed: %s", self.spec.asset, exc)
            return None
        if img is None:
            return None
        frame = Frame(asset=self.spec.asset, t_sim=self._stamp(), t_wall=time.monotonic(),
                      image=img, width=img.shape[1], height=img.shape[0],
                      hfov_deg=self.spec.hfov_deg, vfov_deg=self.spec.vfov_deg)
        with self._lock:
            self._latest = frame
        return frame

    def latest(self) -> Optional[Frame]:
        """Most recent polled frame, or None. Never blocks."""
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------ #
    def poll(self, rate_hz: float = 2.0) -> None:
        """Start the background poller at ``rate_hz``."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            period = 1.0 / max(rate_hz, 0.05)
            while not self._stop.is_set():
                self.grab()
                self._stop.wait(period)

        self._thread = threading.Thread(target=loop, name=f"cam-{self.spec.asset}",
                                        daemon=True)
        self._thread.start()

    def stream(self) -> Iterator[Frame]:
        """Blocking generator over the MJPEG feed (for ``cv2.VideoCapture``)."""
        cap = cv2.VideoCapture(self.spec.stream_url)
        try:
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                yield Frame(asset=self.spec.asset, t_sim=self._stamp(),
                            t_wall=time.monotonic(), image=img,
                            width=img.shape[1], height=img.shape[0],
                            hfov_deg=self.spec.hfov_deg, vfov_deg=self.spec.vfov_deg)
        finally:
            cap.release()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self._session.close()
        except Exception:
            pass
