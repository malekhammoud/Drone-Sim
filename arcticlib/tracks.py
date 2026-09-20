"""Track submission client.

POST a detection to the competition: create on first sighting, update by the
same ``name`` thereafter. Verified contract (RECON.md §5):

* create  -> ``{"ok":true,"created":true,"name","uuid","lat","lon","timestamp"}``
* update  -> ``created:false``, bumps ``fixes``
* list    -> ``{"ok":true,"count","tracks":[{name,uuid,...,lat,lon,heading,speed}]}``

Client-side rate limiting keeps us from ever spamming the endpoint: at most one
request per ``min_interval`` seconds, plus retries with backoff.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import requests

from .geo import bearing_deg, distance_m

log = logging.getLogger("arcticlib.tracks")


class TrackClient:
    """Thin, rate-limited wrapper over ``/api/tracks``."""

    def __init__(self, base_url: str, min_interval: float = 0.5,
                 timeout: float = 5.0, retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval
        self.timeout = timeout
        self.retries = retries
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._last_post = 0.0
        self._last_fix: dict[str, tuple[float, float, float]] = {}  # name -> (lat, lon, t)

    # ------------------------------------------------------------------ #
    def _post(self, path: str, payload: dict) -> Optional[dict]:
        for attempt in range(self.retries + 1):
            try:
                r = self._session.post(self.base_url + path, json=payload,
                                       timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                if attempt >= self.retries:
                    log.warning("tracks: POST %s failed: %s", path, exc)
                    return None
                time.sleep(0.25 * (2 ** attempt))
        return None

    def post(self, name: str, lat: float, lon: float,
             heading: Optional[float] = None,
             speed: Optional[float] = None) -> Optional[dict]:
        """Create or update a track. Returns the response dict or None.

        Heading is degrees and speed is m/s; both are optional on creation but
        should be supplied on updates.
        """
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_post)
            if wait > 0:
                time.sleep(wait)
            self._last_post = time.monotonic()
        payload: dict[str, Any] = {"name": name, "lat": float(lat), "lon": float(lon)}
        if heading is not None:
            payload["heading"] = float(heading)
        if speed is not None:
            payload["speed"] = float(speed)
        return self._post("/api/tracks", payload)

    def post_fix(self, name: str, lat: float, lon: float,
                 t: Optional[float] = None) -> Optional[dict]:
        """Post a fix, deriving ``heading`` (deg true) and ``speed`` (m/s) from
        the previous fix for the same ``name``.

        ``t`` is a monotonic/sim time in seconds. The first fix has no heading or
        speed (that matches the API: create takes just name/lat/lon; updates add
        heading/speed). Falls back to a plain :meth:`post` when ``t`` is None.
        """
        heading = speed = None
        if t is not None:
            prev = self._last_fix.get(name)
            if prev is not None:
                plat, plon, pt = prev
                dt = t - pt
                dist = distance_m(plat, plon, lat, lon)
                if dt > 1e-3:
                    speed = dist / dt
                if dist > 0.5:
                    heading = bearing_deg(plat, plon, lat, lon)
            self._last_fix[name] = (float(lat), float(lon), float(t))
        return self.post(name, lat, lon, heading=heading, speed=speed)

    def list(self) -> list[dict]:
        """List current tracks (empty list on failure)."""
        try:
            r = self._session.get(self.base_url + "/api/tracks", timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("tracks", [])
        except Exception as exc:
            log.warning("tracks: GET failed: %s", exc)
            return []
