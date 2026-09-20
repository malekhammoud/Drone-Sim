"""Track submission client.

POST a detection to the competition: create on first sighting, update by the
same ``name`` thereafter. Verified contract (RECON.md §5):

* create  -> ``{"ok":true,"created":true,"name","uuid","lat","lon","timestamp"}``
* update  -> ``created:false``, bumps ``fixes``
* list    -> ``{"ok":true,"count","tracks":[{name,uuid,...,lat,lon,heading,speed}]}``

This endpoint rate-limits, so the client is deliberately gentle:

* at most one request per ``min_interval`` seconds **globally** (default 1 s);
* at most one request per ``post_every_s`` seconds **per track** (default 5 s),
  and only when the target actually moved more than ``min_move_m`` (default 5 m);
* HTTP 429 is respected: the client backs off (honouring ``Retry-After``) instead
  of hammering, and retries are spaced exponentially.

Raise the intervals if the server still throttles you; lower them only if you
know the limit.
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

    def __init__(self, base_url: str, min_interval: float = 1.0,
                 post_every_s: float = 5.0, min_move_m: float = 5.0,
                 timeout: float = 5.0, retries: int = 3) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval          # global spacing between POSTs
        self.post_every_s = post_every_s          # per-track spacing
        self.min_move_m = min_move_m              # per-track movement gate
        self.timeout = timeout
        self.retries = retries
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._last_post = 0.0
        self._rate_limited_until = 0.0
        self._last_fix: dict[str, tuple[float, float, float]] = {}         # motion derivation
        self._last_posted_fix: dict[str, tuple[float, float, float]] = {}  # throttling

    # ------------------------------------------------------------------ #
    def _post(self, path: str, payload: dict) -> Optional[dict]:
        for attempt in range(self.retries + 1):
            wait = self._rate_limited_until - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                r = self._session.post(self.base_url + path, json=payload,
                                       timeout=self.timeout)
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After", "")
                    try:
                        delay = max(1.0, float(retry_after))
                    except ValueError:
                        delay = min(30.0, 2.0 ** (attempt + 1))
                    self._rate_limited_until = time.monotonic() + delay
                    log.warning("tracks: 429 rate-limited; backing off %.1fs", delay)
                    if attempt >= self.retries:
                        return None
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                if attempt >= self.retries:
                    log.warning("tracks: POST %s failed: %s", path, exc)
                    return None
                time.sleep(0.5 * (2 ** attempt))
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
                 t: Optional[float] = None,
                 post_every_s: Optional[float] = None,
                 min_move_m: Optional[float] = None) -> Optional[dict]:
        """Post a fix, deriving ``heading`` (deg true) and ``speed`` (m/s) from
        the previous fix for the same ``name`` — but **throttled** so we do not
        get rate-limited.

        A POST is sent only when at least ``post_every_s`` seconds have passed
        since the last *posted* fix for this name **or** the target moved more
        than ``min_move_m``. Skipped calls return ``None`` (not an error). The
        first fix for a name is always posted (create). ``t`` is a sim/monotonic
        time in seconds; with ``t=None`` the throttle is bypassed.
        """
        pe = self.post_every_s if post_every_s is None else post_every_s
        mm = self.min_move_m if min_move_m is None else min_move_m

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

            last = self._last_posted_fix.get(name)
            if last is not None and pe and pe > 0:
                plat, plon, pt = last
                if (t - pt) < pe and distance_m(plat, plon, lat, lon) < mm:
                    return None                      # too soon and barely moved

        res = self.post(name, lat, lon, heading=heading, speed=speed)
        if res is not None and t is not None:
            self._last_posted_fix[name] = (float(lat), float(lon), float(t))
        return res

    def list(self) -> list[dict]:
        """List current tracks (empty list on failure)."""
        try:
            r = self._session.get(self.base_url + "/api/tracks", timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("tracks", [])
        except Exception as exc:
            log.warning("tracks: GET failed: %s", exc)
            return []
