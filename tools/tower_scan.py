#!/usr/bin/env python3
"""Pan/tilt tower watch: sweep the strait, detect boats, triangulate a tip.

The two AntennaTracker masts (``tower-1``/``tower-2``) are static — MAVLink only
moves their pan/tilt head, never the mast. This module turns them into a
persistent search sensor:

1. **Sweep.** Each tower pans back and forth across an arc centred on the water,
   stepping through a few tilt levels (a serpentine raster). Every dwell it grabs
   a frame and runs the *same* CV the wing and quad use
   (:class:`tools.detect_verified.VerifiedDetector` — colour anomaly -> CNN
   verifier). Its pixel-space temporal filter is off here: a panning head jumps a
   boat across pixels between dwells, so persistence is done in lat/lon instead
   (see :class:`TowerWatch`).
2. **Geolocate.** A verified hit is turned into a lat/lon with the same trig
   pipeline the rest of the project uses (:mod:`arcticlib.geolocate`). A tower's
   camera direction is not its body attitude, so we build an *effective* pose
   from the commanded pan/tilt: ``yaw = camera bearing``, ``pitch = elevation``.
3. **Triangulate.** A single tower gives a *bearing line*; two towers seeing the
   same boat at the same time intersect to a fix and a search radius. Either way
   the wing gets a point to investigate.

Camera-bearing calibration
--------------------------
The model spawns with world yaw 0 (grid +X = grid east) and the pan joint angle
equals the commanded azimuth in degrees. World yaw ``t`` bears
``convergence + 90 - t`` from true north (``arcticlib.geo``), so::

    bearing_deg = base_yaw_deg - az_deg        base_yaw_deg = convergence + 90

Verified live against the vessel: commanding
``az = base_yaw - bearing`` put the true ship at pixel (640, 360) on a
1280x720 tower frame. ``base_yaw_deg`` is overridable per tower for a site
whose convergence or model heading differs.

Run standalone (live)::

    python tools/tower_scan.py --duration 60
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcticlib.config import Config, load_config
from arcticlib.fleet import Fleet
from arcticlib.geo import Georef, bearing_deg, destination, distance_m
from arcticlib.geolocate import GeoConfig
from arcticlib.types import Pose
from tools.detect_verified import VerifiedDetector

log = logging.getLogger("tower_scan")

# Camera centre height on the mast, from terrain/tower.py HEAD_Z. The mast base
# is at the ground, so the camera sits this far above it.
CAMERA_HEAD_M = 2.70

# Default mast sites (lat, lon) and the ground elevation the terrain build placed
# them on. Overridable per tower with ARCTICSIM_TOWER_<N>_LAT / _LON / _GROUND_M
# so a rebuilt world needs no code edit.
DEFAULT_TOWER_SITES: dict[str, tuple[float, float]] = {
    "tower-1": (71.996396, -94.891696),
    "tower-2": (71.986899, -94.778238),
}
DEFAULT_TOWER_GROUND_M: dict[str, float] = {
    # Measured from the sim (GLOBAL_POSITION_INT alt) once the masts are placed.
    "tower-1": 79.4,
    "tower-2": 96.0,
}

# Towers look across water at shallow depression, so the wing's 10 deg grazing
# floor would flag nearly every fix. 1.5 deg still rejects a ray at the horizon.
TOWER_MIN_DEPRESSION_DEG = 1.5


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Sweep plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SweepPlan:
    """Serpentine pan/tilt raster centred on ``center_bearing_deg``.

    ``pan_step_deg`` should be well under the camera's 60 deg HFOV so successive
    columns overlap; ``tilts_deg`` are elevations (positive up, negative down)
    chosen so the near and far water are both sampled.
    """

    center_bearing_deg: float
    half_width_deg: float = 55.0
    pan_step_deg: float = 15.0
    tilts_deg: tuple[float, ...] = (-3.0, -8.0, -14.0)

    def steps(self) -> list[tuple[float, float]]:
        """All (bearing_deg, elev_deg) dwells, sweeping back and forth."""
        half = self.half_width_deg
        n = max(1, int(round(2 * half / max(self.pan_step_deg, 1e-6))))
        bearings = [self.center_bearing_deg - half + i * (2 * half / n)
                    for i in range(n + 1)]
        out: list[tuple[float, float]] = []
        for row, tilt in enumerate(self.tilts_deg):
            row_bearings = bearings if row % 2 == 0 else list(reversed(bearings))
            out += [((b + 180.0) % 360.0 - 180.0, tilt) for b in row_bearings]
        return out


# --------------------------------------------------------------------------- #
# Contacts and tips
# --------------------------------------------------------------------------- #
@dataclass
class TowerContact:
    """One verified, geolocated boat sighting from one tower."""

    tower: str
    t_sim: float
    t_wall: float
    bearing_deg: float
    elevation_deg: float
    lat: float
    lon: float
    error_radius_m: float
    ground_range_m: float
    score: float
    confirmed: bool
    hits: int = 1

    @property
    def fresh(self) -> float:
        return time.monotonic() - self.t_wall


@dataclass
class TowerTip:
    """A point handed to the wing to investigate.

    ``source`` is ``"triangulated"`` when two towers crossed, else ``"single"``.
    ``bearing_deg``/``bearing_tower`` describe the line when only one tower saw
    the boat, so the wing can search along it rather than a point fix.
    """

    id: str
    lat: float
    lon: float
    sigma_m: float
    source: str
    towers: tuple[str, ...]
    t: float
    bearing_deg: Optional[float] = None
    bearing_tower: Optional[str] = None
    origin_lat: Optional[float] = None
    origin_lon: Optional[float] = None
    ground_range_m: Optional[float] = None


# --------------------------------------------------------------------------- #
# Trig: bearing-line intersection
# --------------------------------------------------------------------------- #
def line_intersection(lat1: float, lon1: float, brg1_deg: float,
                      lat2: float, lon2: float, brg2_deg: float,
                      georef: Optional[Georef] = None,
                      min_cross_deg: float = 8.0
                      ) -> Optional[tuple[float, float, float, float]]:
    """Intersect two bearing rays. Returns ``(lat, lon, t1_m, t2_m)`` or None.

    Worked in local ENU metres: a ray from ``p`` on true bearing ``b`` is
    ``p + t * (sin b, cos b)`` (east, north). Parallel rays and intersections
    *behind* either sensor are rejected — a boat cannot be behind a tower.
    ``min_cross_deg`` rejects a near-parallel crossing whose error explodes.
    """
    if georef is None:
        georef = Georef(lat1, lon1)
    e1, n1, _ = georef.to_enu(lat1, lon1)
    e2, n2, _ = georef.to_enu(lat2, lon2)
    a1, a2 = math.radians(brg1_deg), math.radians(brg2_deg)
    d1 = (math.sin(a1), math.cos(a1))
    d2 = (math.sin(a2), math.cos(a2))

    # Cross product of the two directions; near zero means parallel.
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < math.sin(math.radians(min_cross_deg)):
        return None

    de, dn = e2 - e1, n2 - n1
    t1 = (de * d2[1] - dn * d2[0]) / cross
    t2 = (de * d1[1] - dn * d1[0]) / cross
    if t1 <= 0.0 or t2 <= 0.0:
        return None

    lat, lon, _ = georef.to_latlon(e1 + t1 * d1[0], n1 + t1 * d1[1])
    return lat, lon, t1, t2


def triangulation_sigma_m(t1_m: float, t2_m: float, cross_deg: float,
                          bearing_sigma_deg: float = 3.0) -> float:
    """Rough 1-sigma position radius from two crossing bearings.

    Cross-track error at each range is ``r * sigma``; dividing by ``sin(cross)``
    projects it along the other ray. Reported as the larger of the two arms.
    """
    s = math.radians(bearing_sigma_deg)
    denom = max(math.sin(math.radians(cross_deg)), 1e-3)
    return max(t1_m, t2_m) * s / denom


# --------------------------------------------------------------------------- #
# One tower
# --------------------------------------------------------------------------- #
class TowerScanner:
    """Sweeps one mast and turns verified hits into :class:`TowerContact`s."""

    def __init__(self, fleet: Fleet, name: str, *,
                 lat: Optional[float] = None, lon: Optional[float] = None,
                 ground_elev_m: Optional[float] = None,
                 base_yaw_deg: Optional[float] = None,
                 detector: Optional[VerifiedDetector] = None,
                 geo: Optional[GeoConfig] = None,
                 sweep: Optional[SweepPlan] = None,
                 settle_s: float = 0.8) -> None:
        self.fleet = fleet
        self.name = name
        self.cfg = fleet.config
        self.tower = fleet.vehicle(name)
        self.cam = fleet.cams.get(name)
        if self.tower is None or self.cam is None:
            raise RuntimeError(f"{name}: no tower vehicle/camera")

        key = name.upper().replace("-", "_")
        self.lat = _envf(f"{key}_LAT", DEFAULT_TOWER_SITES[name][0]) if lat is None else lat
        self.lon = _envf(f"{key}_LON", DEFAULT_TOWER_SITES[name][1]) if lon is None else lon
        self.ground_elev_m = (_envf(f"{key}_GROUND_M", DEFAULT_TOWER_GROUND_M[name])
                              if ground_elev_m is None else ground_elev_m)
        # See module docstring: true bearing of pan centre is convergence + 90.
        self.base_yaw_deg = ((self.cfg.convergence_deg + 90.0) % 360.0
                             if base_yaw_deg is None else base_yaw_deg)

        self.intrinsics = self.cfg.assets[name].camera.intrinsics()
        self.detector = detector or VerifiedDetector(
            model_path="models/patch_verifier.pt", min_color_score=0.25,
            min_verify_prob=0.50, enable_temporal=True, min_hits=4, max_misses=4)
        self.geo = geo or GeoConfig(alt_ref="amsl", ground_elevation_m=0.0,
                                    camera_pitch_deg=0.0,
                                    min_depression_deg=TOWER_MIN_DEPRESSION_DEG,
                                    reject_grazing=False)
        self.sweep = sweep or SweepPlan(
            center_bearing_deg=bearing_deg(self.lat, self.lon,
                                           self.cfg.origin_lat, self.cfg.origin_lon))
        self.settle_s = settle_s
        self._steps = self.sweep.steps()
        self._i = 0
        self.frame_idx = 0

    # -- pointing ------------------------------------------------------- #
    def camera_bearing_deg(self, az_deg: float) -> float:
        """True bearing the camera looks along for a commanded azimuth."""
        b = (self.base_yaw_deg - az_deg) % 360.0
        return 0.0 if b >= 360.0 else b

    def az_for_bearing(self, bearing_deg: float) -> float:
        """Commanded azimuth (deg) that points the camera at a true bearing."""
        return (self.base_yaw_deg - bearing_deg + 180.0) % 360.0 - 180.0

    def point_bearing(self, bearing_deg: float, elev_deg: float) -> bool:
        return self.tower.point(self.az_for_bearing(bearing_deg), elev_deg)

    def _refresh_position(self) -> None:
        """Adopt the mast's live lat/lon/alt when the link has them.

        The configured site is the requested placement; the sim is the truth.
        Using the live altitude matters most — it is the camera's height above
        sea level and directly scales the range estimate.
        """
        if (self.tower.connected and abs(getattr(self.tower, "lat", 0.0)) > 1.0
                and abs(getattr(self.tower, "lon", 0.0)) > 1.0):
            self.lat = float(self.tower.lat)
            self.lon = float(self.tower.lon)
            self.ground_elev_m = float(getattr(self.tower, "alt_amsl", self.ground_elev_m))

    def effective_pose(self, bearing_deg: float, elev_deg: float,
                       t_sim: float = 0.0) -> Pose:
        """Camera pose for geolocation: bearing/elevation baked into yaw/pitch."""
        return Pose(asset=self.name, t_sim=t_sim, t_wall=time.monotonic(),
                    lat=self.lat, lon=self.lon, alt_rel=0.0,
                    alt_amsl=self.ground_elev_m + CAMERA_HEAD_M,
                    roll=0.0, pitch=math.radians(elev_deg),
                    yaw=math.radians(bearing_deg))

    # -- sweep ---------------------------------------------------------- #
    def next_step(self) -> tuple[float, float]:
        bearing, elev = self._steps[self._i]
        self._i = (self._i + 1) % len(self._steps)
        return bearing, elev

    def reset_sweep(self) -> None:
        self._i = 0

    def step(self) -> list[TowerContact]:
        """One dwell: aim, let the servos settle, grab, detect, geolocate."""
        if not self.tower.connected:
            return []
        self._refresh_position()
        if self.tower.mode.upper() != "MANUAL":
            self.tower.set_mode("MANUAL", timeout=2.0)

        bearing, elev = self.next_step()
        if not self.point_bearing(bearing, elev):
            log.debug("%s: point failed", self.name)
            return []
        time.sleep(self.settle_s)

        frame = self.cam.grab()
        if frame is None:
            return []
        self.frame_idx += 1
        pose = self.effective_pose(bearing, elev, frame.t_sim)

        dets = self.detector.detect(frame.image, frame_idx=self.frame_idx,
                                    t_sim=frame.t_sim, pose=pose,
                                    cam_intrinsics=self.intrinsics)
        contacts: list[TowerContact] = []
        for c in dets:
            est = self.geo.locate(c.cx, c.cy, pose, self.name, self.intrinsics)
            if est is None:
                continue
            contacts.append(TowerContact(
                tower=self.name, t_sim=frame.t_sim, t_wall=time.monotonic(),
                bearing_deg=bearing, elevation_deg=elev,
                lat=est.lat, lon=est.lon, error_radius_m=est.error_radius_m,
                ground_range_m=est.ground_range_m, score=float(c.score),
                # Stage 1+2 verified this frame. Temporal persistence is done
                # across dwells in TowerWatch, in lat/lon, because the panning
                # camera jumps a boat across pixels between dwells.
                confirmed=True, hits=int(getattr(c, "hits", 1))))
        return contacts


# --------------------------------------------------------------------------- #
# Both towers, in a background thread
# --------------------------------------------------------------------------- #
class TowerWatch:
    """Continuously sweeps every tower and publishes the best available tip.

    ``latest_tip()`` is the current best: a two-tower triangulation when both
    masts have a fresh confirmed contact, else the strongest single-tower line.
    Tips carry an id so a consumer can avoid re-investigating the same one.
    """

    def __init__(self, fleet: Fleet, config: Optional[Config] = None, *,
                 names: tuple[str, ...] = ("tower-1", "tower-2"),
                 detector_factory: Optional[Callable[[], VerifiedDetector]] = None,
                 contact_ttl_s: float = 60.0,
                 min_confirm: int = 2,
                 cluster_gate_m: float = 400.0,
                 **scanner_kw) -> None:
        self.fleet = fleet
        self.cfg = config or fleet.config
        self.names = names
        self.contact_ttl_s = contact_ttl_s
        self.min_confirm = min_confirm
        self.cluster_gate_m = cluster_gate_m
        # No temporal filter: it associates in pixel space, which a panning head
        # breaks. TowerWatch persists sightings in lat/lon across dwells instead.
        factory = detector_factory or (lambda: VerifiedDetector(
            model_path="models/patch_verifier.pt", min_color_score=0.25,
            min_verify_prob=0.50, enable_temporal=False))
        self.scanners: list[TowerScanner] = []
        for n in names:
            try:
                self.scanners.append(TowerScanner(fleet, n, detector=factory(),
                                                  **scanner_kw))
            except Exception as exc:                       # missing asset/camera
                log.warning("tower watch: skipping %s (%s)", n, exc)
        self.georef = Georef(self.cfg.origin_lat, self.cfg.origin_lon,
                             ps_centre_x=self.cfg.ps_centre_x,
                             ps_centre_y=self.cfg.ps_centre_y)
        self._lock = threading.Lock()
        self._contacts: list[TowerContact] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.cycles = 0

    # -- lifecycle ------------------------------------------------------ #
    def start(self) -> None:
        if not self.scanners or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="tower-watch",
                                        daemon=True)
        self._thread.start()
        log.info("tower watch: sweeping %s", ", ".join(s.name for s in self.scanners))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            for sc in self.scanners:
                if self._stop.is_set():
                    break
                try:
                    found = sc.step()
                except Exception as exc:
                    log.debug("tower %s step error: %s", sc.name, exc)
                    found = []
                if found:
                    with self._lock:
                        self._contacts += found
                    for c in found:
                        if c.confirmed:
                            log.info("tower %s: CONFIRMED boat at %.5f,%.5f "
                                     "(brg %.0f, %.0f m, score %.2f)",
                                     c.tower, c.lat, c.lon, c.bearing_deg,
                                     c.ground_range_m, c.score)
            self._prune()
            self.cycles += 1

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.contact_ttl_s
        with self._lock:
            self._contacts = [c for c in self._contacts if c.t_wall >= cutoff]

    # -- consumers ------------------------------------------------------ #
    def contacts(self, confirmed_only: bool = False) -> list[TowerContact]:
        with self._lock:
            out = list(self._contacts)
        if confirmed_only:
            out = [c for c in out if c.confirmed]
        return out

    def _cluster_contacts(self, contacts: list[TowerContact]) -> list[list[TowerContact]]:
        """Greedy union of sightings that plausibly describe the same boat.

        The gate is the position error radii combined in quadrature, so a
        grazing-angle fix with a large radius does not split off on its own,
        capped so two genuinely different boats do not merge.
        """
        clusters: list[list[TowerContact]] = []
        for c in sorted(contacts, key=lambda c: -c.score):
            placed = False
            for cl in clusters:
                for m in cl:
                    gate = 2.0 * math.hypot(c.error_radius_m, m.error_radius_m)
                    gate = max(self.cluster_gate_m, min(gate, 2000.0))
                    if distance_m(c.lat, c.lon, m.lat, m.lon) <= gate:
                        cl.append(c)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                clusters.append([c])
        return clusters

    def _triangulate_pairs(self, contacts: list[TowerContact],
                           max_range_m: float = 2500.0):
        """Best crossing of two different masts' bearing lines, or None.

        Bearing-only association is ambiguous, but with one target and a short
        TTL the strongest well-crossed pair is the right call. Crossings behind
        a mast or beyond the camera range are rejected.
        """
        best = None
        for i, a in enumerate(contacts):
            for b in contacts[i + 1:]:
                if a.tower == b.tower:
                    continue
                hit = line_intersection(a.lat, a.lon, a.bearing_deg,
                                        b.lat, b.lon, b.bearing_deg, self.georef)
                if hit is None:
                    continue
                lat, lon, t1, t2 = hit
                if not (50.0 <= t1 <= max_range_m and 50.0 <= t2 <= max_range_m):
                    continue
                cross = _angle_between(a.bearing_deg, b.bearing_deg)
                sigma = triangulation_sigma_m(t1, t2, cross)
                score = (cross, -sigma, a.score + b.score)
                if best is None or score > best[0]:
                    best = (score, lat, lon, sigma, a, b)
        return best

    def latest_tip(self) -> Optional[TowerTip]:
        """Best current tip, or None.

        A well-crossed pair of bearings from the two masts triangulates to a
        point. Failing that, one mast's sightings are grouped in lat/lon and a
        group needs ``min_confirm`` distinct dwells before it counts.
        """
        contacts = self.contacts(confirmed_only=True)
        if not contacts:
            return None

        tri = self._triangulate_pairs(contacts)
        if tri is not None:
            _, lat, lon, sigma, a, b = tri
            return TowerTip(
                id=f"tri-{int(min(a.t_wall, b.t_wall) * 10)}",
                lat=lat, lon=lon, sigma_m=sigma, source="triangulated",
                towers=(a.tower, b.tower), t=min(a.t_sim, b.t_sim))

        valid = []
        for cl in self._cluster_contacts(contacts):
            dwells = {(c.tower, round(c.t_wall, 1)) for c in cl}
            if len(dwells) >= self.min_confirm:
                valid.append(cl)
        if not valid:
            return None
        cl = max(valid, key=lambda c: (len(c), max(x.score for x in c)))
        c = max(cl, key=lambda c: c.score)
        sc = next((s for s in self.scanners if s.name == c.tower), None)
        return TowerTip(
            id=f"one-{c.tower}-{int(c.t_wall * 10)}",
            lat=c.lat, lon=c.lon, sigma_m=max(c.error_radius_m, 50.0),
            source="single", towers=(c.tower,), t=c.t_sim,
            bearing_deg=c.bearing_deg, bearing_tower=c.tower,
            origin_lat=sc.lat if sc else None, origin_lon=sc.lon if sc else None,
            ground_range_m=c.ground_range_m)

    def search_point(self, tip: TowerTip, max_range_m: float = 1500.0,
                     min_range_m: float = 100.0) -> tuple[float, float]:
        """A finite point to fly to.

        A triangulated tip is already a point. A single-tower line is clamped to
        a plausible range along its bearing — the raw ray/water intersection can
        be tens of km away at a 2 deg depression.
        """
        if tip.source == "triangulated" or tip.bearing_deg is None:
            return tip.lat, tip.lon
        if tip.origin_lat is None or tip.origin_lon is None:
            return tip.lat, tip.lon
        rng = min(max(tip.ground_range_m or tip.sigma_m, min_range_m), max_range_m)
        return destination(tip.origin_lat, tip.origin_lon, tip.bearing_deg, rng)


def _angle_between(b1: float, b2: float) -> float:
    """Acute angle between two bearings, degrees."""
    d = abs((b1 - b2) % 360.0)
    return d if d <= 180.0 else 360.0 - d


# --------------------------------------------------------------------------- #
# Standalone live runner
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sweep the towers and report boat contacts.")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds (0 = forever)")
    ap.add_argument("--out", default="tower_output", help="output root")
    ap.add_argument("--model", default="models/patch_verifier.pt")
    ap.add_argument("--no-fly", action="store_true", help="accepted for symmetry; towers never fly")
    ap.add_argument("--min-hits", type=int, default=4)
    return ap


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    cfg = load_config()
    fleet = Fleet.from_config(cfg)
    fleet.wait_ready(20)

    stamp = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join(args.out, stamp)
    os.makedirs(run_dir, exist_ok=True)

    watch = TowerWatch(fleet, cfg)
    watch.start()
    end = time.monotonic() + args.duration if args.duration > 0 else float("inf")
    last_log = 0.0
    try:
        while time.monotonic() < end:
            tip = watch.latest_tip()
            if tip is not None and time.monotonic() - last_log > 3.0:
                pt = watch.search_point(tip)
                log.info("TIP [%s] %.6f,%.6f +/-%.0f m (%s) -> search %.6f,%.6f",
                         tip.source, tip.lat, tip.lon, tip.sigma_m,
                         ",".join(tip.towers), pt[0], pt[1])
                last_log = time.monotonic()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        watch.stop()
        contacts = watch.contacts()
        with open(os.path.join(run_dir, "contacts.jsonl"), "w") as fh:
            for c in contacts:
                fh.write(_contact_json(c) + "\n")
        fleet.shutdown()
    print(f"\nTower scan complete: {len(contacts)} contact(s) in {run_dir}")
    return 0


def _contact_json(c: TowerContact) -> str:
    import json
    return json.dumps({
        "tower": c.tower, "t_sim": c.t_sim, "bearing_deg": c.bearing_deg,
        "elevation_deg": c.elevation_deg, "lat": c.lat, "lon": c.lon,
        "error_radius_m": c.error_radius_m, "ground_range_m": c.ground_range_m,
        "score": c.score, "confirmed": c.confirmed, "hits": c.hits})


if __name__ == "__main__":
    sys.exit(main())
