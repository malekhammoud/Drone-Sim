"""Single source of truth for endpoints, camera intrinsics and the site origin.

Everything is overridable, in increasing priority:

1. built-in defaults (measured from the live sim — see RECON.md)
2. a ``config.yaml`` next to the repo root (if PyYAML is installed)
3. ``ARCTICSIM_*`` environment variables

Priority order matters: env wins over the file, which wins over defaults, so a
teammate can override one port for a one-off run without editing anything.

The defaults describe the cloud sim as reached over WireGuard. For a local
``docker compose`` deployment run with ``ARCTICSIM_HOST=localhost``.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Optional

DEFAULT_HOST = "10.99.1.1"

# Host-published ports on the sim machine (RECON.md §4). Reach each asset at
# ``udpout:<host>:<port>``.
DEFAULT_ASSET_PORTS = {
    "quadcopter": 14550,
    "fixed-wing": 14560,
    "tower-1": 14580,
    "tower-2": 14590,
    "rover": 14600,
}
DEFAULT_SYSIDS = {
    "quadcopter": 1,
    "fixed-wing": 2,
    "tower-1": 4,
    "tower-2": 5,
    "rover": 6,
}
# MJPEG camera ports live on the sim container, slot-keyed 8600 + 10*slot.
DEFAULT_CAM_PORTS = {
    "quadcopter": 8600,
    "fixed-wing": 8610,
    "tower-1": 8630,
    "tower-2": 8640,
    "rover": 8650,
}
DEFAULT_KINDS = {
    "quadcopter": "copter",
    "fixed-wing": "plane",
    "tower-1": "tower",
    "tower-2": "tower",
    "rover": "rover",
}
# Camera geometry, **measured from the sim's own sensor SDFs** (RECON.md §2):
#   quad  gimbal_small_2d  horizontal_fov 2.000 rad @ 960x720
#   plane skywalker_x8     horizontal_fov 1.204 rad @ 1280x720
#   towers tower.py        horizontal_fov 1.047 rad @ 1280x720
#   rover rover_front_cam  horizontal_fov 1.500 rad @ 960x720
# VFOV is derived from HFOV by the pinhole aspect relation. Note the slides
# quote 640x480 / 640x360 — that is the display size, not the sensor. Trust the
# sim: these are the frames /snapshot.jpg actually returns.
DEFAULT_CAMERA_GEOM = {
    "quadcopter": (960, 720, 114.59, 98.88),
    "fixed-wing": (1280, 720, 68.98, 42.19),
    "tower-1": (1280, 720, 60.0, 36.0),
    "tower-2": (1280, 720, 60.0, 36.0),
    "rover": (960, 720, 85.94, 69.90),
}


def _env(name: str, default):
    return os.environ.get(f"ARCTICSIM_{name}", default)


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ[f"ARCTICSIM_{name}"])
    except (KeyError, ValueError):
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ[f"ARCTICSIM_{name}"])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class CameraSpec:
    """One camera stream plus the intrinsics implied by its FOV/resolution."""

    asset: str
    port: int
    width: int
    height: int
    hfov_deg: float
    vfov_deg: float
    host: str = DEFAULT_HOST

    @property
    def snapshot_url(self) -> str:
        """Single-JPEG endpoint — the right one for polling CV grabs."""
        return f"http://{self.host}:{self.port}/snapshot.jpg"

    @property
    def stream_url(self) -> str:
        """MJPEG endpoint — the right one for ``cv2.VideoCapture``."""
        return f"http://{self.host}:{self.port}/stream"

    def intrinsics(self) -> dict:
        """Pinhole intrinsics in pixels from the horizontal/vertical FOV."""
        fx = (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        fy = (self.height / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)
        return {"fx": fx, "fy": fy, "cx": self.width / 2.0, "cy": self.height / 2.0,
                "width": self.width, "height": self.height,
                "hfov_deg": self.hfov_deg, "vfov_deg": self.vfov_deg}


@dataclass(frozen=True)
class AssetSpec:
    """A controllable asset: its MAVLink endpoint, kind, sysid and camera."""

    name: str
    kind: str                 # copter | plane | tower | rover
    sysid: int
    host: str
    port: int                 # host GCS udp port
    camera: Optional[CameraSpec] = None

    @property
    def master(self) -> str:
        """pymavlink connection string. udpout, not udp: the sim listens."""
        return f"udpout:{self.host}:{self.port}"


@dataclass
class Config:
    """Resolved configuration for one process."""

    host: str = DEFAULT_HOST
    control_port: int = 8090
    tracks_port: int = 8010
    gzweb_port: int = 8080
    origin_lat: float = 71.991960
    origin_lon: float = -94.822428
    site_name: str = "fort_ross"
    site_extent_m: float = 6500.0
    # World (EPSG:3413) coordinates of the site centre, used to map Gazebo world
    # metres to lat/lon. Filled from /api/site when available; the fallback is
    # the published Fort Ross bounds centre.
    ps_centre_x: float = -1502373.733
    ps_centre_y: float = -1268596.501
    assets: dict = field(default_factory=dict)

    # Timeouts / rates (seconds, Hz).
    heartbeat_timeout: float = 5.0
    pose_history_s: float = 30.0
    command_timeout: float = 5.0

    def asset(self, name: str) -> AssetSpec:
        try:
            return self.assets[name]
        except KeyError:
            raise KeyError(f"unknown asset {name!r}; have {sorted(self.assets)}") from None

    def url(self, port: int) -> str:
        return f"http://{self.host}:{port}"


def _apply_yaml(cfg: Config, path: str) -> Config:
    try:
        import yaml  # optional dependency
    except ImportError:
        return cfg
    if not os.path.exists(path):
        return cfg
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    for key, val in data.items():
        if key == "assets" and isinstance(val, dict):
            for name, spec in val.items():
                if name in cfg.assets and isinstance(spec, dict):
                    cfg.assets[name] = replace(cfg.assets[name], **spec)
        elif hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg


def load_config(path: str = "config.yaml") -> Config:
    """Build a :class:`Config` from defaults, ``config.yaml`` and env vars."""
    host = _env("HOST", DEFAULT_HOST)
    cfg = Config(
        host=host,
        control_port=_envi("CONTROL_PORT", 8090),
        tracks_port=_envi("TRACKS_PORT", 8010),
        gzweb_port=_envi("GZWEB_PORT", 8080),
        origin_lat=_envf("ORIGIN_LAT", 71.991960),
        origin_lon=_envf("ORIGIN_LON", -94.822428),
        site_name=_env("SITE_NAME", "fort_ross"),
        site_extent_m=_envf("SITE_EXTENT", 6500.0),
        ps_centre_x=_envf("PS_CENTRE_X", -1502373.733),
        ps_centre_y=_envf("PS_CENTRE_Y", -1268596.501),
        heartbeat_timeout=_envf("HEARTBEAT_TIMEOUT", 5.0),
        pose_history_s=_envf("POSE_HISTORY_S", 30.0),
        command_timeout=_envf("COMMAND_TIMEOUT", 5.0),
    )
    cfg = _apply_yaml(cfg, path)

    # Asset roster: names come from the defaults; ports/sysids/cameras can be
    # overridden per asset with ARCTICSIM_<NAME>_PORT etc. (name sanitised).
    assets = {}
    for name, kind in DEFAULT_KINDS.items():
        key = name.upper().replace("-", "_")
        port = _envi(f"{key}_PORT", DEFAULT_ASSET_PORTS[name])
        sysid = _envi(f"{key}_SYSID", DEFAULT_SYSIDS[name])
        cam_port = _envi(f"{key}_CAM_PORT", DEFAULT_CAM_PORTS[name])
        w, h, hf, vf = DEFAULT_CAMERA_GEOM[name]
        cam = CameraSpec(name, cam_port, w, h, hf, vf, host)
        assets[name] = AssetSpec(name, kind, sysid, host, port, cam)
    cfg.assets = assets
    return cfg
