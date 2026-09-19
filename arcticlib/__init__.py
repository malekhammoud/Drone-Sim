"""arcticlib — infrastructure for the ArcticSim drone challenge.

Other teammates import this package. The public surface is re-exported below;
see README.md for a 30-line quickstart and RECON.md for the measured sim facts.
"""
from .config import Config, CameraSpec, AssetSpec, load_config
from .geo import Georef, bearing_deg, distance_m, destination, ps_to_latlon, latlon_to_ps
from .types import Pose, Frame, Detection, Battery, AssetStatus
from .vehicle import Vehicle, Copter, Plane, Tower, connect_vehicle
from .camera import CameraSource
from .tracks import TrackClient
from .simctl import SimClient
from .fleet import Fleet

__all__ = [
    "Config", "CameraSpec", "AssetSpec", "load_config",
    "Georef", "bearing_deg", "distance_m", "destination", "ps_to_latlon", "latlon_to_ps",
    "Pose", "Frame", "Detection", "Battery", "AssetStatus",
    "Vehicle", "Copter", "Plane", "Tower", "connect_vehicle",
    "CameraSource", "TrackClient", "SimClient", "Fleet",
]
