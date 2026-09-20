# API reference

Everything importable from `arcticlib`. Signatures are copied from the source;
if they ever diverge, the source wins.

```python
from arcticlib import (Config, CameraSpec, AssetSpec, load_config,
                       Georef, bearing_deg, distance_m, destination,
                       ps_to_latlon, latlon_to_ps,
                       Pose, Frame, Detection, Battery, AssetStatus,
                       Vehicle, Copter, Plane, Tower, connect_vehicle,
                       CameraSource, TrackClient, SimClient, Fleet)
```

Not re-exported on purpose: `arcticlib.groundtruth` (dev-only) and
`arcticlib.mock` (dev/test only). Import them by full path.

---

## `arcticlib.config`

```python
from arcticlib.config import load_config, Config, AssetSpec, CameraSpec
```

### `load_config(path: str = "config.yaml") -> Config`

Resolves configuration in increasing priority: built-in defaults → `config.yaml`
(if PyYAML is installed and the file exists) → `ARCTICSIM_*` environment
variables. Also builds the asset roster.

**Environment variables**

| Variable | Meaning | Default |
|---|---|---|
| `ARCTICSIM_HOST` | sim host | `127.0.0.1` |
| `ARCTICSIM_CONTROL_PORT` | control API | `8090` |
| `ARCTICSIM_TRACKS_PORT` | track API | `8010` |
| `ARCTICSIM_GZWEB_PORT` | gzweb HTTP/WS | `8080` |
| `ARCTICSIM_ORIGIN_LAT` / `_LON` | site centre | `71.991960` / `-94.822428` |
| `ARCTICSIM_SITE_NAME` | site name | `fort_ross` |
| `ARCTICSIM_SITE_EXTENT` | extent, m | `6500` |
| `ARCTICSIM_CONVERGENCE_DEG` | grid vs true north | `-49.804793` |
| `ARCTICSIM_PS_CENTRE_X` / `_Y` | world `(0,0)` in EPSG:3413 | Fort Ross |
| `ARCTICSIM_HEARTBEAT_TIMEOUT` | link watchdog, s | `5` |
| `ARCTICSIM_POSE_HISTORY_S` | pose ring buffer, s | `30` |
| `ARCTICSIM_COMMAND_TIMEOUT` | default ACK wait, s | `5` |
| `ARCTICSIM_<ASSET>_PORT` | per-asset UDP port | see RECON |
| `ARCTICSIM_<ASSET>_SYSID` | per-asset sysid | see RECON |
| `ARCTICSIM_<ASSET>_CAM_PORT` | per-asset camera port | see RECON |

`<ASSET>` is the uppercased name with `-` → `_`, e.g.
`ARCTICSIM_FIXED_WING_PORT`, `ARCTICSIM_TOWER_1_CAM_PORT`.

### `Config`

| Field | Type | Notes |
|---|---|---|
| `host` | str | sim host |
| `control_port`, `tracks_port`, `gzweb_port` | int | 8090, 8010, 8080 |
| `origin_lat`, `origin_lon`, `site_name`, `site_extent_m` | | site |
| `convergence_deg` | float | grid-north offset from true north |
| `ps_centre_x`, `ps_centre_y` | float | world origin in EPSG:3413 |
| `assets` | `dict[str, AssetSpec]` | roster |
| `heartbeat_timeout`, `pose_history_s`, `command_timeout` | float | tuning |

Methods: `asset(name) -> AssetSpec` (raises `KeyError`), `url(port) -> str`.

### `AssetSpec`

`name, kind, sysid, host, port, camera`. `kind` is `copter | plane | tower | rover`.
Property `master -> str` gives the pymavlink string, e.g. `udpout:127.0.0.1:14550`.

### `CameraSpec`

`asset, port, width, height, hfov_deg, vfov_deg, host`.

* `snapshot_url -> str` — `http://<host>:<port>/snapshot.jpg`
* `stream_url -> str` — `http://<host>:<port>/stream`
* `intrinsics() -> dict` — `{fx, fy, cx, cy, width, height, hfov_deg, vfov_deg}`

---

## Geography — `arcticlib.geo`

```python
from arcticlib.geo import (Georef, distance_m, bearing_deg, destination,
                           ps_to_latlon, latlon_to_ps)
```

### Free functions

| Function | Returns |
|---|---|
| `distance_m(lat1, lon1, lat2, lon2)` | haversine metres |
| `bearing_deg(lat1, lon1, lat2, lon2)` | initial true bearing, `[0, 360)` |
| `destination(lat, lon, bearing, dist_m)` | `(lat, lon)` along a true bearing |
| `ps_to_latlon(x, y)` | EPSG:3413 metres → `(lat, lon)` |
| `latlon_to_ps(lat, lon)` | `(lat, lon)` → EPSG:3413 metres |

### `Georef(lat0, lon0, alt0=0.0, ps_centre_x=0.0, ps_centre_y=0.0)`

| Method | Returns |
|---|---|
| `to_enu(lat, lon, alt=0)` | `(east, north, up)` metres from the origin |
| `to_latlon(east, north, up=0)` | `(lat, lon, alt)` |
| `world_to_latlon(x, y)` | `(lat, lon)` — sim Gazebo world → true lat/lon |
| `latlon_to_world(lat, lon)` | `(x, y)` — true lat/lon → sim world |
| `Georef.bearing(...)` / `Georef.distance(...)` | static passthroughs |

`to_enu`/`to_latlon` use the WGS84 meridian and prime-vertical radii at the
origin (accurate well under a metre over a few km). `world_to_latlon` and friends
need `ps_centre_x/y` (they come from `Config`).

**Pitfall:** ENU is a tangent plane while `distance_m` is great-circle; over
~2 km they differ by ~0.3%. Assert with that tolerance, not 0.

---

## The contract — `arcticlib.types`

All angles are radians unless the name ends in `_deg`. `t_sim` is Gazebo sim
seconds; `t_wall` is `time.monotonic()`.

### `Pose`

```
asset, t_sim, t_wall, lat, lon,
alt_rel=0, alt_amsl=0,
roll=0, pitch=0, yaw=0,
vx=0, vy=0, vz=0,             # NED, m/s
gimbal_pitch=None, gimbal_yaw=None
```
Property `speed` — horizontal ground speed.

### `Frame`

```
asset, t_sim, t_wall, image,   # image: np.ndarray BGR HxWx3 uint8
width, height, hfov_deg, vfov_deg
```
Property `shape`.

### `Detection`

```
asset, t_sim, lat, lon, sigma_m, conf,
bearing_only=False, extras={}
```
`bearing_only=True` marks a single-camera bearing with no range. `sigma_m` is
1-sigma position uncertainty in metres. `conf` is in `[0, 1]`.

### `Battery`

`voltage_v=None, current_a=None, remaining_pct=None`.

### `AssetStatus`

`asset, connected, mode="?", armed=False, battery=None,
last_heartbeat_age=inf`.

---

## Vehicles — `arcticlib.vehicle`

```python
from arcticlib.vehicle import Vehicle, Copter, Plane, Tower, connect_vehicle
```

### `Vehicle(spec, config=None, sim_time_fn=None, auto_connect=True)`

Base class. `spec` is an `AssetSpec`; `sim_time_fn` is any `() -> float`
(`Fleet` passes `SimClient.sim_time`).

**Cached state** (thread-safe, read directly):

| Attribute | Meaning |
|---|---|
| `connected` | heartbeat within `heartbeat_timeout` |
| `mode`, `armed` | flight mode name, arm state |
| `lat`, `lon`, `alt_rel`, `alt_amsl` | position |
| `roll`, `pitch`, `yaw` | attitude |
| `vn`, `ve`, `vd` | NED velocity, m/s |
| `servo` | `{channel: raw_pwm}` |
| `params` | `{NAME: value}` cache |
| `statustexts` | `deque[(t_wall, text)]` |

**Methods**

| Method | Notes |
|---|---|
| `connect() -> bool` | open link, start reader; idempotent, never raises |
| `pose() -> Pose` | latest snapshot |
| `pose_at(t, clock="sim") -> Pose \| None` | linear interpolation on `"sim"` or `"wall"`; clamps out-of-range to nearest |
| `send_global_target(lat, lon, alt, yaw=None, repeats=5) -> bool` | GUIDED position setpoint (unacknowledged) |
| `get_param(name, timeout=3) -> float \| None` | fetches once, then cached |
| `mode_number(name) -> int \| None` | per-type mode map |
| `set_mode(name, timeout=6) -> bool` | confirms via HEARTBEAT |
| `arm(timeout=60) -> bool` | retries while pre-arm settles |
| `disarm(timeout=5) -> bool` | |
| `status() -> AssetStatus` | |
| `shutdown()` | stop reader, close link |

**Do not** call `recv_match` yourself — the reader thread owns the socket.

### `Copter`

| Method | Notes |
|---|---|
| `takeoff(alt=15, timeout=120) -> bool` | GUIDED → arm → `NAV_TAKEOFF`; retries while the EKF settles and the autopilot auto-disarms |
| `wait_alt(alt_rel, timeout=45) -> bool` | |
| `goto(lat, lon, alt, timeout=6) -> bool` | `SET_POSITION_TARGET_GLOBAL_INT` |
| `land() -> bool` / `rtl() -> bool` | mode switch |
| `set_speed(mps) -> bool` | `WPNAV_SPEED` |
| `has_gimbal() -> bool` | always `False` here (`MNT1_TYPE 0`) |
| `gimbal(pitch_deg=None, yaw_deg=None) -> bool` | always `False` |

### `Plane`

| Method | Notes |
|---|---|
| `takeoff(alt=80, timeout=120) -> bool` | GUIDED → arm → mode `TAKEOFF`, retries |
| `goto(lat, lon, alt, timeout=6) -> bool` | loiters around the point |
| `loiter(lat, lon, alt, radius=None) -> bool` | `goto` plus optional `WP_LOITER_RAD` |
| `set_airspeed(mps) -> bool` | `DO_CHANGE_SPEED` + `AIRSPEED_CRUISE` |
| `rtl() -> bool` | |

### `Tower`

| Method | Notes |
|---|---|
| `set_pan_pwm(pwm, timeout=4) -> bool` | `DO_SET_SERVO` channel 1 |
| `set_tilt_pwm(pwm, timeout=4) -> bool` | channel 2 |
| `point(az_deg, el_deg) -> bool` | local azimuth (0 = pan 1500 µs) and elevation (+ up) |
| `aim_at(lat, lon, alt=0) -> bool` | uses own position + `base_yaw_deg` from calib |
| `scan() -> bool` / `stop_scan() -> bool` | mode `SCAN` / `MANUAL` |
| `center() -> bool` | `point(0, 0)` |
| `load_calibration(path=None) -> dict` | reads `calib/tower_<name>.json`, falls back to model defaults |

Effective servo range is **1100–1900 µs** (`SERVO1/2_MIN/MAX`). Calibration
defaults: pan `1500 µs = 0° @ 2.2222 µs/°`, tilt `1420 µs = level @ 10.667 µs/°`.

### `connect_vehicle(spec, config=None, sim_time_fn=None) -> Vehicle`

Factory selecting `Copter` / `Plane` / `Tower` by `spec.kind`.

---

## Cameras — `arcticlib.camera`

### `CameraSource(spec, sim_time_fn=None, timeout=5.0)`

| Method | Notes |
|---|---|
| `grab() -> Frame \| None` | one `/snapshot.jpg` GET + decode; `None` on failure |
| `latest() -> Frame \| None` | most recent polled frame; never blocks |
| `poll(rate_hz=2.0)` | start the background poller |
| `stream() -> Iterator[Frame]` | blocking MJPEG generator (OpenCV) |
| `stop()` | stop poller, close session |

Frames are stamped with `sim_time_fn()` if given, else `time.monotonic()`.

---

## Tracks — `arcticlib.tracks`

### `TrackClient(base_url, min_interval=0.5, timeout=5.0, retries=2)`

| Method | Notes |
|---|---|
| `post(name, lat, lon, heading=None, speed=None) -> dict \| None` | create on first call, update thereafter; rate-limited |
| `list() -> list[dict]` | `[]` on failure |

Response on create: `{"ok", "created": true, "name", "uuid", "lat", "lon",
"timestamp"}`. Updates set `created: false` and increment `fixes`.

---

## Sim control — `arcticlib.simctl`

### `SimClient(config=None, pose_callback=None, autostart=True)`

| Method | Notes |
|---|---|
| `sim_time() -> float` | Gazebo sim seconds from `~/world_stats`; wall time until first sample |
| `world_stats() -> dict` | last `world_stats` payload |
| `status() -> dict` | `/api/status` (`state`, `detail`), `state="unreachable"` on error |
| `assets() -> list[dict]` | `/api/assets` (per-role `mavlink`, `camera`, ports) |
| `ready() -> bool` | every rostered asset answers MAVLink |
| `reset() -> bool` | `POST /api/reset` (non-blocking server-side) |
| `wait_until_ready(timeout=240, poll=5, wait_for_reset=False) -> bool` | blocks until ready; with `wait_for_reset` it first waits for `working` so it cannot return the pre-reset state |
| `start()` / `stop()` | clock thread |
| `clocks` | count of `world_stats` samples seen |

`pose_callback(body)` is called with each `~/pose/info` entity (used by
`GroundTruth`).

---

## Fleet — `arcticlib.fleet`

### `Fleet.from_config(config=None, connect=True) -> Fleet`

Attributes after connect: `fleet.quad`, `fleet.plane`, `fleet.tower1`,
`fleet.tower2` (or `None` if not rostered), `fleet.vehicles` (dict),
`fleet.cams` (dict), `fleet.tracks`, `fleet.sim`, `fleet.config`.

| Method | Notes |
|---|---|
| `wait_ready(timeout=20) -> bool` | all connected vehicles have a heartbeat |
| `vehicle(asset) -> Vehicle \| None` | |
| `pose(asset)` / `pose_at(asset, t, clock="sim")` | |
| `frame(asset, poll=False, rate_hz=2.0)` | latest frame; `poll=True` starts the poller |
| `status() -> dict` | sim status, sim time, per-asset `AssetStatus`, camera liveness |
| `on_reset(wait=240) -> bool` | wait for the sim after a Reset and reconnect |
| `shutdown()` | stop every thread and link |

---

## Mock — `arcticlib.mock`

### `MockFleet(config=None, speedup=100.0)`

Drop-in for `Fleet` with the same public methods and attributes. Synthetic 2D
strait, kinematic assets, a moving ship (`target_vessel` equivalent exposed via
`_ship_latlon()`), rendered frames, and a `MockTrackClient`. Nothing touches the
network, so the CV/planner/tracker teammates can iterate fast.

```python
from arcticlib.mock import MockFleet
fleet = MockFleet(speedup=100)      # sim time runs 100×
q = fleet.quad
q.takeoff(20); q.goto(*latlon, 20)
frame = fleet.frame("quadcopter")
```

---

## Ground truth (dev only) — `arcticlib.groundtruth`

```python
import os; os.environ["ARCTICSIM_DEV"] = "1"
from arcticlib.groundtruth import GroundTruth, project_point
```

### `GroundTruth(config=None, georef=None, ship_name="target_vessel")`

Raises `RuntimeError` unless `ARCTICSIM_DEV=1`, and logs a loud warning.

| Method | Returns |
|---|---|
| `ship_world() -> (x, y, z) \| None` | true Gazebo world metres |
| `ship_latlon() -> (lat, lon) \| None` | |
| `ship_pose() -> dict \| None` | `{lat, lon, heading, speed, world}`; heading/speed smoothed over a ~1 s window |
| `stop()` | |

### `project_point(point_world, cam_world, cam_yaw_deg, intrinsics, cam_pitch_deg=0.0, cam_roll_deg=0.0) -> (u, v) \| None`

Approximate pinhole projection for auto-labelling. `None` when behind the camera.
Good enough for labels, not for calibration. Never use in the judged run.