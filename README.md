# drone-sim / arcticlib

Infrastructure for the ArcticSim (Hack The North) challenge: connect to the four
ArduPilot assets, read telemetry and camera frames, command flight, control the
tower masts, and submit tracks. Other teammates build CV, planning and tracking
on top of this package.

Measured sim facts (endpoints, cameras, ground truth, quirks) are in
[RECON.md](RECON.md). **Read that first.**

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | components, data flow, threading, coordinate frames, file map |
| [docs/API.md](docs/API.md) | reference for every public class and method |
| [docs/TOOLS.md](docs/TOOLS.md) | the scripts in `tools/` and how to run them |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | setup, tests, conventions, extending, troubleshooting |
| [RECON.md](RECON.md) | measured sim facts |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 30-line quickstart

```python
from arcticlib import Fleet, load_config
from arcticlib.geo import destination

cfg = load_config()                      # ARCTICSIM_* env vars override
fleet = Fleet.from_config(cfg)           # connects 4 MAVLink links + cameras
fleet.wait_ready(20)

q = fleet.quad                           # Copter; fleet.plane, fleet.tower1/2
q.takeoff(20)                            # GUIDED -> arm -> takeoff (atomic)
lat, lon = destination(q.pose().lat, q.pose().lon, bearing=45, dist_m=400)
q.goto(lat, lon, 25)                     # fly there and hold

pose = fleet.pose("quadcopter")          # latest Pose (lat/lon/alt/yaw/…)
p_lag = fleet.pose_at("quadcopter", pose.t_sim - 0.1)   # interpolated by sim time

frame = fleet.frame("quadcopter", poll=True)            # non-blocking latest
print(frame.width, frame.height, frame.image.shape)     # BGR ndarray

fleet.tower1.point(az_deg=30, el_deg=-5) # calibrated pan/tilt
fleet.tower2.scan()

fleet.tracks.post("Sierra One", 71.9965, -94.8448, heading=315, speed=6.5)
print(fleet.tracks.list())

fleet.shutdown()
```

Run `python tools/smoke_test.py` to verify every piece against the live sim
(23 checks: links, poses, copter/plane flight, towers, cameras, tracks).

## Layout

```
arcticlib/
  config.py   endpoints, camera intrinsics, site origin (env / config.yaml)
  geo.py      lat/lon <-> local ENU metres, bearings, sim world frame
  types.py    Pose, Frame, Detection, Battery, AssetStatus  (the contract)
  vehicle.py  Vehicle + Copter/Plane/Tower: reconnect, pose history, commands
  camera.py   CameraSource: latest() polling + stream() MJPEG
  tracks.py   TrackClient: post/list with retries and rate limiting
  simctl.py   SimClient: sim_time(), reset(), wait_until_ready()
  groundtruth.py  target vessel's true pose — DEV ONLY, gated on ARCTICSIM_DEV=1
  fleet.py    Fleet.from_config(): everything wired together
  mock.py     MockFleet: identical API, synthetic world, no network
tools/
  smoke_test.py     end-to-end PASS/FAIL against the live sim
  record.py         labelled dataset recorder (JPEG + JSONL sidecar)
  dashboard_cli.py  live terminal ENU map + status table
  calibrate_tower.py  verify/write calib/tower_<name>.json
tests/          unit tests for geo and pose interpolation
```

## Tests

```bash
python -m unittest discover -s tests
```

## Simulation control

If `14550` goes silent, check `/api/status` before blaming the tunnel (see
RECON.md §3): the sim server may just be idle. `fleet.sim.reset()` restarts it
and `fleet.sim.wait_until_ready()` blocks until the assets answer again; the
reader threads reconnect on their own.

## Keyboard flight controller

The original manual controller is still here — `./fly.sh` (arrows fly, WASD
altitude/yaw, T takeoff, L land). See the docstring in `keyboard_control.py`.

## Ground truth (dev only)

`arcticlib/groundtruth.py` reads the target vessel's true pose from the gzweb
`~/pose/info` stream. It **refuses to construct** unless `ARCTICSIM_DEV=1` and
logs a loud warning. It exists for auto-labelling and offline scoring only —
using it in the judged run is cheating, so never import it from autonomous code.