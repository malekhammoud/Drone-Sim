# arcticlib documentation

Infrastructure layer for the ArcticSim / Hack The North challenge. This is what
the codebase is, how it fits together, and how to use it.

| Document | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System overview, components, data flow, threading, coordinate frames, file map |
| [API.md](API.md) | Reference for every public class and method in `arcticlib` |
| [TOOLS.md](TOOLS.md) | The scripts in `tools/` and how to run them |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Setup, tests, conventions, extending the library, sim quirks |
| [`../RECON.md`](../RECON.md) | **Measured sim facts** — endpoints, ports, cameras, ground truth. Read this first. |
| [`../README.md`](../README.md) | 30-line quickstart |

## TL;DR for a teammate

```python
from arcticlib import Fleet, load_config
from arcticlib.geo import destination

fleet = Fleet.from_config(load_config())   # connects 4 MAVLink links + cameras
fleet.wait_ready(20)

q = fleet.quad
q.takeoff(20)
lat, lon = destination(q.pose().lat, q.pose().lon, 45, 400)
q.goto(lat, lon, 25)

pose = fleet.pose("quadcopter")
frame = fleet.frame("quadcopter", poll=True)   # BGR ndarray
fleet.tower1.point(az_deg=30, el_deg=-5)
fleet.tracks.post("Sierra One", 71.9965, -94.8448, heading=315, speed=6.5)
fleet.shutdown()
```

Working without the real sim: swap `Fleet` for `arcticlib.mock.MockFleet` — same
API, no network.

## Status — what is implemented

| Area | State |
|---|---|
| Config, site origin, camera intrinsics, env overrides | done |
| `geo.py` — lat/lon ↔ ENU, bearings, EPSG:3413 world frame | done, unit-tested |
| `types.py` — Pose / Frame / Detection / Battery / AssetStatus | done (stable contract) |
| `vehicle.py` — connect, reconnect, telemetry, pose history, commands | done; verified on all 4 assets |
| `camera.py` — snapshot polling + MJPEG stream | done; verified on all 4 cameras |
| `tracks.py` — create/update/list, retries, rate limiting | done; verified against `:8010` |
| `simctl.py` — sim clock (`~/world_stats`), reset, readiness | done; verified |
| `groundtruth.py` — `target_vessel` true pose | done; **dev only** (`ARCTICSIM_DEV=1`) |
| `fleet.py` — everything wired together | done |
| `mock.py` — `MockFleet`, identical API | done |
| `tools/smoke_test.py` | 23 checks; **23/23 passed when the sim was healthy** |
| `tools/record.py`, `tools/dashboard_cli.py`, `tools/calibrate_tower.py` | done; verified |
| Tower pan/tilt calibration | done; 0 servo-tracking failures |
| Unit tests (`tests/`) | 16 passing |

**Known issue (external):** at the time of writing the live sim's **quadcopter
would not translate in GUIDED** (it climbed and hovered, but `goto` and GUIDED
velocity left it in place, while STABILIZE + RC override moved it), and it
needed ~1–2 min of EKF settling after a Reset before motors would spin. The
library handles the settle time by retrying; the GUIDED horizontal fault is on
the sim side. The fixed-wing and both towers were unaffected. Details in
[`../RECON.md`](../RECON.md) §Sim health caveat.