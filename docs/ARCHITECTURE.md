# Architecture

How the pieces fit together, why they are built this way, and where each
responsibility lives. Measured sim facts live in [`../RECON.md`](../RECON.md).

## The system in one picture

```
                    local docker compose (host 127.0.0.1)
                                    │
                    ┌───────────────┴────────────────┐
                    │        ArcticSim server 127.0.0.1
                    │                                │
   MAVLink (UDP)    │  quadcopter  :14550  sysid 1   │
   ◄────────────────┤  fixed-wing  :14560  sysid 2   │
                    │  tower-1     :14580  sysid 4   │
                    │  tower-2     :14590  sysid 5   │
   MJPEG (HTTP)     │  cameras  8600/8610/8630/8640   │
   ◄────────────────┤                                │
   WebSocket        │  gzweb :8080  ~/world_stats    │  sim clock + true poses
   ◄────────────────┤                                │
   HTTP/JSON        │  control :8090  tracks :8010   │
   ◄────────────────┤                                │
                    └────────────────────────────────┘

                    arcticlib (this machine)
   Fleet ──┬── Vehicle (Copter/Plane/Tower) × 4   ── reader threads
           ├── CameraSource × 4                    ── poll threads
           ├── TrackClient
           └── SimClient                            ── WebSocket clock thread
```

`Fleet` is the only object the rest of the system needs. Everything else is a
component it wires together.

## Components

| Module | Responsibility |
|---|---|
| `config.py` | One place for endpoints, camera intrinsics, site origin. Defaults → `config.yaml` → `ARCTICSIM_*` env. |
| `geo.py` | lat/lon ↔ local ENU metres, bearings, and the sim's EPSG:3413 world frame. |
| `types.py` | `Pose`, `Frame`, `Detection`, `Battery`, `AssetStatus` — **the contract**. |
| `vehicle.py` | Link management, reconnect, telemetry, pose history, commands. |
| `camera.py` | MJPEG snapshot/stream access with a background poller. |
| `tracks.py` | Rate-limited client for the competition track API. |
| `simctl.py` | Sim clock from gzweb, reset, readiness; minimal embedded WebSocket client. |
| `groundtruth.py` | Target vessel's true pose. **DEV ONLY**, gated on `ARCTICSIM_DEV=1`. |
| `fleet.py` | Wires all of the above; exposes `fleet.quad`, `fleet.plane`, … |
| `mock.py` | `MockFleet` — identical API, synthetic world, no network. |

## Data flow

**Telemetry.** Each `Vehicle` owns exactly one background reader thread, which is
the *only* consumer of that MAVLink socket. It parses messages into a
lock-protected latest snapshot and appends a `Pose` to a ring buffer (~last 30 s)
on every position update. Callers read snapshots with `pose()` and interpolate
with `pose_at(t)`. Commands are sent from the caller's thread and confirmed by
`COMMAND_ACK`, which the reader thread routes to a condition variable.

**Frames.** `CameraSource.grab()` does an HTTP GET of `/snapshot.jpg` and
decodes it. `poll(rate_hz)` runs that on a background thread so `latest()` is
non-blocking. `stream()` is a blocking MJPEG generator for OpenCV.

**Clock.** `SimClient` holds one WebSocket to gzweb and reads `~/world_stats`
into `sim_time()`. That value is handed to every vehicle and camera so poses and
frames share one clock. Fallbacks: MAVLink `SYSTEM_TIME` (per vehicle), then
`time.monotonic()`.

**Ground truth.** The same gzweb socket carries `~/pose/info`; `GroundTruth`
filters `target_vessel`. This is dev-only and must never be imported by the
autonomous run.

## Threading model

Threads, not asyncio, because pymavlink is blocking.

| Thread | Owns | Lifetime |
|---|---|---|
| `mav-<asset>` (×4) | reader loop, heartbeat, reconnect | until `shutdown()` |
| `simclock` | gzweb WebSocket | until `stop()` |
| `cam-<asset>` (×N) | snapshot polling | until `stop()` |

Rules that keep it safe:

* **One reader per link.** Never call `recv_match` from your own code on a
  `Vehicle`'s connection; the reader will have consumed the message. Use the
  cached properties (`q.mode`, `q.params`, `q.servo`, `q.statustexts`, `q.pose()`).
* **Commands are thread-safe.** `_send_command` guards the ACK map with a
  condition variable and never blocks the reader.
* **Nothing raises into the caller.** Commands return `False` on failure/timeout.
* **`shutdown()` joins threads** and closes sockets.

## Coordinate frames

This is the single easiest thing to get wrong. See [`API.md`](API.md#geography)
and `geo.py`.

| Frame | Units | Used by | Convert with |
|---|---|---|---|
| lat/lon | degrees | MAVLink, detections, track API | — |
| local ENU | metres east/north/up | filters, dashboards, spacing | `Georef.to_enu` / `to_latlon` |
| sim world | metres (EPSG:3413) | gzweb `~/pose/info`, ground truth | `Georef.world_to_latlon` / `latlon_to_world` |

World axes are polar-stereographic grid axes, **not** true ENU. At Fort Ross grid
north bears −49.80° from true north (`convergence_deg`). `ps_to_latlon` /
`latlon_to_ps` implement the projection exactly as the sim UI does.

At 72 °N a degree of longitude is ~0.31 of a degree of latitude. Never treat
lat/lon as a flat grid by hand.

## Command strategy per vehicle

| Asset | Takeoff | Move | Notes |
|---|---|---|---|
| Copter | GUIDED → arm → `NAV_TAKEOFF` (retries while EKF settles) | `SET_POSITION_TARGET_GLOBAL_INT` | `DO_REPOSITION` is unsupported here |
| Plane | GUIDED → arm → mode `TAKEOFF` | `SET_POSITION_TARGET_GLOBAL_INT` (loiters) | cannot hover |
| Tower | n/a | `DO_SET_SERVO` 1=pan 2=tilt | servos 1100–1900 µs |

Tower pointing is calibrated in `calib/tower_<name>.json`; the fit is derived
from the sim's own plugin maths and verified by `tools/calibrate_tower.py`.

## Failure handling

* **Heartbeat watchdog** — stale heartbeat → `_drop()` → reopen → re-request
  message rates → continue. Never propagates an exception.
* **Reset** — `SimClient.reset()` posts `/api/reset`; `wait_until_ready()` polls
  `/api/status` and `/api/assets` until the fleet answers. `Fleet.on_reset()`
  blocks for that and reconnects.
* **Settle time** — straight after a Reset the EKF takes ~1–2 min. Arming may
  succeed but takeoff motors stay idle and auto-disarm. `takeoff()` retries until
  it actually climbs.

## File map

```
arcticlib/            the package (see Components)
tools/
  smoke_test.py       end-to-end PASS/FAIL against the live sim
  record.py           labelled dataset recorder (JPEG + JSONL)
  dashboard_cli.py    live terminal ENU map + status table
  calibrate_tower.py  verify/write calib/tower_<name>.json
tests/
  test_geo.py         lat/lon, ENU, polar-stereographic round trips
  test_pose.py        pose_at interpolation (incl. yaw wrap)
calib/                tower calibration (committed)
keyboard_control.py   standalone manual flight controller (./fly.sh)
RECON.md              measured sim facts
README.md             30-line quickstart
```