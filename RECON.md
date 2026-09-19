# RECON — ArcticSim infrastructure (Phase 0)

Recon as of the live WireGuard sim. **Sources:** the real upstream repo
(`github.com/Dominion-Dynamics/arctic-sim`, cloned to `/tmp/opencode/arctic-sim`),
the running sim APIs, and live MAVLink/camera/track probes. The competition
slides are `/home/malek/Downloads/ArcticSim.pdf`.

> The prompt said "read README, docker-compose.yml, .env.example, and the source
> tree". None of those were in this repo (`malekhammoud/Drone-Sim` only had the
> keyboard script), so I cloned the real upstream repo and verified everything
> against the running sim. Findings below are measured, not assumed.

---

## 1. Architecture

`docker compose` with a `10.23.0.0/24` bridge network. No ROS. Gazebo Classic +
gzweb + ArduPilot SITL, shipped by ArcticSim (extracted from Damn Vulnerable
Drone). World axes are **EPSG:3413 polar-stereographic** metres offset by the
site centre; a right-click in the UI yields world `x y z` and true `lat lon`.

| service (container) | compose IP | role |
|---|---|---|
| `terrain` | 10.23.0.10 | one-shot: downloads DEM/imagery, builds world, exits |
| `sim` (`arctic-sim`) | 10.23.0.5 | Gazebo Classic + gzweb + **all** camera sensors |
| `control` (`arctic-control`) | 10.23.0.20 | HTTP API on **8090** (env/status/reset/rebuild/logs) |
| `quadcopter` | 10.23.0.100 | ArduCopter SITL |
| `fixed-wing` | 10.23.0.101 | ArduPlane SITL |
| `boat` | 10.23.0.102 | reserved; no model bundled (idles) |
| `tower-1` | 10.23.0.103 | ArduPilot AntennaTracker |
| `tower-2` | 10.23.0.104 | ArduPilot AntennaTracker |
| `rover` | 10.23.0.105 | ArduRover skid-steer (not rostered here) |

**Reachable from our WireGuard client at `10.99.1.1`** (tunnel peer;
`10.99.1.0/24`, our addr `10.99.1.4`). Assets are reached on the host-published
ports. Rule from the repo: *page host + asset host port*.

Protocols: ArduPilot over **MAVLink** (UDP `udpin` listeners + TCP), **MJPEG/HTTP**
cameras, **gzweb WebSocket** (Gazebo transport) for sim time/poses, **HTTP/JSON**
for control and the track-submission API.

## 2. Cameras — programmatic source

Gazebo camera sensors are rendered server-side in the `sim` container and served
by `sim/plugins/CameraStreamPlugin.cc` as **MJPEG over HTTP** (chosen over RTSP;
OpenCV one-liner: `cv2.VideoCapture("http://host:8630/stream")`). Endpoints:

| path | content |
|---|---|
| `/snapshot.jpg` | one JPEG frame (non-blocking, best for CV grabs) |
| `/stream` | `multipart/x-mixed-replace; boundary=arcticframe` MJPEG |
| `/` | HTML wrapper |

Ports are `8600 + 10*slot` **on the sim host** (not per asset), proven live:

| asset | camera | URL | verified |
|---|---|---|---|
| quadcopter | gimbal | `http://10.99.1.1:8600/snapshot.jpg` | 66 KB JPEG |
| fixed-wing | FPV | `:8610` | 28 KB JPEG |
| tower-1 | EO | `:8630` | 30 KB JPEG |
| tower-2 | EO | `:8640` | 39 KB JPEG |
| rover | FPV | `:8650` | absent (not rostered) |

Resolution/FOV: **measured from the sensor SDFs**, which differ from the slides
(those quote the 640-wide display size, not the sensor):

| asset | sensor | resolution | HFOV | VFOV |
|---|---|---|---|---|
| quadcopter | `gimbal_small_2d` | 960×720 | 114.59° | 98.88° |
| fixed-wing | `skywalker_x8` | 1280×720 | 68.98° | 42.19° |
| tower-1/2 | `terrain/tower.py` | 1280×720 | 60.0° | 36.0° |
| rover | `rover_front_camera` | 960×720 | 85.94° | 69.90° |

(Verified live: `/snapshot.jpg` returns exactly these sizes.)
**Bottleneck:** encoding is skipped when no client is connected, but the sensor
renders regardless; each live camera costs ~a CPU core. Use `/snapshot.jpg` at a
chosen rate rather than holding `/stream` open when possible.

`CAMERAS` is `live`; `SHADOWS=0`. Fog is **off** (`FOG=0`) and the README warns
camera fog rendering is currently buggy.

## 3. Sim time, reset, poses, ground truth

- **Sim time (authoritative):** gzweb WebSocket `~/world_stats` →
  `{sim_time:{sec,nsec}, real_time:{...}, paused, iterations}`. Verified:
  `sim_time≈4922s`, `real_time≈4925s` (they drift; **use sim_time**).
- **Entity poses / ship ground truth:** gzweb WebSocket `~/pose/info`, one
  entity per message: `{name,id,position:{x,y,z},orientation:{x,y,z,w}}` in world
  metres. The **target ship's model name is `target_vessel`** (verified moving at
  world `x=-381.4, y=124.5, z=0`). **Ground truth exists.** DEV ONLY.
- **Reset:** `POST http://<sim>:8090/api/reset` (recreates sim + all assets;
  non-blocking, 202, poll `/api/status` until `state != "working"`). Also
  `POST /api/rebuild` (regenerates terrain first). `/api/status` gives
  `{state, detail}`; `/api/assets` gives per-role `mavlink`/`camera` booleans.
  **After a reset everything restarts and clients must reconnect.**
- **World↔lat/lon:** the panel's `ps2ll` (polar stereographic, `lat_ts=70°`,
  `lon_0=-45°`, WGS84) with `cx,cy` = centre of `/api/site` `bounds3413`.
  `convergence_deg≈-49.80` at Fort Ross. World `(0,0)` is the site centre.

### Discrepancy vs the prompt
`10.99.0.1` (slides) is a **different** sim instance's WireGuard subnet. **Ours is
`10.99.1.1`** — that is where the track API (`:8010`) and everything else answers.

## 4. MAVLink endpoints, IDs and messages

One ArduPilot instance per container, always instance 0, **stock in-container
ports**; host ports are strided. `SYSID_THISMAV = slot+1`. Endpoints are MAVProxy
`udpin` **listeners — the client must transmit first** (`udpout`, not `udp:`).

| asset | our endpoint | sysid | type | autopilot | mode now | armed |
|---|---|---|---|---|---|---|
| quadcopter | `udpout:10.99.1.1:14550` | 1 | QUADROTOR | ArduCopter | GUIDED | yes |
| fixed-wing | `udpout:10.99.1.1:14560` | 2 | FIXED_WING | ArduPlane | GUIDED | yes |
| tower-1 | `udpout:10.99.1.1:14580` | 4 | ANTENNA_TRACKER | AntennaTracker | MANUAL | yes |
| tower-2 | `udpout:10.99.1.1:14590` | 5 | ANTENNA_TRACKER | AntennaTracker | MANUAL | yes |

(rover 14600 sysid 6; boat 14570 reserved.)

Streams observed (after `REQUEST_DATA_STREAM(ALL,4Hz)`): `ATTITUDE`,
`GLOBAL_POSITION_INT`, `LOCAL_POSITION_NED`, `SYSTEM_TIME`, `SYS_STATUS`,
`BATTERY_STATUS`, `GPS_RAW_INT`, `VFR_HUD`, `SERVO_OUTPUT_RAW`, `RC_CHANNELS`,
`NAV_CONTROLLER_OUTPUT`, `EKF_STATUS_REPORT`, `SIMSTATE`, `WIND`; plane adds
`AOA_SSA`, `POSITION_TARGET_GLOBAL_INT`.

**Critical:** SITL's `SERIAL0` blocks until a GCS attaches, so a vehicle only
steps while someone is connected. Our connections literally drive the sim.

### Commands (verified)
- **Copter:** `mode GUIDED` → `arm` → `MAV_CMD_NAV_TAKEOFF(alt)` **immediately**
  (arm auto-disarms after ~3 s). `goto(lat,lon,alt)` via
  `SET_POSITION_TARGET_GLOBAL_INT` (`MAV_FRAME_GLOBAL_RELATIVE_ALT_INT`).
  `land()` = `mode LAND`, `rtl()` = `mode RTL`.
  Velocity via `SET_POSITION_TARGET_LOCAL_NED` (`BODY_OFFSET_NED`, ~30 Hz) — this
  is what the existing keyboard script uses and it is proven.
  **`MAV_CMD_DO_REPOSITION` is UNSUPPORTED here** (`MAV_RESULT_UNSUPPORTED`,
  verified on copter *and* plane) — use the global setpoint, which is also what
  MAVProxy's `guided` uses.
- **Plane:** `mode GUIDED` → `arm` → `mode TAKEOFF` (it climbs and circles); then
  `goto` = loiter around the point. Cannot hover.
- **Tower (AntennaTracker):** `MAV_CMD_DO_SET_SERVO` servo 1 = pan, servo 2 =
  tilt — **verified live**: `MAV_RESULT_ACCEPTED` and `SERVO_OUTPUT_RAW` changed
  to exactly the requested PWM. `set_mode("SCAN")` works (mode → `SCAN`).
- **Quad gimbal: NOT controllable.** `copter.parm` sets `MNT1_TYPE 0` and no
  mount params are exposed, so the "quadcopter gimbal" camera is a **fixed**
  camera. `Pose.gimbal_*` stays `None`; there is no gimbal command surface.

## 5. Track submission API

`http://10.99.1.1:8010/api/tracks`. Verified create/update/list:

- `POST {"name","lat","lon"}` → `{"ok":true,"created":true,"name","uuid","lat","lon","timestamp"}`
- `POST {"name",...,"heading","speed"}` → `created:false`, updates, bumps `fixes`
- `GET` → `{"ok":true,"count","tracks":[{name,uuid,disposition,vessel,lat,lon,heading,speed,updated,fixes}]}`

Every track is tagged `disposition:"hostile"`, `vessel:"kingston-class-mcdv"`.
No delete endpoint. Rate-limit client-side.

## 6. Existing keyboard script (reuse)

`/home/malek/dev/drone-sim/keyboard_control.py` — reuse:
`udpout` connect + **GCS heartbeat loop** (required; the listen sockets stay mute
until we transmit), `BODY_OFFSET_NED` velocity control at ~30 Hz, slew-limited
commands, `MAV_CMD_NAV_TAKEOFF`, `mode LAND`, background-free polling, and the
`:14550` master. It already handles the disarmed-after-reset case with a `T` key.

## 7. Live `.env` (roster)

```
SITE fort_ross  71.991960,-94.822428  extent 6500 m  grid 1025
ASSET_1=copter,quadcopter,71.995807,-94.839300
ASSET_2=tower,tower-1,71.980671,-94.853711
ASSET_3=plane,fixed-wing,71.998195,-94.841967,>71.997790,-94.846245
ASSET_5=tower,tower-2,72.011778,-94.804721
SHIP=1 SHIP_MOVING=1 SHIP_SPEED=3.0 SHIP_START=random SHIP_MODEL=fishing_vessel
SPEEDUP=1 PHYSICS_RATE=250 LOOP_RATE=150 CAMERAS=live FOG=0 SHADOWS=0
```

### Discrepancy vs the prompt
Prompt says terrain ≈ **25 km × 2 km**; the live site is a **6.5 km square**
(`SITE_EXTENT=6500`). Trust the sim: plan against 6.5 km. The strait and ship are
inside it.

## 8. Decisions

- **Camera:** MJPEG HTTP, `/snapshot.jpg` for polling, `/stream` for OpenCV —
  chosen because it is a documented, stable HTTP endpoint (no browser scraping).
- **Sim time:** gzweb `~/world_stats`; fall back to MAVLink `SYSTEM_TIME`.
- **Reset detection:** poll `/api/status`; on `working→idle` (or heartbeat loss)
  reconnect vehicles/cameras/WS.
- **Ground truth:** gzweb `~/pose/info` `target_vessel`, **DEV ONLY**, gated on
  `ARCTICSIM_DEV=1`.
- **Ports are discovered, not guessed** — the table above is measured.