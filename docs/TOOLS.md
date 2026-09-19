# Tools

Scripts under `tools/` (and the two root helpers). All read configuration the
same way as the library, so `ARCTICSIM_HOST=localhost` etc. work everywhere.

Run from the repo root with the venv active, or via `.venv/bin/python`.

---

## `tools/smoke_test.py` — end-to-end health check

The fastest way to answer "is everything working?". Each check is independent and
reports PASS/FAIL; the process exits non-zero if any fail.

```bash
python tools/smoke_test.py                 # full, including flight
python tools/smoke_test.py --no-fly        # telemetry/cameras/tracks only
python tools/smoke_test.py --host localhost --outdir /tmp/frames
```

| Flag | Default | Meaning |
|---|---|---|
| `--host HOST` | config | overrides `ARCTICSIM_HOST` |
| `--no-fly` | off | skip takeoff/goto (safe on a shared sim) |
| `--outdir DIR` | `smoke_frames` | where camera PNGs are written |

Checks:

1. control plane reachable
2. sim clock ticking
3. every **rostered** asset: heartbeat + valid pose + pose history
4. quadcopter takes off (or is already airborne) and flies to a waypoint
5. fixed-wing takes off if grounded and flies to a waypoint
6. tower-1 and tower-2 sweep pan/tilt, verified against `SERVO_OUTPUT_RAW`, and enter `SCAN`
7. one frame from each of the four cameras, saved as PNG
8. a test track (`TestTrack`) is posted and appears in the list

Example tail:

```
  [PASS] sim control plane reachable — state=idle
  [PASS] quadcopter: heartbeat + pose — mode=GUIDED armed=True lat=... alt=41.5m
  [PASS] tower-1: pan/tilt servo — commanded 1800/1150, SERVO_OUTPUT_RAW=1800/1150
  [PASS] camera quadcopter — 960x720 -> smoke_frames/quadcopter.png
  [PASS] tracks: create + list — uuid=entity-... listed=True

23/23 checks passed
```

Non-rostered assets (e.g. `rover`) are skipped and named in the output.

---

## `tools/record.py` — labelled dataset recorder

Grabs frames from chosen assets at N Hz and writes one JPEG per frame plus a
`sidecar.jsonl` line with pose, camera intrinsics, sim time and — **only with
`ARCTICSIM_DEV=1`** — the target vessel's ground-truth position and an
approximate pixel point.

```bash
python tools/record.py --assets quadcopter fixed-wing --hz 2 --duration 60
ARCTICSIM_DEV=1 python tools/record.py --assets quadcopter --hz 1 --duration 300 --out data
```

| Flag | Default | Meaning |
|---|---|---|
| `--assets ...` | all with a camera | assets to record |
| `--hz N` | `2.0` | frames per second per asset |
| `--duration S` | `60` | seconds (`0` = forever) |
| `--out DIR` | `data` | output root; a timestamped subdir is created |

Output:

```
data/2026-09-19T18-00-00/
  meta.json                 site, hz, camera intrinsics, dev flag
  quadcopter/
    00000.jpg ...
    sidecar.jsonl           one JSON object per frame
```

Sidecar line (dev fields present only when enabled):

```json
{"asset":"quadcopter","frame":"quadcopter/00000.jpg","index":0,
 "t_sim":6169.1,"t_wall":8517.29,"width":960,"height":720,
 "camera":{"fx":...,"fy":...,"cx":...,"cy":...},
 "pose":{"lat":...,"lon":...,"alt_rel":...,"roll":...,"pitch":...,"yaw":...},
 "groundtruth":{"lat":...,"lon":...,"heading":...,"speed":...,"world":[x,y,z],
                "point_px_approx":[u,v]}}
```

**Never enable `ARCTICSIM_DEV` for the judged run.** It is cheating there; use it
only to build training data and to score the tracker offline.

---

## `tools/dashboard_cli.py` — live terminal view

Status table plus a rough north-up ASCII map of every asset in local ENU metres.
`S` is the target vessel (dev only).

```bash
python tools/dashboard_cli.py              # refresh every second
python tools/dashboard_cli.py --once
ARCTICSIM_DEV=1 python tools/dashboard_cli.py   # include the ship
```

| Flag | Default | Meaning |
|---|---|---|
| `--once` | off | print one frame and exit |
| `--interval S` | `1.0` | refresh period |

```
      asset        link mode        armed    alt    E(m)    N(m)   spd
  Q  quadcopter     up GUIDED          Y    41.5    -995       9   0.0
  1  tower-1        up SERVO_TEST      Y     0.0   -1084   -1262   0.0
```

---

## `tools/calibrate_tower.py` — tower pan/tilt calibration

Sweeps pan and tilt, verifies the tracker tracks the commanded PWM via
`SERVO_OUTPUT_RAW`, and writes `calib/tower_<name>.json`. The fit is derived from
the sim's own plugin maths (corrected for the 1100–1900 µs servo range), so this
is a verification as much as a calibration.

```bash
python tools/calibrate_tower.py --tower tower-1
python tools/calibrate_tower.py --tower tower-2 --grab-frames calib/frames
```

| Flag | Default | Meaning |
|---|---|---|
| `--tower NAME` | `tower-1` | which mast |
| `--out-dir DIR` | `calib` | where the JSON is written |
| `--grab-frames DIR` | off | also save a JPEG at each sweep step |

```
  pan  cmd=1500  ack=True  SERVO_OUTPUT_RAW=1500  OK
  ...
servo tracking: 0 failures
wrote calib/tower_tower-1.json
```

`base_yaw_deg` (the world yaw of pan 1500 µs) cannot be measured from MAVLink —
the tower is a static model, so joint angles are not published. Set it by hand
if you need absolute `aim_at()`.

---

## `./fly.sh` — manual keyboard controller

Standalone, does not use `arcticlib`. `./fly.sh` runs `keyboard_control.py`);
arrow keys fly, `W/S` climb, `A/D` yaw, `T` takeoff, `L` land, `Space` stop,
`Q` quit.

```bash
./fly.sh
./fly.sh --master=udpout:10.99.1.1:14550 --max-speed=2
./fly.sh --self-test                     # scripted takeoff/fly/land check
```

---

## `./test_mavproxy.sh` — MAVProxy smoke test

Reproduces the original setup check: runs
`mavproxy.py --master=udpout:10.99.1.1:14550`, feeds it `exit`, and reports PASS
if MAVProxy starts and opens the link.

```bash
./test_mavproxy.sh
MASTER=udpout:127.0.0.1:14550 RUN_SECONDS=30 ./test_mavproxy.sh
```

---

## Tests

```bash
python -m unittest discover -s tests
```

Covers `geo.py` (distance/bearing/destination, ENU round trips, polar
stereographic) and `vehicle.py`'s `pose_at` interpolation (including yaw wrap).
No radio needed.