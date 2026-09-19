# drone-sim

MAVProxy setup plus a keyboard flight controller for an ArduCopter in GUIDED mode.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes `setuptools<81`, which is required on Python 3.14
because MAVProxy imports `pkg_resources` (removed from setuptools 81+).

## MAVProxy smoke test

```bash
./test_mavproxy.sh
```

Runs `mavproxy.py --master=udpout:10.99.1.1:14550`, feeds it `exit`, and reports
PASS if MAVProxy starts and opens the master link.

## Keyboard flight control

```bash
./fly.sh
```

or directly:

```bash
source .venv/bin/activate
python keyboard_control.py --master=udpout:10.99.1.1:14550
```

### Controls

| Key            | Action                        |
|----------------|-------------------------------|
| Up / Down      | fly forward / backward        |
| Left / Right   | strafe left / right           |
| W / S          | climb / descend               |
| A / D          | yaw left / right              |
| Space          | stop now (hover)              |
| T              | take off (arms if needed, climbs to takeoff altitude) |
| L              | land                          |
| Q / Ctrl-C     | quit (stops and hovers)       |

The HUD shows the current mode, armed state, altitude, speed and the body-frame
command being streamed.

### How it works

The aircraft is in **GUIDED** mode. The program streams
`SET_POSITION_TARGET_LOCAL_NED` messages at ~30 Hz with a velocity vector in
`MAV_FRAME_BODY_OFFSET_NED` plus a yaw rate. Because the frame is relative to
the aircraft's heading, "forward" always means "where the nose points".

Commands are slew-limited so the aircraft does not jerk, and no keys held means
zero velocity (brake to hover). Since a terminal has no key-up events, a key
counts as held until it has not been seen for ~0.65 s (key auto-repeat keeps it
alive). Press Space to stop immediately.

### Options

```
--master MASTER        MAVLink connection string (default udpout:10.99.1.1:14550)
--max-speed M          max horizontal speed, m/s (default 3)
--max-climb M          max vertical speed, m/s (default 1.5)
--max-yaw-rate DEG     max yaw rate, deg/s (default 60)
--takeoff-alt M        altitude for the T key, m (default 15)
--no-guided            do not switch flight mode; only warn if not GUIDED
--self-test            run a scripted takeoff/forward/land test and exit
```

### Verify it works

The built-in self-test takes off if the vehicle is on the ground, flies forward
for a few seconds, brakes, then lands:

```bash
./fly.sh --self-test
```

Expected output ends with `SELF-TEST PASS`.

## Sim server (ArcticSim)

The vehicle lives behind a competition sim server reachable through the
WireGuard tunnel at `10.99.1.1`. It exposes a control API on port **8090** and a
web panel on port **8080**:

```bash
curl -s http://10.99.1.1:8090/api/status          # {"state": "...", "detail": "..."}
curl -s http://10.99.1.1:8090/api/assets          # per-asset mavlink/camera flags
curl -s "http://10.99.1.1:8090/api/logs?tail=50"  # server/container log tail
```

If `state` is `idle` and the `quadcopter` asset shows `"mavlink": false`, the
simulation is not running: nothing will answer on `14550` even though the tunnel
is fine. Use the **Reset** button on http://10.99.1.1:8080 (or
`POST /api/reset`) and wait a minute or two for gzweb and the assets to come
back. `POST /api/rebuild` does a full rebuild and takes several minutes.

A healthy asset looks like `"mavlink": true, "camera": true`. Note that a reset
leaves the drone **disarmed on the ground in STABILIZE**; press **T** in the
controller to arm and take off.

## Notes

- `Failed to load module: No module named 'adsb'` in MAVProxy is harmless; it is
  an optional module.
- On Python 3.14, MAVProxy fails with `No module named 'pkg_resources'` unless
  `setuptools<81` is installed.
- GUIDED velocity control only works while the vehicle is armed and flying. If
  it is disarmed, press **T** (the controller switches to GUIDED and arms
  automatically), or arm it yourself.
- If `14550` stops answering, check `/api/status` before blaming the tunnel:
  `ping 10.99.1.1` working while MAVLink is silent usually means the sim is
  stopped, not that WireGuard is broken.
