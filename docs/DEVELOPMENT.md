# Development guide

Setup, conventions, how to extend the library, and what to do when the sim
misbehaves. Sim facts live in [`../RECON.md`](../RECON.md); API in
[`API.md`](API.md).

## Setup

Requires Python 3.11+ (developed on 3.14).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes `setuptools<81`: on Python 3.14, MAVProxy imports
`pkg_resources`, which setuptools 81+ removed.

Sanity check once the sim is up:

```bash
python tools/smoke_test.py --no-fly     # safe
python tools/smoke_test.py              # includes flight
```

## Tests

```bash
python -m unittest discover -s tests
```

Only `geo.py` and pose interpolation are unit-tested. Everything else is verified
against the live sim by `tools/smoke_test.py`, because the sim *is* the
specification — see the rules below.

## Rules of the codebase

1. **Never raise into a caller's control loop.** Commands return `False` on
   failure or timeout and log. This matters because the planner and tracker call
   these in tight loops.
2. **One reader per link.** A `Vehicle`'s background thread is the sole
   `recv_match` consumer. Read cached attributes; don't read the socket yourself.
   If you need new telemetry, add it to `_on_message` and expose an attribute.
3. **Type hints and short docstrings everywhere.** This package is an API other
   people import.
4. **Trust the sim over the docs.** If a prompt, slide or README disagrees with
   the running sim, the sim wins — record the finding in `RECON.md`.
5. **Discover, don't guess.** Ports, message names and camera endpoints are
   measured, not assumed.
6. **Use sim time for anything feeding a filter.** `Pose.t_sim` and
   `Frame.t_sim` come from the gzweb `~/world_stats` clock.
7. **Keep ground truth out of autonomous code.** `arcticlib.groundtruth` is
   dev-only and gated on `ARCTICSIM_DEV=1`.

## Changing endpoints or adding an asset

Endpoints are role-based (`RECON.md` §4). To add or move one:

1. Add an `AssetSpec` in `config.py` (or override the port with
   `ARCTICSIM_<ASSET>_PORT`).
2. If it needs a new `kind`, add a subclass in `vehicle.py` and register it in
   `connect_vehicle`.
3. If it needs a camera, add a `CameraSpec` with the **real** sensor
   resolution/FOV (query `/snapshot.jpg` size and read the model SDF — do not
   trust the slides).
4. Wire it into `Fleet._CANON` in `fleet.py` and mirror it in `mock.py` so the
   mock stays a drop-in.

## Extending the library

* New telemetry → add to `Vehicle._on_message`; store under the lock; expose a
  property or attribute.
* New flight command → add a method that calls `_send_command` (ACK-checked) or
  `send_global_target` (unacknowledged setpoint), and document which.
* New positional source → put the maths in `geo.py` and unit-test it. Do not
  scatter coordinate transforms around call sites.
* New data → extend `types.py` with a dataclass and note it in `API.md`.

## Simulation quick reference

Full detail in [`../RECON.md`](../RECON.md).

| Thing | Value |
|---|---|
| Sim host (local) | `127.0.0.1` (`ARCTICSIM_HOST`) |
| Endpoints | quad `:14550`, plane `:14560`, tower-1 `:14580`, tower-2 `:14590` |
| Sysids | quad 1, plane 2, tower-1 4, tower-2 5 |
| Cameras | `:8600` quad, `:8610` plane, `:8630`/`:8640` towers (`/snapshot.jpg`, `/stream`) |
| Control / tracks / gzweb | `:8090` / `:8010` / `:8080` |
| Site | Fort Ross, 6.5 km, convergence −49.80° |

Three quirks that bite everyone:

* **Endpoints are `udpin` listeners.** Use `udpout`, and transmit *first* — the
  vehicle stays mute until it sees a GCS. This is also why SITL only steps while
  someone is connected.
* **`MAV_CMD_DO_REPOSITION` is unsupported.** `goto` uses
  `SET_POSITION_TARGET_GLOBAL_INT`.
* **Tower servos clamp to 1100–1900 µs** and the quad has **no gimbal**
  (`MNT1_TYPE 0`).

## Troubleshooting

**No heartbeat / `14550` silent.** Check `curl http://127.0.0.1:8090/api/status`.
If `state` is `idle` and the asset shows `"mavlink": false`, the sim is stopped —
that is not a network problem. Reset the sim (`POST /api/reset`, or the UI) and
wait for `/api/assets` to show `mavlink: true`. See `SimClient`.

**Vehicle connects but will not arm / take off.** Straight after a Reset the EKF
needs ~1–2 min. `takeoff()` already retries; give it time. If it still will not
climb, check `q.statustexts`.

**An asset commands fine but does not move.** That is a sim/autopilot fault, not
ours — most likely the physics backend. Reset from the UI and tell DD staff. We
observed the quadcopter hold position in GUIDED while STABILIZE + RC override
moved it, with `arm`/`NAV_TAKEOFF` accepted. See `RECON.md` §Sim health caveat.

**`pose/info` JSON parse errors.** gzweb fragments large WebSocket messages;
`simctl._WSClient` reassembles continuation frames. If you write your own client,
handle FIN=0.

**Camera returns nothing.** Confirm `camera: true` in `/api/assets`. The rover has
no feed when it is not rostered, and encoding is skipped while no client is
attached (the first snapshot after connecting is the current frame).

## Known limitations

* The quadcopter's GUIDED **horizontal** control was failing on the live sim at
  the time of writing; vertical/takeoff works (see `RECON.md`).
* `Plane.loiter(radius=...)` sets `WP_LOITER_RAD`, but the sim circles at the
  autopilot's own radius; treat the radius as advisory.
* `groundtruth.project_point` is approximate (body-fixed, level-attitude camera);
  labels only.
* `Fleet.status()["cameras"]` reports whether a frame is buffered, not whether
  the sensor is healthy.

## Commit conventions

Commits are milestone-sized and runnable on their own. Before committing:

```bash
python -m py_compile arcticlib/*.py tools/*.py
python -m unittest discover -s tests
```

Keep `RECON.md` and `docs/` in step with code changes — those are the parts the
rest of the team reads.