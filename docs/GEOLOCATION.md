# Pixel → GPS Geolocation

`arcticlib/geolocate.py` turns an image pixel into a lat/lon by casting a ray
from the camera through the pixel, rotating it into NED, intersecting flat
ground, and converting the horizontal offset to lat/lon with a WGS84 geodesic.

## Integration with the 3-stage detector (merged)

The GPS tools are wired into the merged detector pipeline
(`tools/detect_verified.py`: **Step 1** color anomaly → **Step 2** CNN verifier
with the trained `models/patch_verifier.pt` → **Step 3** temporal persistence):

* `tools/detect_verified_gps.py` imports `VerifiedDetector` from
  `tools/detect_verified` (no duplicated copy) and drives it with temporal
  context (`frame_idx`, `t_sim`, `pose`, `cam_intrinsics`), then geolocates every
  surviving hit with `arcticlib.geolocate` and overlays lat/lon labels.
* `tools/detect_verified.pixel_to_latlon` (used by the detector's geo-mode
  temporal association) now delegates to the trig geolocator instead of the old
  flat-earth approximation.
* `tools/patrol_and_record_gps.py` is rebuilt on the remote patrol rework (safe
  3-tier strait waypoints + 3-stage detector) with the GPS recording,
  `detections.jsonl`, `tracks.jsonl` and HUD annotations re-applied.
* `tools/quad_follow_ship.py` and `tools/two_step_mission.py` feed the temporal
  filter the same per-frame context.

On the ground-truth dataset (`tools/gt_runs/2026-09-19T21-26-43`) the merged
pipeline produced a single fused ship track of **244 hits over 190 frames**
(mean score 0.76), and the CNN cut Stage-1 false positives by ~97%.

## Track API — map the boat

`arcticlib/tracks.py` implements the competition create/update/list contract:

* `TrackClient.post(name, lat, lon, heading=None, speed=None)` — creates on the
  first POST with a name, updates thereafter (bumps `fixes`).
* `TrackClient.post_fix(name, lat, lon, t)` — same, but derives **heading**
  (deg true) and **speed** (m/s) from the previous fix for that name.
* `TrackClient.list()`.

Endpoint: `config.url(config.tracks_port)` → `http://127.0.0.1:8010`. Note the
local `arctic-sim` compose has **no 8010 service**; that endpoint is the
competition track API (the curl examples point at `<SIM-IP>:8010`).

CLI (the curl examples, as a tool):

```bash
python tools/tracks_cli.py post --name "Sierra One" --lat 71.9965 --lon -94.8448
python tools/tracks_cli.py post --name "Sierra One" --lat 71.9975 --lon -94.8450 --heading 315 --speed 6.5
python tools/tracks_cli.py list
```

Publishing from a mission: `--publish-tracks` (`main.py`, `quad_follow_ship.py`,
`patrol_and_record_gps.py`). The wing posts its best fused track with
course/speed; the quad and live detector post each fix with derived heading/speed.

## Where it is wired in

| File | Role |
|---|---|
| `arcticlib/geolocate.py` | The maths: `pixel_to_gps`, `camera_ray_ned`, `intersect_ground`, `GeoConfig`, `GeoEstimate`. |
| `tools/patrol_and_record_gps.py` | Patrol recorder that logs a `gps` block + `gps_track.csv` per frame and writes geolocated detections to `detections.jsonl`, annotated on the MP4. |
| `tools/detect_verified_gps.py` | Two-stage detector (color + CNN) that geolocates every verified detection in `--image`, `--dataset` and `--live` modes and can post fixes to the track API. |
| `tests/test_geolocate.py` | Hand-checkable geometry tests. |

## Conventions

* Body = **(forward, right, down)**, NED = **(north, east, down)**; both have +z down.
* Angles are **radians** (repo convention). `yaw` is clockwise from **true north**,
  `pitch` is positive nose-up, `roll` is positive right-wing-down — exactly
  MAVLink `ATTITUDE`, so `Pose.yaw/pitch/roll` go straight in.
* Camera frame is OpenCV: x right, y down, z forward. Camera→body is the fixed
  remap `R_cb = [[0,0,1],[1,0,0],[0,1,0]]`.
* Body→NED is `Rz(yaw)·Ry(pitch)·Rx(roll)`.
* The camera mount is a separate pitch, `R_gimbal_to_body = Ry(gimbal_pitch)`,
  with `gimbal_pitch` in the "0 = horizon, −90 = straight down" convention.

## Camera mount angles (from the sim's sensor SDF)

The sim exposes no gimbal telemetry, but the sensor SDF is available from gzweb
`~/scene`. The camera is body-fixed with a pure pitch offset:

| asset | sensor | mount pitch |
|---|---|---|
| `fixed-wing` | `skywalker_x8/fpv_camera` | **−8.021°** |
| `quadcopter` | `gimbal_small_2d/webcam` | **−20.002°** |
| `tower-1` / `tower-2` | `eo_camera` | derived from pan/tilt servos — not handled here |

The fixed-wing value comes from the SDF quaternion
`(x=0, y=0.069942847, z=0, w=0.997551)` on `base_link`:
`2·atan2(0.069942847, 0.997551) = 8.021°`, which tips the camera's +X optical
axis downward. Override with `--camera-pitch-deg` if the model changes.

## Assumptions (and how much they bias the result)

1. **Height above the target is the dominant term.** We use
   `agl = alt_amsl − ground_elevation_m` (default `ground_elevation_m = 0`, i.e.
   the target sits at sea level, which matches the vessel at world `z = 0`).
   `alt_ref="rel"` (takeoff-relative) is available but is only correct if the
   launch point is at the target's ground level. **A wrong ground elevation or
   MSL/AGL mix-up biases range almost linearly** and is the biggest single risk.
2. **The camera mount pitch is critical at shallow angles.** Ground-range error
   grows as `h / sin²(δ)`. At the fixed-wing's typical 8° depression a 1° mount
   error moves the fix by tens of metres; at 45° it is negligible. The SDF value
   is authoritative but should be re-read after any model change.
3. **Flat ground.** No DEM. `intersect_ground` is the single seam where a
   terrain ray-march would drop in. Over the sea this is exact; over land it
   biases long if the terrain rises.
4. **Attitude sign/unit conventions.** `yaw` is assumed **true** heading (the
   repo converts true→grid via `convergence_deg` for `project_point`, which
   implies `ATTITUDE.yaw` is true). If it were magnetic, the fix would be rotated
   by the local declination. Pitch/roll signs follow MAVLink.
5. **No lens distortion.** Intrinsics are FOV-derived pinhole (`CameraSpec.intrinsics()`);
   there are no distortion coefficients in the sim. Near frame edges this biases
   the ray slightly.
6. **Timestamps.** `fleet.pose_at(asset, frame.t_sim, clock="sim")` interpolates
   telemetry to the frame's sim time. Residual latency at 20 m/s is ~1 m per
   50 ms; a real frame-exposure vs telemetry offset would show up as a lateral bias.
7. **The mount is static** (no gimbal command surface exists). `Pose.gimbal_*`
   is always `None` and is not used.

## Error radius

`GeoEstimate.error_radius_m` is an approximate 1σ radial error from the analytic
sensitivity `dr/dδ = −h / sin²δ`, plus altitude (`cot δ`), lateral yaw
(`r·σ_yaw`) and input position error, combined in quadrature. Defaults are
0.5° attitude, 3 m position, 2 m altitude; override on the CLI
(`--attitude-sigma`, `--position-sigma`, `--altitude-sigma`).

Rays shallower than `--min-depression` (default 10°) are flagged `grazing`
(and dropped with `--reject-grazing`); they are where the flat-ground
assumption and the error model both degrade fastest.

## Ground-truth validation (2026-09-19 follow-ship run)

Recorded with `patrol_and_record_gps.py --follow-ship` and `ARCTICSIM_DEV=1`
(300 frames, the true vessel lat/lon in every sidecar entry). Scoring via
`detect_verified_gps.py --dataset ... ` (it now reports "Geolocation vs GROUND
TRUTH"):

| mount pitch | matched frames | median error | mean | RMSE |
|---|---|---|---|---|
| −8.021° (SDF) | 176/300 | **79 m** | 78 m | 90 m |
| −8.50° (calibrated) | 199 | **38 m** | 34 m | 36 m |

The calibrated value was chosen on even frames and confirmed on odd frames
(train −8.50° → 37.5 m; held-out → 38.2 m), so it is not overfit. Error vs.
depression angle (single frame, calibrated mount):

| depression | median error |
|---|---|
| 20–40° | **13 m** |
| 12–20° | 30 m |
| 8–12° | 47 m |
| 4–8° | 101 m |
| 0–4° | 190 m |

That is the expected `h / sin²δ` behaviour. The predicted 1σ radius tracked the
actual error well (median 63 m predicted vs 79 m actual at the SDF mount), so the
uncertainty estimate is honest.

**Caveat:** the fused-track error radius (`FusedTrack.error_radius_m`) assumes
independent per-frame errors, so it collapses toward ~1 m over hundreds of frames
even when a systematic bias remains. Use the per-frame ground-truth error above,
not the fused radius, to judge absolute accuracy.

Set the calibrated mount with `--camera-pitch-deg -8.5` (or
`CALIBRATED_CAMERA_PITCH_DEG` in `geolocate.py`); the SDF default stays the
principled one.

## Step 2: quadcopter follow (`tools/quad_follow_ship.py`)

Step 1 (glider) produces a lat/lon; step 2 flies the quadcopter there, keeps the
ship in view and refines it:

```
python tools/quad_follow_ship.py --tracks tools/gt_runs/<stamp>/eval/tracks.jsonl
ARCTICSIM_DEV=1 python tools/quad_follow_ship.py --from-ship --duration 240   # DEV test
```

It holds a **standoff** so the ship sits at `--view-depression` (default 45 deg)
instead of directly below (which a forward-down camera cannot see), and re-aims
the hold point as the ship moves. It scores each fix against the true vessel in
DEV and can post to the track API.

### The gimbal cannot be pointed down

The quadcopter camera is a **fixed 20.0 deg-down mount**:

* SDF joint `gimbal_small_2d::tilt_joint` is **type 8 (fixed)**, with no
  controller plugin on the model.
* `MNT1_TYPE=0`; setting it to 1 (servo) or 2 (MAVLink) and sending
  `DO_MOUNT_CONTROL` / `DO_GIMBAL_MANAGER_PITCHYAW` does **not** move the camera
  (servo outputs stay 0, frames unchanged), and a bad mount type fails pre-arm
  ("Mount: check TYPE").
* gzweb's `~/link` message carries only link properties, not pose, so it cannot
  reorient the sensor either.

So `--nadir` (`-90 deg`) is only correct if the sim model is changed. With the
real mount, the quad still gives good geometry: its 98.9 deg vertical FOV reaches
~69 deg depression at the bottom of the frame, far better than the glider's ~8 deg.

### One-command handoff: `tools/two_step_mission.py`

Chains both phases over a single `Fleet` connection (the glider keeps flying while
the quad works):

```
ARCTICSIM_DEV=1 python tools/two_step_mission.py --from-ship \
    --search-duration 150 --duration 180 --alt 25 --speed 18
```

* Phase 1 flies the glider (DEV: steers at the live vessel), detects + geolocates
  until `--min-hits` fixes, then fuses them with `refine_tracks` and hands off the
  best target.
* Phase 2 calls `quad_follow_ship.run_quad_follow` with that target.
* `--tracks <step-1 tracks.jsonl>` or `--lat/--lon` skips the search and hands off
  directly.

Measured on 2026-09-20: the glider found it and the quad refined the fix to a
**12.4 m median (2.2 m best)** at the 45 deg standoff.
