# Fixed demo config — wing + towers + quad, reproducible

The `.env` is **not enough on its own**. Three pieces make the demo repeatable:

1. **The sim stack** (`arctic-sim`: docker compose + Gazebo + SITL). The `.env`
   sets the asset roster, the fixed spawn positions, the moored ship and the
   camera mode.
2. **The mission code** (this repo). Includes the CV weights
   (`models/patch_verifier.pt`) and tower calibration, both committed.
3. **Python deps** (`requirements.txt`).

## 1. Sim side (`arctic-sim`)

Copy `demo.sim.env` into the sim repo root as `.env`:

```bash
cp demo.sim.env /path/to/arctic-sim/.env
# put your own Mapbox token in it if you want the terrain rebuilt from imagery
cd /path/to/arctic-sim
docker compose up -d --build          # first run builds terrain + ArduPilot
```

If the world is already built on that machine, you can instead hit the control
panel/API **Rebuild** (it regenerates the world from `.env` and recreates the
fleet) — that is what applies the fixed positions.

The values that matter:

| key | value | why |
|---|---|---|
| `SITE_NAME` | `fort_ross` | the world/terrain to load |
| `ASSET_1` | `copter,quadcopter,71.995809,-94.838879` | quad on flat dry ground |
| `ASSET_3` | `plane,fixed-wing,71.995785,-94.839483,>71.993000,-94.870000` | plane on flat ground, facing the ship |
| `ASSET_2` / `ASSET_5` | `tower-1` / `tower-2` at the two ends | sweep coverage |
| `SHIP_START_AT` | `71.993000,-94.870000` | ship moored 828 m from tower-1 |
| `SHIP_MOVING` | `0` | moored = deterministic |
| `CAMERAS` | `live` | **essential** — `ondemand` parks the sensors, the plane spawns tilted and will not fly |

## 2. Mission side (this repo)

```bash
git clone https://github.com/malekhammoud/Drone-Sim.git
cd Drone-Sim
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

If the sim is **not** on the same machine, point the client at it:

```bash
export ARCTICSIM_HOST=<sim-host>     # default is 127.0.0.1
```

## 3. Run

```bash
python main.py --search-timeout 180 --quad-duration 90
```

`--search-timeout` bounds the glider search and `--quad-duration` bounds the
quad follow, so the demo finishes in a few minutes instead of running the
defaults. Outputs land in `mission_output/<stamp>/` (`wing/`, `quad/`, and the
merged top-down MP4).

## Fixed positions

| asset | lat, lon | note |
|---|---|---|
| `fixed-wing` | 71.995785, -94.839483 | terrain's flat dry spawn, 75.8 m, faces the ship |
| `quadcopter` | 71.995809, -94.838879 | flat dry ground, 75.7 m |
| `ship` | 71.993000, -94.870000 | moored; 828 m from tower-1, 1104 m from the plane |
| `tower-1` | 71.996396, -94.891696 | west end |
| `tower-2` | 71.986899, -94.778238 | east end |
