# crowd-tom

2D crowd evacuation on a Helbing social-force model, with **limited vision** and optional **ToM-0** (beliefs about exits). Python 3.12, run with [uv](https://docs.astral.sh/uv/).

This is meant for teammates to swap layouts, toggle ToM, and compare runs. Physics (pushing, walls) stay Helbing; ToM only changes **which exit** an agent heads toward.

## Setup

```bash
git clone git@github.com:sebapusch/crowd-tom.git
cd crowd-tom
git checkout tom0-environments   # current experiment branch
uv sync
```

Needs Python **≥ 3.12** (see `.python-version`). Use `uv run` so you get that version, not system 3.11.

## Run

```bash
uv run python main.py
uv run python main.py experiments/pillar.yaml
```

A pygame window opens: the room on the left, live plots on the right. Close the window to quit.

### Bundled scenes (`experiments/`)

| File | Layout |
|---|---|
| `two-north-one-south.yaml` | Default: two north doors, one south (no interior obstacles) |
| `north-south.yaml` | One north, one south |
| `four-doors.yaml` | North, south, east, west |
| `pillar.yaml` | Two north + one south, plus a bar and a circular pillar |

`tom_order` in YAML: `0` = ToM-0, `null` / `none` = reactive (no memory).

## Controls

| Key | Action |
|---|---|
| Space | Pause |
| R | Reset this experiment (new spawn) |
| + / - | Simulation speed |
| T | Toggle **ToM-0** ↔ **reactive** (no memory) |
| [ / ] | Previous / next YAML in `experiments/` |
| E | Edit mode (pauses). Place geometry, then the crowd respawns |
| W / C | In edit mode: draw a **wall** or **circle** (click-drag) |
| Click a wall edge | In edit mode: add a width-3 door on that side |
| Backspace | Undo last circle, interior wall, or door |
| F | Save current layout to `experiments/last.yaml` (gitignored) |
| G | Save plot time series to `experiments/last-metrics.csv` (gitignored) |
| D | Debug: range circles, green = seeing a door, orange = remembered door |

When the last agent leaves, the HUD shows **all escaped in Xs** and writes `experiments/last-metrics.csv`.

## What the agents do

**Perception:** 360° view, range **40**, blocked by walls/circles. Unseen doors are not “visible”.

**Reactive (`tom_order: none`):** walk to the nearest *currently* visible door. If none, follow someone who sees a door, else wander.

**ToM-0 (`tom_order: 0`):** on sight, store last-seen time and door position. Uncertainty grows as \(\sigma_0 + \alpha\sqrt{\Delta t}\). Each step pick the believed door with highest utility (closer, less crowded cone, more certain). They can keep walking to a door they **no longer see**. If they have never seen any door, they follow a committed agent.

ToM-1 is not implemented. **T** only switches reactive vs ToM-0.

In an empty rectangle, walking toward a remembered door often **keeps it inside range 40**, so “unseen choice” can stay 0. Pillars, extra walls, or leaving range make ToM-0 visible (orange debug lines, “ToM-0 unseen choice” in the HUD).

## Write your own experiment

Copy a file under `experiments/` or press **F** in the editor. Perimeter walls are generated from doors; you only list gaps and interior obstacles.

```yaml
name: my-hall
width: 100
height: 100
tom_order: 0          # 0 = ToM-0, omit or null = reactive
agents:
  count: 400
  radius: 0.3
  desired_speed: 2.0
perception:
  view_range: 40
  fov_deg: 360
exits:
  - {side: north, center: 21.5, width: 3, name: NW}
  - {side: south, center: 51.5, width: 3, name: south}
obstacles:
  walls:
    - {start: [30, 40], end: [70, 40]}
  circles:
    - {center: [50, 50], radius: 8}
```

`side` is `north` | `south` | `west` | `east`. `center` is the position **along that wall** (x on north/south, y on east/west).

## HUD and plots

- Simulation time (not wall-clock; **+**/**-** change how fast time advances)
- Remaining / escaped per door
- Seeing a door vs holding a belief vs never seen
- **ToM-0 unseen choice:** heading to a remembered door that is out of sight (should stay 0 in reactive mode)
- Right panel: remaining vs escaped, and ToM-state series over time

## Project layout

| Path | Role |
|---|---|
| `main.py` | Window, keys, editor, plots |
| `experiment.py` | Load/save YAML, spawn agents |
| `layout.py` | Doors on walls → perimeter segments |
| `environment.py` | Time step, forces, ToM vs reactive choice |
| `tom0.py` | Belief scoring (distance, occupancy, uncertainty) |
| `perception.py` | Range, FOV, line of sight |
| `agent.py` | Helbing agent parameters |
| `obstacle.py` / `exit.py` | Walls, circles, exit crossing |
| `plots.py` | Live metric panel + CSV export |
| `experiments/` | Scene files to share |

## Commit / push (for this branch)

Do **not** commit `experiments/last.yaml` or `experiments/last-metrics.csv` (gitignored run output).

```bash
git checkout tom0-environments
git add README.md .gitignore pyproject.toml uv.lock \
  main.py environment.py experiment.py layout.py plots.py tom0.py perception.py \
  experiments/*.yaml
git status   # confirm no last.yaml / last-metrics.csv
git commit -m "Add ToM-0, YAML experiments, editor, and live metrics."
git push -u origin tom0-environments
```

Teammates: `git fetch && git checkout tom0-environments && uv sync && uv run python main.py`.
