# crowd-tom

2D crowd evacuation on a Helbing social-force model, with **limited vision**, ToM-0 exit beliefs, and ToM-1 predictions of nearby agents' exit choices. Python 3.12+, run with [uv](https://docs.astral.sh/uv/).

This is meant for teammates to swap layouts, toggle ToM, and compare runs. Physics (pushing, walls) stay Helbing; ToM only changes **which exit** an agent heads toward.

## Setup

```bash
git clone git@github.com:sebapusch/crowd-tom.git
cd crowd-tom
git checkout tom1-environment
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


| File                       | Layout                                                      |
| -------------------------- | ----------------------------------------------------------- |
| `two-north-one-south.yaml` | Default: two north doors, one south (no interior obstacles) |
| `north-south.yaml`         | One north, one south                                        |
| `four-doors.yaml`          | North, south, east, west                                    |
| `mixed-tom.yaml`           | Four doors with equal ToM-0 and ToM-1 proportions          |
| `pillar.yaml`              | Two north + one south, plus a bar and a circular pillar     |
| `station-concourse.yaml`   | Clustered arrivals, ticket-gate obstacles, three signed platform entrances |


`tom_order` in YAML: `1` = ToM-1, `0` = ToM-0, `null` / `none` / `reactive` = reactive. Omitting it defaults to ToM-0.

For a mixed population, add `tom_proportions: {tom0: 0.6, tom1: 0.4}` at the top level. Both values must be between 0 and 1 and sum to 1. The simulator rounds the ToM-1 count to the nearest agent, assigns types randomly at spawn, and keeps each agent's type for the run. This setting takes precedence over `tom_order`. See `experiments/mixed-tom.yaml`.

## Controls


| Key               | Action                                                                |
| ----------------- | --------------------------------------------------------------------- |
| Space             | Pause                                                                 |
| R                 | Reset this experiment (new spawn)                                     |
| + / -             | Simulation speed                                                      |
| T                 | Cycle reactive / ToM-0 / ToM-1; mixed scenes also cycle back to the mixture |
| [ / ]             | Previous / next YAML in `experiments/`                                |
| E                 | Edit mode (pauses). Place geometry, then the crowd respawns           |
| W / C             | In edit mode: draw a **wall** or **circle** (click-drag)              |
| Click a wall edge | In edit mode: add a width-3 door on that side                         |
| Backspace         | Undo last circle, interior wall, or door                              |
| F                 | Save current layout to `experiments/last.yaml` (gitignored)           |
| G                 | Save plot time series to `experiments/last-metrics-<mode>.csv` (gitignored) |
| D                 | Debug: each chosen exit gets a color; arrows show agents' chosen headings |


When the last agent leaves, the HUD shows **all escaped in Xs** and writes a mode-specific metrics CSV.

## What the agents do

**Perception:** 360° view, range **40**, blocked by walls/circles. Unseen doors are not “visible”.

**Exit signs:** a sign names an exit by its zero-based index in the YAML `exits` list. An agent that can see the sign learns that exit's location even when the exit itself is out of sight. Walls, range, and field of view also apply to signs. ToM-0 and ToM-1 agents remember the information; reactive agents use it only while the sign is visible. Signs appear as numbered teal diamonds.

**Reactive (**`tom_order: none`**):** walk to the nearest *currently* visible door. If none, follow someone who sees a door, else wander.

**ToM-0 (**`tom_order: 0`**):** on sight, store last-seen time and door position. Uncertainty grows as `sigma0 + sigma_alpha * sqrt(age)`. Each step pick the believed door with highest utility (closer, less crowded cone, more certain). They can keep walking to a door they **no longer see**. If they have never seen any door, they follow a visible informed agent.

**ToM-1 (**`tom_order: 1`**):** start with each agent's own ToM-0 score. For currently visible agents within `model_range`, estimate their ToM-0 choices using only the observer's exit memories, observed positions, and visible velocity as a heading proxy. Each observer keeps a bounded, expiring record of exits it inferred other agents saw. Predicted exit demand adds a negative, confidence-weighted score term. ToM-1 uses the same forces, movement, and follow/wander fallback as ToM-0. These are heuristic predictions, not direct reads of other agents' private memories.

In an empty rectangle, walking toward a remembered door often **keeps it inside range 40**, so “unseen choice” can stay 0. Pillars, extra walls, or leaving range make ToM-0's unseen choices visible in the HUD. In debug mode, each arrow points along an agent's chosen heading and matches the color of its selected exit; agents following others or wandering have no exit-choice arrow.

## Write your own experiment

Copy a file under `experiments/` or press **F** in the editor. Perimeter walls are generated from doors; you only list gaps and interior obstacles.

```yaml
name: my-hall
width: 100
height: 100
tom_order: 1          # 1 = ToM-1, 0 = ToM-0, null = reactive
tom1:                 # optional; these are the defaults
  model_range: 12.0
  memory_agents: 16
  memory_horizon: 30.0
  confidence_decay: 10.0
  knowledge_prior: 0.2
  uncertainty_prior: 0.5
  choice_temperature: 1.0
  demand_weight: -1.0
agents:
  count: 400
  radius: 0.3
  desired_speed: 2.0
  spawn_region:          # optional rectangle for initial agent centers
    min: [10, 10]
    max: [40, 35]
perception:
  view_range: 40
  fov_deg: 360
exits:
  - {side: north, center: 21.5, width: 3, name: NW}
  - {side: south, center: 51.5, width: 3, name: south}
signs:
  - {position: [50, 30], exit_index: 0}  # points to the first exit
obstacles:
  walls:
    - {start: [30, 40], end: [70, 40]}
  circles:
    - {center: [50, 50], radius: 8}
```

`side` is `north` | `south` | `west` | `east`. `center` is the position **along that wall** (x on north/south, y on east/west).

`agents.spawn_region` limits initial agent centers to the rectangle from `min` to `max`. It must fit inside the room with clearance for each agent's radius. Spawned agents avoid obstacles and each other; an area too small for the requested count raises an error. Omit it to spawn throughout the room as before. See `experiments/train-uneven-doors.yaml` for a cluster near the narrow train doors.

## HUD and plots

- Simulation time (not wall-clock; **+**/**-** change how fast time advances)
- Remaining / escaped per door
- Remaining ToM-0 and ToM-1 agents, both in the HUD and in the live graph. The graph also shows total remaining and escaped agents. Press **G** to export the time series, including each type's remaining count.
- Seeing a door vs holding a belief vs never seen
- **ToM-0 unseen choice:** heading to a remembered door that is out of sight (should stay 0 in reactive mode)
- **ToM-1 changed exit:** agents whose current ToM-1 choice differs from their ToM-0 choice
- Right panel: remaining agents by type and ToM-state series over time. In a mixed scene, the type counts follow the agents' assigned types even when **T** temporarily overrides their behavior.



## Project layout


| Path                      | Role                                              |
| ------------------------- | ------------------------------------------------- |
| `main.py`                 | Window, keys, editor, plots                       |
| `experiment.py`           | Load/save YAML, spawn agents                      |
| `layout.py`               | Doors on walls → perimeter segments               |
| `environment.py`          | Time step, forces, ToM vs reactive choice         |
| `tom0.py`                 | Belief scoring (distance, occupancy, uncertainty) |
| `tom1.py`                 | Limited observation memory and predicted demand   |
| `perception.py`           | Range, FOV, line of sight                         |
| `agent.py`                | Helbing agent parameters                          |
| `obstacle.py` / `exit.py` | Walls, circles, exit crossing                     |
| `plots.py`                | Live metric panel + CSV export                    |
| `experiments/`            | Scene files to share                              |




## Tests

```bash
uv run python -m unittest discover -s tests -v
```
