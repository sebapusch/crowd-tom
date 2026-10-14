"""Paired, headless comparison for a mixed ToM-0/ToM-1 scene."""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment import ExperimentConfig, build_environment, load_experiment


@dataclass
class Result:
    # Agents still inside at max_time contribute max_time (restricted mean).
    mean_by_type: dict[int, float]
    escaped_by_type: dict[int, int]
    exits_by_type: dict[int, list[int]]


def run(config: ExperimentConfig, seed: int, dt: float, max_time: float,
        all_tom0: bool) -> Result:
    random.seed(seed)
    environment = build_environment(config)
    if all_tom0:
        # Keep the same random spawn, headings, and type labels for measurement.
        environment.tom_order = 0

    group = {agent.idx: agent.tom_order for agent in environment.agents}
    departures: dict[int, float] = {}
    exits_by_type = {0: [0] * len(environment.exits), 1: [0] * len(environment.exits)}
    while environment.agents and environment._t < max_time:
        before = {agent.idx: (agent, agent.position.copy()) for agent in environment.agents}
        environment.tick(dt)
        remaining = {agent.idx for agent in environment.agents}
        for ident in before.keys() - remaining:
            agent, previous = before[ident]
            departures[ident] = environment._t
            for index, exit_ in enumerate(environment.exits):
                if exit_.is_crossed(previous, agent.position):
                    exits_by_type[group[ident]][index] += 1
                    break

    means = {
        order: statistics.mean(departures.get(ident, max_time) for ident in group if group[ident] == order)
        for order in (0, 1)
    }
    escaped = {
        order: sum(ident in departures for ident in group if group[ident] == order)
        for order in (0, 1)
    }
    return Result(means, escaped, exits_by_type)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', nargs='?', default=ROOT / 'experiments/mixed-tom-advantage.yaml',
                        type=Path)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--first-seed', type=int, default=0)
    parser.add_argument('--dt', type=float, default=1 / 60,
                        help='simulation step in seconds (default: the UI step, 1/60)')
    parser.add_argument('--max-time', type=float, default=120)
    args = parser.parse_args()
    if args.seeds < 1 or args.first_seed < 0 or args.dt <= 0 or args.max_time <= 0:
        parser.error('--seeds, --dt, and --max-time must be positive; --first-seed must be nonnegative')
    config = load_experiment(args.scene)
    if config.tom_proportions is None or (config.tom_proportions.tom0, config.tom_proportions.tom1) != (0.5, 0.5):
        parser.error('scene must assign 50% ToM-0 and 50% ToM-1')

    gaps = []
    control_gaps = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        mixed = run(config, seed, args.dt, args.max_time, all_tom0=False)
        control = run(config, seed, args.dt, args.max_time, all_tom0=True)
        gap = mixed.mean_by_type[0] - mixed.mean_by_type[1]
        control_gap = control.mean_by_type[0] - control.mean_by_type[1]
        gaps.append(gap)
        control_gaps.append(control_gap)
        print(f'seed {seed}: mixed capped mean ToM-0={mixed.mean_by_type[0]:.2f}s, '
              f'ToM-1={mixed.mean_by_type[1]:.2f}s, advantage={gap:+.2f}s; '
              f'all-ToM-0 labeled-group gap={control_gap:+.2f}s; '
              f'escaped mixed={mixed.escaped_by_type}, control={control.escaped_by_type}; '
              f'mixed exits by type={mixed.exits_by_type}', flush=True)

    print(f'Across {args.seeds} seeds (unescaped agents count as {args.max_time:g}s): '
          f'mean mixed advantage '
          f'{statistics.mean(gaps):+.2f}s; positive in {sum(g > 0 for g in gaps)}/{args.seeds} runs; '
          f'mean all-ToM-0 labeled-group gap {statistics.mean(control_gaps):+.2f}s; '
          f'paired difference {statistics.mean(a - b for a, b in zip(gaps, control_gaps)):+.2f}s')


if __name__ == '__main__':
    main()
