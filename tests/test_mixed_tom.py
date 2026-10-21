import tempfile
import unittest
import csv
import io
from pathlib import Path

import numpy as np

from agent import Agent
from environment import Environment
from exit import Exit
from experiment import (
    ExperimentConfig, ToMProportions, build_environment, load_experiment,
    save_experiment,
)
from plots import History
from tom0 import ToM0Params
from tom1 import ToM1Params


class MixedToMTests(unittest.TestCase):
    def test_proportions_validate_and_round_trip(self):
        for tom0, tom1 in [(0.6, 0.5), (-0.1, 1.1), (0.2, 0.2)]:
            with self.assertRaises(ValueError):
                ToMProportions(tom0, tom1)
        config = ExperimentConfig(tom_proportions=ToMProportions(0.6, 0.4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mixed.yaml'
            save_experiment(config, path)
            loaded = load_experiment(path)
        self.assertEqual(loaded.tom_proportions, config.tom_proportions)

    def test_exact_assignment_and_initial_history(self):
        config = ExperimentConfig(
            width=30, height=30, agent_count=11,
            tom_proportions=ToMProportions(0.6, 0.4),
        )
        env = build_environment(config)
        stats = env.tom0_metrics()
        self.assertIsNone(env.tom_order)
        self.assertEqual(stats['remaining_tom1'], 4)
        self.assertEqual(stats['remaining_tom0'], 7)
        self.assertEqual(sum(agent.tom_order == 1 for agent in env.agents), 4)
        history = History()
        history.record(stats)
        self.assertEqual(history.remaining_tom0, [7.0])
        self.assertEqual(history.remaining_tom1, [4.0])
        rows = list(csv.DictReader(io.StringIO(history.to_csv())))
        self.assertEqual({row['agent_type'] for row in rows}, {'reactive', 'tom0', 'tom1'})
        self.assertEqual(int(next(row['remaining'] for row in rows if row['agent_type'] == 'tom0')), 7)
        self.assertEqual(int(next(row['remaining'] for row in rows if row['agent_type'] == 'tom1')), 4)

    def test_mixed_dispatch_only_models_tom1_observer(self):
        def make_env(order):
            agents = [
                Agent(
                    idx=i, mass=1.0, radius=0.3,
                    social_repulsion=(2000.0, 0.08),
                    position=np.array([float(2 * i), 0.0]),
                    velocity=np.zeros(2), desired_speed=2.0,
                    desired_direction=np.array([1.0, 0.0]), tau=1.0,
                    tom_order=(1 if i == 0 else 0),
                )
                for i in range(2)
            ]
            env = Environment(
                30, 30, agents, view_range=10.0,
                tom0=ToM0Params(w_occupancy=0.0),
                tom1=ToM1Params(demand_weight=-10.0), tom_order=order,
            )
            env.add_exit(Exit((5.0, -1.0), (5.0, 1.0)))
            env.add_exit(Exit((-10.0, -1.0), (-10.0, 1.0)))
            env._desired_directions(env._pos, 1 / 60)
            return env

        baseline = make_env(0)
        mixed = make_env(None)
        self.assertEqual(int(baseline._chosen_exit[0]), 0)
        self.assertEqual(int(mixed._chosen_exit[0]), 1)
        self.assertEqual(int(mixed._chosen_exit[1]), int(baseline._chosen_exit[1]))
        self.assertEqual(mixed.tom0_metrics()['tom1_changed'], 1)
        self.assertNotIn(1, mixed.tom1.memories)

    def test_departures_and_csv_metrics_keep_agent_types(self):
        agents = [
            Agent(
                idx=i, mass=1.0, radius=0.3,
                social_repulsion=(2000.0, 0.08),
                position=np.array([4.8 if i == 0 else 0.0, 0.0]),
                velocity=np.zeros(2), desired_speed=2.0,
                desired_direction=np.array([1.0, 0.0]), tau=1.0,
                tom_order=i,
            )
            for i in range(2)
        ]
        env = Environment(20, 20, agents, tom_order=None)
        env.add_exit(Exit((5.0, -1.0), (5.0, 1.0)))
        history = History()
        history.record(env.tom0_metrics())
        env.tick(0.5)
        stats = env.tom0_metrics()
        history.record(stats)

        self.assertEqual(stats['by_type']['tom0']['escaped'], 1)
        self.assertEqual(stats['by_type']['tom1']['escaped'], 0)
        self.assertEqual(stats['by_type']['tom0']['escaped_by_exit'], [1])
        self.assertEqual(stats['by_type']['tom0']['departed_exits_step'], [1])
        self.assertEqual(stats['by_type']['tom1']['remaining'], 1)
        for key in ('remaining', 'escaped', 'seeing_exit', 'seeing_sign',
                    'with_belief', 'memory_guided', 'no_belief',
                    'tom1_changed', 'blind_committed'):
            expected = stats['n'] if key == 'remaining' else stats[key]
            self.assertEqual(sum(values[key] for values in stats['by_type'].values()), expected)
        rows = list(csv.DictReader(io.StringIO(history.to_csv())))
        self.assertEqual(len(rows), 6)
        tom0_last = next(row for row in rows if row['agent_type'] == 'tom0' and float(row['t']) == 0.5)
        self.assertEqual(int(tom0_last['escaped']), 1)
        self.assertEqual(int(tom0_last['remaining']), 0)
        self.assertEqual(int(tom0_last['escaped_exit_0']), 1)
        self.assertEqual(int(tom0_last['departed_exit_0']), 1)

    def test_metrics_keep_assigned_type_when_behavior_is_overridden(self):
        config = ExperimentConfig(
            width=20, height=20, agent_count=2,
            tom_proportions=ToMProportions(0.5, 0.5),
        )
        env = build_environment(config)
        env.tom_order = 0
        env.tick(0.1)
        stats = env.tom0_metrics()['by_type']
        self.assertAlmostEqual(stats['tom0']['agent_seconds'], 0.1)
        self.assertAlmostEqual(stats['tom1']['agent_seconds'], 0.1)


if __name__ == '__main__':
    unittest.main()
