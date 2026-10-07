import tempfile
import unittest
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
        self.assertIn('remaining_tom0,remaining_tom1', history.to_csv())

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


if __name__ == '__main__':
    unittest.main()
