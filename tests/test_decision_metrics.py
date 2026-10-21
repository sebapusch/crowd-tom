import csv
import io
import unittest

import numpy as np

from agent import Agent
from environment import Environment
from exit import Exit
from plots import History


class DecisionMetricsTests(unittest.TestCase):
    def test_selected_exit_counts_and_valid_choice_switch_rate(self):
        agents = [
            Agent(
                idx=index, mass=1.0, radius=0.3,
                social_repulsion=(2000.0, 0.08),
                position=np.array(position, dtype=float), velocity=np.zeros(2),
                desired_speed=0.0, desired_direction=np.array([0.0, 1.0]),
                tau=1.0,
            )
            for index, position in enumerate(((2.0, 5.0), (5.0, 8.0)))
        ]
        env = Environment(20, 20, agents, view_range=20.0, tom_order=-1)
        env.add_exit(Exit((0.0, 4.0), (0.0, 6.0)))
        env.add_exit(Exit((10.0, 4.0), (10.0, 6.0)))
        history = History()
        history.record(env.tom0_metrics())

        env.tick(0.1)
        first = env.tom0_metrics()
        self.assertEqual(first['selected_exits'], [2, 0])
        self.assertEqual(first['switches_step'], 0)  # Initial selection is not a switch.
        history.record(first)

        env._pos[0] = [8.0, 5.0]
        env.tick(0.1)
        second = env.tom0_metrics()
        self.assertEqual(second['selected_exits'], [1, 1])
        self.assertEqual(second['switches_step'], 1)
        self.assertEqual(second['switches_total'], 1)
        self.assertAlmostEqual(second['agent_seconds'], 0.4)
        self.assertAlmostEqual(second['switch_rate'], 2.5)
        history.record(second)

        env.view_range = 1.0
        env.tick(0.1)
        self.assertEqual(env.tom0_metrics()['switches_step'], 0)
        env.view_range = 20.0
        env.tick(0.1)
        self.assertEqual(env.tom0_metrics()['switches_step'], 0)  # Fallback to exit is not a switch.
        self.assertEqual(env.tom0_metrics()['switches_total'], 1)

        rows = list(csv.DictReader(io.StringIO(history.to_csv())))
        latest = next(row for row in rows if row['agent_type'] == 'reactive' and float(row['t']) == 0.2)
        self.assertEqual(int(latest['selected_exit_0']), 1)
        self.assertEqual(int(latest['selected_exit_1']), 1)
        self.assertEqual(int(latest['escaped_exit_0']), 0)
        self.assertEqual(int(latest['switches_step']), 1)
        self.assertAlmostEqual(float(latest['switch_rate']), 2.5)


if __name__ == '__main__':
    unittest.main()
