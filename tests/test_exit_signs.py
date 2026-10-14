import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent import Agent
from environment import Environment, ExitSign
from exit import Exit
from experiment import ExperimentConfig, load_experiment, save_experiment
from layout import SideExit
from obstacle import Wall


def make_environment(tom_order=0):
    agents = [
        Agent(
            idx=i, mass=1.0, radius=0.3, social_repulsion=(2000.0, 0.08),
            position=np.array([0.0, float(10 * i)]), velocity=np.zeros(2),
            desired_speed=2.0, desired_direction=np.array([1.0, 0.0]),
            tau=1.0,
        )
        for i in range(2)
    ]
    env = Environment(20, 20, agents, view_range=15.0, tom_order=tom_order)
    env.add_exit(Exit((10.0, -1.0), (10.0, 1.0)))
    env.add_obstacle(Wall((5.0, -2.0), (5.0, 2.0)))
    return env


class ExitSignTests(unittest.TestCase):
    def test_visible_sign_informs_tom_agent_without_exit_sight(self):
        for order in (0, 1):
            with self.subTest(tom_order=order):
                env = make_environment(order)
                env._desired_directions(env._pos, 1 / 60)
                self.assertFalse(env._has_belief[0, 0])

                env.add_sign(ExitSign((2.0, 0.0), 0))
                heading = env._desired_directions(env._pos, 1 / 60)
                self.assertFalse(env._seen_exits[0, 0])
                self.assertTrue(env._has_belief[0, 0])
                self.assertEqual(env._chosen_exit[0], 0)
                np.testing.assert_allclose(heading[0], [1.0, 0.0])

                env.signs.clear()
                env._desired_directions(env._pos, 1 / 60)
                self.assertEqual(env._chosen_exit[0], 0)

    def test_blocked_sign_does_not_inform_agent(self):
        env = make_environment()
        env.add_sign(ExitSign((7.0, 0.0), 0))
        env._desired_directions(env._pos, 1 / 60)
        self.assertFalse(env._has_belief[0, 0])
        self.assertEqual(env._chosen_exit[0], -1)

    def test_reactive_agent_uses_sign_only_while_visible(self):
        env = make_environment(-1)
        env.add_sign(ExitSign((2.0, 0.0), 0))
        env._desired_directions(env._pos, 1 / 60)
        self.assertEqual(env._chosen_exit[0], 0)
        self.assertFalse(env._has_belief[0, 0])

        env.signs.clear()
        env._desired_directions(env._pos, 1 / 60)
        self.assertEqual(env._chosen_exit[0], -1)

    def test_signs_round_trip_and_validate_exit_reference(self):
        config = ExperimentConfig(
            exits=[SideExit(side='north', center=5.0, width=3.0)],
            signs=[ExitSign((3.0, 4.0), 0)],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'signs.yaml'
            save_experiment(config, path)
            loaded = load_experiment(path)
        self.assertEqual(loaded.signs, config.signs)
        with self.assertRaisesRegex(ValueError, 'missing exit'):
            make_environment().add_sign(ExitSign((2.0, 0.0), 1))


if __name__ == '__main__':
    unittest.main()
