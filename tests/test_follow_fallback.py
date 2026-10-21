import unittest

import numpy as np

from agent import Agent
from environment import Environment
from exit import Exit
from obstacle import Wall


def make_agent(index, position):
    return Agent(
        idx=index, mass=1.0, radius=0.3, social_repulsion=(2000.0, 0.08),
        position=np.array(position, dtype=float), velocity=np.zeros(2),
        desired_speed=2.0, desired_direction=np.array([0.0, 1.0]), tau=1.0,
    )


class FollowFallbackTests(unittest.TestCase):
    def test_follows_nearest_visible_agent_even_when_uninformed(self):
        for order in (-1, 0, 1):
            with self.subTest(tom_order=order):
                agents = [
                    make_agent(0, (0.0, 0.0)),
                    make_agent(1, (1.0, 0.0)),
                    make_agent(2, (0.0, 4.0)),
                ]
                env = Environment(200, 200, agents, view_range=10.0,
                                  wander_turn=0.0, tom_order=order)
                env.add_exit(Exit((100.0, -1.0), (100.0, 1.0)))
                env._has_belief[2, 0] = True
                env._belief_mu[2, 0] = [100.0, 0.0]

                direction = env._desired_directions(env._pos, 1 / 60)
                self.assertEqual(env._chosen_exit[0], -1)
                np.testing.assert_allclose(direction[0], [1.0, 0.0])
                stats = env.tom0_metrics()
                self.assertGreaterEqual(stats['following'], 1)
                self.assertEqual(
                    stats['following'] + stats['wandering'] + sum(stats['selected_exits']),
                    stats['n'],
                )

    def test_wanders_when_no_other_agent_is_visible(self):
        agents = [make_agent(0, (0.0, 0.0)), make_agent(1, (1.0, 0.0))]
        env = Environment(200, 200, agents, view_range=10.0, wander_turn=0.0)
        env.add_exit(Exit((100.0, -1.0), (100.0, 1.0)))
        env.add_obstacle(Wall((0.5, -1.0), (0.5, 1.0)))

        direction = env._desired_directions(env._pos, 1 / 60)
        np.testing.assert_allclose(direction[0], [0.0, 1.0])
        self.assertEqual(env.tom0_metrics()['wandering'], 2)
        self.assertEqual(env.tom0_metrics()['following'], 0)


if __name__ == '__main__':
    unittest.main()
