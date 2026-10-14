import unittest

import numpy as np

from tom0 import ToM0Params, distance_penalty, utilities
from tom1 import ToM1Params, ToM1Reasoner


class DistanceScoreTests(unittest.TestCase):
    def test_distance_score_keeps_small_differences_small(self):
        params = ToM0Params(distance_scale=10.0)
        scores = utilities(
            np.array([[5.0, 5.01]]), np.zeros((1, 2)),
            np.zeros((1, 2)), np.ones((1, 2), dtype=bool), params,
        )
        difference = distance_penalty(5.01, 10.0) - distance_penalty(5.0, 10.0)
        self.assertAlmostEqual(scores[0, 0] - scores[0, 1], difference)
        self.assertLess(difference, 0.001)
        self.assertAlmostEqual(distance_penalty(30.0, 10.0), 1.0)

    def test_tom1_predicts_with_same_distance_curve(self):
        reasoner = ToM1Reasoner(ToM1Params())
        _, _, demand = reasoner.choose(
            ids=np.array([10, 11]),
            pos=np.array([[0.0, 0.0], [2.0, 0.0]]),
            vel=np.zeros((2, 2)),
            mu=np.array([[[5.0, 0.0], [10.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
            has_belief=np.array([[True, True], [False, False]]),
            base_scores=np.zeros((2, 2)),
            visible_agents=np.array([[False, True], [True, False]]),
            obstacles=[], view_range=1.0, fov_rad=2.0 * np.pi,
            tom0=ToM0Params(w_occupancy=0.0, w_uncertainty=0.0), now=0.0,
            observer_mask=np.array([True, False]),
        )
        distance_scores = -distance_penalty(np.array([3.0, 8.0]), 10.0)
        weights = np.exp(distance_scores)
        expected = 0.36 * weights[0] / weights.sum()
        self.assertAlmostEqual(demand[0, 0], expected)


if __name__ == "__main__":
    unittest.main()
