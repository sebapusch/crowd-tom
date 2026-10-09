import unittest

import numpy as np

from tom0 import ToM0Params, uncertainty, utilities
from tom1 import ToM1Params, ToM1Reasoner


class UncertaintyTests(unittest.TestCase):
    def test_age_penalty_uses_a_fixed_scale(self):
        params = ToM0Params(uncertainty_time=10.0)
        np.testing.assert_allclose(
            uncertainty(np.array([0.0, 0.01, 10.0]), params.uncertainty_time),
            [0.0, 0.01 / 10.01, 0.5],
        )
        scores = utilities(
            np.zeros((1, 2)), np.zeros((1, 2)),
            np.array([[0.0, 0.01]]), np.ones((1, 2), dtype=bool), params,
        )
        self.assertAlmostEqual(scores[0, 0] - scores[0, 1], 0.01 / 10.01)

    def test_only_known_exit_retains_age_penalty(self):
        scores = utilities(
            np.zeros((1, 2)), np.zeros((1, 2)),
            np.array([[10.0, 0.0]]), np.array([[True, False]]), ToM0Params(),
        )
        self.assertAlmostEqual(scores[0, 0], -0.25)
        self.assertTrue(np.isneginf(scores[0, 1]))

    def test_tom1_prediction_uses_same_uncertainty_scale(self):
        reasoner = ToM1Reasoner(ToM1Params(knowledge_prior=0.999))
        reasoner.memories = {10: {11: {0: 0.0}}}
        _, _, demand = reasoner.choose(
            ids=np.array([10, 11]),
            pos=np.array([[0.0, 0.0], [2.0, 0.0]]),
            vel=np.zeros((2, 2)),
            mu=np.array([[[5.0, 0.0], [-1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
            has_belief=np.array([[True, True], [False, False]]),
            base_scores=np.zeros((2, 2)),
            visible_agents=np.array([[False, True], [True, False]]),
            obstacles=[], view_range=1.0, fov_rad=2.0 * np.pi,
            tom0=ToM0Params(w_distance=0.0, w_occupancy=0.0), now=0.0,
            observer_mask=np.array([True, False]),
        )
        expected = 1.0 / (1.0 + 0.999 * np.exp(-0.5))
        self.assertAlmostEqual(demand[0, 0], expected)


if __name__ == "__main__":
    unittest.main()
