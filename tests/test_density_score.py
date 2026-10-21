import unittest

import numpy as np

from tom0 import ToM0Params, occupancy_penalty, utilities
from tom1 import ToM1Params, ToM1Reasoner


class DensityScoreTests(unittest.TestCase):
    def test_fixed_density_scale_keeps_small_differences_small(self):
        params = ToM0Params(occupancy_half_density=0.04)
        np.testing.assert_allclose(
            occupancy_penalty(np.array([0.0, 0.04, 0.08]), 0.04),
            [0.0, 0.5, 2.0 / 3.0],
        )
        one_agent_density = 1.0 / (0.5 * 12.0**2 * np.pi / 3.0)
        scores = utilities(
            np.zeros((1, 2)), np.array([[0.0, one_agent_density]]),
            np.zeros((1, 2)), np.ones((1, 2), dtype=bool), params,
        )
        difference = scores[0, 0] - scores[0, 1]
        self.assertAlmostEqual(difference, occupancy_penalty(one_agent_density, 0.04))
        self.assertLess(difference, 0.3)

    def test_tom1_prediction_uses_same_density_scale(self):
        reasoner = ToM1Reasoner(ToM1Params())
        _, _, demand = reasoner.choose(
            ids=np.array([10, 11]),
            pos=np.array([[0.0, 0.0], [2.0, 0.0]]),
            vel=np.zeros((2, 2)),
            mu=np.array([[[5.0, 0.0], [-1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
            has_belief=np.array([[True, True], [False, False]]),
            base_scores=np.zeros((2, 2)),
            visible_agents=np.array([[False, True], [True, False]]),
            obstacles=[], view_range=1.0, fov_rad=2.0 * np.pi,
            tom0=ToM0Params(w_distance=0.0, w_uncertainty=0.0), now=0.0,
            observer_mask=np.array([True, False]),
        )
        one_agent_density = 1.0 / (0.5 * 12.0**2 * np.pi / 3.0)
        penalty = occupancy_penalty(one_agent_density, 0.04)
        expected = 0.36 / (1.0 + np.exp(-penalty))
        self.assertAlmostEqual(demand[0, 0], expected)

    def test_batched_occupancy_respects_each_observers_visible_group(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        mu = np.array([
            [[4.0, 0.0], [0.0, 4.0]],
            [[4.0, 1.0], [0.0, 3.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[4.0, 3.0], [0.0, 4.0]],
        ])
        modeled = np.array([
            [False, True, True, False],
            [True, False, False, True],
            [False, False, False, False],
            [False, True, False, False],
        ])
        delta = pos[None, :, :] - pos[:, None, :]
        distance = np.linalg.norm(delta, axis=2)
        directions = delta / np.maximum(distance[:, :, None], 1e-12)
        params = ToM0Params(occupancy_range=3.0, occupancy_fov_rad=np.pi / 2)
        eligible = (distance > 1e-12) & (distance <= params.occupancy_range)
        actual = ToM1Reasoner(ToM1Params())._estimated_occupancy(
            pos, mu, np.array([[True, True], [True, True], [False, False], [True, True]]),
            modeled, np.array([True, True, False, True]), directions, eligible, params,
        )
        area = 0.5 * params.occupancy_range**2 * params.occupancy_fov_rad
        expected = np.zeros_like(actual)
        for i in (0, 1, 3):
            group = set(np.flatnonzero(modeled[i])) | {i}
            for j in range(len(pos)):
                for k in range(2):
                    axis = mu[i, k] - pos[j]
                    axis /= max(np.linalg.norm(axis), 1e-12)
                    expected[i, j, k] = sum(
                        eligible[j, other]
                        and np.dot(directions[j, other], axis) >= np.cos(np.pi / 4)
                        for other in group
                    ) / area
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
