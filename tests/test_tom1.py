import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent import Agent
from environment import Environment
from exit import Exit
from experiment import build_environment, load_experiment, save_experiment, ExperimentConfig, tom_order_label
from main import _metrics_path
from tom0 import ToM0Params
from tom1 import ToM1Params, ToM1Reasoner, demand_penalty


def reasoner_inputs(other_beliefs=False):
    pos = np.array([[0.0, 0.0], [2.0, 0.0]])
    has_belief = np.array([[True, True], [other_beliefs, other_beliefs]])
    mu = np.zeros((2, 2, 2))
    mu[0] = [[5.0, 0.0], [-10.0, 0.0]]
    visible = np.array([[False, True], [True, False]])
    return {
        "ids": np.array([10, 11]),
        "pos": pos,
        "vel": np.zeros_like(pos),
        "mu": mu,
        "has_belief": has_belief,
        "base_scores": np.zeros((2, 2)),
        "visible_agents": visible,
        "obstacles": [],
        "view_range": 10.0,
        "fov_rad": 2.0 * np.pi,
        "tom0": ToM0Params(),
        "now": 0.0,
    }


class ToM1Tests(unittest.TestCase):
    def test_low_knowledge_confidence_weakens_demand_penalty(self):
        inputs = reasoner_inputs(False)
        inputs["mu"][0] = [[20.0, 0.0], [-20.0, 0.0]]
        inputs["base_scores"][0] = [0.0, -0.02]
        low = ToM1Reasoner(ToM1Params(knowledge_prior=0.001))
        high = ToM1Reasoner(ToM1Params(knowledge_prior=0.5))

        _, low_choice, low_demand = low.choose(**inputs)
        _, high_choice, high_demand = high.choose(**inputs)

        self.assertEqual(low_choice[0], 0)
        self.assertEqual(high_choice[0], 1)
        self.assertLess(low_demand[0].sum(), high_demand[0].sum())

    def test_predicted_demand_uses_fixed_scale(self):
        np.testing.assert_allclose(
            demand_penalty(np.array([0.0, 0.01, 2.0]), 2.0),
            [0.0, 0.01 / 2.01, 0.5],
        )
        with self.assertRaises(ValueError):
            ToM1Params(demand_half_count=0.0)

    def test_prediction_uses_observations_not_another_agents_beliefs(self):
        params = ToM1Params(demand_weight=-2.0)
        first = ToM1Reasoner(params)
        second = ToM1Reasoner(params)
        _, choice_a, demand_a = first.choose(**reasoner_inputs(False))
        _, choice_b, demand_b = second.choose(**reasoner_inputs(True))

        self.assertEqual(choice_a[0], 1)
        self.assertEqual(choice_b[0], 1)
        self.assertGreater(demand_a[0, 0], demand_a[0, 1])
        np.testing.assert_allclose(demand_a[0], demand_b[0])
        self.assertEqual(first.memories[10][11][0], 0.0)
        self.assertNotIn(1, first.memories[10][11])

        departed = reasoner_inputs(False)
        departed["ids"] = departed["ids"][:1]
        departed["pos"] = departed["pos"][:1]
        departed["vel"] = departed["vel"][:1]
        departed["mu"] = departed["mu"][:1]
        departed["has_belief"] = departed["has_belief"][:1]
        departed["base_scores"] = departed["base_scores"][:1]
        departed["visible_agents"] = departed["visible_agents"][:1, :1]
        departed["now"] = 0.1
        first.choose(**departed)
        self.assertNotIn(11, first.memories.get(10, {}))

    def test_memory_is_bounded_and_expires(self):
        params = ToM1Params(memory_agents=2, memory_horizon=2.0)
        reasoner = ToM1Reasoner(params)
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        mu = np.zeros((4, 1, 2))
        mu[0, 0] = [5.0, 0.0]
        visible = np.zeros((4, 4), dtype=bool)
        visible[0, 1:] = True
        inputs = {
            "ids": np.arange(4),
            "pos": pos,
            "vel": np.zeros_like(pos),
            "mu": mu,
            "has_belief": np.array([[True], [False], [False], [False]]),
            "base_scores": np.zeros((4, 1)),
            "visible_agents": visible,
            "obstacles": [],
            "view_range": 20.0,
            "fov_rad": 2.0 * np.pi,
            "tom0": ToM0Params(),
            "now": 0.0,
        }
        reasoner.choose(**inputs)
        self.assertEqual(set(reasoner.memories[0]), {1, 2})

        inputs["now"] = 3.0
        inputs["visible_agents"] = np.zeros_like(visible)
        reasoner.choose(**inputs)
        self.assertNotIn(0, reasoner.memories)

    def test_witness_confidence_decays_without_new_sighting(self):
        inputs = reasoner_inputs(False)
        inputs["has_belief"][0, 1] = False
        reasoner = ToM1Reasoner(
            ToM1Params(knowledge_prior=0.2, confidence_decay=5.0)
        )
        _, _, fresh_demand = reasoner.choose(**inputs)

        inputs["pos"][1] = [-6.0, 0.0]
        inputs["now"] = 5.0
        _, _, stale_demand = reasoner.choose(**inputs)
        self.assertAlmostEqual(fresh_demand[0, 0], 1.0)
        self.assertAlmostEqual(
            stale_demand[0, 0],
            0.2 + 0.8 * np.exp(-1.0),
        )

    def test_environment_mode_one_changes_exit_selection(self):
        agents = [
            Agent(
                idx=index, mass=1.0, radius=0.3,
                social_repulsion=(2000.0, 0.08),
                position=np.array([float(2 * index), 0.0]),
                velocity=np.zeros(2), desired_speed=2.0,
                desired_direction=np.array([1.0, 0.0]), tau=1.0,
            )
            for index in range(2)
        ]
        env = Environment(
            30, 30, agents, view_range=10.0,
            tom0=ToM0Params(w_occupancy=0.0),
            tom1=ToM1Params(demand_weight=-10.0),
            tom_order=0,
        )
        env.add_exit(Exit((5.0, -1.0), (5.0, 1.0)))
        env.add_exit(Exit((-10.0, -1.0), (-10.0, 1.0)))
        env._desired_directions(env._pos, 1 / 60)
        tom0_choice = int(env._chosen_exit[0])

        env.tom_order = 1
        env._desired_directions(env._pos, 1 / 60)
        self.assertEqual(tom0_choice, 0)
        self.assertEqual(env._chosen_exit[0], 1)
        self.assertEqual(env.tom0_metrics()["tom1_changed"], 1)

    def test_yaml_round_trip_and_mode_label(self):
        config = ExperimentConfig(
            tom_order=1,
            view_range=8.0,
            tom1=ToM1Params(model_range=8.0, demand_weight=-0.5),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.yaml"
            save_experiment(config, path)
            loaded = load_experiment(path)
        self.assertEqual(loaded.tom_order, 1)
        self.assertEqual(loaded.tom1, config.tom1)
        self.assertEqual(tom_order_label(1), "ToM-1")
        self.assertEqual(_metrics_path(1).name, "last-metrics-tom-1.csv")
        self.assertEqual(_metrics_path(0).name, "last-metrics-tom-0.csv")

    def test_yaml_view_range_controls_behavioral_radii(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.yaml"
            path.write_text(
                "perception:\n  view_range: 7.0\n"
                "tom1:\n  model_range: 2.0\n"
                "agents:\n  count: 2\n"
            )
            config = load_experiment(path)
            self.assertEqual(config.tom1.model_range, 7.0)
            env = build_environment(config)
            self.assertEqual(env.tom0.occupancy_range, 7.0)
            self.assertEqual(env.tom1.params.model_range, 7.0)
            save_experiment(config, path)
            self.assertNotIn("model_range", path.read_text())


if __name__ == "__main__":
    unittest.main()
