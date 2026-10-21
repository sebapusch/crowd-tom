import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiment import (
    DEFAULT_TEST_SEED, ExperimentConfig, SpawnRegion, ToMProportions,
    build_environment, initial_state_path, save_experiment,
)
from layout import SideExit
from main import main, run_seed_range


def crowd_arrays(environment):
    ids = np.array([agent.idx for agent in environment.agents])
    positions = np.array([agent.position for agent in environment.agents])
    headings = np.array([agent.desired_direction for agent in environment.agents])
    return ids, positions, headings


class SeedingTests(unittest.TestCase):
    def test_same_seed_replays_initial_state_and_wandering(self):
        config = ExperimentConfig(
            width=25, height=25, agent_count=5, view_range=2.0,
            exits=[SideExit('east', 12.5, 3.0)],
        )
        first = build_environment(config, seed=42)
        second = build_environment(config, seed=42)
        different = build_environment(config, seed=43)
        for left, right in zip(crowd_arrays(first), crowd_arrays(second)):
            np.testing.assert_array_equal(left, right)
        self.assertFalse(np.array_equal(first._pos, different._pos))

        for _ in range(12):
            first.tick(1 / 60)
            second.tick(1 / 60)
        np.testing.assert_array_equal(first._pos, second._pos)
        np.testing.assert_array_equal(first._vel, second._vel)
        np.testing.assert_array_equal(first._heading, second._heading)
        self.assertEqual(first.evacuation_times, second.evacuation_times)

    def test_same_seed_replays_departures(self):
        config = ExperimentConfig(
            width=10, height=10, agent_count=2,
            spawn_region=SpawnRegion((7.0, 4.0), (8.5, 6.0)),
            view_range=5.0, exits=[SideExit('east', 5.0, 4.0)],
        )
        first = build_environment(config, seed=42)
        second = build_environment(config, seed=42)
        for _ in range(180):
            first.tick(1 / 60)
            second.tick(1 / 60)
        self.assertEqual(first.evacuation_times, second.evacuation_times)
        self.assertEqual(len(first.evacuation_times), 2)

    def test_policy_modes_share_saved_initial_crowd(self):
        config = ExperimentConfig(
            width=25, height=25, agent_count=12,
            exits=[SideExit('east', 12.5, 3.0)],
        )
        with tempfile.TemporaryDirectory() as directory:
            baseline = build_environment(config, seed=7, initial_state_dir=directory)
            saved_path = initial_state_path(config, 7, directory)
            self.assertTrue(saved_path.exists())

            for order in (-1, 1):
                changed = config.clone()
                changed.tom_order = order
                env = build_environment(changed, seed=7, initial_state_dir=directory)
                for left, right in zip(crowd_arrays(baseline), crowd_arrays(env)):
                    np.testing.assert_array_equal(left, right)

            mixed = config.clone()
            mixed.tom_proportions = ToMProportions(0.5, 0.5)
            first_mix = build_environment(mixed, seed=7, initial_state_dir=directory)
            second_mix = build_environment(mixed, seed=7, initial_state_dir=directory)
            uncached_mix = build_environment(mixed, seed=7)
            for left, right in zip(crowd_arrays(baseline), crowd_arrays(first_mix)):
                np.testing.assert_array_equal(left, right)
            for left, right in zip(crowd_arrays(baseline), crowd_arrays(uncached_mix)):
                np.testing.assert_array_equal(left, right)
            self.assertEqual(
                [agent.tom_order for agent in first_mix.agents],
                [agent.tom_order for agent in second_mix.agents],
            )
            self.assertEqual(len(list(Path(directory).rglob('*.npz'))), 1)
            with np.load(saved_path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['positions'], baseline._pos)
                np.testing.assert_array_equal(saved['headings'], baseline._heading)

    def test_cli_defaults_and_inclusive_seed_range(self):
        config = ExperimentConfig(
            name='seed-test', width=20, height=20, agent_count=2,
            exits=[SideExit('east', 10.0, 2.0)],
        )
        with tempfile.TemporaryDirectory() as directory:
            scene = save_experiment(config, Path(directory) / 'scene.yaml')
            results = Path(directory) / 'results'
            with patch('main.run_simulation') as interactive:
                main([str(scene), '--results-dir', str(results)])
                self.assertEqual(interactive.call_args.kwargs['seed'], DEFAULT_TEST_SEED)
                main([str(scene), '--seed', '9', '--results-dir', str(results)])
                self.assertEqual(interactive.call_args.kwargs['seed'], 9)

            with contextlib.redirect_stdout(io.StringIO()):
                summaries = run_seed_range(config, 4, 5, 1 / 60, 1 / 60, results)
                repeated = run_seed_range(config, 4, 5, 1 / 60, 1 / 60, results)
            self.assertEqual([item['seed'] for item in summaries], [4, 5])
            self.assertEqual(summaries, repeated)
            self.assertEqual(len(list((results / 'initial-states').rglob('*.npz'))), 2)
            self.assertEqual(
                sorted(path.name for path in (results / 'seed-test').glob('metrics-*.csv')),
                ['metrics-4.csv', 'metrics-5.csv'],
            )
            self.assertIn('agent_type', (results / 'seed-test' / 'metrics-4.csv').read_text().splitlines()[0])


if __name__ == '__main__':
    unittest.main()
