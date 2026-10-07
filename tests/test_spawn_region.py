import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiment import (
    ExperimentConfig, SpawnRegion, build_environment, load_experiment,
    save_experiment,
)


class SpawnRegionTests(unittest.TestCase):
    def test_cluster_stays_in_bounds_without_overlaps_and_round_trips(self):
        region = SpawnRegion((3.0, 3.0), (30.0, 23.0))
        config = ExperimentConfig(
            width=50, height=50, agent_count=180, spawn_region=region,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cluster.yaml'
            save_experiment(config, path)
            loaded = load_experiment(path)
        self.assertEqual(loaded.spawn_region, region)
        self.assertEqual(config.clone().spawn_region, region)

        positions = np.array([agent.position for agent in build_environment(loaded).agents])
        self.assertTrue(np.all(positions >= region.minimum))
        self.assertTrue(np.all(positions <= region.maximum))
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        self.assertGreaterEqual(distances.min(), 2 * config.agent_radius + 0.05 - 1e-9)

    def test_invalid_regions_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, 'minimum must be below maximum'):
            SpawnRegion((5, 5), (5, 10))
        with self.assertRaisesRegex(ValueError, 'must fit inside the room'):
            build_environment(ExperimentConfig(
                width=50, height=50, spawn_region=SpawnRegion((0, 3), (10, 10)),
            ))

    def test_train_scene_uses_cluster(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/train-uneven-doors.yaml'
        config = load_experiment(path)
        region = config.spawn_region
        self.assertIsNotNone(region)
        positions = np.array([agent.position for agent in build_environment(config).agents])
        self.assertEqual(len(positions), 180)
        self.assertTrue(np.all(positions >= region.minimum))
        self.assertTrue(np.all(positions <= region.maximum))


if __name__ == '__main__':
    unittest.main()
