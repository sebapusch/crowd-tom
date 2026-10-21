import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiment import (
    ExperimentConfig, SpawnRegion, build_environment, initial_state_path,
    load_experiment, save_experiment,
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

    def test_multiple_regions_round_trip_and_spawn_in_each(self):
        regions = [
            SpawnRegion((3, 3), (12, 12)),
            SpawnRegion((35, 35), (44, 44)),
        ]
        config = ExperimentConfig(width=50, height=50, agent_count=80,
                                  spawn_regions=regions)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'clusters.yaml'
            save_experiment(config, path)
            loaded = load_experiment(path)
            self.assertEqual(loaded.spawn_regions, regions)
            self.assertEqual(config.clone().spawn_regions, regions)
            self.assertIsNot(config.clone().spawn_regions, config.spawn_regions)
            self.assertNotEqual(
                initial_state_path(config, 7, directory),
                initial_state_path(ExperimentConfig(width=50, height=50, agent_count=80,
                                                  spawn_regions=regions[:1]), 7, directory),
            )

        positions = np.array([agent.position for agent in build_environment(loaded).agents])
        membership = [np.all((positions >= region.minimum) & (positions <= region.maximum),
                             axis=1) for region in regions]
        self.assertTrue(np.all(np.logical_or.reduce(membership)))
        self.assertTrue(all(np.any(group) for group in membership))
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        self.assertGreaterEqual(distances.min(), 2 * config.agent_radius + 0.05 - 1e-9)

    def test_multiple_region_configuration_validation(self):
        region = SpawnRegion((3, 3), (12, 12))
        with self.assertRaisesRegex(ValueError, 'either spawn_region or spawn_regions'):
            ExperimentConfig(spawn_region=region, spawn_regions=[region])
        with self.assertRaisesRegex(ValueError, 'nonempty list'):
            ExperimentConfig(spawn_regions=[])
        with self.assertRaisesRegex(ValueError, 'must fit inside the room'):
            build_environment(ExperimentConfig(
                width=50, height=50, agent_count=1,
                spawn_regions=[region, SpawnRegion((45, 45), (50, 50))],
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
