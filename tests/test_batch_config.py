import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import (
    ExperimentConfig, ToMOption, ToMProportions, load_experiment, save_experiment,
)
from layout import SideExit
from main import main, run_seed_range


class BatchConfigTests(unittest.TestCase):
    def test_options_and_seed_range_round_trip_and_validate(self):
        config = ExperimentConfig(
            name='batch-test', agent_count=2, seed_range=(4, 5), score_trace=False,
            tom_options=[
                ToMOption(tom_order=-1), ToMOption(tom_order=1),
                ToMOption(tom_proportions=ToMProportions(0.5, 0.5)),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            scene = save_experiment(config, Path(directory) / 'scene.yaml')
            loaded = load_experiment(scene)
            self.assertEqual(loaded.seed_range, (4, 5))
            self.assertFalse(loaded.score_trace)
            self.assertFalse(config.clone().score_trace)
            self.assertEqual([option.label for option in loaded.tom_options],
                             ['reactive', 'tom-1', 'mixed-tom0-0p5-tom1-0p5'])
            for invalid in (
                'seed: 1\nseed_range: [1, 2]\n',
                'seed_range: [3, 2]\n',
                'tom_options: []\n',
                'tom_options: [{tom_order: 0, tom_proportions: {tom0: 0.5, tom1: 0.5}}]\n',
                'tom_options: [{tom_order: 0}, {tom_order: 0}]\n',
                'score_trace: maybe\n',
            ):
                scene.write_text(invalid)
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    load_experiment(scene)

    def test_cli_seed_options_override_config(self):
        config = ExperimentConfig(
            name='batch-test', agent_count=2, seed_range=(4, 5),
            tom_options=[ToMOption(tom_order=0), ToMOption(tom_order=1)],
        )
        with tempfile.TemporaryDirectory() as directory:
            scene = save_experiment(config, Path(directory) / 'scene.yaml')
            with patch('main.run_seed_range', return_value=[]) as run:
                main([str(scene), '--seed', '9'])
                self.assertEqual(run.call_args.args[1:3], (9, 9))
                main([str(scene), '--seed-range', '7', '8'])
                self.assertEqual(run.call_args.args[1:3], (7, 8))
                main([str(scene)])
                self.assertEqual(run.call_args.args[1:3], (4, 5))
            with patch('main.run_seed_range', return_value=[{'error': 'failed'}]):
                with self.assertRaises(SystemExit) as exit_status:
                    main([str(scene)])
                self.assertEqual(exit_status.exception.code, 1)

            config.seed_range = None
            config.seed = 6
            save_experiment(config, scene)
            with patch('main.run_seed_range', return_value=[]) as run:
                main([str(scene)])
                self.assertEqual(run.call_args.args[1:3], (6, 6))

            config.tom_options = None
            save_experiment(config, scene)
            with patch('main.run_simulation') as interactive, patch('main.run_seed_range', return_value=[]) as run:
                main([str(scene)])
                self.assertEqual(interactive.call_args.kwargs['seed'], 6)
                main([str(scene), '--seed', '8'])
                self.assertEqual(interactive.call_args.kwargs['seed'], 8)
                main([str(scene), '--seed-range', '2', '3'])
                self.assertEqual(run.call_args.args[1:3], (2, 3))
            # The explicit range dispatches headlessly even for a single policy.

    def test_multiple_tom_settings_share_typed_seed_csv(self):
        config = ExperimentConfig(
            name='batch-test', width=20, height=20, agent_count=2,
            tom_proportions=ToMProportions(0.5, 0.5),
            tom_options=[
                ToMOption(tom_order=0), ToMOption(tom_order=1),
                ToMOption(tom_proportions=ToMProportions(0.5, 0.5)),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                summaries = run_seed_range(config, 2, 3, 0.1, 0.1, results)
            self.assertEqual(len(summaries), 6)
            self.assertEqual(len(list((results / 'initial-states').rglob('*.npz'))), 2)
            first_csv = (results / 'batch-test' / 'metrics-2.csv').read_text()
            with contextlib.redirect_stdout(io.StringIO()):
                run_seed_range(config, 2, 2, 0.1, 0.1, results)
            self.assertEqual((results / 'batch-test' / 'metrics-2.csv').read_text(), first_csv)
            for seed in (2, 3):
                path = results / 'batch-test' / f'metrics-{seed}.csv'
                with path.open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual({row['tom_config'] for row in rows},
                                 {'tom-0', 'tom-1', 'mixed-tom0-0p5-tom1-0p5'})
                self.assertEqual({row['seed'] for row in rows}, {str(seed)})
                self.assertEqual({row['dt'] for row in rows}, {'0.1'})
                self.assertEqual({row['max_time'] for row in rows}, {'0.1'})
                self.assertEqual({row['agent_type'] for row in rows},
                                 {'reactive', 'tom0', 'tom1'})
                self.assertEqual(len(rows), 18)
                starts = {(row['tom_config'], row['agent_type']): int(row['remaining'])
                          for row in rows if float(row['t']) == 0.0}
                self.assertEqual(starts['tom-0', 'tom0'], 2)
                self.assertEqual(starts['tom-0', 'tom1'], 0)
                self.assertEqual(starts['tom-1', 'tom1'], 2)
                self.assertEqual(starts['tom-1', 'tom0'], 0)
                self.assertEqual(starts['mixed-tom0-0p5-tom1-0p5', 'tom0'], 1)
                self.assertEqual(starts['mixed-tom0-0p5-tom1-0p5', 'tom1'], 1)
                for label in ('tom-0', 'tom-1', 'mixed-tom0-0p5-tom1-0p5'):
                    snapshot = results / 'batch-test' / f'config-{seed}-{label}.yaml'
                    self.assertTrue(snapshot.exists())
                    self.assertEqual(load_experiment(snapshot).seed, seed)

    def test_failed_setting_is_logged_and_next_setting_runs(self):
        config = ExperimentConfig(
            name='failure-test', width=20, height=20, agent_count=2,
            tom_options=[ToMOption(tom_order=0), ToMOption(tom_order=1)],
        )
        from experiment import build_environment

        def maybe_fail(resolved, **kwargs):
            if resolved.tom_order == 0:
                raise RuntimeError('deliberate failure')
            return build_environment(resolved, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            with patch('main.build_environment', side_effect=maybe_fail):
                with contextlib.redirect_stdout(io.StringIO()):
                    summaries = run_seed_range(config, 4, 4, 0.1, 0.1, results)
            self.assertIn('error', summaries[0])
            self.assertEqual(summaries[1]['tom_config'], 'tom-1')
            with (results / 'failure-test' / 'metrics-4.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row['tom_config'] for row in rows}, {'tom-1'})
            failures = [json.loads(line) for line in
                        (results / 'failure-test' / 'failures.jsonl').read_text().splitlines()]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]['seed'], 4)
            self.assertEqual(failures[0]['tom_config'], 'tom-0')
            self.assertIn('deliberate failure', failures[0]['traceback'])

    def test_nonzero_seed_reports_when_no_agent_has_two_exit_choices(self):
        config = ExperimentConfig(
            name='no-trace-candidate', width=20, height=20, agent_count=2,
            exits=[SideExit('east', 10.0, 2.0)],
            tom_options=[ToMOption(tom_order=1)],
        )
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                run_seed_range(config, 7, 7, 0.1, 0.1, Path(directory))
            path = Path(directory) / 'no-trace-candidate' / 'score-trace-7-tom-1.csv'
            with path.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['status'], 'no_eligible_agent')

    def test_config_and_cli_toggle_traces_for_every_seed(self):
        config = ExperimentConfig(
            name='trace-toggle', width=20, height=20, agent_count=2,
            seed_range=(7, 8), score_trace=False,
            exits=[SideExit('east', 10.0, 2.0)],
            tom_options=[ToMOption(tom_order=1)],
        )
        with tempfile.TemporaryDirectory() as directory:
            scene = save_experiment(config, Path(directory) / 'scene.yaml')
            argv = [str(scene), '--dt', '0.1', '--max-time', '0.1',
                    '--results-dir', directory]
            with contextlib.redirect_stdout(io.StringIO()):
                main(argv)
            for seed in (7, 8):
                trace = Path(directory) / 'trace-toggle' / f'score-trace-{seed}-tom-1.csv'
                self.assertFalse(trace.exists())
                self.assertFalse(load_experiment(trace.parent / f'config-{seed}-tom-1.yaml').score_trace)
            with contextlib.redirect_stdout(io.StringIO()):
                main([*argv, '--score-trace'])
            for seed in (7, 8):
                trace = Path(directory) / 'trace-toggle' / f'score-trace-{seed}-tom-1.csv'
                with trace.open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(rows[0]['seed'], str(seed))
                self.assertEqual(rows[0]['status'], 'no_eligible_agent')
                self.assertTrue(load_experiment(trace.parent / f'config-{seed}-tom-1.yaml').score_trace)
            config.score_trace = True
            save_experiment(config, scene)
            with contextlib.redirect_stdout(io.StringIO()):
                main([*argv, '--no-score-trace'])
            for seed in (7, 8):
                trace = Path(directory) / 'trace-toggle' / f'score-trace-{seed}-tom-1.csv'
                self.assertFalse(trace.exists())
                self.assertFalse(load_experiment(trace.parent / f'config-{seed}-tom-1.yaml').score_trace)


if __name__ == '__main__':
    unittest.main()
