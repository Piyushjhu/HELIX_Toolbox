"""Compile successful ALPSS traces while reporting per-file failures."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ALPSS'))
import alpss_main
import helix_cli_runner as runner
from helix_analysis_toolbox import AnalysisThread

FIXTURE = ROOT / 'regression/data/C1--20251022--00001--vel-smooth-with-uncert.csv'


class PartialProcessingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = json.loads((ROOT / 'helix_master_config.json').read_text())
        self.config['alpss_config'].update(
            save_velocity_csv=False, save_velocity_smooth_csv=False,
            save_velocity_uncert_csv=False, save_velocity_smooth_uncert_csv=True,
            save_results_csv=False, save_noise_csv=False, smart_selection_enabled=False)
        self.config['spade_config'] = {
            'spall_analysis_enabled': True, 'experiment_spall_analysis': True,
            'hel_detection_enabled': False, 'experiment_hel_detection': False,
            'plot_individual': False, 'generate_all_velocity_plot': False,
        }

    def write_velocity(self, output, stem):
        path = output / (stem + '--vel-smooth-with-uncert.csv')
        shutil.copyfile(FIXTURE, path)
        return str(path)

    @contextlib.contextmanager
    def processing(self, failure='false'):
        def process(**params):
            stem = Path(params['filename']).stem
            if stem == 'bad':
                if failure == 'exception':
                    raise ValueError('No IQ threshold crossing')
                if failure == 'false':
                    return False
                if failure == 'missing':
                    return True  # Required output was never generated.
            self.write_velocity(Path(params['out_files_dir']), stem)
            return True

        with contextlib.ExitStack() as stack:
            # Keep actual SPADE processing and both CSV compilers; suppress plots.
            for name in dir(AnalysisThread):
                if name.startswith('generate_') and name not in (
                        'generate_velocity_shots_summary', 'generate_spall_analysis_summary'):
                    stack.enter_context(patch.object(AnalysisThread, name))
            stack.enter_context(patch('importlib.reload', side_effect=lambda module: module))
            stack.enter_context(patch.object(alpss_main, 'detect_sample_rate', return_value=1e9))
            stack.enter_context(patch.object(alpss_main, 'alpss_main', side_effect=process))
            yield

    def test_partial_run_compiles_only_successes_and_can_recover(self):
        for failure in ('false', 'exception', 'missing'):
            with self.subTest(failure=failure):
                folder = self.root / failure
                folder.mkdir()
                output = folder / 'Output'
                output.mkdir()
                inputs = [folder / (stem + '.csv') for stem in ('bad', 'good')]
                for source in inputs:
                    source.write_text('Time,Ampl\n0,1\n')
                # Old failed-shot outputs must never enter this run's summaries.
                if failure != 'missing':
                    self.write_velocity(output, 'bad')
                self.write_velocity(output, 'unrelated')
                thread = AnalysisThread(
                    runner._transform_alpss_params_for_analysis(self.config['alpss_config']),
                    self.config['spade_config'], list(map(str, inputs)), str(output),
                    analysis_mode='both')
                finished = []
                thread.finished_signal.connect(lambda ok, message: finished.append((ok, message)))
                with self.processing(failure), contextlib.redirect_stdout(io.StringIO()):
                    thread.run()
                self.assertTrue(finished[-1][0], finished[-1][1])
                self.assertIn('1 failed ALPSS input', finished[-1][1])
                self.assertEqual(thread.successful_files, [str(inputs[1])])
                failures = pd.read_csv(output / 'failed_data_files.csv')
                self.assertEqual(failures['input_file'].tolist(), ['bad.csv'])
                spade = output / 'SPADE_analysis'
                velocity = pd.read_csv(spade / 'velocity_shots_summary.csv')
                self.assertEqual(velocity['file_name'].tolist(), ['good'])
                spall = pd.read_csv(spade / 'spall_summary.csv')
                self.assertEqual(spall['Filename'].tolist(), ['good'])
                master = pd.read_csv(spade / f'{failure}-Data_Summary.csv')
                self.assertEqual(len(master), 1)
                snapshot = json.loads((spade / f'{failure}-Run_Config.json').read_text())
                self.assertEqual(snapshot['alpss_processing']['successful_files'], [str(inputs[1])])
                self.assertEqual(len(snapshot['alpss_processing']['failed_files']), 1)
                scanned = [str(p) for p in output.glob('*--vel-smooth-with-uncert.csv')]
                self.assertEqual(thread._scope_files(scanned),
                                 [str(output / 'good--vel-smooth-with-uncert.csv')])
                for column in ('file_name', 'Filename', 'filename', 'PDV_FileName'):
                    frame = pd.DataFrame({column: ['bad', 'good', 'unrelated']})
                    self.assertEqual(thread._scope_summary(frame)[column].tolist(), ['good'])

                with self.processing(failure=None), contextlib.redirect_stdout(io.StringIO()):
                    thread.run()
                self.assertTrue(finished[-1][0], finished[-1][1])
                self.assertEqual(thread.failed_files, [])
                self.assertFalse((output / 'failed_data_files.csv').exists())
                velocity = pd.read_csv(spade / 'velocity_shots_summary.csv')
                self.assertEqual(set(velocity['file_name']), {'bad', 'good'})

    def test_batch_completes_partial_folder_and_reports_failures(self):
        for group, stems in [('partial', ('good', 'bad')), ('complete', ('good',))]:
            folder = self.root / 'inputs' / group
            folder.mkdir(parents=True)
            for stem in stems:
                (folder / (stem + '.csv')).write_text('Time,Ampl\n0,1\n')
        self.config['cli_settings'].update(
            input_dir=str(self.root / 'inputs'), input_files=None, output_dir=None,
            param_folder=None, batch_mode=True, data_mode='single_pdv',
            analysis_mode='both', spade_mode='auto', output_subdir_name='Output',
            combined_output_dir=None,
            input_preparation={'cache_dir': str(self.root / 'cache')})
        cfg = self.root / 'config.json'
        cfg.write_text(json.dumps(self.config))
        stdout = io.StringIO()
        with self.processing(), contextlib.redirect_stdout(stdout), \
                patch.object(sys, 'argv', ['runner', '--config', str(cfg)]), \
                self.assertRaises(SystemExit) as result:
            runner.main()
        self.assertEqual(result.exception.code, 0, stdout.getvalue()[-4000:])
        self.assertIn('partial: compiled 1 successful trace(s); 1 failed', stdout.getvalue())
        self.assertIn('2/2 subfolder(s) completed; 1 with failed inputs', stdout.getvalue())
        for group in ('partial', 'complete'):
            summary = self.root / 'inputs' / group / 'Output/SPADE_analysis/velocity_shots_summary.csv'
            self.assertEqual(pd.read_csv(summary)['file_name'].tolist(), ['good'])


if __name__ == '__main__':
    unittest.main()
