"""Cloud availability, reconciliation, and incomplete-run regressions."""
import contextlib
import io
import json
import os
import hashlib
import subprocess
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ALPSS'))
from helix_file_io import InputUnavailableError, file_availability, read_parameter_table, require_local_files, download_and_verify
from helix_input_audit import audit_inputs
from helix_data_source import match_logged_files
from helix_cli_runner import _load_parameter_folder, _resolve_file_list, _transform_alpss_params_for_analysis
from helix_analysis_toolbox import AnalysisThread
import alpss_main


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.params = self.root / 'parameters'
        self.params.mkdir()
        self.data = self.root / 'sample'
        self.data.mkdir()

    def test_placeholder_flags_without_reading_content(self):
        for flags, attributes in [(0x40000000, 0), (0, 0x1000), (0, 0x400000)]:
            info = types.SimpleNamespace(st_mode=0o100644, st_size=500,
                                         st_flags=flags, st_file_attributes=attributes)
            with patch('helix_file_io.os.stat', return_value=info):
                self.assertEqual(file_availability('cloud.csv'), 'cloud_only')
                with self.assertRaisesRegex(InputUnavailableError, 'Always Keep'):
                    require_local_files(['cloud.csv'])

    def test_missing_empty_and_local_files(self):
        path = self.data / 'trace.csv'
        self.assertEqual(file_availability(path), 'missing')
        path.touch()
        self.assertEqual(file_availability(path), 'empty')
        path.write_text('Time,Ampl\n0,1\n')
        self.assertEqual(file_availability(path), 'local')
        with self.assertRaisesRegex(FileNotFoundError, 'run stopped'):
            _resolve_file_list([str(path), str(self.data/'missing.csv')], None, '*.csv')

    def test_parameter_errors_are_not_skipped(self):
        path = self.params / 'log.csv'
        path.write_text('PDV_FileName\ntrace\n')
        with patch('pandas.read_csv', side_effect=TimeoutError('sync failed')):
            with self.assertRaisesRegex(InputUnavailableError, 'metadata was not skipped'):
                _load_parameter_folder(str(self.params))

    def test_parameter_timeout_does_not_wait_for_worker(self):
        path = self.params / 'log.csv'
        path.write_text('PDV_FileName\ntrace\n')
        release = threading.Event()
        try:
            with patch('pandas.read_csv', side_effect=lambda _: release.wait(10)):
                with self.assertRaisesRegex(InputUnavailableError, 'timed out'):
                    read_parameter_table(path, timeout=0.02)
        finally:
            release.set()

    def settings(self):
        return {'input_dir': str(self.root), 'param_folder': str(self.params),
                'batch_mode': True, 'data_mode': 'single_pdv'}

    def test_audit_reconciles_missing_and_intentionally_excluded(self):
        pd.DataFrame([{'PDV_10_FileName': 'C1--sample_shot01', 'PDV_6_FileName': 'C3--sample_shot01'},
                      {'PDV_10_FileName': 'C1--sample_shot02'},
                      {'PDV_10_FileName': None}]).to_csv(self.params/'log.csv', index=False)
        for name in ('C1--sample_shot01', 'C2--sample_shot01', 'C3--sample_shot01'):
            (self.data/(name+'.csv')).write_text('Time,Ampl\n0,1\n')
        report = audit_inputs(self.settings(), emit=lambda _: None)
        self.assertTrue(report['reconciliation_complete'])
        self.assertFalse(report['ready'])
        self.assertEqual(report['summary']['expected_status'], {'local': 1, 'missing': 1})
        self.assertEqual(report['summary']['file_roles'], {'expected': 1, 'noncentral_mpdv': 2})
        self.assertEqual(report['summary']['waveform_files'], 3)
        (self.data/'C1--sample_shot02.csv').write_text('Time,Ampl\n0,1\n')
        self.assertTrue(audit_inputs(self.settings(), emit=lambda _: None)['ready'])

    def test_legacy_export_wrapper_is_unambiguous_and_mpdv_stays_exact(self):
        key = 'sample_2026-07-14_13-00-00_shot01'
        first = 'C1--' + key + '--00000.csv'
        second = 'C2--' + key + '--00000.csv'
        self.assertEqual(match_logged_files(key, [first]), [first])
        self.assertEqual(match_logged_files(key, [first, second]), [first, second])
        self.assertEqual(match_logged_files(key, [first], 'PDV_10'), [])
        self.assertEqual(match_logged_files(first, [second], 'PDV_10'), [])

    def test_audit_does_not_certify_unreadable_metadata(self):
        path = self.params/'log.csv'
        path.write_text('PDV_FileName\ntrace\n')
        actual = file_availability
        with patch('helix_input_audit.file_availability', side_effect=lambda p: 'cloud_only' if Path(p)==path else actual(p)), \
                patch('helix_input_audit.read_parameter_table') as read:
            report = audit_inputs(self.settings(), emit=lambda _: None)
        read.assert_not_called()
        self.assertFalse(report['ready'])
        self.assertFalse(report['reconciliation_complete'])

    def test_io_failure_returns_false_and_does_not_retry_for_error_plot(self):
        with patch.object(alpss_main, 'detect_sample_rate', return_value=1e9), \
                patch.object(alpss_main, 'spall_doi_finder', side_effect=TimeoutError('cloud read')), \
                patch.object(alpss_main.pd, 'read_csv') as reader, \
                contextlib.redirect_stdout(io.StringIO()):
            result = alpss_main.alpss_main(sample_rate=1e9)
        self.assertFalse(result)
        reader.assert_not_called()

    def test_failed_pipeline_cannot_reuse_old_outputs_or_report_success(self):
        source = self.data/'trace.csv'
        source.write_text('Time,Ampl\n0,1\n')
        out = self.root/'Output'
        out.mkdir()
        for suffix in ('velocity--smooth','vel-smooth-with-uncert','results','noise--frac'):
            (out/f'trace--{suffix}.csv').write_text('old output\n')
        params = json.loads((ROOT/'helix_master_config.json').read_text())['alpss_config']
        thread = AnalysisThread(_transform_alpss_params_for_analysis(params), {}, [str(source)],
                                str(out), analysis_mode='alpss_only')
        finished = []
        thread.finished_signal.connect(lambda ok, message: finished.append((ok, message)))
        with patch('importlib.reload', side_effect=lambda module: module), \
                patch.object(alpss_main, 'detect_sample_rate', return_value=1e9), \
                patch.object(alpss_main, 'alpss_main', return_value=False), \
                contextlib.redirect_stdout(io.StringIO()):
            thread.run()
        self.assertFalse(finished[-1][0])
        self.assertIn('Incomplete ALPSS run', finished[-1][1])
        self.assertEqual(thread.successful_files, [])
        self.assertTrue((out/'failed_data_files.csv').exists())


class DownloadTests(unittest.TestCase):
    def test_verified_copy_and_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'trace.csv'
            content = b'Time,Ampl\n0,1\n' * 100
            source.write_bytes(content)
            cache = Path(folder)/'cache'
            result = download_and_verify([source], cache, emit=lambda _: None)
            self.assertEqual(Path(result[str(source)]).read_bytes(), content)
            report = json.loads((cache/'download_manifest.json').read_text())
            self.assertTrue(report['ready'])
            self.assertEqual(report['files'][0]['sha256'], hashlib.sha256(content).hexdigest())

    def test_retries_exhausted_never_accept_existing_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'trace.csv'
            source.write_text('valid data')
            cache = Path(folder)/'cache'
            old = download_and_verify([source], cache, emit=lambda _: None)
            with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('worker', 0.1)) as run:
                with self.assertRaisesRegex(InputUnavailableError, 'Analysis has not started'):
                    download_and_verify([source], cache, timeout=0.1, attempts=2, retry_delay=0, emit=lambda _: None)
            self.assertEqual(run.call_count, 2)
            self.assertTrue(Path(old[str(source)]).exists())
            self.assertFalse(json.loads((cache/'download_manifest.json').read_text())['ready'])

    def test_retry_can_recover(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'trace.csv'
            source.write_text('valid data')
            actual = subprocess.run
            attempts = []
            def run(*args, **kwargs):
                attempts.append(1)
                if len(attempts) == 1:
                    return subprocess.CompletedProcess(args[0], 1, '', 'temporary sync timeout')
                return actual(*args, **kwargs)
            with patch('subprocess.run', side_effect=run):
                result = download_and_verify([source], Path(folder)/'cache', attempts=2, retry_delay=0, emit=lambda _: None)
            self.assertEqual(Path(result[str(source)]).read_text(), 'valid data')
            self.assertEqual(len(attempts), 2)

    def test_preflight_maps_legacy_wrapper_to_correct_shot_metadata(self):
        import helix_cli_runner as runner
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root/'sample'; data.mkdir()
            params = root/'params'; params.mkdir()
            pd.DataFrame([{'PDV_FileName': 'sample_shot01', 'Exp_ID': 1},
                          {'PDV_FileName': 'sample_shot02', 'Exp_ID': 2}]).to_csv(params/'log.csv', index=False)
            for stem in ('C2--sample_shot01--00000', 'C1--sample_shot02--00000'):
                (data/(stem+'.csv')).write_text('Time,Ampl\n0,1\n')
            options = dict(input_dir=str(data), input_files=None, input_pattern='*.csv',
                           param_folder=str(params), data_mode='single_pdv', batch_mode=False,
                           subfolder_pattern='*', analysis_mode='alpss_only', spade_mode='auto',
                           spade_input_files=None, spade_input_dir=None, spade_input_pattern='*.csv',
                           options={'cache_dir': str(root/'cache')})
            with contextlib.redirect_stdout(io.StringIO()):
                metadata, groups, _ = runner._prepare_run_sources(**options)
            self.assertEqual(len(groups['single']), 2)
            self.assertEqual(metadata['C2--sample_shot01--00000']['Exp_ID'], 1)
            self.assertEqual(metadata['C1--sample_shot02--00000']['Exp_ID'], 2)
            (data/'C1--sample_shot01--00000.csv').write_text('Time,Ampl\n0,1\n')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Ambiguous input'):
                runner._prepare_run_sources(**options)
            (data/'C1--sample_shot01--00000.csv').unlink()
            (data/'C1--sample_shot02--00000.csv').unlink()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Logged shot files are missing'):
                runner._prepare_run_sources(**options)

    def test_prepare_only_downloads_but_does_not_analyze(self):
        import helix_cli_runner as runner
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root/'sample'; data.mkdir()
            (data/'trace.csv').write_text('Time,Ampl\n0,1\n')
            config = json.loads((ROOT/'helix_master_config.json').read_text())
            config['cli_settings'].update(input_dir=str(data), input_files=None,
                output_dir=str(root/'Output'), param_folder=None, batch_mode=False,
                input_preparation={'cache_dir': str(root/'cache')})
            cfg = root/'config.json'; cfg.write_text(json.dumps(config))
            with patch.object(sys, 'argv', ['runner', '--config', str(cfg), '--prepare-only']), \
                    patch.object(AnalysisThread, 'run') as analyze, \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                runner.main()
            self.assertEqual(result.exception.code, 0)
            analyze.assert_not_called()
            manifests = list((root/'cache').rglob('download_manifest.json'))
            self.assertTrue(manifests)
            self.assertTrue(all(json.loads(p.read_text())['ready'] for p in manifests))

    def test_batch_does_not_start_when_later_folder_is_unavailable(self):
        import helix_cli_runner as runner
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root/'inputs'; data.mkdir()
            for group, content in [('a', 'Time,Ampl\n0,1\n'), ('b', '')]:
                (data/group).mkdir(); (data/group/'trace.csv').write_text(content)
            config = json.loads((ROOT/'helix_master_config.json').read_text())
            config['cli_settings'].update(input_dir=str(data), input_files=None,
                output_dir=None, param_folder=None, batch_mode=True,
                input_preparation={'cache_dir': str(root/'cache')})
            cfg = root/'config.json'; cfg.write_text(json.dumps(config))
            with patch.object(sys, 'argv', ['runner', '--config', str(cfg)]), \
                    patch.object(runner, '_run_analysis') as analyze, \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(InputUnavailableError):
                runner.main()
            analyze.assert_not_called()


if __name__ == '__main__':
    unittest.main()
