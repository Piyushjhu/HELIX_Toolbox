"""Cloud availability, reconciliation, and incomplete-run regressions."""
import contextlib
import io
import json
import os
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
from helix_file_io import InputUnavailableError, file_availability, read_parameter_table, require_local_files
from helix_input_audit import audit_inputs
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


if __name__ == '__main__':
    unittest.main()
