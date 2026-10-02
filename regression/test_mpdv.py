"""Source-selection regression tests. Run with QT_QPA_PLATFORM=offscreen."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helix_data_source import (
    MPDV_NOTE, central_summary, load_mpdv_parameters, mpdv_header_lines,
    select_central_files, trace_key, validate_data_mode,
)
from helix_analysis_toolbox import AnalysisThread, HELIXAnalysisToolbox, QApplication, load_config_from_file
from helix_cli_runner import _load_parameter_folder
import helix_cli_runner


CENTRAL = 'C1--SAMPLE_2026-08-12_17-59-49_shot10--00000'
OTHER = CENTRAL.replace('C1--', 'C3--')


class MPDVTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.rows = pd.DataFrame([
            {'Exp_ID': 12, 'Sample_IGSN': 'APLMAL00006-001',
             'PDV_6_FileName': OTHER, 'PDV_10_FileName': CENTRAL,
             'PDV_10_Target_Wavelength (m)': 1.55e-6,
             'PDV_10_Return_Power (dBm)': -22,
             'Laser_Target_Energy (mJ)': 1777, 'Flyer_material': 'Al'},
            {'Exp_ID': 13, 'PDV_10_FileName': None},
        ])
        self.rows.to_csv(self.folder / 'log.csv', index=False)
        self.messages = []

    def load(self):
        return load_mpdv_parameters(self.folder, self.messages.append)

    def thread(self, **kwargs):
        opts = dict(alpss_params={}, spade_params={}, input_files=[],
                    output_dir=str(self.folder), param_data=self.load(), data_mode='mpdv')
        opts.update(kwargs)
        return AnalysisThread(**opts)

    def test_mapping_uses_explicit_probe_and_preserves_row(self):
        data = self.load()
        self.assertEqual(list(data), [CENTRAL])
        self.assertEqual(data[CENTRAL]['Exp_ID'], 12)
        self.assertEqual(data[CENTRAL]['Laser_Target_Energy (mJ)'], 1777)
        self.assertEqual(data[CENTRAL]['PDV_Return_Power (dBm)'], -22)
        self.assertEqual(data[CENTRAL]['PDV_FileName'], CENTRAL)
        self.assertTrue(any('skipped 1 rows' in m for m in self.messages))

    def test_paths_and_derived_suffixes_preserve_shot_identity(self):
        for value in (CENTRAL, CENTRAL + '.csv', 'C:\\scope\\' + CENTRAL + '.csv',
                      '/data/' + CENTRAL + '--vel-smooth-with-uncert.csv'):
            self.assertEqual(trace_key(value), CENTRAL)
        self.assertNotEqual(trace_key(OTHER), CENTRAL)

    def test_blank_values(self):
        for value in (None, float('nan'), '', ' ', 0):
            self.assertEqual(trace_key(value), '')

    def test_selection_filters_other_channels(self):
        files = [OTHER + '.csv', CENTRAL + '.csv']
        self.assertEqual(select_central_files(files, self.load()), [CENTRAL + '.csv'])

    def test_no_central_match_is_error(self):
        with self.assertRaisesRegex(ValueError, 'No input files match'):
            select_central_files([OTHER + '.csv'], self.load())

    def test_duplicate_inputs_are_error(self):
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            select_central_files(['/a/' + CENTRAL + '.csv', '/b/' + CENTRAL + '.csv'], self.load())

    def test_duplicate_metadata_is_error(self):
        self.rows.to_csv(self.folder / 'duplicate.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            self.load()

    def test_invalid_wavelength_is_error(self):
        self.rows.loc[0, 'PDV_10_Target_Wavelength (m)'] = -1
        self.rows.to_csv(self.folder / 'log.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'wavelength'):
            self.load()

    def test_required_parameter_schema(self):
        with self.assertRaises(ValueError):
            load_mpdv_parameters(None)
        (self.folder / 'log.csv').write_text('PDV_FileName\nlegacy\n')
        with self.assertRaisesRegex(ValueError, 'PDV_10_FileName'):
            self.load()

    def test_excel_mapping(self):
        (self.folder / 'log.csv').unlink()
        self.rows.to_excel(self.folder / 'log.xlsx', index=False)
        self.assertEqual(list(self.load()), [CENTRAL])

    def test_summary_excludes_stale_other_probe_outputs(self):
        df = pd.DataFrame({'Filename': [CENTRAL, OTHER], 'value': [1, 2]})
        self.assertEqual(central_summary(df, self.load())['value'].tolist(), [1])
        with self.assertRaises(ValueError):
            central_summary(pd.DataFrame({'value': [1]}), self.load())

    def test_thread_filters_raw_and_manual_velocity_inputs(self):
        thread = self.thread(input_files=[CENTRAL + '.csv', OTHER + '.csv'],
                             spade_auto_mode=False,
                             spade_input_files=[CENTRAL + '--vel-smooth-with-uncert.csv',
                                                OTHER + '--vel-smooth-with-uncert.csv'])
        thread.progress_signal.connect(self.messages.append)
        thread._prepare_data_source()
        self.assertEqual(len(thread.input_files), 1)
        self.assertEqual(len(thread.spade_input_files), 1)
        self.assertIn(MPDV_NOTE, self.messages)
        self.assertEqual(thread._scope_files([OTHER + '.csv', CENTRAL + '.csv']), [CENTRAL + '.csv'])

    def test_spade_only_selection(self):
        thread = self.thread(analysis_mode='spade_only', spade_auto_mode=False,
                             spade_input_files=[CENTRAL + '--vel-smooth-with-uncert.csv',
                                                OTHER + '--vel-smooth-with-uncert.csv'])
        thread._prepare_data_source()
        self.assertEqual(len(thread.spade_input_files), 1)

    def test_mpdv_never_matches_by_exp_or_partial_name(self):
        thread = self.thread()
        self.assertEqual(thread.get_param_data_for_file(OTHER), {})
        self.assertEqual(thread.get_param_data_for_file('APLMAL00006-001'), {})
        self.assertEqual(thread.get_param_data_for_file(CENTRAL)['Exp_ID'], 12)

    def test_igsn_material_precedence_and_child_override(self):
        thread = self.thread(igsn_material_map={'APLMAL00006': 'Ti', 'APLMAL00006-001': 'Ti64'})
        info = thread.get_param_data_for_file(CENTRAL)
        self.assertEqual(thread.resolve_sample_material(CENTRAL, info), 'Ti64')
        thread.igsn_material_map = {}
        self.assertEqual(thread.resolve_sample_material(CENTRAL, info), 'Unknown')

    def test_single_pdv_default_is_passthrough(self):
        thread = AnalysisThread({}, {}, [CENTRAL, OTHER], str(self.folder))
        thread._prepare_data_source()
        self.assertEqual(thread.data_mode, 'single_pdv')
        self.assertEqual(thread.input_files, [CENTRAL, OTHER])
        df = pd.DataFrame({'anything': [1]})
        self.assertIs(thread._scope_summary(df), df)

    def test_legacy_loader_default_equals_explicit_single(self):
        params = Path(__file__).parent / 'data' / 'params'
        with contextlib.redirect_stdout(io.StringIO()):
            default = _load_parameter_folder(str(params))
            explicit = _load_parameter_folder(str(params), data_mode='single_pdv')
        pd.testing.assert_frame_equal(pd.DataFrame(default), pd.DataFrame(explicit))

    def test_mpdv_loader_ignores_experiment_folder_name_filter(self):
        with contextlib.redirect_stdout(io.StringIO()):
            data = _load_parameter_folder(str(self.folder), experiment_id='unrelated_parent', data_mode='mpdv')
        self.assertEqual(list(data), [CENTRAL])

    def test_header_detection(self):
        path = self.folder / 'waveform.csv'
        path.write_text('LECROY,Waveform\nSegments,1\nSegment,TrigTime\n#1,date\nTime,Ampl\n0,1\n')
        self.assertEqual(mpdv_header_lines(path, 22), 4)
        path.write_text('t,v\n0,1\n')
        self.assertEqual(mpdv_header_lines(path, 22), 22)

    def test_invalid_mode_does_not_silently_use_legacy(self):
        with self.assertRaises(ValueError):
            validate_data_mode('multiplex')

    def test_run_snapshot_records_selection(self):
        thread = self.thread(input_files=[CENTRAL + '.csv', OTHER + '.csv'])
        thread._prepare_data_source()
        thread._save_run_config(str(self.folder))
        config = json.loads(next(self.folder.glob('*Run_Config.json')).read_text())
        self.assertEqual(config['selected_probe'], 'PDV_10')
        self.assertEqual(config['mpdv_selection']['raw_files_suppressed'], 1)

    def test_cli_batch_propagates_mode_and_filters_every_subfolder(self):
        parent = self.folder / 'traces'
        records = []
        for sample in ('SAMPLE_A', 'SAMPLE_B'):
            directory = parent / sample
            directory.mkdir(parents=True)
            central = f'C1--{sample}_shot01--00000'
            other = central.replace('C1--', 'C3--')
            for name in (central, other):
                (directory / (name + '.csv')).write_text('Time,Ampl\n0,1\n')
            records.append({'Sample_IGSN': sample, 'PDV_10_FileName': central})
        pd.DataFrame(records).to_csv(self.folder / 'log.csv', index=False)
        cfg = {'cli_settings': {'batch_mode': True, 'data_mode': 'mpdv',
                               'input_dir': str(parent), 'param_folder': str(self.folder),
                               'analysis_mode': 'alpss_only'},
               'alpss_config': {}, 'spade_config': {}}
        path = self.folder / 'batch.json'
        path.write_text(json.dumps(cfg))
        selected = []

        def run(**kwargs):
            self.assertEqual(kwargs['data_mode'], 'mpdv')
            files = select_central_files(kwargs['resolved_input_files'], kwargs['param_data'])
            selected.extend(files)
            return True

        with patch.object(sys, 'argv', ['helix_cli_runner.py', '--config', str(path)]), \
                patch.object(helix_cli_runner, '_run_analysis', side_effect=run), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
            helix_cli_runner.main()
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(Path(f).name.startswith('C1--') for f in selected))

    def test_sensitivity_excludes_other_probes_before_sweep(self):
        from argparse import Namespace
        from helix_sensitivity_analysis import resolve_input_files
        paths = [self.folder / (stem + '.csv') for stem in (CENTRAL, OTHER)]
        for path in paths:
            path.touch()
        # Keep raw waveforms outside the parameter directory.
        params = self.folder / 'params'
        params.mkdir()
        (self.folder / 'log.csv').rename(params / 'log.csv')
        args = Namespace(input_files=list(map(str, paths)), input_dir=None, input_file=None)
        cfg = {'cli_settings': {'data_mode': 'mpdv', 'param_folder': str(params)}}
        with contextlib.redirect_stdout(io.StringIO()):
            traces = resolve_input_files(args, cfg)
        self.assertEqual([path for _, path in traces], [str(paths[0])])

    def test_velocity_summary_with_hel_disabled(self):
        fixture = Path(__file__).parent / 'data' / 'C1--20251022--00001--vel-smooth-with-uncert.csv'
        thread = AnalysisThread({}, {'hel_detection_enabled': False,
                                    'experiment_hel_detection': False}, [], str(self.folder),
                                analysis_mode='spade_only', spade_input_files=[str(fixture)])
        with contextlib.ExitStack() as stack:
            for name in dir(AnalysisThread):
                if name.startswith('generate_') and name != 'generate_velocity_shots_summary':
                    stack.enter_context(patch.object(thread, name))
            thread.generate_velocity_shots_summary(str(self.folder))
        summary = pd.read_csv(self.folder / 'velocity_shots_summary.csv')
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]['sample_material'], 'Unknown')


class GUISelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_gui_source_default_master_load_and_persistence(self):
        with patch.object(HELIXAnalysisToolbox, 'load_settings'):
            gui = HELIXAnalysisToolbox()
        with tempfile.TemporaryDirectory() as folder:
            gui.config_file = str(Path(folder) / 'settings.json')
            self.assertEqual(gui.data_mode_combo.currentData(), 'single_pdv')
            config = {'cli_settings': {'data_mode': 'mpdv', 'input_files': [], 'analysis_mode': 'both'},
                      'alpss_config': {}, 'spade_config': {}, 'igsn_material_map': {'APLMAL00006': 'Ti'}}
            gui.apply_master_config(config, '/tmp/test-master.yml')
            self.assertEqual(gui.data_mode_combo.currentData(), 'mpdv')
            self.assertEqual(gui._master_material_settings['igsn_material_map'], {'APLMAL00006': 'Ti'})
            gui.save_settings()
            gui.data_mode_combo.setCurrentIndex(0)
            gui.load_settings()
            self.assertEqual(gui.data_mode_combo.currentData(), 'mpdv')
            gui.close()

    def test_gui_loads_full_master_json_and_yaml(self):
        root = Path(__file__).resolve().parents[1]
        with patch.object(HELIXAnalysisToolbox, 'load_settings'):
            gui = HELIXAnalysisToolbox()
        with tempfile.TemporaryDirectory() as folder:
            gui.config_file = str(Path(folder) / 'settings.json')
            for name in ('helix_master_config.json', 'helix_master_config.yml'):
                ok, config, _ = load_config_from_file(root / name)
                self.assertTrue(ok)
                self.assertIn(config['cli_settings']['data_mode'], ('single_pdv', 'mpdv'))
                config['cli_settings'].update(data_mode='mpdv', input_dir=folder,
                                               input_files=None, param_folder=None,
                                               output_dir=folder)
                gui.apply_master_config(config, str(root / name))
                self.assertEqual(gui.data_mode_combo.currentData(), 'mpdv')
                self.assertEqual(gui.multi_file_path.text(), folder)
                self.assertFalse(gui.single_file_radio.isChecked())
                self.assertTrue(gui.multi_file_radio.isChecked())
                self.assertEqual(gui._master_material_settings['material_properties'], config['material_properties'])
            gui.close()


if __name__ == '__main__':
    unittest.main()
