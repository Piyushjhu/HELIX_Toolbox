"""Mixed single-PDV / MPDV runner selection regressions."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import helix_cli_runner as runner
from helix_analysis_toolbox import AnalysisThread

CENTRAL = 'C1--JHAMAC00003-S4R4C2_68efdf9ebe3476695206a1c0_1_2046_2026-08-26_13-04-54_shot01--00000'
UNLISTED = CENTRAL.replace('C1--', 'C2--')
OTHER = CENTRAL.replace('C1--', 'C3--')
LEGACY = 'C2--20251022--00001'


class MixedPDVTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.params = self.root / 'params'
        self.params.mkdir()
        # Deliberately put a noncentral probe first.
        self.rows = pd.DataFrame([{'PDV_6_FileName': OTHER, 'PDV_10_FileName': CENTRAL,
                                   'PDV_10_Target_Wavelength (m)': 1.55e-6,
                                   'PDV_10_Return_Power (dBm)': -38.59,
                                   'Sample_IGSN': 'JHAMAC00003-S4R4C2', 'Exp_ID': 1}])
        self.write_mpdv()
        pd.DataFrame([{'PDV_FileName': LEGACY, 'Sample material': 'Ti'}]).to_csv(
            self.params / 'legacy.csv', index=False)

    def write_mpdv(self):
        self.rows.to_csv(self.params / 'mpdv.csv', index=False)

    def load(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner._load_parameter_folder(str(self.params), **kwargs)

    def thread(self, **kwargs):
        options = dict(alpss_params={}, spade_params={}, input_files=[],
                       output_dir=str(self.root), param_data=self.load())
        options.update(kwargs)
        return AnalysisThread(**options)

    def test_mixed_loader_adapts_probe_ten_and_preserves_legacy(self):
        data = self.load()
        self.assertEqual(set(data), {CENTRAL, LEGACY})
        self.assertEqual(data[CENTRAL]['PDV_FileName'], CENTRAL)
        self.assertEqual(data[CENTRAL]['PDV_Return_Power (dBm)'], -38.59)
        self.assertEqual(data[LEGACY]['Sample material'], 'Ti')

    def test_combined_export_shared_probe_filename_uses_probe_ten(self):
        shared = 'SAMPLE_2026-09-30_15-55-37_shot25'
        self.rows['PDV_6_FileName'] = shared
        self.rows['PDV_10_FileName'] = shared
        self.rows['PDV_6_Target_Wavelength (m)'] = 1.53e-6
        self.write_mpdv()
        data = self.load()
        self.assertEqual(data.select([shared + '.csv'], lambda _: None), [shared + '.csv'])
        self.assertEqual(data[shared]['PDV_Target_Wavelength (m)'], 1.55e-6)
        self.assertEqual(data[shared]['selected_probe'], 'PDV_10')

    def test_missing_spall_column_skips_plot_without_exception(self):
        thread = self.thread()
        messages = []
        thread.progress_signal.connect(messages.append)
        thread.generate_spall_vs_shock_stress_plot(
            pd.DataFrame({'Filename': ['shot'], 'Spall_OK': [False]}), str(self.root))
        self.assertTrue(any('Spall Strength column not found' in m for m in messages))
        self.assertFalse(any('Traceback' in m for m in messages))

    def test_filters_raw_manual_and_existing_outputs(self):
        files = [s + '.csv' for s in (CENTRAL, UNLISTED, OTHER, LEGACY)]
        velocity = [s + '--vel-smooth-with-uncert.csv' for s in (CENTRAL, UNLISTED, OTHER, LEGACY)]
        thread = self.thread(input_files=files, spade_auto_mode=False, spade_input_files=velocity)
        thread._prepare_data_source()
        self.assertEqual(thread.input_files, [files[0], files[3]])
        self.assertEqual(thread.spade_input_files, [velocity[0], velocity[3]])
        self.assertEqual(thread._scope_files(velocity), [velocity[0], velocity[3]])
        for column in ('file_name', 'Filename', 'filename', 'PDV_FileName'):
            df = pd.DataFrame({column: velocity})
            self.assertEqual(thread._scope_summary(df)[column].tolist(), [velocity[0], velocity[3]])
        # Post-processing can filter directly, without preparing raw inputs.
        self.assertEqual(self.thread()._scope_files(velocity), [velocity[0], velocity[3]])

    def test_spade_only_and_snapshot(self):
        thread = self.thread(analysis_mode='spade_only', spade_auto_mode=False,
                             spade_input_files=[CENTRAL, UNLISTED, LEGACY])
        thread._prepare_data_source()
        self.assertEqual(thread.spade_input_files, [CENTRAL, LEGACY])
        thread._save_run_config(str(self.root))
        config = json.loads(next(self.root.glob('*Run_Config.json')).read_text())
        self.assertEqual(config['data_mode'], 'single_pdv')
        self.assertEqual(config['mpdv_selection']['velocity_files_suppressed'], 1)

    def test_central_channel_is_not_hardcoded(self):
        self.rows['PDV_10_FileName'] = UNLISTED
        self.write_mpdv()
        self.assertEqual(self.load().select([CENTRAL, UNLISTED, OTHER, LEGACY], lambda _: None),
                         [UNLISTED, LEGACY])

    def test_missing_central_stops_instead_of_using_sibling(self):
        for inputs in ([UNLISTED, OTHER, LEGACY], [UNLISTED]):
            with self.assertRaisesRegex(ValueError, 'Missing PDV_10 input'):
                self.load().select(inputs, lambda _: None)

    def test_blank_central_keeps_peripheral_membership(self):
        self.rows['PDV_10_FileName'] = None
        self.write_mpdv()
        data = self.load()
        self.assertEqual(set(data), {LEGACY})
        self.assertEqual(data.select([CENTRAL, UNLISTED, OTHER, LEGACY], lambda _: None), [LEGACY])
        (self.params / 'legacy.csv').unlink()
        # Empty dict still carries the exclusion index.
        thread = self.thread(input_files=[OTHER])
        with self.assertRaisesRegex(ValueError, 'No eligible inputs'):
            thread._prepare_data_source()

    def test_missing_central_column_rejected(self):
        self.rows.drop(columns=['PDV_10_FileName']).to_csv(self.params / 'mpdv.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'no PDV_10_FileName'):
            self.load()

    def test_duplicate_metadata_and_inputs_rejected(self):
        data = self.load()
        with self.assertRaisesRegex(ValueError, 'Ambiguous PDV_10 input'):
            data.select(['/a/' + CENTRAL + '.csv', '/b/' + CENTRAL + '.csv'])
        self.rows.to_csv(self.params / 'duplicate.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'Ambiguous PDV_10 filename'):
            self.load()

    def test_invalid_wavelength_rejected(self):
        self.rows['PDV_10_Target_Wavelength (m)'] = -1
        self.write_mpdv()
        with self.assertRaisesRegex(ValueError, 'wavelength'):
            self.load()

    def test_metadata_only_matches_exact_mpdv_trace(self):
        thread = self.thread()
        self.assertEqual(thread.get_param_data_for_file('/a/' + CENTRAL + '--velocity.csv')['Exp_ID'], 1)
        self.assertEqual(thread.get_param_data_for_file(UNLISTED), {})
        self.assertEqual(thread.get_param_data_for_file('JHAMAC00003-S4R4C2'), {})
        self.assertEqual(thread.get_param_data_for_file(LEGACY)['Sample material'], 'Ti')

    def test_only_channel_token_is_ignored_for_membership(self):
        data = self.load()
        for unrelated in (UNLISTED.replace('shot01', 'shot02'),
                          UNLISTED.replace('13-04-54', '13-04-55'),
                          UNLISTED.replace('S4R4C2', 'S4R4C3')):
            self.assertFalse(data.is_mpdv(unrelated))
            self.assertTrue(data.allows(unrelated))

    def test_excel_paths_extensions_and_spaces(self):
        (self.params / 'mpdv.csv').unlink()
        self.rows['PDV_10_FileName'] = 'C:\\scope\\' + CENTRAL + '.csv'
        self.rows.rename(columns={'PDV_10_FileName': ' PDV_10_FileName '}).to_excel(
            self.params / 'mpdv.xlsx', index=False)
        self.assertEqual(self.load().select([CENTRAL + '.csv', UNLISTED, LEGACY], lambda _: None),
                         [CENTRAL + '.csv', LEGACY])

    def test_legacy_filename_filter_does_not_hide_mpdv_log(self):
        data = self.load(experiment_id='legacy')
        self.assertEqual(set(data), {CENTRAL, LEGACY})
        self.assertFalse(data.allows(UNLISTED))

    def test_per_file_wavelength_and_header(self):
        path = self.root / (CENTRAL + '.csv')
        path.write_text('LECROY,Waveform\nSegments,1\nSegment,TrigTime\n#1,date\nTime,Ampl\n0,1\n')
        thread = self.thread()
        settings = {'lam': 9e-6, 'header_lines': 22}
        thread._apply_mpdv_input_settings(str(path), settings)
        self.assertEqual(settings, {'lam': 1.55e-6, 'header_lines': 4})
        ordinary = {'lam': 9e-6, 'header_lines': 22}
        thread._apply_mpdv_input_settings(LEGACY, ordinary)
        self.assertEqual(ordinary, {'lam': 9e-6, 'header_lines': 22})

    def test_cli_single_and_batch_propagate_selection(self):
        parent = self.root / 'inputs'
        for group, stems in [('mixed', (CENTRAL, UNLISTED, OTHER, LEGACY)),
                             ('ordinary', ('C1--20251022--00002',))]:
            folder = parent / group
            folder.mkdir(parents=True)
            for stem in stems:
                (folder / (stem + '.csv')).write_text('Time,Ampl\n0,1\n')
        for batch in (False, True):
            cfg = {'cli_settings': {'batch_mode': batch, 'data_mode': 'single_pdv',
                   'input_dir': str(parent if batch else parent / 'mixed'),
                   'output_dir': str(self.root / 'output'), 'param_folder': str(self.params),
                   'analysis_mode': 'alpss_only'},
                   'alpss_config': json.loads((Path(runner.__file__).parent / 'helix_master_config.json').read_text())['alpss_config'],
                   'spade_config': {}}
            path = self.root / 'config.json'
            path.write_text(json.dumps(cfg))
            selected = []

            def run(thread):
                thread._prepare_data_source()
                selected.extend(Path(f).stem for f in thread.input_files)
                thread.finished_signal.emit(True, 'Selection verified')

            with patch.object(sys, 'argv', ['helix_cli_runner.py', '--config', str(path)]), \
                    patch.object(AnalysisThread, 'run', run), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                runner.main()
            self.assertEqual(result.exception.code, 0)
            expected = {CENTRAL, LEGACY} | ({'C1--20251022--00002'} if batch else set())
            self.assertEqual(set(selected), expected)


if __name__ == '__main__':
    unittest.main()
