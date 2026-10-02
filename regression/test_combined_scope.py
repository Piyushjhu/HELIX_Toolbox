"""Synthetic combined scope inputs; no experimental data is stored here."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ALPSS'))
from alpss_main import read_scope_trace, detect_sample_rate


class CombinedScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'shot.csv'

    def test_selects_named_channel_and_preserves_first_sample(self):
        self.path.write_text(
            ',Channel 2,Channel 1,Channel 3\nPoints:,3,3,3\n'
            'Time Tags (Channel 1),Channel 2,Channel 1,Channel 3\n'
            '0,90,1,80\n1e-9,91,2,81\n2e-9,92,3,82\n')
        data = read_scope_trace(self.path, 22)
        np.testing.assert_array_equal(data.Ampl, [1, 2, 3])
        np.testing.assert_allclose(data.Time, [0, 1e-9, 2e-9])
        data = read_scope_trace(self.path, 22, sample_offset=1, nrows=1)
        np.testing.assert_array_equal(data.Ampl, [2])
        np.testing.assert_allclose(detect_sample_rate(
            exp_data_dir=self.tmp.name, filename='shot.csv', header_lines=22), 1e9)

    def test_missing_channel_one_fails(self):
        self.path.write_text('Time Tags (Channel 2),Channel 2,Channel 3\n0,2,3\n')
        with self.assertRaisesRegex(ValueError, 'Channel 1'):
            read_scope_trace(self.path, 22)

    def test_legacy_header_and_offset_are_preserved(self):
        self.path.write_text('scope metadata\nTime,Ampl\n0,1\n1e-9,2\n2e-9,3\n')
        np.testing.assert_array_equal(read_scope_trace(self.path, 1).Ampl, [1, 2, 3])
        np.testing.assert_array_equal(
            read_scope_trace(self.path, 1, sample_offset=1, nrows=1).Ampl, [2])


if __name__ == '__main__':
    unittest.main()
