"""Tests for helix.config."""
import json
import pytest
from helix.config.loader import load_config, save_config, load_master_config
from helix.config.transform import transform_alpss_params, transform_spade_params


@pytest.fixture
def tmp_config(tmp_path):
    cfg = {"foo": 1, "bar": "baz"}
    path = tmp_path / "test.json"
    path.write_text(json.dumps(cfg))
    return path, cfg


def test_load_config(tmp_config):
    path, expected = tmp_config
    assert load_config(path) == expected


def test_load_config_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.json")


def test_save_config(tmp_path):
    dest = tmp_path / "out.json"
    save_config({"a": 1}, dest)
    assert json.loads(dest.read_text()) == {"a": 1}


def test_load_master_config(tmp_path):
    master = {
        "cli_settings": {},
        "alpss_config": {"lam": 1.5e-6},
        "spade_config": {"density": 8960},
    }
    path = tmp_path / "master.json"
    path.write_text(json.dumps(master))
    loaded = load_master_config(path)
    assert "alpss_config" in loaded


def test_load_master_config_missing_section(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"cli_settings": {}}))
    with pytest.raises(ValueError, match="alpss_config"):
        load_master_config(path)


class TestTransformAlpss:
    def test_blur_kernel_from_xy(self):
        p = transform_alpss_params({"blur_kernel_x": 7, "blur_kernel_y": 9})
        assert p["blur_kernel"] == (7, 9)

    def test_defaults_filled(self):
        p = transform_alpss_params({})
        assert p["plot_figsize"] == (30, 10)
        assert p["window"] == "hann"
        assert p["display_plots"] == "no"

    def test_int_coercion(self):
        p = transform_alpss_params({"header_lines": 5.0, "nperseg": "512"})
        assert isinstance(p["header_lines"], int)
        assert p["nperseg"] == 512


class TestTransformSpade:
    def test_experiment_toggles(self):
        p = transform_spade_params({"experiment_velocity_shots": True, "experiment_hel_detection": "true"})
        assert p["velocity_shots_enabled"] is True
        assert p["hel_detection_enabled"] is True

    def test_signal_length_full(self):
        p = transform_spade_params({"signal_length": "Full Signal (None)"})
        assert p["signal_length_ns"] is None

    def test_signal_length_numeric(self):
        p = transform_spade_params({"signal_length": "500"})
        assert p["signal_length_ns"] == 500.0
