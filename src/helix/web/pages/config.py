"""Configuration page — analysis mode, ALPSS params, SPADE params, material properties."""
from __future__ import annotations

from typing import TYPE_CHECKING

import panel as pn

from helix.materials.database import list_available_materials

if TYPE_CHECKING:
    from helix.web.app import HelixApp


# Default ALPSS params matching the monolith's defaults
DEFAULT_ALPSS = {
    "start_time_user": "otsu",
    "carrier_filter_type": "gaussian_notch",
    "spall_calculation": "yes",
    "window": "hann",
    "header_lines": 5,
    "time_to_skip": 0.0,
    "time_to_take": 10e-6,
    "t_before": 5e-9,
    "t_after": 5e-8,
    "start_time_correction": 0.0,
    "freq_min": 1.5e9,
    "freq_max": 4.0e9,
    "sample_rate": 80e9,
    "nperseg": 512,
    "noverlap": 435,
    "nfft": 5120,
    "smoothing_window": 601,
    "smoothing_wid": 3,
    "smoothing_amp": 1,
    "smoothing_sigma": 1,
    "smoothing_mu": 0,
    "pb_neighbors": 400,
    "pb_idx_correction": 0,
    "rc_neighbors": 400,
    "rc_idx_correction": 0,
    "blur_kernel_x": 5,
    "blur_kernel_y": 5,
    "blur_sigx": 0,
    "blur_sigy": 0,
    "carrier_band_time": 2.5e-7,
    "uncert_mult": 100,
    "order": 6,
    "wid": 50e6,
    "lam": 1.547461e-6,
    "iq_threshold_factor": 0.4,
    "cusum_offset": 5,
    "cusum_threshold": 1000,
    "t_fit_begin": 20,
    "t_fit_end": 300,
    "cmap": "viridis",
    "plot_figsize_x": 80,
    "plot_figsize_y": 40,
    "plot_dpi": 300,
    "display_plots": "no",
    "save_data": "yes",
}

DEFAULT_SPADE = {
    "spall_analysis_enabled": True,
    "threshold": 10.0,
    "min_pullback_pct": 0.05,
    "max_recomp_pct": 0.8,
    "dns_threshold": 0.02,
}

DEFAULT_MATERIAL = {
    "C0": 4540,
    "density": 1730,
    "delta_rho": 9,
    "delta_C0": 23,
    "delta_lam": 8e-18,
    "theta": 0,
    "delta_theta": 5,
}


class ConfigPage:
    def __init__(self, app: HelixApp):
        self.app = app

        # --- Analysis mode ---
        self.analysis_mode = pn.widgets.RadioButtonGroup(
            name="Analysis Mode",
            options=["ALPSS + SPADE", "ALPSS Only", "SPADE Only"],
            value="ALPSS + SPADE",
        )

        # --- Config file loader ---
        self.config_json = pn.widgets.TextInput(
            name="Load Config JSON (optional)",
            placeholder="/path/to/config.json",
            width=500,
        )
        self.load_config_btn = pn.widgets.Button(
            name="Load", button_type="default", width=100,
        )
        self.load_config_btn.on_click(self._on_load_config)

        # --- Material ---
        materials_list = list_available_materials()
        self.material_select = pn.widgets.AutocompleteInput(
            name="Material",
            options=materials_list,
            value="Ti-6Al-4V" if "Ti-6Al-4V" in materials_list else materials_list[0] if materials_list else "",
            placeholder="Type material name...",
            width=300,
        )
        self.material_select.param.watch(self._on_material_change, "value")

        self.C0 = pn.widgets.FloatInput(name="C0 (m/s)", value=DEFAULT_MATERIAL["C0"], step=1, width=150)
        self.density = pn.widgets.FloatInput(name="Density (kg/m3)", value=DEFAULT_MATERIAL["density"], step=1, width=150)
        self.delta_rho = pn.widgets.FloatInput(name="delta_rho", value=DEFAULT_MATERIAL["delta_rho"], step=0.1, width=150)
        self.delta_C0 = pn.widgets.FloatInput(name="delta_C0", value=DEFAULT_MATERIAL["delta_C0"], step=0.1, width=150)
        self.lam = pn.widgets.FloatInput(name="Wavelength (m)", value=DEFAULT_ALPSS["lam"], step=1e-9, format="0.000000e+0", width=200)
        self.delta_lam = pn.widgets.FloatInput(name="delta_lam", value=DEFAULT_MATERIAL["delta_lam"], step=1e-18, format="0.0e+0", width=200)
        self.theta = pn.widgets.FloatInput(name="theta (deg)", value=DEFAULT_MATERIAL["theta"], step=0.1, width=150)
        self.delta_theta = pn.widgets.FloatInput(name="delta_theta (deg)", value=DEFAULT_MATERIAL["delta_theta"], step=0.1, width=150)

        # --- Basic ALPSS params ---
        self.start_time_user = pn.widgets.TextInput(name="Start Time User", value="otsu", width=200)
        self.carrier_filter_type = pn.widgets.Select(
            name="Carrier Filter", options=["gaussian_notch", "none"], value="gaussian_notch", width=200,
        )
        self.spall_calculation = pn.widgets.Select(
            name="Spall Calculation", options=["yes", "no"], value="yes", width=150,
        )
        self.header_lines = pn.widgets.IntInput(name="Header Lines", value=5, step=1, start=0, width=100)

        # --- Time params ---
        self.time_to_skip = pn.widgets.FloatInput(name="Time to Skip (s)", value=0.0, step=1e-7, format="0.00e+0", width=200)
        self.time_to_take = pn.widgets.FloatInput(name="Time to Take (s)", value=10e-6, step=1e-7, format="0.00e+0", width=200)
        self.t_before = pn.widgets.FloatInput(name="t_before (s)", value=5e-9, step=1e-9, format="0.0e+0", width=200)
        self.t_after = pn.widgets.FloatInput(name="t_after (s)", value=5e-8, step=1e-9, format="0.0e+0", width=200)
        self.start_time_correction = pn.widgets.FloatInput(name="Start Time Correction (s)", value=0.0, step=1e-9, format="0.0e+0", width=200)

        # --- Frequency params ---
        self.freq_min = pn.widgets.FloatInput(name="Freq Min (Hz)", value=1.5e9, step=1e8, format="0.00e+0", width=200)
        self.freq_max = pn.widgets.FloatInput(name="Freq Max (Hz)", value=4.0e9, step=1e8, format="0.00e+0", width=200)
        self.sample_rate = pn.widgets.FloatInput(name="Sample Rate (Hz)", value=80e9, step=1e9, format="0.0e+0", width=200)

        # --- Spectrogram params ---
        self.nperseg = pn.widgets.IntInput(name="nperseg", value=512, step=1, width=120)
        self.noverlap = pn.widgets.IntInput(name="noverlap", value=435, step=1, width=120)
        self.nfft = pn.widgets.IntInput(name="nfft", value=5120, step=1, width=120)
        self.window = pn.widgets.Select(name="Window", options=["hann", "hamming", "blackman", "boxcar"], value="hann", width=150)

        # --- Smoothing params ---
        self.smoothing_window = pn.widgets.IntInput(name="Smoothing Window", value=601, step=2, width=150)
        self.smoothing_wid = pn.widgets.IntInput(name="Smoothing Width", value=3, step=1, width=120)

        # --- Spall detection ---
        self.pb_neighbors = pn.widgets.IntInput(name="Pullback Neighbors", value=400, step=10, width=150)
        self.pb_idx_correction = pn.widgets.IntInput(name="PB Index Correction", value=0, step=1, width=150)
        self.rc_neighbors = pn.widgets.IntInput(name="Recomp Neighbors", value=400, step=10, width=150)
        self.rc_idx_correction = pn.widgets.IntInput(name="RC Index Correction", value=0, step=1, width=150)

        # --- SPADE params ---
        self.spade_enabled = pn.widgets.Checkbox(name="Enable SPADE Analysis", value=True)
        self.dns_threshold = pn.widgets.FloatInput(name="DNS Threshold", value=0.02, step=0.01, width=150)

        # --- HEL params ---
        self.hel_enabled = pn.widgets.Checkbox(name="Enable HEL Detection", value=False)
        self.hel_threshold = pn.widgets.FloatInput(name="HEL Threshold (m/s)", value=10.0, step=1, width=150)

        self.config_status = pn.pane.Alert("", alert_type="info", visible=False)

    def _on_material_change(self, event):
        from helix.materials.database import get_material_properties
        props = get_material_properties(event.new)
        if props:
            self.C0.value = props.get("acoustic_velocity", self.C0.value)
            self.density.value = props.get("density", self.density.value)

    def _on_load_config(self, event):
        import json
        path = self.config_json.value.strip()
        if not path:
            return
        try:
            with open(path) as f:
                cfg = json.load(f)
            # Populate ALPSS params from config
            alpss = cfg.get("alpss_config", cfg)
            for key, widget in self._alpss_widget_map().items():
                if key in alpss:
                    widget.value = alpss[key]
            self.config_status.object = f"Loaded config from {path}"
            self.config_status.alert_type = "success"
            self.config_status.visible = True
        except Exception as e:
            self.config_status.object = f"Error loading config: {e}"
            self.config_status.alert_type = "danger"
            self.config_status.visible = True

    def _alpss_widget_map(self):
        """Map param names to widgets for bulk load/save."""
        return {
            "start_time_user": self.start_time_user,
            "carrier_filter_type": self.carrier_filter_type,
            "spall_calculation": self.spall_calculation,
            "header_lines": self.header_lines,
            "time_to_skip": self.time_to_skip,
            "time_to_take": self.time_to_take,
            "t_before": self.t_before,
            "t_after": self.t_after,
            "start_time_correction": self.start_time_correction,
            "freq_min": self.freq_min,
            "freq_max": self.freq_max,
            "sample_rate": self.sample_rate,
            "nperseg": self.nperseg,
            "noverlap": self.noverlap,
            "nfft": self.nfft,
            "window": self.window,
            "smoothing_window": self.smoothing_window,
            "smoothing_wid": self.smoothing_wid,
            "pb_neighbors": self.pb_neighbors,
            "pb_idx_correction": self.pb_idx_correction,
            "rc_neighbors": self.rc_neighbors,
            "rc_idx_correction": self.rc_idx_correction,
        }

    def get_alpss_params(self) -> dict:
        """Collect current ALPSS parameter values into a dict."""
        params = dict(DEFAULT_ALPSS)  # start with defaults
        for key, widget in self._alpss_widget_map().items():
            params[key] = widget.value

        # Material properties
        params["C0"] = self.C0.value
        params["density"] = self.density.value
        params["delta_rho"] = self.delta_rho.value
        params["delta_C0"] = self.delta_C0.value
        params["lam"] = self.lam.value
        params["delta_lam"] = self.delta_lam.value
        params["theta"] = self.theta.value
        params["delta_theta"] = self.delta_theta.value

        # Fix composite types
        params["blur_kernel"] = (DEFAULT_ALPSS["blur_kernel_x"], DEFAULT_ALPSS["blur_kernel_y"])
        params.pop("blur_kernel_x", None)
        params.pop("blur_kernel_y", None)
        params["plot_figsize"] = (DEFAULT_ALPSS["plot_figsize_x"], DEFAULT_ALPSS["plot_figsize_y"])
        params.pop("plot_figsize_x", None)
        params.pop("plot_figsize_y", None)

        # Parse start_time_user — could be a float
        stu = params["start_time_user"]
        if stu not in ("otsu", "iq", "none"):
            try:
                params["start_time_user"] = float(stu)
            except ValueError:
                pass

        # HEL params
        if self.hel_enabled.value:
            params["hel_enabled"] = True
            params["hel_threshold"] = self.hel_threshold.value

        return params

    def get_spade_params(self) -> dict:
        """Collect current SPADE parameter values."""
        return {
            "spall_analysis_enabled": self.spade_enabled.value,
            "dns_threshold": self.dns_threshold.value,
            "density": self.density.value,
            "acoustic_velocity": self.C0.value,
        }

    def get_analysis_mode(self) -> str:
        mode_map = {
            "ALPSS + SPADE": "both",
            "ALPSS Only": "alpss_only",
            "SPADE Only": "spade_only",
        }
        return mode_map.get(self.analysis_mode.value, "both")

    def panel(self) -> pn.Column:
        mode_section = pn.Column(
            pn.pane.Markdown("# Configuration"),
            pn.pane.Markdown("### Analysis Mode"),
            self.analysis_mode,
        )

        config_loader = pn.Column(
            pn.pane.Markdown("### Load Config"),
            pn.Row(self.config_json, self.load_config_btn),
            self.config_status,
        )

        material_section = pn.Card(
            pn.Row(self.material_select),
            pn.Row(self.C0, self.density),
            pn.Row(self.delta_rho, self.delta_C0),
            pn.Row(self.lam, self.delta_lam),
            pn.Row(self.theta, self.delta_theta),
            title="Material Properties",
            collapsed=False,
        )

        basic_section = pn.Card(
            pn.Row(self.start_time_user, self.carrier_filter_type),
            pn.Row(self.spall_calculation, self.header_lines),
            title="Basic Parameters",
            collapsed=False,
        )

        time_section = pn.Card(
            pn.Row(self.time_to_skip, self.time_to_take),
            pn.Row(self.t_before, self.t_after),
            self.start_time_correction,
            title="Time Parameters",
            collapsed=True,
        )

        freq_section = pn.Card(
            pn.Row(self.freq_min, self.freq_max),
            self.sample_rate,
            title="Frequency Parameters",
            collapsed=True,
        )

        spect_section = pn.Card(
            pn.Row(self.nperseg, self.noverlap, self.nfft),
            self.window,
            title="Spectrogram Parameters",
            collapsed=True,
        )

        smooth_section = pn.Card(
            pn.Row(self.smoothing_window, self.smoothing_wid),
            title="Smoothing Parameters",
            collapsed=True,
        )

        spall_section = pn.Card(
            pn.Row(self.pb_neighbors, self.pb_idx_correction),
            pn.Row(self.rc_neighbors, self.rc_idx_correction),
            title="Spall Detection Parameters",
            collapsed=True,
        )

        spade_section = pn.Card(
            self.spade_enabled,
            self.dns_threshold,
            title="SPADE Parameters",
            collapsed=True,
        )

        hel_section = pn.Card(
            self.hel_enabled,
            self.hel_threshold,
            title="HEL Detection",
            collapsed=True,
        )

        return pn.Column(
            mode_section,
            pn.layout.Divider(),
            config_loader,
            pn.layout.Divider(),
            material_section,
            basic_section,
            time_section,
            freq_section,
            spect_section,
            smooth_section,
            spall_section,
            spade_section,
            hel_section,
            sizing_mode="stretch_width",
        )
