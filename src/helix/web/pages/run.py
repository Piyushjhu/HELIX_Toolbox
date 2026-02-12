"""Run page — execute the HELIX pipeline with progress feedback."""
from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

import panel as pn

if TYPE_CHECKING:
    from helix.web.app import HelixApp


class RunPage:
    def __init__(self, app: HelixApp):
        self.app = app

        self.run_btn = pn.widgets.Button(
            name="Run Pipeline", button_type="success", width=200,
        )
        self.run_btn.on_click(self._on_run)

        self.stop_btn = pn.widgets.Button(
            name="Stop", button_type="danger", width=100, visible=False,
        )

        self.progress = pn.indicators.Progress(
            name="Pipeline Progress",
            active=False,
            bar_color="primary",
            width=600,
        )

        self.log = pn.widgets.TextAreaInput(
            name="Pipeline Log",
            value="",
            height=400,
            width=700,
            disabled=True,
        )

        self.status = pn.pane.Alert("Ready to run.", alert_type="info")
        self._running = False

    def _append_log(self, msg: str):
        self.log.value += msg + "\n"

    def _on_run(self, event):
        if self._running:
            return

        # Validate
        if not self.app.input_files and self.app.analysis_mode != "spade_only":
            self.status.object = "No input files selected. Go to the Files tab."
            self.status.alert_type = "danger"
            return

        self._running = True
        self.run_btn.disabled = True
        self.stop_btn.visible = True
        self.progress.active = True
        self.log.value = ""
        self.status.object = "Running pipeline..."
        self.status.alert_type = "warning"

        # Collect params from config page
        alpss_params = self.app.config_page.get_alpss_params()
        spade_params = self.app.config_page.get_spade_params()
        analysis_mode = self.app.config_page.get_analysis_mode()
        output_dir = self.app.output_dir

        # Store on app
        self.app.alpss_params = alpss_params
        self.app.spade_params = spade_params
        self.app.analysis_mode = analysis_mode

        os.makedirs(output_dir, exist_ok=True)

        def _run_in_thread():
            try:
                from helix.pipeline.runner import run_pipeline

                result = run_pipeline(
                    input_files=self.app.input_files,
                    output_dir=output_dir,
                    alpss_params=alpss_params,
                    spade_params=spade_params,
                    analysis_mode=analysis_mode,
                    progress_callback=self._append_log,
                )
                self.app.pipeline_result = result
                n_ok = len(result.get("successful_files", []))
                n_fail = len(result.get("failed_files", []))
                self.status.object = f"Pipeline complete. {n_ok} succeeded, {n_fail} failed."
                self.status.alert_type = "success" if n_fail == 0 else "warning"
            except Exception as e:
                self.status.object = f"Pipeline error: {e}"
                self.status.alert_type = "danger"
                self._append_log(f"ERROR: {e}")
            finally:
                self._running = False
                self.run_btn.disabled = False
                self.stop_btn.visible = False
                self.progress.active = False

        thread = threading.Thread(target=_run_in_thread, daemon=True)
        thread.start()

    def panel(self) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown("# Run Pipeline"),
            self.status,
            pn.Row(self.run_btn, self.stop_btn),
            self.progress,
            pn.layout.Divider(),
            pn.pane.Markdown("### Log"),
            self.log,
            sizing_mode="stretch_width",
        )
