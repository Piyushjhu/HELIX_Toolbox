"""Connection page — authenticate with Girder or work locally."""
from __future__ import annotations

from typing import TYPE_CHECKING

import panel as pn

if TYPE_CHECKING:
    from helix.web.app import HelixApp


class ConnectionPage:
    def __init__(self, app: HelixApp):
        self.app = app

        # Widgets
        self.mode = pn.widgets.RadioButtonGroup(
            name="Data Source",
            options=["Girder (HTMDEC)", "Local Files"],
            value="Girder (HTMDEC)",
        )

        self.api_url = pn.widgets.TextInput(
            name="Girder API URL",
            value="https://data.htmdec.org/api/v1",
            width=500,
        )
        self.api_key = pn.widgets.PasswordInput(
            name="API Key",
            placeholder="Enter your HTMDEC API key...",
            width=500,
        )
        self.connect_btn = pn.widgets.Button(
            name="Connect", button_type="primary", width=200,
        )
        self.connect_btn.on_click(self._on_connect)

        self.status = pn.pane.Alert(
            "Enter your Girder credentials to connect, or switch to Local Files mode.",
            alert_type="info",
        )

        # React to mode changes
        self.mode.param.watch(self._on_mode_change, "value")

    def _on_mode_change(self, event):
        is_girder = event.new == "Girder (HTMDEC)"
        self.api_url.visible = is_girder
        self.api_key.visible = is_girder
        self.connect_btn.visible = is_girder
        if not is_girder:
            self.app.girder_client = None
            self.status.object = "Local file mode selected. Go to the Files tab to select input files."
            self.status.alert_type = "success"

    def _on_connect(self, event):
        try:
            from helix.girder.client import HelixGirderClient
            client = HelixGirderClient(
                api_url=self.api_url.value,
                api_key=self.api_key.value,
            )
            client.connect()
            self.app.girder_client = client
            self.status.object = f"Connected to {self.api_url.value}"
            self.status.alert_type = "success"
        except Exception as e:
            self.status.object = f"Connection failed: {e}"
            self.status.alert_type = "danger"

    def panel(self) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown("# Connection"),
            pn.pane.Markdown("Connect to the HTMDEC data portal or work with local files."),
            self.mode,
            pn.layout.Divider(),
            self.api_url,
            self.api_key,
            self.connect_btn,
            self.status,
            sizing_mode="stretch_width",
        )
