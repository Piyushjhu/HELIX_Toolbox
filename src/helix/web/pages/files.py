"""Files page — select PDV input files from Girder or local filesystem."""
from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import panel as pn

if TYPE_CHECKING:
    from helix.web.app import HelixApp


class FilesPage:
    def __init__(self, app: HelixApp):
        self.app = app

        # --- Girder browser ---
        self.folder_id = pn.widgets.TextInput(
            name="Girder Folder ID",
            placeholder="Paste a Girder folder ID...",
            width=500,
        )
        self.browse_btn = pn.widgets.Button(
            name="Browse Folder", button_type="primary", width=200,
        )
        self.browse_btn.on_click(self._on_browse_girder)

        self.girder_items = pn.widgets.MultiSelect(
            name="Available Files",
            options=[],
            size=10,
            width=600,
        )
        self.download_btn = pn.widgets.Button(
            name="Download Selected", button_type="success", width=200,
        )
        self.download_btn.on_click(self._on_download)

        # --- Local file input ---
        self.local_input = pn.widgets.TextAreaInput(
            name="Local File Paths (one per line)",
            placeholder="/path/to/pdv_file1.csv\n/path/to/pdv_file2.csv",
            height=150,
            width=600,
        )
        self.local_dir = pn.widgets.TextInput(
            name="Or Input Directory",
            placeholder="/path/to/pdv_data_folder/",
            width=500,
        )
        self.local_pattern = pn.widgets.TextInput(
            name="File Pattern", value="*.csv", width=200,
        )
        self.scan_btn = pn.widgets.Button(
            name="Scan Directory", button_type="primary", width=200,
        )
        self.scan_btn.on_click(self._on_scan_local)

        # --- Output dir ---
        self.output_dir = pn.widgets.TextInput(
            name="Output Directory",
            value=os.path.join(os.path.expanduser("~"), "HELIX_output"),
            width=500,
        )

        # --- Selected files display ---
        self.file_list = pn.pane.Markdown("_No files selected yet._")
        self.status = pn.pane.Alert("", alert_type="info", visible=False)

    def _on_browse_girder(self, event):
        gc = self.app.girder_client
        if gc is None:
            self.status.object = "Not connected to Girder. Go to Connection tab first."
            self.status.alert_type = "warning"
            self.status.visible = True
            return

        try:
            items = gc.list_items(self.folder_id.value)
            csv_items = [it for it in items if it["name"].endswith(".csv")]
            self.girder_items.options = {it["name"]: it["_id"] for it in csv_items}
            self.status.object = f"Found {len(csv_items)} CSV files in folder."
            self.status.alert_type = "info"
            self.status.visible = True
        except Exception as e:
            self.status.object = f"Error browsing folder: {e}"
            self.status.alert_type = "danger"
            self.status.visible = True

    def _on_download(self, event):
        gc = self.app.girder_client
        if gc is None:
            return

        selected_ids = self.girder_items.value
        if not selected_ids:
            self.status.object = "No files selected."
            self.status.alert_type = "warning"
            self.status.visible = True
            return

        dest = os.path.join(tempfile.gettempdir(), "helix_girder_input")
        os.makedirs(dest, exist_ok=True)
        downloaded = []

        # Get name→id mapping
        name_to_id = {v: k for k, v in self.girder_items.options.items()} if isinstance(self.girder_items.options, dict) else {}
        id_to_name = {v: k for k, v in (self.girder_items.options.items() if isinstance(self.girder_items.options, dict) else [])}

        for item_id in selected_ids:
            try:
                item_name = id_to_name.get(item_id, item_id)
                files = gc.list_files(item_id)
                if files:
                    local_path = os.path.join(dest, item_name)
                    gc.download_file_to_path(files[0]["_id"], local_path)
                    downloaded.append(local_path)
            except Exception as e:
                self.status.object = f"Error downloading {item_id}: {e}"
                self.status.alert_type = "danger"
                self.status.visible = True

        self.app.input_files = downloaded
        self._update_file_list()
        self.status.object = f"Downloaded {len(downloaded)} files to {dest}"
        self.status.alert_type = "success"
        self.status.visible = True

    def _on_scan_local(self, event):
        import glob
        directory = self.local_dir.value.strip()
        if not directory or not os.path.isdir(directory):
            self.status.object = f"Directory not found: {directory}"
            self.status.alert_type = "danger"
            self.status.visible = True
            return

        pattern = os.path.join(directory, self.local_pattern.value)
        files = sorted(glob.glob(pattern))
        self.app.input_files = files
        self._update_file_list()
        self.status.object = f"Found {len(files)} files matching {self.local_pattern.value}"
        self.status.alert_type = "success"
        self.status.visible = True

    def _update_file_list(self):
        # Also pick up manually typed paths
        manual_text = self.local_input.value.strip()
        if manual_text:
            manual_files = [f.strip() for f in manual_text.split("\n") if f.strip()]
            existing = set(self.app.input_files)
            for f in manual_files:
                if f not in existing and os.path.isfile(f):
                    self.app.input_files.append(f)

        self.app.output_dir = self.output_dir.value.strip()

        if self.app.input_files:
            lines = [f"- `{os.path.basename(f)}`" for f in self.app.input_files]
            self.file_list.object = f"**{len(lines)} file(s) selected:**\n\n" + "\n".join(lines)
        else:
            self.file_list.object = "_No files selected yet._"

    def panel(self) -> pn.Column:
        girder_section = pn.Column(
            pn.pane.Markdown("### Girder Browser"),
            self.folder_id,
            self.browse_btn,
            self.girder_items,
            self.download_btn,
            pn.layout.Divider(),
        )

        local_section = pn.Column(
            pn.pane.Markdown("### Local Files"),
            self.local_input,
            pn.Row(self.local_dir, self.local_pattern, self.scan_btn),
            pn.layout.Divider(),
        )

        return pn.Column(
            pn.pane.Markdown("# File Selection"),
            pn.pane.Markdown("Select PDV input files from Girder or your local filesystem."),
            self.status,
            girder_section,
            local_section,
            pn.pane.Markdown("### Output"),
            self.output_dir,
            pn.layout.Divider(),
            self.file_list,
            sizing_mode="stretch_width",
        )
