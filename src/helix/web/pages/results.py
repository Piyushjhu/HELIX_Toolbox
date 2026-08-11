"""Results page — view pipeline output figures, tables, and download files."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import panel as pn

if TYPE_CHECKING:
    from helix.web.app import HelixApp


class ResultsPage:
    def __init__(self, app: HelixApp):
        self.app = app

        self.refresh_btn = pn.widgets.Button(
            name="Refresh Results", button_type="primary", width=200,
        )
        self.refresh_btn.on_click(self._on_refresh)

        self.upload_btn = pn.widgets.Button(
            name="Upload to Girder", button_type="success", width=200,
        )
        self.upload_btn.on_click(self._on_upload)

        self.upload_folder_id = pn.widgets.TextInput(
            name="Target Girder Folder ID",
            placeholder="Paste the Girder folder ID for uploads...",
            width=400,
        )

        self.summary = pn.pane.Markdown("_Run the pipeline first to see results._")
        self.figures = pn.Column()
        self.file_table = pn.pane.Markdown("")
        self.status = pn.pane.Alert("", alert_type="info", visible=False)

    def _on_refresh(self, event):
        result = self.app.pipeline_result
        if result is None:
            self.summary.object = "_No results yet. Run the pipeline first._"
            self.figures.clear()
            return

        # Summary
        n_ok = len(result.get("successful_files", []))
        n_fail = len(result.get("failed_files", []))
        lines = [
            f"**Successful:** {n_ok} files",
            f"**Failed:** {n_fail} files",
        ]
        if result.get("failed_files"):
            lines.append("\n**Failed files:**")
            for f in result["failed_files"]:
                lines.append(f"- `{f}`")
        self.summary.object = "\n\n".join(lines)

        # List output files
        output_dir = self.app.output_dir
        if os.path.isdir(output_dir):
            all_files = sorted(os.listdir(output_dir))
            png_files = [f for f in all_files if f.endswith(".png")]
            csv_files = [f for f in all_files if f.endswith(".csv")]

            # Show figures
            self.figures.clear()
            for png in png_files[:20]:  # limit to 20 figures
                png_path = os.path.join(output_dir, png)
                self.figures.append(
                    pn.Column(
                        pn.pane.Markdown(f"**{png}**"),
                        pn.pane.PNG(png_path, width=800),
                    )
                )

            # File table
            file_lines = ["### Output Files\n"]
            file_lines.append(f"**Directory:** `{output_dir}`\n")
            file_lines.append(f"| File | Size |")
            file_lines.append(f"|------|------|")
            for f in all_files:
                fpath = os.path.join(output_dir, f)
                if os.path.isfile(fpath):
                    size = os.path.getsize(fpath)
                    if size > 1_000_000:
                        size_str = f"{size / 1_000_000:.1f} MB"
                    elif size > 1_000:
                        size_str = f"{size / 1_000:.1f} KB"
                    else:
                        size_str = f"{size} B"
                    file_lines.append(f"| `{f}` | {size_str} |")
            self.file_table.object = "\n".join(file_lines)

    def _on_upload(self, event):
        gc = self.app.girder_client
        if gc is None:
            self.status.object = "Not connected to Girder."
            self.status.alert_type = "danger"
            self.status.visible = True
            return

        folder_id = self.upload_folder_id.value.strip()
        if not folder_id:
            self.status.object = "Enter a target Girder folder ID."
            self.status.alert_type = "warning"
            self.status.visible = True
            return

        try:
            gc.upload_results(self.app.output_dir, folder_id)
            self.status.object = "Results uploaded to Girder."
            self.status.alert_type = "success"
            self.status.visible = True
        except Exception as e:
            self.status.object = f"Upload failed: {e}"
            self.status.alert_type = "danger"
            self.status.visible = True

    def panel(self) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown("# Results"),
            self.refresh_btn,
            self.summary,
            pn.layout.Divider(),
            self.figures,
            pn.layout.Divider(),
            self.file_table,
            pn.layout.Divider(),
            pn.pane.Markdown("### Upload to Girder"),
            pn.Row(self.upload_folder_id, self.upload_btn),
            self.status,
            sizing_mode="stretch_width",
        )
