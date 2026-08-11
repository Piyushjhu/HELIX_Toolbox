"""Sidebar navigation for HELIX web app."""
from __future__ import annotations

from typing import TYPE_CHECKING

import panel as pn

if TYPE_CHECKING:
    from helix.web.app import HelixApp


def create_sidebar(app: HelixApp) -> pn.Column:
    """Build the sidebar with navigation buttons and status indicators."""

    status = pn.pane.Markdown("**Status:** Not connected", name="status_md")

    def _make_nav_btn(label: str, idx: int):
        btn = pn.widgets.Button(name=label, button_type="light", width=240)
        btn.on_click(lambda e: app.navigate(idx))
        return btn

    nav_buttons = pn.Column(
        pn.pane.Markdown("### Navigation"),
        _make_nav_btn("1. Connection", 0),
        _make_nav_btn("2. Files", 1),
        _make_nav_btn("3. Configuration", 2),
        _make_nav_btn("4. Run", 3),
        _make_nav_btn("5. Results", 4),
    )

    version_info = pn.pane.Markdown(
        "_HELIX Toolbox v2.0_",
        styles={"color": "#888", "font-size": "11px"},
    )

    return pn.Column(
        pn.pane.Markdown("## HELIX"),
        status,
        pn.layout.Divider(),
        nav_buttons,
        pn.layout.Divider(),
        version_info,
        sizing_mode="stretch_width",
    )
