"""HELIX Toolbox — Panel web application.

Launch with::

    helix-web                  # uses entry point
    panel serve src/helix/web/app.py   # direct
"""
from __future__ import annotations

import logging
import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import panel as pn

from helix.web.sidebar import create_sidebar
from helix.web.pages.connection import ConnectionPage
from helix.web.pages.files import FilesPage
from helix.web.pages.config import ConfigPage
from helix.web.pages.run import RunPage
from helix.web.pages.results import ResultsPage

pn.extension("tabulator", sizing_mode="stretch_width")

logger = logging.getLogger("helix.web")

ACCENT = "#2c6fbb"


class HelixApp:
    """Main application state container."""

    def __init__(self):
        # Shared state across pages
        self.girder_client = None  # HelixGirderClient or None
        self.input_files = []
        self.output_dir = os.path.join(tempfile.gettempdir(), "helix_output")
        self.alpss_params = {}
        self.spade_params = {}
        self.analysis_mode = "both"
        self.pipeline_result = None

        # Build pages
        self.connection_page = ConnectionPage(self)
        self.files_page = FilesPage(self)
        self.config_page = ConfigPage(self)
        self.run_page = RunPage(self)
        self.results_page = ResultsPage(self)

        # Build template
        self.template = pn.template.FastListTemplate(
            title="HELIX Toolbox",
            accent_base_color=ACCENT,
            header_background=ACCENT,
            sidebar_width=280,
        )

        # Sidebar navigation
        self.nav = create_sidebar(self)
        self.template.sidebar.append(self.nav)

        # Main area — tabs
        self.tabs = pn.Tabs(
            ("Connection", self.connection_page.panel()),
            ("Files", self.files_page.panel()),
            ("Configuration", self.config_page.panel()),
            ("Run", self.run_page.panel()),
            ("Results", self.results_page.panel()),
        )
        self.template.main.append(self.tabs)

    def navigate(self, tab_index: int):
        """Switch to a specific tab."""
        self.tabs.active = tab_index

    def servable(self):
        return self.template.servable()


def create_app() -> HelixApp:
    """Create and return the HELIX web app."""
    return HelixApp()


# Allow `panel serve app.py`
if pn.state.curdoc:
    app = create_app()
    app.servable()
