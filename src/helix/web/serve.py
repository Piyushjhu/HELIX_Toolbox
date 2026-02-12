"""Entry point for `helix-web` command — launches the Panel server."""
from __future__ import annotations

import sys


def main():
    import panel as pn
    from helix.web.app import create_app

    app = create_app()

    # Serve the app
    pn.serve(
        {"/": app.template},
        port=5006,
        show=True,
        title="HELIX Toolbox",
        websocket_origin="*",
    )


if __name__ == "__main__":
    main()
