"""Entry point for `helix-web` command — launches the Panel server."""
from __future__ import annotations

import argparse
import logging

logger = logging.getLogger("helix.web")


def _build_app():
    """Build a fresh HelixApp per session (required by Panel's multi-session model)."""
    import panel as pn
    pn.extension("tabulator", sizing_mode="stretch_width")

    from helix.web.app import HelixApp
    app = HelixApp()
    return app.template


def main():
    parser = argparse.ArgumentParser(description="HELIX Toolbox Web App")
    parser.add_argument("--port", type=int, default=5006, help="Port to serve on (default: 5006)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--address", default="0.0.0.0", help="Address to bind (default: 0.0.0.0)")
    args = parser.parse_args()

    import panel as pn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Starting HELIX Toolbox web app on port %d...", args.port)

    pn.serve(
        {"helix": _build_app},
        port=args.port,
        address=args.address,
        show=not args.no_browser,
        title="HELIX Toolbox",
        websocket_origin="*",
    )


if __name__ == "__main__":
    main()
