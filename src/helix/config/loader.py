"""Configuration loading and saving for HELIX Toolbox."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger("helix")


def load_config(file_path: str | Path) -> Dict[str, Any]:
    """Load a JSON config file and return the parsed dictionary."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)


def save_config(config: Dict[str, Any], file_path: str | Path) -> None:
    """Save configuration dictionary to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info("Configuration saved to %s", file_path)


def load_master_config(file_path: str | Path) -> Dict[str, Any]:
    """Load master config containing cli_settings, alpss_config, spade_config.

    Returns the full master dict with validated sections.
    """
    config = load_config(file_path)
    required = ["cli_settings", "alpss_config", "spade_config"]
    missing = [s for s in required if s not in config]
    if missing:
        raise ValueError(
            f"Master config must contain sections: {', '.join(missing)}"
        )
    return config
