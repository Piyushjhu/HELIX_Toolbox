"""Material property resolution with multi-source fallback hierarchy.

Priority:
  1. Config ``material_properties`` section
  2. Built-in material database
  3. Provided defaults / Copper fallback
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from helix.materials.database import get_material_properties

logger = logging.getLogger("helix")


class MaterialResolver:
    """Resolves material properties from config overrides + database."""

    def __init__(
        self,
        config_materials: Optional[Dict[str, Dict[str, Any]]] = None,
        default_density: Optional[float] = None,
        default_acoustic_velocity: Optional[float] = None,
    ):
        self._config = config_materials or {}
        self._default_density = default_density or 8960
        self._default_acoustic_velocity = default_acoustic_velocity or 3950

    def resolve(self, material_name: str) -> Dict[str, Any]:
        """Return properties dict with density, bulk_wave_speed, C_L, source."""
        name = str(material_name).strip() if material_name else "Unknown"

        # Priority 1: config overrides (exact then case-insensitive)
        props = self._lookup_config(name)
        if props:
            return props

        # Priority 2+3: database → defaults
        db = get_material_properties(name, self._default_density, self._default_acoustic_velocity)
        db["source"] = "database" if db["material_found"] else "default"
        db.setdefault("C_L", db.get("bulk_wave_speed", self._default_acoustic_velocity))
        return db

    # ------------------------------------------------------------------
    def _lookup_config(self, name: str) -> Optional[Dict[str, Any]]:
        hit = self._config.get(name)
        if hit is None:
            for k, v in self._config.items():
                if k.lower() == name.lower():
                    hit = v
                    break
        if hit is None:
            return None

        density = float(hit.get("density", self._default_density))
        bws = float(hit.get("bulk_wave_speed", hit.get("C0", self._default_acoustic_velocity)))
        c_l = float(hit.get("C_L", bws))
        return {
            "density": density,
            "bulk_wave_speed": bws,
            "C_L": c_l,
            "material_found": True,
            "material_name": name,
            "source": "config",
        }
