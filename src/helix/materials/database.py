"""Material properties database for shock physics experiments."""
from __future__ import annotations

from typing import Dict, List, Optional

MATERIAL_DATABASE: Dict[str, Dict[str, float]] = {
    # Metals
    "Copper": {"density": 8960, "bulk_wave_speed": 3940},
    "Cu": {"density": 8960, "bulk_wave_speed": 3940},
    "Aluminum": {"density": 2700, "bulk_wave_speed": 5240},
    "Al": {"density": 2700, "bulk_wave_speed": 5240},
    "Iron": {"density": 7874, "bulk_wave_speed": 4910},
    "Fe": {"density": 7874, "bulk_wave_speed": 4910},
    "Steel": {"density": 7850, "bulk_wave_speed": 4570},
    "Stainless Steel": {"density": 7900, "bulk_wave_speed": 4570},
    "Titanium": {"density": 4506, "bulk_wave_speed": 4950},
    "Ti": {"density": 4506, "bulk_wave_speed": 4950},
    "Nickel": {"density": 8908, "bulk_wave_speed": 4970},
    "Ni": {"density": 8908, "bulk_wave_speed": 4970},
    "Gold": {"density": 19300, "bulk_wave_speed": 3240},
    "Au": {"density": 19300, "bulk_wave_speed": 3240},
    "Silver": {"density": 10490, "bulk_wave_speed": 3650},
    "Ag": {"density": 10490, "bulk_wave_speed": 3650},
    "Tantalum": {"density": 16690, "bulk_wave_speed": 3400},
    "Ta": {"density": 16690, "bulk_wave_speed": 3400},
    "Tungsten": {"density": 19250, "bulk_wave_speed": 4030},
    "W": {"density": 19250, "bulk_wave_speed": 4030},
    "Magnesium": {"density": 1738, "bulk_wave_speed": 4940},
    "Mg": {"density": 1738, "bulk_wave_speed": 4940},
    "Zinc": {"density": 7140, "bulk_wave_speed": 3700},
    "Zn": {"density": 7140, "bulk_wave_speed": 3700},
    "Lead": {"density": 11340, "bulk_wave_speed": 2160},
    "Pb": {"density": 11340, "bulk_wave_speed": 2160},
    # Polymers
    "PMMA": {"density": 1190, "bulk_wave_speed": 2680},
    "Polycarbonate": {"density": 1200, "bulk_wave_speed": 2270},
    "PC": {"density": 1200, "bulk_wave_speed": 2270},
    "Polyethylene": {"density": 950, "bulk_wave_speed": 2430},
    "PE": {"density": 950, "bulk_wave_speed": 2430},
    "Polystyrene": {"density": 1050, "bulk_wave_speed": 2350},
    "PS": {"density": 1050, "bulk_wave_speed": 2350},
    "Teflon": {"density": 2200, "bulk_wave_speed": 1350},
    "PTFE": {"density": 2200, "bulk_wave_speed": 1350},
    # Ceramics and glasses
    "Glass": {"density": 2500, "bulk_wave_speed": 5660},
    "Fused Silica": {"density": 2203, "bulk_wave_speed": 5968},
    "SiO2": {"density": 2203, "bulk_wave_speed": 5968},
    "Sapphire": {"density": 3980, "bulk_wave_speed": 11190},
    "Al2O3": {"density": 3980, "bulk_wave_speed": 11190},
    "Silicon": {"density": 2329, "bulk_wave_speed": 8433},
    "Si": {"density": 2329, "bulk_wave_speed": 8433},
    "Silicon Carbide": {"density": 3210, "bulk_wave_speed": 12000},
    "SiC": {"density": 3210, "bulk_wave_speed": 12000},
    # Other
    "Water": {"density": 1000, "bulk_wave_speed": 1480},
    "H2O": {"density": 1000, "bulk_wave_speed": 1480},
    "Diamond": {"density": 3515, "bulk_wave_speed": 18000},
    "Graphite": {"density": 2260, "bulk_wave_speed": 2500},
}

DEFAULT_DENSITY = 8960  # kg/m³ (Copper)
DEFAULT_BULK_WAVE_SPEED = 3940  # m/s (Copper)


def get_material_properties(
    material_name: str,
    default_density: Optional[float] = None,
    default_acoustic_velocity: Optional[float] = None,
) -> Dict:
    """Look up material properties by name.

    Tries exact match, then case-insensitive. Falls back to provided defaults
    or Copper properties.
    """
    material_name = str(material_name).strip()
    fallback_density = default_density if default_density is not None else DEFAULT_DENSITY
    fallback_velocity = (
        default_acoustic_velocity if default_acoustic_velocity is not None else DEFAULT_BULK_WAVE_SPEED
    )

    # exact match
    if material_name in MATERIAL_DATABASE:
        props = MATERIAL_DATABASE[material_name].copy()
        props.update(material_found=True, material_name=material_name)
        return props

    # case-insensitive
    for db_name, db_props in MATERIAL_DATABASE.items():
        if db_name.lower() == material_name.lower():
            props = db_props.copy()
            props.update(material_found=True, material_name=db_name)
            return props

    return {
        "density": fallback_density,
        "bulk_wave_speed": fallback_velocity,
        "material_found": False,
        "material_name": material_name,
    }


def list_available_materials() -> List[str]:
    return sorted(MATERIAL_DATABASE.keys())


def add_material(name: str, density: float, bulk_wave_speed: float) -> None:
    MATERIAL_DATABASE[name] = {"density": density, "bulk_wave_speed": bulk_wave_speed}
