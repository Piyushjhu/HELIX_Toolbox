#!/usr/bin/env python3
"""
Material Properties Database
Contains density and bulk wave speed for common materials used in shock physics experiments
"""

# Material properties database: {material_name: {'density': kg/m³, 'bulk_wave_speed': m/s}}
MATERIAL_DATABASE = {
    # Metals
    'Copper': {'density': 8960, 'bulk_wave_speed': 3940},
    'Cu': {'density': 8960, 'bulk_wave_speed': 3940},
    'Aluminum': {'density': 2700, 'bulk_wave_speed': 5240},
    'Al': {'density': 2700, 'bulk_wave_speed': 5240},
    'Iron': {'density': 7874, 'bulk_wave_speed': 4910},
    'Fe': {'density': 7874, 'bulk_wave_speed': 4910},
    'Steel': {'density': 7850, 'bulk_wave_speed': 4570},
    'Stainless Steel': {'density': 7900, 'bulk_wave_speed': 4570},
    'Titanium': {'density': 4506, 'bulk_wave_speed': 4950},
    'Ti': {'density': 4506, 'bulk_wave_speed': 4950},
    'Nickel': {'density': 8908, 'bulk_wave_speed': 4970},
    'Ni': {'density': 8908, 'bulk_wave_speed': 4970},
    'Gold': {'density': 19300, 'bulk_wave_speed': 3240},
    'Au': {'density': 19300, 'bulk_wave_speed': 3240},
    'Silver': {'density': 10490, 'bulk_wave_speed': 3650},
    'Ag': {'density': 10490, 'bulk_wave_speed': 3650},
    'Tantalum': {'density': 16690, 'bulk_wave_speed': 3400},
    'Ta': {'density': 16690, 'bulk_wave_speed': 3400},
    'Tungsten': {'density': 19250, 'bulk_wave_speed': 4030},
    'W': {'density': 19250, 'bulk_wave_speed': 4030},
    'Magnesium': {'density': 1738, 'bulk_wave_speed': 4940},
    'Mg': {'density': 1738, 'bulk_wave_speed': 4940},
    'Zinc': {'density': 7140, 'bulk_wave_speed': 3700},
    'Zn': {'density': 7140, 'bulk_wave_speed': 3700},
    'Lead': {'density': 11340, 'bulk_wave_speed': 2160},
    'Pb': {'density': 11340, 'bulk_wave_speed': 2160},
    
    # Polymers and organics
    'PMMA': {'density': 1190, 'bulk_wave_speed': 2680},
    'Polycarbonate': {'density': 1200, 'bulk_wave_speed': 2270},
    'PC': {'density': 1200, 'bulk_wave_speed': 2270},
    'Polyethylene': {'density': 950, 'bulk_wave_speed': 2430},
    'PE': {'density': 950, 'bulk_wave_speed': 2430},
    'Polystyrene': {'density': 1050, 'bulk_wave_speed': 2350},
    'PS': {'density': 1050, 'bulk_wave_speed': 2350},
    'Teflon': {'density': 2200, 'bulk_wave_speed': 1350},
    'PTFE': {'density': 2200, 'bulk_wave_speed': 1350},
    
    # Ceramics and glasses
    'Glass': {'density': 2500, 'bulk_wave_speed': 5660},
    'Fused Silica': {'density': 2203, 'bulk_wave_speed': 5968},
    'SiO2': {'density': 2203, 'bulk_wave_speed': 5968},
    'Sapphire': {'density': 3980, 'bulk_wave_speed': 11190},
    'Al2O3': {'density': 3980, 'bulk_wave_speed': 11190},
    'Silicon': {'density': 2329, 'bulk_wave_speed': 8433},
    'Si': {'density': 2329, 'bulk_wave_speed': 8433},
    'Silicon Carbide': {'density': 3210, 'bulk_wave_speed': 12000},
    'SiC': {'density': 3210, 'bulk_wave_speed': 12000},
    
    # Other materials
    'Water': {'density': 1000, 'bulk_wave_speed': 1480},
    'H2O': {'density': 1000, 'bulk_wave_speed': 1480},
    'Diamond': {'density': 3515, 'bulk_wave_speed': 18000},
    'Graphite': {'density': 2260, 'bulk_wave_speed': 2500},
}

def get_material_properties(material_name):
    """
    Get material properties from the database.

    No fallback values are substituted for an unrecognized material -- callers
    must treat 'material_found': False as an error condition (e.g. flag the
    trace as "Mat not found") rather than compute with a guessed density/speed.

    Parameters:
    -----------
    material_name : str
        Name of the material to look up

    Returns:
    --------
    dict : Dictionary containing:
        - 'density': float (kg/m³) or None if not found
        - 'bulk_wave_speed': float (m/s) or None if not found
        - 'material_found': bool (True if found in database)
        - 'material_name': str (the material name used)
    """
    # Clean up the material name
    material_name = str(material_name).strip()

    # Try exact match first
    if material_name in MATERIAL_DATABASE:
        props = MATERIAL_DATABASE[material_name].copy()
        props['material_found'] = True
        props['material_name'] = material_name
        return props

    # Try case-insensitive match
    for db_name, db_props in MATERIAL_DATABASE.items():
        if db_name.lower() == material_name.lower():
            props = db_props.copy()
            props['material_found'] = True
            props['material_name'] = db_name
            return props

    # Material not found -- no fallback
    return {
        'density': None,
        'bulk_wave_speed': None,
        'material_found': False,
        'material_name': material_name
    }


def list_available_materials():
    """
    Get a list of all available materials in the database.
    
    Returns:
    --------
    list : Sorted list of material names
    """
    return sorted(MATERIAL_DATABASE.keys())


def add_material(name, density, bulk_wave_speed):
    """
    Add a new material to the database.
    
    Parameters:
    -----------
    name : str
        Material name
    density : float
        Density in kg/m³
    bulk_wave_speed : float
        Bulk wave speed in m/s
    """
    MATERIAL_DATABASE[name] = {
        'density': density,
        'bulk_wave_speed': bulk_wave_speed
    }


if __name__ == '__main__':
    # Test the module
    print("Available materials:")
    for mat in list_available_materials():
        props = get_material_properties(mat)
        print(f"  {mat}: ρ = {props['density']} kg/m³, c = {props['bulk_wave_speed']} m/s")
    
    print("\nTesting unknown material:")
    props = get_material_properties('Unknown Material')
    print(f"  Unknown Material: ρ = {props['density']} kg/m³, c = {props['bulk_wave_speed']} m/s")
    print(f"  Material found: {props['material_found']}")

