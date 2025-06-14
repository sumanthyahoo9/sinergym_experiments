def create_uk_material_database():
    """
    Create a database of UK equivalent materials based on current US materials
    """
    
    print("\n🇬🇧 UK MATERIAL CONVERSION DATABASE")
    print("=" * 50)
    
    # Current US materials that need UK equivalents
    us_materials = {
        'BR01': 'Brick (US Standard)',
        'CC03': 'Heavy Weight Concrete', 
        'GP01': 'Gypsum Board (US)',
        'GP02': 'Gypsum Board (US)',
        'IN02': 'Insulation Board (US)',
        'IN46': 'Insulation (US)',
        'PW03': 'Plywood (US)',
        'RG01': 'Roof Gravel (US)',
        'WD01': 'Wood Siding (US)',
        'WD10': 'Wood Siding (US)'
    }
    
    # UK equivalent materials with properties
    uk_material_equivalents = {
        'BR01': {
            'uk_name': 'UK_BRICK_FACING',
            'description': 'UK Facing Brick',
            'conductivity': 0.84,  # UK brick typically higher conductivity
            'density': 1900.0,     # UK brick typically denser
            'specific_heat': 800.0,
            'thermal_absorptance': 0.9,
            'solar_absorptance': 0.7,
            'roughness': 'MediumRough',
            'source': 'CIBSE Guide A'
        },
        'CC03': {
            'uk_name': 'UK_CONCRETE_DENSE',
            'description': 'UK Dense Concrete',
            'conductivity': 1.4,   # UK concrete standards
            'density': 2100.0,
            'specific_heat': 840.0,
            'thermal_absorptance': 0.9,
            'solar_absorptance': 0.65,
            'roughness': 'MediumRough',
            'source': 'BS EN 12524'
        },
        'GP01': {
            'uk_name': 'UK_PLASTERBOARD_12MM',
            'description': 'UK Plasterboard 12.5mm',
            'conductivity': 0.25,  # UK plasterboard
            'density': 950.0,
            'specific_heat': 840.0,
            'thermal_absorptance': 0.9,
            'solar_absorptance': 0.5,
            'roughness': 'Smooth',
            'source': 'British Gypsum'
        },
        'GP02': {
            'uk_name': 'UK_PLASTERBOARD_15MM',
            'description': 'UK Plasterboard 15mm',
            'conductivity': 0.25,
            'density': 950.0,
            'specific_heat': 840.0,
            'thermal_absorptance': 0.9,
            'solar_absorptance': 0.5,
            'roughness': 'Smooth',
            'source': 'British Gypsum'
        },
        'IN02': {
            'uk_name': 'UK_MINERAL_WOOL_100MM',
            'description': 'UK Mineral Wool Insulation 100mm',
            'conductivity': 0.038,  # UK insulation standards
            'density': 25.0,
            'specific_heat': 840.0,
            'thermal_absorptance': 0.9,
            'solar_absorptance': 0.5,
            'roughness': 'Rough',
            'source': 'Rockwool UK'
        },
        'IN46': {
            'uk_name': 'UK_MINERAL_WOOL_150MM',
            'description': 'UK Mineral Wool Insulation 150mm',
            'conductivity': 0.035,  # Better UK insulation
            'density': 30.0,
            'specific_heat': 840.0,
            'thermal_absorptance': 0.9,
            'solar_absorptance': 0.5,
            'roughness': 'Rough',
            'source': 'Knauf UK'
        },
        'PW03': {
            'uk_name': 'UK_OSB_18MM',
            'description': 'UK Oriented Strand Board 18mm',
            'conductivity': 0.13,   # UK timber products
            'density': 650.0,
            'specific_heat': 1700.0,
            'thermal_absorptance': 0.9,
            'solar_absorptance': 0.78,
            'roughness': 'MediumRough',
            'source': 'Norbord Europe'
        },
        'RG01': {
            'uk_name': 'UK_ROOF_FELT_MEMBRANE',
            'description': 'UK Built-up Roof Membrane',
            'conductivity': 0.19,   # UK roofing materials
            'density': 1100.0,
            'specific_heat': 1000.0,
            'thermal_absorptance': 0.9,
            'solar_absorptance': 0.65,
            'roughness': 'Rough',
            'source': 'IKO PLC'
        },
        'WD01': {
            'uk_name': 'UK_TIMBER_CLADDING',
            'description': 'UK Timber Cladding',
            'conductivity': 0.14,   # UK softwood
            'density': 550.0,
            'specific_heat': 1600.0,
            'thermal_absorptance': 0.9,
            'solar_absorptance': 0.78,
            'roughness': 'MediumRough',
            'source': 'TRADA'
        },
        'WD10': {
            'uk_name': 'UK_TIMBER_EXTERNAL',
            'description': 'UK External Grade Timber',
            'conductivity': 0.14,
            'density': 550.0,
            'specific_heat': 1600.0,
            'thermal_absorptance': 0.9,
            'solar_absorptance': 0.78,
            'roughness': 'MediumRough',
            'source': 'TRADA'
        }
    }
    
    # Print conversion table
    print("\n📊 MATERIAL CONVERSION TABLE:")
    print("-" * 80)
    print(f"{'US Material':<10} {'UK Equivalent':<25} {'Conductivity':<12} {'Source':<15}")
    print("-" * 80)
    
    for us_mat, uk_data in uk_material_equivalents.items():
        us_desc = us_materials[us_mat]
        uk_name = uk_data['uk_name']
        conductivity = uk_data['conductivity']
        source = uk_data['source']
        print(f"{us_mat:<10} {uk_name:<25} {conductivity:<12.3f} {source:<15}")
    
    return uk_material_equivalents

def create_uk_construction_assemblies():
    """
    Create UK construction assemblies using UK materials
    """
    
    print("\n\n🏗️ UK CONSTRUCTION ASSEMBLIES")
    print("=" * 50)
    
    # Current US constructions that need UK equivalents
    us_constructions = [
        'WALL-1', 'ROOF-1', 'FLOOR-SLAB-1', 'INT-WALL-1', 'CLNG-1'
    ]
    
    # UK construction assemblies following UK Building Regulations
    uk_construction_assemblies = {
        'WALL-1': {
            'uk_name': 'UK_EXTERNAL_WALL_CAVITY',
            'description': 'UK External Cavity Wall (Part L Compliant)',
            'layers': [
                'UK_BRICK_FACING',           # External brick
                'UK_CAVITY_50MM',            # Cavity
                'UK_MINERAL_WOOL_100MM',     # Cavity insulation
                'UK_CONCRETE_BLOCK_100MM',   # Inner block
                'UK_PLASTERBOARD_12MM'       # Internal finish
            ],
            'u_value_target': 0.28,  # W/m²K - UK Building Regs Part L
            'source': 'Approved Document Part L'
        },
        'ROOF-1': {
            'uk_name': 'UK_PITCHED_ROOF_INSULATED',
            'description': 'UK Pitched Roof with Insulation',
            'layers': [
                'UK_ROOF_TILE_CONCRETE',     # External tiles
                'UK_ROOF_FELT_MEMBRANE',     # Underlay
                'UK_ROOF_BATTEN_25MM',       # Battens
                'UK_MINERAL_WOOL_200MM',     # Insulation between rafters
                'UK_VAPOUR_BARRIER',         # Vapour control
                'UK_PLASTERBOARD_12MM'       # Internal finish
            ],
            'u_value_target': 0.16,  # W/m²K - UK Building Regs
            'source': 'NHBC Standards'
        },
        'FLOOR-SLAB-1': {
            'uk_name': 'UK_GROUND_FLOOR_INSULATED',
            'description': 'UK Insulated Ground Floor',
            'layers': [
                'UK_CONCRETE_SCREED_50MM',   # Floor screed
                'UK_RIGID_INSULATION_100MM', # Floor insulation
                'UK_CONCRETE_SLAB_150MM',    # Structural slab
                'UK_DPM_MEMBRANE',           # Damp proof membrane
                'UK_HARDCORE_150MM'          # Sub-base
            ],
            'u_value_target': 0.22,  # W/m²K
            'source': 'Concrete Centre'
        },
        'INT-WALL-1': {
            'uk_name': 'UK_INTERNAL_PARTITION',
            'description': 'UK Internal Partition Wall',
            'layers': [
                'UK_PLASTERBOARD_12MM',      # Finish one side
                'UK_METAL_STUD_100MM',       # Structural frame
                'UK_MINERAL_WOOL_100MM',     # Acoustic insulation
                'UK_PLASTERBOARD_12MM'       # Finish other side
            ],
            'u_value_target': 'Not applicable',  # Internal wall
            'source': 'British Gypsum'
        },
        'CLNG-1': {
            'uk_name': 'UK_SUSPENDED_CEILING',
            'description': 'UK Suspended Ceiling System',
            'layers': [
                'UK_MINERAL_FIBRE_TILE',     # Ceiling tiles
                'UK_CEILING_GRID_SYSTEM',    # Support grid
                'UK_MINERAL_WOOL_100MM'      # Acoustic insulation
            ],
            'u_value_target': 'Variable',
            'source': 'Armstrong Ceilings'
        }
    }
    
    # Print construction assemblies
    for us_const, uk_data in uk_construction_assemblies.items():
        print(f"\n🔸 {us_const} → {uk_data['uk_name']}")
        print(f"   Description: {uk_data['description']}")
        print(f"   U-value target: {uk_data['u_value_target']}")
        print(f"   Layers:")
        for i, layer in enumerate(uk_data['layers'], 1):
            print(f"     {i}. {layer}")
        print(f"   Source: {uk_data['source']}")
    
    return uk_construction_assemblies

if __name__ == "__main__":
    # Run Step 3
    uk_materials = create_uk_material_database()
    uk_constructions = create_uk_construction_assemblies()