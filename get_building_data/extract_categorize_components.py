from analyse_US_building_structure import analyze_current_building_structure, identify_uk_conversion_priorities

def extract_material_properties(building_data):
    """
    Extract all material properties that need UK equivalents
    """
    
    print("\n\n🧱 MATERIAL PROPERTIES ANALYSIS")
    print("=" * 50)
    
    materials_found = {}
    
    # Extract materials
    if "Material" in building_data:
        materials = building_data["Material"]
        print(f"📦 Found {len(materials)} materials:")
        
        for mat_name, properties in materials.items():
            print(f"\n🔹 {mat_name}:")
            materials_found[mat_name] = properties
            
            # Show key thermal properties
            key_thermal_props = [
                "conductivity", "density", "specific_heat", 
                "thermal_absorptance", "solar_absorptance"
            ]
            
            for prop in key_thermal_props:
                if prop in properties:
                    print(f"   {prop}: {properties[prop]}")
    
    # Extract constructions (assemblies of materials)
    constructions_found = {}
    if "Construction" in building_data:
        constructions = building_data["Construction"]
        print(f"\n🏗️ Found {len(constructions)} constructions:")
        
        for const_name, properties in constructions.items():
            print(f"\n🔸 {const_name}:")
            constructions_found[const_name] = properties
            
            # Show layer structure
            if "outside_layer" in properties:
                print(f"   Outside layer: {properties['outside_layer']}")
            
            # Show additional layers
            layer_count = 1
            while f"layer_{layer_count+1}" in properties:
                layer_count += 1
                layer_key = f"layer_{layer_count}"
                print(f"   Layer {layer_count}: {properties[layer_key]}")
    
    return materials_found, constructions_found

def extract_hvac_equipment(building_data):
    """
    Extract HVAC equipment that needs UK specifications
    """
    
    print("\n\n🌡️ HVAC EQUIPMENT ANALYSIS")
    print("=" * 50)
    
    hvac_categories = [
        "Coil:Cooling:DX:SingleSpeed",
        "Coil:Heating:Electric",
        "Fan:VariableVolume",
        "AirLoopHVAC"
    ]
    
    hvac_equipment = {}
    
    for category in hvac_categories:
        if category in building_data:
            equipment = building_data[category]
            hvac_equipment[category] = equipment
            
            print(f"\n🔧 {category}: {len(equipment)} units")
            
            # Show first equipment example
            if equipment:
                first_equipment = list(equipment.values())[0]
                print(f"   Example equipment:")
                
                # Show key performance parameters
                key_performance_params = [
                    "rated_total_cooling_capacity", "rated_sensible_heat_ratio",
                    "rated_cop", "nominal_capacity", "efficiency"
                ]
                
                for param in key_performance_params:
                    if param in first_equipment:
                        print(f"   {param}: {first_equipment[param]}")
    
    return hvac_equipment

def extract_occupancy_schedules(building_data):
    """
    Extract occupancy and operational schedules for UK adaptation
    """
    
    print("\n\n👥 OCCUPANCY & SCHEDULES ANALYSIS")
    print("=" * 50)
    
    schedule_categories = [
        "Schedule:Compact",
        "People", 
        "Lights",
        "ElectricEquipment"
    ]
    
    schedules_found = {}
    
    for category in schedule_categories:
        if category in building_data:
            schedules = building_data[category]
            schedules_found[category] = schedules
            
            print(f"\n📅 {category}: {len(schedules)} items")
            
            # Show first schedule example
            if schedules:
                first_schedule = list(schedules.values())[0]
                print(f"   Example:")
                
                # Show key schedule parameters
                if category == "People":
                    key_params = ["number_of_people", "activity_level_schedule_name"]
                elif category == "Schedule:Compact":
                    key_params = ["schedule_type_limits_name"]
                else:
                    key_params = list(first_schedule.keys())[:3]
                
                for param in key_params:
                    if param in first_schedule:
                        print(f"   {param}: {first_schedule[param]}")
    
    return schedules_found

if __name__ == "__main__":
    # Run the extraction analysis
    analysis_summary, building_data = analyze_current_building_structure()
    identify_uk_conversion_priorities(building_data)
    materials_found, constructions_found = extract_material_properties(building_data)
    hvac_equipment = extract_hvac_equipment(building_data)
    schedules_found = extract_occupancy_schedules(building_data)

    print("\n\n✅ EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"📦 Materials to convert: {len(materials_found)}")
    print(f"🏗️ Constructions to convert: {len(constructions_found)}")
    print(f"🌡️ HVAC equipment to convert: {sum(len(equip) for equip in hvac_equipment.values())}")
    print(f"📅 Schedules to convert: {sum(len(sched) for sched in schedules_found.values())}")