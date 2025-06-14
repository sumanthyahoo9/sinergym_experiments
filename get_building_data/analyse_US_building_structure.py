import json
import pandas as pd
from collections import defaultdict

def analyze_current_building_structure(json_file="building_samp_file.json"):
    """
    Analyze the current US building structure to understand what needs UK conversion
    """
    
    # Load the building file
    with open(json_file, 'r') as f:
        building_data = json.load(f)
    
    print("🏗️ BUILDING STRUCTURE ANALYSIS")
    print("=" * 50)
    
    # Get main object categories
    object_categories = list(building_data.keys())
    print(f"📋 Total Object Categories: {len(object_categories)}")
    
    # Analyze each category
    analysis_summary = {}
    
    for category in object_categories:
        objects_in_category = building_data[category]
        count = len(objects_in_category) if isinstance(objects_in_category, dict) else 1
        analysis_summary[category] = count
        
        print(f"\n📂 {category}: {count} objects")
        
        # Show first object as example
        if isinstance(objects_in_category, dict):
            first_object_name = list(objects_in_category.keys())[0]
            first_object = objects_in_category[first_object_name]
            print(f"   Example: {first_object_name}")
            
            # Show key parameters
            if isinstance(first_object, dict):
                key_params = list(first_object.keys())[:3]  # First 3 parameters
                print(f"   Key params: {', '.join(key_params)}")
    
    return analysis_summary, building_data

def identify_uk_conversion_priorities(building_data):
    """
    Identify which building components are highest priority for UK conversion
    """
    
    print("\n\n🎯 UK CONVERSION PRIORITIES")
    print("=" * 50)
    
    # High priority categories (affect energy performance most)
    high_priority = [
        "Material",
        "Construction", 
        "BuildingSurface:Detailed",
        "People",
        "Lights",
        "ElectricEquipment"
    ]
    
    # Medium priority categories (affect HVAC performance)
    medium_priority = [
        "Coil:Cooling:DX:SingleSpeed",
        "Coil:Heating:Electric", 
        "Fan:VariableVolume",
        "Schedule:Compact"
    ]
    
    # Low priority categories (mostly structural/output)
    low_priority = [
        "Output:Variable",
        "Output:Meter",
        "Zone"
    ]
    
    print("🔴 HIGH PRIORITY (Building Envelope & Loads):")
    for category in high_priority:
        if category in building_data:
            count = len(building_data[category])
            print(f"   ✓ {category}: {count} objects")
        else:
            print(f"   ✗ {category}: Not found")
    
    print("\n🟡 MEDIUM PRIORITY (HVAC Equipment):")
    for category in medium_priority:
        if category in building_data:
            count = len(building_data[category])
            print(f"   ✓ {category}: {count} objects")
        else:
            print(f"   ✗ {category}: Not found")
    
    print("\n🟢 LOW PRIORITY (Structure & Output):")
    for category in low_priority:
        if category in building_data:
            count = len(building_data[category])
            print(f"   ✓ {category}: {count} objects")
        else:
            print(f"   ✗ {category}: Not found")

if __name__ == "__main__":
    # Run the analysis
    analysis_summary, building_data = analyze_current_building_structure()
    identify_uk_conversion_priorities(building_data)