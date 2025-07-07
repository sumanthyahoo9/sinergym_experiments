"""
Master AstraZeneca Building Conversion Script
Combines all 5 steps to convert working template to AstraZeneca specifications
Run this script to get final building file ready for Sinergym training
"""

import json
import math
from pathlib import Path

class AstraZenecaBuildingConverter:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.building_data = None
    
    def load_building_file(self):
        """Load the base building file"""
        print(f"🏗️  Loading base building file: {self.input_file}")
        try:
            with open(self.input_file, 'r') as f:
                self.building_data = json.load(f)
            print(f"   ✅ Loaded successfully - {len(self.building_data)} main sections")
            return True
        except FileNotFoundError:
            print(f"   ❌ Error: File {self.input_file} not found")
            return False
        except json.JSONDecodeError as e:
            print(f"   ❌ Error: Invalid JSON format - {e}")
            return False
    
    def step1_building_identity(self):
        """Step 1: Replace Building Identity"""
        print("\n📋 Step 1: Updating building identity...")
        
        # Update Building object
        if "Building" in self.building_data:
            old_building_keys = list(self.building_data["Building"].keys())
            for old_key in old_building_keys:
                del self.building_data["Building"][old_key]
        else:
            self.building_data["Building"] = {}
        
        self.building_data["Building"]["AstraZeneca_Macclesfield"] = {
            "north_axis": 0,
            "terrain": "City",
            "loads_convergence_tolerance_value": 0.04,
            "temperature_convergence_tolerance_value": 0.4,
            "solar_distribution": "FullExterior",
            "maximum_number_of_warmup_days": 25
        }
        
        # Update Version
        if "Version" in self.building_data:
            version_key = list(self.building_data["Version"].keys())[0]
            self.building_data["Version"][version_key]["version_identifier"] = "23.1"
        
        print("   ✅ Building identity updated to AstraZeneca Macclesfield")
    
    def step2_zone_scaling(self):
        """Step 2: Scale Zones to AstraZeneca Areas"""
        print("\n🏢 Step 2: Scaling zones to 6,700m²...")
        
        # AstraZeneca zone specifications
        az_zones = {
            "Ground_Floor_Office": {"floor_area": 3000, "ceiling_height": 3.0, "volume": 9000},
            "First_Floor_Office": {"floor_area": 3000, "ceiling_height": 3.0, "volume": 9000},
            "Meeting_Rooms": {"floor_area": 500, "ceiling_height": 3.0, "volume": 1500},
            "Server_Room": {"floor_area": 100, "ceiling_height": 3.0, "volume": 300},
            "Canteen_Breakout": {"floor_area": 100, "ceiling_height": 3.0, "volume": 300}
        }
        
        # Clear existing zones
        self.building_data["Zone"] = {}
        
        # Add AstraZeneca zones
        for zone_name, zone_data in az_zones.items():
            self.building_data["Zone"][zone_name] = {
                "ceiling_height": zone_data["ceiling_height"],
                "volume": zone_data["volume"],
                "floor_area": zone_data["floor_area"],
                "zone_inside_convection_algorithm": "TARP",
                "zone_outside_convection_algorithm": "DOE-2",
                "zone_air_distribution_effectiveness_in_cooling_mode": 1.0,
                "zone_air_distribution_effectiveness_in_heating_mode": 1.0,
                "part_of_total_floor_area": "Yes"
            }
        
        total_area = sum(zone["floor_area"] for zone in az_zones.values())
        print(f"   ✅ Zones updated - Total area: {total_area}m² ({len(az_zones)} zones)")
    
    def step3_hvac_scaling(self):
        """Step 3: Scale HVAC to 995kW VRF System"""
        print("\n❄️  Step 3: Scaling HVAC to 995kW VRF system...")
        
        # Clear existing HVAC objects
        hvac_objects_to_clear = [
            "AirLoopHVAC", "AirLoopHVAC:ControllerList", "AirLoopHVAC:OutdoorAirSystem",
            "Fan:ConstantVolume", "Fan:VariableVolume", "Coil:Cooling:DX:SingleSpeed",
            "Coil:Heating:Gas", "Controller:OutdoorAir", "OutdoorAir:Mixer"
        ]
        
        for hvac_obj in hvac_objects_to_clear:
            if hvac_obj in self.building_data:
                self.building_data[hvac_obj] = {}
        
        # Add VRF Outdoor Unit
        self.building_data["AirConditioner:VariableRefrigerantFlow"] = {
            "AZ_VRF_Outdoor_Unit": {
                "heat_pump_name": "AZ_VRF_System",
                "zone_terminal_unit_list_name": "AZ_VRF_Terminal_List",
                "refrigerant_type": "R410A",
                "rated_total_cooling_capacity": 995000,
                "rated_total_heating_capacity": 800000,
                "cooling_capacity_ratio_modifier_function_of_low_temperature_curve_name": "VRF_Cool_Cap_FT_Low",
                "heating_capacity_ratio_modifier_function_of_low_temperature_curve_name": "VRF_Heat_Cap_FT_Low",
                "equivalent_piping_length_used_for_piping_correction_factor_in_cooling_mode": 30,
                "vertical_height_used_for_piping_correction_factor": 6,
                "crankcase_heater_power_per_compressor": 33,
                "number_of_compressors": 4,
                "defrost_strategy": "ReverseCycle",
                "defrost_control": "Timed"
            }
        }
        
        print("   ✅ VRF system updated - 995kW capacity, R410A refrigerant, 4 compressors")
    
    def step4_uk_schedules(self):
        """Step 4: Add UK Office Schedules"""
        print("\n📅 Step 4: Adding UK office schedules...")
        
        if "Schedule:Compact" not in self.building_data:
            self.building_data["Schedule:Compact"] = {}
        
        # Add UK Office Occupancy Schedule
        self.building_data["Schedule:Compact"]["UK_Office_Occupancy"] = {
            "schedule_type_limits_name": "Fraction",
            "field_1": "Through: 12/31",
            "field_2": "For: Weekdays",
            "field_3": "Until: 07:00", "field_4": "0.0",
            "field_5": "Until: 08:00", "field_6": "0.2",
            "field_7": "Until: 09:00", "field_8": "0.7",
            "field_9": "Until: 12:00", "field_10": "1.0",
            "field_11": "Until: 13:00", "field_12": "0.6",
            "field_13": "Until: 17:00", "field_14": "1.0",
            "field_15": "Until: 18:00", "field_16": "0.5",
            "field_17": "Until: 24:00", "field_18": "0.0",
            "field_19": "For: Saturday",
            "field_20": "Until: 24:00", "field_21": "0.1",
            "field_22": "For: Sunday Holidays",
            "field_23": "Until: 24:00", "field_24": "0.0"
        }
        
        # Add HVAC Operation Schedule
        self.building_data["Schedule:Compact"]["UK_HVAC_Operation"] = {
            "schedule_type_limits_name": "Fraction",
            "field_1": "Through: 12/31",
            "field_2": "For: Weekdays",
            "field_3": "Until: 06:00", "field_4": "0.0",
            "field_5": "Until: 07:00", "field_6": "0.5",
            "field_7": "Until: 19:00", "field_8": "1.0",
            "field_9": "Until: 21:00", "field_10": "0.3",
            "field_11": "Until: 24:00", "field_12": "0.0",
            "field_13": "For: Saturday",
            "field_14": "Until: 24:00", "field_15": "0.1",
            "field_16": "For: Sunday Holidays",
            "field_17": "Until: 24:00", "field_18": "0.0"
        }
        
        print("   ✅ UK schedules added - 8AM-6PM office hours, pre-conditioning HVAC")
    
    def step5_location_update(self):
        """Step 5: Update Location to Macclesfield, UK"""
        print("\n🌍 Step 5: Updating location to Macclesfield, UK...")
        
        # Update Site Location
        self.building_data["Site:Location"] = {
            "Macclesfield_UK": {
                "latitude": 53.26,
                "longitude": -2.12,
                "time_zone": 0.0,
                "elevation": 118
            }
        }
        
        # Add UK Design Days
        self.building_data["SizingPeriod:DesignDay"] = {
            "Macclesfield_Summer_Design_Day": {
                "month": 7,
                "day_of_month": 21,
                "day_type": "SummerDesignDay",
                "maximum_dry_bulb_temperature": 28.0,
                "daily_dry_bulb_temperature_range": 8.0,
                "humidity_condition_type": "WetBulb",
                "wetbulb_or_dewpoint_at_maximum_dry_bulb": 19.5,
                "barometric_pressure": 101325,
                "wind_speed": 3.5,
                "wind_direction": 230,
                "solar_model_indicator": "ASHRAEClearSky"
            },
            "Macclesfield_Winter_Design_Day": {
                "month": 1,
                "day_of_month": 21,
                "day_type": "WinterDesignDay",
                "maximum_dry_bulb_temperature": -3.0,
                "daily_dry_bulb_temperature_range": 0.0,
                "humidity_condition_type": "WetBulb",
                "wetbulb_or_dewpoint_at_maximum_dry_bulb": -4.0,
                "barometric_pressure": 101325,
                "wind_speed": 5.0,
                "wind_direction": 270,
                "solar_model_indicator": "ASHRAEClearSky"
            }
        }
        
        print("   ✅ Location updated - 53.26°N, 2.12°W, GMT timezone, UK design conditions")
    
    def save_building_file(self):
        """Save the final building file"""
        print(f"\n💾 Saving final building file: {self.output_file}")
        try:
            with open(self.output_file, 'w') as f:
                json.dump(self.building_data, f, indent=2)
            
            # Get file size
            file_size = Path(self.output_file).stat().st_size / 1024  # KB
            num_sections = len(self.building_data)
            
            print(f"   ✅ File saved successfully!")
            print(f"   📊 File size: {file_size:.1f} KB")
            print(f"   📋 Sections: {num_sections}")
            return True
        except Exception as e:
            print(f"   ❌ Error saving file: {e}")
            return False
    
    def validate_building(self):
        """Basic validation of the building file"""
        print("\n🔍 Validating building file...")
        
        required_sections = [
            "Version", "Building", "Zone", "Site:Location", 
            "Schedule:Compact", "SizingPeriod:DesignDay"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in self.building_data or not self.building_data[section]:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"   ⚠️  Missing sections: {missing_sections}")
        else:
            print("   ✅ All required sections present")
        
        # Check zone count and total area
        if "Zone" in self.building_data:
            zones = self.building_data["Zone"]
            total_area = sum(zone.get("floor_area", 0) for zone in zones.values())
            print(f"   📐 Total floor area: {total_area}m² (target: 6,700m²)")
            print(f"   🏢 Number of zones: {len(zones)}")
        
        return len(missing_sections) == 0
    
    def convert(self):
        """Run the complete conversion process"""
        print("🚀 Starting AstraZeneca Building Conversion")
        print("=" * 60)
        
        # Load file
        if not self.load_building_file():
            return False
        
        # Run all conversion steps
        self.step1_building_identity()
        self.step2_zone_scaling()
        self.step3_hvac_scaling()
        self.step4_uk_schedules()
        self.step5_location_update()
        
        # Validate and save
        self.validate_building()
        success = self.save_building_file()
        
        if success:
            print("\n🎉 CONVERSION COMPLETE!")
            print("=" * 60)
            print(f"📁 Output file: {self.output_file}")
            print("📋 Ready for Sinergym training with:")
            print("   • AstraZeneca building specifications (6,700m²)")
            print("   • 995kW VRF HVAC system")
            print("   • UK office schedules and location")
            print("   • Macclesfield weather compatibility")
            print("\n🎯 Next steps:")
            print("   1. Copy to Sinergym buildings directory")
            print("   2. Update environment config")
            print("   3. Start training!")
        
        return success

def main():
    """Main function to run the conversion"""
    import sys
    
    # Default file paths
    input_file = "astrazeneca_base.epJSON"  # Copy of working template
    output_file = "astrazeneca_macclesfield.epJSON"  # Final output
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Run conversion
    converter = AstraZenecaBuildingConverter(input_file, output_file)
    success = converter.convert()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()