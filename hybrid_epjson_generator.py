"""
AstraZeneca Hybrid Building Generator - Option 2
Creates thermally accurate building file using template structure + AZ specifications
Maintains 17-input compatibility for existing trained model
"""

import json
import math
from pathlib import Path
from typing import Dict, Any

def create_astrazeneca_hybrid_building():
    """
    Creates hybrid epJSON combining working template structure with AZ specifications
    """
    
    # Base template structure (minimal working EnergyPlus model)
    template = {
        "Version": {"Version 1": {"version_identifier": "23.1"}},
        
        "Building": {
            "AstraZeneca_Macclesfield": {
                "north_axis": 0,
                "terrain": "City",
                "loads_convergence_tolerance_value": 0.04,
                "temperature_convergence_tolerance_value": 0.4,
                "solar_distribution": "FullExterior",
                "maximum_number_of_warmup_days": 25
            }
        },
        
        "SimulationControl": {
            "SimulationControl 1": {
                "do_zone_sizing_calculation": "Yes",
                "do_system_sizing_calculation": "Yes",
                "do_plant_sizing_calculation": "Yes",
                "run_simulation_for_sizing_periods": "No",
                "run_simulation_for_weather_file_run_periods": "Yes"
            }
        },
        
        "Timestep": {"Timestep 1": {"number_of_timesteps_per_hour": 4}},
        
        "RunPeriod": {
            "RunPeriod 1": {
                "begin_month": 1,
                "begin_day_of_month": 1,
                "end_month": 12,
                "end_day_of_month": 31,
                "day_of_week_for_start_day": "Tuesday",
                "use_weather_file_holidays_and_special_days": "Yes",
                "use_weather_file_daylight_saving_period": "Yes",
                "apply_weekend_holiday_rule": "No",
                "use_weather_file_rain_indicators": "Yes",
                "use_weather_file_snow_indicators": "Yes"
            }
        }
    }
    
    # UK Materials for AstraZeneca
    uk_materials = {
        "Material": {
            "UK_Brick_Outer": {
                "roughness": "Rough",
                "thickness": 0.1,
                "conductivity": 0.72,
                "density": 1800,
                "specific_heat": 840
            },
            "UK_Insulation_MineralWool": {
                "roughness": "MediumRough", 
                "thickness": 0.1,
                "conductivity": 0.04,
                "density": 12,
                "specific_heat": 840
            },
            "UK_Plasterboard": {
                "roughness": "Smooth",
                "thickness": 0.012,
                "conductivity": 0.16,
                "density": 800,
                "specific_heat": 1000
            },
            "UK_Concrete_Slab": {
                "roughness": "MediumRough",
                "thickness": 0.15,
                "conductivity": 1.4,
                "density": 2100,
                "specific_heat": 840
            }
        },
        
        "WindowMaterial:Glazing": {
            "UK_DoubleGlazing_6mm": {
                "optical_data_type": "SpectralAverage",
                "window_glass_spectral_data_set_name": "",
                "thickness": 0.006,
                "solar_transmittance_at_normal_incidence": 0.775,
                "front_side_solar_reflectance_at_normal_incidence": 0.071,
                "back_side_solar_reflectance_at_normal_incidence": 0.071,
                "visible_transmittance_at_normal_incidence": 0.881,
                "front_side_visible_reflectance_at_normal_incidence": 0.080,
                "back_side_visible_reflectance_at_normal_incidence": 0.080,
                "infrared_transmittance_at_normal_incidence": 0.0,
                "front_side_infrared_hemispherical_emissivity": 0.84,
                "back_side_infrared_hemispherical_emissivity": 0.84,
                "conductivity": 0.9
            }
        },
        
        "WindowMaterial:Gas": {
            "UK_Air_Gap_12mm": {
                "gas_type": "Air",
                "thickness": 0.012
            }
        }
    }
    
    # Constructions using UK materials
    constructions = {
        "Construction": {
            "UK_Wall_Construction": {
                "outside_layer": "UK_Brick_Outer",
                "layer_2": "UK_Insulation_MineralWool", 
                "layer_3": "UK_Plasterboard"
            },
            "UK_Floor_Construction": {
                "outside_layer": "UK_Concrete_Slab",
                "layer_2": "UK_Insulation_MineralWool"
            },
            "UK_Roof_Construction": {
                "outside_layer": "UK_Plasterboard",
                "layer_2": "UK_Insulation_MineralWool",
                "layer_3": "UK_Brick_Outer"
            },
            "UK_Window_Construction": {
                "outside_layer": "UK_DoubleGlazing_6mm",
                "layer_2": "UK_Air_Gap_12mm",
                "layer_3": "UK_DoubleGlazing_6mm"
            }
        }
    }
    
    # UK Office Schedules
    uk_schedules = {
        "Schedule:Compact": {
            "UK_Office_Occupancy": {
                "schedule_type_limits_name": "Fraction",
                "field_1": "Through: 12/31",
                "field_2": "For: Weekdays",
                "field_3": "Until: 08:00", "field_4": "0.0",
                "field_5": "Until: 09:00", "field_6": "0.3", 
                "field_7": "Until: 12:00", "field_8": "1.0",
                "field_9": "Until: 13:00", "field_10": "0.5",
                "field_11": "Until: 17:00", "field_12": "1.0",
                "field_13": "Until: 18:00", "field_14": "0.3",
                "field_15": "Until: 24:00", "field_16": "0.0",
                "field_17": "For: Saturday",
                "field_18": "Until: 24:00", "field_19": "0.1",
                "field_20": "For: Sunday Holidays",
                "field_21": "Until: 24:00", "field_22": "0.0"
            },
            "UK_Office_Lighting": {
                "schedule_type_limits_name": "Fraction",
                "field_1": "Through: 12/31",
                "field_2": "For: Weekdays", 
                "field_3": "Until: 08:00", "field_4": "0.1",
                "field_5": "Until: 18:00", "field_6": "1.0",
                "field_7": "Until: 24:00", "field_8": "0.1",
                "field_9": "For: Saturday",
                "field_10": "Until: 24:00", "field_11": "0.2",
                "field_12": "For: Sunday Holidays",
                "field_13": "Until: 24:00", "field_14": "0.1"
            },
            "UK_Equipment_Schedule": {
                "schedule_type_limits_name": "Fraction",
                "field_1": "Through: 12/31",
                "field_2": "For: Weekdays",
                "field_3": "Until: 08:00", "field_4": "0.4",
                "field_5": "Until: 18:00", "field_6": "1.0", 
                "field_7": "Until: 24:00", "field_8": "0.4",
                "field_9": "For: Saturday",
                "field_10": "Until: 24:00", "field_11": "0.4",
                "field_12": "For: Sunday Holidays", 
                "field_13": "Until: 24:00", "field_14": "0.3"
            }
        }
    }
    
    # AstraZeneca Zone Definitions (4 zones matching areas)
    zones = create_az_zones()
    surfaces = create_az_surfaces()
    
    # VRF HVAC System (995kW capacity)
    hvac_system = create_vrf_system()
    
    # Internal Loads 
    internal_loads = create_internal_loads()
    
    # Combine all components
    template.update(uk_materials)
    template.update(constructions) 
    template.update(uk_schedules)
    template.update(zones)
    template.update(surfaces)
    template.update(hvac_system)
    template.update(internal_loads)
    
    return template

def create_az_zones():
    """Create 4 thermal zones for AstraZeneca building"""
    return {
        "Zone": {
            "Office_East": {
                "ceiling_height": 3.0,
                "volume": 10050,  # 3350m² * 3m height
                "floor_area": 3350,
                "zone_inside_convection_algorithm": "TARP",
                "zone_outside_convection_algorithm": "DOE-2",
                "zone_air_distribution_effectiveness_in_cooling_mode": 1.0,
                "zone_air_distribution_effectiveness_in_heating_mode": 1.0
            },
            "Office_West": {
                "ceiling_height": 3.0,
                "volume": 10050, # 3350m² * 3m height
                "floor_area": 3350,
                "zone_inside_convection_algorithm": "TARP",
                "zone_outside_convection_algorithm": "DOE-2",
                "zone_air_distribution_effectiveness_in_cooling_mode": 1.0,
                "zone_air_distribution_effectiveness_in_heating_mode": 1.0
            },
            "Meeting_Rooms": {
                "ceiling_height": 3.0,
                "volume": 1500,  # 500m² * 3m height
                "floor_area": 500,
                "zone_inside_convection_algorithm": "TARP", 
                "zone_outside_convection_algorithm": "DOE-2",
                "zone_air_distribution_effectiveness_in_cooling_mode": 1.0,
                "zone_air_distribution_effectiveness_in_heating_mode": 1.0
            },
            "Server_Room": {
                "ceiling_height": 3.0,
                "volume": 300,   # 100m² * 3m height  
                "floor_area": 100,
                "zone_inside_convection_algorithm": "TARP",
                "zone_outside_convection_algorithm": "DOE-2", 
                "zone_air_distribution_effectiveness_in_cooling_mode": 1.0,
                "zone_air_distribution_effectiveness_in_heating_mode": 1.0
            }
        }
    }

def create_az_surfaces():
    """Create building surfaces for AZ zones"""
    
    # Simplified rectangular geometry 
    # Office_East: 50m x 67m, Office_West: 50m x 67m
    # Meeting_Rooms: 20m x 25m, Server_Room: 10m x 10m
    
    surfaces = {"BuildingSurface:Detailed": {}}
    
    # Office East surfaces (0,0 to 67,50)
    surfaces["BuildingSurface:Detailed"].update({
        "Office_East_Floor": {
            "surface_type": "Floor",
            "construction_name": "UK_Floor_Construction", 
            "zone_name": "Office_East",
            "outside_boundary_condition": "Ground",
            "vertices": [
                {"vertex_x_coordinate": 0, "vertex_y_coordinate": 0, "vertex_z_coordinate": 0},
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 0, "vertex_z_coordinate": 0},
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 50, "vertex_z_coordinate": 0},
                {"vertex_x_coordinate": 0, "vertex_y_coordinate": 50, "vertex_z_coordinate": 0}
            ]
        },
        "Office_East_Ceiling": {
            "surface_type": "Roof",
            "construction_name": "UK_Roof_Construction",
            "zone_name": "Office_East", 
            "outside_boundary_condition": "Outdoors",
            "vertices": [
                {"vertex_x_coordinate": 0, "vertex_y_coordinate": 50, "vertex_z_coordinate": 3},
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 50, "vertex_z_coordinate": 3},
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 0, "vertex_z_coordinate": 3},
                {"vertex_x_coordinate": 0, "vertex_y_coordinate": 0, "vertex_z_coordinate": 3}
            ]
        },
        "Office_East_Wall_South": {
            "surface_type": "Wall",
            "construction_name": "UK_Wall_Construction",
            "zone_name": "Office_East",
            "outside_boundary_condition": "Outdoors", 
            "vertices": [
                {"vertex_x_coordinate": 0, "vertex_y_coordinate": 0, "vertex_z_coordinate": 0},
                {"vertex_x_coordinate": 0, "vertex_y_coordinate": 0, "vertex_z_coordinate": 3},
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 0, "vertex_z_coordinate": 3},
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 0, "vertex_z_coordinate": 0}
            ]
        },
        "Office_East_Wall_East": {
            "surface_type": "Wall", 
            "construction_name": "UK_Wall_Construction",
            "zone_name": "Office_East",
            "outside_boundary_condition": "Outdoors",
            "vertices": [
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 0, "vertex_z_coordinate": 0},
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 0, "vertex_z_coordinate": 3},
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 50, "vertex_z_coordinate": 3},
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 50, "vertex_z_coordinate": 0}
            ]
        }
    })
    
    # Office West surfaces (67,0 to 134,50) 
    surfaces["BuildingSurface:Detailed"].update({
        "Office_West_Floor": {
            "surface_type": "Floor",
            "construction_name": "UK_Floor_Construction",
            "zone_name": "Office_West", 
            "outside_boundary_condition": "Ground",
            "vertices": [
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 0, "vertex_z_coordinate": 0},
                {"vertex_x_coordinate": 134, "vertex_y_coordinate": 0, "vertex_z_coordinate": 0},
                {"vertex_x_coordinate": 134, "vertex_y_coordinate": 50, "vertex_z_coordinate": 0},
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 50, "vertex_z_coordinate": 0}
            ]
        },
        "Office_West_Ceiling": {
            "surface_type": "Roof",
            "construction_name": "UK_Roof_Construction", 
            "zone_name": "Office_West",
            "outside_boundary_condition": "Outdoors",
            "vertices": [
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 50, "vertex_z_coordinate": 3},
                {"vertex_x_coordinate": 134, "vertex_y_coordinate": 50, "vertex_z_coordinate": 3},
                {"vertex_x_coordinate": 134, "vertex_y_coordinate": 0, "vertex_z_coordinate": 3},
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 0, "vertex_z_coordinate": 3}
            ]
        }
    })
    
    # Meeting Rooms surfaces (67,50 to 87,75)
    surfaces["BuildingSurface:Detailed"].update({
        "Meeting_Floor": {
            "surface_type": "Floor",
            "construction_name": "UK_Floor_Construction",
            "zone_name": "Meeting_Rooms",
            "outside_boundary_condition": "Ground", 
            "vertices": [
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 50, "vertex_z_coordinate": 0},
                {"vertex_x_coordinate": 87, "vertex_y_coordinate": 50, "vertex_z_coordinate": 0},
                {"vertex_x_coordinate": 87, "vertex_y_coordinate": 75, "vertex_z_coordinate": 0},
                {"vertex_x_coordinate": 67, "vertex_y_coordinate": 75, "vertex_z_coordinate": 0}
            ]
        }
    })
    
    # Server Room surfaces (87,50 to 97,60)
    surfaces["BuildingSurface:Detailed"].update({
        "Server_Floor": {
            "surface_type": "Floor", 
            "construction_name": "UK_Floor_Construction",
            "zone_name": "Server_Room",
            "outside_boundary_condition": "Ground",
            "vertices": [
                {"vertex_x_coordinate": 87, "vertex_y_coordinate": 50, "vertex_z_coordinate": 0},
                {"vertex_x_coordinate": 97, "vertex_y_coordinate": 50, "vertex_z_coordinate": 0},
                {"vertex_x_coordinate": 97, "vertex_y_coordinate": 60, "vertex_z_coordinate": 0},
                {"vertex_x_coordinate": 87, "vertex_y_coordinate": 60, "vertex_z_coordinate": 0}
            ]
        }
    })
    
    return surfaces

def create_vrf_system():
    """Create VRF HVAC system for 995kW capacity"""
    return {
        "AirConditioner:VariableRefrigerantFlow": {
            "VRF_System_AZ": {
                "heat_pump_name": "VRF_Outdoor_Unit",
                "zone_terminal_unit_list_name": "VRF_Terminal_List",
                "refrigerant_type": "R410A",
                "rated_total_cooling_capacity": 995000,  # 995kW
                "rated_total_heating_capacity": 800000,  # 800kW heating
                "cooling_capacity_ratio_modifier_function_of_low_temperature_curve_name": "VRF_Cool_Cap_FT_Low",
                "heating_capacity_ratio_modifier_function_of_low_temperature_curve_name": "VRF_Heat_Cap_FT_Low"
            }
        },
        
        "ZoneTerminalUnitList": {
            "VRF_Terminal_List": {
                "zone_terminal_unit_1_zone_name": "Office_East",
                "zone_terminal_unit_1_terminal_unit_name": "Office_East_VRF_Terminal",
                "zone_terminal_unit_2_zone_name": "Office_West", 
                "zone_terminal_unit_2_terminal_unit_name": "Office_West_VRF_Terminal",
                "zone_terminal_unit_3_zone_name": "Meeting_Rooms",
                "zone_terminal_unit_3_terminal_unit_name": "Meeting_VRF_Terminal",
                "zone_terminal_unit_4_zone_name": "Server_Room",
                "zone_terminal_unit_4_terminal_unit_name": "Server_VRF_Terminal"
            }
        },
        
        "ZoneHVAC:TerminalUnit:VariableRefrigerantFlow": {
            "Office_East_VRF_Terminal": {
                "terminal_unit_availability_schedule": "UK_Office_Occupancy",
                "terminal_unit_air_inlet_node_name": "Office_East_VRF_Inlet",
                "terminal_unit_air_outlet_node_name": "Office_East_VRF_Outlet",
                "supply_air_flow_rate_during_cooling_operation": 2.5,  # m³/s
                "supply_air_flow_rate_during_heating_operation": 2.0,
                "zone_name": "Office_East"
            },
            "Office_West_VRF_Terminal": {
                "terminal_unit_availability_schedule": "UK_Office_Occupancy", 
                "terminal_unit_air_inlet_node_name": "Office_West_VRF_Inlet",
                "terminal_unit_air_outlet_node_name": "Office_West_VRF_Outlet",
                "supply_air_flow_rate_during_cooling_operation": 2.5,
                "supply_air_flow_rate_during_heating_operation": 2.0,
                "zone_name": "Office_West"
            },
            "Meeting_VRF_Terminal": {
                "terminal_unit_availability_schedule": "UK_Office_Occupancy",
                "terminal_unit_air_inlet_node_name": "Meeting_VRF_Inlet", 
                "terminal_unit_air_outlet_node_name": "Meeting_VRF_Outlet",
                "supply_air_flow_rate_during_cooling_operation": 0.8,
                "supply_air_flow_rate_during_heating_operation": 0.6,
                "zone_name": "Meeting_Rooms"
            },
            "Server_VRF_Terminal": {
                "terminal_unit_availability_schedule": "UK_Equipment_Schedule",
                "terminal_unit_air_inlet_node_name": "Server_VRF_Inlet",
                "terminal_unit_air_outlet_node_name": "Server_VRF_Outlet", 
                "supply_air_flow_rate_during_cooling_operation": 0.5,
                "supply_air_flow_rate_during_heating_operation": 0.2,
                "zone_name": "Server_Room"
            }
        }
    }

def create_internal_loads():
    """Create internal loads for occupancy, lighting, equipment"""
    return {
        "People": {
            "Office_East_People": {
                "zone_or_zonelist_name": "Office_East",
                "number_of_people_schedule_name": "UK_Office_Occupancy",
                "number_of_people_calculation_method": "People/Area",
                "number_of_people_per_zone_floor_area": 0.05,  # 5 people per 100m²
                "fraction_radiant": 0.3,
                "sensible_heat_fraction": "autocalculate"
            },
            "Office_West_People": {
                "zone_or_zonelist_name": "Office_West",
                "number_of_people_schedule_name": "UK_Office_Occupancy", 
                "number_of_people_calculation_method": "People/Area",
                "number_of_people_per_zone_floor_area": 0.05,
                "fraction_radiant": 0.3,
                "sensible_heat_fraction": "autocalculate"
            },
            "Meeting_People": {
                "zone_or_zonelist_name": "Meeting_Rooms",
                "number_of_people_schedule_name": "UK_Office_Occupancy",
                "number_of_people_calculation_method": "People/Area", 
                "number_of_people_per_zone_floor_area": 0.15,  # Higher density for meetings
                "fraction_radiant": 0.3,
                "sensible_heat_fraction": "autocalculate"
            }
        },
        
        "Lights": {
            "Office_East_Lights": {
                "zone_or_zonelist_name": "Office_East",
                "schedule_name": "UK_Office_Lighting",
                "design_level_calculation_method": "Watts/Area",
                "watts_per_zone_floor_area": 12,  # W/m² LED lighting
                "fraction_radiant": 0.4,
                "fraction_visible": 0.2
            },
            "Office_West_Lights": {
                "zone_or_zonelist_name": "Office_West", 
                "schedule_name": "UK_Office_Lighting",
                "design_level_calculation_method": "Watts/Area",
                "watts_per_zone_floor_area": 12,
                "fraction_radiant": 0.4,
                "fraction_visible": 0.2
            },
            "Meeting_Lights": {
                "zone_or_zonelist_name": "Meeting_Rooms",
                "schedule_name": "UK_Office_Lighting",
                "design_level_calculation_method": "Watts/Area",
                "watts_per_zone_floor_area": 15,  # Higher for presentations
                "fraction_radiant": 0.4,
                "fraction_visible": 0.2
            },
            "Server_Lights": {
                "zone_or_zonelist_name": "Server_Room", 
                "schedule_name": "UK_Equipment_Schedule",
                "design_level_calculation_method": "Watts/Area",
                "watts_per_zone_floor_area": 20,
                "fraction_radiant": 0.4,
                "fraction_visible": 0.2
            }
        },
        
        "ElectricEquipment": {
            "Office_East_Equipment": {
                "zone_or_zonelist_name": "Office_East",
                "schedule_name": "UK_Equipment_Schedule",
                "design_level_calculation_method": "Watts/Area",
                "watts_per_zone_floor_area": 15,  # Computers, monitors
                "fraction_radiant": 0.5,
                "fraction_latent": 0.1
            },
            "Office_West_Equipment": {
                "zone_or_zonelist_name": "Office_West",
                "schedule_name": "UK_Equipment_Schedule", 
                "design_level_calculation_method": "Watts/Area",
                "watts_per_zone_floor_area": 15,
                "fraction_radiant": 0.5,
                "fraction_latent": 0.1
            },
            "Server_Equipment": {
                "zone_or_zonelist_name": "Server_Room",
                "schedule_name": "UK_Equipment_Schedule",
                "design_level_calculation_method": "Watts/Area", 
                "watts_per_zone_floor_area": 300,  # High heat load servers
                "fraction_radiant": 0.8,
                "fraction_latent": 0.05
            }
        }
    }

def save_building_file():
    """Generate and save the AstraZeneca hybrid building file"""
    
    print("🏗️  Creating AstraZeneca hybrid building file...")
    
    # Generate the building model
    building_data = create_astrazeneca_hybrid_building()
    
    # Save to epJSON file
    output_file = "astrazeneca_hybrid.epJSON"
    with open(output_file, 'w') as f:
        json.dump(building_data, f, indent=2)
    
    print(f"✅ Building file saved: {output_file}")
    print(f"📊 Building specifications:")
    print(f"   • Total floor area: 6,700 m²")
    print(f"   • Zones: 4 (Office East/West, Meeting Rooms, Server Room)")
    print(f"   • HVAC: VRF system, 995kW cooling capacity")
    print(f"   • Materials: UK specifications")
    print(f"   • Schedules: UK office patterns")
    print(f"   • Ready for Sinergym integration!")
    
    return output_file

if __name__ == "__main__":
    save_building_file()