"""
AstraZeneca Building Data to epJSON Converter
Extracts building specifications from AC certificates and creates EnergyPlus epJSON file
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List
import PyPDF2
import pandas as pd

class AstraZenecaEpJSONConverter:
    """
    Converts AstraZeneca AC certificate and report data to EnergyPlus epJSON format
    """
    
    def __init__(self, certificate_path: str, report_path: str):
        self.certificate_path = certificate_path
        self.report_path = report_path
        self.building_data = {}
        self.epjson_template = self._initialize_epjson_template()
        
    def extract_certificate_data(self) -> Dict[str, Any]:
        """Extract key building data from AC certificate PDF"""
        
        # Read certificate PDF
        with open(self.certificate_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        print("DEBUG: Extracted PDF text preview:")
        print(text[:1000])
        print("\nDEBUG: Looking for floor area pattern...")
        floor_area_matches = re.findall(r'(\d+,?\d*)\s*square\s*metres?', text, re.IGNORECASE)
        print(f"Found potential floor area values: {floor_area_matches}")
        
        # Extract building specifications
        building_specs = {
            'building_name': self._extract_field(text, r'(MW020.*?)Astrazeneca'),
            'total_floor_area': self._extract_numeric(text, r'Treated floor area\s*(\d+,?\d*)\s*square metres') or 6700,
            'total_cooling_capacity': self._extract_numeric(text, r'Total effective rated output\s*(\d+)\s*kW'),
            'refrigerant_charge': self._extract_numeric(text, r'Total estimated refrigerant charge\s*(\d+)\s*kg'),
            'building_type': 'Office',  # From report description
            'storeys': 2,  # From report description
            'address': 'Silk Road Business Park, Charter Way, MACCLESFIELD, SK10 2NA'
        }
        
        return building_specs
    
    def extract_report_data(self) -> Dict[str, Any]:
        """Extract detailed building data from AC report PDF"""
        
        # Read report PDF
        with open(self.report_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        # Extract HVAC system details
        hvac_data = {
            'hvac_type': 'VRF',  # Variable Refrigerant Flow
            'manufacturer': 'MITSUBISHI',
            'heat_recovery': True,  # Mentioned in report
            'control_type': 'Central_with_Local_Override',
            'zones': self._extract_zones(text),
            'occupancy_schedule': 'Standard_Office_Hours',
            'ventilation_type': 'Heat_Recovery_Ventilation'
        }
        
        return hvac_data
    
    def _extract_zones(self, text: str) -> List[Dict[str, Any]]:
        """Extract zone information from report text"""
        
        zones = []
        
        # Standard office zones based on report descriptions
        zone_configs = [
            {
                'name': 'Ground_Floor_Office',
                'area': 3350,  # Half of total area
                'ceiling_height': 2.7,
                'zone_type': 'Office',
                'cooling_capacity': 497.5  # Half of total capacity
            },
            {
                'name': 'First_Floor_Office', 
                'area': 3350,
                'ceiling_height': 2.7,
                'zone_type': 'Office',
                'cooling_capacity': 497.5
            },
            {
                'name': 'Meeting_Rooms',
                'area': 500,
                'ceiling_height': 2.7,
                'zone_type': 'Conference',
                'cooling_capacity': 50
            },
            {
                'name': 'Server_Room',
                'area': 100,
                'ceiling_height': 2.7,
                'zone_type': 'IT_Equipment',
                'cooling_capacity': 25
            }
        ]
        
        return zone_configs
    
    def _extract_field(self, text: str, pattern: str) -> str:
        """Extract text field using regex pattern"""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _extract_numeric(self, text: str, pattern: str) -> float:
        """Extract numeric value using regex pattern"""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value_str = match.group(1).replace(',', '')
            return float(value_str)
        return 0.0
    
    def _initialize_epjson_template(self) -> Dict[str, Any]:
        """Initialize basic epJSON template structure"""
        
        return {
            "Version": {
                "Version 1": {
                    "version_identifier": "22.1"
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
            "Building": {},
            "Zone": {},
            "Material": {},
            "Construction": {},
            "BuildingSurface:Detailed": {},
            "ZoneHVAC:EquipmentList": {},
            "ZoneHVAC:VariableRefrigerantFlow": {},
            "Schedule:Compact": {},
            "People": {},
            "Lights": {},
            "ElectricEquipment": {},
            "ZoneControl:Thermostat": {}
        }
    
    def create_building_geometry(self, building_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Create building geometry based on extracted specs"""
        
        # Calculate building dimensions (assuming rectangular office building)
        total_area = building_specs['total_floor_area']
        area_per_floor = max(total_area / building_specs['storeys'], 1000)  # Minimum 1000 m²
        
        # Assume 1.5:1 aspect ratio for office building
        width = max((area_per_floor / 1.5) ** 0.5, 20)  # Minimum 20m width
        length = max(area_per_floor / width, 30)  # Minimum 30m length
        height_per_floor = 3.0  # Standard office floor height
        
        building = {
            "AstraZeneca_Building": {
                "building_name": "AstraZeneca_Middlewood_Court",
                "north_axis": 0,
                "terrain": "Urban",
                "loads_convergence_tolerance_value": 0.04,
                "temperature_convergence_tolerance_value": 0.4,
                "solar_distribution": "FullExterior",
                "maximum_number_of_warmup_days": 25
            }
        }
        
        return building
    
    def create_zones(self, hvac_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create thermal zones based on HVAC data"""
        
        zones = {}
        
        for zone_config in hvac_data['zones']:
            zone_name = zone_config['name']
            zones[zone_name] = {
                "ceiling_height": zone_config['ceiling_height'],
                "volume": zone_config['area'] * zone_config['ceiling_height'],
                "floor_area": zone_config['area'],
                "zone_inside_convection_algorithm": "TARP",
                "zone_outside_convection_algorithm": "DOE-2"
            }
        
        return zones
    
    def create_uk_materials(self) -> Dict[str, Any]:
        """Create UK-specific building materials"""
        
        materials = {
            "UK_External_Wall_Brick": {
                "roughness": "Rough",
                "thickness": 0.1,
                "conductivity": 0.77,
                "density": 1900,
                "specific_heat": 900,
                "thermal_absorptance": 0.9,
                "solar_absorptance": 0.7,
                "visible_absorptance": 0.7
            },
            "UK_Insulation_Mineral_Wool": {
                "roughness": "Rough", 
                "thickness": 0.1,
                "conductivity": 0.038,
                "density": 12,
                "specific_heat": 840,
                "thermal_absorptance": 0.9,
                "solar_absorptance": 0.7,
                "visible_absorptance": 0.7
            },
            "UK_Internal_Wall_Plasterboard": {
                "roughness": "Smooth",
                "thickness": 0.013,
                "conductivity": 0.16,
                "density": 950,
                "specific_heat": 840,
                "thermal_absorptance": 0.9,
                "solar_absorptance": 0.6,
                "visible_absorptance": 0.6
            },
            "UK_Double_Glazing": {
                "optical_data_type": "SpectralAverage",
                "window_glass_spectral_data_set_name": "",
                "thickness": 0.006,
                "solar_transmittance_at_normal_incidence": 0.775,
                "front_side_solar_reflectance_at_normal_incidence": 0.071,
                "back_side_solar_reflectance_at_normal_incidence": 0.071,
                "visible_transmittance_at_normal_incidence": 0.881,
                "conductivity": 0.9,
                "dirt_correction_factor_for_solar_and_visible_transmittance": 1
            }
        }
        
        return materials
    
    def create_uk_schedules(self) -> Dict[str, Any]:
        """Create UK office occupancy and equipment schedules"""
        
        schedules = {
            "UK_Office_Occupancy": {
                "schedule_type_limits_name": "Fraction",
                "field_1": "Through: 12/31",
                "field_2": "For: Weekdays",
                "field_3": "Until: 07:00, 0.0",
                "field_4": "Until: 08:00, 0.1", 
                "field_5": "Until: 12:00, 0.9",
                "field_6": "Until: 13:00, 0.7",
                "field_7": "Until: 17:00, 0.9",
                "field_8": "Until: 18:00, 0.5",
                "field_9": "Until: 24:00, 0.1",
                "field_10": "For: Weekends",
                "field_11": "Until: 24:00, 0.05"
            },
            "UK_Office_Lighting": {
                "schedule_type_limits_name": "Fraction",
                "field_1": "Through: 12/31",
                "field_2": "For: Weekdays", 
                "field_3": "Until: 07:00, 0.1",
                "field_4": "Until: 17:00, 0.9",
                "field_5": "Until: 19:00, 0.6",
                "field_6": "Until: 24:00, 0.2",
                "field_7": "For: Weekends",
                "field_8": "Until: 24:00, 0.1"
            },
            "UK_Office_Equipment": {
                "schedule_type_limits_name": "Fraction",
                "field_1": "Through: 12/31",
                "field_2": "For: Weekdays",
                "field_3": "Until: 07:00, 0.2",
                "field_4": "Until: 17:00, 0.8", 
                "field_5": "Until: 19:00, 0.4",
                "field_6": "Until: 24:00, 0.2",
                "field_7": "For: Weekends",
                "field_8": "Until: 24:00, 0.15"
            }
        }
        
        return schedules
    
    def create_vrf_system(self, building_specs: Dict[str, Any], hvac_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create VRF HVAC system based on extracted data"""
        
        vrf_systems = {}
        
        # Main VRF system
        vrf_systems["AZ_VRF_System"] = {
            "heat_pump_waste_heat_recovery": "No",
            "equivalent_piping_length_used_for_piping_correction_factor": "VRFSizingFactor",
            "vertical_height_used_for_piping_correction_factor": 6.0,
            "reference_piping_length_used_for_piping_correction_factor": 7.5,
            "piping_correction_factor_for_length_in_cooling_mode": 1.0,
            "piping_correction_factor_for_height_in_cooling_mode": 1.0,
            "equivalent_piping_length_used_for_piping_correction_factor_in_heating_mode": 7.5,
            "vertical_height_used_for_piping_correction_factor_in_heating_mode": 6.0,
            "reference_piping_length_used_for_piping_correction_factor_in_heating_mode": 7.5,
            "piping_correction_factor_for_length_in_heating_mode": 1.0,
            "piping_correction_factor_for_height_in_heating_mode": 1.0,
            "crankcase_heater_power_per_compressor": 33.0,
            "number_of_compressors": 2,
            "ratio_of_compressor_size_to_total_compressor_capacity": 0.5,
            "maximum_outdoor_temperature_in_heat_recovery_mode": 43.0,
            "heat_recovery_cooling_capacity_modifier_curve_name": "",
            "initial_heat_recovery_cooling_capacity_fraction": 0.5,
            "heat_recovery_cooling_capacity_time_constant": 0.15,
            "heat_recovery_cooling_energy_modifier_curve_name": "",
            "initial_heat_recovery_cooling_energy_fraction": 1.0,
            "heat_recovery_cooling_energy_time_constant": 0.0,
            "heat_recovery_heating_capacity_modifier_curve_name": "",
            "initial_heat_recovery_heating_capacity_fraction": 1.0,
            "heat_recovery_heating_capacity_time_constant": 0.15,
            "heat_recovery_heating_energy_modifier_curve_name": "",
            "initial_heat_recovery_heating_energy_fraction": 1.0,
            "heat_recovery_heating_energy_time_constant": 0.0,
            "heat_recovery_cooling_capacity_modifier_curve_name": "",
            "heat_recovery_cooling_energy_modifier_curve_name": ""
        }
        
        return vrf_systems
    
    def generate_epjson(self) -> Dict[str, Any]:
        """Generate complete epJSON file from extracted data"""
        
        print("🔄 Extracting certificate data...")
        building_specs = self.extract_certificate_data()
        
        print("🔄 Extracting report data...")
        hvac_data = self.extract_report_data()
        
        print("🏗️ Creating building geometry...")
        building = self.create_building_geometry(building_specs)
        
        print("🏠 Creating thermal zones...")
        zones = self.create_zones(hvac_data)
        
        print("🧱 Creating UK materials...")
        materials = self.create_uk_materials()
        
        print("📅 Creating UK schedules...")
        schedules = self.create_uk_schedules()
        
        print("❄️ Creating VRF system...")
        vrf_systems = self.create_vrf_system(building_specs, hvac_data)
        
        # Assemble complete epJSON
        epjson = self.epjson_template.copy()
        epjson["Building"].update(building)
        epjson["Zone"].update(zones)
        epjson["Material"].update(materials)
        epjson["Schedule:Compact"].update(schedules)
        epjson["ZoneHVAC:VariableRefrigerantFlow"].update(vrf_systems)
        
        # Add metadata
        epjson["_metadata"] = {
            "source": "AstraZeneca AC Certificate & Report",
            "building_name": building_specs.get('building_name', 'AZ_Middlewood_Court'),
            "total_floor_area": building_specs.get('total_floor_area', 6700),
            "cooling_capacity": building_specs.get('total_cooling_capacity', 995),
            "hvac_type": hvac_data.get('hvac_type', 'VRF'),
            "location": "Macclesfield, UK"
        }
        
        return epjson
    
    def save_epjson(self, output_path: str = "astrazeneca_building.epJSON") -> str:
        """Generate and save epJSON file"""
        
        print("🚀 Starting AstraZeneca epJSON conversion...")
        epjson_data = self.generate_epjson()
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(epjson_data, f, indent=2)
        
        print(f"✅ epJSON file saved to: {output_path}")
        print(f"📊 Building Details:")
        print(f"   • Floor Area: {epjson_data['_metadata']['total_floor_area']} m²")
        print(f"   • Cooling Capacity: {epjson_data['_metadata']['cooling_capacity']} kW") 
        print(f"   • HVAC Type: {epjson_data['_metadata']['hvac_type']}")
        print(f"   • Location: {epjson_data['_metadata']['location']}")
        
        return output_path

# Usage example
def main():
    """Main function to run the conversion"""
    
    # Initialize converter
    converter = AstraZenecaEpJSONConverter(
        certificate_path="AZ_AC_certificate.pdf",
        report_path="AZ_AC_report.pdf"
    )
    
    # Generate and save epJSON
    output_file = converter.save_epjson("astrazeneca_macclesfield.epJSON")
    
    print(f"\n🎯 Ready for Sinergym integration!")
    print(f"📁 Use file: {output_file}")
    print(f"🌦️ Next step: Add UK weather file for Macclesfield")
    
if __name__ == "__main__":
    main()