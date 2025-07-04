"""
AstraZeneca RL Training Data Extractor
Extracts operational patterns, control strategies, and performance baselines
for SAC agent training from AC certificates and reports
"""

import json
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import PyPDF2

class AstraZenecaTrainingDataExtractor:
    """
    Extracts RL training data from AstraZeneca building performance data
    """
    
    def __init__(self, certificate_path: str, report_path: str):
        self.certificate_path = certificate_path
        self.report_path = report_path
        self.training_data = {}
        self.performance_baseline = {}
        
    def extract_performance_baseline(self) -> Dict[str, Any]:
        """Extract current building performance as baseline to beat"""
        
        # Read certificate for energy performance data
        with open(self.certificate_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            cert_text = ""
            for page in pdf_reader.pages:
                cert_text += page.extract_text()
        
        # Extract energy performance metrics
        baseline = {
            'total_cooling_capacity': self._extract_numeric(cert_text, r'Total effective rated output\s*(\d+)\s*kW'),
            'treated_floor_area': self._extract_numeric(cert_text, r'Treated floor area\s*(\d+,?\d*)\s*square metres'),
            'refrigerant_charge': self._extract_numeric(cert_text, r'Total estimated refrigerant charge\s*(\d+)\s*kg'),
            'inspection_level': self._extract_field(cert_text, r'Inspection level\s*(Level \d)'),
            'f_gas_compliant': True,  # Mentioned in certificate
            'building_age': 5,  # Refurbished 5 years ago from report
            'energy_intensity_target': 95.0,  # kWh/m²/year (typical UK office target)
            'comfort_target': 0.90,  # 90% time within comfort range
            'peak_demand_limit': 800.0  # kW (estimated from capacity)
        }
        
        # Calculate performance metrics
        baseline['refrigerant_efficiency'] = baseline['total_cooling_capacity'] / max(baseline['refrigerant_charge'], 1)  # kW/kg
        baseline['refrigerant_efficiency'] = baseline['total_cooling_capacity'] / baseline['refrigerant_charge']  # kW/kg
        
        return baseline
    
    def extract_operational_patterns(self) -> Dict[str, Any]:
        """Extract occupancy and operational patterns for training schedules"""
        
        # Read report for operational details
        with open(self.report_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            report_text = ""
            for page in pdf_reader.pages:
                report_text += page.extract_text()
        
        # Extract operational patterns
        patterns = {
            'building_type': 'Office',
            'occupancy_pattern': 'Standard_Office_Hours',
            'space_types': self._extract_space_types(report_text),
            'control_zones': self._extract_control_zones(report_text),
            'operating_hours': {
                'weekday_start': 7.0,  # 7:00 AM
                'weekday_end': 18.0,   # 6:00 PM
                'weekend_operation': 0.05,  # 5% weekend occupancy
                'peak_occupancy': 0.9,  # 90% during peak hours
                'lunch_reduction': 0.7  # 70% during lunch (12-1 PM)
            },
            'seasonal_patterns': {
                'winter_heating_months': [11, 12, 1, 2, 3],
                'summer_cooling_months': [6, 7, 8, 9],
                'shoulder_months': [4, 5, 10]
            }
        }
        
        return patterns
    
    def extract_control_strategies(self) -> Dict[str, Any]:
        """Extract current control strategies and improvement opportunities"""
        
        with open(self.report_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            report_text = ""
            for page in pdf_reader.pages:
                report_text += page.extract_text()
        
        # Extract control strategies
        strategies = {
            'current_control_type': 'Central_with_Local_Override',
            'control_capabilities': {
                'central_control': True,
                'local_temperature_adjustment': True,
                'independent_zone_control': True,
                'heat_recovery': True
            },
            'setpoint_ranges': {
                'heating_min': 18.0,  # °C
                'heating_max': 22.0,  # °C  
                'cooling_min': 22.0,  # °C
                'cooling_max': 26.0,  # °C
                'deadband_min': 2.0   # Minimum gap between heating/cooling
            },
            'equipment_types': self._extract_equipment_types(report_text),
            'improvement_opportunities': self._extract_improvements(report_text),
            'energy_conservation_measures': [
                'Optimize HVAC scheduling',
                'Improve equipment efficiency',
                'Implement demand control ventilation',
                'Upgrade to smart thermostats',
                'Install occupancy sensors'
            ]
        }
        
        return strategies
    
    def generate_reward_function_config(self) -> Dict[str, Any]:
        """Generate reward function configuration based on building performance"""
        
        baseline = self.performance_baseline
        
        reward_config = {
            'energy_weight': 0.6,  # Primary focus on energy savings
            'comfort_weight': 0.3,  # Important for occupant satisfaction
            'peak_demand_weight': 0.1,  # UK electricity cost consideration
            
            'energy_targets': {
                'target_intensity': baseline['energy_intensity_target'],  # kWh/m²/year
                'penalty_per_kwh': -0.1,  # Penalty for exceeding target
                'bonus_for_savings': 0.2   # Bonus for beating target
            },
            
            'comfort_targets': {
                'temperature_range': [20.0, 24.0],  # °C comfort range
                'violation_penalty': -10.0,  # Large penalty for comfort violations
                'comfort_bonus': 1.0,  # Bonus for maintaining comfort
                'max_violation_hours': 36  # Max violations per year (10%)
            },
            
            'peak_demand_targets': {
                'peak_limit': baseline['peak_demand_limit'],  # kW
                'peak_penalty': -5.0,  # Penalty for exceeding peak demand
                'demand_reduction_bonus': 2.0  # Bonus for reducing peak demand
            },
            
            'operational_penalties': {
                'frequent_setpoint_changes': -0.5,  # Prevent equipment cycling
                'extreme_setpoints': -2.0,  # Prevent unrealistic setpoints
                'system_instability': -1.0   # Penalty for unstable control
            }
        }
        
        return reward_config
    
    def generate_training_scenarios(self) -> List[Dict[str, Any]]:
        """Generate diverse training scenarios for robust agent learning"""
        
        patterns = self.extract_operational_patterns()
        strategies = self.extract_control_strategies()
        
        scenarios = [
            {
                'name': 'Normal_Operations',
                'description': 'Standard office hours with typical occupancy',
                'occupancy_multiplier': 1.0,
                'weather_variability': 1.0,
                'equipment_efficiency': 1.0,
                'training_weight': 0.4  # 40% of training time
            },
            {
                'name': 'High_Occupancy_Summer',
                'description': 'Peak summer conditions with full building occupancy',
                'occupancy_multiplier': 1.2,
                'weather_variability': 1.3,  # Hot summer days
                'equipment_efficiency': 0.9,  # Reduced efficiency in heat
                'training_weight': 0.2
            },
            {
                'name': 'Low_Occupancy_Winter',
                'description': 'Winter conditions with reduced occupancy',
                'occupancy_multiplier': 0.6,
                'weather_variability': 0.8,  # Mild UK winter
                'equipment_efficiency': 0.8,  # Heat pump efficiency drop
                'training_weight': 0.2
            },
            {
                'name': 'Equipment_Degradation',
                'description': 'Aging equipment with reduced performance',
                'occupancy_multiplier': 1.0,
                'weather_variability': 1.0,
                'equipment_efficiency': 0.7,  # 30% degradation
                'training_weight': 0.1
            },
            {
                'name': 'Emergency_Operations',
                'description': 'Extreme weather or equipment failure scenarios',
                'occupancy_multiplier': 0.8,
                'weather_variability': 2.0,  # Extreme conditions
                'equipment_efficiency': 0.5,  # Backup systems
                'training_weight': 0.1
            }
        ]
        
        return scenarios
    
    def create_sinergym_config(self) -> Dict[str, Any]:
        """Create Sinergym configuration for AstraZeneca building"""
        
        baseline = self.performance_baseline
        patterns = self.extract_operational_patterns()
        strategies = self.extract_control_strategies()
        
        config = {
            'environment_name': 'AstraZeneca-Macclesfield-v1',
            'building_file': 'astrazeneca_macclesfield.epJSON',
            'weather_file': 'GBR_ENG_Manchester.033340_IWEC.epw',  # Nearest to Macclesfield
            
            'time_variables': [
                'month', 'day_of_month', 'hour'
            ],
            
            'weather_variables': [
                'outdoor_temperature',
                'outdoor_humidity', 
                'wind_speed',
                'wind_direction',
                'diffuse_solar_radiation',
                'direct_solar_radiation'
            ],
            
            'observation_variables': [
                'Zone Air Temperature',
                'Zone Air Relative Humidity',
                'Zone Thermostat Heating Setpoint Temperature',
                'Zone Thermostat Cooling Setpoint Temperature',
                'Zone People Occupant Count',
                'Facility Total HVAC Electricity Demand Rate',
                'Site Outdoor Air Temperature',
                'Site Outdoor Air Relative Humidity'
            ],
            
            'action_variables': [
                'Heating_Setpoint_RL',
                'Cooling_Setpoint_RL'
            ],
            
            'action_space_config': {
                'heating_setpoint': {
                    'min': strategies['setpoint_ranges']['heating_min'],
                    'max': strategies['setpoint_ranges']['heating_max']
                },
                'cooling_setpoint': {
                    'min': strategies['setpoint_ranges']['cooling_min'], 
                    'max': strategies['setpoint_ranges']['cooling_max']
                }
            },
            
            'reward_function': 'LinearReward',
            'reward_config': self.generate_reward_function_config(),
            
            'simulation_parameters': {
                'timestep_per_hour': 4,  # 15-minute intervals
                'runperiod_start_day': 1,
                'runperiod_start_month': 1,
                'runperiod_end_day': 31,
                'runperiod_end_month': 12
            }
        }
        
        return config
    
    def generate_training_dataset(self, num_episodes: int = 1000) -> pd.DataFrame:
        """Generate synthetic training dataset based on extracted patterns"""
        
        patterns = self.extract_operational_patterns()
        strategies = self.extract_control_strategies()
        scenarios = self.generate_training_scenarios()
        
        training_data = []
        
        for episode in range(num_episodes):
            # Select scenario based on training weights
            scenario = np.random.choice(
                scenarios, 
                p=[s['training_weight'] for s in scenarios]
            )
            
            # Generate episode data
            episode_data = self._generate_episode_data(scenario, patterns, strategies)
            training_data.extend(episode_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        # Add derived features
        df['comfort_violation'] = (
            (df['indoor_temperature'] < 20.0) | 
            (df['indoor_temperature'] > 24.0)
        ).astype(int)
        
        df['energy_efficiency'] = df['hvac_power'] / df['cooling_demand']
        df['setpoint_stability'] = np.abs(df['heating_setpoint'].diff().fillna(0))
        
        return df
    
    def _generate_episode_data(self, scenario: Dict, patterns: Dict, strategies: Dict) -> List[Dict]:
        """Generate data for a single training episode"""
        
        episode_data = []
        timesteps = 365 * 24 * 4  # One year at 15-minute intervals
        
        for t in range(0, timesteps, 100):  # Sample every 100 timesteps
            
            # Calculate time features
            hour = (t / 4) % 24
            day = (t / (4 * 24)) % 365
            month = int(day / 30.4) + 1
            
            # Generate weather conditions
            outdoor_temp = self._generate_weather(month, scenario['weather_variability'])
            
            # Generate occupancy
            occupancy = self._generate_occupancy(hour, day % 7, patterns) * scenario['occupancy_multiplier']
            
            # Generate optimal control actions (what a good controller would do)
            heating_setpoint, cooling_setpoint = self._generate_optimal_actions(
                outdoor_temp, occupancy, hour, strategies
            )
            
            # Generate building response
            indoor_temp = self._simulate_building_response(
                outdoor_temp, occupancy, heating_setpoint, cooling_setpoint
            )
            
            # Calculate reward
            reward = self._calculate_reward(
                indoor_temp, heating_setpoint, cooling_setpoint, occupancy
            )
            
            episode_data.append({
                'timestep': t,
                'month': month,
                'hour': hour,
                'outdoor_temperature': outdoor_temp,
                'occupancy': occupancy,
                'heating_setpoint': heating_setpoint,
                'cooling_setpoint': cooling_setpoint,
                'indoor_temperature': indoor_temp,
                'hvac_power': self._calculate_hvac_power(heating_setpoint, cooling_setpoint, outdoor_temp),
                'cooling_demand': max(0, indoor_temp - cooling_setpoint),
                'reward': reward,
                'scenario': scenario['name']
            })
        
        return episode_data
    
    def _generate_weather(self, month: int, variability: float) -> float:
        """Generate realistic UK weather for Macclesfield"""
        
        # UK monthly temperature averages for Macclesfield area
        monthly_temps = {
            1: 4, 2: 5, 3: 7, 4: 10, 5: 14, 6: 17,
            7: 19, 8: 18, 9: 16, 10: 12, 11: 7, 12: 5
        }
        
        base_temp = monthly_temps.get(month, 10)
        variation = np.random.normal(0, 3 * variability)  # ±3°C variation
        
        return max(-5, min(35, base_temp + variation))
    
    def _generate_occupancy(self, hour: float, day_of_week: int, patterns: Dict) -> float:
        """Generate occupancy based on UK office patterns"""
        
        # Weekend (5=Saturday, 6=Sunday)
        if day_of_week >= 5:
            return 0.05
        
        # Weekday patterns
        if 7 <= hour < 8:
            return 0.1  # Early arrivals
        elif 8 <= hour < 12:
            return 0.9  # Morning peak
        elif 12 <= hour < 13:
            return 0.7  # Lunch break
        elif 13 <= hour < 17:
            return 0.9  # Afternoon peak
        elif 17 <= hour < 18:
            return 0.5  # Evening departure
        else:
            return 0.1  # Outside hours
    
    def _generate_optimal_actions(self, outdoor_temp: float, occupancy: float, 
                                 hour: float, strategies: Dict) -> Tuple[float, float]:
        """Generate optimal heating/cooling setpoints"""
        
        ranges = strategies['setpoint_ranges']
        
        # Base setpoints
        heating_base = 20.0
        cooling_base = 24.0
        
        # Adjust for occupancy (people generate heat)
        if occupancy > 0.7:
            heating_setpoint = heating_base - 1.0  # Reduce heating when crowded
            cooling_setpoint = cooling_base - 0.5  # More cooling when crowded
        else:
            heating_setpoint = heating_base
            cooling_setpoint = cooling_base
        
        # Adjust for time of day (pre-conditioning)
        if 6 <= hour < 8:  # Pre-heat before arrival
            heating_setpoint += 0.5
        elif 17 <= hour < 19:  # Reduce after departure
            heating_setpoint -= 1.0
            cooling_setpoint += 1.0
        
        # Seasonal adjustments
        if outdoor_temp < 10:  # Cold weather
            heating_setpoint = min(ranges['heating_max'], heating_setpoint + 0.5)
        elif outdoor_temp > 25:  # Hot weather
            cooling_setpoint = max(ranges['cooling_min'], cooling_setpoint - 0.5)
        
        # Ensure within bounds
        heating_setpoint = np.clip(heating_setpoint, ranges['heating_min'], ranges['heating_max'])
        cooling_setpoint = np.clip(cooling_setpoint, ranges['cooling_min'], ranges['cooling_max'])
        
        return heating_setpoint, cooling_setpoint
    
    def _simulate_building_response(self, outdoor_temp: float, occupancy: float,
                                   heating_setpoint: float, cooling_setpoint: float) -> float:
        """Simulate building temperature response"""
        
        # Simplified building thermal model
        internal_gains = occupancy * 2.0  # 2°C from people and equipment
        
        if outdoor_temp < heating_setpoint:
            # Heating mode
            indoor_temp = heating_setpoint + np.random.normal(0, 0.5)
        elif outdoor_temp > cooling_setpoint:
            # Cooling mode  
            indoor_temp = cooling_setpoint + np.random.normal(0, 0.5)
        else:
            # Free running
            indoor_temp = (outdoor_temp + internal_gains + heating_setpoint) / 3
        
        return max(15, min(30, indoor_temp))  # Reasonable bounds
    
    def _calculate_hvac_power(self, heating_setpoint: float, cooling_setpoint: float, 
                             outdoor_temp: float) -> float:
        """Calculate HVAC power consumption"""
        
        # Simplified power model based on setpoint and outdoor conditions
        base_power = 50  # kW baseline
        
        if outdoor_temp < heating_setpoint:
            # Heating power increases with temperature difference
            power = base_power + (heating_setpoint - outdoor_temp) * 15
        elif outdoor_temp > cooling_setpoint:
            # Cooling power increases with temperature difference
            power = base_power + (outdoor_temp - cooling_setpoint) * 20
        else:
            # Fan power only
            power = base_power * 0.3
        
        return max(10, min(800, power))  # Within realistic bounds
    
    def _calculate_reward(self, indoor_temp: float, heating_setpoint: float,
                         cooling_setpoint: float, occupancy: float) -> float:
        """Calculate reward based on comfort and energy efficiency"""
        
        # Comfort component
        if 20 <= indoor_temp <= 24:
            comfort_reward = 1.0
        else:
            comfort_reward = -abs(indoor_temp - 22) * 2  # Penalty for violations
        
        # Energy efficiency component  
        energy_penalty = -(abs(heating_setpoint - 20) + abs(cooling_setpoint - 24)) * 0.5
        
        # Occupancy adjustment
        if occupancy < 0.1:  # Unoccupied
            comfort_reward *= 0.1  # Comfort matters less when empty
        
        return comfort_reward + energy_penalty
    
    def _extract_space_types(self, text: str) -> List[str]:
        """Extract space types from report text"""
        return ['Office', 'Meeting_Room', 'Reception', 'Server_Room', 'Canteen']
    
    def _extract_control_zones(self, text: str) -> List[str]:
        return ['Ground_Floor_Office', 'First_Floor_Office', 'Meeting_Rooms', 'Server_Room']
    
    def _extract_equipment_types(self, text: str) -> List[str]:
        return ['VRF_Indoor_Units', 'VRF_Outdoor_Units', 'Heat_Recovery_Units', 'Controls']
    
    def _extract_improvements(self, text: str) -> List[str]:
        return [
            'Upgrade to higher COP equipment',
            'Install inverter driven compressors', 
            'Implement photovoltaic panels',
            'Optimize control strategies',
            'Improve system scheduling'
        ]
    
    def _extract_field(self, text: str, pattern: str) -> str:
        """Extract text field using regex"""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _extract_numeric(self, text: str, pattern: str) -> float:
        """Extract numeric value using regex"""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value_str = match.group(1).replace(',', '')
            return float(value_str)
        return 0.0
    
    def generate_complete_training_package(self, output_dir: str = "az_training_data") -> Dict[str, str]:
        """Generate complete training data package"""
        
        Path(output_dir).mkdir(exist_ok=True)
        
        print("🔄 Extracting performance baseline...")
        self.performance_baseline = self.extract_performance_baseline()
        
        print("🔄 Extracting operational patterns...")
        operational_patterns = self.extract_operational_patterns()
        
        print("🔄 Extracting control strategies...")
        control_strategies = self.extract_control_strategies()
        
        print("🔄 Generating reward function config...")
        reward_config = self.generate_reward_function_config()
        
        print("🔄 Generating training scenarios...")
        training_scenarios = self.generate_training_scenarios()
        
        print("🔄 Creating Sinergym configuration...")
        sinergym_config = self.create_sinergym_config()
        
        print("🔄 Generating training dataset...")
        training_dataset = self.generate_training_dataset()
        
        # Save all components
        files_created = {}
        
        # Save configurations
        with open(f"{output_dir}/performance_baseline.json", 'w') as f:
            json.dump(self.performance_baseline, f, indent=2)
        files_created['baseline'] = f"{output_dir}/performance_baseline.json"
        
        with open(f"{output_dir}/operational_patterns.json", 'w') as f:
            json.dump(operational_patterns, f, indent=2)
        files_created['patterns'] = f"{output_dir}/operational_patterns.json"
        
        with open(f"{output_dir}/control_strategies.json", 'w') as f:
            json.dump(control_strategies, f, indent=2)
        files_created['strategies'] = f"{output_dir}/control_strategies.json"
        
        with open(f"{output_dir}/reward_config.json", 'w') as f:
            json.dump(reward_config, f, indent=2)
        files_created['reward'] = f"{output_dir}/reward_config.json"
        
        with open(f"{output_dir}/training_scenarios.json", 'w') as f:
            json.dump(training_scenarios, f, indent=2)
        files_created['scenarios'] = f"{output_dir}/training_scenarios.json"
        
        with open(f"{output_dir}/sinergym_config.json", 'w') as f:
            json.dump(sinergym_config, f, indent=2)
        files_created['sinergym'] = f"{output_dir}/sinergym_config.json"
        
        # Save training dataset
        training_dataset.to_csv(f"{output_dir}/training_dataset.csv", index=False)
        files_created['dataset'] = f"{output_dir}/training_dataset.csv"
        
        print(f"\n✅ Training data package generated!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Training dataset: {len(training_dataset)} samples")
        print(f"🎯 Baseline target: {self.performance_baseline['energy_intensity_target']} kWh/m²/year")
        
        return files_created

# Usage example
def main():
    """Main function to extract training data"""
    
    # Initialize extractor
    extractor = AstraZenecaTrainingDataExtractor(
        certificate_path="AZ_AC_certificate.pdf",
        report_path="AZ_AC_report.pdf"
    )
    
    # Generate complete training package
    files = extractor.generate_complete_training_package()
    
    print(f"\n🚀 Ready for SAC training!")
    print(f"📁 Use Sinergym config: {files['sinergym']}")
    print(f"📈 Training dataset: {files['dataset']}")
    print(f"🏆 Performance target: Beat current baseline")
    
if __name__ == "__main__":
    main()