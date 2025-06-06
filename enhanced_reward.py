import numpy as np
from sinergym.utils.rewards import LinearReward

class EnhancedLinearReward(LinearReward):
    """Enhanced LinearReward that uses more of the 17 observation variables"""
    
    def __init__(self, temperature_variable, energy_variable, range_comfort_winter, 
                 range_comfort_summer, summer_start=(6, 1), summer_final=(9, 30),
                 energy_weight=0.5, lambda_energy=1e-4, lambda_temperature=1.0):
        
        super().__init__(temperature_variable, energy_variable, range_comfort_winter,
                        range_comfort_summer, summer_start, summer_final, 
                        energy_weight, lambda_energy, lambda_temperature)
    
    def __call__(self, observation):
        # Get base reward from LinearReward
        base_reward = super().__call__(observation)
        
        # Extract additional variables for enhancement
        try:
            outdoor_temp = observation[0] if len(observation) > 0 else 20
            outdoor_humidity = observation[1] if len(observation) > 1 else 50  
            zone_humidity = observation[10] if len(observation) > 10 else 50
            people_count = observation[11] if len(observation) > 11 else 0
            hour = observation[16] if len(observation) > 16 else 12
            
            # Enhancement 1: Humidity comfort penalty
            target_humidity = 50.0
            humidity_penalty = abs(zone_humidity - target_humidity) / 100.0
            
            # Enhancement 2: Occupancy-based weighting
            occupancy_multiplier = 1.0
            if people_count > 5:
                occupancy_multiplier = 1.2  # Higher penalty when occupied
            elif people_count == 0:
                occupancy_multiplier = 0.8  # Lower penalty when empty
            
            # Enhancement 3: Peak hour penalty multiplier
            time_multiplier = 1.0
            if 14 <= hour <= 18:  # Peak hours
                time_multiplier = 1.5
            elif 22 <= hour or hour <= 6:  # Off-peak
                time_multiplier = 0.8
            
            # Apply enhancements
            enhanced_penalty = (humidity_penalty * 0.1 +  # Small humidity component
                              base_reward * occupancy_multiplier * time_multiplier)
            
            return enhanced_penalty
            
        except (IndexError, TypeError):
            # Fallback to base reward if observation structure is different
            return base_reward