from sinergym.utils.rewards import LinearReward

class EnhancedLinearReward(LinearReward):
    """Enhanced LinearReward with occupancy, humidity, and time-of-day weighting"""
    
    def __init__(self, 
                 temperature_variable='air_temperature',
                 energy_variable='HVAC_electricity_demand_rate',
                 range_comfort_winter=(20.0, 23.5), 
                 range_comfort_summer=(23.0, 26.0),
                 summer_start=(6, 1), 
                 summer_final=(9, 30),
                 energy_weight=0.5, 
                 lambda_energy=1e-4, 
                 lambda_temperature=1.0):
        
        super().__init__(temperature_variable, energy_variable, range_comfort_winter,
                        range_comfort_summer, summer_start, summer_final, 
                        energy_weight, lambda_energy, lambda_temperature)
        
        # FIX: Ensure energy_names and temp_names are lists, not strings
        if isinstance(self.energy_names, str):
            self.energy_names = [self.energy_names]
        if isinstance(self.temp_names, str):
            self.temp_names = [self.temp_names]
    
    def __call__(self, obs_dict):
        # Get base reward from LinearReward
        base_reward, rw_terms = super().__call__(obs_dict)
        
        try:
            # Extract additional variables
            air_humidity = obs_dict.get('air_humidity', 50)
            people_count = obs_dict.get('people_occupant', 0)
            hour = obs_dict.get('hour', 12)
            
            # Enhancement 1: Humidity comfort penalty
            target_humidity = 50.0
            humidity_penalty = abs(air_humidity - target_humidity) / 100.0
            
            # Enhancement 2: Occupancy-based weighting
            occupancy_multiplier = 1.0
            if people_count > 5:
                occupancy_multiplier = 1.2
            elif people_count == 0:
                occupancy_multiplier = 0.8
            
            # Enhancement 3: Peak hour penalty multiplier
            time_multiplier = 1.0
            if 14 <= hour <= 18:
                time_multiplier = 1.5
            elif 22 <= hour or hour <= 6:
                time_multiplier = 0.8
            
            enhanced_reward = base_reward * occupancy_multiplier * time_multiplier - humidity_penalty * 0.1
            return enhanced_reward, rw_terms
            
        except (KeyError, TypeError):
            return base_reward, rw_terms