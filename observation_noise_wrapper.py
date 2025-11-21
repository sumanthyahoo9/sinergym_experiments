"""
Use this to add noise to the internal sensor variables.
"""
import gymnasium as gym
import numpy as np
from typing import Dict

class InternalSensorNoiseWrapper(gym.ObservationWrapper):
    """
    Add realistic noise to internal sensor observations only.
    Weather variables remain clean (from accurate API).
    """
    
    def __init__(self, env, noise_config: Dict[str, float]):
        """
        Args:
            env: Sinergym environment
            noise_config: Dict mapping observation keys to noise std dev
                Example: {'air_temperature': 0.2, 'air_humidity': 1.0}
        """
        super().__init__(env)
        self.noise_config = noise_config
        
    def observation(self, obs_dict):
        """Add Gaussian noise to specified internal sensors"""
        noisy_obs = obs_dict.copy()
        
        for var_name, noise_std in self.noise_config.items():
            if var_name in noisy_obs:
                # Add zero-mean Gaussian noise
                noise = np.random.normal(0, noise_std)
                noisy_obs[var_name] += noise
                
        return noisy_obs