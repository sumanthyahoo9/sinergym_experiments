import argparse
import logging
import sys
import os
import torch
import torch.nn as nn
from datetime import datetime

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
sys.path.append(".")
# Make it available globally for eval()
from sinergym.utils.common import (
    is_wrapped,
    process_algorithm_parameters,
    process_environment_parameters,
)
from sinergym.utils.constants import *
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *

def check_gpu_availability():
    """
    Check if there's a GPU for usage
    """
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"🚀 GPU available: {device}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return 'cuda'
    else:
        print("⚠️  No GPU available, using CPU")
        return 'cpu'

class DropoutSACPolicy(SACPolicy):
    """SAC Policy with dropout for uncertainty estimation"""
    
    def __init__(self, *args, dropout_rate=0.1, **kwargs):
        self.dropout_rate = dropout_rate
        super().__init__(*args, **kwargs)
    
    def make_actor(self, features_extractor=None):
        """Add dropout layers to actor network"""
        actor = super().make_actor(features_extractor)
        
        # Insert dropout after each activation
        new_layers = []
        for i, layer in enumerate(actor.latent_pi):
            new_layers.append(layer)
            # Add dropout after ReLU activations (odd indices)
            if i % 2 == 1:  
                new_layers.append(nn.Dropout(p=self.dropout_rate))
        
        actor.latent_pi = nn.Sequential(*new_layers)
        return actor

class EnhancedLinearReward(LinearReward):
    """Enhanced LinearReward that uses more of the 17 observation variables"""
    
    def __init__(self, 
                 temperature_variables=["air_temperature"], 
                 energy_variables=["HVAC_electricity_demand_rate"],  # ← Changed to plural
                 range_comfort_winter=(20.0, 23.5), 
                 range_comfort_summer=(23.0, 26.0),
                 summer_start=(6, 1), 
                 summer_final=(9, 30),
                 energy_weight=0.5, 
                 lambda_energy=1e-4, 
                 lambda_temperature=1.0):
        
        # Convert to singular for parent class
        temperature_variable = temperature_variables[0] if isinstance(temperature_variables, list) else temperature_variables
        energy_variable = energy_variables[0] if isinstance(energy_variables, list) else energy_variables
        
        # Set defaults if not provided
        """
        if temperature_variable is None:
            temperature_variable = 'air_temperature'  # ← Use SinerGym internal name
        if energy_variable is None:
            energy_variable = 'HVAC_electricity_demand_rate'  # ← Use SinerGym internal name
        """
        temperature_variable = temperature_variables[0] if isinstance(temperature_variables, list) else temperature_variables
        energy_variable = energy_variables[0] if isinstance(energy_variables, list) else energy_variables
        # DEBUG: Check what we're passing to parent
        print(f"DEBUG EnhancedLinearReward: temperature_variable = {temperature_variable}")
        print(f"DEBUG EnhancedLinearReward: energy_variable = {energy_variable}")
        print(f"DEBUG EnhancedLinearReward: type(energy_variable) = {type(energy_variable)}")
            
        super().__init__(temperature_variable, energy_variable, range_comfort_winter,
                        range_comfort_summer, summer_start, summer_final, 
                        energy_weight, lambda_energy, lambda_temperature)
        # FIX: Ensure energy_names is a list, not a string
        if isinstance(self.energy_names, str):
            self.energy_names = [self.energy_names]
        if isinstance(self.temp_names, str):
            self.temp_names = [self.temp_names]
        # DEBUG: Check what parent class set
        print(f"DEBUG EnhancedLinearReward: self.energy_names = {getattr(self, 'energy_names', 'NOT_SET')}")
        print(f"DEBUG EnhancedLinearReward: self.energy_name = {getattr(self, 'energy_name', 'NOT_SET')}")
        # DEBUG: Check both fixes worked
        print(f"DEBUG FIXED: self.energy_names = {self.energy_names}")
        print(f"DEBUG FIXED: self.temp_names = {self.temp_names}")
        print(f"DEBUG FIXED: type(self.energy_names) = {type(self.energy_names)}")
        print(f"DEBUG FIXED: type(self.temp_names) = {type(self.temp_names)}")
    
    def __call__(self, obs_dict):  # ← Make sure this matches LinearReward signature
        # Get base reward from LinearReward
        base_reward, rw_terms = super().__call__(obs_dict)
        
        # Extract additional variables for enhancement from obs_dict
        try:
            # Use SinerGym variable names from obs_dict
            outdoor_temp = obs_dict.get('outdoor_temperature', 20)
            outdoor_humidity = obs_dict.get('outdoor_humidity', 50)  
            air_humidity = obs_dict.get('air_humidity', 50)
            people_count = obs_dict.get('people_occupant', 0)
            # Extract hour from time variables or obs_dict
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
            # Extract reward components
            energy_penalty = obs_dict.get('energy_term', 0)  
            comfort_penalty = obs_dict.get('comfort_term', 0)
            
            # Apply different logic to each component
            if people_count == 0:
                # Empty building: strict on energy, lenient on comfort
                energy_multiplier = 1.5    # Amplify energy penalties
                comfort_multiplier = 0.5   # Reduce comfort penalties
            else:
                # Occupied: balance energy and comfort  
                energy_multiplier = 1.0
                comfort_multiplier = 1.0
            enhanced_reward = base_reward * occupancy_multiplier * time_multiplier - humidity_penalty * 0.1

            
            return enhanced_reward, rw_terms
            
        except (KeyError, TypeError):
            return base_reward


def main():
    # ---------------------------------------------------------------------------- #
    #                                  Parameters                                  #
    # ---------------------------------------------------------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        required=True,
        type=str,
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()

    # Terminal logger
    terminal_logger = TerminalLogger()
    logger = terminal_logger.getLogger(name='TRAINING', level=logging.INFO)

    # ---------------------------------------------------------------------------- #
    #                             Read yaml parameters                             #
    # ---------------------------------------------------------------------------- #
    with open(args.config, 'r') as yaml_conf:
        config = yaml.safe_load(yaml_conf)

    try:
        # ---------------------------------------------------------------------------- #
        #                                Training name                                 #
        # ---------------------------------------------------------------------------- #
        training_date = datetime.today().strftime('%Y-%m-%d_%H-%M')
        training_name = config['algorithm'] + '-' + config['environment'] + \
            '-episodes-' + str(config['episodes'])
        training_name += '_' + training_date

        logger.info(f"Starting training: {training_name}")

        # ---------------------------------------------------------------------------- #
        #                                Set random seed                               #
        # ---------------------------------------------------------------------------- #
        if config.get('seed'):
            np.random.seed(config['seed'])
            logger.info(f"Random seed set to: {config['seed']}")

        # ---------------------------------------------------------------------------- #
        #                            Environment definition                            #
        # ---------------------------------------------------------------------------- #
        env_params = {}
        if config.get('environment_parameters'):
            env_params = config['environment_parameters'].copy()

            # DEBUG: Check what we're actually getting
            if 'weather_variability' in env_params:
                temp_param = env_params['weather_variability']['Dry Bulb Temperature']
                print(f"DEBUG: Type: {type(temp_param)}")
                print(f"DEBUG: Value: {temp_param}")
                print(f"DEBUG: Length: {len(temp_param)}")
                
                # Convert to tuple if it's a list
                if isinstance(temp_param, list):
                    env_params['weather_variability']['Dry Bulb Temperature'] = tuple(temp_param)
                    print(f"DEBUG: Converted to tuple: {env_params['weather_variability']['Dry Bulb Temperature']}")
    
            # Handle custom reward manually
            if env_params['reward'] == 'EnhancedLinearReward':
                env_params['reward'] = EnhancedLinearReward
            elif env_params['reward'] == 'LinearReward':
                env_params['reward'] = LinearReward
            elif env_params['reward'] == 'SAC':  # Add this line
                env_params['reward'] = eval(env_params['reward'])
            else:
                env_params['reward'] = eval(env_params['reward'])
            
            # Process other parameters normally
            for key, value in env_params.items():
                if key != 'reward' and isinstance(value, str) and '.' in value:
                    env_params[key] = eval(value)

            env_params.update({'env_name': training_name})
            env = gym.make(config['environment'], **env_params)

        # ---------------------------------------------------------------------------- #
        #                                   Wrappers                                   #
        # ---------------------------------------------------------------------------- #
        if config.get('wrappers'):
            for wrapper_config in config['wrappers']:
                wrapper_name = wrapper_config[0]
                wrapper_params = wrapper_config[1]
                
                wrapper_class = eval(wrapper_name)
                for name, value in wrapper_params.items():
                    if isinstance(value, str) and '.' in value and '.txt' not in value:
                        wrapper_params[name] = eval(value)
                
                env = wrapper_class(env=env, **wrapper_params)

        logger.info("Environment and wrappers configured successfully")
        device = check_gpu_availability()

        # ---------------------------------------------------------------------------- #
        #                           DRL model initialization                           #
        # ---------------------------------------------------------------------------- #
        algorithm_parameters = process_algorithm_parameters(config.get('algorithm_parameters', {}))
        algorithm_parameters['device'] = device  # Force detected device
        algorithm_class = eval(config['algorithm'])

        # Set seed in algorithm parameters if provided
        if config.get('seed'):
            algorithm_parameters['seed'] = config['seed']

        logger.info(f"Initializing {config['algorithm']} algorithm")
        if config.get('model', None) is None:
            # Use dropout policy if training SAC with dropout
            if config['algorithm'] == 'SAC' and config.get('use_dropout', False):
                logger.info("Using SAC with Dropout layers")
                algorithm_parameters['policy'] = DropoutSACPolicy
                algorithm_parameters['policy_kwargs'] = {
                    'dropout_rate': config.get('dropout_rate', 0.1),
                    'net_arch': [256, 256]
                }
            model = algorithm_class(env=env, **algorithm_parameters)

        # ---------------------------------------------------------------------------- #
        #                          Application of callback(s)                          #
        # ---------------------------------------------------------------------------- #
        callbacks = []

        # Evaluation Callback if evaluations are specified
        if config.get('evaluation'):
            from sinergym.utils.callbacks import LoggerEvalCallback
            
            # Create evaluation environment
            eval_env_params = env_params.copy()
            eval_env_params.update({'env_name': training_name + '-EVAL'})
            eval_env = gym.make(config['environment'], **eval_env_params)
            
            # Apply same wrappers to eval env (except WandB logger)
            if config.get('wrappers'):
                for wrapper_config in config['wrappers']:
                    wrapper_name = wrapper_config[0]
                    if wrapper_name != 'WandBLogger':  # Skip WandB logger
                        wrapper_params = wrapper_config[1]
                        wrapper_class = eval(wrapper_name)
                        for name, value in wrapper_params.items():
                            if isinstance(value, str) and '.' in value and '.txt' not in value:
                                wrapper_params[name] = eval(value)
                        eval_env = wrapper_class(env=eval_env, **wrapper_params)

            eval_callback = LoggerEvalCallback(
                eval_env=eval_env,
                train_env=env,
                n_eval_episodes=config['evaluation']['eval_length'],
                eval_freq_episodes=config['evaluation']['eval_freq'],
                deterministic=True,
                verbose=1
            )
            callbacks.append(eval_callback)

        callback = CallbackList(callbacks)

        # ---------------------------------------------------------------------------- #
        #                                 DRL training                                 #
        # ---------------------------------------------------------------------------- #
        timesteps = config['episodes'] * (env.get_wrapper_attr('timestep_per_episode') - 1)
        
        logger.info(f"Starting training for {timesteps} timesteps")
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            log_interval=config['log_interval']
        )

        # ---------------------------------------------------------------------------- #
        #                                Saving results                                #
        # ---------------------------------------------------------------------------- #
        # Create models directory if it doesn't exist
        os.makedirs('./models', exist_ok=True)
        
        model_save_path = f'./models/{training_name}'
        model.save(model_save_path)
        logger.info(f"Model saved to: {model_save_path}")

        # Save training configuration
        config_save_path = f'./models/{training_name}_config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)
        logger.info(f"Configuration saved to: {config_save_path}")

        # ---------------------------------------------------------------------------- #
        #                               Close environment                              #
        # ---------------------------------------------------------------------------- #
        if env.get_wrapper_attr('is_running'):
            env.close()

        logger.info("Training completed successfully!")

    except (Exception, KeyboardInterrupt) as err:
        logger.error(f"Error or interruption detected: {err}")
        
        # Save current model state
        try:
            model.save(f'./models/{training_name}_interrupted')
            logger.info("Interrupted model saved")
        except:
            logger.error("Could not save interrupted model")
        
        if 'env' in locals():
            env.close()
        raise err

if __name__ == '__main__':
    main()