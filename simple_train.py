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
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.sac.policies import SACPolicy
sys.path.append(".")

from sinergym.utils.common import process_algorithm_parameters
from sinergym.utils.constants import *
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *
from observation_noise_wrapper import InternalSensorNoiseWrapper

def check_gpu_availability():
    """Check if there's a GPU for usage"""
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
    
    def __init__(self, *args, dropout_rate=0.15, **kwargs):
        self.dropout_rate = dropout_rate
        super().__init__(*args, **kwargs)
    
    def make_actor(self, features_extractor=None):
        """Add dropout layers to actor network"""
        actor = super().make_actor(features_extractor)
        
        new_layers = []
        for i, layer in enumerate(actor.latent_pi):
            new_layers.append(layer)
            if i % 2 == 1:  # Add dropout after ReLU activations
                new_layers.append(nn.Dropout(p=self.dropout_rate))
        
        actor.latent_pi = nn.Sequential(*new_layers)
        return actor

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
        
        # CRITICAL FIX: Force energy_names to be a list
        # Parent class might set it as string during __init__
        self.energy_names = [energy_variable] if isinstance(energy_variable, str) else energy_variable
        self.temp_names = [temperature_variable] if isinstance(temperature_variable, str) else temperature_variable
    
    def _get_energy_consumed(self, obs_dict):
        """Override parent method to handle string properly"""
        # Ensure we always work with a list
        energy_names = self.energy_names if isinstance(self.energy_names, list) else [self.energy_names]
        return [obs_dict[v] for v in energy_names]
    
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        required=True,
        type=str,
        help='Path to configuration YAML file',
        default="policy_configs/SAC_sweep_v2.yaml"
    )
    args = parser.parse_args()

    terminal_logger = TerminalLogger()
    logger = terminal_logger.getLogger(name='TRAINING', level=logging.INFO)

    with open(args.config, 'r') as yaml_conf:
        config = yaml.safe_load(yaml_conf)

    try:
        # Training name
        training_date = datetime.today().strftime('%Y-%m-%d_%H-%M')
        training_name = config['algorithm'] + '-' + config['environment'] + \
            '-episodes-' + str(config['episodes'])
        training_name += '_' + training_date

        logger.info(f"Starting training: {training_name}")

        # Set random seed
        if config.get('seed'):
            np.random.seed(config['seed'])
            logger.info(f"Random seed set to: {config['seed']}")

        # Environment definition
        env_params = {}
        if config.get('environment_parameters'):
            env_params = config['environment_parameters'].copy()

            # Handle custom reward
            if env_params.get('reward') == 'EnhancedLinearReward':
                env_params['reward'] = EnhancedLinearReward
            elif env_params.get('reward') == 'LinearReward':
                env_params['reward'] = LinearReward
            else:
                if isinstance(env_params.get('reward'), str):
                    env_params['reward'] = eval(env_params['reward'])

            env_params.update({'env_name': training_name})
            env = gym.make(config['environment'], **env_params)

        # Apply wrappers
        if config.get('wrappers'):
            for wrapper_config in config['wrappers']:
                wrapper_name = wrapper_config[0]
                wrapper_params = wrapper_config[1]
                
                wrapper_class = eval(wrapper_name)
                for name, value in wrapper_params.items():
                    if isinstance(value, str) and '.' in value and '.txt' not in value:
                        wrapper_params[name] = eval(value)
                
                env = wrapper_class(env=env, **wrapper_params)

        # ---------------------------------------------------------------------------- #
        #                    Internal Sensor Noise (Production-like)                  #
        # ---------------------------------------------------------------------------- #
        internal_sensor_noise = {
            'air_temperature': 0.2,           # ±0.2°C HVAC sensor drift
            'air_humidity': 1.0,              # ±1% RH sensor error
            'people_occupant': 1.0,           # ±1 person counting error
            'HVAC_electricity_demand_rate': 100.0,  # ±100W power meter accuracy
        }
        
        env = InternalSensorNoiseWrapper(env, internal_sensor_noise)
        logger.info("Internal sensor noise wrapper applied (production-like conditions)")

        logger.info("Environment and wrappers configured successfully")
        device = check_gpu_availability()

        # DRL model initialization
        algorithm_parameters = process_algorithm_parameters(config.get('algorithm_parameters', {}))
        algorithm_parameters['device'] = device
        algorithm_class = eval(config['algorithm'])

        if config.get('seed'):
            algorithm_parameters['seed'] = config['seed']

        logger.info(f"Initializing {config['algorithm']} algorithm")
        if config.get('model', None) is None:
            if config['algorithm'] == 'SAC' and config.get('use_dropout', False):
                logger.info("Using SAC with Dropout layers")
                algorithm_parameters['policy'] = DropoutSACPolicy
                algorithm_parameters['policy_kwargs'] = {
                    'dropout_rate': config.get('dropout_rate', 0.1),
                    'net_arch': [256, 256]
                }
            model = algorithm_class(env=env, **algorithm_parameters)

        # Callbacks
        callbacks = []

        if config.get('evaluation'):
            from sinergym.utils.callbacks import LoggerEvalCallback
            
            eval_env_params = env_params.copy()
            eval_env_params.update({'env_name': training_name + '-EVAL'})
            eval_env = gym.make(config['environment'], **eval_env_params)
            
            # Apply same wrappers to eval env
            if config.get('wrappers'):
                for wrapper_config in config['wrappers']:
                    wrapper_name = wrapper_config[0]
                    if wrapper_name != 'WandBLogger':
                        wrapper_params = wrapper_config[1]
                        wrapper_class = eval(wrapper_name)
                        for name, value in wrapper_params.items():
                            if isinstance(value, str) and '.' in value and '.txt' not in value:
                                wrapper_params[name] = eval(value)
                        eval_env = wrapper_class(env=eval_env, **wrapper_params)
            
            # Apply sensor noise to eval env too
            eval_env = InternalSensorNoiseWrapper(eval_env, internal_sensor_noise)

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

        # Training
        timesteps = config['episodes'] * (env.get_wrapper_attr('timestep_per_episode') - 1)
        
        logger.info(f"Starting training for {timesteps} timesteps")
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            log_interval=config['log_interval']
        )

        # Save results
        os.makedirs('./models', exist_ok=True)
        
        model_save_path = f'./models/{training_name}'
        model.save(model_save_path)
        logger.info(f"Model saved to: {model_save_path}")

        config_save_path = f'./models/{training_name}_config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)
        logger.info(f"Configuration saved to: {config_save_path}")

        # Close environment
        if env.get_wrapper_attr('is_running'):
            env.close()

        logger.info("Training completed successfully!")

    except (Exception, KeyboardInterrupt) as err:
        logger.error(f"Error or interruption detected: {err}")
        
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