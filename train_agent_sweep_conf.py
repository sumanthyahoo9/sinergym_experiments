import argparse
import logging
import sys
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import Logger as SB3Logger

import sinergym
from sinergym.utils.common import (
    is_wrapped,
    process_algorithm_parameters,
    process_environment_parameters,
)
from sinergym.utils.constants import *
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *

def train(config_path):
    """
    Training function without W&B dependency
    """
    try:
        # Load configuration
        with open(config_path, 'r') as yaml_conf:
            config = yaml.safe_load(yaml_conf)

        # ---------------------------------------------------------------------------- #
        #                                Training name                                 #
        # ---------------------------------------------------------------------------- #
        training_date = datetime.today().strftime('%Y-%m-%d_%H-%M')
        training_name = config['algorithm'] + '-' + config['environment'] + \
            '-episodes-' + str(config['episodes'])
        training_name += '_' + training_date

        # ---------------------------------------------------------------------------- #
        #                                Set random seed                               #
        # ---------------------------------------------------------------------------- #
        if config.get('seed'):
            np.random.seed(config['seed'])

        # --------------------- Overwrite environment parameters --------------------- #
        env_params = {}
        if config.get('environment_parameters'):
            env_params = process_environment_parameters(config['environment_parameters'])

        # ---------------------------------------------------------------------------- #
        #                            Environment definition                            #
        # ---------------------------------------------------------------------------- #
        env_params.update({'env_name': training_name})
        env = gym.make(config['environment'], **env_params)

        # ---------------------------------------------------------------------------- #
        #                                   Wrappers                                   #
        # ---------------------------------------------------------------------------- #
        if config.get('wrappers'):
            for wrapper_config in config['wrappers']:
                wrapper_name = wrapper_config[0]
                wrapper_parameters = wrapper_config[1]
                wrapper_class = eval(wrapper_name)
                for name, value in wrapper_parameters.items():
                    if isinstance(value, str) and '.' in value and '.txt' not in value:
                        wrapper_parameters[name] = eval(value)
                env = wrapper_class(env=env, **wrapper_parameters)

        # ---------------------------------------------------------------------------- #
        #                           DRL model initialization                           #
        # ---------------------------------------------------------------------------- #
        algorithm_parameters = process_algorithm_parameters(
            config.get('algorithm_parameters', {'policy': 'MlpPolicy'}))
        algorithm_class = eval(config['algorithm'])

        # Set seed in algorithm if provided
        if config.get('seed'):
            algorithm_parameters['seed'] = config['seed']

        model = algorithm_class(env=env, **algorithm_parameters)

        # ---------------------------------------------------------------------------- #
        #                          Application of callback(s)                          #
        # ---------------------------------------------------------------------------- #
        callbacks = []

        # Set standard logger (no W&B)
        logger = SB3Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout, max_length=120)])
        model.set_logger(logger)

        # Evaluation Callback if evaluations are specified
        evaluation = config.get('evaluation', False)
        if evaluation:
            from sinergym.utils.callbacks import LoggerEvalCallback
            
            # Create evaluation environment
            eval_env_params = env_params.copy()
            eval_env_params.update({'env_name': training_name + '-EVAL'})
            eval_env = gym.make(config['environment'], **eval_env_params)
            
            # Apply wrappers to eval env
            if config.get('wrappers'):
                for wrapper_config in config['wrappers']:
                    wrapper_name = wrapper_config[0]
                    wrapper_parameters = wrapper_config[1]
                    wrapper_class = eval(wrapper_name)
                    for name, value in wrapper_parameters.items():
                        if isinstance(value, str) and '.' in value and '.txt' not in value:
                            wrapper_parameters[name] = eval(value)
                    eval_env = wrapper_class(env=eval_env, **wrapper_parameters)

            eval_callback = LoggerEvalCallback(
                eval_env=eval_env,
                train_env=env,
                n_eval_episodes=config['evaluation']['eval_length'],
                eval_freq_episodes=config['evaluation']['eval_freq'],
                deterministic=True,
                verbose=1)
            callbacks.append(eval_callback)

        callback = CallbackList(callbacks)

        # ---------------------------------------------------------------------------- #
        #                                 DRL training                                 #
        # ---------------------------------------------------------------------------- #
        timesteps = config['episodes'] * (env.get_wrapper_attr('timestep_per_episode') - 1)
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            log_interval=config['log_interval'])

        # ---------------------------------------------------------------------------- #
        #                                Saving results                                #
        # ---------------------------------------------------------------------------- #
        os.makedirs('./models', exist_ok=True)
        model.save(f'./models/{training_name}')

        env.close()

    except (Exception, KeyboardInterrupt) as err:
        print("Error or interruption in process detected")
        print(f"Error: {err}")
        
        try:
            model.save(f'./models/{training_name}_interrupted')
        except:
            pass
            
        env.close()
        raise err

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, type=str, help='Path to config YAML file')
    args = parser.parse_args()
    
    train(args.config)