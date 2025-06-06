import sinergym
import gymnasium as gym
import numpy as np
import os

# Print available environments
print("Available Sinergym environments:")
for env_id in [env for env in gym.envs.registry.keys() if env.startswith('Eplus')]:
    print(f"  - {env_id}")

# Create environment
env_name = 'Eplus-5Zone-hot-continuous-v1'
print(f"\nCreating environment: {env_name}")
env = gym.make(env_name)

# Print space information
print(f"\nObservation Space: {env.observation_space}")
print(f"Action Space: {env.action_space}")

# Reset environment
observation, info = env.reset()
print(f"\nInitial observation shape: {observation.shape}")
print(f"First 5 observation values: {observation[:5]}")

# Run a few simulation steps
print("\nRunning 10 simulation steps with random actions...")
for i in range(10):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {i+1}:")
    print(f"  Action: {action}")
    print(f"  Reward: {reward:.4f}")
    
    if terminated or truncated:
        print("  Episode ended, resetting environment")
        observation, info = env.reset()

# Get episode results
if hasattr(env, 'get_episode_results'):
    results = env.get_episode_results()
    print("\nEpisode results summary:")
    for key, value in results.items():
        if isinstance(value, (int, float, str)):
            print(f"  {key}: {value}")

# Print working directory (to confirm where files are saved)
print(f"\nCurrent working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# Close the environment
env.close()
print("\nEnvironment closed successfully!")
