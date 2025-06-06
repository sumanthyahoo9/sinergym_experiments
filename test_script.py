import gymnasium as gym
import sinergym
from pprint import pprint

# Get all Sinergym environments
sinergym_envs = [env for env in gym.envs.registry.keys() if env.startswith('Eplus')]

print(f"Found {len(sinergym_envs)} Sinergym environments:\n")

# Group environments by building
buildings = {}
for env_id in sorted(sinergym_envs):
    parts = env_id.split('-')
    if len(parts) >= 2:
        building = parts[1]
        if building not in buildings:
            buildings[building] = []
        buildings[building].append(env_id)

# Print environments grouped by building
for building, envs in buildings.items():
    print(f"Building: {building}")
    for env_id in envs:
        # Try to create the environment to get more info
        try:
            env = gym.make(env_id)
            obs_shape = env.observation_space.shape
            act_shape = env.action_space.shape
            print(f"  - {env_id}")
            print(f"    Observation space: {obs_shape}, Action space: {act_shape}")
            env.close()
        except Exception as e:
            print(f"  - {env_id} (Error: {str(e)})")
    print()
