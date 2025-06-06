import gymnasium as gym
import numpy as np
import yaml
from sinergym.utils.rewards import LinearReward
from sinergym.utils.wrappers import NormalizeObservation, NormalizeAction

def debug_observation_structure():
    """Debug the observation structure to find correct energy index"""
    print("🔍 DEBUGGING OBSERVATION STRUCTURE")
    print("="*60)
    
    # Load config
    with open('PPO_sweep_example.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup environment parameters
    env_params = {}
    if config.get('environment_parameters'):
        env_params = config['environment_parameters'].copy()
        
        if 'reward' in env_params:
            env_params['reward'] = LinearReward
            
        if 'weather_variability' in env_params:
            for var_name, var_param in env_params['weather_variability'].items():
                if isinstance(var_param, list):
                    env_params['weather_variability'][var_name] = tuple(var_param)
    
    # Test 1: Raw environment (no wrappers)
    print("\n1️⃣ RAW ENVIRONMENT (No Wrappers)")
    print("-" * 40)
    env_params['env_name'] = 'debug_raw'
    env_raw = gym.make(config['environment'], **env_params)
    
    obs_raw, _ = env_raw.reset()
    print(f"Raw observation shape: {obs_raw.shape}")
    print(f"Raw observation sample values:")
    for i, val in enumerate(obs_raw):
        print(f"  Index {i:2d}: {val:12.4f}")
    
    # Take one step to see energy changes
    action_raw = np.array([22.0, 25.0], dtype=np.float32)
    obs_raw_2, reward_raw, _, _, info = env_raw.step(action_raw)
    
    print(f"\nAfter one step:")
    print(f"  Reward: {reward_raw:.4f}")
    print(f"  Energy-related changes:")
    for i, (val1, val2) in enumerate(zip(obs_raw, obs_raw_2)):
        if abs(val2 - val1) > 0.001:  # Show indices that changed
            print(f"    Index {i:2d}: {val1:12.4f} → {val2:12.4f} (change: {val2-val1:+.4f})")
    
    env_raw.close()
    
    # Test 2: Wrapped environment (with normalization)
    print("\n2️⃣ WRAPPED ENVIRONMENT (With Normalization)")
    print("-" * 40)
    env_params['env_name'] = 'debug_wrapped'
    env_wrapped = gym.make(config['environment'], **env_params)
    env_wrapped = NormalizeObservation(env_wrapped)
    env_wrapped = NormalizeAction(env_wrapped)
    
    obs_wrapped, _ = env_wrapped.reset()
    print(f"Wrapped observation shape: {obs_wrapped.shape}")
    print(f"Wrapped observation sample values:")
    for i, val in enumerate(obs_wrapped):
        print(f"  Index {i:2d}: {val:12.4f}")
    
    # Take one step to see changes
    action_wrapped = np.array([-0.5, 0.2], dtype=np.float32)  # Normalized action
    obs_wrapped_2, reward_wrapped, _, _, info = env_wrapped.step(action_wrapped)
    
    print(f"\nAfter one step:")
    print(f"  Reward: {reward_wrapped:.4f}")
    print(f"  Energy-related changes:")
    for i, (val1, val2) in enumerate(zip(obs_wrapped, obs_wrapped_2)):
        if abs(val2 - val1) > 0.001:
            print(f"    Index {i:2d}: {val1:12.4f} → {val2:12.4f} (change: {val2-val1:+.4f})")
    
    env_wrapped.close()
    
    # Test 3: Check info dictionary
    print("\n3️⃣ INFO DICTIONARY CONTENTS")
    print("-" * 40)
    env_params['env_name'] = 'debug_info'
    env_info = gym.make(config['environment'], **env_params)
    obs_info, info_initial = env_info.reset()
    
    print("Info at reset:")
    for key, value in info_initial.items():
        print(f"  {key}: {value}")
    
    obs_info_2, reward_info, _, _, info_step = env_info.step(np.array([22.0, 25.0], dtype=np.float32))
    print("\nInfo after step:")
    for key, value in info_step.items():
        print(f"  {key}: {value}")
    
    env_info.close()
    
    # Test 4: Expected variable mapping from your training data
    print("\n4️⃣ EXPECTED VARIABLE MAPPING")
    print("-" * 40)
    expected_mapping = {
        0: "obs_outdoor_temperature",
        1: "obs_outdoor_humidity", 
        2: "obs_wind_speed",
        3: "obs_wind_direction",
        4: "obs_diffuse_solar_radiation",
        5: "obs_direct_solar_radiation",
        6: "obs_htg_setpoint",
        7: "obs_clg_setpoint", 
        8: "obs_air_temperature",
        9: "obs_air_humidity",
        10: "obs_people_occupant",
        11: "obs_co2_emission",
        12: "obs_HVAC_electricity_demand_rate",  # This might be the correct index!
        13: "obs_total_electricity_HVAC",
        14: "month",
        15: "day_of_month", 
        16: "hour"
    }
    
    print("Based on your training data, expected mapping:")
    for idx, name in expected_mapping.items():
        if idx < len(obs_raw):
            print(f"  Index {idx:2d}: {name:35s} = {obs_raw[idx]:12.4f}")
    
    # Test 5: Look for large values that could be energy
    print("\n5️⃣ POTENTIAL ENERGY INDICES (Large values)")
    print("-" * 40)
    print("Indices with large absolute values (potential energy consumption):")
    for i, val in enumerate(obs_raw):
        if abs(val) > 1000:  # Energy values are typically large
            print(f"  Index {i:2d}: {val:12.2f} ← Potential energy index")
    
    print("\n" + "="*60)
    print("🎯 RECOMMENDED ACTIONS:")
    print("1. Try index 12 for energy (obs_HVAC_electricity_demand_rate)")
    print("2. Check indices with large values for energy consumption")
    print("3. Use raw environment for baseline, wrapped for PPO")
    print("4. Verify energy values are reasonable (>0, <100000)")
    print("="*60)

if __name__ == '__main__':
    debug_observation_structure()