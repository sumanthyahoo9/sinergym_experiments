# Test if energy and reward are properly correlated
import gymnasium as gym
import numpy as np
from sinergym.utils.rewards import LinearReward
def test_reward_energy_correlation():
    env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1', reward=LinearReward)
    obs, _ = env.reset()
    
    # Test different energy strategies
    strategies = {
        'energy_waste': [23.0, 24.0],      # Narrow deadband = high energy
        'energy_save': [18.0, 28.0],       # Wide deadband = low energy  
        'baseline': [22.0, 25.0]           # Your current baseline
    }
    
    for name, action in strategies.items():
        obs, reward, _, _, info = env.step(np.array(action, dtype=np.float32))
        energy = info.get('total_power_demand', 0)
        print(f"{name}: Energy={energy:.1f}W, Reward={reward:.3f}")


if __name__ == "__main__":
    test_reward_energy_correlation()