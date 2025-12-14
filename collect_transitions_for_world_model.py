"""
Collect Transition Data for World Model Training
================================================

This script collects diverse building dynamics data (transitions) WITHOUT needing
a trained RL policy. We use random and scripted exploration strategies to capture
diverse state-action-next_state sequences across different conditions.

Why random/scripted policies work for world model data:
- We need DIVERSITY in transitions, not optimality
- Random actions explore the full action space
- Scripted policies (aggressive cooling, aggressive heating) capture extreme conditions
- Combination gives comprehensive coverage of building dynamics

Usage:
    python collect_transitions_for_world_model.py --episodes 50 --building 5zone
"""

import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
import yaml

import sys
sys.path.append(".")

from sinergym.utils.wrappers import NormalizeObservation, NormalizeAction
from observation_noise_wrapper import InternalSensorNoiseWrapper


class TransitionCollector:
    """Collects and saves transition data from building simulations"""
    
    def __init__(self, output_dir: str = "./world_model_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.transitions = []
        self.metadata = {
            'building_type': None,
            'total_episodes': 0,
            'total_transitions': 0,
            'collection_strategy': [],
            'observation_vars': [],
            'action_vars': []
        }
    
    def collect_episode(self, env, policy_fn, episode_num: int, 
                       strategy_name: str) -> Dict:
        """
        Collect transitions from one episode
        
        Args:
            env: Gym environment
            policy_fn: Function that takes observation and returns action
            episode_num: Episode number
            strategy_name: Name of exploration strategy
        
        Returns:
            Episode statistics
        """
        obs, info = env.reset()
        done = False
        truncated = False
        
        episode_transitions = []
        episode_reward = 0
        steps = 0
        
        print(f"\n📊 Episode {episode_num} - Strategy: {strategy_name}")
        
        while not (done or truncated):
            # Get action from policy
            action = policy_fn(obs, info)
            
            # Take step in environment
            next_obs, reward, done, truncated, next_info = env.step(action)
            
            # Store transition
            transition = {
                'episode': episode_num,
                'timestep': steps,
                'strategy': strategy_name,
                'state': obs.tolist(),
                'action': action.tolist() if isinstance(action, np.ndarray) else [action],
                'next_state': next_obs.tolist(),
                'reward': float(reward),
                'done': done,
                'truncated': truncated
            }
            
            # Add relevant info fields
            for key in ['outdoor_temperature', 'outdoor_humidity', 'people_occupant']:
                if key in info:
                    transition[f'info_{key}'] = float(info[key])
            
            episode_transitions.append(transition)
            
            # Update for next iteration
            obs = next_obs
            info = next_info
            episode_reward += reward
            steps += 1
            
            # Progress indicator
            if steps % 1000 == 0:
                print(f"   Step {steps}/35040 ({steps/350.4:.1f}%)")
        
        self.transitions.extend(episode_transitions)
        
        print(f"   ✅ Episode complete: {steps} steps, reward: {episode_reward:.2f}")
        
        return {
            'episode': episode_num,
            'strategy': strategy_name,
            'steps': steps,
            'total_reward': episode_reward,
            'avg_reward': episode_reward / steps if steps > 0 else 0
        }
    
    def save_transitions(self, building_type: str):
        """Save collected transitions to files"""
        
        # Save as pickle (compact, preserves types)
        pickle_path = os.path.join(self.output_dir, f'transitions_{building_type}.pkl')
        df = pd.DataFrame(self.transitions)
        df.to_pickle(pickle_path)
        print(f"\n💾 Saved {len(self.transitions):,} transitions to {pickle_path}")
        
        # Save as CSV (human-readable, for inspection)
        csv_path = os.path.join(self.output_dir, f'transitions_{building_type}.csv')
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved CSV to {csv_path}")
        
        # Save metadata
        self.metadata['building_type'] = building_type
        self.metadata['total_episodes'] = len(set(df['episode']))
        self.metadata['total_transitions'] = len(self.transitions)
        
        metadata_path = os.path.join(self.output_dir, f'metadata_{building_type}.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"💾 Saved metadata to {metadata_path}")
        
        # Summary statistics
        print(f"\n📈 Collection Summary:")
        print(f"   Building: {building_type}")
        print(f"   Episodes: {self.metadata['total_episodes']}")
        print(f"   Transitions: {self.metadata['total_transitions']:,}")
        print(f"   Strategies: {', '.join(set(df['strategy']))}")


# ============================================================================ #
#                         EXPLORATION STRATEGIES                               #
# ============================================================================ #

def random_policy(obs: np.ndarray, info: Dict) -> np.ndarray:
    """Completely random actions - maximum exploration"""
    # NormalizeAction expects actions in [-1, 1] range
    # These will be denormalized to: heating [12, 23.25], cooling [23.25, 30]
    heating_normalized = np.random.uniform(-1.0, 1.0)
    cooling_normalized = np.random.uniform(-1.0, 1.0)
    return np.array([heating_normalized, cooling_normalized], dtype=np.float32)


def aggressive_cooling_policy(obs: np.ndarray, info: Dict) -> np.ndarray:
    """Always push for maximum cooling - captures cooling dynamics"""
    # Minimum heating (-1 = 12°C), Minimum cooling (-1 = 23.25°C)
    return np.array([-1.0, -1.0], dtype=np.float32)


def aggressive_heating_policy(obs: np.ndarray, info: Dict) -> np.ndarray:
    """Always push for maximum heating - captures heating dynamics"""
    # Maximum heating (+1 = 23.25°C), Maximum cooling (+1 = 30°C)
    return np.array([1.0, 1.0], dtype=np.float32)


def moderate_policy(obs: np.ndarray, info: Dict) -> np.ndarray:
    """Moderate setpoints - captures typical operation"""
    # Mid-range: heating ~20°C, cooling ~24°C
    # Normalize: 20°C → ~0.4, 24°C → ~0.2
    return np.array([0.4, 0.2], dtype=np.float32)


def oscillating_policy(obs: np.ndarray, info: Dict) -> np.ndarray:
    """Oscillate between extremes - captures transient dynamics"""
    # Use timestep from observation if available
    try:
        hour = obs[2] if len(obs) > 2 else 12  # obs[2] is hour
        if hour % 4 < 2:  # Every 2 hours switch
            return aggressive_cooling_policy(obs, info)
        else:
            return aggressive_heating_policy(obs, info)
    except:
        return moderate_policy(obs, info)


def occupancy_based_policy(obs: np.ndarray, info: Dict) -> np.ndarray:
    """Adjust based on occupancy - captures load-dependent dynamics"""
    people = info.get('people_occupant', 0)
    
    if people > 10:  # High occupancy
        # Tight control: heating 21°C, cooling 23.5°C
        return np.array([0.6, -0.8], dtype=np.float32)
    elif people > 5:  # Medium occupancy
        # Moderate: heating 20°C, cooling 24°C
        return np.array([0.4, 0.2], dtype=np.float32)
    else:  # Low/no occupancy
        # Relaxed: heating 18°C, cooling 26°C
        return np.array([0.05, 0.6], dtype=np.float32)


# ============================================================================ #
#                              MAIN COLLECTION                                 #
# ============================================================================ #

def main():
    parser = argparse.ArgumentParser(
        description="Collect transition data for world model training"
    )
    parser.add_argument(
        '--building',
        type=str,
        choices=['5zone', 'warehouse', 'datacenter', 'office'],
        default='warehouse',
        help='Building type to collect data from'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='Number of episodes to collect (default: 10)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./world_model_data',
        help='Output directory for transition data'
    )
    parser.add_argument(
        '--weather',
        type=str,
        choices=['hot', 'mixed', 'cool'],
        default='hot',
        help='Weather conditions (default: hot)'
    )
    parser.add_argument(
        '--apply-noise',
        action='store_true',
        help='Apply sensor noise (production-like conditions)'
    )
    
    args = parser.parse_args()
    
    # Map building types to Sinergym environments
    env_map = {
        '5zone': f'Eplus-5zone-{args.weather}-continuous-v1',
        'warehouse': f'Eplus-warehouse-{args.weather}-continuous-v1',
        'datacenter': f'Eplus-datacenter-{args.weather}-continuous-v1',
        'office': f'Eplus-office-{args.weather}-continuous-v1',
    }
    
    env_name = env_map[args.building]
    
    print("="*80)
    print("🎯 World Model Transition Data Collection")
    print("="*80)
    print(f"Building: {args.building}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {args.episodes}")
    print(f"Weather: {args.weather}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    # Create environment
    env = gym.make(
        env_name,
        max_ep_data_store_num=args.episodes  # Store all episodes
    )
    
    # Apply wrappers (same as training for consistency)
    env = NormalizeObservation(env)
    env = NormalizeAction(env)
    
    if args.apply_noise:
        print("\n🔊 Applying sensor noise (production-like conditions)")
        internal_sensor_noise = {
            'air_temperature': 0.2,
            'air_humidity': 1.0,
            'people_occupant': 1.0,
            'HVAC_electricity_demand_rate': 100.0,
        }
        env = InternalSensorNoiseWrapper(env, internal_sensor_noise)
    
    # Initialize collector
    collector = TransitionCollector(output_dir=args.output_dir)
    
    # Define exploration strategies and their distribution
    strategies = [
        ('random', random_policy, 0.3),  # 30% random exploration
        ('moderate', moderate_policy, 0.2),  # 20% typical operation
        ('aggressive_cooling', aggressive_cooling_policy, 0.15),  # 15%
        ('aggressive_heating', aggressive_heating_policy, 0.15),  # 15%
        ('oscillating', oscillating_policy, 0.1),  # 10% transients
        ('occupancy_based', occupancy_based_policy, 0.1)  # 10% adaptive
    ]
    
    # Calculate episodes per strategy
    episodes_per_strategy = []
    remaining_episodes = args.episodes
    
    for name, policy, ratio in strategies[:-1]:
        count = int(args.episodes * ratio)
        episodes_per_strategy.append((name, policy, count))
        remaining_episodes -= count
    
    # Last strategy gets remaining episodes
    episodes_per_strategy.append((strategies[-1][0], strategies[-1][1], remaining_episodes))
    
    print("\n📋 Collection Strategy:")
    for name, _, count in episodes_per_strategy:
        print(f"   {name}: {count} episodes")
    
    # Collect transitions
    episode_stats = []
    episode_num = 0
    
    for strategy_name, policy_fn, num_episodes in episodes_per_strategy:
        for i in range(num_episodes):
            stats = collector.collect_episode(
                env, 
                policy_fn, 
                episode_num, 
                strategy_name
            )
            episode_stats.append(stats)
            episode_num += 1
            
            collector.metadata['collection_strategy'].append({
                'episode': episode_num,
                'strategy': strategy_name
            })
    
    # Save all collected data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    building_id = f"{args.building}_{args.weather}_{timestamp}"
    collector.save_transitions(building_id)
    
    # Save episode statistics
    stats_df = pd.DataFrame(episode_stats)
    stats_path = os.path.join(args.output_dir, f'episode_stats_{building_id}.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"💾 Saved episode statistics to {stats_path}")
    
    print("\n" + "="*80)
    print("✅ Data collection complete!")
    print("="*80)
    print(f"\n📁 Output files in: {args.output_dir}")
    print(f"   - transitions_{building_id}.pkl (main data)")
    print(f"   - transitions_{building_id}.csv (human-readable)")
    print(f"   - metadata_{building_id}.json (collection info)")
    print(f"   - episode_stats_{building_id}.csv (episode summaries)")
    
    # Clean up
    env.close()
    
    print("\n🚀 Next steps:")
    print("   1. Run this script for other building types (warehouse, datacenter)")
    print("   2. Extract building metadata from epJSON files")
    print("   3. Train world model with collected transitions")
    print("   4. Validate physics constraints (PINNs)")


if __name__ == '__main__':
    main()