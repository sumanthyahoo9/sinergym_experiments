import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
import sys

# Add your project path
sys.path.append(".")

# Import SinerGym reward classes and wrappers
from sinergym.utils.rewards import LinearReward
from sinergym.utils.wrappers import NormalizeObservation, NormalizeAction, LoggerWrapper, CSVLogger

# Import your custom reward (if still using it)
try:
    from simple_train import EnhancedLinearReward
except ImportError:
    print("Warning: Could not import EnhancedLinearReward. Using LinearReward for PPO analysis.")
    EnhancedLinearReward = LinearReward

def run_baseline_controller(episodes=1, config_path="PPO_sweep_example.yaml"):
    """Run building with simple rule-based controller"""
    print("🏠 Running Baseline Controller...")
    
    # Create environment with same config as training
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    env_params = {}
    if config.get('environment_parameters'):
        env_params = config['environment_parameters'].copy()
        
        # Handle reward (use LinearReward for baseline)
        if 'reward' in env_params:
            env_params['reward'] = LinearReward  # Use LinearReward class for baseline
            
        # Handle weather variability
        if 'weather_variability' in env_params:
            for var_name, var_param in env_params['weather_variability'].items():
                if isinstance(var_param, list):
                    env_params['weather_variability'][var_name] = tuple(var_param)
    
    # Create baseline environment (NO wrappers - raw actions)
    env_params['env_name'] = 'baseline_energy_analysis'
    env = gym.make(config['environment'], **env_params)
    
    total_results = []
    
    for episode in range(episodes):
        print(f"  Episode {episode + 1}/{episodes}")
        
        obs, _ = env.reset()
        episode_energy = 0
        episode_rewards = []
        
        for step in range(35040):  # Full year simulation
            # Use conservative baseline: 22°C heating, 25°C cooling
            action = np.array([22.0, 25.0], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            
            # Extract energy consumption from info dictionary (most reliable)
            if 'total_power_demand' in info:
                energy_demand = info['total_power_demand']  # Watts
            else:
                # Fallback: use observation index 15 (raw power demand)
                energy_demand = obs[15] if len(obs) > 15 else 0
            
            episode_energy += energy_demand
            
            if terminated or truncated:
                break
        
        total_results.append({
            'episode': episode + 1,
            'total_energy': episode_energy,
            'total_reward': sum(episode_rewards),
            'avg_reward': np.mean(episode_rewards)
        })
        
        print(f"    Episode {episode + 1} Energy: {episode_energy:.2f}")
    
    env.close()
    return total_results

def run_ppo_controller(model_path, episodes=1, config_path="PPO_sweep_example.yaml"):
    """Run building with trained PPO agent"""
    print("🤖 Running PPO Controller...")
    
    # Load trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = PPO.load(model_path)
    print(f"  Loaded model: {model_path}")
    
    # Create environment with same config as training
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    env_params = {}
    if config.get('environment_parameters'):
        env_params = config['environment_parameters'].copy()
        
        # Handle custom reward - same logic as training script
        if 'reward' in env_params:
            if env_params['reward'] == 'EnhancedLinearReward':
                env_params['reward'] = EnhancedLinearReward
            elif env_params['reward'] == 'LinearReward':
                env_params['reward'] = LinearReward
            else:
                # Try to evaluate other reward types
                try:
                    env_params['reward'] = eval(env_params['reward'])
                except:
                    print(f"Warning: Could not evaluate reward {env_params['reward']}, using LinearReward")
                    env_params['reward'] = LinearReward
                
        # Handle weather variability
        if 'weather_variability' in env_params:
            for var_name, var_param in env_params['weather_variability'].items():
                if isinstance(var_param, list):
                    env_params['weather_variability'][var_name] = tuple(var_param)
    
    # Create PPO environment with SAME wrappers as training
    env_params['env_name'] = 'ppo_energy_analysis'
    env = gym.make(config['environment'], **env_params)
    
    # Apply the same wrappers that were used during training
    if config.get('wrappers'):
        for wrapper_config in config['wrappers']:
            wrapper_name = wrapper_config[0]
            wrapper_params = wrapper_config[1]
            
            # Skip CSVLogger and LoggerWrapper for analysis (just need action/obs normalization)
            if wrapper_name in ['CSVLogger', 'LoggerWrapper']:
                continue
                
            wrapper_class = eval(wrapper_name)
            for name, value in wrapper_params.items():
                if isinstance(value, str) and '.' in value and '.txt' not in value:
                    wrapper_params[name] = eval(value)
            
            env = wrapper_class(env=env, **wrapper_params)
            print(f"  Applied wrapper: {wrapper_name}")
    else:
        # Fallback: Apply essential wrappers manually if not in config
        env = NormalizeObservation(env)
        env = NormalizeAction(env)
        print("  Applied default wrappers: NormalizeObservation, NormalizeAction")
    
    total_results = []
    
    for episode in range(episodes):
        print(f"  Episode {episode + 1}/{episodes}")
        
        obs, _ = env.reset()
        episode_energy = 0
        episode_rewards = []
        
        for step in range(35040):  # Full year simulation
            # PPO agent decides actions
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            
            # Extract energy consumption from info dictionary (most reliable)
            if 'total_power_demand' in info:
                energy_demand = info['total_power_demand']  # Watts
            else:
                # Fallback: use observation index 15 (raw power demand)
                energy_demand = obs[15] if len(obs) > 15 else 0
            
            episode_energy += energy_demand
            
            if terminated or truncated:
                break
        
        total_results.append({
            'episode': episode + 1,
            'total_energy': episode_energy,
            'total_reward': sum(episode_rewards),
            'avg_reward': np.mean(episode_rewards)
        })
        
        print(f"    Episode {episode + 1} Energy: {episode_energy:.2f}")
    
    env.close()
    return total_results

def calculate_savings(baseline_results, ppo_results, electricity_rate=0.12):
    """Calculate energy and cost savings"""
    print("\n📊 Calculating Energy Savings...")
    
    # Average across episodes
    baseline_energy = np.mean([r['total_energy'] for r in baseline_results])
    ppo_energy = np.mean([r['total_energy'] for r in ppo_results])
    
    baseline_reward = np.mean([r['total_reward'] for r in baseline_results])
    ppo_reward = np.mean([r['total_reward'] for r in ppo_results])
    
    # Convert demand rate to annual consumption
    # Energy demand is in watts, timestep is 15 minutes
    timestep_hours = 15 / 60  # 0.25 hours
    
    baseline_kwh = (baseline_energy * timestep_hours) / 1000  # Convert W to kW
    ppo_kwh = (ppo_energy * timestep_hours) / 1000
    
    # Calculate savings
    energy_savings_kwh = baseline_kwh - ppo_kwh
    savings_percentage = (energy_savings_kwh / baseline_kwh) * 100
    cost_savings = energy_savings_kwh * electricity_rate
    
    # Calculate reward improvement
    reward_improvement = ppo_reward - baseline_reward
    reward_improvement_pct = (reward_improvement / abs(baseline_reward)) * 100 if baseline_reward != 0 else 0
    
    results = {
        'baseline_kwh': baseline_kwh,
        'ppo_kwh': ppo_kwh,
        'energy_savings_kwh': energy_savings_kwh,
        'savings_percentage': savings_percentage,
        'cost_savings_usd': cost_savings,
        'baseline_reward': baseline_reward,
        'ppo_reward': ppo_reward,
        'reward_improvement': reward_improvement,
        'reward_improvement_pct': reward_improvement_pct
    }
    
    return results

def create_comparison_plot(baseline_results, ppo_results, savings_results):
    """Create visualization of energy comparison"""
    print("📈 Creating comparison plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Energy Consumption Comparison
    categories = ['Baseline\n(Rule-based)', 'PPO Agent']
    energy_values = [savings_results['baseline_kwh'], savings_results['ppo_kwh']]
    colors = ['lightcoral', 'lightgreen']
    
    bars = ax1.bar(categories, energy_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Annual Energy Consumption (kWh)')
    ax1.set_title('Energy Consumption Comparison')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, energy_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energy_values)*0.01,
                f'{value:,.0f} kWh', ha='center', va='bottom', fontweight='bold')
    
    # Add savings annotation
    savings_text = f"Savings: {savings_results['energy_savings_kwh']:,.0f} kWh\n({savings_results['savings_percentage']:.1f}%)"
    ax1.text(0.5, max(energy_values) * 0.8, savings_text, 
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
             fontsize=12, fontweight='bold')
    
    # Plot 2: Reward Comparison
    reward_categories = ['Baseline', 'PPO Agent']
    reward_values = [savings_results['baseline_reward'], savings_results['ppo_reward']]
    reward_colors = ['lightcoral', 'lightgreen']
    
    bars2 = ax2.bar(reward_categories, reward_values, color=reward_colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Performance Reward Comparison')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, reward_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(reward_values) - min(reward_values))*0.02,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Add improvement annotation
    improvement_text = f"Improvement: {savings_results['reward_improvement']:.0f}\n({savings_results['reward_improvement_pct']:.1f}%)"
    ax2.text(0.5, min(reward_values) + (max(reward_values) - min(reward_values)) * 0.2, improvement_text,
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('energy_savings_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  Plot saved as: energy_savings_comparison.png")

def print_results(savings_results, baseline_results, ppo_results):
    """Print detailed results with business case information"""
    print("\n" + "="*80)
    print("🎯 ENERGY SAVINGS ANALYSIS RESULTS")
    print("="*80)
    
    print(f"📊 Energy Consumption Analysis:")
    print(f"  Baseline Controller:     {savings_results['baseline_kwh']:>12,.0f} kWh/year")
    print(f"  PPO Agent:               {savings_results['ppo_kwh']:>12,.0f} kWh/year")
    print(f"  Energy Savings:          {savings_results['energy_savings_kwh']:>12,.0f} kWh/year")
    print(f"  Savings Percentage:      {savings_results['savings_percentage']:>12.1f}%")
    
    print(f"\n💰 Financial Impact:")
    print(f"  Annual Cost Savings:     ${savings_results['cost_savings_usd']:>12,.0f}")
    print(f"  5-year Savings:          ${savings_results['cost_savings_usd']*5:>12,.0f}")
    print(f"  10-year Savings:         ${savings_results['cost_savings_usd']*10:>12,.0f}")
    
    print(f"\n🏆 Performance Metrics:")
    print(f"  Baseline Reward:         {savings_results['baseline_reward']:>12,.0f}")
    print(f"  PPO Reward:              {savings_results['ppo_reward']:>12,.0f}")
    print(f"  Reward Improvement:      {savings_results['reward_improvement']:>12,.0f} ({savings_results['reward_improvement_pct']:.1f}%)")
    
    # Environmental impact
    co2_reduction = savings_results['energy_savings_kwh'] * 0.5  # kg CO2 per kWh (average grid)
    print(f"\n🌍 Environmental Impact:")
    print(f"  CO2 Reduction:           {co2_reduction:>12,.0f} kg/year ({co2_reduction/1000:.1f} tons)")
    
    # Building deployment insights
    print(f"\n🏢 Real Building Deployment Potential:")
    if savings_results['savings_percentage'] > 15:
        deployment_rating = "🌟 EXCELLENT"
    elif savings_results['savings_percentage'] > 10:
        deployment_rating = "✅ GOOD"
    elif savings_results['savings_percentage'] > 5:
        deployment_rating = "⚠️  MODERATE"
    else:
        deployment_rating = "❌ POOR"
        
    print(f"  Deployment Viability:    {deployment_rating}")
    print(f"  Typical Payback Period:  {2000 / max(savings_results['cost_savings_usd'], 1):.1f} years")
    print(f"  Recommended for:         {'Large commercial buildings' if savings_results['cost_savings_usd'] > 1000 else 'Small/medium buildings'}")
    
    # Detailed episode statistics
    print(f"\n📈 Episode Statistics:")
    if len(baseline_results) > 1:
        baseline_energies = [r['total_energy'] for r in baseline_results]
        ppo_energies = [r['total_energy'] for r in ppo_results]
        print(f"  Baseline Std Dev:        {np.std(baseline_energies):>12,.0f} kWh")
        print(f"  PPO Std Dev:             {np.std(ppo_energies):>12,.0f} kWh")
        print(f"  Consistency (PPO):       {'High' if np.std(ppo_energies) < np.std(baseline_energies) else 'Similar'}")
    
    print("="*80)
    print("💡 Key Insights for Real Building Deployment:")
    
    if savings_results['savings_percentage'] > 0:
        print(f"  ✓ PPO agent successfully reduces energy consumption")
        print(f"  ✓ Estimated annual savings: ${savings_results['cost_savings_usd']:,.0f}")
        if savings_results['reward_improvement'] > 0:
            print(f"  ✓ Maintains/improves comfort while saving energy")
        else:
            print(f"  ⚠️  Energy savings may come with comfort trade-offs")
    else:
        print(f"  ❌ PPO agent uses MORE energy than baseline")
        print(f"  💡 Consider: different reward function, more training, or different baseline")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Analyze energy savings of PPO HVAC controller')
    parser.add_argument('--model', required=True, help='Path to trained PPO model (e.g., ./models/PPO-...-model)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run for each controller')
    parser.add_argument('--config', default='PPO_sweep_example.yaml', help='Path to config file')
    parser.add_argument('--electricity-rate', type=float, default=0.12, help='Electricity rate ($/kWh)')
    
    args = parser.parse_args()
    
    try:
        # Run baseline controller
        baseline_results = run_baseline_controller(episodes=args.episodes, config_path=args.config)
        
        # Run PPO controller
        ppo_results = run_ppo_controller(args.model, episodes=args.episodes, config_path=args.config)
        
        # Calculate savings
        savings_results = calculate_savings(baseline_results, ppo_results, args.electricity_rate)
        
        # Create visualizations
        create_comparison_plot(baseline_results, ppo_results, savings_results)
        
        # Print detailed results
        print_results(savings_results, baseline_results, ppo_results)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == '__main__':
    main()