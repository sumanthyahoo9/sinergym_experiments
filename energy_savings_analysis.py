import argparse
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import SAC
import sys

# Add your project path
sys.path.append(".")

# Import SinerGym reward classes and wrappers
from sinergym.utils.rewards import LinearReward
from sinergym.utils.wrappers import NormalizeObservation, NormalizeAction
from simple_train import EnhancedLinearReward

def run_baseline_controller(episodes=1, config_path="SAC_sweep_example.yaml"):
    """Run building with simple rule-based controller"""
    print("🏠 Running Baseline Controller...")
    
    # Create environment with same config as training
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    env_params = {}
    if config.get('environment_parameters'):
        env_params = config['environment_parameters'].copy()
        
        # Handle reward configuration
        if 'reward' in env_params:
            reward_type = env_params['reward']
            
            # Always use LinearReward for baseline (simpler, more stable)
            env_params['reward'] = LinearReward
            
            # Fix reward_kwargs for LinearReward compatibility
            if 'reward_kwargs' in env_params:
                old_kwargs = env_params['reward_kwargs']
                
                # LinearReward parameters (NO temperature_variable or energy_variable)
                env_params['reward_kwargs'] = {
                    'temperature_variables': ['air_temperature'],  # PLURAL + LIST
                    'energy_variables': ['HVAC_electricity_demand_rate'],  # PLURAL + LIST
                    'energy_weight': old_kwargs.get('energy_weight', 0.5),
                    'lambda_energy': old_kwargs.get('lambda_energy', 1.0),
                    'lambda_temperature': old_kwargs.get('lambda_temperature', 1.0),
                    'range_comfort_winter': old_kwargs.get('range_comfort_winter', (20.0, 23.5)),
                    'range_comfort_summer': old_kwargs.get('range_comfort_summer', (23.0, 26.0)),
                    'summer_start': old_kwargs.get('summer_start', (6, 1)),
                    'summer_final': old_kwargs.get('summer_final', (9, 30))
                }
                
                print(f"  Using LinearReward for baseline (original config: {reward_type})")
        
        # Handle weather variability - convert lists to tuples
        if 'weather_variability' in env_params:
            for var_name, var_param in env_params['weather_variability'].items():
                if isinstance(var_param, list):
                    env_params['weather_variability'][var_name] = tuple(var_param)
    
    # Create baseline environment
    env_params['env_name'] = 'baseline_energy_analysis'
    env = gym.make(config['environment'], **env_params)
    
    total_results = []
    
    for episode in range(episodes):
        print(f"  Episode {episode + 1}/{episodes}")
        
        obs, _ = env.reset()
        episode_energy = 0
        episode_rewards = []
        steps_completed = 0
        
        for step in range(35040):  # Full year simulation
            # Conservative baseline: 22°C heating, 25°C cooling
            action = np.array([22.0, 25.0], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            steps_completed = step + 1
            
            # Extract energy from info (most reliable)
            if 'total_power_demand' in info:
                energy_demand = info['total_power_demand']
            else:
                energy_demand = obs[15] if len(obs) > 15 else 0
            
            episode_energy += energy_demand
            
            if terminated or truncated:
                print(f"    ⚠️ Episode {episode + 1} terminated early at step {steps_completed} ({(steps_completed/35040)*100:.1f}%)")
                break
        
        completion_pct = (steps_completed / 35040) * 100
        total_results.append({
            'episode': episode + 1,
            'total_energy': episode_energy,
            'total_reward': sum(episode_rewards),
            'avg_reward': np.mean(episode_rewards),
            'steps_completed': steps_completed,
            'completion_pct': completion_pct
        })
        
        if completion_pct >= 100:
            print(f"    ✓ Episode {episode + 1} Energy: {episode_energy:.2f} Wh ({steps_completed} steps, 100%)")
        else:
            print(f"    ✗ Episode {episode + 1} Energy: {episode_energy:.2f} Wh ({steps_completed} steps, {completion_pct:.1f}%)")
    
    env.close()
    
    # Check completion rate
    complete_episodes = sum(1 for r in total_results if r['completion_pct'] >= 100)
    print(f"\n  Baseline Summary: {complete_episodes}/{episodes} episodes completed (100%)")
    
    return total_results

def run_sac_controller(model_path, episodes=1, config_path="SAC_sweep_example.yaml"):
    """Run building with trained SAC agent"""
    print("\n🤖 Running SAC Controller...")
    
    # Load trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = SAC.load(model_path)
    print(f"  Loaded model: {model_path}")
    
    # Create environment with same config as training
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    env_params = {}
    if config.get('environment_parameters'):
        env_params = config['environment_parameters'].copy()
        
        # Handle custom reward - KEEP the same reward as training
        if 'reward' in env_params:
            if env_params['reward'] == 'EnhancedLinearReward':
                env_params['reward'] = EnhancedLinearReward
                print(f"  Using EnhancedLinearReward for SAC evaluation")
            elif env_params['reward'] == 'LinearReward':
                env_params['reward'] = LinearReward
                print(f"  Using LinearReward for SAC evaluation")
            else:
                # Try to evaluate other reward types
                try:
                    env_params['reward'] = eval(env_params['reward'])
                except:
                    print(f"  Warning: Could not evaluate reward {env_params['reward']}, using LinearReward")
                    env_params['reward'] = LinearReward
                
        # Handle weather variability
        if 'weather_variability' in env_params:
            for var_name, var_param in env_params['weather_variability'].items():
                if isinstance(var_param, list):
                    env_params['weather_variability'][var_name] = tuple(var_param)
    
    # Create SAC environment with SAME wrappers as training
    env_params['env_name'] = 'SAC_energy_analysis'
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
        comfort_violations = 0
        total_timesteps = 0
        steps_completed = 0
        
        for step in range(35040):  # Full year simulation
            # SAC agent decides actions
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            steps_completed = step + 1

            # Track comfort violations
            if "total_temperature_violation" in info:
                total_temp_violation = info["total_temperature_violation"]
                total_timesteps += 1
                comfort_violations += (total_temp_violation > 0)  # Binary, violated or not
            
            episode_rewards.append(reward)
            
            # Extract energy consumption from info dictionary (most reliable)
            if 'total_power_demand' in info:
                energy_demand = info['total_power_demand']  # Watts
            else:
                # Fallback: use observation index 15 (raw power demand)
                energy_demand = obs[15] if len(obs) > 15 else 0
            
            episode_energy += energy_demand
            
            if terminated or truncated:
                print(f"    ⚠️ Episode {episode + 1} terminated early at step {steps_completed} ({(steps_completed/35040)*100:.1f}%)")
                break
        
        comfort_violation_percent = (comfort_violations / total_timesteps * 100) if total_timesteps > 0 else 0
        completion_pct = (steps_completed / 35040) * 100
        
        total_results.append({
            'episode': episode + 1,
            'total_energy': episode_energy,
            'total_reward': sum(episode_rewards),
            'avg_reward': np.mean(episode_rewards),
            'comfort_violation_percentage': comfort_violation_percent,
            'steps_completed': steps_completed,
            'completion_pct': completion_pct
        })
        
        if completion_pct >= 100:
            print(f"    ✓ Episode {episode + 1} Energy: {episode_energy:.2f} Wh | Comfort: {comfort_violation_percent:.1f}% ({steps_completed} steps, 100%)")
        else:
            print(f"    ✗ Episode {episode + 1} Energy: {episode_energy:.2f} Wh | Comfort: {comfort_violation_percent:.1f}% ({steps_completed} steps, {completion_pct:.1f}%)")
    
    env.close()
    
    # Check completion rate
    complete_episodes = sum(1 for r in total_results if r['completion_pct'] >= 100)
    avg_comfort = np.mean([r['comfort_violation_percentage'] for r in total_results])
    print(f"\n  SAC Summary: {complete_episodes}/{episodes} episodes completed (100%)")
    print(f"  Average comfort violation: {avg_comfort:.1f}%")
    
    return total_results

def calculate_savings(baseline_results, sac_results, electricity_rate=0.12):
    """Calculate energy and cost savings"""
    print("\n📊 Calculating Energy Savings...")
    
    # Filter out incomplete episodes for fair comparison
    complete_baseline = [r for r in baseline_results if r['completion_pct'] >= 100]
    complete_sac = [r for r in sac_results if r['completion_pct'] >= 100]
    
    if len(complete_baseline) == 0 or len(complete_sac) == 0:
        print("  ⚠️ WARNING: Some episodes did not complete!")
        print(f"    Baseline: {len(complete_baseline)}/{len(baseline_results)} complete")
        print(f"    SAC: {len(complete_sac)}/{len(sac_results)} complete")
        print("    Using ALL episodes (including incomplete) - results may be inaccurate!")
        complete_baseline = baseline_results
        complete_sac = sac_results
    else:
        print(f"  Using {len(complete_baseline)} baseline and {len(complete_sac)} SAC episodes (all 100% complete)")
    
    # Average across complete episodes only
    baseline_energy = np.mean([r['total_energy'] for r in complete_baseline])
    sac_energy = np.mean([r['total_energy'] for r in complete_sac])
    
    baseline_reward = np.mean([r['total_reward'] for r in complete_baseline])
    sac_reward = np.mean([r['total_reward'] for r in complete_sac])
    
    # Convert demand rate to annual consumption
    # Energy demand is in watts, timestep is 15 minutes
    timestep_hours = 15 / 60  # 0.25 hours
    
    baseline_kwh = (baseline_energy * timestep_hours) / 1000  # Convert W to kWh
    sac_kwh = (sac_energy * timestep_hours) / 1000
    
    # Calculate savings
    energy_savings_kwh = baseline_kwh - sac_kwh
    savings_percentage = (energy_savings_kwh / baseline_kwh) * 100
    cost_savings = energy_savings_kwh * electricity_rate
    
    # Calculate reward improvement
    reward_improvement = sac_reward - baseline_reward
    reward_improvement_pct = (reward_improvement / abs(baseline_reward)) * 100 if baseline_reward != 0 else 0
    
    # Calculate standard deviations for consistency
    baseline_std = np.std([r['total_energy'] for r in complete_baseline])
    sac_std = np.std([r['total_energy'] for r in complete_sac])
    
    results = {
        'baseline_kwh': baseline_kwh,
        'sac_kwh': sac_kwh,
        'energy_savings_kwh': energy_savings_kwh,
        'savings_percentage': savings_percentage,
        'cost_savings_usd': cost_savings,
        'baseline_reward': baseline_reward,
        'sac_reward': sac_reward,
        'reward_improvement': reward_improvement,
        'reward_improvement_pct': reward_improvement_pct,
        'baseline_std': baseline_std,
        'sac_std': sac_std,
        'episodes_used': {
            'baseline': len(complete_baseline),
            'sac': len(complete_sac)
        }
    }
    
    avg_comfort_violation = np.mean([r['comfort_violation_percentage'] for r in complete_sac])
    print(f"  Average comfort violation: {avg_comfort_violation:.1f}%")
    
    return results

def create_comparison_plot(baseline_results, sac_results, savings_results):
    """Create visualization of energy comparison"""
    print("\n📈 Creating comparison plots...")
    
    # Filter to complete episodes only
    complete_baseline = [r for r in baseline_results if r.get('completion_pct', 100) >= 100]
    complete_sac = [r for r in sac_results if r.get('completion_pct', 100) >= 100]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # Plot 1: Energy Consumption Comparison
    categories = ['Baseline\n(Rule-based)', 'SAC Agent']
    energy_values = [savings_results['baseline_kwh'], savings_results['sac_kwh']]
    colors = ['lightcoral', 'lightgreen']
    
    bars = ax1.bar(categories, energy_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Annual Energy Consumption (kWh)', fontsize=12)
    ax1.set_title('Energy Consumption Comparison', fontsize=14, fontweight='bold')
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
    reward_categories = ['Baseline', 'SAC Agent']
    reward_values = [savings_results['baseline_reward'], savings_results['sac_reward']]
    reward_colors = ['lightcoral', 'lightgreen']
    
    bars2 = ax2.bar(reward_categories, reward_values, color=reward_colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Total Reward', fontsize=12)
    ax2.set_title('Performance Reward Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, reward_values):
        y_offset = (max(reward_values) - min(reward_values)) * 0.02
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Add improvement annotation
    improvement_text = f"Improvement: {savings_results['reward_improvement']:.0f}\n({savings_results['reward_improvement_pct']:.1f}%)"
    y_pos = min(reward_values) + (max(reward_values) - min(reward_values)) * 0.2
    ax2.text(0.5, y_pos, improvement_text,
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
             fontsize=12, fontweight='bold')
    
    # Plot 3: Energy Distribution across episodes
    baseline_energies = np.array([float(r["total_energy"]) for r in complete_baseline])
    sac_energies = np.array([float(r["total_energy"]) for r in complete_sac])
    
    print(f"  Baseline energy array shape: {baseline_energies.shape}")
    print(f"  SAC energy array shape: {sac_energies.shape}")
    
    box_data = [baseline_energies, sac_energies]
    bp = ax3.boxplot(box_data, labels=["Baseline", "SAC"], patch_artist=True)
    
    # Color the boxes
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    
    ax3.set_ylabel("Energy consumed per episode (Wh)", fontsize=12)
    ax3.set_title("Energy Consistency Across Episodes", fontsize=14, fontweight='bold')
    ax3.grid(axis="y", alpha=0.3)
    
    # Add episode count annotation
    ax3.text(0.5, 0.95, f"n={len(baseline_energies)} episodes", 
             transform=ax3.transAxes, ha='center', va='top', fontsize=10)

    # Plot 4: Comfort violations distribution
    if "comfort_violation_percentage" in complete_sac[0]:
        comfort_data = [r["comfort_violation_percentage"] for r in complete_sac]
        ax4.hist(comfort_data, bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax4.set_xlabel("Comfort Violation Percentage (%)", fontsize=12)
        ax4.set_ylabel("Number of Episodes", fontsize=12)
        ax4.set_title("Comfort Violation Distribution", fontsize=14, fontweight='bold')
        mean_comfort = np.mean(comfort_data)
        ax4.axvline(mean_comfort, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_comfort:.1f}%')
        ax4.legend(fontsize=10)
        ax4.grid(axis='y', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Comfort data not available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
    
    plt.tight_layout()
    plt.savefig('energy_savings_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Plot saved as: energy_savings_comparison.png")

def print_results(savings_results, baseline_results, sac_results):
    """Print detailed results with business case information"""
    print("\n" + "="*80)
    print("🎯 ENERGY SAVINGS ANALYSIS RESULTS")
    print("="*80)
    
    print("\n📊 Energy Consumption Analysis:")
    print(f"  Baseline Controller:     {savings_results['baseline_kwh']:>12,.0f} kWh/year")
    print(f"  SAC Agent:               {savings_results['sac_kwh']:>12,.0f} kWh/year")
    print(f"  Energy Savings:          {savings_results['energy_savings_kwh']:>12,.0f} kWh/year")
    print(f"  Savings Percentage:      {savings_results['savings_percentage']:>12.1f}%")
    
    print("\n💰 Financial Impact:")
    print(f"  Annual Cost Savings:     ${savings_results['cost_savings_usd']:>12,.0f}")
    print(f"  5-year Savings:          ${savings_results['cost_savings_usd']*5:>12,.0f}")
    print(f"  10-year Savings:         ${savings_results['cost_savings_usd']*10:>12,.0f}")
    
    print("\n🏆 Performance Metrics:")
    print(f"  Baseline Reward:         {savings_results['baseline_reward']:>12,.0f}")
    print(f"  SAC Reward:              {savings_results['sac_reward']:>12,.0f}")
    print(f"  Reward Improvement:      {savings_results['reward_improvement']:>12,.0f} ({savings_results['reward_improvement_pct']:.1f}%)")
    
    # Environmental impact
    co2_reduction = savings_results['energy_savings_kwh'] * 0.5  # kg CO2 per kWh (average grid)
    print("\n🌍 Environmental Impact:")
    print(f"  CO2 Reduction:           {co2_reduction:>12,.0f} kg/year ({co2_reduction/1000:.1f} tons)")
    
    # Building deployment insights
    print("\n🏢 Real Building Deployment Potential:")
    if savings_results['savings_percentage'] > 15:
        deployment_rating = "🌟 EXCELLENT"
    elif savings_results['savings_percentage'] > 10:
        deployment_rating = "✅ GOOD"
    elif savings_results['savings_percentage'] > 5:
        deployment_rating = "⚠️  MODERATE"
    else:
        deployment_rating = "❌ POOR"
        
    print(f"  Deployment Viability:    {deployment_rating}")
    
    payback_years = 2000 / max(savings_results['cost_savings_usd'], 1)
    print(f"  Typical Payback Period:  {payback_years:.1f} years")
    print(f"  Recommended for:         {'Large commercial buildings' if savings_results['cost_savings_usd'] > 1000 else 'Small/medium buildings'}")
    
    # Episode statistics
    print("\n📈 Episode Statistics:")
    print(f"  Episodes Used:           Baseline: {savings_results['episodes_used']['baseline']}, SAC: {savings_results['episodes_used']['sac']}")
    print(f"  Baseline Std Dev:        {savings_results['baseline_std']:>12,.0f} Wh")
    print(f"  SAC Std Dev:             {savings_results['sac_std']:>12,.0f} Wh")
    consistency = 'High' if savings_results['sac_std'] < savings_results['baseline_std'] else 'Similar'
    print(f"  Consistency (SAC):       {consistency}")
    
    print("="*80)
    print("\n💡 Key Insights for Real Building Deployment:")
    
    if savings_results['savings_percentage'] > 0:
        print("  ✓ SAC agent successfully reduces energy consumption")
        print(f"  ✓ Estimated annual savings: ${savings_results['cost_savings_usd']:,.0f}")
        if savings_results['reward_improvement'] > 0:
            print(f"  ✓ Maintains/improves comfort while saving energy")
        else:
            print(f"  ⚠️  Energy savings may come with comfort trade-offs")
    else:
        print(f"  ❌ SAC agent uses MORE energy than baseline")
        print(f"  💡 Consider: different reward function, more training, or different baseline")
    
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze energy savings of SAC HVAC controller')
    parser.add_argument('--model', required=True, help='Path to trained SAC model (e.g., ./models/SAC-...-model)')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes to run for each controller')
    parser.add_argument('--config', default='SAC_sweep_example.yaml', help='Path to config file')
    parser.add_argument('--electricity-rate', type=float, default=0.12, help='Electricity rate ($/kWh)')
    
    args = parser.parse_args()
    
    try:
        # Run baseline controller
        baseline_results = run_baseline_controller(episodes=args.episodes, config_path=args.config)
        
        # Run the SAC controller
        sac_results = run_sac_controller(args.model, episodes=args.episodes, config_path=args.config)
        
        # Calculate savings
        savings_results = calculate_savings(baseline_results, sac_results, args.electricity_rate)
        
        # Create visualizations
        create_comparison_plot(baseline_results, sac_results, savings_results)
        
        # Print detailed results
        print_results(savings_results, baseline_results, sac_results)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()