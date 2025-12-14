"""
World Model Data Collection with Automatic Shutdown
===================================================

Run this script to collect transition data and initiate automatic shutdown.
This allows unattended overnight data collection on cloud VMs.

Example usage:
    nohup python collect_transitions_with_shutdown.py --building 5zone --episodes 30 > collection_$(date +%Y%m%d_%H%M%S).log 2>&1 &

The PID will be displayed - note it down for monitoring.
"""

import subprocess
import os
import sys
import time
import argparse
from datetime import datetime


def run_collection_with_shutdown(building_type, episodes, weather='hot', apply_noise=True):
    """Run transition data collection and shutdown when complete"""
    
    import os  # Move import to the TOP
    
    start_time = datetime.now()
    print("=" * 80)
    print("🚀 World Model Data Collection with Auto-Shutdown")
    print("=" * 80)
    print(f"Start time: {start_time}")
    print(f"Building: {building_type}")
    print(f"Episodes: {episodes}")
    print(f"Weather: {weather}")
    print(f"Sensor noise: {apply_noise}")
    print(f"PID: {os.getpid()}")
    print("=" * 80)
    
    # Get absolute path for output directory
    output_dir = os.path.join(os.getcwd(), "world_model_data")
    
    # Build command
    cmd = [
        'python', 
        'collect_transitions_for_world_model.py',
        '--building', building_type,
        '--episodes', str(episodes),
        '--weather', weather,
        '--output-dir', output_dir
    ]
    
    if apply_noise:
        cmd.append('--apply-noise')
    
    print(f"\n📝 Command: {' '.join(cmd)}\n")
    
    # Start collection process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Stream output in real-time
    print("📊 Collection output:")
    print("-" * 80)
    
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
            sys.stdout.flush()
    
    # Wait for process to complete
    return_code = process.wait()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("-" * 80)
    print(f"\n✅ Collection completed at {end_time}")
    print(f"⏱️  Total duration: {duration}")
    print(f"📦 Return code: {return_code}")
    
    if return_code == 0:
        print("✨ Collection successful!")
    else:
        print(f"⚠️  Collection exited with code {return_code}")
    
    # Give 2 minutes to ensure all files are written
    print("\n⏳ Waiting 2 minutes to ensure all files are saved...")
    time.sleep(120)
    
    # Shutdown the machine
    print("🔌 Initiating machine shutdown...")
    print("=" * 80)
    os.system("sudo shutdown -h now")


def main():
    parser = argparse.ArgumentParser(
        description="Collect transition data with automatic shutdown"
    )
    parser.add_argument(
        '--building',
        type=str,
        choices=['5zone', 'warehouse', 'datacenter'],
        required=True,
        help='Building type to collect data from'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=30,
        help='Number of episodes to collect (default: 30)'
    )
    parser.add_argument(
        '--weather',
        type=str,
        choices=['hot', 'mixed', 'cool'],
        default='hot',
        help='Weather conditions (default: hot)'
    )
    parser.add_argument(
        '--no-noise',
        action='store_true',
        help='Disable sensor noise (default: noise enabled)'
    )
    
    args = parser.parse_args()
    
    run_collection_with_shutdown(
        building_type=args.building,
        episodes=args.episodes,
        weather=args.weather,
        apply_noise=not args.no_noise
    )


if __name__ == "__main__":
    main()