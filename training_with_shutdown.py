import subprocess
import os
import time
from datetime import datetime

def run_training_with_shutdown(config_file):
    """Run training and shutdown when complete"""
    start_time = datetime.now()
    print(f"🚀 Starting training at {start_time}")
    
    # Start training
    process = subprocess.Popen(['python', 'simple_train.py', '--config', config_file])
    
    # Monitor process
    while True:
        if process.poll() is not None:  # Process finished
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"✅ Training completed at {end_time}")
            print(f"⏱️  Total duration: {duration}")
            
            # Give 2 minutes to save everything
            print("⏳ Waiting 2 minutes before shutdown...")
            time.sleep(120)
            
            # Shutdown the machine
            print("🔌 Shutting down machine...")
            os.system("sudo shutdown now")
            break
            
        time.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    CONFIG_FILE = "SAC_sweep_example.yaml"
    run_training_with_shutdown(CONFIG_FILE)