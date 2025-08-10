#!/usr/bin/env python3
"""Monitor RETGEN v2 training progress."""

import time
import os
from pathlib import Path

def monitor_training():
    """Monitor the training progress."""
    log_file = "retgen_v2_training.log"
    
    print("Monitoring RETGEN v2 Training Progress")
    print("="*50)
    
    while True:
        if os.path.exists(log_file):
            # Get file size
            size = os.path.getsize(log_file) / 1024  # KB
            
            # Read last lines
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            # Find progress indicators
            for line in reversed(lines[-100:]):
                if "Training batches:" in line or "Extracting patterns:" in line or "Encoding:" in line:
                    print(f"\r{line.strip()}", end="", flush=True)
                    break
                elif "TRAINING COMPLETE" in line:
                    print("\n\nTraining completed!")
                    print("="*50)
                    # Print summary
                    for summary_line in lines[-30:]:
                        if any(keyword in summary_line for keyword in ["Total patterns", "Memory", "Compression", "time"]):
                            print(summary_line.strip())
                    return
                elif "ERROR" in line or "Traceback" in line:
                    print(f"\n\nError detected: {line.strip()}")
                    return
            
            print(f" | Log size: {size:.1f} KB", end="", flush=True)
        
        time.sleep(5)  # Check every 5 seconds

if __name__ == "__main__":
    monitor_training()