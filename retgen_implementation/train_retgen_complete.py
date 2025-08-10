#!/usr/bin/env python3
"""Complete training script for RETGEN with monitoring and visualization."""

import os
import sys
import json
import time
import logging
import argparse
import threading
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import psutil
import GPUtil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("Warning: datasets library not available")
    
from core.retgen_fixed import RETGENFixed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Monitor training progress and system resources."""
    
    def __init__(self, output_dir: str = "training_status"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.status = {
            'start_time': None,
            'current_time': None,
            'elapsed_time': 0,
            'status': 'initializing',
            'samples_processed': 0,
            'total_samples': 0,
            'patterns_extracted': 0,
            'current_batch': 0,
            'total_batches': 0,
            'progress_percent': 0,
            'samples_per_second': 0,
            'estimated_time_remaining': 0,
            'gpu_memory_used': 0,
            'gpu_memory_total': 0,
            'gpu_utilization': 0,
            'cpu_percent': 0,
            'ram_used': 0,
            'ram_total': 0,
            'checkpoints': [],
            'errors': [],
            'logs': []
        }
        
        self.running = True
        self.monitor_thread = None
        
    def start(self):
        """Start monitoring in background thread."""
        self.status['start_time'] = datetime.now().isoformat()
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Training monitor started")
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("Training monitor stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                # Update system stats
                self._update_system_stats()
                
                # Save status to JSON
                self._save_status()
                
                # Sleep for 2 seconds
                time.sleep(2)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
    
    def _update_system_stats(self):
        """Update system resource statistics."""
        try:
            # CPU and RAM
            self.status['cpu_percent'] = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            self.status['ram_used'] = memory.used / (1024**3)  # GB
            self.status['ram_total'] = memory.total / (1024**3)  # GB
            
            # GPU stats
            if torch.cuda.is_available():
                self.status['gpu_memory_used'] = torch.cuda.memory_allocated() / (1024**3)
                self.status['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.status['gpu_utilization'] = gpu.load * 100
        except Exception as e:
            logger.warning(f"Error updating system stats: {e}")
    
    def _save_status(self):
        """Save current status to JSON file."""
        try:
            status_file = self.output_dir / "status.json"
            with open(status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving status: {e}")
    
    def update_training_progress(self, samples_processed: int, patterns_extracted: int, 
                                 current_batch: int, total_batches: int):
        """Update training progress metrics."""
        self.status['samples_processed'] = samples_processed
        self.status['patterns_extracted'] = patterns_extracted
        self.status['current_batch'] = current_batch
        self.status['total_batches'] = total_batches
        
        # Calculate progress
        if total_batches > 0:
            self.status['progress_percent'] = (current_batch / total_batches) * 100
        
        # Calculate speed
        if self.status['start_time']:
            elapsed = (datetime.now() - datetime.fromisoformat(self.status['start_time'])).total_seconds()
            self.status['elapsed_time'] = elapsed
            
            if elapsed > 0 and samples_processed > 0:
                self.status['samples_per_second'] = samples_processed / elapsed
                
                # Estimate remaining time
                if self.status['total_samples'] > 0:
                    remaining_samples = self.status['total_samples'] - samples_processed
                    self.status['estimated_time_remaining'] = remaining_samples / self.status['samples_per_second']
        
        self.status['current_time'] = datetime.now().isoformat()
    
    def add_checkpoint(self, path: str, samples: int, patterns: int):
        """Add checkpoint information."""
        checkpoint = {
            'path': path,
            'samples': samples,
            'patterns': patterns,
            'timestamp': datetime.now().isoformat()
        }
        self.status['checkpoints'].append(checkpoint)
        self.add_log(f"Checkpoint saved: {path}")
    
    def add_error(self, error: str):
        """Add error to status."""
        self.status['errors'].append({
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_log(self, message: str):
        """Add log message to status."""
        self.status['logs'].append({
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 100 logs
        if len(self.status['logs']) > 100:
            self.status['logs'] = self.status['logs'][-100:]
    
    def set_status(self, status: str):
        """Set current training status."""
        self.status['status'] = status
        self.add_log(f"Status changed to: {status}")


def load_training_data(max_samples: int = 1000000) -> List[str]:
    """Load training data from available datasets."""
    logger.info("Loading training data...")
    texts = []
    
    # First try to load from pre-generated file
    if os.path.exists("training_data.json"):
        logger.info("Loading from training_data.json...")
        try:
            with open("training_data.json", 'r') as f:
                texts = json.load(f)
            logger.info(f"Loaded {len(texts)} samples from training_data.json")
            return texts[:max_samples]
        except Exception as e:
            logger.warning(f"Could not load training_data.json: {e}")
    
    # Fallback to loading datasets
    try:
        # Try loading WikiText-103
        logger.info("Loading WikiText-103...")
        from datasets import load_dataset
        wikitext = load_dataset("wikitext", "wikitext-103-v1", split="train")
        
        for example in tqdm(wikitext, desc="Processing WikiText"):
            text = example['text'].strip()
            if len(text.split()) >= 10:  # Min 10 words
                texts.append(text)
                if len(texts) >= max_samples:
                    break
    except Exception as e:
        logger.error(f"Error loading WikiText: {e}")
    
    logger.info(f"Loaded {len(texts)} text samples")
    return texts


def train_retgen(
    texts: List[str],
    config: Dict,
    monitor: TrainingMonitor,
    batch_size: int = 32,
    checkpoint_interval: int = 10000
) -> RETGENFixed:
    """Train RETGEN model with monitoring."""
    
    monitor.set_status("training")
    monitor.status['total_samples'] = len(texts)
    
    # Initialize model
    logger.info("Initializing RETGEN model...")
    model = RETGENFixed(config)
    
    # Training variables
    total_patterns = 0
    samples_processed = 0
    
    # Calculate total batches
    total_batches = (len(texts) + batch_size - 1) // batch_size
    monitor.status['total_batches'] = total_batches
    
    try:
        # Process in batches
        for batch_idx in tqdm(range(0, len(texts), batch_size), desc="Training batches"):
            batch = texts[batch_idx:batch_idx+batch_size]
            
            # Train on batch
            try:
                model.train_on_texts(batch, batch_size=16)
                samples_processed += len(batch)
                total_patterns = model.total_patterns
                
                # Update monitor
                monitor.update_training_progress(
                    samples_processed=samples_processed,
                    patterns_extracted=total_patterns,
                    current_batch=(batch_idx // batch_size) + 1,
                    total_batches=total_batches
                )
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                monitor.add_error(str(e))
                continue
            
            # Checkpoint
            if samples_processed % checkpoint_interval == 0 and samples_processed > 0:
                checkpoint_path = f"models/checkpoints/retgen_checkpoint_{samples_processed}.pkl"
                try:
                    os.makedirs("models/checkpoints", exist_ok=True)
                    model.save(checkpoint_path)
                    monitor.add_checkpoint(checkpoint_path, samples_processed, total_patterns)
                    logger.info(f"Checkpoint saved at {samples_processed} samples")
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")
                    monitor.add_error(f"Checkpoint error: {e}")
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        monitor.set_status("completed")
        logger.info(f"Training completed. Processed {samples_processed} samples, extracted {total_patterns} patterns")
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        monitor.add_error(f"Training error: {e}")
        monitor.set_status("error")
        raise
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RETGEN model")
    parser.add_argument('--max-samples', type=int, default=1000000, help='Maximum training samples')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--checkpoint-interval', type=int, default=50000, help='Checkpoint interval')
    parser.add_argument('--embedding-dim', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--retrieval-k', type=int, default=100, help='Number of patterns to retrieve')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'embedding_dim': args.embedding_dim,
        'resolutions': [1, 2, 3, 5, 8],
        'retrieval_k': args.retrieval_k,
        'min_pattern_frequency': 2,
        'use_gpu': torch.cuda.is_available(),
        'temperature': 1.0
    }
    
    # Create directories
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("training_status", exist_ok=True)
    
    # Initialize monitor
    monitor = TrainingMonitor()
    
    # Handle interrupts gracefully
    def signal_handler(sig, frame):
        logger.info("Interrupt received, stopping training...")
        monitor.set_status("interrupted")
        monitor.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start monitoring
        monitor.start()
        monitor.set_status("loading_data")
        
        # Load data
        texts = load_training_data(args.max_samples)
        
        if not texts:
            logger.error("No training data loaded")
            monitor.set_status("error")
            monitor.add_error("No training data available")
            return
        
        # Train model
        model = train_retgen(
            texts=texts,
            config=config,
            monitor=monitor,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval
        )
        
        # Save final model
        monitor.set_status("saving_model")
        final_path = f"models/retgen_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model.save(final_path)
        monitor.add_log(f"Final model saved to {final_path}")
        
        # Test generation
        monitor.set_status("testing")
        test_prompt = "The future of artificial intelligence"
        generated = model.generate(test_prompt, max_length=50)
        logger.info(f"Test generation:\nPrompt: {test_prompt}\nGenerated: {generated}")
        monitor.add_log(f"Test generation successful: {generated[:100]}...")
        
        monitor.set_status("completed")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        monitor.set_status("error")
        monitor.add_error(str(e))
        raise
    finally:
        # Stop monitor
        monitor.stop()
        logger.info("Training script completed")


if __name__ == "__main__":
    main()