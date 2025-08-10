"""
Large-scale GPU-accelerated training script for RetGen.
Trains on C4 and Wikipedia datasets with 1M+ samples.
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import gc

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Will import after dependencies are installed
try:
    import torch
    import numpy as np
    from tqdm import tqdm
    from torch.cuda.amp import autocast, GradScaler
    import psutil
    import GPUtil
except ImportError:
    print("Dependencies not yet installed. Will be available after installation.")

from src.core.config import RETGENConfig
from src.core.retgen import RETGEN
from src.training.large_scale_dataset_loader import (
    DatasetConfig, LargeScaleDatasetLoader, DatasetStatistics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_gpu.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor GPU usage during training."""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device = torch.cuda.current_device()
            self.gpu_name = torch.cuda.get_device_name(self.device)
            logger.info(f"Using GPU: {self.gpu_name}")
    
    def get_stats(self) -> Dict:
        """Get current GPU statistics."""
        if not self.gpu_available:
            return {}
        
        stats = {
            'gpu_memory_used': torch.cuda.memory_allocated() / 1024**3,  # GB
            'gpu_memory_cached': torch.cuda.memory_reserved() / 1024**3,  # GB
            'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
        }
        
        # Get GPU utilization
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            stats['gpu_utilization'] = gpu.load * 100
            stats['gpu_temperature'] = gpu.temperature
        
        return stats
    
    def log_stats(self):
        """Log current GPU statistics."""
        stats = self.get_stats()
        if stats:
            logger.info(f"GPU Memory: {stats['gpu_memory_used']:.2f}/{stats['gpu_memory_total']:.2f} GB "
                       f"(Cached: {stats['gpu_memory_cached']:.2f} GB)")
            if 'gpu_utilization' in stats:
                logger.info(f"GPU Utilization: {stats['gpu_utilization']:.1f}%, "
                           f"Temperature: {stats['gpu_temperature']}°C")


class LargeScaleTrainer:
    """Trainer for large-scale RetGen models with GPU optimization."""
    
    def __init__(self, config: RETGENConfig, dataset_config: DatasetConfig):
        self.config = config
        self.dataset_config = dataset_config
        
        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            # Enable GPU optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set GPU memory management
            torch.cuda.empty_cache()
            
            logger.info(f"GPU Optimizations enabled for {torch.cuda.get_device_name()}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        else:
            logger.warning("CUDA not available, training on CPU (will be slower)")
        
        # Initialize model
        self.model = RETGEN(config)
        
        # Setup monitoring
        self.gpu_monitor = GPUMonitor()
        
        # Training statistics
        self.stats = {
            'samples_processed': 0,
            'patterns_extracted': 0,
            'training_time': 0,
            'gpu_stats': [],
            'checkpoints': []
        }
    
    def optimize_gpu_memory(self):
        """Optimize GPU memory usage."""
        if self.device.type == 'cuda':
            # Clear cache
            torch.cuda.empty_cache()
            
            # Run garbage collection
            gc.collect()
            
            # Log memory stats
            self.gpu_monitor.log_stats()
    
    def train_batch_gpu(self, texts: List[str]) -> Dict:
        """Train on a batch of texts with GPU acceleration."""
        batch_stats = {
            'batch_size': len(texts),
            'patterns_extracted': 0,
            'processing_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Move processing to GPU where possible
            if hasattr(self.model, 'embedder') and hasattr(self.model.embedder, 'model'):
                # Ensure embedding model is on GPU
                if hasattr(self.model.embedder.model, 'to'):
                    self.model.embedder.model = self.model.embedder.model.to(self.device)
            
            # Process texts in batch
            for text in texts:
                patterns = self.model.pattern_extractor.extract_patterns(text)
                batch_stats['patterns_extracted'] += len(patterns)
                
                # Add patterns to model
                for pattern in patterns:
                    self.model.add_pattern(pattern)
            
            # Build/update index with GPU acceleration if using FAISS-GPU
            if self.model.vector_db and hasattr(self.model.vector_db, 'use_gpu'):
                self.model.vector_db.use_gpu = True
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise
        
        batch_stats['processing_time'] = time.time() - start_time
        
        return batch_stats
    
    def train(self, checkpoint_interval: int = 10000):
        """Train RetGen on large-scale dataset."""
        logger.info("=" * 80)
        logger.info("Starting Large-Scale RetGen Training")
        logger.info(f"Dataset: {self.dataset_config.dataset_name}")
        logger.info(f"Target samples: {self.dataset_config.max_samples:,}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 80)
        
        # Initialize dataset loader
        loader = LargeScaleDatasetLoader(self.dataset_config)
        dataset_stats = DatasetStatistics()
        
        # Training loop
        start_time = time.time()
        last_checkpoint = 0
        
        try:
            # Process in batches
            for batch_idx, batch_texts in enumerate(tqdm(
                loader.create_batched_iterator(),
                desc="Training batches",
                total=self.dataset_config.max_samples // self.dataset_config.batch_size
            )):
                # Train on batch
                batch_stats = self.train_batch_gpu(batch_texts)
                
                # Update statistics
                self.stats['samples_processed'] += batch_stats['batch_size']
                self.stats['patterns_extracted'] += batch_stats['patterns_extracted']
                
                # Update dataset statistics
                for text in batch_texts:
                    source = 'c4' if batch_idx % 2 == 0 else 'wikipedia'  # Alternating assumption
                    dataset_stats.update(text, source)
                
                # Periodic monitoring and checkpointing
                if self.stats['samples_processed'] - last_checkpoint >= checkpoint_interval:
                    self.checkpoint(batch_idx)
                    last_checkpoint = self.stats['samples_processed']
                    
                    # Log progress
                    self.log_progress()
                    
                    # Optimize memory
                    self.optimize_gpu_memory()
                
                # Check if we've reached target
                if self.stats['samples_processed'] >= self.dataset_config.max_samples:
                    logger.info(f"Reached target of {self.dataset_config.max_samples:,} samples")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        finally:
            # Final statistics
            self.stats['training_time'] = time.time() - start_time
            
            # Save final model and stats
            self.save_final_model()
            self.save_training_stats(dataset_stats)
            
            # Log summary
            self.log_summary(dataset_stats)
    
    def checkpoint(self, batch_idx: int):
        """Save model checkpoint."""
        checkpoint_dir = Path("models/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"retgen_checkpoint_{self.stats['samples_processed']}.pkl"
        
        logger.info(f"Saving checkpoint at {self.stats['samples_processed']:,} samples...")
        
        # Save model
        self.model.save(str(checkpoint_path))
        
        # Record checkpoint
        self.stats['checkpoints'].append({
            'batch_idx': batch_idx,
            'samples': self.stats['samples_processed'],
            'patterns': self.stats['patterns_extracted'],
            'timestamp': datetime.now().isoformat(),
            'path': str(checkpoint_path)
        })
        
        # Also save training stats
        stats_path = checkpoint_dir / f"training_stats_{self.stats['samples_processed']}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def save_final_model(self):
        """Save final trained model."""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f"retgen_large_scale_{timestamp}.pkl"
        
        logger.info(f"Saving final model to {model_path}...")
        self.model.save(str(model_path))
        
        # Also save config
        config_path = model_dir / f"retgen_config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info("Model saved successfully")
    
    def save_training_stats(self, dataset_stats: DatasetStatistics):
        """Save training statistics."""
        stats_dir = Path("models/stats")
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_path = stats_dir / f"training_stats_{timestamp}.json"
        
        # Combine all statistics
        all_stats = {
            'training_stats': self.stats,
            'dataset_stats': dataset_stats.summary(),
            'config': {
                'model': self.config.to_dict(),
                'dataset': self.dataset_config.__dict__
            },
            'timestamp': timestamp
        }
        
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        logger.info(f"Training statistics saved to {stats_path}")
    
    def log_progress(self):
        """Log training progress."""
        elapsed = time.time() - (self.stats.get('start_time', time.time()))
        samples_per_sec = self.stats['samples_processed'] / max(elapsed, 1)
        
        logger.info("-" * 60)
        logger.info(f"Progress Update:")
        logger.info(f"  Samples processed: {self.stats['samples_processed']:,}")
        logger.info(f"  Patterns extracted: {self.stats['patterns_extracted']:,}")
        logger.info(f"  Processing rate: {samples_per_sec:.1f} samples/sec")
        logger.info(f"  Elapsed time: {elapsed/3600:.2f} hours")
        
        # GPU stats
        self.gpu_monitor.log_stats()
        
        # Memory stats
        process = psutil.Process()
        ram_usage = process.memory_info().rss / 1024**3  # GB
        logger.info(f"  RAM usage: {ram_usage:.2f} GB")
        logger.info("-" * 60)
    
    def log_summary(self, dataset_stats: DatasetStatistics):
        """Log training summary."""
        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        
        # Training stats
        logger.info("Training Statistics:")
        logger.info(f"  Total samples: {self.stats['samples_processed']:,}")
        logger.info(f"  Total patterns: {self.stats['patterns_extracted']:,}")
        logger.info(f"  Training time: {self.stats['training_time']/3600:.2f} hours")
        logger.info(f"  Average rate: {self.stats['samples_processed']/self.stats['training_time']:.1f} samples/sec")
        
        # Dataset stats
        ds_summary = dataset_stats.summary()
        if ds_summary:
            logger.info("\nDataset Statistics:")
            logger.info(f"  Total tokens: {ds_summary['total_tokens']:,}")
            logger.info(f"  Average length: {ds_summary['avg_length']:.1f} tokens")
            logger.info(f"  Source distribution: {ds_summary['source_distribution']}")
        
        # Model stats
        if hasattr(self.model, 'vector_db') and self.model.vector_db:
            logger.info("\nModel Statistics:")
            logger.info(f"  Index size: {self.model.vector_db.index_size:,} patterns")
            logger.info(f"  Embedding dimension: {self.config.embedding_dim}")
            logger.info(f"  Resolutions: {self.config.resolutions}")
        
        logger.info("=" * 80)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Large-scale RetGen training on GPU")
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='both',
                       choices=['c4', 'wikipedia', 'both'],
                       help='Dataset to train on')
    parser.add_argument('--max-samples', type=int, default=1_000_000,
                       help='Maximum number of samples to train on')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    
    # Model arguments
    parser.add_argument('--embedding-dim', type=int, default=768,
                       help='Embedding dimension')
    parser.add_argument('--resolutions', nargs='+', type=int, 
                       default=[1, 2, 3, 5, 8],
                       help='Pattern resolutions')
    parser.add_argument('--retrieval-k', type=int, default=100,
                       help='Number of patterns to retrieve')
    
    # Training arguments
    parser.add_argument('--checkpoint-interval', type=int, default=50000,
                       help='Checkpoint save interval')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # GPU arguments
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU device ID to use')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    
    args = parser.parse_args()
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    
    # Create configurations
    retgen_config = RETGENConfig(
        embedding_dim=args.embedding_dim,
        resolutions=args.resolutions,
        retrieval_k=args.retrieval_k,
        min_pattern_frequency=2,
        index_type='IVF4096,PQ64',  # Optimized for large-scale
        use_gpu=torch.cuda.is_available(),
        device=f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    )
    
    dataset_config = DatasetConfig(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        streaming=True,  # Always stream for large datasets
        cache_dir="./data/large_scale_cache"
    )
    
    # Log configurations
    logger.info("Configuration:")
    logger.info(f"  Model config: {retgen_config.to_dict()}")
    logger.info(f"  Dataset config: {dataset_config.__dict__}")
    
    # Create trainer
    trainer = LargeScaleTrainer(retgen_config, dataset_config)
    
    # Start training
    trainer.train(checkpoint_interval=args.checkpoint_interval)
    
    logger.info("Training script completed successfully!")


if __name__ == "__main__":
    main()