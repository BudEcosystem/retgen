#!/usr/bin/env python3
"""Memory-Optimized RETGEN Training with Checkpointing and GPU->RAM->Disk Offloading."""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pickle
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import psutil
import GPUtil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_memory_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class CheckpointManager:
    """Manages checkpoints and incremental saves."""
    
    checkpoint_dir: str = "models/checkpoints"
    max_checkpoints: int = 5
    
    def __post_init__(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoints = deque(maxlen=self.max_checkpoints)
    
    def save_checkpoint(self, model, iteration: int, patterns: int):
        """Save a checkpoint with automatic cleanup of old checkpoints."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_iter{iteration}_patterns{patterns}.pkl"
        )
        
        try:
            # Save checkpoint
            model.save(checkpoint_path)
            
            # Track checkpoint
            self.checkpoints.append(checkpoint_path)
            
            # Clean up old checkpoints if exceeded max
            if len(self.checkpoints) == self.max_checkpoints:
                oldest = self.checkpoints[0]
                if os.path.exists(oldest):
                    os.remove(oldest)
                    os.remove(oldest.replace('.pkl', '_index.faiss'))
                    logger.info(f"Removed old checkpoint: {oldest}")
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def get_latest_checkpoint(self):
        """Get the most recent checkpoint."""
        checkpoints = list(Path(self.checkpoint_dir).glob("checkpoint_*.pkl"))
        if checkpoints:
            return str(max(checkpoints, key=lambda x: x.stat().st_mtime))
        return None


class MemoryOptimizedVectorDB:
    """Vector database with GPU->RAM->Disk offloading for memory management."""
    
    def __init__(self, dimension: int = 384, max_ram_gb: float = 8.0, 
                 index_dir: str = "models/index_shards"):
        self.dimension = dimension
        self.max_ram_gb = max_ram_gb
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Main index (kept small in RAM)
        self.index = faiss.IndexFlatL2(dimension)
        
        # Shard management
        self.current_shard = 0
        self.shard_indices = []  # List of saved shard files
        self.patterns_per_shard = []
        self.continuations_per_shard = []
        
        # Current working buffers
        self.patterns = []
        self.continuations = []
        self.embeddings_buffer = []
        
        # Memory thresholds
        self.buffer_size = 10000
        self.max_index_size = 500000  # Max patterns in RAM index
        self.current_index_size = 0
        
        logger.info(f"Initialized MemoryOptimizedVectorDB with {max_ram_gb}GB RAM limit")
    
    def add_batch(self, patterns: List[str], continuations: List[str], embeddings: np.ndarray):
        """Add batch with automatic offloading when memory threshold reached."""
        self.patterns.extend(patterns)
        self.continuations.extend(continuations)
        self.embeddings_buffer.append(embeddings)
        
        # Check if we need to flush to index
        buffer_size = sum(e.shape[0] for e in self.embeddings_buffer)
        if buffer_size >= self.buffer_size:
            self.flush_buffer()
        
        # Check if we need to offload to disk
        if self.current_index_size >= self.max_index_size:
            self.offload_to_disk()
    
    def flush_buffer(self):
        """Flush embedding buffer to main index."""
        if self.embeddings_buffer:
            all_embeddings = np.vstack(self.embeddings_buffer).astype(np.float32)
            self.index.add(all_embeddings)
            self.current_index_size += all_embeddings.shape[0]
            self.embeddings_buffer = []
            logger.info(f"Flushed {all_embeddings.shape[0]} embeddings. Index size: {self.current_index_size}")
    
    def offload_to_disk(self):
        """Offload current index to disk and create new index."""
        if self.index.ntotal == 0:
            return
        
        logger.info(f"Offloading shard {self.current_shard} to disk...")
        
        # Save current index shard
        shard_path = self.index_dir / f"shard_{self.current_shard}.faiss"
        faiss.write_index(self.index, str(shard_path))
        
        # Save patterns and continuations for this shard
        meta_path = self.index_dir / f"shard_{self.current_shard}_meta.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'patterns': self.patterns,
                'continuations': self.continuations
            }, f)
        
        # Track shard
        self.shard_indices.append(str(shard_path))
        self.patterns_per_shard.append(self.patterns.copy())
        self.continuations_per_shard.append(self.continuations.copy())
        
        # Reset for new shard
        self.index = faiss.IndexFlatL2(self.dimension)
        self.patterns = []
        self.continuations = []
        self.current_index_size = 0
        self.current_shard += 1
        
        # Force garbage collection
        gc.collect()
        
        logger.info(f"Shard {self.current_shard-1} saved. Starting new shard.")
    
    def search_all_shards(self, query_embeddings: np.ndarray, k: int = 100):
        """Search across all shards (RAM + disk)."""
        all_distances = []
        all_indices = []
        offset = 0
        
        # Search current RAM index
        if self.index.ntotal > 0:
            D, I = self.index.search(query_embeddings.astype(np.float32), min(k, self.index.ntotal))
            all_distances.append(D)
            all_indices.append(I + offset)
            offset += self.index.ntotal
        
        # Search disk shards if needed
        for shard_path in self.shard_indices[-2:]:  # Only search last 2 shards for speed
            shard_index = faiss.read_index(shard_path)
            if shard_index.ntotal > 0:
                D, I = shard_index.search(query_embeddings.astype(np.float32), min(k, shard_index.ntotal))
                all_distances.append(D)
                all_indices.append(I + offset)
                offset += shard_index.ntotal
        
        if all_distances:
            # Combine results and get top-k
            distances = np.hstack(all_distances)
            indices = np.hstack(all_indices)
            
            # Get top-k across all results
            top_k_idx = np.argsort(distances[0])[:k]
            return distances[0][top_k_idx], indices[0][top_k_idx]
        
        return None, None
    
    def get_total_patterns(self):
        """Get total number of patterns across all shards."""
        total = self.current_index_size
        for shard_path in self.shard_indices:
            # Quick metadata check instead of loading full index
            meta_path = shard_path.replace('.faiss', '_meta.pkl')
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    total += len(meta['patterns'])
        return total


class GPUOptimizedPatternExtractor:
    """GPU-optimized pattern extraction with memory management."""
    
    def __init__(self, resolutions=[1, 2, 3, 5, 8], device='cuda'):
        self.resolutions = resolutions
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def extract_patterns_batch(self, texts: List[str], max_length: int = 512) -> Tuple[List[str], List[str]]:
        """Extract patterns from a batch of texts."""
        all_patterns = []
        all_continuations = []
        
        # Tokenize all texts at once
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False
        )
        
        for tokens in encodings['input_ids']:
            if len(tokens) < 2:
                continue
                
            for resolution in self.resolutions:
                for i in range(len(tokens) - resolution):
                    pattern_tokens = tokens[i:i+resolution]
                    continuation_token = tokens[i+resolution] if i+resolution < len(tokens) else self.tokenizer.pad_token_id
                    
                    pattern_text = self.tokenizer.decode(pattern_tokens, skip_special_tokens=True)
                    continuation_text = self.tokenizer.decode([continuation_token], skip_special_tokens=True)
                    
                    all_patterns.append(pattern_text)
                    all_continuations.append(continuation_text)
        
        return all_patterns, all_continuations


class GPUDataset(Dataset):
    """PyTorch Dataset for GPU training."""
    
    def __init__(self, texts: List[str]):
        self.texts = texts
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]


class MemoryOptimizedRETGEN:
    """Memory-Optimized RETGEN with checkpointing and offloading."""
    
    def __init__(self, config: Dict, checkpoint_manager: CheckpointManager):
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing Memory-Optimized RETGEN on {self.device}")
        
        # Pattern extractor
        self.pattern_extractor = GPUOptimizedPatternExtractor(
            resolutions=config.get('resolutions', [1, 2, 3, 5, 8]),
            device=self.device
        )
        
        # Sentence encoder
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        
        # Enable half precision for faster encoding
        if self.device.type == 'cuda':
            self.encoder = self.encoder.half()
        
        # Memory-optimized vector database
        self.vector_db = MemoryOptimizedVectorDB(
            dimension=384,
            max_ram_gb=config.get('max_ram_gb', 8.0),
            index_dir=config.get('index_dir', 'models/index_shards')
        )
        
        # Statistics
        self.total_patterns = 0
        self.samples_processed = 0
        self.last_checkpoint_patterns = 0
        
    def process_batch_gpu(self, texts: List[str], batch_size: int = 256):
        """Process a batch with GPU acceleration and memory management."""
        
        # Extract patterns
        patterns, continuations = self.pattern_extractor.extract_patterns_batch(texts)
        
        if not patterns:
            return 0
        
        # Filter by frequency
        unique_patterns = {}
        for p, c in zip(patterns, continuations):
            if p not in unique_patterns:
                unique_patterns[p] = []
            unique_patterns[p].append(c)
        
        min_freq = self.config.get('min_pattern_frequency', 2)
        filtered_patterns = []
        filtered_continuations = []
        
        for pattern, conts in unique_patterns.items():
            if len(conts) >= min_freq:
                filtered_patterns.append(pattern)
                filtered_continuations.append(max(set(conts), key=conts.count))
        
        if not filtered_patterns:
            return 0
        
        # Batch encode on GPU
        chunk_size = batch_size
        all_embeddings = []
        
        for i in range(0, len(filtered_patterns), chunk_size):
            chunk = filtered_patterns[i:i+chunk_size]
            
            with torch.no_grad():
                embeddings = self.encoder.encode(
                    chunk,
                    batch_size=min(len(chunk), batch_size),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                all_embeddings.append(embeddings)
        
        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
            
            # Add to vector database (with automatic offloading)
            self.vector_db.add_batch(
                filtered_patterns,
                filtered_continuations,
                final_embeddings
            )
            
            self.total_patterns += len(filtered_patterns)
            
        return len(filtered_patterns)
    
    def train_with_checkpoints(self, texts: List[str], batch_size: int = 256, 
                              num_workers: int = 4, checkpoint_interval: int = 50000):
        """Train with automatic checkpointing and memory management."""
        
        logger.info(f"Starting training on {len(texts)} texts")
        logger.info(f"Batch size: {batch_size}, Workers: {num_workers}")
        logger.info(f"Checkpoint interval: {checkpoint_interval} samples")
        
        # Create DataLoader
        dataset = GPUDataset(texts)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        # Training loop
        total_patterns = 0
        
        with tqdm(total=len(texts), desc="Training") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                # Process batch
                patterns_added = self.process_batch_gpu(batch, batch_size=min(256, len(batch)))
                total_patterns += patterns_added
                self.samples_processed += len(batch)
                
                # Update progress
                pbar.update(len(batch))
                pbar.set_postfix({
                    'patterns': total_patterns,
                    'shards': self.vector_db.current_shard,
                    'gpu_mem': f"{torch.cuda.memory_allocated()/1024**3:.2f}GB",
                    'ram': f"{psutil.virtual_memory().percent:.1f}%"
                })
                
                # Checkpoint if needed
                if (self.samples_processed % checkpoint_interval == 0 and 
                    self.samples_processed > 0):
                    
                    # Flush buffers before checkpoint
                    self.vector_db.flush_buffer()
                    
                    # Save checkpoint
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        self, 
                        self.samples_processed,
                        self.total_patterns
                    )
                    
                    if checkpoint_path:
                        logger.info(f"Checkpoint saved at {self.samples_processed} samples, {self.total_patterns} patterns")
                    
                    # Clear GPU cache
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Memory management
                if pbar.n % 10000 == 0:
                    torch.cuda.empty_cache()
                    
                    # Check system memory
                    mem_percent = psutil.virtual_memory().percent
                    if mem_percent > 85:
                        logger.warning(f"High RAM usage: {mem_percent}%. Forcing disk offload...")
                        self.vector_db.offload_to_disk()
                        gc.collect()
        
        # Final flush
        self.vector_db.flush_buffer()
        
        logger.info(f"Training complete. Total patterns: {self.total_patterns}")
        logger.info(f"Total shards created: {self.vector_db.current_shard + 1}")
        
        return total_patterns
    
    def save(self, path: str):
        """Save model with all shards."""
        save_data = {
            'config': self.config,
            'total_patterns': self.total_patterns,
            'samples_processed': self.samples_processed,
            'current_shard': self.vector_db.current_shard,
            'shard_indices': self.vector_db.shard_indices
        }
        
        # Save current index
        if self.vector_db.index.ntotal > 0:
            faiss.write_index(self.vector_db.index, path.replace('.pkl', '_current_index.faiss'))
        
        # Save metadata
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from checkpoint."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.config = save_data['config']
        self.total_patterns = save_data['total_patterns']
        self.samples_processed = save_data.get('samples_processed', 0)
        self.vector_db.current_shard = save_data.get('current_shard', 0)
        self.vector_db.shard_indices = save_data.get('shard_indices', [])
        
        # Load current index if exists
        index_path = path.replace('.pkl', '_current_index.faiss')
        if os.path.exists(index_path):
            self.vector_db.index = faiss.read_index(index_path)
            self.vector_db.current_index_size = self.vector_db.index.ntotal
        
        logger.info(f"Model loaded from {path}")
        logger.info(f"Resumed at {self.samples_processed} samples, {self.total_patterns} patterns")


def monitor_system():
    """Monitor system resources."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    logger.info(f"System: CPU {cpu_percent:.1f}% | RAM {memory.percent:.1f}% ({memory.used/1024**3:.1f}/{memory.total/1024**3:.1f}GB)")
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            logger.info(f"GPU: {gpu.name} | Util: {gpu.load*100:.1f}% | Mem: {gpu_mem:.2f}/{gpu_total:.2f}GB | Temp: {gpu.temperature}°C")


def main():
    """Main training function with checkpointing and memory management."""
    parser = argparse.ArgumentParser(description="Memory-Optimized RETGEN Training")
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for GPU processing')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--max-samples', type=int, default=2000000, help='Maximum training samples')
    parser.add_argument('--checkpoint-interval', type=int, default=25000, help='Checkpoint interval (samples)')
    parser.add_argument('--max-ram-gb', type=float, default=8.0, help='Maximum RAM for index (GB)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'resolutions': [1, 2, 3, 5, 8],
        'min_pattern_frequency': 2,
        'embedding_dim': 384,
        'max_ram_gb': args.max_ram_gb,
        'index_dir': 'models/index_shards'
    }
    
    # GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
        
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info("GPU optimizations enabled")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir="models/checkpoints",
        max_checkpoints=5
    )
    
    # Initialize or load model
    model = MemoryOptimizedRETGEN(config, checkpoint_manager)
    
    # Resume from checkpoint if specified
    start_idx = 0
    if args.resume:
        if os.path.exists(args.resume):
            model.load(args.resume)
            start_idx = model.samples_processed
            logger.info(f"Resumed from checkpoint: {args.resume}")
        else:
            logger.warning(f"Checkpoint not found: {args.resume}")
    elif checkpoint_manager.get_latest_checkpoint():
        latest = checkpoint_manager.get_latest_checkpoint()
        logger.info(f"Found checkpoint: {latest}")
        response = input("Resume from latest checkpoint? (y/n): ")
        if response.lower() == 'y':
            model.load(latest)
            start_idx = model.samples_processed
    
    # Load training data
    logger.info("Loading training data...")
    with open("training_data.json", 'r') as f:
        all_texts = json.load(f)
    
    # Resume from where we left off
    texts = all_texts[start_idx:min(start_idx + args.max_samples, len(all_texts))]
    logger.info(f"Training on {len(texts)} samples (starting from sample {start_idx})")
    
    # Monitor initial state
    monitor_system()
    
    # Train with checkpoints
    start_time = time.time()
    
    try:
        total_patterns = model.train_with_checkpoints(
            texts,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            checkpoint_interval=args.checkpoint_interval
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed/60:.2f} minutes")
        logger.info(f"Processing speed: {len(texts)/elapsed:.1f} samples/second")
        logger.info(f"Total patterns extracted: {total_patterns}")
        
        # Monitor final state
        monitor_system()
        
        # Save final model
        final_path = f"models/retgen_memory_optimized_final.pkl"
        model.save(final_path)
        logger.info(f"Final model saved to {final_path}")
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        
        # Save emergency checkpoint
        emergency_path = f"models/emergency_checkpoint_{model.samples_processed}.pkl"
        try:
            model.save(emergency_path)
            logger.info(f"Emergency checkpoint saved: {emergency_path}")
        except:
            logger.error("Failed to save emergency checkpoint")
        
        raise
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()