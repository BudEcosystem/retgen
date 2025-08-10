#!/usr/bin/env python3
"""GPU-Optimized RETGEN Training with Maximum Parallelization."""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
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
        logging.FileHandler('training_gpu_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GPUOptimizedPatternExtractor:
    """GPU-optimized pattern extraction using parallel processing."""
    
    def __init__(self, resolutions=[1, 2, 3, 5, 8], device='cuda'):
        self.resolutions = resolutions
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def extract_patterns_batch(self, texts: List[str], max_length: int = 512) -> Tuple[List[str], List[str]]:
        """Extract patterns from a batch of texts in parallel."""
        all_patterns = []
        all_continuations = []
        
        # Tokenize all texts at once (batch tokenization is faster)
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False
        )
        
        # Process each text in parallel
        for tokens in encodings['input_ids']:
            if len(tokens) < 2:
                continue
                
            # Extract patterns at all resolutions
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
    
    def __init__(self, texts: List[str], batch_size: int = 256):
        self.texts = texts
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]


class GPUOptimizedVectorDB:
    """GPU-optimized vector database using FAISS GPU."""
    
    def __init__(self, dimension: int = 384, use_gpu: bool = True):
        self.dimension = dimension
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Create index - using CPU FAISS but keeping GPU for embeddings
        # Note: faiss-gpu not available, using CPU index with GPU embeddings
        self.index = faiss.IndexFlatL2(dimension)
        logger.info("Created CPU FAISS index (GPU used for embeddings)")
        
        self.patterns = []
        self.continuations = []
        self.embeddings_buffer = []
        self.buffer_size = 10000  # Buffer before adding to index
        
    def add_batch(self, patterns: List[str], continuations: List[str], embeddings: np.ndarray):
        """Add a batch of patterns to the database."""
        self.patterns.extend(patterns)
        self.continuations.extend(continuations)
        self.embeddings_buffer.append(embeddings)
        
        # Add to index when buffer is full
        if sum(e.shape[0] for e in self.embeddings_buffer) >= self.buffer_size:
            self.flush_buffer()
    
    def flush_buffer(self):
        """Flush embedding buffer to index."""
        if self.embeddings_buffer:
            all_embeddings = np.vstack(self.embeddings_buffer).astype(np.float32)
            self.index.add(all_embeddings)
            self.embeddings_buffer = []
            logger.info(f"Added {all_embeddings.shape[0]} embeddings to index. Total: {self.index.ntotal}")
    
    def search_batch(self, query_embeddings: np.ndarray, k: int = 100):
        """Batch search for nearest neighbors."""
        if self.index.ntotal == 0:
            return None, None
        
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embeddings.astype(np.float32), k)
        return distances, indices


class GPUOptimizedRETGEN:
    """GPU-Optimized RETGEN with maximum parallelization."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        logger.info(f"Initializing GPU-Optimized RETGEN on {self.device}")
        
        # Pattern extractor with parallel processing
        self.pattern_extractor = GPUOptimizedPatternExtractor(
            resolutions=config.get('resolutions', [1, 2, 3, 5, 8]),
            device=self.device
        )
        
        # Sentence encoder on GPU
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()  # Set to eval mode for faster inference
        
        # Enable half precision for faster encoding
        if self.device.type == 'cuda':
            self.encoder = self.encoder.half()
        
        # Vector database
        self.vector_db = GPUOptimizedVectorDB(
            dimension=384,  # all-MiniLM-L6-v2 dimension
            use_gpu=self.device.type == 'cuda'
        )
        
        # Pattern statistics
        self.pattern_stats = {}
        self.total_patterns = 0
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
    def process_batch_gpu(self, texts: List[str], batch_size: int = 256):
        """Process a batch of texts with full GPU acceleration."""
        
        # Extract patterns in parallel
        patterns, continuations = self.pattern_extractor.extract_patterns_batch(texts)
        
        if not patterns:
            return 0
        
        # Filter by frequency (do this before embedding to save computation)
        unique_patterns = {}
        for p, c in zip(patterns, continuations):
            if p not in unique_patterns:
                unique_patterns[p] = []
            unique_patterns[p].append(c)
        
        # Filter patterns by minimum frequency
        min_freq = self.config.get('min_pattern_frequency', 2)
        filtered_patterns = []
        filtered_continuations = []
        
        for pattern, conts in unique_patterns.items():
            if len(conts) >= min_freq:
                # Take the most common continuation
                filtered_patterns.append(pattern)
                filtered_continuations.append(max(set(conts), key=conts.count))
        
        if not filtered_patterns:
            return 0
        
        # Batch encode all patterns on GPU
        # Process in chunks to avoid OOM
        chunk_size = batch_size
        all_embeddings = []
        
        for i in range(0, len(filtered_patterns), chunk_size):
            chunk = filtered_patterns[i:i+chunk_size]
            
            # Encode on GPU with no_grad for faster inference
            with torch.no_grad():
                embeddings = self.encoder.encode(
                    chunk,
                    batch_size=min(len(chunk), batch_size),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
            
            # Add to vector database
            self.vector_db.add_batch(
                filtered_patterns,
                filtered_continuations,
                final_embeddings
            )
            
            self.total_patterns += len(filtered_patterns)
            
        return len(filtered_patterns)
    
    def train_parallel(self, texts: List[str], batch_size: int = 256, num_workers: int = 4):
        """Train with parallel data loading and GPU processing."""
        
        logger.info(f"Starting parallel training on {len(texts)} texts")
        logger.info(f"Batch size: {batch_size}, Workers: {num_workers}")
        
        # Create DataLoader for efficient batching
        dataset = GPUDataset(texts, batch_size)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,  # Pin memory for faster GPU transfer
            prefetch_factor=2,  # Prefetch batches
            persistent_workers=True  # Keep workers alive
        )
        
        # Training loop with progress bar
        total_patterns = 0
        
        with tqdm(total=len(texts), desc="Training") as pbar:
            for batch in dataloader:
                # Process batch on GPU
                patterns_added = self.process_batch_gpu(batch, batch_size=min(256, len(batch)))
                total_patterns += patterns_added
                
                # Update progress
                pbar.update(len(batch))
                pbar.set_postfix({
                    'patterns': total_patterns,
                    'gpu_mem': f"{torch.cuda.memory_allocated()/1024**3:.2f}GB"
                })
                
                # Clear GPU cache periodically
                if pbar.n % 10000 == 0:
                    torch.cuda.empty_cache()
        
        # Flush remaining embeddings
        self.vector_db.flush_buffer()
        
        logger.info(f"Training complete. Total patterns: {self.total_patterns}")
        
        return total_patterns
    
    def save(self, path: str):
        """Save model to disk."""
        save_data = {
            'config': self.config,
            'total_patterns': self.total_patterns,
            'pattern_stats': self.pattern_stats
        }
        
        # Save FAISS index (already on CPU)
        faiss.write_index(self.vector_db.index, path.replace('.pkl', '_index.faiss'))
        
        # Save metadata
        with open(path, 'wb') as f:
            import pickle
            pickle.dump(save_data, f)
        
        logger.info(f"Model saved to {path}")


def monitor_gpu():
    """Monitor GPU utilization."""
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            logger.info(f"GPU: {gpu.name} | Util: {gpu.load*100:.1f}% | Mem: {gpu_mem:.2f}/{gpu_total:.2f}GB | Temp: {gpu.temperature}°C")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="GPU-Optimized RETGEN Training")
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for GPU processing')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--max-samples', type=int, default=1000000, help='Maximum training samples')
    parser.add_argument('--checkpoint-interval', type=int, default=50000, help='Checkpoint interval')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'resolutions': [1, 2, 3, 5, 8],
        'min_pattern_frequency': 2,
        'embedding_dim': 384
    }
    
    # GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info("GPU optimizations enabled")
    
    # Load training data
    logger.info("Loading training data...")
    with open("training_data.json", 'r') as f:
        texts = json.load(f)[:args.max_samples]
    logger.info(f"Loaded {len(texts)} samples")
    
    # Initialize model
    model = GPUOptimizedRETGEN(config)
    
    # Monitor initial GPU state
    monitor_gpu()
    
    # Train with maximum GPU utilization
    start_time = time.time()
    
    try:
        total_patterns = model.train_parallel(
            texts,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed/60:.2f} minutes")
        logger.info(f"Processing speed: {len(texts)/elapsed:.1f} samples/second")
        logger.info(f"Total patterns extracted: {total_patterns}")
        
        # Monitor final GPU state
        monitor_gpu()
        
        # Save model
        model.save("models/retgen_gpu_optimized.pkl")
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    finally:
        # Cleanup
        if hasattr(model, 'thread_pool'):
            model.thread_pool.shutdown()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()