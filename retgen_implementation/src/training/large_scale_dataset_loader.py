"""
Large-scale dataset loader for C4 and Wikipedia datasets.
Optimized for training RetGen with 1M+ samples.
"""

import os
import json
import random
from typing import List, Dict, Iterator, Optional, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Will import after dependencies are installed
try:
    from datasets import load_dataset, Dataset
    import torch
    from torch.utils.data import DataLoader, IterableDataset
except ImportError:
    print("Dependencies not yet installed. Will be available after installation.")

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for large-scale dataset loading."""
    dataset_name: str  # 'c4', 'wikipedia', 'both'
    max_samples: int = 1_000_000
    max_length: int = 512
    min_length: int = 10
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    streaming: bool = True  # Use streaming for large datasets
    cache_dir: str = "./data/cache"
    c4_config: str = "en"  # C4 configuration
    wiki_config: str = "20220301.en"  # Wikipedia configuration
    shuffle_buffer_size: int = 10_000
    preprocessing_num_workers: int = 4


class LargeScaleDatasetLoader:
    """Loader for large-scale datasets optimized for GPU training."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        random.seed(config.seed)
        
        # Create cache directory
        os.makedirs(config.cache_dir, exist_ok=True)
        
    def load_c4(self, streaming: bool = True) -> Dataset:
        """Load C4 dataset with streaming support."""
        logger.info(f"Loading C4 dataset (streaming={streaming})")
        
        # Use allenai/c4 dataset
        dataset = load_dataset(
            "allenai/c4",
            self.config.c4_config,
            streaming=streaming,
            cache_dir=self.config.cache_dir,
            split="train"
        )
        
        if streaming:
            # Shuffle with buffer for streaming
            dataset = dataset.shuffle(
                seed=self.config.seed,
                buffer_size=self.config.shuffle_buffer_size
            )
        
        return dataset
    
    def load_wikipedia(self, streaming: bool = True) -> Dataset:
        """Load Wikipedia dataset with streaming support."""
        logger.info(f"Loading Wikipedia dataset (streaming={streaming})")
        
        dataset = load_dataset(
            "wikipedia",
            self.config.wiki_config,
            streaming=streaming,
            cache_dir=self.config.cache_dir,
            split="train"
        )
        
        if streaming:
            dataset = dataset.shuffle(
                seed=self.config.seed,
                buffer_size=self.config.shuffle_buffer_size
            )
        
        return dataset
    
    def preprocess_text(self, example: Dict) -> Dict:
        """Preprocess text for RetGen training."""
        # Get text field (different for C4 and Wikipedia)
        if 'text' in example:
            text = example['text']
        elif 'content' in example:
            text = example['content']
        else:
            text = str(example.get('title', '')) + ' ' + str(example.get('text', ''))
        
        # Clean and filter
        text = text.strip()
        
        # Skip if too short or too long
        if len(text.split()) < self.config.min_length:
            return None
        
        # Truncate if too long
        words = text.split()
        if len(words) > self.config.max_length:
            text = ' '.join(words[:self.config.max_length])
        
        return {'text': text, 'length': len(text.split())}
    
    def load_combined_dataset(self) -> Iterator[str]:
        """Load and combine C4 and Wikipedia datasets."""
        logger.info("Loading combined C4 and Wikipedia datasets")
        
        datasets = []
        
        if self.config.dataset_name in ['c4', 'both']:
            c4_dataset = self.load_c4(streaming=self.config.streaming)
            datasets.append(('c4', c4_dataset))
        
        if self.config.dataset_name in ['wikipedia', 'both']:
            wiki_dataset = self.load_wikipedia(streaming=self.config.streaming)
            datasets.append(('wikipedia', wiki_dataset))
        
        # Process samples
        sample_count = 0
        
        # Round-robin between datasets
        iterators = [(name, iter(dataset)) for name, dataset in datasets]
        
        with tqdm(total=self.config.max_samples, desc="Loading samples") as pbar:
            while sample_count < self.config.max_samples:
                for dataset_name, iterator in iterators:
                    try:
                        # Get next sample
                        sample = next(iterator)
                        
                        # Preprocess
                        processed = self.preprocess_text(sample)
                        
                        if processed is not None:
                            yield processed['text']
                            sample_count += 1
                            pbar.update(1)
                            
                            if sample_count >= self.config.max_samples:
                                break
                    
                    except StopIteration:
                        # Dataset exhausted
                        logger.info(f"Dataset {dataset_name} exhausted")
                        continue
                
                # Check if all datasets are exhausted
                if all(isinstance(it, StopIteration) for _, it in iterators):
                    logger.warning("All datasets exhausted before reaching target samples")
                    break
        
        logger.info(f"Loaded {sample_count} samples total")
    
    def create_batched_iterator(self) -> Iterator[List[str]]:
        """Create batched iterator for efficient processing."""
        batch = []
        
        for text in self.load_combined_dataset():
            batch.append(text)
            
            if len(batch) >= self.config.batch_size:
                yield batch
                batch = []
        
        # Yield remaining samples
        if batch:
            yield batch
    
    def parallel_preprocess(self, texts: List[str], num_workers: int = None) -> List[str]:
        """Parallel preprocessing of text batches."""
        if num_workers is None:
            num_workers = self.config.preprocessing_num_workers
        
        with mp.Pool(num_workers) as pool:
            # Simple cleaning function for parallel processing
            clean_func = lambda x: x.strip().replace('\n', ' ').replace('\t', ' ')
            processed = pool.map(clean_func, texts)
        
        return processed


class RetGenDataset(IterableDataset):
    """PyTorch IterableDataset for RetGen training."""
    
    def __init__(self, loader: LargeScaleDatasetLoader):
        self.loader = loader
    
    def __iter__(self):
        """Iterate over dataset samples."""
        for text in self.loader.load_combined_dataset():
            yield text


def create_dataloader(config: DatasetConfig) -> DataLoader:
    """Create PyTorch DataLoader for GPU training."""
    loader = LargeScaleDatasetLoader(config)
    dataset = RetGenDataset(loader)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,  # For GPU training
        prefetch_factor=2  # Prefetch batches
    )
    
    return dataloader


def download_and_prepare_datasets(config: DatasetConfig):
    """Download and prepare datasets for offline use."""
    loader = LargeScaleDatasetLoader(config)
    
    logger.info("Downloading and preparing datasets...")
    
    # Save samples to disk for faster loading
    output_file = os.path.join(config.cache_dir, f"{config.dataset_name}_prepared.jsonl")
    
    with open(output_file, 'w') as f:
        for i, text in enumerate(loader.load_combined_dataset()):
            f.write(json.dumps({'id': i, 'text': text}) + '\n')
    
    logger.info(f"Saved prepared dataset to {output_file}")
    return output_file


# Statistics collection for dataset analysis
class DatasetStatistics:
    """Collect statistics about the dataset."""
    
    def __init__(self):
        self.total_samples = 0
        self.total_tokens = 0
        self.length_distribution = []
        self.source_distribution = {'c4': 0, 'wikipedia': 0}
    
    def update(self, text: str, source: str):
        """Update statistics with new sample."""
        self.total_samples += 1
        tokens = len(text.split())
        self.total_tokens += tokens
        self.length_distribution.append(tokens)
        self.source_distribution[source] += 1
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        import numpy as np
        
        if not self.length_distribution:
            return {}
        
        return {
            'total_samples': self.total_samples,
            'total_tokens': self.total_tokens,
            'avg_length': np.mean(self.length_distribution),
            'median_length': np.median(self.length_distribution),
            'min_length': np.min(self.length_distribution),
            'max_length': np.max(self.length_distribution),
            'source_distribution': self.source_distribution
        }


if __name__ == "__main__":
    # Test configuration
    config = DatasetConfig(
        dataset_name='both',
        max_samples=1000,  # Small test
        batch_size=32,
        streaming=True
    )
    
    # Test loading
    loader = LargeScaleDatasetLoader(config)
    
    # Collect some samples
    samples = []
    for i, text in enumerate(loader.load_combined_dataset()):
        samples.append(text)
        if i >= 10:
            break
    
    print(f"Loaded {len(samples)} test samples")
    print(f"First sample preview: {samples[0][:200]}...")