#!/usr/bin/env python3
"""Complete RETGEN system demonstration with proper imports."""

import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set environment variable
os.environ['PYTHONPATH'] = str(src_path)

import logging
import time
import json
import numpy as np
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RETGENSystem:
    """Complete RETGEN system implementation."""
    
    def __init__(self, config=None):
        """Initialize RETGEN system."""
        # Import configuration
        from core.config import RETGENConfig
        
        self.config = config or RETGENConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
            min_pattern_frequency=1,
            retrieval_k=20,
            max_generation_length=100,
            device="cpu",  # Force CPU
            use_gpu=False,
            index_type="Flat",  # Use simple flat index for demo
            use_positional_encoding=False  # Simplify for demo
        )
        
        # Import core components
        from data.pattern_extraction import PatternExtractor
        from embeddings.context_embeddings import RETGENEmbedder
        from indexing.vector_database import VectorDatabase
        from training.dataset_loader import DatasetLoader
        from inference.generator import RETGENGenerator
        
        # Initialize components
        self.pattern_extractor = PatternExtractor(self.config)
        self.embedder = RETGENEmbedder(self.config)
        self.database = VectorDatabase(self.config)
        self.dataset_loader = DatasetLoader
        self.generator = None
        
        self.is_trained = False
        self.training_metrics = {
            'training_time': 0,
            'pattern_count': 0,
            'model_size_mb': 0
        }
    
    def train(self, train_docs: List[str], val_docs: List[str] = None):
        """Train the RETGEN model."""
        logger.info(f"Training RETGEN with {len(train_docs)} documents")
        start_time = time.time()
        
        # Step 1: Extract patterns
        logger.info("Extracting patterns...")
        patterns = self.pattern_extractor.extract_from_corpus(train_docs, show_progress=True)
        
        # Step 2: Compute embeddings
        logger.info("Computing embeddings...")
        embeddings = self.embedder.embed_patterns(
            patterns,
            batch_size=self.config.batch_size,
            show_progress=True
        )
        
        # Step 3: Build database
        logger.info("Building vector database...")
        self.database.add_patterns(patterns, embeddings)
        
        # Step 4: Initialize generator
        from inference.generator import RETGENGenerator
        self.generator = RETGENGenerator(self.embedder, self.database, self.config)
        
        # Update metrics
        self.training_metrics['training_time'] = time.time() - start_time
        self.training_metrics['pattern_count'] = len(patterns)
        self.training_metrics['model_size_mb'] = self._estimate_model_size()
        
        self.is_trained = True
        
        logger.info(f"Training completed in {self.training_metrics['training_time']:.2f}s")
        logger.info(f"Extracted {self.training_metrics['pattern_count']:,} patterns")
        
        return self.training_metrics
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generation")
        
        return self.generator.generate(prompt, **kwargs)
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in MB."""
        stats = self.database.get_stats()
        pattern_count = stats['pattern_count']
        embedding_dim = stats['embedding_dim']
        
        # Embeddings + metadata + index overhead
        embeddings_size_mb = (pattern_count * embedding_dim * 4) / (1024 * 1024)
        metadata_size_mb = (pattern_count * 100) / (1024 * 1024)
        index_overhead_mb = embeddings_size_mb * 0.2
        
        return embeddings_size_mb + metadata_size_mb + index_overhead_mb
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save(path / "config.json")
        
        # Save database
        self.database.save(path / "database")
        
        # Save metrics
        with open(path / "metrics.json", 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        """Load model from disk."""
        from core.config import RETGENConfig
        
        path = Path(path)
        
        # Load config
        config = RETGENConfig.load(path / "config.json")
        
        # Create system
        system = cls(config)
        
        # Load database
        system.database.load(path / "database")
        
        # Load metrics
        with open(path / "metrics.json", 'r') as f:
            system.training_metrics = json.load(f)
        
        # Initialize generator
        from inference.generator import RETGENGenerator
        system.generator = RETGENGenerator(system.embedder, system.database, system.config)
        
        system.is_trained = True
        
        return system


def demonstrate_retgen():
    """Demonstrate RETGEN functionality."""
    print("=" * 80)
    print("RETGEN Complete System Demonstration")
    print("=" * 80)
    
    # Create RETGEN system
    print("\n1. Creating RETGEN system...")
    retgen = RETGENSystem()
    print("✓ System initialized")
    
    # Create dataset
    print("\n2. Creating sample dataset...")
    docs = retgen.dataset_loader.create_sample_dataset(50)
    train_docs, val_docs, test_docs = retgen.dataset_loader.split_dataset(docs)
    print(f"✓ Created {len(train_docs)} training documents")
    
    # Train model
    print("\n3. Training RETGEN model...")
    metrics = retgen.train(train_docs, val_docs)
    print(f"✓ Training completed:")
    print(f"  - Time: {metrics['training_time']:.2f}s")
    print(f"  - Patterns: {metrics['pattern_count']:,}")
    print(f"  - Size: {metrics['model_size_mb']:.1f} MB")
    
    # Test generation
    print("\n4. Testing text generation...")
    test_prompts = [
        "The future of artificial intelligence",
        "Natural language processing is",
        "Machine learning algorithms can",
        "Deep learning models have",
        "Text generation requires"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        generated = retgen.generate(prompt, max_length=50, temperature=1.0)
        print(f"\n{i}. '{prompt}'")
        print(f"   → '{generated}'")
    
    # Test save/load
    print("\n5. Testing model persistence...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        
        # Save
        retgen.save(save_path)
        print("✓ Model saved")
        
        # Load
        loaded_retgen = RETGENSystem.load(save_path)
        print("✓ Model loaded")
        
        # Test loaded model
        test_gen = loaded_retgen.generate("Test prompt", max_length=20)
        print(f"✓ Loaded model generates: '{test_gen}'")
    
    print("\n" + "=" * 80)
    print("🎉 RETGEN system demonstration completed successfully!")
    
    return retgen


def run_benchmarks(retgen: RETGENSystem):
    """Run performance benchmarks."""
    print("\n" + "=" * 80)
    print("Running RETGEN Benchmarks")
    print("=" * 80)
    
    from benchmarks.performance import PerformanceBenchmark
    from benchmarks.accuracy import AccuracyBenchmark
    
    # Performance benchmark
    print("\n1. Performance Analysis:")
    perf_bench = PerformanceBenchmark(retgen)
    
    # Memory usage
    memory_results = perf_bench.benchmark_memory_usage()
    print(f"  - Model size: {memory_results['model_size_mb']:.1f} MB")
    print(f"  - Pattern count: {memory_results['pattern_count']:,}")
    print(f"  - Cache hit rate: {memory_results['cache_hit_rate']:.2%}")
    
    # Inference speed
    test_prompts = ["Test prompt"] * 5
    speed_results = perf_bench.benchmark_inference_speed(
        test_prompts,
        generation_lengths=[10, 25, 50],
        num_runs=3
    )
    
    print("\n2. Inference Speed:")
    for i, length in enumerate(speed_results['generation_lengths']):
        tokens_per_sec = speed_results['tokens_per_second'][i]
        print(f"  - {length} tokens: {tokens_per_sec:.1f} tokens/sec")
    
    # Accuracy benchmark
    print("\n3. Generation Quality:")
    acc_bench = AccuracyBenchmark(retgen)
    
    test_patterns = ["natural language", "machine learning", "deep learning"]
    retrieval_results = acc_bench.benchmark_retrieval_quality(test_patterns, k_values=[1, 5, 10])
    
    print("  Retrieval Quality (k, similarity, coverage):")
    for i, k in enumerate(retrieval_results['k_values']):
        sim = retrieval_results['avg_similarities'][i]
        cov = retrieval_results['coverage_rates'][i]
        print(f"  - k={k}: {sim:.3f}, {cov:.3f}")


if __name__ == "__main__":
    try:
        # Run demonstration
        retgen = demonstrate_retgen()
        
        # Run benchmarks
        run_benchmarks(retgen)
        
        print("\n✅ All systems operational!")
        print("\nNext: Run scripts/train_retgen.py for full training")
        print("      Run scripts/benchmark_retgen.py for comprehensive benchmarks")
        
    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback
        traceback.print_exc()