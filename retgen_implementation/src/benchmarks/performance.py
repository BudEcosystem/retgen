"""Performance benchmarking for RETGEN."""

import time
import psutil
import gc
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.retgen import RETGEN
from training.dataset_loader import DatasetLoader


logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Performance benchmark suite for RETGEN."""
    
    def __init__(self, model: RETGEN):
        """Initialize performance benchmark.
        
        Args:
            model: RETGEN model to benchmark
        """
        self.model = model
        self.results = {}
    
    def benchmark_training_speed(
        self,
        dataset_sizes: List[int] = [100, 500, 1000, 2000],
        num_runs: int = 1
    ) -> Dict[str, Any]:
        """Benchmark training speed across different dataset sizes.
        
        Args:
            dataset_sizes: List of dataset sizes to test
            num_runs: Number of runs per size for averaging
            
        Returns:
            Dictionary with training speed results
        """
        logger.info(f"Benchmarking training speed on sizes: {dataset_sizes}")
        
        results = {
            'dataset_sizes': dataset_sizes,
            'training_times': [],
            'patterns_per_second': [],
            'memory_usage': []
        }
        
        for size in dataset_sizes:
            logger.info(f"Training on {size} documents...")
            
            size_times = []
            size_patterns_per_sec = []
            size_memory = []
            
            for run in range(num_runs):
                # Create fresh dataset
                docs = DatasetLoader.create_sample_dataset(size)
                train_docs, val_docs, _ = DatasetLoader.split_dataset(docs)
                
                # Create fresh model
                model = RETGEN(self.model.config)
                
                # Monitor memory before training
                gc.collect()
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # Time training
                start_time = time.time()
                metrics = model.train(train_docs, val_docs)
                training_time = time.time() - start_time
                
                # Monitor memory after training
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                # Calculate patterns processed per second
                total_patterns = metrics.index_size[-1] if metrics.index_size else 0
                patterns_per_sec = total_patterns / training_time if training_time > 0 else 0
                
                size_times.append(training_time)
                size_patterns_per_sec.append(patterns_per_sec)
                size_memory.append(memory_used)
                
                # Cleanup
                del model
                gc.collect()
            
            # Average across runs
            results['training_times'].append(np.mean(size_times))
            results['patterns_per_second'].append(np.mean(size_patterns_per_sec))
            results['memory_usage'].append(np.mean(size_memory))
        
        self.results['training_speed'] = results
        return results
    
    def benchmark_inference_speed(
        self,
        prompts: List[str],
        generation_lengths: List[int] = [10, 25, 50, 100],
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """Benchmark inference speed for different generation lengths.
        
        Args:
            prompts: List of test prompts
            generation_lengths: List of maximum generation lengths to test
            num_runs: Number of runs per configuration
            
        Returns:
            Dictionary with inference speed results
        """
        logger.info(f"Benchmarking inference speed on {len(prompts)} prompts")
        
        if not self.model.is_trained:
            logger.warning("Model not trained, skipping inference benchmark")
            return {}
        
        results = {
            'generation_lengths': generation_lengths,
            'avg_times': [],
            'tokens_per_second': [],
            'memory_usage': []
        }
        
        for max_length in generation_lengths:
            logger.info(f"Testing generation length: {max_length}")
            
            length_times = []
            length_memory = []
            
            for run in range(num_runs):
                # Monitor memory
                gc.collect()
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # Time generation
                start_time = time.time()
                generated_texts = self.model.generate_batch(
                    prompts,
                    max_length=max_length
                )
                generation_time = time.time() - start_time
                
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                length_times.append(generation_time)
                length_memory.append(memory_used)
            
            # Calculate metrics
            avg_time = np.mean(length_times)
            total_tokens = len(prompts) * max_length
            tokens_per_sec = total_tokens / avg_time if avg_time > 0 else 0
            avg_memory = np.mean(length_memory)
            
            results['avg_times'].append(avg_time)
            results['tokens_per_second'].append(tokens_per_sec)
            results['memory_usage'].append(avg_memory)
        
        self.results['inference_speed'] = results
        return results
    
    def benchmark_scaling(
        self,
        pattern_counts: List[int] = [1000, 5000, 10000, 50000],
        query_batch_sizes: List[int] = [1, 10, 50, 100]
    ) -> Dict[str, Any]:
        """Benchmark scaling with database size and query batch size.
        
        Args:
            pattern_counts: List of pattern counts to test
            query_batch_sizes: List of query batch sizes to test
            
        Returns:
            Dictionary with scaling results
        """
        logger.info("Benchmarking scaling behavior")
        
        results = {
            'pattern_counts': pattern_counts,
            'query_batch_sizes': query_batch_sizes,
            'search_times': {},
            'memory_usage': {}
        }
        
        # Test different database sizes
        for pattern_count in pattern_counts:
            logger.info(f"Testing with {pattern_count} patterns")
            
            # Create dataset with appropriate size to get target pattern count
            docs_needed = max(10, pattern_count // 50)  # Estimate
            docs = DatasetLoader.create_sample_dataset(docs_needed)
            
            # Train model
            model = RETGEN(self.model.config)
            model.train(docs)
            
            # Test different query batch sizes
            results['search_times'][pattern_count] = []
            results['memory_usage'][pattern_count] = []
            
            for batch_size in query_batch_sizes:
                test_prompts = ["Test prompt for scaling"] * batch_size
                
                # Time search operations
                search_times = []
                for _ in range(10):  # Multiple runs
                    start_time = time.time()
                    _ = model.generate_batch(test_prompts, max_length=10)
                    search_time = time.time() - start_time
                    search_times.append(search_time)
                
                avg_search_time = np.mean(search_times)
                model_memory = model.get_size_mb()
                
                results['search_times'][pattern_count].append(avg_search_time)
                results['memory_usage'][pattern_count].append(model_memory)
            
            # Cleanup
            del model
            gc.collect()
        
        self.results['scaling'] = results
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark detailed memory usage breakdown.
        
        Returns:
            Dictionary with memory usage breakdown
        """
        logger.info("Benchmarking memory usage")
        
        if not self.model.is_trained:
            logger.warning("Model not trained, creating sample model")
            docs = DatasetLoader.create_sample_dataset(100)
            self.model.train(docs)
        
        results = {}
        
        # Overall model size
        results['model_size_mb'] = self.model.get_size_mb()
        
        # Database statistics
        db_stats = self.model.trainer.database.get_stats()
        results['pattern_count'] = db_stats['pattern_count']
        results['embedding_dim'] = db_stats['embedding_dim']
        
        # Estimate component sizes
        pattern_count = db_stats['pattern_count']
        embedding_dim = db_stats['embedding_dim']
        
        # Embeddings size
        embeddings_size_mb = (pattern_count * embedding_dim * 4) / (1024 * 1024)  # float32
        results['embeddings_size_mb'] = embeddings_size_mb
        
        # Metadata size (estimated)
        metadata_size_mb = (pattern_count * 100) / (1024 * 1024)  # ~100 bytes per pattern
        results['metadata_size_mb'] = metadata_size_mb
        
        # Index overhead
        index_overhead_mb = embeddings_size_mb * 0.2  # Estimated 20% overhead
        results['index_overhead_mb'] = index_overhead_mb
        
        # Cache size
        cache_stats = self.model.trainer.embedder.get_cache_stats()
        results['cache_entries'] = cache_stats['size']
        results['cache_hit_rate'] = cache_stats['hit_rate']
        
        self.results['memory_usage'] = results
        return results
    
    def run_full_benchmark(
        self,
        save_results: bool = True,
        results_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run complete performance benchmark suite.
        
        Args:
            save_results: Whether to save results to file
            results_path: Path to save results (optional)
            
        Returns:
            Dictionary with all benchmark results
        """
        logger.info("Running full performance benchmark suite")
        
        # Training speed benchmark
        try:
            self.benchmark_training_speed()
            logger.info("✓ Training speed benchmark completed")
        except Exception as e:
            logger.error(f"Training speed benchmark failed: {e}")
        
        # Create test prompts
        test_prompts = [
            "The future of artificial intelligence",
            "Natural language processing",
            "Machine learning algorithms",
            "Deep learning models",
            "Text generation systems"
        ]
        
        # Inference speed benchmark
        try:
            if self.model.is_trained:
                self.benchmark_inference_speed(test_prompts)
                logger.info("✓ Inference speed benchmark completed")
        except Exception as e:
            logger.error(f"Inference speed benchmark failed: {e}")
        
        # Scaling benchmark
        try:
            self.benchmark_scaling()
            logger.info("✓ Scaling benchmark completed")
        except Exception as e:
            logger.error(f"Scaling benchmark failed: {e}")
        
        # Memory usage benchmark
        try:
            self.benchmark_memory_usage()
            logger.info("✓ Memory usage benchmark completed")
        except Exception as e:
            logger.error(f"Memory usage benchmark failed: {e}")
        
        # Save results
        if save_results:
            if results_path is None:
                results_path = Path("benchmark_results_performance.json")
            
            import json
            with open(results_path, 'w') as f:
                # Convert numpy types for JSON serialization
                json_results = self._convert_for_json(self.results)
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Saved performance benchmark results to {results_path}")
        
        return self.results
    
    def _convert_for_json(self, obj):
        """Convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj