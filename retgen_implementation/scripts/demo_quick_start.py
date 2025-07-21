#!/usr/bin/env python3
"""Quick demonstration of RETGEN capabilities."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.retgen import RETGEN
from core.config import RETGENConfig
from training.dataset_loader import DatasetLoader
from benchmarks.performance import PerformanceBenchmark
from benchmarks.accuracy import AccuracyBenchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_usage():
    """Demonstrate basic RETGEN usage."""
    print("=" * 60)
    print("RETGEN Quick Start Demo")
    print("=" * 60)
    
    # Create configuration
    config = RETGENConfig(
        min_pattern_frequency=1,
        retrieval_k=20,
        max_generation_length=100,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Faster model
        embedding_dim=384
    )
    
    print("Configuration:")
    print(f"  - Min pattern frequency: {config.min_pattern_frequency}")
    print(f"  - Retrieval k: {config.retrieval_k}")
    print(f"  - Max generation length: {config.max_generation_length}")
    print(f"  - Embedding model: {config.embedding_model}")
    print()
    
    # Create and train model
    print("Creating RETGEN model...")
    model = RETGEN(config)
    
    # Create sample dataset
    print("Creating sample dataset...")
    train_docs = DatasetLoader.create_sample_dataset(num_docs=50)
    val_docs = DatasetLoader.create_sample_dataset(num_docs=10)
    
    dataset_info = DatasetLoader.get_dataset_info(train_docs)
    print(f"Dataset info:")
    print(f"  - Training documents: {dataset_info['num_docs']}")
    print(f"  - Average document length: {dataset_info['avg_doc_length']:.1f} chars")
    print()
    
    # Train model
    print("Training RETGEN model...")
    metrics = model.train(train_docs, val_docs)
    
    training_time = metrics.training_time[-1] if metrics.training_time else 0
    pattern_count = metrics.index_size[-1] if metrics.index_size else 0
    
    print(f"Training completed!")
    print(f"  - Training time: {training_time:.2f} seconds")
    print(f"  - Patterns extracted: {pattern_count:,}")
    print(f"  - Model size: {model.get_size_mb():.1f} MB")
    print()
    
    # Test generation
    print("Testing text generation...")
    test_prompts = [
        "The future of artificial intelligence",
        "Natural language processing is",
        "Machine learning algorithms can",
        "Deep learning models have",
        "Text generation requires"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        try:
            generated = model.generate(
                prompt,
                max_length=50,
                temperature=1.0,
                top_p=0.9
            )
            print(f"   Generated: {generated}")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    return model


def demo_performance_benchmark(model):
    """Demonstrate performance benchmarking."""
    print("RETGEN Performance Benchmark")
    print("=" * 60)
    
    # Create benchmark
    benchmark = PerformanceBenchmark(model)
    
    # Test prompts for speed benchmark
    test_prompts = [
        "The cat sat on",
        "Natural language processing",
        "Machine learning is",
        "Deep learning models",
        "Artificial intelligence"
    ]
    
    print("Running inference speed benchmark...")
    try:
        speed_results = benchmark.benchmark_inference_speed(
            prompts=test_prompts,
            generation_lengths=[10, 25, 50],
            num_runs=3
        )
        
        print("Inference Speed Results:")
        for i, length in enumerate(speed_results['generation_lengths']):
            time_taken = speed_results['avg_times'][i]
            tokens_per_sec = speed_results['tokens_per_second'][i]
            print(f"  - Max length {length}: {time_taken:.3f}s, {tokens_per_sec:.1f} tokens/sec")
        
    except Exception as e:
        print(f"Speed benchmark failed: {e}")
    
    print("\nRunning memory usage analysis...")
    try:
        memory_results = benchmark.benchmark_memory_usage()
        
        print("Memory Usage:")
        print(f"  - Total model size: {memory_results['model_size_mb']:.1f} MB")
        print(f"  - Embeddings: {memory_results['embeddings_size_mb']:.1f} MB")
        print(f"  - Metadata: {memory_results['metadata_size_mb']:.1f} MB")
        print(f"  - Cache entries: {memory_results['cache_entries']}")
        print(f"  - Cache hit rate: {memory_results['cache_hit_rate']:.2%}")
        
    except Exception as e:
        print(f"Memory analysis failed: {e}")
    
    print("\n" + "=" * 60)
    return benchmark


def demo_accuracy_benchmark(model):
    """Demonstrate accuracy benchmarking."""
    print("RETGEN Accuracy Benchmark")
    print("=" * 60)
    
    # Create benchmark
    benchmark = AccuracyBenchmark(model)
    
    # Test generation quality
    print("Testing generation quality...")
    prompt_sets = {
        'general': [
            "The future of",
            "Natural language",
            "Machine learning"
        ]
    }
    
    try:
        quality_results = benchmark.benchmark_generation_quality(prompt_sets)
        
        for set_name, results in quality_results.items():
            if 'metrics' in results:
                metrics = results['metrics']
                print(f"\n{set_name.title()} Prompts:")
                
                if 'retgen_distinct_1' in metrics:
                    print(f"  - Diversity (distinct-1): {metrics['retgen_distinct_1']:.3f}")
                if 'retgen_avg_length' in metrics:
                    print(f"  - Average length: {metrics['retgen_avg_length']:.1f} tokens")
                
                print("  - Sample outputs:")
                for i, output in enumerate(results.get('sample_outputs', [])[:3], 1):
                    print(f"    {i}. {output}")
    
    except Exception as e:
        print(f"Generation quality benchmark failed: {e}")
    
    # Test retrieval quality
    print("\nTesting retrieval quality...")
    test_patterns = [
        "natural language processing",
        "machine learning algorithm",
        "deep neural network"
    ]
    
    try:
        retrieval_results = benchmark.benchmark_retrieval_quality(
            test_patterns,
            k_values=[1, 5, 10]
        )
        
        print("Retrieval Quality:")
        for i, k in enumerate(retrieval_results['k_values']):
            similarity = retrieval_results['avg_similarities'][i]
            coverage = retrieval_results['coverage_rates'][i]
            print(f"  - k={k}: Avg similarity={similarity:.3f}, Coverage={coverage:.3f}")
    
    except Exception as e:
        print(f"Retrieval quality benchmark failed: {e}")
    
    print("\n" + "=" * 60)
    return benchmark


def demo_model_comparison():
    """Demonstrate comparing different RETGEN configurations."""
    print("RETGEN Model Comparison")
    print("=" * 60)
    
    # Create different configurations
    configs = {
        'small': RETGENConfig(
            retrieval_k=10,
            min_pattern_frequency=2,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384
        ),
        'medium': RETGENConfig(
            retrieval_k=25,
            min_pattern_frequency=1,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384
        )
    }
    
    # Train models
    models = {}
    train_docs = DatasetLoader.create_sample_dataset(30)
    
    for name, config in configs.items():
        print(f"\nTraining {name} model...")
        model = RETGEN(config)
        model.train(train_docs)
        models[name] = model
        
        info = model.get_model_info()
        print(f"  - Patterns: {info['pattern_count']:,}")
        print(f"  - Size: {model.get_size_mb():.1f} MB")
    
    # Compare generation
    print("\nGeneration Comparison:")
    test_prompt = "The future of artificial intelligence"
    
    for name, model in models.items():
        try:
            generated = model.generate(test_prompt, max_length=50)
            print(f"\n{name.title()} model:")
            print(f"  {generated}")
        except Exception as e:
            print(f"{name.title()} model failed: {e}")
    
    print("\n" + "=" * 60)


def main():
    """Run the complete demo."""
    print("RETGEN: Retrieval-Enhanced Text Generation")
    print("Comprehensive Demonstration")
    print("=" * 80)
    
    try:
        # Basic usage demo
        model = demo_basic_usage()
        
        # Performance benchmark
        demo_performance_benchmark(model)
        
        # Accuracy benchmark
        demo_accuracy_benchmark(model)
        
        # Model comparison
        demo_model_comparison()
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Try training on larger datasets")
        print("2. Experiment with different configurations")
        print("3. Compare with transformer baselines")
        print("4. Deploy as a REST API")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()