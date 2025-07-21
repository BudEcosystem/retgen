#!/usr/bin/env python3
"""Comprehensive benchmarking script for RETGEN models."""

import argparse
import sys
import logging
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.retgen import RETGEN
from core.config import RETGENConfig
from training.dataset_loader import DatasetLoader
from benchmarks.performance import PerformanceBenchmark
from benchmarks.accuracy import AccuracyBenchmark


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark RETGEN model")
    
    # Model arguments
    parser.add_argument("--model-path", type=str,
                        help="Path to trained RETGEN model (optional)")
    parser.add_argument("--train-new", action="store_true",
                        help="Train a new model for benchmarking")
    parser.add_argument("--num-docs", type=int, default=500,
                        help="Number of documents for training new model")
    
    # Benchmark arguments
    parser.add_argument("--benchmark-type", type=str, default="both",
                        choices=["performance", "accuracy", "both"],
                        help="Type of benchmark to run")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick benchmark (reduced test sizes)")
    
    # Performance benchmark arguments
    parser.add_argument("--perf-training-sizes", type=int, nargs="+",
                        default=[100, 500, 1000],
                        help="Dataset sizes for training speed benchmark")
    parser.add_argument("--perf-gen-lengths", type=int, nargs="+",
                        default=[10, 25, 50, 100],
                        help="Generation lengths for inference speed benchmark")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save visualization plots")
    
    # Other arguments
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    return parser.parse_args()


def load_or_train_model(args):
    """Load existing model or train new one."""
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        try:
            model = RETGEN.load(Path(args.model_path))
            print("✓ Model loaded successfully")
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            if not args.train_new:
                raise
    
    if args.train_new or not args.model_path:
        print(f"Training new model with {args.num_docs} documents...")
        
        # Create configuration optimized for benchmarking
        config = RETGENConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
            min_pattern_frequency=1,
            retrieval_k=30
        )
        
        # Train model
        model = RETGEN(config)
        docs = DatasetLoader.create_sample_dataset(args.num_docs)
        train_docs, val_docs, _ = DatasetLoader.split_dataset(docs)
        
        model.train(train_docs, val_docs)
        print("✓ Model trained successfully")
        
        return model
    
    raise ValueError("No model to benchmark")


def run_performance_benchmark(model, args):
    """Run performance benchmark."""
    print("\n" + "=" * 60)
    print("RUNNING PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    benchmark = PerformanceBenchmark(model)
    
    if args.quick:
        # Quick benchmark settings
        training_sizes = args.perf_training_sizes[:2]  # First 2 sizes only
        gen_lengths = args.perf_gen_lengths[:3]        # First 3 lengths only
        num_runs = 1
    else:
        # Full benchmark settings
        training_sizes = args.perf_training_sizes
        gen_lengths = args.perf_gen_lengths
        num_runs = 3
    
    results = {}
    
    # Training speed benchmark
    print("1. Training Speed Benchmark")
    print("-" * 30)
    try:
        training_results = benchmark.benchmark_training_speed(
            dataset_sizes=training_sizes,
            num_runs=num_runs
        )
        results['training_speed'] = training_results
        
        print("Training Speed Results:")
        for i, size in enumerate(training_results['dataset_sizes']):
            time_taken = training_results['training_times'][i]
            patterns_per_sec = training_results['patterns_per_second'][i]
            memory_mb = training_results['memory_usage'][i]
            
            print(f"  {size:4d} docs: {time_taken:6.2f}s, "
                  f"{patterns_per_sec:8.1f} patterns/sec, "
                  f"{memory_mb:6.1f} MB")
        
    except Exception as e:
        print(f"Training speed benchmark failed: {e}")
    
    # Inference speed benchmark
    print("\n2. Inference Speed Benchmark")
    print("-" * 30)
    
    test_prompts = [
        "The future of artificial intelligence",
        "Natural language processing is",
        "Machine learning algorithms can",
        "Deep learning models have",
        "Text generation requires understanding"
    ]
    
    try:
        inference_results = benchmark.benchmark_inference_speed(
            prompts=test_prompts,
            generation_lengths=gen_lengths,
            num_runs=num_runs
        )
        results['inference_speed'] = inference_results
        
        print("Inference Speed Results:")
        for i, length in enumerate(inference_results['generation_lengths']):
            time_taken = inference_results['avg_times'][i]
            tokens_per_sec = inference_results['tokens_per_second'][i]
            memory_mb = inference_results['memory_usage'][i]
            
            print(f"  {length:3d} tokens: {time_taken:6.3f}s, "
                  f"{tokens_per_sec:8.1f} tokens/sec, "
                  f"{memory_mb:6.1f} MB")
        
    except Exception as e:
        print(f"Inference speed benchmark failed: {e}")
    
    # Memory usage benchmark
    print("\n3. Memory Usage Analysis")
    print("-" * 30)
    try:
        memory_results = benchmark.benchmark_memory_usage()
        results['memory_usage'] = memory_results
        
        print("Memory Usage Breakdown:")
        print(f"  Total model size:     {memory_results['model_size_mb']:8.1f} MB")
        print(f"  Embeddings:          {memory_results['embeddings_size_mb']:8.1f} MB")
        print(f"  Metadata:            {memory_results['metadata_size_mb']:8.1f} MB")
        print(f"  Index overhead:      {memory_results['index_overhead_mb']:8.1f} MB")
        print(f"  Pattern count:       {memory_results['pattern_count']:8,}")
        print(f"  Embedding dimension: {memory_results['embedding_dim']:8,}")
        print(f"  Cache entries:       {memory_results['cache_entries']:8,}")
        print(f"  Cache hit rate:      {memory_results['cache_hit_rate']:8.2%}")
        
    except Exception as e:
        print(f"Memory usage benchmark failed: {e}")
    
    return results


def run_accuracy_benchmark(model, args):
    """Run accuracy benchmark."""
    print("\n" + "=" * 60)
    print("RUNNING ACCURACY BENCHMARK")
    print("=" * 60)
    
    benchmark = AccuracyBenchmark(model)
    results = {}
    
    # Test datasets
    if args.quick:
        test_size = 20
        prompt_size = 5
    else:
        test_size = 50
        prompt_size = 10
    
    test_datasets = {
        'sample': DatasetLoader.create_sample_dataset(test_size)
    }
    
    # Perplexity benchmark
    print("1. Perplexity Evaluation")
    print("-" * 30)
    try:
        ppl_results = benchmark.benchmark_perplexity(test_datasets)
        results['perplexity'] = ppl_results
        
        for dataset_name, dataset_results in ppl_results.items():
            if 'retgen_ppl' in dataset_results:
                ppl = dataset_results['retgen_ppl']
                print(f"  {dataset_name}: {ppl:.2f}")
            else:
                print(f"  {dataset_name}: ERROR")
        
    except Exception as e:
        print(f"Perplexity benchmark failed: {e}")
    
    # Generation quality benchmark
    print("\n2. Generation Quality Evaluation")
    print("-" * 30)
    
    prompt_sets = {
        'general': [
            "The future of artificial intelligence",
            "Natural language processing is",
            "Machine learning algorithms",
            "Deep learning models",
            "Text generation systems"
        ][:prompt_size],
        'technical': [
            "The algorithm works by",
            "The neural network architecture",
            "The optimization process",
            "The evaluation metrics",
            "The implementation details"
        ][:prompt_size]
    }
    
    try:
        quality_results = benchmark.benchmark_generation_quality(prompt_sets)
        results['generation_quality'] = quality_results
        
        for set_name, set_results in quality_results.items():
            print(f"\n  {set_name.title()} prompts:")
            if 'metrics' in set_results:
                metrics = set_results['metrics']
                
                if 'retgen_distinct_1' in metrics:
                    print(f"    Diversity (distinct-1): {metrics['retgen_distinct_1']:.3f}")
                if 'retgen_avg_length' in metrics:
                    print(f"    Average length:        {metrics['retgen_avg_length']:.1f} tokens")
                
                print("    Sample outputs:")
                for i, output in enumerate(set_results.get('sample_outputs', [])[:2], 1):
                    print(f"      {i}. {output}")
        
    except Exception as e:
        print(f"Generation quality benchmark failed: {e}")
    
    # Retrieval quality benchmark
    print("\n3. Retrieval Quality Evaluation")
    print("-" * 30)
    
    test_patterns = [
        "natural language processing",
        "machine learning algorithm", 
        "deep neural network",
        "artificial intelligence system",
        "text generation model"
    ][:prompt_size]
    
    try:
        retrieval_results = benchmark.benchmark_retrieval_quality(
            test_patterns,
            k_values=[1, 5, 10, 20]
        )
        results['retrieval_quality'] = retrieval_results
        
        print("Retrieval Quality Results:")
        print("    k   Avg Similarity   Coverage Rate")
        print("  " + "-" * 36)
        
        for i, k in enumerate(retrieval_results['k_values']):
            similarity = retrieval_results['avg_similarities'][i]
            coverage = retrieval_results['coverage_rates'][i]
            print(f"  {k:3d}      {similarity:.3f}         {coverage:.3f}")
        
    except Exception as e:
        print(f"Retrieval quality benchmark failed: {e}")
    
    return results


def save_results(results, args):
    """Save benchmark results to files."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    for benchmark_type, benchmark_results in results.items():
        output_file = output_dir / f"benchmark_{benchmark_type}.json"
        
        with open(output_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        print(f"✓ Saved {benchmark_type} results to {output_file}")
    
    # Save summary report
    summary_file = output_dir / "benchmark_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("RETGEN Benchmark Summary\n")
        f.write("=" * 50 + "\n\n")
        
        if 'performance' in results:
            f.write("PERFORMANCE RESULTS:\n")
            f.write("-" * 20 + "\n")
            
            perf = results['performance']
            
            if 'memory_usage' in perf:
                mem = perf['memory_usage']
                f.write(f"Model Size: {mem['model_size_mb']:.1f} MB\n")
                f.write(f"Patterns: {mem['pattern_count']:,}\n")
                f.write(f"Cache Hit Rate: {mem['cache_hit_rate']:.2%}\n")
            
            if 'inference_speed' in perf:
                inf = perf['inference_speed']
                if inf['tokens_per_second']:
                    avg_speed = sum(inf['tokens_per_second']) / len(inf['tokens_per_second'])
                    f.write(f"Avg Speed: {avg_speed:.1f} tokens/sec\n")
            
            f.write("\n")
        
        if 'accuracy' in results:
            f.write("ACCURACY RESULTS:\n")
            f.write("-" * 20 + "\n")
            
            acc = results['accuracy']
            
            if 'perplexity' in acc:
                for dataset, ppl_data in acc['perplexity'].items():
                    if 'retgen_ppl' in ppl_data:
                        f.write(f"Perplexity ({dataset}): {ppl_data['retgen_ppl']:.2f}\n")
            
            if 'retrieval_quality' in acc:
                ret = acc['retrieval_quality']
                if ret['avg_similarities']:
                    avg_sim = sum(ret['avg_similarities']) / len(ret['avg_similarities'])
                    f.write(f"Avg Retrieval Similarity: {avg_sim:.3f}\n")
    
    print(f"✓ Saved summary report to {summary_file}")


def main():
    """Main benchmarking function."""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("RETGEN Comprehensive Benchmark Suite")
    print("=" * 80)
    
    try:
        # Load or train model
        model = load_or_train_model(args)
        
        # Print model info
        model_info = model.get_model_info()
        print(f"\nModel Information:")
        print(f"Pattern count: {model_info.get('pattern_count', 'N/A'):,}")
        print(f"Model size: {model.get_size_mb():.1f} MB")
        print(f"Embedding dim: {model_info.get('embedding_dim', 'N/A')}")
        
        # Run benchmarks
        results = {}
        
        if args.benchmark_type in ['performance', 'both']:
            results['performance'] = run_performance_benchmark(model, args)
        
        if args.benchmark_type in ['accuracy', 'both']:
            results['accuracy'] = run_accuracy_benchmark(model, args)
        
        # Save results
        save_results(results, args)
        
        print(f"\n🎉 Benchmark completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()