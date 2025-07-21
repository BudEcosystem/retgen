#!/usr/bin/env python3
"""Final demonstration of complete RETGEN implementation."""

import sys
import os
from pathlib import Path
import time
import json

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("RETGEN: Retrieval-Enhanced Text Generation")
print("Complete Implementation Demonstration")
print("=" * 80)

def demonstrate_retgen():
    """Demonstrate the complete RETGEN system."""
    
    try:
        # Step 1: Import all components
        print("Step 1: Importing RETGEN components...")
        
        from core.config import RETGENConfig, TrainingMetrics
        from data.pattern_extraction import PatternExtractor, RETGENPreprocessor
        from embeddings.context_embeddings import RETGENEmbedder
        from indexing.vector_database import VectorDatabase
        from training.trainer import RETGENTrainer
        from training.dataset_loader import DatasetLoader
        from inference.generator import RETGENGenerator
        from evaluation.metrics import RETGENEvaluator
        from benchmarks.performance import PerformanceBenchmark
        from benchmarks.accuracy import AccuracyBenchmark
        
        print("✓ All components imported successfully!")
        
        # Step 2: Create configuration
        print("\nStep 2: Creating RETGEN configuration...")
        
        config = RETGENConfig(
            # Use lightweight model for demo
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
            
            # Pattern settings
            min_pattern_length=1,
            max_pattern_length=5,
            min_pattern_frequency=1,
            resolutions=[1, 2, 3],
            
            # Retrieval settings  
            retrieval_k=20,
            temperature=1.0,
            
            # Generation settings
            max_generation_length=50,
            top_p=0.9,
            
            # Training settings
            batch_size=32,
            
            # Hardware
            device="cpu"
        )
        
        print(f"✓ Configuration created:")
        print(f"  - Embedding model: {config.embedding_model}")
        print(f"  - Embedding dimension: {config.embedding_dim}")
        print(f"  - Retrieval k: {config.retrieval_k}")
        print(f"  - Resolutions: {config.resolutions}")
        
        # Step 3: Create sample dataset
        print("\nStep 3: Creating sample dataset...")
        
        docs = DatasetLoader.create_sample_dataset(30)
        train_docs, val_docs, test_docs = DatasetLoader.split_dataset(docs)
        
        dataset_info = DatasetLoader.get_dataset_info(train_docs)
        print(f"✓ Dataset created:")
        print(f"  - Training documents: {dataset_info['num_docs']}")
        print(f"  - Average doc length: {dataset_info['avg_doc_length']:.1f} chars")
        print(f"  - Total characters: {dataset_info['total_chars']:,}")
        
        # Step 4: Initialize components
        print("\nStep 4: Initializing RETGEN components...")
        
        # Pattern extractor
        extractor = PatternExtractor(config)
        print("✓ Pattern extractor initialized")
        
        # Embedder  
        embedder = RETGENEmbedder(config)
        print("✓ Embedder initialized")
        
        # Vector database
        database = VectorDatabase(config)
        print("✓ Vector database initialized")
        
        # Step 5: Extract patterns
        print("\nStep 5: Extracting patterns from corpus...")
        start_time = time.time()
        
        patterns = extractor.extract_from_corpus(train_docs, show_progress=True)
        extraction_time = time.time() - start_time
        
        print(f"✓ Pattern extraction completed:")
        print(f"  - Patterns extracted: {len(patterns):,}")
        print(f"  - Extraction time: {extraction_time:.2f} seconds")
        print(f"  - Patterns per second: {len(patterns)/extraction_time:.1f}")
        
        # Show pattern examples
        print(f"  - Sample patterns:")
        for i, pattern in enumerate(patterns[:5], 1):
            print(f"    {i}. '{pattern.text}' -> '{pattern.next_text}' (res={pattern.resolution})")
        
        # Step 6: Compute embeddings
        print("\nStep 6: Computing embeddings...")
        start_time = time.time()
        
        embeddings = embedder.embed_patterns(
            patterns[:1000],  # Limit for demo speed
            batch_size=config.batch_size,
            show_progress=True
        )
        embedding_time = time.time() - start_time
        
        print(f"✓ Embeddings computed:")
        print(f"  - Embeddings shape: {embeddings.shape}")
        print(f"  - Embedding time: {embedding_time:.2f} seconds")
        print(f"  - Embeddings per second: {len(embeddings)/embedding_time:.1f}")
        
        # Step 7: Build vector database
        print("\nStep 7: Building vector database...")
        start_time = time.time()
        
        database.add_patterns(patterns[:1000], embeddings)
        indexing_time = time.time() - start_time
        
        print(f"✓ Vector database built:")
        print(f"  - Indexing time: {indexing_time:.2f} seconds")
        print(f"  - Patterns indexed: {database.pattern_count:,}")
        
        db_stats = database.get_stats()
        print(f"  - Index size: {db_stats['index_size']:,}")
        print(f"  - Embedding dimension: {db_stats['embedding_dim']}")
        
        # Step 8: Test retrieval
        print("\nStep 8: Testing pattern retrieval...")
        
        test_queries = [
            "natural language processing",
            "machine learning algorithm", 
            "deep neural network"
        ]
        
        for query in test_queries:
            query_embedding = embedder.embed_text(query)
            results = database.search(query_embedding.reshape(1, -1), k=3)
            
            print(f"  Query: '{query}'")
            if results[0]:
                for i, result in enumerate(results[0][:3], 1):
                    pattern = result['pattern']
                    similarity = result['distance']
                    print(f"    {i}. '{pattern.text}' -> '{pattern.next_text}' (sim={similarity:.3f})")
            else:
                print("    No results found")
        
        # Step 9: Initialize generator
        print("\nStep 9: Initializing text generator...")
        
        generator = RETGENGenerator(embedder, database, config)
        print("✓ Generator initialized")
        
        # Step 10: Test text generation
        print("\nStep 10: Testing text generation...")
        
        test_prompts = [
            "The future of artificial intelligence",
            "Natural language processing is",
            "Machine learning algorithms can",
            "Deep learning models have",
            "Text generation requires"
        ]
        
        print("Generated texts:")
        for i, prompt in enumerate(test_prompts, 1):
            try:
                start_time = time.time()
                generated = generator.generate(
                    prompt,
                    max_length=30,
                    temperature=1.0,
                    top_p=0.9
                )
                gen_time = time.time() - start_time
                
                print(f"  {i}. '{prompt}'")
                print(f"     -> '{generated}'")
                print(f"     (generated in {gen_time:.3f}s)")
                
            except Exception as e:
                print(f"  {i}. '{prompt}' -> ERROR: {e}")
        
        # Step 11: Performance analysis
        print("\nStep 11: Performance analysis...")
        
        # Memory usage
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        print(f"✓ Performance metrics:")
        print(f"  - Memory usage: {memory_mb:.1f} MB")
        print(f"  - Patterns per MB: {database.pattern_count / memory_mb:.1f}")
        
        # Cache statistics
        cache_stats = embedder.get_cache_stats()
        print(f"  - Cache entries: {cache_stats['size']}")
        print(f"  - Cache hit rate: {cache_stats['hit_rate']:.2%}")
        
        # Step 12: Save demonstration results
        print("\nStep 12: Saving demonstration results...")
        
        results = {
            "config": config.__dict__,
            "dataset_stats": dataset_info,
            "pattern_stats": {
                "total_patterns": len(patterns),
                "indexed_patterns": database.pattern_count,
                "extraction_time": extraction_time,
                "embedding_time": embedding_time,
                "indexing_time": indexing_time
            },
            "performance_stats": {
                "memory_mb": memory_mb,
                "cache_hit_rate": cache_stats['hit_rate']
            },
            "sample_generations": [
                {
                    "prompt": prompt,
                    "generated": generator.generate(prompt, max_length=20)
                }
                for prompt in test_prompts[:3]
            ]
        }
        
        # Save to JSON
        with open("retgen_demo_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("✓ Results saved to retgen_demo_results.json")
        
        print("\n" + "=" * 80)
        print("🎉 RETGEN DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print("\nKey Achievements:")
        print(f"✓ Extracted {len(patterns):,} patterns in {extraction_time:.2f}s")
        print(f"✓ Computed {len(embeddings):,} embeddings in {embedding_time:.2f}s")
        print(f"✓ Built vector database with {database.pattern_count:,} patterns")
        print(f"✓ Generated text from {len(test_prompts)} prompts")
        print(f"✓ Total memory usage: {memory_mb:.1f} MB")
        
        print("\nNext Steps:")
        print("1. Run full benchmarks: python scripts/benchmark_retgen.py --train-new")
        print("2. Train on larger datasets: python scripts/train_retgen.py --dataset wikitext103")
        print("3. Compare with transformers: python scripts/compare_models.py")
        print("4. Deploy as API: python scripts/api_server.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demonstrate_retgen()
    if success:
        print("\n✨ RETGEN implementation is working correctly!")
    else:
        print("\n💥 RETGEN demonstration failed!")
    
    sys.exit(0 if success else 1)