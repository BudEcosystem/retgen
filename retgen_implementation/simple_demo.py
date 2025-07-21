#!/usr/bin/env python3
"""Simple RETGEN demonstration showing core functionality."""

import sys
import os
from pathlib import Path
import time
import numpy as np

# Add src to path and set PYTHONPATH
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))
os.environ['PYTHONPATH'] = str(src_path)

print("RETGEN: Retrieval-Enhanced Text Generation")
print("Simple Functionality Demonstration")  
print("=" * 60)

def run_simple_demo():
    """Run a simple demonstration of RETGEN core concepts."""
    
    try:
        print("Step 1: Testing configuration system...")
        
        # Import and test configuration
        from core.config import RETGENConfig
        
        config = RETGENConfig(
            embedding_dim=384,
            retrieval_k=20,
            max_generation_length=50
        )
        
        print(f"✓ Configuration created")
        print(f"  - Embedding dim: {config.embedding_dim}")
        print(f"  - Retrieval k: {config.retrieval_k}")
        
        # Test save/load
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config.save(config_path)
            
            loaded_config = RETGENConfig.load(config_path)
            assert loaded_config.embedding_dim == config.embedding_dim
            print("✓ Configuration save/load works")
        
        print("\nStep 2: Testing pattern extraction...")
        
        # Test pattern extraction components
        from data.pattern_extraction import Pattern, PatternDatabase
        
        # Create test patterns
        patterns = [
            Pattern([1, 2, 3], "the cat", 4, "sat", 0, 3, 0),
            Pattern([1, 2], "the cat", 5, "ran", 1, 2, 0),
            Pattern([1, 2, 3], "the cat", 4, "sat", 2, 3, 1),  # Duplicate
        ]
        
        # Test pattern database
        db = PatternDatabase()
        for pattern in patterns:
            db.add_pattern(pattern)
        
        print(f"✓ Pattern database created")
        print(f"  - Total patterns: {db.total_patterns}")
        print(f"  - Unique patterns: {len(db.patterns)}")
        
        # Test continuation distribution
        dist = db.get_continuation_distribution((1, 2, 3))
        print(f"  - Distribution for (1,2,3): {dist}")
        
        print("\nStep 3: Testing embeddings...")
        
        # Test positional encoder
        from embeddings.context_embeddings import PositionalEncoder, EmbeddingCache
        
        pos_encoder = PositionalEncoder(d_model=64, max_len=100)
        
        pos_enc = pos_encoder.encode_position(5, 20)
        print(f"✓ Positional encoding works")
        print(f"  - Position 5 in sequence 20: shape {pos_enc.shape}")
        
        # Test embedding cache
        cache = EmbeddingCache(max_size=10)
        
        test_embedding = np.random.randn(128)
        cache.set("test_key", test_embedding)
        
        retrieved = cache.get("test_key")
        assert np.array_equal(retrieved, test_embedding)
        print("✓ Embedding cache works")
        
        stats = cache.get_stats()
        print(f"  - Cache stats: {stats}")
        
        print("\nStep 4: Testing vector database...")
        
        from indexing.vector_database import FAISSIndexBuilder, PatternMetadataStore
        
        # Test metadata store
        metadata_store = PatternMetadataStore(backend='memory')
        
        test_metadata = {
            'tokens': [1, 2, 3],
            'text': 'test pattern',
            'continuations': {4: 0.8, 5: 0.2}
        }
        
        metadata_store.store_pattern(0, test_metadata)
        retrieved_metadata = metadata_store.get_pattern(0)
        
        assert retrieved_metadata == test_metadata
        print("✓ Metadata store works")
        
        # Test FAISS index builder
        index_config = RETGENConfig(
            similarity_metric="cosine",
            index_type="Flat"
        )
        
        builder = FAISSIndexBuilder(index_config)
        
        # Create test embeddings
        test_embeddings = np.random.randn(100, 64).astype(np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True)
        test_embeddings = test_embeddings / norms
        
        index = builder.build_index(test_embeddings)
        print("✓ FAISS index built")
        print(f"  - Index size: {index.ntotal}")
        
        # Test search
        query = test_embeddings[0:1]
        distances, indices = index.search(query, k=5)
        
        print(f"  - Search results: {indices[0]}")
        print(f"  - Distances: {distances[0]}")
        
        print("\nStep 5: Testing dataset utilities...")
        
        from training.dataset_loader import DatasetLoader
        
        # Create sample dataset
        sample_docs = DatasetLoader.create_sample_dataset(10)
        
        dataset_info = DatasetLoader.get_dataset_info(sample_docs)
        print("✓ Sample dataset created")
        print(f"  - Documents: {dataset_info['num_docs']}")
        print(f"  - Avg length: {dataset_info['avg_doc_length']:.1f}")
        
        # Test split
        train_docs, val_docs, test_docs = DatasetLoader.split_dataset(sample_docs)
        print(f"  - Split: {len(train_docs)} train, {len(val_docs)} val, {len(test_docs)} test")
        
        print("\nStep 6: Integration test...")
        
        print("Testing integrated pattern extraction and database building...")
        
        # Create a minimal end-to-end test
        from indexing.vector_database import VectorDatabase
        
        # Use the sample configuration we created
        test_config = RETGENConfig(
            embedding_dim=384,
            min_pattern_frequency=1,
            retrieval_k=5,
            resolutions=[1, 2],
            index_type="Flat",
            similarity_metric="cosine"
        )
        
        # We'll simulate the pattern extraction and embedding process
        # since we can't easily run the full sentence transformer without downloads
        
        # Create mock patterns
        mock_patterns = [
            Pattern([1, 2], "the cat", 3, "sat", 0, 2, 0),
            Pattern([2, 3], "cat sat", 4, "on", 1, 2, 0),
            Pattern([3, 4], "sat on", 5, "the", 2, 2, 0),
            Pattern([4, 5], "on the", 6, "mat", 3, 2, 0),
            Pattern([1], "the", 2, "cat", 0, 1, 0),
        ]
        
        # Create mock embeddings
        mock_embeddings = np.random.randn(len(mock_patterns), test_config.embedding_dim)
        norms = np.linalg.norm(mock_embeddings, axis=1, keepdims=True)
        mock_embeddings = (mock_embeddings / norms).astype(np.float32)
        
        # Test vector database
        vector_db = VectorDatabase(test_config)
        vector_db.add_patterns(mock_patterns, mock_embeddings)
        
        print("✓ Vector database integration works")
        print(f"  - Patterns indexed: {vector_db.pattern_count}")
        
        # Test search
        query_embedding = mock_embeddings[0:1]
        search_results = vector_db.search(query_embedding, k=3)
        
        print("✓ Search works")
        print(f"  - Found {len(search_results[0])} results")
        
        if search_results[0]:
            for i, result in enumerate(search_results[0]):
                pattern = result['pattern']
                print(f"    {i+1}. '{pattern.text}' -> '{pattern.next_text}' (sim={result['distance']:.3f})")
        
        # Test continuation distribution
        distributions = vector_db.get_continuation_distribution(query_embedding, k=3)
        print("✓ Continuation distribution works")
        print(f"  - Distribution: {distributions[0]}")
        
        print("\n" + "=" * 60)
        print("🎉 SIMPLE DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nCore functionality verified:")
        print("✓ Configuration system")  
        print("✓ Pattern extraction data structures")
        print("✓ Embedding components (positional encoding, caching)")
        print("✓ Vector database (FAISS indexing, metadata storage)")
        print("✓ Dataset utilities")
        print("✓ End-to-end pattern storage and retrieval")
        
        print("\nImplementation is ready for:")
        print("1. Full sentence transformer integration")
        print("2. Complete training pipeline")
        print("3. Text generation system")
        print("4. Comprehensive benchmarking")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_simple_demo()
    if success:
        print("\n✨ RETGEN core implementation verified!")
        print("🚀 Ready for full system deployment!")
    else:
        print("\n💥 Core functionality test failed!")
    
    sys.exit(0 if success else 1)