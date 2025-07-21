"""Tests for FAISS-based vector indexing."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.indexing.vector_database import (
    FAISSIndexBuilder,
    PatternMetadataStore,
    VectorDatabase,
    IndexType
)
from src.core.config import RETGENConfig
from src.data.pattern_extraction import Pattern


class TestFAISSIndexBuilder:
    """Test suite for FAISSIndexBuilder."""
    
    def test_initialization(self):
        """Test index builder initialization."""
        config = RETGENConfig()
        builder = FAISSIndexBuilder(config)
        
        assert builder.config == config
        assert builder.index is None
    
    def test_select_index_type(self):
        """Test automatic index type selection."""
        config = RETGENConfig()
        builder = FAISSIndexBuilder(config)
        
        # Small dataset should use flat index
        index_type = builder._select_index_type(1000, 128)
        assert index_type == IndexType.FLAT
        
        # Medium dataset should use IVF
        index_type = builder._select_index_type(50000, 128)
        assert index_type == IndexType.IVF_FLAT
        
        # Large dataset should use IVF_PQ
        index_type = builder._select_index_type(5000000, 128)
        assert index_type == IndexType.IVF_PQ
    
    def test_build_flat_index(self):
        """Test building flat index."""
        config = RETGENConfig(index_type="Flat")
        builder = FAISSIndexBuilder(config)
        
        # Generate test embeddings
        n_vectors = 100
        dim = 128
        embeddings = np.random.randn(n_vectors, dim).astype(np.float32)
        
        # Build index
        index = builder.build_index(embeddings)
        
        assert index is not None
        assert index.ntotal == n_vectors
        assert index.d == dim
    
    def test_build_ivf_index(self):
        """Test building IVF index."""
        config = RETGENConfig(index_type="IVF100,Flat")
        builder = FAISSIndexBuilder(config)
        
        # Generate test embeddings
        n_vectors = 1000
        dim = 128
        embeddings = np.random.randn(n_vectors, dim).astype(np.float32)
        
        # Build index
        index = builder.build_index(embeddings)
        
        assert index is not None
        assert index.ntotal == n_vectors
        assert index.d == dim
        assert hasattr(index, 'nprobe')
    
    def test_search_index(self):
        """Test searching in built index."""
        config = RETGENConfig(index_type="Flat", similarity_metric="cosine")
        builder = FAISSIndexBuilder(config)
        
        # Generate test embeddings
        n_vectors = 100
        dim = 64
        embeddings = np.random.randn(n_vectors, dim).astype(np.float32)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        # Build index
        index = builder.build_index(embeddings)
        
        # Search for nearest neighbors
        query = embeddings[0:5]  # Use first 5 as queries
        k = 10
        distances, indices = index.search(query, k)
        
        assert distances.shape == (5, k)
        assert indices.shape == (5, k)
        
        # First neighbor should be itself with distance ~1.0 (cosine similarity)
        assert np.all(indices[:, 0] == [0, 1, 2, 3, 4])
        assert np.allclose(distances[:, 0], 1.0, atol=1e-6)
    
    def test_save_load_index(self):
        """Test saving and loading index."""
        config = RETGENConfig()
        builder = FAISSIndexBuilder(config)
        
        # Build index
        embeddings = np.random.randn(100, 64).astype(np.float32)
        index = builder.build_index(embeddings)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save index
            save_path = Path(tmpdir) / "test_index.faiss"
            builder.save_index(save_path)
            assert save_path.exists()
            
            # Load index
            loaded_index = builder.load_index(save_path)
            assert loaded_index.ntotal == index.ntotal
            assert loaded_index.d == index.d


class TestPatternMetadataStore:
    """Test suite for PatternMetadataStore."""
    
    def test_memory_backend(self):
        """Test memory backend for metadata storage."""
        store = PatternMetadataStore(backend='memory')
        
        # Store pattern metadata
        pattern_data = {
            'tokens': [1, 2, 3],
            'text': 'test pattern',
            'continuations': {4: 0.7, 5: 0.3},
            'count': 10
        }
        
        store.store_pattern(0, pattern_data)
        
        # Retrieve pattern
        retrieved = store.get_pattern(0)
        assert retrieved == pattern_data
        
        # Non-existent pattern
        assert store.get_pattern(999) is None
    
    def test_lmdb_backend(self):
        """Test LMDB backend for metadata storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PatternMetadataStore(backend='lmdb', db_path=tmpdir)
            
            # Store multiple patterns
            for i in range(10):
                pattern_data = {
                    'tokens': [i, i+1, i+2],
                    'text': f'pattern {i}',
                    'continuations': {i+3: 1.0},
                    'count': i * 2
                }
                store.store_pattern(i, pattern_data)
            
            # Retrieve patterns
            for i in range(10):
                retrieved = store.get_pattern(i)
                assert retrieved is not None
                assert retrieved['text'] == f'pattern {i}'
            
            # Close and reopen
            store.close()
            
            # Create new store instance
            store2 = PatternMetadataStore(backend='lmdb', db_path=tmpdir)
            
            # Should still have data
            retrieved = store2.get_pattern(5)
            assert retrieved is not None
            assert retrieved['text'] == 'pattern 5'
            
            store2.close()
    
    def test_batch_operations(self):
        """Test batch storage and retrieval."""
        store = PatternMetadataStore(backend='memory')
        
        # Batch store
        patterns = {}
        for i in range(100):
            patterns[i] = {
                'tokens': [i],
                'text': f'pattern {i}',
                'count': i
            }
        
        store.store_patterns_batch(patterns)
        
        # Batch retrieve
        indices = list(range(0, 100, 10))
        retrieved = store.get_patterns_batch(indices)
        
        assert len(retrieved) == 10
        assert all(i in retrieved for i in indices)
        assert retrieved[50]['text'] == 'pattern 50'


class TestVectorDatabase:
    """Test suite for VectorDatabase."""
    
    def test_initialization(self):
        """Test vector database initialization."""
        config = RETGENConfig()
        db = VectorDatabase(config)
        
        assert db.config == config
        assert db.index_builder is not None
        assert db.metadata_store is not None
        assert db.index is None
        assert db.pattern_count == 0
    
    def test_add_patterns(self):
        """Test adding patterns to database."""
        config = RETGENConfig(embedding_dim=64)
        db = VectorDatabase(config)
        
        # Create test patterns and embeddings
        patterns = [
            Pattern([i, i+1], f"pattern {i}", i+2, "x", i, 2, 0)
            for i in range(100)
        ]
        
        embeddings = np.random.randn(100, 64).astype(np.float32)
        
        # Add patterns
        db.add_patterns(patterns, embeddings)
        
        assert db.pattern_count == 100
        assert db.index is not None
        assert db.index.ntotal == 100
    
    def test_search(self):
        """Test searching in database."""
        config = RETGENConfig(
            embedding_dim=64,
            similarity_metric="cosine"
        )
        db = VectorDatabase(config)
        
        # Add patterns
        patterns = []
        embeddings = []
        
        for i in range(100):
            pattern = Pattern([i], f"pattern {i}", i+1, "next", i, 1, 0)
            patterns.append(pattern)
            
            # Create embedding (normalized for cosine similarity)
            emb = np.random.randn(64).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        
        embeddings = np.vstack(embeddings)
        db.add_patterns(patterns, embeddings)
        
        # Search with query
        query = embeddings[10:11]  # Use pattern 10 as query
        results = db.search(query, k=5)
        
        assert len(results) == 1
        assert len(results[0]) == 5
        
        # First result should be pattern 10 itself
        first_result = results[0][0]
        assert first_result['index'] == 10
        assert first_result['pattern'].text == "pattern 10"
        assert first_result['distance'] > 0.99  # High cosine similarity
    
    def test_get_continuation_distribution(self):
        """Test getting continuation distribution."""
        config = RETGENConfig(embedding_dim=64)
        db = VectorDatabase(config)
        
        # Create patterns with different continuations
        patterns = []
        embeddings = []
        
        # Pattern "the cat" with continuations
        for i in range(10):
            if i < 7:
                next_token = 100  # "sat"
            else:
                next_token = 200  # "ran"
            
            pattern = Pattern(
                tokens=[1, 2],  # "the cat"
                text="the cat",
                next_token=next_token,
                next_text="sat" if next_token == 100 else "ran",
                position=i,
                resolution=2,
                document_id=i
            )
            patterns.append(pattern)
            
            # Same embedding for same pattern
            emb = np.ones(64).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        
        embeddings = np.vstack(embeddings)
        db.add_patterns(patterns, embeddings)
        
        # Get continuation distribution
        query = embeddings[0:1]
        distribution = db.get_continuation_distribution(query, k=10)
        
        assert len(distribution) == 1
        assert len(distribution[0]) == 2  # Two different continuations
        assert distribution[0][100] == 0.7  # 7/10
        assert distribution[0][200] == 0.3  # 3/10
    
    def test_save_load(self):
        """Test saving and loading database."""
        config = RETGENConfig(embedding_dim=64)
        db = VectorDatabase(config)
        
        # Add patterns
        patterns = [
            Pattern([i], f"pattern {i}", i+1, "next", i, 1, 0)
            for i in range(50)
        ]
        embeddings = np.random.randn(50, 64).astype(np.float32)
        db.add_patterns(patterns, embeddings)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            
            # Save database
            db.save(save_path)
            
            # Create new database and load
            db2 = VectorDatabase(config)
            db2.load(save_path)
            
            assert db2.pattern_count == 50
            assert db2.index.ntotal == 50
            
            # Test search in loaded database
            query = embeddings[0:1]
            results = db2.search(query, k=5)
            assert len(results[0]) == 5