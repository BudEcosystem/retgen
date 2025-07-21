"""Tests for RETGEN embedding system."""

import pytest
import numpy as np
import torch
from typing import List

from src.embeddings.context_embeddings import (
    PositionalEncoder,
    ContextAwareEmbedder,
    RETGENEmbedder,
    EmbeddingCache
)
from src.core.config import RETGENConfig
from src.data.pattern_extraction import Pattern


class TestPositionalEncoder:
    """Test suite for PositionalEncoder."""
    
    def test_initialization(self):
        """Test positional encoder initialization."""
        d_model = 128
        max_len = 1000
        encoder = PositionalEncoder(d_model=d_model, max_len=max_len)
        
        assert encoder.d_model == d_model
        assert encoder.max_len == max_len
        assert encoder.pe_matrix.shape == (max_len, d_model)
    
    def test_sinusoidal_encoding(self):
        """Test sinusoidal position encoding properties."""
        encoder = PositionalEncoder(d_model=64, max_len=100)
        
        # Check encoding shape
        pe_matrix = encoder.pe_matrix
        assert pe_matrix.shape == (100, 64)
        
        # Check that values are bounded
        assert np.all(np.abs(pe_matrix) <= 1.0)
        
        # Check alternating sin/cos pattern
        # Even indices should be sin, odd should be cos
        pos_0 = encoder.encode_position(0, 10)
        pos_1 = encoder.encode_position(1, 10)
        
        # Positions should be different
        assert not np.allclose(pos_0, pos_1)
    
    def test_encode_position(self):
        """Test position encoding with absolute and relative positions."""
        encoder = PositionalEncoder(d_model=32, max_len=100)
        
        # Test encoding
        position = 5
        seq_len = 20
        encoding = encoder.encode_position(position, seq_len)
        
        # Should concatenate absolute and relative positions
        assert encoding.shape == (64,)  # 2 * d_model
        
        # Test edge cases
        encoding_start = encoder.encode_position(0, seq_len)
        encoding_end = encoder.encode_position(seq_len - 1, seq_len)
        
        assert encoding_start.shape == encoding_end.shape


class TestContextAwareEmbedder:
    """Test suite for ContextAwareEmbedder."""
    
    def test_initialization(self):
        """Test context-aware embedder initialization."""
        config = RETGENConfig(
            embedding_dim=768,
            positional_encoding_dim=128
        )
        embedder = ContextAwareEmbedder(config)
        
        assert embedder.config == config
        assert embedder.sentence_encoder is not None
        assert embedder.positional_encoder is not None
    
    def test_embed_local_context(self):
        """Test local context embedding."""
        config = RETGENConfig(embedding_dim=384)
        embedder = ContextAwareEmbedder(config)
        
        text = "This is a test"
        embedding = embedder.embed_local_context(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        
        # Test normalization
        if config.normalize_embeddings:
            norm = np.linalg.norm(embedding)
            assert np.isclose(norm, 1.0, atol=1e-6)
    
    def test_embed_with_position(self):
        """Test embedding with positional information."""
        config = RETGENConfig(
            embedding_dim=384,
            positional_encoding_dim=64,
            use_positional_encoding=True
        )
        embedder = ContextAwareEmbedder(config)
        
        pattern = Pattern(
            tokens=[1, 2, 3],
            text="test pattern",
            next_token=4,
            next_text="next",
            position=10,
            resolution=3,
            document_id=0
        )
        
        embedding = embedder.embed_pattern(pattern, sequence_length=50)
        
        # Should include positional encoding
        expected_dim = 384 + 2 * 64  # embedding_dim + 2 * positional_dim
        assert embedding.shape == (expected_dim,)
    
    def test_embed_without_position(self):
        """Test embedding without positional information."""
        config = RETGENConfig(
            embedding_dim=384,
            use_positional_encoding=False
        )
        embedder = ContextAwareEmbedder(config)
        
        pattern = Pattern(
            tokens=[1, 2, 3],
            text="test pattern",
            next_token=4,
            next_text="next",
            position=10,
            resolution=3,
            document_id=0
        )
        
        embedding = embedder.embed_pattern(pattern, sequence_length=50)
        
        # Should not include positional encoding
        assert embedding.shape == (384,)
    
    def test_batch_embedding(self):
        """Test batch embedding of patterns."""
        config = RETGENConfig(embedding_dim=384)
        embedder = ContextAwareEmbedder(config)
        
        patterns = [
            Pattern([1, 2], "test 1", 3, "a", 0, 2, 0),
            Pattern([4, 5], "test 2", 6, "b", 1, 2, 0),
            Pattern([7, 8], "test 3", 9, "c", 2, 2, 0),
        ]
        
        embeddings = embedder.embed_patterns(patterns, sequence_length=10)
        
        assert embeddings.shape == (3, 384)
        
        # Each embedding should be normalized if configured
        if config.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-6)


class TestRETGENEmbedder:
    """Test suite for RETGENEmbedder."""
    
    def test_initialization(self):
        """Test RETGEN embedder initialization."""
        config = RETGENConfig()
        embedder = RETGENEmbedder(config)
        
        assert embedder.config == config
        assert embedder.context_embedder is not None
        assert embedder.cache is not None
    
    def test_embed_text(self):
        """Test text embedding."""
        config = RETGENConfig(embedding_dim=384)
        embedder = RETGENEmbedder(config)
        
        text = "This is a test sentence"
        embedding = embedder.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
    
    def test_embed_pattern_with_cache(self):
        """Test pattern embedding with caching."""
        config = RETGENConfig(embedding_dim=384)
        embedder = RETGENEmbedder(config)
        
        pattern = Pattern(
            tokens=[1, 2, 3],
            text="cached pattern",
            next_token=4,
            next_text="next",
            position=0,
            resolution=3,
            document_id=0
        )
        
        # First call should compute
        embedding1 = embedder.embed_pattern(pattern)
        
        # Second call should use cache
        embedding2 = embedder.embed_pattern(pattern)
        
        assert np.array_equal(embedding1, embedding2)
        assert embedder.cache.hits > 0
    
    def test_embed_patterns_batch(self):
        """Test batch embedding of patterns."""
        config = RETGENConfig(
            embedding_dim=384,
            use_positional_encoding=True,
            positional_encoding_dim=64
        )
        embedder = RETGENEmbedder(config)
        
        patterns = [
            Pattern([i, i+1], f"pattern {i}", i+2, "x", i, 2, 0)
            for i in range(10)
        ]
        
        embeddings = embedder.embed_patterns(patterns, batch_size=4)
        
        expected_dim = 384 + 2 * 64
        assert embeddings.shape == (10, expected_dim)
        
        # Check all embeddings are valid
        assert not np.any(np.isnan(embeddings))
        assert not np.any(np.isinf(embeddings))
    
    def test_similarity_computation(self):
        """Test similarity computation between embeddings."""
        config = RETGENConfig(
            embedding_dim=384,
            similarity_metric="cosine"
        )
        embedder = RETGENEmbedder(config)
        
        text1 = "The cat sat on the mat"
        text2 = "The dog sat on the mat"
        text3 = "Something completely different"
        
        emb1 = embedder.embed_text(text1)
        emb2 = embedder.embed_text(text2)
        emb3 = embedder.embed_text(text3)
        
        # Similar texts should have higher similarity
        sim_12 = embedder.compute_similarity(emb1, emb2)
        sim_13 = embedder.compute_similarity(emb1, emb3)
        
        assert sim_12 > sim_13
        
        # Self-similarity should be maximum (1.0 for cosine)
        sim_11 = embedder.compute_similarity(emb1, emb1)
        assert np.isclose(sim_11, 1.0, atol=1e-6)


class TestEmbeddingCache:
    """Test suite for EmbeddingCache."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = EmbeddingCache(max_size=100)
        
        assert cache.max_size == 100
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_cache_operations(self):
        """Test cache get and set operations."""
        cache = EmbeddingCache(max_size=3)
        
        # Test miss
        key = "test_key"
        result = cache.get(key)
        assert result is None
        assert cache.misses == 1
        
        # Test set
        embedding = np.random.randn(128)
        cache.set(key, embedding)
        
        # Test hit
        result = cache.get(key)
        assert np.array_equal(result, embedding)
        assert cache.hits == 1
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = EmbeddingCache(max_size=2)
        
        # Fill cache
        emb1 = np.array([1, 2, 3])
        emb2 = np.array([4, 5, 6])
        emb3 = np.array([7, 8, 9])
        
        cache.set("key1", emb1)
        cache.set("key2", emb2)
        
        # Access key1 to make it more recently used
        _ = cache.get("key1")
        
        # Add key3, should evict key2 (least recently used)
        cache.set("key3", emb3)
        
        assert cache.get("key1") is not None
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") is not None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = EmbeddingCache(max_size=10)
        
        # Generate some hits and misses
        for i in range(5):
            cache.set(f"key{i}", np.random.randn(10))
        
        for i in range(10):
            if i < 5:
                cache.get(f"key{i}")  # Hit
            else:
                cache.get(f"key{i}")  # Miss
        
        stats = cache.get_stats()
        assert stats["size"] == 5
        assert stats["hits"] == 5
        assert stats["misses"] == 5
        assert stats["hit_rate"] == 0.5