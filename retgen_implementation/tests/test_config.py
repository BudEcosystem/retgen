"""Tests for RETGEN configuration."""

import pytest
import json
from pathlib import Path
import tempfile
from src.core.config import RETGENConfig, TrainingMetrics


class TestRETGENConfig:
    """Test suite for RETGENConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RETGENConfig()
        
        # Pattern extraction
        assert config.min_pattern_length == 1
        assert config.max_pattern_length == 10
        assert config.pattern_stride == 1
        assert config.min_pattern_frequency == 2
        
        # Multi-resolution
        assert config.resolutions == [1, 2, 3, 5, 8]
        assert config.resolution_weights is None
        
        # Embedding
        assert config.embedding_dim == 768
        assert config.embedding_model == "sentence-transformers/all-mpnet-base-v2"
        assert config.normalize_embeddings is True
        
        # Retrieval
        assert config.retrieval_k == 50
        assert config.temperature == 1.0
        assert config.similarity_metric == "cosine"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RETGENConfig(
            min_pattern_length=2,
            max_pattern_length=20,
            embedding_dim=512,
            retrieval_k=100,
            temperature=0.5
        )
        
        assert config.min_pattern_length == 2
        assert config.max_pattern_length == 20
        assert config.embedding_dim == 512
        assert config.retrieval_k == 100
        assert config.temperature == 0.5
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = RETGENConfig()
        config.validate()
        
        # Invalid min_pattern_length
        with pytest.raises(AssertionError):
            config = RETGENConfig(min_pattern_length=0)
            config.validate()
        
        # Invalid max < min pattern length
        with pytest.raises(AssertionError):
            config = RETGENConfig(min_pattern_length=10, max_pattern_length=5)
            config.validate()
        
        # Invalid temperature
        with pytest.raises(AssertionError):
            config = RETGENConfig(temperature=-1)
            config.validate()
        
        # Invalid top_p
        with pytest.raises(AssertionError):
            config = RETGENConfig(top_p=1.5)
            config.validate()
        
        # Invalid similarity metric
        with pytest.raises(AssertionError):
            config = RETGENConfig(similarity_metric="invalid")
            config.validate()
        
        # Invalid resolution weights
        with pytest.raises(AssertionError):
            config = RETGENConfig(resolution_weights=[0.5, 0.3])  # Wrong length
            config.validate()
        
        with pytest.raises(AssertionError):
            config = RETGENConfig(
                resolutions=[1, 2, 3],
                resolution_weights=[0.3, 0.3, 0.3]  # Doesn't sum to 1
            )
            config.validate()
    
    def test_save_load_config(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            # Create and save config
            config1 = RETGENConfig(
                min_pattern_length=3,
                embedding_dim=512,
                retrieval_k=75,
                resolutions=[1, 3, 5, 7]
            )
            config1.save(config_path)
            
            # Load config
            config2 = RETGENConfig.load(config_path)
            
            # Verify loaded values
            assert config2.min_pattern_length == 3
            assert config2.embedding_dim == 512
            assert config2.retrieval_k == 75
            assert config2.resolutions == [1, 3, 5, 7]
    
    def test_get_index_params(self):
        """Test getting FAISS index parameters."""
        config = RETGENConfig(
            index_type="IVF512,PQ32",
            similarity_metric="dot",
            nprobe=20,
            use_gpu=True
        )
        
        params = config.get_index_params()
        assert params["index_type"] == "IVF512,PQ32"
        assert params["metric"] == "dot"
        assert params["nprobe"] == 20
        assert params["use_gpu"] is True
    
    def test_get_embedding_params(self):
        """Test getting embedding parameters."""
        config = RETGENConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            normalize_embeddings=False,
            max_sequence_length=256,
            device="cpu"
        )
        
        params = config.get_embedding_params()
        assert params["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert params["normalize"] is False
        assert params["max_length"] == 256
        assert params["device"] == "cpu"


class TestTrainingMetrics:
    """Test suite for TrainingMetrics."""
    
    def test_add_metrics(self):
        """Test adding metrics."""
        metrics = TrainingMetrics()
        
        # Add first set of metrics
        metrics.add_metrics(
            coverage=0.85,
            retrieval_quality=0.92,
            perplexity=45.3,
            index_size=10000,
            inference_speed=150.5,
            training_time=120.0
        )
        
        assert len(metrics.coverage) == 1
        assert metrics.coverage[0] == 0.85
        assert metrics.retrieval_quality[0] == 0.92
        assert metrics.perplexity[0] == 45.3
        assert metrics.index_size[0] == 10000
        assert metrics.inference_speed[0] == 150.5
        assert metrics.training_time[0] == 120.0
        
        # Add second set
        metrics.add_metrics(
            coverage=0.88,
            retrieval_quality=0.94,
            perplexity=42.1,
            index_size=20000,
            inference_speed=145.2,
            training_time=240.0
        )
        
        assert len(metrics.coverage) == 2
        assert metrics.coverage[1] == 0.88
        assert metrics.index_size[1] == 20000
    
    def test_save_load_metrics(self):
        """Test saving and loading metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            
            # Create and populate metrics
            metrics1 = TrainingMetrics()
            for i in range(3):
                metrics1.add_metrics(
                    coverage=0.8 + i * 0.05,
                    retrieval_quality=0.9 + i * 0.02,
                    perplexity=50 - i * 2,
                    index_size=(i + 1) * 10000,
                    inference_speed=150 - i * 5,
                    training_time=(i + 1) * 100
                )
            
            # Save metrics
            metrics1.save(metrics_path)
            
            # Load metrics
            metrics2 = TrainingMetrics.load(metrics_path)
            
            # Verify loaded values
            assert len(metrics2.coverage) == 3
            assert metrics2.coverage == metrics1.coverage
            assert metrics2.perplexity == metrics1.perplexity
            assert metrics2.index_size == metrics1.index_size