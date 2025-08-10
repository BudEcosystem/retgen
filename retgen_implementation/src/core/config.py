"""Configuration classes for RETGEN system."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class RETGENConfig:
    """Configuration for RETGEN model."""
    
    # Pattern extraction settings
    min_pattern_length: int = 1
    max_pattern_length: int = 10
    pattern_stride: int = 1
    min_pattern_frequency: int = 2
    
    # Multi-resolution settings
    resolutions: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 8])
    resolution_weights: Optional[List[float]] = None
    
    # Embedding settings
    embedding_dim: int = 768
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    normalize_embeddings: bool = True
    max_sequence_length: int = 512
    
    # Context settings
    use_local_context: bool = True
    use_global_context: bool = True
    use_positional_encoding: bool = True
    positional_encoding_dim: int = 128
    
    # Retrieval settings
    retrieval_k: int = 50
    temperature: float = 1.0
    similarity_metric: str = "cosine"  # cosine, dot, l2
    
    # Index settings
    index_type: str = "IVF1024,PQ64"  # FAISS index factory string
    nprobe: int = 10  # Number of clusters to search
    use_gpu: bool = False
    
    # Storage settings
    max_index_size: int = 10_000_000  # Maximum patterns to store
    metadata_backend: str = "lmdb"  # lmdb, redis, memory
    compression: bool = True
    
    # Generation settings
    max_generation_length: int = 100
    beam_size: int = 1
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0
    
    # Training settings
    batch_size: int = 256
    validation_split: float = 0.1
    checkpoint_interval: int = 10000
    
    # Hardware settings
    device: str = "cuda"  # cuda, cpu
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Logging settings
    log_level: str = "INFO"
    log_interval: int = 100
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.__dict__.copy()
        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'RETGENConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.min_pattern_length > 0, "min_pattern_length must be positive"
        assert self.max_pattern_length >= self.min_pattern_length, \
            "max_pattern_length must be >= min_pattern_length"
        assert self.pattern_stride > 0, "pattern_stride must be positive"
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        assert self.retrieval_k > 0, "retrieval_k must be positive"
        assert self.temperature > 0, "temperature must be positive"
        assert 0 < self.top_p <= 1, "top_p must be in (0, 1]"
        assert self.similarity_metric in ["cosine", "dot", "l2"], \
            f"Unknown similarity metric: {self.similarity_metric}"
        
        if self.resolution_weights is not None:
            assert len(self.resolution_weights) == len(self.resolutions), \
                "resolution_weights must match resolutions length"
            assert abs(sum(self.resolution_weights) - 1.0) < 1e-6, \
                "resolution_weights must sum to 1"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)
    
    def get_index_params(self) -> Dict[str, Any]:
        """Get parameters for FAISS index construction."""
        return {
            "index_type": self.index_type,
            "metric": self.similarity_metric,
            "nprobe": self.nprobe,
            "use_gpu": self.use_gpu,
        }
    
    def get_embedding_params(self) -> Dict[str, Any]:
        """Get parameters for embedding model."""
        return {
            "model_name": self.embedding_model,
            "normalize": self.normalize_embeddings,
            "max_length": self.max_sequence_length,
            "device": self.device,
        }


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    coverage: List[float] = field(default_factory=list)
    retrieval_quality: List[float] = field(default_factory=list)
    perplexity: List[float] = field(default_factory=list)
    index_size: List[int] = field(default_factory=list)
    inference_speed: List[float] = field(default_factory=list)
    training_time: List[float] = field(default_factory=list)
    
    def add_metrics(
        self,
        coverage: float,
        retrieval_quality: float,
        perplexity: float,
        index_size: int,
        inference_speed: float,
        training_time: float
    ) -> None:
        """Add new metrics."""
        self.coverage.append(coverage)
        self.retrieval_quality.append(retrieval_quality)
        self.perplexity.append(perplexity)
        self.index_size.append(index_size)
        self.inference_speed.append(inference_speed)
        self.training_time.append(training_time)
    
    def save(self, path: Path) -> None:
        """Save metrics to JSON file."""
        metrics_dict = {
            "coverage": self.coverage,
            "retrieval_quality": self.retrieval_quality,
            "perplexity": self.perplexity,
            "index_size": self.index_size,
            "inference_speed": self.inference_speed,
            "training_time": self.training_time,
        }
        
        with open(path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'TrainingMetrics':
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            metrics_dict = json.load(f)
        
        return cls(**metrics_dict)