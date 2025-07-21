"""Context-aware embedding system for RETGEN."""

import numpy as np
import torch
from typing import List, Optional, Dict, Any, Tuple, Union
from collections import OrderedDict
import logging
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import RETGENConfig
from data.pattern_extraction import Pattern


logger = logging.getLogger(__name__)


class PositionalEncoder:
    """Sinusoidal positional encoding for sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoder.
        
        Args:
            d_model: Dimension of the encoding
            max_len: Maximum sequence length
        """
        self.d_model = d_model
        self.max_len = max_len
        self.pe_matrix = self._create_pe_matrix()
    
    def _create_pe_matrix(self) -> np.ndarray:
        """Create sinusoidal position encoding matrix.
        
        Returns:
            Position encoding matrix of shape (max_len, d_model)
        """
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len).reshape(-1, 1)
        
        # Create div_term for sinusoidal pattern
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * 
            -(np.log(10000.0) / self.d_model)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cos to odd indices
        if self.d_model % 2 == 0:
            pe[:, 1::2] = np.cos(position * div_term)
        else:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        
        return pe
    
    def encode_position(self, position: int, seq_len: int) -> np.ndarray:
        """Encode absolute and relative position.
        
        Args:
            position: Absolute position in sequence
            seq_len: Total sequence length
            
        Returns:
            Concatenated absolute and relative position encoding
        """
        # Absolute position encoding
        abs_pos = self.pe_matrix[position].copy()
        
        # Relative position from end
        rel_pos = self.pe_matrix[seq_len - position - 1].copy()
        
        # Concatenate absolute and relative
        return np.concatenate([abs_pos, rel_pos])


class ContextAwareEmbedder:
    """Embedder that incorporates context information."""
    
    def __init__(self, config: RETGENConfig):
        """Initialize context-aware embedder.
        
        Args:
            config: RETGEN configuration
        """
        self.config = config
        
        # Initialize sentence encoder
        import torch
        # Force CPU if CUDA not available
        device = config.device
        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"
            logger.info("CUDA not available, using CPU")
        
        self.sentence_encoder = SentenceTransformer(
            config.embedding_model,
            device=device
        )
        
        # Update embedding dimension based on actual model
        actual_dim = self.sentence_encoder.get_sentence_embedding_dimension()
        if actual_dim != config.embedding_dim:
            logger.warning(
                f"Model {config.embedding_model} has embedding dim {actual_dim}, "
                f"not {config.embedding_dim}. Using {actual_dim}."
            )
            self.embedding_dim = actual_dim
        else:
            self.embedding_dim = config.embedding_dim
        
        # Initialize positional encoder if needed
        if config.use_positional_encoding:
            self.positional_encoder = PositionalEncoder(
                d_model=config.positional_encoding_dim,
                max_len=config.max_sequence_length
            )
        else:
            self.positional_encoder = None
        
        # Calculate total embedding dimension
        self.total_dim = self.embedding_dim
        if config.use_positional_encoding:
            self.total_dim += 2 * config.positional_encoding_dim
    
    def embed_local_context(self, text: str) -> np.ndarray:
        """Embed text using sentence transformer.
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding
        """
        embedding = self.sentence_encoder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings
        )
        
        return embedding
    
    def embed_pattern(
        self,
        pattern: Pattern,
        sequence_length: Optional[int] = None
    ) -> np.ndarray:
        """Embed a pattern with optional positional information.
        
        Args:
            pattern: Pattern to embed
            sequence_length: Total sequence length for relative positioning
            
        Returns:
            Pattern embedding with optional positional encoding
        """
        # Get base text embedding
        text_embedding = self.embed_local_context(pattern.text)
        
        # Add positional encoding if configured
        if self.config.use_positional_encoding and self.positional_encoder:
            if sequence_length is None:
                sequence_length = pattern.position + pattern.resolution + 10
            
            pos_encoding = self.positional_encoder.encode_position(
                pattern.position,
                sequence_length
            )
            
            # Concatenate text and positional embeddings
            embedding = np.concatenate([text_embedding, pos_encoding])
        else:
            embedding = text_embedding
        
        return embedding
    
    def embed_patterns(
        self,
        patterns: List[Pattern],
        sequence_length: Optional[int] = None,
        batch_size: int = 256,
        show_progress: bool = False
    ) -> np.ndarray:
        """Embed multiple patterns in batches.
        
        Args:
            patterns: List of patterns to embed
            sequence_length: Total sequence length for relative positioning
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        if not patterns:
            return np.array([])
        
        # Extract texts for batch encoding
        texts = [p.text for p in patterns]
        
        # Batch encode texts
        text_embeddings = self.sentence_encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=show_progress
        )
        
        # Add positional encodings if needed
        if self.config.use_positional_encoding and self.positional_encoder:
            embeddings = []
            
            for i, pattern in enumerate(patterns):
                text_emb = text_embeddings[i]
                
                if sequence_length is None:
                    seq_len = pattern.position + pattern.resolution + 10
                else:
                    seq_len = sequence_length
                
                pos_encoding = self.positional_encoder.encode_position(
                    pattern.position,
                    seq_len
                )
                
                combined = np.concatenate([text_emb, pos_encoding])
                embeddings.append(combined)
            
            return np.vstack(embeddings)
        else:
            return text_embeddings


class EmbeddingCache:
    """LRU cache for embeddings."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached embedding or None
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key].copy()
        else:
            self.misses += 1
            return None
    
    def set(self, key: str, embedding: np.ndarray) -> None:
        """Set embedding in cache.
        
        Args:
            key: Cache key
            embedding: Embedding to cache
        """
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = embedding.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class RETGENEmbedder:
    """Main embedder for RETGEN system."""
    
    def __init__(self, config: RETGENConfig):
        """Initialize RETGEN embedder.
        
        Args:
            config: RETGEN configuration
        """
        self.config = config
        self.context_embedder = ContextAwareEmbedder(config)
        self.cache = EmbeddingCache(max_size=10000)
        
        # Update total dimension from context embedder
        self.embedding_dim = self.context_embedder.total_dim
    
    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Embed text with caching.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cache
            
        Returns:
            Text embedding
        """
        if use_cache:
            # Check cache
            cache_key = f"text:{text}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Compute embedding
        embedding = self.context_embedder.embed_local_context(text)
        
        if use_cache:
            self.cache.set(cache_key, embedding)
        
        return embedding
    
    def embed_pattern(
        self,
        pattern: Pattern,
        sequence_length: Optional[int] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """Embed pattern with caching.
        
        Args:
            pattern: Pattern to embed
            sequence_length: Total sequence length
            use_cache: Whether to use cache
            
        Returns:
            Pattern embedding
        """
        if use_cache:
            # Create cache key including position info
            cache_key = f"pattern:{pattern.get_key()}:{pattern.position}:{sequence_length}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Compute embedding
        embedding = self.context_embedder.embed_pattern(pattern, sequence_length)
        
        if use_cache:
            self.cache.set(cache_key, embedding)
        
        return embedding
    
    def embed_patterns(
        self,
        patterns: List[Pattern],
        sequence_length: Optional[int] = None,
        batch_size: int = 256,
        show_progress: bool = False,
        use_cache: bool = False
    ) -> np.ndarray:
        """Embed multiple patterns.
        
        Args:
            patterns: List of patterns
            sequence_length: Total sequence length
            batch_size: Batch size for encoding
            show_progress: Whether to show progress
            use_cache: Whether to use cache (not recommended for large batches)
            
        Returns:
            Array of embeddings
        """
        if use_cache:
            embeddings = []
            for pattern in patterns:
                emb = self.embed_pattern(pattern, sequence_length, use_cache=True)
                embeddings.append(emb)
            return np.vstack(embeddings)
        else:
            return self.context_embedder.embed_patterns(
                patterns,
                sequence_length,
                batch_size,
                show_progress
            )
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score
        """
        if self.config.similarity_metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        elif self.config.similarity_metric == "dot":
            # Dot product similarity
            return np.dot(embedding1, embedding2)
        
        elif self.config.similarity_metric == "l2":
            # Negative L2 distance (higher is more similar)
            return -np.linalg.norm(embedding1 - embedding2)
        
        else:
            raise ValueError(f"Unknown similarity metric: {self.config.similarity_metric}")
    
    def get_embedding_dim(self) -> int:
        """Get total embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        return self.cache.get_stats()