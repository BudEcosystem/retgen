"""FAISS-based vector database for RETGEN."""

from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import logging
from collections import defaultdict
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")

try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False
    print("LMDB not available. Using memory backend only.")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import RETGENConfig
from data.pattern_extraction import Pattern


logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Types of FAISS indices."""
    FLAT = "Flat"
    IVF_FLAT = "IVF,Flat"
    IVF_PQ = "IVF,PQ"
    HNSW = "HNSW"


class FAISSIndexBuilder:
    """Builder for FAISS vector indices."""
    
    def __init__(self, config: RETGENConfig):
        """Initialize FAISS index builder.
        
        Args:
            config: RETGEN configuration
        """
        self.config = config
        self.index = None
    
    def _select_index_type(self, n_vectors: int, dim: int) -> IndexType:
        """Select appropriate index type based on dataset size.
        
        Args:
            n_vectors: Number of vectors
            dim: Vector dimension
            
        Returns:
            Appropriate index type
        """
        if n_vectors < 10_000:
            return IndexType.FLAT
        elif n_vectors < 1_000_000:
            return IndexType.IVF_FLAT
        else:
            return IndexType.IVF_PQ
    
    def _create_metric_type(self) -> int:
        """Create FAISS metric type from config.
        
        Returns:
            FAISS metric type
        """
        if self.config.similarity_metric == "cosine":
            return faiss.METRIC_INNER_PRODUCT  # Assumes normalized vectors
        elif self.config.similarity_metric == "dot":
            return faiss.METRIC_INNER_PRODUCT
        elif self.config.similarity_metric == "l2":
            return faiss.METRIC_L2
        else:
            raise ValueError(f"Unsupported metric: {self.config.similarity_metric}")
    
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index from embeddings.
        
        Args:
            embeddings: Array of embeddings (n_vectors, dim)
            
        Returns:
            Built FAISS index
        """
        n_vectors, dim = embeddings.shape
        embeddings = embeddings.astype(np.float32)
        
        logger.info(f"Building index for {n_vectors} vectors of dimension {dim}")
        
        # Normalize embeddings for cosine similarity
        if self.config.similarity_metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms
        
        # Create metric type
        metric = self._create_metric_type()
        
        # Parse index type from config or select automatically
        if self.config.index_type == "auto":
            index_type = self._select_index_type(n_vectors, dim)
        else:
            # Use config-specified index type
            index_factory_str = self.config.index_type
        
        # Create index based on type
        if self.config.index_type == "Flat":
            # Exact search
            if metric == faiss.METRIC_INNER_PRODUCT:
                index = faiss.IndexFlatIP(dim)
            else:
                index = faiss.IndexFlatL2(dim)
                
        elif "IVF" in self.config.index_type:
            # Parse IVF parameters
            if "," in self.config.index_type:
                parts = self.config.index_type.split(",")
                if parts[0].startswith("IVF"):
                    nlist = int(parts[0][3:])  # Extract number after "IVF"
                else:
                    nlist = min(4 * int(np.sqrt(n_vectors)), n_vectors // 39)
            else:
                nlist = min(4 * int(np.sqrt(n_vectors)), n_vectors // 39)
            
            # Create quantizer
            if metric == faiss.METRIC_INNER_PRODUCT:
                quantizer = faiss.IndexFlatIP(dim)
            else:
                quantizer = faiss.IndexFlatL2(dim)
            
            # Create IVF index
            if "PQ" in self.config.index_type:
                # Product quantization
                m = min(64, dim // 4)  # Number of subquantizers
                index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
            else:
                # Flat storage within clusters
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric)
        
        else:
            # Use factory string
            index = faiss.index_factory(dim, self.config.index_type, metric)
        
        # Move to GPU if requested and available
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # Train index if needed
        if not index.is_trained:
            logger.info("Training index...")
            index.train(embeddings)
        
        # Add vectors to index
        logger.info("Adding vectors to index...")
        index.add(embeddings)
        
        # Set search parameters
        if hasattr(index, 'nprobe'):
            index.nprobe = self.config.nprobe
        
        logger.info(f"Built index with {index.ntotal} vectors")
        
        self.index = index
        return index
    
    def save_index(self, path: Path) -> None:
        """Save index to disk.
        
        Args:
            path: Path to save index
        """
        if self.index is None:
            raise RuntimeError("No index to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move to CPU if on GPU
        if hasattr(self.index, 'index'):  # GPU index wrapper
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
        
        faiss.write_index(cpu_index, str(path))
        logger.info(f"Saved index to {path}")
    
    def load_index(self, path: Path) -> faiss.Index:
        """Load index from disk.
        
        Args:
            path: Path to load index from
            
        Returns:
            Loaded FAISS index
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        
        index = faiss.read_index(str(path))
        
        # Move to GPU if requested
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # Set search parameters
        if hasattr(index, 'nprobe'):
            index.nprobe = self.config.nprobe
        
        self.index = index
        logger.info(f"Loaded index with {index.ntotal} vectors from {path}")
        return index


class PatternMetadataStore:
    """Storage for pattern metadata."""
    
    def __init__(self, backend: str = "memory", db_path: Optional[str] = None):
        """Initialize metadata store.
        
        Args:
            backend: Storage backend ("memory", "lmdb")
            db_path: Path for persistent storage
        """
        self.backend = backend
        
        if backend == "memory":
            self.data: Dict[int, Dict[str, Any]] = {}
        elif backend == "lmdb":
            if not LMDB_AVAILABLE:
                raise ImportError("LMDB not available")
            
            if db_path is None:
                db_path = "./pattern_metadata"
            
            # Open LMDB environment
            self.env = lmdb.open(
                db_path,
                map_size=100 * 1024 * 1024 * 1024,  # 100GB
                max_dbs=1
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def store_pattern(self, index: int, pattern_data: Dict[str, Any]) -> None:
        """Store pattern metadata.
        
        Args:
            index: Pattern index
            pattern_data: Pattern metadata
        """
        if self.backend == "memory":
            self.data[index] = pattern_data.copy()
        elif self.backend == "lmdb":
            with self.env.begin(write=True) as txn:
                txn.put(
                    str(index).encode(),
                    pickle.dumps(pattern_data)
                )
    
    def get_pattern(self, index: int) -> Optional[Dict[str, Any]]:
        """Get pattern metadata.
        
        Args:
            index: Pattern index
            
        Returns:
            Pattern metadata or None
        """
        if self.backend == "memory":
            return self.data.get(index)
        elif self.backend == "lmdb":
            with self.env.begin() as txn:
                data = txn.get(str(index).encode())
                return pickle.loads(data) if data else None
    
    def store_patterns_batch(self, patterns: Dict[int, Dict[str, Any]]) -> None:
        """Store multiple patterns in batch.
        
        Args:
            patterns: Dictionary mapping indices to pattern data
        """
        if self.backend == "memory":
            self.data.update(patterns)
        elif self.backend == "lmdb":
            with self.env.begin(write=True) as txn:
                for index, pattern_data in patterns.items():
                    txn.put(
                        str(index).encode(),
                        pickle.dumps(pattern_data)
                    )
    
    def get_patterns_batch(self, indices: List[int]) -> Dict[int, Dict[str, Any]]:
        """Get multiple patterns in batch.
        
        Args:
            indices: List of pattern indices
            
        Returns:
            Dictionary mapping indices to pattern data
        """
        result = {}
        
        if self.backend == "memory":
            for index in indices:
                if index in self.data:
                    result[index] = self.data[index]
        elif self.backend == "lmdb":
            with self.env.begin() as txn:
                for index in indices:
                    data = txn.get(str(index).encode())
                    if data:
                        result[index] = pickle.loads(data)
        
        return result
    
    def close(self) -> None:
        """Close the store."""
        if self.backend == "lmdb" and hasattr(self, 'env'):
            self.env.close()


class VectorDatabase:
    """Complete vector database for patterns."""
    
    def __init__(self, config: RETGENConfig):
        """Initialize vector database.
        
        Args:
            config: RETGEN configuration
        """
        self.config = config
        self.index_builder = FAISSIndexBuilder(config)
        self.metadata_store = PatternMetadataStore(
            backend=config.metadata_backend
        )
        
        self.index: Optional[faiss.Index] = None
        self.pattern_count = 0
    
    def add_patterns(
        self,
        patterns: List[Pattern],
        embeddings: np.ndarray
    ) -> None:
        """Add patterns to the database.
        
        Args:
            patterns: List of patterns
            embeddings: Corresponding embeddings
        """
        if len(patterns) != embeddings.shape[0]:
            raise ValueError("Number of patterns and embeddings must match")
        
        # Build or update index
        if self.index is None:
            self.index = self.index_builder.build_index(embeddings)
        else:
            # Add to existing index
            if self.config.similarity_metric == "cosine":
                # Normalize new embeddings
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embeddings = embeddings / norms
            
            self.index.add(embeddings.astype(np.float32))
        
        # Store pattern metadata
        metadata_batch = {}
        for i, pattern in enumerate(patterns):
            pattern_data = {
                'tokens': pattern.tokens,
                'text': pattern.text,
                'next_token': pattern.next_token,
                'next_text': pattern.next_text,
                'position': pattern.position,
                'resolution': pattern.resolution,
                'document_id': pattern.document_id
            }
            metadata_batch[self.pattern_count + i] = pattern_data
        
        self.metadata_store.store_patterns_batch(metadata_batch)
        self.pattern_count += len(patterns)
        
        logger.info(f"Added {len(patterns)} patterns. Total: {self.pattern_count}")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int
    ) -> List[List[Dict[str, Any]]]:
        """Search for similar patterns.
        
        Args:
            query_embeddings: Query embeddings (n_queries, dim)
            k: Number of nearest neighbors
            
        Returns:
            List of results for each query
        """
        if self.index is None:
            raise RuntimeError("No index built")
        
        # Normalize queries for cosine similarity
        if self.config.similarity_metric == "cosine":
            norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            query_embeddings = query_embeddings / norms
        
        # Search index
        distances, indices = self.index.search(
            query_embeddings.astype(np.float32),
            k
        )
        
        # Get metadata for results
        all_results = []
        for i in range(len(query_embeddings)):
            query_results = []
            
            # Get metadata for this query's results
            result_indices = indices[i].tolist()
            metadata_batch = self.metadata_store.get_patterns_batch(result_indices)
            
            for j, idx in enumerate(result_indices):
                if idx in metadata_batch:
                    pattern_data = metadata_batch[idx]
                    result = {
                        'index': idx,
                        'distance': float(distances[i, j]),
                        'pattern': Pattern(**pattern_data)
                    }
                    query_results.append(result)
            
            all_results.append(query_results)
        
        return all_results
    
    def get_continuation_distribution(
        self,
        query_embeddings: np.ndarray,
        k: int
    ) -> List[Dict[int, float]]:
        """Get continuation probability distribution for queries.
        
        Args:
            query_embeddings: Query embeddings
            k: Number of neighbors to consider
            
        Returns:
            List of continuation distributions for each query
        """
        results = self.search(query_embeddings, k)
        
        distributions = []
        for query_results in results:
            # Count continuations weighted by similarity
            continuation_scores = defaultdict(float)
            total_score = 0
            
            for result in query_results:
                # Use distance as weight (higher distance = lower weight)
                if self.config.similarity_metric == "cosine":
                    weight = result['distance']  # Already similarity for cosine
                else:
                    weight = 1.0 / (1.0 + result['distance'])  # Convert distance to weight
                
                continuation_scores[result['pattern'].next_token] += weight
                total_score += weight
            
            # Normalize to probabilities
            if total_score > 0:
                distribution = {
                    token: score / total_score
                    for token, score in continuation_scores.items()
                }
            else:
                distribution = {}
            
            distributions.append(distribution)
        
        return distributions
    
    def save(self, path: Path) -> None:
        """Save database to disk.
        
        Args:
            path: Directory to save database
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save index
        if self.index is not None:
            index_path = path / "index.faiss"
            self.index_builder.save_index(index_path)
        
        # Save metadata about the database
        meta_info = {
            'pattern_count': self.pattern_count,
            'config': self.config.__dict__
        }
        
        with open(path / "database_info.pkl", 'wb') as f:
            pickle.dump(meta_info, f)
        
        logger.info(f"Saved database to {path}")
    
    def load(self, path: Path) -> None:
        """Load database from disk.
        
        Args:
            path: Directory to load database from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Database directory not found: {path}")
        
        # Load database info
        info_path = path / "database_info.pkl"
        if info_path.exists():
            with open(info_path, 'rb') as f:
                meta_info = pickle.load(f)
                self.pattern_count = meta_info['pattern_count']
        
        # Load index
        index_path = path / "index.faiss"
        if index_path.exists():
            self.index = self.index_builder.load_index(index_path)
        
        logger.info(f"Loaded database from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'pattern_count': self.pattern_count,
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dim': self.config.embedding_dim
        }
        
        return stats