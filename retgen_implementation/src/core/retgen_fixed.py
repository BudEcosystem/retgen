"""Fixed RETGEN model implementation with all components."""

import os
import pickle
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

try:
    import torch
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    from tqdm import tqdm
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """A text pattern with its continuation."""
    text: str
    continuation: str
    frequency: int = 1
    embedding: Optional[np.ndarray] = None


class PatternExtractor:
    """Extract patterns from text at multiple resolutions."""
    
    def __init__(self, resolutions: List[int] = [1, 2, 3, 5, 8], min_frequency: int = 2):
        self.resolutions = resolutions
        self.min_frequency = min_frequency
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    def extract_patterns(self, text: str) -> List[Pattern]:
        """Extract patterns from text at multiple resolutions."""
        patterns = []
        
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) < 2:
            return patterns
        
        # Extract patterns at each resolution
        for resolution in self.resolutions:
            for i in range(len(tokens) - resolution):
                pattern_text = " ".join(tokens[i:i+resolution])
                continuation = tokens[i+resolution] if i+resolution < len(tokens) else ""
                
                patterns.append(Pattern(
                    text=pattern_text,
                    continuation=continuation,
                    frequency=1
                ))
        
        return patterns


class VectorDatabase:
    """Vector database for pattern storage and retrieval."""
    
    def __init__(self, dimension: int = 768, use_gpu: bool = False):
        self.dimension = dimension
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Using GPU-accelerated FAISS index")
            except:
                logger.warning("GPU FAISS not available, using CPU")
        
        self.patterns = []
        self.pattern_map = {}
        self.index_size = 0
        
    def add_patterns(self, patterns: List[Pattern], embeddings: np.ndarray):
        """Add patterns with their embeddings to the database."""
        if len(patterns) != embeddings.shape[0]:
            raise ValueError("Number of patterns must match number of embeddings")
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store patterns
        for i, pattern in enumerate(patterns):
            pattern.embedding = embeddings[i]
            self.patterns.append(pattern)
            self.pattern_map[self.index_size + i] = pattern
        
        self.index_size += len(patterns)
        
    def search(self, query_embedding: np.ndarray, k: int = 50) -> List[Tuple[Pattern, float]]:
        """Search for k nearest patterns."""
        if self.index_size == 0:
            return []
        
        k = min(k, self.index_size)
        
        # Search index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32), 
            k
        )
        
        # Retrieve patterns with scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < self.index_size:
                pattern = self.pattern_map[idx]
                score = 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity
                results.append((pattern, score))
        
        return results


class RETGENFixed:
    """Fixed RETGEN model with all components."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize RETGEN model."""
        self.config = config or {}
        
        # Set defaults
        self.embedding_dim = self.config.get('embedding_dim', 768)
        self.resolutions = self.config.get('resolutions', [1, 2, 3, 5, 8])
        self.retrieval_k = self.config.get('retrieval_k', 50)
        self.temperature = self.config.get('temperature', 1.0)
        self.min_pattern_frequency = self.config.get('min_pattern_frequency', 2)
        self.use_gpu = self.config.get('use_gpu', torch.cuda.is_available())
        
        # Initialize components
        logger.info("Initializing RETGEN components...")
        
        # Pattern extractor
        self.pattern_extractor = PatternExtractor(
            resolutions=self.resolutions,
            min_frequency=self.min_pattern_frequency
        )
        
        # Embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        if self.use_gpu:
            self.embedder = self.embedder.cuda()
        
        # Vector database
        self.vector_db = VectorDatabase(
            dimension=384,  # all-MiniLM-L6-v2 dimension
            use_gpu=self.use_gpu
        )
        
        # Pattern statistics
        self.pattern_stats = defaultdict(int)
        self.total_patterns = 0
        self.is_trained = False
        
        logger.info("RETGEN model initialized successfully")
    
    def add_pattern(self, pattern: Pattern):
        """Add a single pattern to the model."""
        # Update frequency
        pattern_key = pattern.text
        self.pattern_stats[pattern_key] += 1
        
        # Compute embedding
        embedding = self.embedder.encode([pattern.text])
        
        # Add to database
        self.vector_db.add_patterns([pattern], embedding)
        self.total_patterns += 1
    
    def train_on_texts(self, texts: List[str], batch_size: int = 32):
        """Train RETGEN on a list of texts."""
        logger.info(f"Training on {len(texts)} texts...")
        
        all_patterns = []
        pattern_texts = []
        
        # Extract patterns from all texts
        for text_idx, text in enumerate(tqdm(texts, desc="Extracting patterns")):
            try:
                patterns = self.pattern_extractor.extract_patterns(text)
                for pattern in patterns:
                    all_patterns.append(pattern)
                    pattern_texts.append(pattern.text)
                    self.pattern_stats[pattern.text] += 1
            except Exception as e:
                logger.warning(f"Error processing text {text_idx}: {e}")
                continue
        
        # Filter by frequency
        filtered_patterns = []
        filtered_texts = []
        for pattern in all_patterns:
            if self.pattern_stats[pattern.text] >= self.min_pattern_frequency:
                filtered_patterns.append(pattern)
                filtered_texts.append(pattern.text)
        
        logger.info(f"Filtered to {len(filtered_patterns)} patterns (min frequency: {self.min_pattern_frequency})")
        
        if not filtered_patterns:
            logger.warning("No patterns to add after filtering")
            return
        
        # Compute embeddings in batches
        logger.info("Computing embeddings...")
        embeddings = []
        for i in tqdm(range(0, len(filtered_texts), batch_size), desc="Embedding batches"):
            batch_texts = filtered_texts[i:i+batch_size]
            batch_embeddings = self.embedder.encode(batch_texts, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        if embeddings:
            all_embeddings = np.vstack(embeddings)
            
            # Add to database
            logger.info("Adding patterns to vector database...")
            self.vector_db.add_patterns(filtered_patterns, all_embeddings)
            self.total_patterns = len(filtered_patterns)
            self.is_trained = True
            
            logger.info(f"Training complete. Total patterns: {self.total_patterns}")
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text from prompt using retrieval."""
        if not self.is_trained:
            return prompt + " [Model not trained]"
        
        generated = prompt
        
        for _ in range(max_length):
            # Encode current context
            context_embedding = self.embedder.encode([generated[-200:]])  # Use last 200 chars as context
            
            # Retrieve similar patterns
            results = self.vector_db.search(context_embedding, k=self.retrieval_k)
            
            if not results:
                break
            
            # Aggregate continuations
            continuation_scores = defaultdict(float)
            for pattern, score in results:
                if pattern.continuation:
                    continuation_scores[pattern.continuation] += score
            
            if not continuation_scores:
                break
            
            # Sample from continuations
            continuations = list(continuation_scores.keys())
            scores = np.array(list(continuation_scores.values()))
            
            # Apply temperature
            scores = scores / self.temperature
            probs = np.exp(scores) / np.sum(np.exp(scores))
            
            # Sample next token
            next_token = np.random.choice(continuations, p=probs)
            generated += " " + next_token
            
            # Stop if we hit a period
            if next_token in ['.', '!', '?']:
                break
        
        return generated
    
    def save(self, path: str):
        """Save model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        save_data = {
            'config': self.config,
            'pattern_stats': dict(self.pattern_stats),
            'total_patterns': self.total_patterns,
            'patterns': self.vector_db.patterns,
            'index_size': self.vector_db.index_size
        }
        
        # Save FAISS index separately
        index_path = path.replace('.pkl', '_index.faiss')
        faiss.write_index(self.vector_db.index, index_path)
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.config = save_data['config']
        self.pattern_stats = defaultdict(int, save_data['pattern_stats'])
        self.total_patterns = save_data['total_patterns']
        
        # Load FAISS index
        index_path = path.replace('.pkl', '_index.faiss')
        self.vector_db.index = faiss.read_index(index_path)
        self.vector_db.patterns = save_data['patterns']
        self.vector_db.index_size = save_data['index_size']
        
        # Rebuild pattern map
        self.vector_db.pattern_map = {
            i: pattern for i, pattern in enumerate(self.vector_db.patterns)
        }
        
        self.is_trained = True
        logger.info(f"Model loaded from {path}")