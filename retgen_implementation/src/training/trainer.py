"""RETGEN training pipeline."""

import time
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import RETGENConfig, TrainingMetrics
from data.pattern_extraction import PatternExtractor
from embeddings.context_embeddings import RETGENEmbedder
from indexing.vector_database import VectorDatabase


logger = logging.getLogger(__name__)


class RETGENTrainer:
    """Trainer for RETGEN model."""
    
    def __init__(self, config: RETGENConfig):
        """Initialize RETGEN trainer.
        
        Args:
            config: RETGEN configuration
        """
        self.config = config
        
        # Initialize components
        self.pattern_extractor = PatternExtractor(config)
        self.embedder = RETGENEmbedder(config)
        self.database = VectorDatabase(config)
        
        # Training state
        self.metrics = TrainingMetrics()
        self.is_trained = False
    
    def train(
        self,
        corpus: List[str],
        validation_corpus: Optional[List[str]] = None,
        save_path: Optional[Path] = None
    ) -> TrainingMetrics:
        """Train RETGEN model on corpus.
        
        Args:
            corpus: Training corpus (list of documents)
            validation_corpus: Optional validation corpus
            save_path: Optional path to save trained model
            
        Returns:
            Training metrics
        """
        logger.info(f"Starting RETGEN training with {len(corpus)} documents")
        start_time = time.time()
        
        # Step 1: Extract patterns from corpus
        logger.info("Step 1: Extracting patterns...")
        pattern_start = time.time()
        patterns = self.pattern_extractor.extract_from_corpus(
            corpus, show_progress=True
        )
        pattern_time = time.time() - pattern_start
        
        logger.info(f"Extracted {len(patterns)} patterns in {pattern_time:.2f}s")
        
        # Step 2: Compute embeddings
        logger.info("Step 2: Computing embeddings...")
        embed_start = time.time()
        embeddings = self.embedder.embed_patterns(
            patterns,
            batch_size=self.config.batch_size,
            show_progress=True
        )
        embed_time = time.time() - embed_start
        
        logger.info(f"Computed embeddings in {embed_time:.2f}s")
        
        # Step 3: Build vector database
        logger.info("Step 3: Building vector database...")
        db_start = time.time()
        self.database.add_patterns(patterns, embeddings)
        db_time = time.time() - db_start
        
        logger.info(f"Built database in {db_time:.2f}s")
        
        # Step 4: Compute validation metrics
        if validation_corpus:
            logger.info("Step 4: Computing validation metrics...")
            val_metrics = self._compute_validation_metrics(validation_corpus)
        else:
            val_metrics = {
                'coverage': 0.0,
                'retrieval_quality': 0.0,
                'perplexity': 0.0
            }
        
        total_time = time.time() - start_time
        
        # Update metrics
        self.metrics.add_metrics(
            coverage=val_metrics['coverage'],
            retrieval_quality=val_metrics['retrieval_quality'],
            perplexity=val_metrics['perplexity'],
            index_size=len(patterns),
            inference_speed=0.0,  # Will be measured during inference
            training_time=total_time
        )
        
        self.is_trained = True
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        logger.info(f"Training completed in {total_time:.2f}s")
        return self.metrics
    
    def _compute_validation_metrics(self, validation_corpus: List[str]) -> Dict[str, float]:
        """Compute validation metrics.
        
        Args:
            validation_corpus: Validation documents
            
        Returns:
            Dictionary with validation metrics
        """
        # Extract validation patterns (sample)
        val_patterns = []
        for doc in validation_corpus[:10]:  # Sample first 10 docs
            patterns = self.pattern_extractor.preprocessor.extract_training_patterns(doc)
            val_patterns.extend(patterns[:100])  # Sample 100 patterns per doc
        
        if not val_patterns:
            return {'coverage': 0.0, 'retrieval_quality': 0.0, 'perplexity': 50.0}
        
        # Compute embeddings for validation patterns
        val_embeddings = self.embedder.embed_patterns(
            val_patterns[:1000],  # Limit for speed
            show_progress=False
        )
        
        # Coverage: percentage of validation patterns with good matches
        coverage_count = 0
        total_similarity = 0.0
        
        for i, emb in enumerate(val_embeddings):
            results = self.database.search(emb.reshape(1, -1), k=1)
            if results[0]:  # Has results
                similarity = results[0][0]['distance']
                if similarity > 0.8:  # Good match threshold
                    coverage_count += 1
                total_similarity += similarity
        
        coverage = coverage_count / len(val_embeddings) if val_embeddings.shape[0] > 0 else 0.0
        retrieval_quality = total_similarity / len(val_embeddings) if val_embeddings.shape[0] > 0 else 0.0
        
        # Perplexity estimation (simplified)
        perplexity = max(10.0, 100.0 * (1.0 - coverage))
        
        return {
            'coverage': coverage,
            'retrieval_quality': retrieval_quality,
            'perplexity': perplexity
        }
    
    def save_model(self, path: Path) -> None:
        """Save trained model to disk.
        
        Args:
            path: Directory to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.config.save(path / "config.json")
        
        # Save database
        self.database.save(path / "database")
        
        # Save training metrics
        self.metrics.save(path / "training_metrics.json")
        
        logger.info(f"Saved model to {path}")
    
    @classmethod
    def load_model(cls, path: Path) -> 'RETGENTrainer':
        """Load trained model from disk.
        
        Args:
            path: Directory containing saved model
            
        Returns:
            Loaded RETGEN trainer
        """
        path = Path(path)
        
        # Load configuration
        config = RETGENConfig.load(path / "config.json")
        
        # Create trainer
        trainer = cls(config)
        
        # Load database
        trainer.database.load(path / "database")
        
        # Load metrics if available
        metrics_path = path / "training_metrics.json"
        if metrics_path.exists():
            trainer.metrics = TrainingMetrics.load(metrics_path)
        
        trainer.is_trained = True
        
        logger.info(f"Loaded model from {path}")
        return trainer
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        db_stats = self.database.get_stats()
        cache_stats = self.embedder.get_cache_stats()
        
        return {
            'is_trained': self.is_trained,
            'pattern_count': db_stats['pattern_count'],
            'embedding_dim': db_stats['embedding_dim'],
            'cache_stats': cache_stats,
            'config': self.config.__dict__
        }