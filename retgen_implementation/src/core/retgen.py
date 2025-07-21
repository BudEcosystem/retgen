"""Core RETGEN model implementation."""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import RETGENConfig, TrainingMetrics
from training.trainer import RETGENTrainer
from training.dataset_loader import DatasetLoader
from inference.generator import RETGENGenerator
from evaluation.metrics import RETGENEvaluator


logger = logging.getLogger(__name__)


class RETGEN:
    """RETGEN: Retrieval-Enhanced Text Generation model."""
    
    def __init__(self, config: Optional[RETGENConfig] = None):
        """Initialize RETGEN model.
        
        Args:
            config: Model configuration. If None, uses default config.
        """
        self.config = config or RETGENConfig()
        self.config.validate()
        
        # Core components
        self.trainer: Optional[RETGENTrainer] = None
        self.generator: Optional[RETGENGenerator] = None
        
        # Training state
        self.is_trained = False
        
        # Set up logging
        logging.basicConfig(level=self.config.log_level)
        
        logger.info("Initialized RETGEN model")
    
    def train(
        self,
        corpus: List[str],
        validation_corpus: Optional[List[str]] = None,
        save_path: Optional[Path] = None
    ) -> TrainingMetrics:
        """Train RETGEN model on corpus.
        
        Args:
            corpus: List of training documents
            validation_corpus: Optional validation documents
            save_path: Optional path to save trained model
            
        Returns:
            Training metrics
        """
        logger.info(f"Training RETGEN with {len(corpus)} documents")
        
        # Initialize trainer
        self.trainer = RETGENTrainer(self.config)
        
        # Train model
        metrics = self.trainer.train(
            corpus=corpus,
            validation_corpus=validation_corpus,
            save_path=save_path
        )
        
        # Initialize generator
        self.generator = RETGENGenerator(
            embedder=self.trainer.embedder,
            database=self.trainer.database,
            config=self.config
        )
        
        self.is_trained = True
        logger.info("Training completed")
        
        return metrics
    
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self.is_trained or self.generator is None:
            raise RuntimeError("Model must be trained before generation")
        
        return self.generator.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters
            
        Returns:
            List of generated texts
        """
        if not self.is_trained or self.generator is None:
            raise RuntimeError("Model must be trained before generation")
        
        return self.generator.batch_generate(prompts, **kwargs)
    
    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text under model.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Perplexity score
        """
        if not self.is_trained or self.generator is None:
            raise RuntimeError("Model must be trained before evaluation")
        
        return self.generator.compute_perplexity(text)
    
    def save(self, path: Path) -> None:
        """Save model to disk.
        
        Args:
            path: Directory to save model
        """
        if not self.is_trained or self.trainer is None:
            raise RuntimeError("Cannot save untrained model")
        
        self.trainer.save_model(path)
        logger.info(f"Saved model to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'RETGEN':
        """Load model from disk.
        
        Args:
            path: Directory containing saved model
            
        Returns:
            Loaded RETGEN model
        """
        # Load trainer (which includes config and database)
        trainer = RETGENTrainer.load_model(path)
        
        # Create RETGEN instance
        model = cls(trainer.config)
        model.trainer = trainer
        
        # Initialize generator
        model.generator = RETGENGenerator(
            embedder=trainer.embedder,
            database=trainer.database,
            config=trainer.config
        )
        
        model.is_trained = True
        
        logger.info(f"Loaded model from {path}")
        return model
    
    def get_size_mb(self) -> float:
        """Get model size in megabytes.
        
        Returns:
            Model size in MB
        """
        if not self.is_trained or self.trainer is None:
            return 0.0
        
        # Estimate size based on database statistics
        stats = self.trainer.database.get_stats()
        
        # Rough estimation: 
        # - Embeddings: pattern_count * embedding_dim * 4 bytes (float32)
        # - Metadata: pattern_count * 100 bytes (estimated)
        # - Index overhead: ~20% of embeddings
        
        embedding_size_mb = (stats['pattern_count'] * stats['embedding_dim'] * 4) / (1024 * 1024)
        metadata_size_mb = (stats['pattern_count'] * 100) / (1024 * 1024)
        index_overhead_mb = embedding_size_mb * 0.2
        
        total_size_mb = embedding_size_mb + metadata_size_mb + index_overhead_mb
        
        return total_size_mb
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'is_trained': self.is_trained,
            'config': self.config.__dict__,
            'size_mb': self.get_size_mb()
        }
        
        if self.trainer:
            trainer_info = self.trainer.get_model_info()
            info.update(trainer_info)
        
        return info
    
    @classmethod  
    def quick_train(
        cls,
        dataset_name: str = "sample",
        num_docs: int = 100,
        config: Optional[RETGENConfig] = None
    ) -> 'RETGEN':
        """Quickly train a RETGEN model for testing.
        
        Args:
            dataset_name: Dataset to use ("sample", "wikitext103")
            num_docs: Number of documents (for sample dataset)
            config: Optional configuration
            
        Returns:
            Trained RETGEN model
        """
        config = config or RETGENConfig(
            min_pattern_frequency=1,
            retrieval_k=20,
            max_generation_length=50
        )
        
        # Load dataset
        if dataset_name == "sample":
            docs = DatasetLoader.create_sample_dataset(num_docs)
            train_docs, val_docs, _ = DatasetLoader.split_dataset(docs)
        elif dataset_name == "wikitext103":
            train_docs, val_docs, _ = DatasetLoader.load_wikitext103()
            train_docs = train_docs[:num_docs]  # Limit for speed
            val_docs = val_docs[:num_docs // 10]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Train model
        model = cls(config)
        model.train(train_docs, val_docs)
        
        return model