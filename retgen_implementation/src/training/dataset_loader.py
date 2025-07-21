"""Dataset loading utilities for RETGEN."""

from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("HuggingFace datasets not available. Using local data only.")

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and prepare datasets for RETGEN training."""
    
    @staticmethod
    def load_wikitext103() -> Tuple[List[str], List[str], List[str]]:
        """Load WikiText-103 dataset.
        
        Returns:
            Tuple of (train, validation, test) document lists
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("HuggingFace datasets required for WikiText-103")
        
        logger.info("Loading WikiText-103 dataset...")
        dataset = load_dataset("wikitext", "wikitext-103-v1")
        
        train_docs = [doc['text'] for doc in dataset['train'] if len(doc['text'].strip()) > 50]
        val_docs = [doc['text'] for doc in dataset['validation'] if len(doc['text'].strip()) > 50]
        test_docs = [doc['text'] for doc in dataset['test'] if len(doc['text'].strip()) > 50]
        
        logger.info(f"Loaded {len(train_docs)} train, {len(val_docs)} val, {len(test_docs)} test docs")
        
        return train_docs, val_docs, test_docs
    
    @staticmethod
    def load_openwebtext(limit: Optional[int] = None) -> List[str]:
        """Load OpenWebText dataset.
        
        Args:
            limit: Optional limit on number of documents
            
        Returns:
            List of documents
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("HuggingFace datasets required for OpenWebText")
        
        logger.info("Loading OpenWebText dataset...")
        dataset = load_dataset("openwebtext", streaming=True)
        
        docs = []
        for i, doc in enumerate(dataset['train']):
            if doc['text'] and len(doc['text'].strip()) > 100:
                docs.append(doc['text'])
            
            if limit and len(docs) >= limit:
                break
        
        logger.info(f"Loaded {len(docs)} documents from OpenWebText")
        return docs
    
    @staticmethod
    def load_local_texts(directory: Path) -> List[str]:
        """Load text files from local directory.
        
        Args:
            directory: Directory containing text files
            
        Returns:
            List of document contents
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        docs = []
        for file_path in directory.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if len(content) > 50:
                        docs.append(content)
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
        
        logger.info(f"Loaded {len(docs)} documents from {directory}")
        return docs
    
    @staticmethod
    def create_sample_dataset(num_docs: int = 100) -> List[str]:
        """Create sample dataset for testing.
        
        Args:
            num_docs: Number of documents to generate
            
        Returns:
            List of sample documents
        """
        import random
        
        # Sample sentences for generating documents
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Natural language processing is a fascinating field of study.",
            "Machine learning algorithms can learn complex patterns from data.",
            "Deep learning models have revolutionized artificial intelligence.",
            "Text generation requires understanding of language structure.",
            "Vector databases enable efficient similarity search.",
            "Retrieval-augmented generation improves language model performance.",
            "Context-aware embeddings capture semantic relationships.",
            "Pattern extraction identifies recurring linguistic structures.",
            "Transformer models use attention mechanisms for sequence processing."
        ]
        
        docs = []
        for i in range(num_docs):
            # Generate document with 5-15 sentences
            num_sentences = random.randint(5, 15)
            doc_sentences = random.choices(sentences, k=num_sentences)
            doc = " ".join(doc_sentences)
            docs.append(doc)
        
        logger.info(f"Generated {len(docs)} sample documents")
        return docs
    
    @staticmethod
    def split_dataset(
        docs: List[str],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split dataset into train/validation/test sets.
        
        Args:
            docs: List of documents
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            
        Returns:
            Tuple of (train, validation, test) document lists
        """
        import random
        
        # Shuffle documents
        shuffled_docs = docs.copy()
        random.shuffle(shuffled_docs)
        
        # Calculate split indices
        n_docs = len(shuffled_docs)
        n_train = int(n_docs * train_ratio)
        n_val = int(n_docs * val_ratio)
        
        # Split
        train_docs = shuffled_docs[:n_train]
        val_docs = shuffled_docs[n_train:n_train + n_val]
        test_docs = shuffled_docs[n_train + n_val:]
        
        logger.info(f"Split {n_docs} docs into {len(train_docs)} train, {len(val_docs)} val, {len(test_docs)} test")
        
        return train_docs, val_docs, test_docs
    
    @staticmethod
    def get_dataset_info(docs: List[str]) -> Dict[str, Any]:
        """Get statistics about a dataset.
        
        Args:
            docs: List of documents
            
        Returns:
            Dictionary with dataset statistics
        """
        if not docs:
            return {
                'num_docs': 0,
                'total_chars': 0,
                'avg_doc_length': 0,
                'min_doc_length': 0,
                'max_doc_length': 0
            }
        
        doc_lengths = [len(doc) for doc in docs]
        
        return {
            'num_docs': len(docs),
            'total_chars': sum(doc_lengths),
            'avg_doc_length': sum(doc_lengths) / len(doc_lengths),
            'min_doc_length': min(doc_lengths),
            'max_doc_length': max(doc_lengths)
        }