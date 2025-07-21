"""Pattern extraction and preprocessing for RETGEN."""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import logging
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import RETGENConfig


logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """A text pattern with its continuation."""
    
    tokens: List[int]
    text: str
    next_token: int
    next_text: str
    position: int
    resolution: int
    document_id: int
    
    def get_key(self) -> Tuple[int, ...]:
        """Get hashable key for this pattern."""
        return tuple(self.tokens)
    
    def __hash__(self) -> int:
        """Hash based on token sequence."""
        return hash(self.get_key())


class RETGENPreprocessor:
    """Preprocessor for RETGEN text data."""
    
    def __init__(self, config: RETGENConfig):
        """Initialize preprocessor.
        
        Args:
            config: RETGEN configuration
        """
        self.config = config
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.embedding_model,
                use_fast=True
            )
        except:
            # Fallback to BERT tokenizer if embedding model doesn't have one
            logger.warning(f"Could not load tokenizer for {config.embedding_model}, using bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Strip whitespace
        text = text.strip()
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Handle special tokens
        text = text.replace('<|endoftext|>', ' ')
        
        return text
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(
            text,
            max_length=self.config.max_sequence_length,
            truncation=True,
            add_special_tokens=True
        )
    
    def extract_training_patterns(
        self,
        text: str,
        doc_id: int = 0
    ) -> List[Pattern]:
        """Extract all patterns from text.
        
        Args:
            text: Input text
            doc_id: Document ID
            
        Returns:
            List of extracted patterns
        """
        # Preprocess and tokenize
        text = self.preprocess_text(text)
        if not text:
            return []
        
        tokens = self.tokenize(text)
        if len(tokens) < 2:  # Need at least 2 tokens for pattern + continuation
            return []
        
        patterns = []
        
        # Extract patterns at each resolution
        for resolution in self.config.resolutions:
            if resolution > len(tokens) - 1:
                continue
            
            # Slide window across tokens
            for i in range(0, len(tokens) - resolution, self.config.pattern_stride):
                if i + resolution >= len(tokens):
                    break
                
                # Extract pattern tokens and next token
                pattern_tokens = tokens[i:i + resolution]
                next_token = tokens[i + resolution]
                
                # Decode to text (for debugging and analysis)
                pattern_text = self.tokenizer.decode(pattern_tokens, skip_special_tokens=True)
                next_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
                
                # Create pattern object
                pattern = Pattern(
                    tokens=pattern_tokens,
                    text=pattern_text,
                    next_token=next_token,
                    next_text=next_text,
                    position=i,
                    resolution=resolution,
                    document_id=doc_id
                )
                
                patterns.append(pattern)
        
        return patterns


class PatternDatabase:
    """Database for storing and querying patterns."""
    
    def __init__(self):
        """Initialize empty pattern database."""
        self.patterns: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        self.total_patterns = 0
    
    def add_pattern(self, pattern: Pattern) -> None:
        """Add a pattern to the database.
        
        Args:
            pattern: Pattern to add
        """
        key = pattern.get_key()
        
        if key not in self.patterns:
            self.patterns[key] = {
                'count': 0,
                'continuations': defaultdict(int),
                'positions': [],
                'resolution': pattern.resolution,
                'text': pattern.text
            }
        
        # Update pattern info
        self.patterns[key]['count'] += 1
        self.patterns[key]['continuations'][pattern.next_token] += 1
        self.patterns[key]['positions'].append({
            'doc_id': pattern.document_id,
            'position': pattern.position
        })
        
        self.total_patterns += 1
    
    def get_pattern_info(self, key: Tuple[int, ...]) -> Optional[Dict[str, Any]]:
        """Get information about a pattern.
        
        Args:
            key: Pattern key (tuple of token IDs)
            
        Returns:
            Pattern information or None if not found
        """
        return self.patterns.get(key)
    
    def get_continuation_distribution(
        self,
        key: Tuple[int, ...]
    ) -> Dict[int, float]:
        """Get probability distribution over continuations.
        
        Args:
            key: Pattern key
            
        Returns:
            Dictionary mapping token IDs to probabilities
        """
        info = self.get_pattern_info(key)
        if info is None:
            return {}
        
        total_count = sum(info['continuations'].values())
        if total_count == 0:
            return {}
        
        return {
            token: count / total_count
            for token, count in info['continuations'].items()
        }
    
    def filter_by_frequency(self, min_frequency: int) -> None:
        """Remove patterns below minimum frequency.
        
        Args:
            min_frequency: Minimum pattern frequency to keep
        """
        filtered_patterns = {
            key: info
            for key, info in self.patterns.items()
            if info['count'] >= min_frequency
        }
        
        removed_count = len(self.patterns) - len(filtered_patterns)
        if removed_count > 0:
            logger.info(f"Filtered out {removed_count} patterns below frequency {min_frequency}")
        
        self.patterns = filtered_patterns
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.patterns:
            return {
                'total_patterns': 0,
                'unique_patterns': 0,
                'avg_continuations': 0,
                'resolution_distribution': {}
            }
        
        resolution_counts = defaultdict(int)
        total_continuations = 0
        
        for info in self.patterns.values():
            resolution_counts[info['resolution']] += 1
            total_continuations += len(info['continuations'])
        
        return {
            'total_patterns': self.total_patterns,
            'unique_patterns': len(self.patterns),
            'avg_continuations': total_continuations / len(self.patterns),
            'resolution_distribution': dict(resolution_counts)
        }


class PatternExtractor:
    """Extract patterns from a corpus."""
    
    def __init__(self, config: RETGENConfig):
        """Initialize pattern extractor.
        
        Args:
            config: RETGEN configuration
        """
        self.config = config
        self.preprocessor = RETGENPreprocessor(config)
    
    def extract_from_corpus(
        self,
        corpus: List[str],
        show_progress: bool = True
    ) -> List[Pattern]:
        """Extract patterns from a corpus of documents.
        
        Args:
            corpus: List of documents
            show_progress: Whether to show progress bar
            
        Returns:
            List of all extracted patterns
        """
        all_patterns = []
        
        # Create pattern database for frequency filtering
        database = PatternDatabase()
        
        # Extract patterns from each document
        iterator = tqdm(corpus, desc="Extracting patterns") if show_progress else corpus
        
        for doc_id, document in enumerate(iterator):
            patterns = self.preprocessor.extract_training_patterns(document, doc_id)
            
            # Add to database for frequency counting
            for pattern in patterns:
                database.add_pattern(pattern)
            
            all_patterns.extend(patterns)
        
        # Log statistics
        stats = database.get_statistics()
        logger.info(f"Extracted {stats['total_patterns']} total patterns")
        logger.info(f"Found {stats['unique_patterns']} unique patterns")
        logger.info(f"Resolution distribution: {stats['resolution_distribution']}")
        
        # Filter by frequency if specified
        if self.config.min_pattern_frequency > 1:
            database.filter_by_frequency(self.config.min_pattern_frequency)
            
            # Keep only patterns that remain in database
            filtered_patterns = []
            for pattern in all_patterns:
                if pattern.get_key() in database.patterns:
                    filtered_patterns.append(pattern)
            
            logger.info(f"Kept {len(filtered_patterns)} patterns after frequency filtering")
            return filtered_patterns
        
        return all_patterns
    
    def create_pattern_database(self, corpus: List[str]) -> PatternDatabase:
        """Create pattern database from corpus.
        
        Args:
            corpus: List of documents
            
        Returns:
            Populated pattern database
        """
        database = PatternDatabase()
        
        for doc_id, document in enumerate(tqdm(corpus, desc="Building pattern database")):
            patterns = self.preprocessor.extract_training_patterns(document, doc_id)
            
            for pattern in patterns:
                database.add_pattern(pattern)
        
        # Filter by frequency
        if self.config.min_pattern_frequency > 1:
            database.filter_by_frequency(self.config.min_pattern_frequency)
        
        return database