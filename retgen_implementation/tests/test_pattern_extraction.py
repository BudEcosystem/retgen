"""Tests for pattern extraction and preprocessing."""

import pytest
import numpy as np
from typing import List
from src.data.pattern_extraction import (
    PatternExtractor,
    Pattern,
    PatternDatabase,
    RETGENPreprocessor
)
from src.core.config import RETGENConfig


class TestPattern:
    """Test suite for Pattern class."""
    
    def test_pattern_creation(self):
        """Test creating a pattern."""
        pattern = Pattern(
            tokens=[101, 2023, 2003],
            text="this is",
            next_token=1037,
            next_text="a",
            position=5,
            resolution=2,
            document_id=0
        )
        
        assert pattern.tokens == [101, 2023, 2003]
        assert pattern.text == "this is"
        assert pattern.next_token == 1037
        assert pattern.next_text == "a"
        assert pattern.position == 5
        assert pattern.resolution == 2
        assert pattern.document_id == 0
    
    def test_pattern_key(self):
        """Test pattern key generation."""
        pattern = Pattern(
            tokens=[101, 2023, 2003],
            text="this is",
            next_token=1037,
            next_text="a",
            position=5,
            resolution=2,
            document_id=0
        )
        
        key = pattern.get_key()
        assert key == (101, 2023, 2003)
    
    def test_pattern_equality(self):
        """Test pattern equality."""
        pattern1 = Pattern(
            tokens=[101, 2023],
            text="this",
            next_token=2003,
            next_text="is",
            position=0,
            resolution=1,
            document_id=0
        )
        
        pattern2 = Pattern(
            tokens=[101, 2023],
            text="this",
            next_token=2003,
            next_text="is",
            position=10,  # Different position
            resolution=1,
            document_id=1  # Different document
        )
        
        # Patterns with same tokens should have same key
        assert pattern1.get_key() == pattern2.get_key()


class TestRETGENPreprocessor:
    """Test suite for RETGENPreprocessor."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        config = RETGENConfig()
        preprocessor = RETGENPreprocessor(config)
        
        assert preprocessor.config == config
        assert preprocessor.tokenizer is not None
    
    def test_text_preprocessing(self):
        """Test text preprocessing."""
        config = RETGENConfig()
        preprocessor = RETGENPreprocessor(config)
        
        # Test whitespace normalization
        text = "  This   is   a   test.  "
        processed = preprocessor.preprocess_text(text)
        assert processed == "This is a test."
        
        # Test special token handling
        text = "Hello <|endoftext|> World"
        processed = preprocessor.preprocess_text(text)
        assert processed == "Hello   World"
        
        # Test empty text
        assert preprocessor.preprocess_text("") == ""
        assert preprocessor.preprocess_text("   ") == ""
    
    def test_tokenization(self):
        """Test tokenization."""
        config = RETGENConfig(embedding_model="bert-base-uncased")
        preprocessor = RETGENPreprocessor(config)
        
        text = "Hello world!"
        tokens = preprocessor.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)
    
    def test_pattern_extraction_single_resolution(self):
        """Test pattern extraction at single resolution."""
        config = RETGENConfig(
            resolutions=[2],
            min_pattern_length=2,
            max_pattern_length=2
        )
        preprocessor = RETGENPreprocessor(config)
        
        text = "The quick brown fox"
        patterns = preprocessor.extract_training_patterns(text, doc_id=0)
        
        # Should extract patterns of length 2
        assert len(patterns) > 0
        assert all(p.resolution == 2 for p in patterns)
        assert all(len(p.tokens) == 2 for p in patterns)
    
    def test_pattern_extraction_multi_resolution(self):
        """Test pattern extraction at multiple resolutions."""
        config = RETGENConfig(
            resolutions=[1, 2, 3],
            min_pattern_length=1,
            max_pattern_length=3
        )
        preprocessor = RETGENPreprocessor(config)
        
        text = "The quick brown fox jumps"
        patterns = preprocessor.extract_training_patterns(text, doc_id=0)
        
        # Should have patterns of different resolutions
        resolutions = set(p.resolution for p in patterns)
        assert resolutions == {1, 2, 3}
        
        # Check pattern lengths match resolutions
        for pattern in patterns:
            assert len(pattern.tokens) == pattern.resolution


class TestPatternExtractor:
    """Test suite for PatternExtractor."""
    
    def test_extractor_initialization(self):
        """Test pattern extractor initialization."""
        config = RETGENConfig()
        extractor = PatternExtractor(config)
        
        assert extractor.config == config
        assert extractor.preprocessor is not None
    
    def test_extract_from_corpus(self):
        """Test extracting patterns from corpus."""
        config = RETGENConfig(
            resolutions=[1, 2],
            min_pattern_frequency=1
        )
        extractor = PatternExtractor(config)
        
        corpus = [
            "The cat sat on the mat.",
            "The dog sat on the floor.",
            "The cat and dog played."
        ]
        
        patterns = extractor.extract_from_corpus(corpus)
        
        # Should have extracted patterns
        assert len(patterns) > 0
        
        # All patterns should have required attributes
        for pattern in patterns:
            assert hasattr(pattern, 'tokens')
            assert hasattr(pattern, 'text')
            assert hasattr(pattern, 'next_token')
            assert hasattr(pattern, 'resolution')
    
    def test_pattern_filtering_by_frequency(self):
        """Test filtering patterns by frequency."""
        config = RETGENConfig(
            resolutions=[1],
            min_pattern_frequency=2
        )
        extractor = PatternExtractor(config)
        
        corpus = [
            "The cat sat.",
            "The dog sat.",
            "A bird flew."
        ]
        
        patterns = extractor.extract_from_corpus(corpus)
        
        # Count pattern occurrences
        pattern_counts = {}
        for p in patterns:
            key = p.get_key()
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
        
        # Only patterns with frequency >= 2 should remain
        # "The" and "sat" should appear at least twice
        high_freq_patterns = [p for p in patterns if pattern_counts[p.get_key()] >= 2]
        assert len(high_freq_patterns) > 0


class TestPatternDatabase:
    """Test suite for PatternDatabase."""
    
    def test_database_initialization(self):
        """Test pattern database initialization."""
        db = PatternDatabase()
        
        assert db.patterns == {}
        assert db.total_patterns == 0
    
    def test_add_pattern(self):
        """Test adding patterns to database."""
        db = PatternDatabase()
        
        pattern = Pattern(
            tokens=[101, 2023],
            text="this",
            next_token=2003,
            next_text="is",
            position=0,
            resolution=1,
            document_id=0
        )
        
        db.add_pattern(pattern)
        
        assert db.total_patterns == 1
        assert pattern.get_key() in db.patterns
        assert db.patterns[pattern.get_key()]['count'] == 1
        assert db.patterns[pattern.get_key()]['continuations'][2003] == 1
    
    def test_add_duplicate_patterns(self):
        """Test adding duplicate patterns."""
        db = PatternDatabase()
        
        pattern1 = Pattern(
            tokens=[101, 2023],
            text="this",
            next_token=2003,
            next_text="is",
            position=0,
            resolution=1,
            document_id=0
        )
        
        pattern2 = Pattern(
            tokens=[101, 2023],
            text="this",
            next_token=2001,  # Different continuation
            next_text="was",
            position=10,
            resolution=1,
            document_id=1
        )
        
        db.add_pattern(pattern1)
        db.add_pattern(pattern2)
        
        assert db.total_patterns == 2
        assert db.patterns[pattern1.get_key()]['count'] == 2
        assert db.patterns[pattern1.get_key()]['continuations'][2003] == 1
        assert db.patterns[pattern1.get_key()]['continuations'][2001] == 1
    
    def test_get_pattern_info(self):
        """Test retrieving pattern information."""
        db = PatternDatabase()
        
        pattern = Pattern(
            tokens=[101, 2023],
            text="this",
            next_token=2003,
            next_text="is",
            position=0,
            resolution=1,
            document_id=0
        )
        
        db.add_pattern(pattern)
        
        info = db.get_pattern_info(pattern.get_key())
        assert info is not None
        assert info['count'] == 1
        assert info['continuations'][2003] == 1
        
        # Non-existent pattern
        assert db.get_pattern_info((999, 999)) is None
    
    def test_get_continuation_distribution(self):
        """Test getting continuation probability distribution."""
        db = PatternDatabase()
        
        # Add patterns with different continuations
        for i in range(3):
            pattern = Pattern(
                tokens=[101, 2023],
                text="this",
                next_token=2003,  # "is"
                next_text="is",
                position=i,
                resolution=1,
                document_id=0
            )
            db.add_pattern(pattern)
        
        pattern = Pattern(
            tokens=[101, 2023],
            text="this",
            next_token=2001,  # "was"
            next_text="was",
            position=3,
            resolution=1,
            document_id=0
        )
        db.add_pattern(pattern)
        
        # Get distribution
        dist = db.get_continuation_distribution((101, 2023))
        
        assert len(dist) == 2
        assert dist[2003] == 0.75  # 3/4
        assert dist[2001] == 0.25  # 1/4
        assert abs(sum(dist.values()) - 1.0) < 1e-6
    
    def test_filter_by_frequency(self):
        """Test filtering database by frequency."""
        db = PatternDatabase()
        
        # Add patterns with different frequencies
        patterns = [
            Pattern([1, 2], "a b", 3, "c", 0, 1, 0),
            Pattern([1, 2], "a b", 4, "d", 1, 1, 0),
            Pattern([2, 3], "b c", 4, "d", 2, 1, 0),
        ]
        
        for p in patterns:
            db.add_pattern(p)
        
        # Filter by frequency >= 2
        db.filter_by_frequency(min_frequency=2)
        
        assert len(db.patterns) == 1
        assert (1, 2) in db.patterns
        assert (2, 3) not in db.patterns