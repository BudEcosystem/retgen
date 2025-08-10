#!/usr/bin/env python3
"""Test RETGEN v2 with WikiText data."""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# Import RETGEN v2
from retgen_v2_improved import RETGENv2, RETGENConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_training_data(limit: int = 10000):
    """Load training data from existing dataset."""
    data_path = Path("training_data.pkl")
    
    if data_path.exists():
        logger.info(f"Loading training data from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            return data[:limit]
    else:
        # Generate sample data if file doesn't exist
        logger.info("Generating sample training data...")
        from datasets import load_dataset
        
        # Load WikiText-103
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        
        # Process texts
        texts = []
        for item in tqdm(dataset, desc="Processing", total=limit):
            text = item['text'].strip()
            if len(text) > 50:  # Filter short texts
                texts.append(text)
            if len(texts) >= limit:
                break
        
        # Save for future use
        with open(data_path, 'wb') as f:
            pickle.dump(texts, f)
        
        return texts


def test_retgen_v2():
    """Test RETGEN v2 with comprehensive evaluation."""
    
    print("\n" + "="*70)
    print("RETGEN v2 COMPREHENSIVE TESTING")
    print("="*70)
    
    # Configuration
    config = RETGENConfig(
        nlist=100,
        nprobe=10,
        pq_nbits=8,
        pq_nsplits=48,
        use_gpu=True
    )
    
    # Initialize model
    print("\n1. Initializing RETGEN v2...")
    model = RETGENv2(config)
    
    # Load training data
    print("\n2. Loading training data...")
    texts = load_training_data(limit=5000)
    print(f"   Loaded {len(texts)} texts")
    
    # Train model
    print("\n3. Training model...")
    model.train_on_corpus(texts[:1000], batch_size=64)  # Use subset for testing
    
    # Test generation
    print("\n4. Testing generation capabilities...")
    test_prompts = [
        "The future of artificial intelligence",
        "Machine learning algorithms",
        "Natural language processing",
        "Deep learning models",
        "Computer vision technology"
    ]
    
    print("\n" + "-"*60)
    print("GENERATION RESULTS")
    print("-"*60)
    
    for prompt in test_prompts:
        generated = model.generate(prompt, max_length=30)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated}'")
    
    # Test retrieval with adaptive policy
    print("\n5. Testing adaptive retrieval policy...")
    print("\n" + "-"*60)
    print("ADAPTIVE RETRIEVAL RESULTS")
    print("-"*60)
    
    test_queries = [
        "artificial intelligence",
        "learning algorithms",
        "neural networks"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Test with adaptive policy
        results_adaptive = model.retrieve_patterns(query, adaptive=True)
        print(f"Adaptive retrieval ({len(results_adaptive)} results):")
        for r in results_adaptive[:3]:
            print(f"  - Pattern: '{r['pattern'][:30]}...' -> '{r['continuation']}'")
        
        # Test without adaptive policy
        results_fixed = model.retrieve_patterns(query, adaptive=False)
        print(f"Fixed retrieval ({len(results_fixed)} results):")
        for r in results_fixed[:3]:
            print(f"  - Pattern: '{r['pattern'][:30]}...' -> '{r['continuation']}'")
    
    # Memory analysis
    print("\n6. Memory usage analysis...")
    memory_usage = model.index.get_memory_usage()
    
    print("\n" + "-"*60)
    print("MEMORY USAGE ANALYSIS")
    print("-"*60)
    
    if model.patterns:
        original_size = len(model.patterns) * 384 * 4 / (1024**3)
        compressed_size = memory_usage['total_gb']
        
        print(f"Patterns indexed: {len(model.patterns):,}")
        print(f"Original size (Flat L2): {original_size:.3f} GB")
        print(f"Compressed size (IVF+PQ): {compressed_size:.3f} GB")
        
        if compressed_size > 0:
            compression_ratio = original_size / compressed_size
            print(f"Compression ratio: {compression_ratio:.1f}x")
            print(f"Memory saved: {(original_size - compressed_size):.3f} GB ({(1 - compressed_size/original_size)*100:.1f}%)")
    else:
        print("No patterns indexed")
    
    # Test policy learning
    print("\n7. Testing policy learning with feedback...")
    print("\n" + "-"*60)
    print("POLICY LEARNING")
    print("-"*60)
    
    # Simulate user feedback
    feedback_data = [
        ("good generation example", 1.0),  # Positive feedback
        ("bad generation example", -1.0),  # Negative feedback
        ("neutral example", 0.0)  # Neutral
    ]
    
    for query, feedback in feedback_data:
        loss = model.update_policy(query, feedback)
        print(f"Query: '{query}' | Feedback: {feedback:+.1f} | Loss: {loss:.4f}")
    
    # Save model
    print("\n8. Saving model...")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model.save(str(model_dir / "retgen_v2_test"))
    print(f"Model saved to {model_dir / 'retgen_v2_test'}")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    
    return model


if __name__ == "__main__":
    model = test_retgen_v2()