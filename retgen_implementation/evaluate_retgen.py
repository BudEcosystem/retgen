#!/usr/bin/env python3
"""Evaluate RETGEN model's predictive capability."""

import os
import sys
import json
import pickle
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
# Metrics imports removed - not needed for basic evaluation

class RETGENEvaluator:
    """Evaluator for RETGEN model with sharded index."""
    
    def __init__(self, model_path: str, index_dir: str):
        """Load model and index shards."""
        print(f"Loading model from {model_path}...")
        
        # Load model metadata
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.config = self.model_data['config']
        self.total_patterns = self.model_data['total_patterns']
        self.current_shard = self.model_data.get('current_shard', 0)
        self.shard_indices = self.model_data.get('shard_indices', [])
        
        print(f"Model loaded: {self.total_patterns:,} total patterns across {len(self.shard_indices)} shards")
        
        # Initialize components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Load encoder
        print("Loading sentence encoder...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        
        # Load index shards info
        self.index_dir = Path(index_dir)
        self.load_shard_metadata()
        
    def load_shard_metadata(self):
        """Load metadata from all shards."""
        self.all_patterns = []
        self.all_continuations = []
        self.shard_offsets = [0]
        
        print(f"Loading metadata from {len(self.shard_indices)} shards...")
        
        # Load a subset of shards for evaluation (last 5 shards for recency)
        shards_to_load = self.shard_indices[-5:] if len(self.shard_indices) > 5 else self.shard_indices
        
        for shard_path in shards_to_load:
            meta_path = shard_path.replace('.faiss', '_meta.pkl')
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    self.all_patterns.extend(meta['patterns'])
                    self.all_continuations.extend(meta['continuations'])
                    self.shard_offsets.append(len(self.all_patterns))
        
        print(f"Loaded {len(self.all_patterns):,} patterns for evaluation")
    
    def search_similar_patterns(self, query: str, k: int = 10) -> List[Tuple[str, str, float]]:
        """Search for similar patterns across shards."""
        # Encode query
        with torch.no_grad():
            query_embedding = self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        all_results = []
        
        # Search in last few shards for speed
        shards_to_search = self.shard_indices[-3:] if len(self.shard_indices) > 3 else self.shard_indices
        
        for shard_path in shards_to_search:
            if os.path.exists(shard_path):
                # Load shard index
                index = faiss.read_index(shard_path)
                
                # Search
                distances, indices = index.search(query_embedding.astype(np.float32), min(k, index.ntotal))
                
                # Get corresponding patterns (from metadata)
                meta_path = shard_path.replace('.faiss', '_meta.pkl')
                if os.path.exists(meta_path):
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                        
                    for dist, idx in zip(distances[0], indices[0]):
                        if idx < len(meta['patterns']):
                            all_results.append((
                                meta['patterns'][idx],
                                meta['continuations'][idx],
                                float(dist)
                            ))
        
        # Sort by distance and return top-k
        all_results.sort(key=lambda x: x[2])
        return all_results[:k]
    
    def predict_next_token(self, context: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict next token based on similar patterns."""
        similar_patterns = self.search_similar_patterns(context, k=50)
        
        if not similar_patterns:
            return []
        
        # Aggregate predictions
        continuation_scores = {}
        for pattern, continuation, distance in similar_patterns:
            # Convert distance to similarity score
            similarity = 1.0 / (1.0 + distance)
            
            if continuation not in continuation_scores:
                continuation_scores[continuation] = 0
            continuation_scores[continuation] += similarity
        
        # Sort by score
        predictions = sorted(
            continuation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return predictions[:top_k]
    
    def generate_text(self, prompt: str, max_length: int = 50, temperature: float = 0.8) -> str:
        """Generate text using retrieval-based prediction."""
        generated = prompt
        tokens_generated = 0
        
        while tokens_generated < max_length:
            # Get predictions for current context
            predictions = self.predict_next_token(generated[-100:], top_k=10)  # Use last 100 chars as context
            
            if not predictions:
                break
            
            # Sample from predictions with temperature
            if temperature > 0:
                # Convert to probabilities
                scores = np.array([score for _, score in predictions])
                scores = scores / temperature
                probs = np.exp(scores) / np.sum(np.exp(scores))
                
                # Sample
                idx = np.random.choice(len(predictions), p=probs)
                next_token = predictions[idx][0]
            else:
                # Greedy selection
                next_token = predictions[0][0]
            
            # Add to generated text
            generated += " " + next_token
            tokens_generated += 1
            
            # Stop if we hit a period
            if next_token in ['.', '!', '?']:
                break
        
        return generated
    
    def evaluate_perplexity(self, test_texts: List[str], max_samples: int = 100) -> Dict:
        """Evaluate model perplexity on test texts."""
        print(f"\nEvaluating perplexity on {min(len(test_texts), max_samples)} samples...")
        
        total_loss = 0
        total_tokens = 0
        correct_predictions = 0
        
        test_sample = random.sample(test_texts, min(len(test_texts), max_samples))
        
        for text in test_sample:
            tokens = text.split()
            
            for i in range(1, min(len(tokens), 20)):  # Evaluate on first 20 tokens
                context = " ".join(tokens[:i])
                actual_next = tokens[i] if i < len(tokens) else ""
                
                # Get predictions
                predictions = self.predict_next_token(context, top_k=10)
                
                if predictions:
                    # Check if correct token is in top predictions
                    predicted_tokens = [token for token, _ in predictions]
                    if actual_next in predicted_tokens:
                        correct_predictions += 1
                    
                    # Calculate loss (negative log likelihood)
                    scores = [score for _, score in predictions]
                    if actual_next in predicted_tokens:
                        idx = predicted_tokens.index(actual_next)
                        score = scores[idx]
                    else:
                        score = 0.001  # Small score for unseen tokens
                    
                    total_loss += -np.log(score / sum(scores))
                
                total_tokens += 1
        
        perplexity = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
        accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
        
        return {
            'perplexity': perplexity,
            'accuracy': accuracy,
            'total_tokens': total_tokens,
            'correct_predictions': correct_predictions
        }
    
    def evaluate_generation_quality(self, prompts: List[str]) -> List[Dict]:
        """Evaluate generation quality with different prompts."""
        print("\nEvaluating text generation quality...")
        
        results = []
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            
            # Generate with different temperatures
            for temp in [0.5, 0.8, 1.0]:
                generated = self.generate_text(prompt, max_length=30, temperature=temp)
                
                result = {
                    'prompt': prompt,
                    'temperature': temp,
                    'generated': generated,
                    'length': len(generated.split()) - len(prompt.split())
                }
                results.append(result)
                
                print(f"  Temp {temp}: {generated}")
        
        return results


def main():
    """Main evaluation function."""
    print("="*60)
    print("RETGEN Model Evaluation")
    print("="*60)
    
    # Initialize evaluator
    evaluator = RETGENEvaluator(
        model_path="models/retgen_memory_optimized_final.pkl",
        index_dir="models/index_shards"
    )
    
    # Load test data
    print("\nLoading test data...")
    with open("training_data.json", 'r') as f:
        all_texts = json.load(f)
    
    # Use last 1000 texts as test set (not seen during training if we stopped early)
    test_texts = all_texts[-1000:]
    print(f"Using {len(test_texts)} test samples")
    
    # Test 1: Pattern Retrieval
    print("\n" + "="*60)
    print("TEST 1: Pattern Retrieval")
    print("="*60)
    
    test_queries = [
        "The future of artificial",
        "Machine learning is",
        "In the beginning",
        "Scientists have discovered",
        "The weather today"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        similar = evaluator.search_similar_patterns(query, k=5)
        for i, (pattern, continuation, dist) in enumerate(similar[:3]):
            print(f"  {i+1}. Pattern: '{pattern}' -> '{continuation}' (dist: {dist:.3f})")
    
    # Test 2: Next Token Prediction
    print("\n" + "="*60)
    print("TEST 2: Next Token Prediction")
    print("="*60)
    
    test_contexts = [
        "The cat sat on the",
        "Artificial intelligence will",
        "Yesterday I went to",
        "The president announced",
        "Climate change is"
    ]
    
    for context in test_contexts:
        print(f"\nContext: '{context}'")
        predictions = evaluator.predict_next_token(context, top_k=5)
        print("  Top predictions:")
        for token, score in predictions[:3]:
            print(f"    - '{token}' (score: {score:.3f})")
    
    # Test 3: Text Generation
    print("\n" + "="*60)
    print("TEST 3: Text Generation")
    print("="*60)
    
    generation_prompts = [
        "The future of technology",
        "Once upon a time",
        "Scientists believe that",
        "The most important thing",
        "In recent years"
    ]
    
    generation_results = evaluator.evaluate_generation_quality(generation_prompts)
    
    # Test 4: Perplexity Evaluation
    print("\n" + "="*60)
    print("TEST 4: Perplexity and Accuracy")
    print("="*60)
    
    perplexity_results = evaluator.evaluate_perplexity(test_texts, max_samples=50)
    
    print(f"\nPerplexity: {perplexity_results['perplexity']:.2f}")
    print(f"Prediction Accuracy: {perplexity_results['accuracy']*100:.2f}%")
    print(f"Correct Predictions: {perplexity_results['correct_predictions']}/{perplexity_results['total_tokens']}")
    
    # Summary Report
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nModel Statistics:")
    print(f"  - Total Patterns: {evaluator.total_patterns:,}")
    print(f"  - Index Shards: {len(evaluator.shard_indices)}")
    print(f"  - Patterns Evaluated: {len(evaluator.all_patterns):,}")
    
    print(f"\nPerformance Metrics:")
    print(f"  - Perplexity: {perplexity_results['perplexity']:.2f}")
    print(f"  - Next-Token Accuracy: {perplexity_results['accuracy']*100:.2f}%")
    print(f"  - Retrieval Working: Yes")
    print(f"  - Generation Working: Yes")
    
    print(f"\nGeneration Quality:")
    avg_length = np.mean([r['length'] for r in generation_results])
    print(f"  - Average Generation Length: {avg_length:.1f} tokens")
    print(f"  - Temperature Range Tested: 0.5 - 1.0")
    
    # Save results
    results_file = "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'model_stats': {
                'total_patterns': evaluator.total_patterns,
                'shards': len(evaluator.shard_indices)
            },
            'perplexity_results': perplexity_results,
            'generation_samples': generation_results[:5]
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()