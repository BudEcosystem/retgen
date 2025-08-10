#!/usr/bin/env python3
"""
RETGEN v2: Improved Retrieval-Enhanced Text Generation
With hierarchical indexing, learned policies, quantization, and energy-based reranking
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RETGENConfig:
    """Configuration for RETGEN v2."""
    embedding_dim: int = 384
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search
    pq_nbits: int = 8  # Bits per sub-quantizer for PQ
    pq_nsplits: int = 48  # Number of sub-quantizers (384/8 = 48)
    energy_temperature: float = 0.1
    learning_rate: float = 0.01
    policy_hidden_dim: int = 256
    max_patterns_per_shard: int = 500000
    use_gpu: bool = True


class LearnedRetrievalPolicy(nn.Module):
    """Learned policy network for adaptive retrieval."""
    
    def __init__(self, config: RETGENConfig):
        super().__init__()
        self.config = config
        
        # Policy network: query embedding -> retrieval parameters
        self.policy_net = nn.Sequential(
            nn.Linear(config.embedding_dim, config.policy_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.policy_hidden_dim, config.policy_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.policy_hidden_dim, 3)  # Output: [k, temperature, diversity_weight]
        )
        
        # Value network for energy estimation
        self.value_net = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.policy_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.policy_hidden_dim, 1)
        )
        
    def forward(self, query_embedding: torch.Tensor) -> Dict[str, float]:
        """Compute retrieval parameters from query."""
        params = self.policy_net(query_embedding)
        
        # Constrain parameters to valid ranges
        k = torch.sigmoid(params[0]) * 100 + 10  # k in [10, 110]
        temperature = torch.sigmoid(params[1]) * 2  # temp in [0, 2]
        diversity = torch.sigmoid(params[2])  # diversity in [0, 1]
        
        return {
            'k': int(k.item()),
            'temperature': temperature.item(),
            'diversity_weight': diversity.item()
        }
    
    def compute_energy(self, query_emb: torch.Tensor, retrieved_emb: torch.Tensor) -> torch.Tensor:
        """Compute energy function for query-pattern pair."""
        combined = torch.cat([query_emb, retrieved_emb], dim=-1)
        return -self.value_net(combined)  # Negative for minimization


class HierarchicalIndex:
    """Hierarchical IVF index with Product Quantization for efficient search."""
    
    def __init__(self, config: RETGENConfig):
        self.config = config
        self.indices = []  # List of FAISS indices for each level
        self.quantizers = []  # Product quantizers for each level
        self.metadata = []  # Pattern metadata for each level
        
    def build_index(self, embeddings: np.ndarray, patterns: List[str], level: int = 0):
        """Build hierarchical index with IVF and PQ."""
        n_samples = embeddings.shape[0]
        d = embeddings.shape[1]
        
        logger.info(f"Building index level {level} with {n_samples} patterns")
        
        # Create quantizer for clustering
        quantizer = faiss.IndexFlatL2(d)
        
        # Create IVF index with Product Quantization
        if n_samples > 50000:
            # Use IVF with PQ for large datasets
            index = faiss.IndexIVFPQ(
                quantizer, d, 
                min(self.config.nlist, n_samples // 100),  # Adaptive number of clusters
                self.config.pq_nsplits,  # Sub-quantizers
                self.config.pq_nbits  # Bits per code
            )
        else:
            # Use IVF with flat storage for smaller datasets
            index = faiss.IndexIVFFlat(
                quantizer, d,
                min(self.config.nlist, n_samples // 100)
            )
        
        # Train the index
        logger.info(f"Training index on {n_samples} vectors...")
        index.train(embeddings.astype(np.float32))
        
        # Add vectors
        index.add(embeddings.astype(np.float32))
        
        # Set search parameters
        index.nprobe = self.config.nprobe
        
        # Store index and metadata
        self.indices.append(index)
        self.metadata.append({
            'patterns': patterns,
            'level': level,
            'size': n_samples
        })
        
        logger.info(f"Index level {level} built successfully")
        
        return index
    
    def search(self, query_embedding: np.ndarray, k: int = 50, level: int = -1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Search in hierarchical index."""
        # Return empty results if no indices built yet
        if not self.indices:
            return np.array([]), np.array([]), []
            
        if level == -1:
            # Search all levels and aggregate
            all_distances = []
            all_indices = []
            all_patterns = []
            
            for l, index in enumerate(self.indices):
                D, I = index.search(query_embedding.astype(np.float32), min(k, index.ntotal))
                
                # Map to actual patterns
                patterns = [self.metadata[l]['patterns'][i] for i in I[0] if i < len(self.metadata[l]['patterns'])]
                
                all_distances.append(D)
                all_indices.append(I)
                all_patterns.extend(patterns)
            
            # Aggregate results
            if all_distances:
                distances = np.hstack(all_distances)
                indices = np.hstack(all_indices)
                
                # Sort by distance
                sorted_idx = np.argsort(distances[0])[:k]
                
                return distances[0][sorted_idx], indices[0][sorted_idx], [all_patterns[i] for i in sorted_idx if i < len(all_patterns)]
            else:
                return np.array([]), np.array([]), []
        else:
            # Search specific level
            if level >= len(self.indices):
                return np.array([]), np.array([]), []
            index = self.indices[level]
            D, I = index.search(query_embedding.astype(np.float32), min(k, index.ntotal))
            patterns = [self.metadata[level]['patterns'][i] for i in I[0] if i < len(self.metadata[level]['patterns'])]
            
            return D[0], I[0], patterns
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Calculate memory usage of indices."""
        total_bytes = 0
        details = {}
        
        for l, index in enumerate(self.indices):
            # Estimate based on index type
            if hasattr(index, 'pq'):
                # PQ compressed index
                bytes_per_vec = self.config.pq_nsplits * self.config.pq_nbits / 8
                size = index.ntotal * bytes_per_vec
            else:
                # Flat index
                size = index.ntotal * self.config.embedding_dim * 4  # float32
            
            details[f'level_{l}'] = size / (1024**3)  # GB
            total_bytes += size
        
        details['total_gb'] = total_bytes / (1024**3)
        return details


class EnergyBasedReranker:
    """Energy-based reranking for improved pattern selection."""
    
    def __init__(self, config: RETGENConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        
        # Energy function components
        self.semantic_weight = nn.Parameter(torch.tensor(1.0))
        self.diversity_weight = nn.Parameter(torch.tensor(0.5))
        self.frequency_weight = nn.Parameter(torch.tensor(0.3))
        
    def compute_energy(self, 
                      query_emb: torch.Tensor,
                      candidate_embs: torch.Tensor,
                      similarities: torch.Tensor,
                      frequencies: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute energy for each candidate pattern.
        Lower energy = better match.
        
        E(x, p) = -w_s * sim(x, p) + w_d * div(p, P) + w_f * log(freq(p))
        """
        batch_size = candidate_embs.shape[0]
        
        # Semantic similarity energy (negative for minimization)
        semantic_energy = -self.semantic_weight * similarities
        
        # Diversity energy (encourage diverse patterns)
        if batch_size > 1:
            # Pairwise distances between candidates
            pairwise_dist = torch.cdist(candidate_embs, candidate_embs, p=2)
            diversity_energy = -self.diversity_weight * pairwise_dist.mean(dim=1)
        else:
            diversity_energy = torch.zeros_like(semantic_energy)
        
        # Frequency energy (prefer common patterns)
        if frequencies is not None:
            frequency_energy = -self.frequency_weight * torch.log(frequencies + 1e-6)
        else:
            frequency_energy = torch.zeros_like(semantic_energy)
        
        # Total energy
        total_energy = semantic_energy + diversity_energy + frequency_energy
        
        return total_energy
    
    def rerank(self, 
               query_emb: np.ndarray,
               candidate_embs: np.ndarray,
               similarities: np.ndarray,
               top_k: int = 10) -> np.ndarray:
        """Rerank candidates based on energy minimization."""
        # Convert to tensors
        query_tensor = torch.from_numpy(query_emb).float().to(self.device)
        candidates_tensor = torch.from_numpy(candidate_embs).float().to(self.device)
        sims_tensor = torch.from_numpy(similarities).float().to(self.device)
        
        # Compute energies
        energies = self.compute_energy(query_tensor, candidates_tensor, sims_tensor)
        
        # Sort by energy (ascending - lower is better)
        sorted_indices = torch.argsort(energies)[:top_k]
        
        return sorted_indices.cpu().numpy()


class RETGENv2:
    """RETGEN v2 with all improvements."""
    
    def __init__(self, config: RETGENConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        
        # Initialize components
        logger.info("Initializing RETGEN v2...")
        
        # Encoder
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encoder = self.encoder.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Hierarchical index
        self.index = HierarchicalIndex(config)
        
        # Learned retrieval policy
        self.policy = LearnedRetrievalPolicy(config).to(self.device)
        
        # Energy-based reranker
        self.reranker = EnergyBasedReranker(config)
        
        # Pattern storage
        self.patterns = []
        self.continuations = []
        self.pattern_frequencies = {}
        
        # Optimizer for learning
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + 
            [self.reranker.semantic_weight, 
             self.reranker.diversity_weight,
             self.reranker.frequency_weight],
            lr=config.learning_rate
        )
        
        logger.info("RETGEN v2 initialized successfully")
    
    def extract_patterns(self, text: str, resolutions: List[int] = [1, 2, 3, 5, 8]) -> List[Tuple[str, str]]:
        """Extract multi-resolution patterns from text."""
        tokens = self.tokenizer.tokenize(text)
        patterns = []
        
        for resolution in resolutions:
            for i in range(len(tokens) - resolution):
                pattern = " ".join(tokens[i:i+resolution])
                continuation = tokens[i+resolution] if i+resolution < len(tokens) else ""
                patterns.append((pattern, continuation))
                
                # Update frequency
                if pattern not in self.pattern_frequencies:
                    self.pattern_frequencies[pattern] = 0
                self.pattern_frequencies[pattern] += 1
        
        return patterns
    
    def train_on_corpus(self, texts: List[str], batch_size: int = 256):
        """Train RETGEN v2 on text corpus."""
        logger.info(f"Training on {len(texts)} texts...")
        
        all_patterns = []
        all_continuations = []
        
        # Extract patterns
        for text in tqdm(texts, desc="Extracting patterns"):
            patterns = self.extract_patterns(text)
            for pattern, continuation in patterns:
                all_patterns.append(pattern)
                all_continuations.append(continuation)
        
        # Remove duplicates while preserving order
        unique_patterns = []
        unique_continuations = []
        seen = set()
        
        for p, c in zip(all_patterns, all_continuations):
            if p not in seen:
                seen.add(p)
                unique_patterns.append(p)
                unique_continuations.append(c)
        
        logger.info(f"Extracted {len(unique_patterns)} unique patterns")
        
        # Encode patterns
        logger.info("Encoding patterns...")
        embeddings = []
        
        for i in tqdm(range(0, len(unique_patterns), batch_size), desc="Encoding"):
            batch = unique_patterns[i:i+batch_size]
            batch_emb = self.encoder.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            embeddings.append(batch_emb)
        
        embeddings = np.vstack(embeddings)
        
        # Build hierarchical index
        logger.info("Building hierarchical index...")
        
        # Split into levels based on pattern length
        levels = {}
        for i, pattern in enumerate(unique_patterns):
            length = len(pattern.split())
            if length not in levels:
                levels[length] = {'embeddings': [], 'patterns': [], 'continuations': []}
            levels[length]['embeddings'].append(embeddings[i])
            levels[length]['patterns'].append(pattern)
            levels[length]['continuations'].append(unique_continuations[i])
        
        # Build index for each level
        for length, data in sorted(levels.items()):
            if len(data['patterns']) > 100:  # Only index levels with enough patterns
                level_embeddings = np.vstack(data['embeddings'])
                self.index.build_index(level_embeddings, data['patterns'], level=length)
                
                # Store patterns
                self.patterns.extend(data['patterns'])
                self.continuations.extend(data['continuations'])
        
        # Report memory usage
        memory_usage = self.index.get_memory_usage()
        logger.info(f"Index memory usage: {memory_usage}")
        
    def retrieve_patterns(self, query: str, adaptive: bool = True) -> List[Dict]:
        """
        Retrieve relevant patterns using learned policy and energy-based reranking.
        """
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        query_tensor = torch.from_numpy(query_embedding[0]).float().to(self.device)
        
        # Get retrieval parameters from learned policy
        if adaptive:
            with torch.no_grad():
                params = self.policy(query_tensor)
            k = params['k']
            temperature = params['temperature']
        else:
            k = 50
            temperature = 1.0
        
        # Search in hierarchical index
        distances, indices, patterns = self.index.search(query_embedding, k=k)
        
        # Check if any results were found
        if len(distances) == 0 or len(patterns) == 0:
            return []
        
        # Get embeddings of retrieved patterns for reranking
        if len(patterns) > 0:
            pattern_embeddings = self.encoder.encode(patterns, convert_to_numpy=True, normalize_embeddings=True)
            similarities = 1.0 / (1.0 + distances)
            
            # Energy-based reranking
            reranked_indices = self.reranker.rerank(
                query_embedding[0],
                pattern_embeddings,
                similarities,
                top_k=min(10, len(patterns))
            )
            
            # Get reranked results
            results = []
            for idx in reranked_indices:
                if idx < len(patterns):
                    pattern_idx = indices[idx]
                    if pattern_idx < len(self.continuations):
                        results.append({
                            'pattern': patterns[idx],
                            'continuation': self.continuations[pattern_idx],
                            'similarity': similarities[idx],
                            'energy': 0  # Would be computed by reranker
                        })
            
            return results
        
        return []
    
    def generate(self, prompt: str, max_length: int = 50) -> str:
        """Generate text using improved retrieval."""
        generated = prompt
        
        for _ in range(max_length):
            # Retrieve patterns
            results = self.retrieve_patterns(generated[-100:])  # Use last 100 chars as context
            
            if not results:
                break
            
            # Aggregate predictions with energy weighting
            predictions = {}
            for r in results:
                cont = r['continuation']
                if cont not in predictions:
                    predictions[cont] = 0
                # Weight by similarity and inverse energy
                predictions[cont] += r['similarity']
            
            # Sample next token
            if predictions:
                # Softmax over predictions
                tokens = list(predictions.keys())
                scores = np.array(list(predictions.values()))
                probs = np.exp(scores) / np.sum(np.exp(scores))
                
                next_token = np.random.choice(tokens, p=probs)
                generated += " " + next_token
                
                if next_token in ['.', '!', '?']:
                    break
        
        return generated
    
    def update_policy(self, query: str, feedback: float):
        """Update retrieval policy based on feedback."""
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        query_tensor = torch.from_numpy(query_embedding[0]).float().to(self.device)
        
        # Compute loss based on feedback (negative feedback = high loss)
        loss = -feedback * self.policy.compute_energy(query_tensor, query_tensor)
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, path: str):
        """Save the model."""
        logger.info(f"Saving model to {path}")
        
        save_data = {
            'config': self.config,
            'patterns': self.patterns,
            'continuations': self.continuations,
            'pattern_frequencies': self.pattern_frequencies,
            'policy_state': self.policy.state_dict(),
            'reranker_weights': {
                'semantic': self.reranker.semantic_weight.item(),
                'diversity': self.reranker.diversity_weight.item(),
                'frequency': self.reranker.frequency_weight.item()
            }
        }
        
        # Save indices
        for i, index in enumerate(self.index.indices):
            faiss.write_index(index, f"{path}_index_level_{i}.faiss")
        
        # Save metadata
        with open(f"{path}_metadata.pkl", 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info("Model saved successfully")


def benchmark_improvements():
    """Benchmark the improvements over baseline RETGEN."""
    
    # Configuration
    config = RETGENConfig(
        nlist=100,
        nprobe=10,
        pq_nbits=8,
        pq_nsplits=48
    )
    
    # Initialize model
    model = RETGENv2(config)
    
    # Sample training data
    sample_texts = [
        "The future of artificial intelligence is bright.",
        "Machine learning models are becoming more sophisticated.",
        "Natural language processing has made significant advances.",
        "Deep learning revolutionized computer vision.",
        "Transformers changed the landscape of NLP."
    ] * 100  # Repeat for more data
    
    # Train
    model.train_on_corpus(sample_texts, batch_size=32)
    
    # Test generation
    test_prompts = [
        "The future of",
        "Machine learning",
        "Natural language"
    ]
    
    print("\n" + "="*60)
    print("RETGEN v2 GENERATION RESULTS")
    print("="*60)
    
    for prompt in test_prompts:
        generated = model.generate(prompt, max_length=20)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        
        # Show retrieval details
        results = model.retrieve_patterns(prompt)
        print(f"Top retrieved patterns:")
        for r in results[:3]:
            print(f"  - '{r['pattern']}' -> '{r['continuation']}' (sim: {r['similarity']:.3f})")
    
    # Memory comparison
    memory_usage = model.index.get_memory_usage()
    print("\n" + "="*60)
    print("MEMORY USAGE COMPARISON")
    print("="*60)
    print(f"Hierarchical IVF+PQ Index: {memory_usage['total_gb']:.2f} GB")
    print(f"Original Flat Index (estimated): {len(model.patterns) * 384 * 4 / (1024**3):.2f} GB")
    if memory_usage['total_gb'] > 0:
        print(f"Compression ratio: {(len(model.patterns) * 384 * 4) / (memory_usage['total_gb'] * 1024**3):.1f}x")
    else:
        print(f"Compression ratio: N/A (no data indexed)")
    
    return model


if __name__ == "__main__":
    # Run benchmark
    model = benchmark_improvements()
    
    # Save model
    model.save("models/retgen_v2")