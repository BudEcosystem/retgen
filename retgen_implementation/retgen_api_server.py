#!/usr/bin/env python3
"""Flask API server for RETGEN model interaction."""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from collections import deque, Counter
from typing import List, Dict, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


class RETGENServer:
    """RETGEN model server for web interface."""
    
    def __init__(self, model_path: str, index_dir: str):
        """Initialize the model server."""
        print("Initializing RETGEN server...")
        
        # Load model metadata
        with open(model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.total_patterns = self.model_data['total_patterns']
        self.num_shards = len(self.model_data.get('shard_indices', []))
        
        print(f"Model loaded: {self.total_patterns:,} patterns, {self.num_shards} shards")
        
        # Initialize components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        print("Loading encoder...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        
        # Load shards
        self.index_dir = Path(index_dir)
        self.load_shards()
        
        print("RETGEN server ready!")
    
    def load_shards(self, num_shards: int = 3):
        """Load multiple shards for better coverage."""
        shard_files = sorted(self.index_dir.glob("shard_*.faiss"))
        
        if not shard_files:
            raise ValueError("No shard files found!")
        
        # Load evenly distributed shards
        indices_to_load = [
            min(10, len(shard_files)-1),
            min(25, len(shard_files)-1),
            min(40, len(shard_files)-1)
        ]
        
        self.indices = []
        self.all_patterns = []
        self.all_continuations = []
        
        for idx in indices_to_load[:num_shards]:
            shard_file = shard_files[idx]
            print(f"Loading shard: {shard_file.name}")
            
            # Load index
            index = faiss.read_index(str(shard_file))
            self.indices.append(index)
            
            # Load metadata
            meta_path = str(shard_file).replace('.faiss', '_meta.pkl')
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                self.all_patterns.extend(meta['patterns'][:170000])  # Limit for memory
                self.all_continuations.extend(meta['continuations'][:170000])
        
        print(f"Loaded {len(self.all_patterns):,} patterns")
    
    def search_patterns(self, query: str, k: int = 30) -> List[Dict]:
        """Search for similar patterns."""
        with torch.no_grad():
            query_embedding = self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        all_results = []
        pattern_offset = 0
        
        for idx, index in enumerate(self.indices):
            patterns_per_shard = len(self.all_patterns) // len(self.indices)
            
            distances, indices = index.search(
                query_embedding.astype(np.float32), 
                min(k, index.ntotal)
            )
            
            for dist, i in zip(distances[0], indices[0]):
                actual_idx = pattern_offset + i
                if actual_idx < len(self.all_patterns):
                    all_results.append({
                        'pattern': self.all_patterns[actual_idx],
                        'continuation': self.all_continuations[actual_idx],
                        'distance': float(dist),
                        'similarity': 1.0 / (1.0 + float(dist))
                    })
            
            pattern_offset += patterns_per_shard
        
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        return all_results[:k]
    
    def nucleus_sampling(self, predictions: List[Tuple[str, float]], p: float = 0.9, temperature: float = 1.0) -> str:
        """Nucleus sampling implementation."""
        if not predictions:
            return ""
        
        scores = np.array([score for _, score in predictions])
        
        if temperature > 0:
            scores = scores / temperature
        else:
            return predictions[0][0]
        
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        cumsum = np.cumsum(sorted_probs)
        nucleus_size = np.searchsorted(cumsum, p) + 1
        nucleus_size = max(1, min(nucleus_size, len(predictions)))
        
        nucleus_indices = sorted_indices[:nucleus_size]
        nucleus_probs = probs[nucleus_indices]
        nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
        
        chosen_idx = np.random.choice(nucleus_indices, p=nucleus_probs)
        return predictions[chosen_idx][0]
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        sampling_method: str = "nucleus"
    ) -> Dict:
        """Generate text with specified parameters."""
        generated = prompt
        generated_tokens = prompt.split()
        tokens_generated = []
        
        for i in range(max_tokens):
            context = generated[-100:]  # Use last 100 chars as context
            
            # Search for similar patterns
            results = self.search_patterns(context, k=50)
            
            if not results:
                break
            
            # Aggregate predictions
            predictions = {}
            for r in results:
                if r['similarity'] < 0.1:
                    continue
                
                cont = r['continuation']
                if cont.startswith('##'):
                    continue
                
                weight = r['similarity']
                if cont not in predictions:
                    predictions[cont] = 0
                predictions[cont] += weight
            
            if not predictions:
                break
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                recent_tokens = generated_tokens[-20:]
                for token in set(recent_tokens):
                    if token in predictions:
                        count = recent_tokens.count(token)
                        predictions[token] /= (repetition_penalty ** count)
            
            # Sort predictions
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
            # Sample next token
            if sampling_method == "nucleus":
                next_token = self.nucleus_sampling(sorted_preds, p=top_p, temperature=temperature)
            elif sampling_method == "greedy":
                next_token = sorted_preds[0][0]
            else:  # top-k
                k = min(10, len(sorted_preds))
                top_k_preds = sorted_preds[:k]
                scores = np.array([s for _, s in top_k_preds])
                if temperature > 0:
                    scores = scores / temperature
                    exp_scores = np.exp(scores - np.max(scores))
                    probs = exp_scores / np.sum(exp_scores)
                    idx = np.random.choice(len(top_k_preds), p=probs)
                    next_token = top_k_preds[idx][0]
                else:
                    next_token = top_k_preds[0][0]
            
            generated += " " + next_token
            generated_tokens.append(next_token)
            tokens_generated.append(next_token)
            
            if next_token in ['.', '!', '?'] and len(tokens_generated) > 5:
                break
        
        return {
            'generated': generated,
            'tokens_added': tokens_generated,
            'num_tokens': len(tokens_generated)
        }


# Initialize model server
model_server = None

@app.route('/')
def index():
    """Serve the HTML interface."""
    return send_from_directory('.', 'retgen_interface.html')

@app.route('/retrieval')
def retrieval_viz():
    """Serve the retrieval visualization interface."""
    return send_from_directory('.', 'retgen_retrieval_viz.html')

@app.route('/report')
def pattern_report():
    """Serve the pattern visualization report."""
    return send_from_directory('.', 'pattern_visualization_report.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get model status."""
    if model_server:
        return jsonify({
            'status': 'ready',
            'total_patterns': model_server.total_patterns,
            'num_shards': model_server.num_shards,
            'patterns_loaded': len(model_server.all_patterns),
            'device': str(model_server.device)
        })
    else:
        return jsonify({'status': 'not_initialized'})

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate text endpoint."""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 50)
        temperature = data.get('temperature', 0.8)
        top_p = data.get('top_p', 0.9)
        repetition_penalty = data.get('repetition_penalty', 1.2)
        sampling_method = data.get('sampling_method', 'nucleus')
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        result = model_server.generate_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            sampling_method=sampling_method
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search():
    """Search for similar patterns."""
    try:
        data = request.json
        query = data.get('query', '')
        k = data.get('k', 10)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        results = model_server.search_patterns(query, k=k)
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """Main function to start the server."""
    global model_server
    
    print("Starting RETGEN API Server...")
    
    # Initialize model
    model_server = RETGENServer(
        model_path="models/retgen_memory_optimized_final.pkl",
        index_dir="models/index_shards"
    )
    
    # Start Flask server
    print("\nServer ready! Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()