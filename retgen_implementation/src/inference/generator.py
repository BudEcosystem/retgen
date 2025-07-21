"""RETGEN text generation engine."""

import time
import random
import logging
from typing import List, Optional, Dict, Any
import numpy as np
from transformers import AutoTokenizer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import RETGENConfig
from embeddings.context_embeddings import RETGENEmbedder
from indexing.vector_database import VectorDatabase


logger = logging.getLogger(__name__)


class RETGENGenerator:
    """Text generator using RETGEN approach."""
    
    def __init__(
        self,
        embedder: RETGENEmbedder,
        database: VectorDatabase,
        config: RETGENConfig
    ):
        """Initialize RETGEN generator.
        
        Args:
            embedder: RETGEN embedder
            database: Vector database
            config: RETGEN configuration
        """
        self.embedder = embedder
        self.database = database
        self.config = config
        
        # Initialize tokenizer for decoding
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
    
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop_tokens: Optional[List[str]] = None
    ) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            stop_tokens: Tokens to stop generation
            
        Returns:
            Generated text
        """
        # Use config defaults if not specified
        max_length = max_length or self.config.max_generation_length
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        repetition_penalty = repetition_penalty or self.config.repetition_penalty
        stop_tokens = stop_tokens or [".", "!", "?", "\n"]
        
        generated_text = prompt
        generation_steps = 0
        
        while generation_steps < max_length:
            # Get next token probabilities
            token_probs = self.get_next_token_probs(
                generated_text,
                temperature=temperature
            )
            
            if not token_probs:
                break
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                token_probs = self._apply_repetition_penalty(
                    token_probs,
                    generated_text,
                    repetition_penalty
                )
            
            # Sample next token
            next_token_id = self._sample_token(token_probs, top_p=top_p, top_k=top_k)
            
            # Decode token
            next_token = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            
            # Check for stop tokens
            if next_token.strip() in stop_tokens:
                generated_text += next_token
                break
            
            # Add to generated text
            generated_text += next_token
            generation_steps += 1
            
            # Stop if we hit special tokens
            if next_token_id in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                break
        
        return generated_text
    
    def get_next_token_probs(
        self,
        context: str,
        temperature: float = 1.0,
        k: Optional[int] = None
    ) -> Dict[int, float]:
        """Get probability distribution for next token.
        
        Args:
            context: Current context
            temperature: Sampling temperature
            k: Number of patterns to retrieve
            
        Returns:
            Dictionary mapping token IDs to probabilities
        """
        k = k or self.config.retrieval_k
        
        # Embed context
        context_embedding = self.embedder.embed_text(context)
        query = context_embedding.reshape(1, -1)
        
        # Get continuation distributions from database
        distributions = self.database.get_continuation_distribution(query, k=k)
        
        if not distributions or not distributions[0]:
            return {}
        
        distribution = distributions[0]
        
        # Apply temperature
        if temperature != 1.0:
            # Convert to logits and apply temperature
            logits = {}
            for token_id, prob in distribution.items():
                logit = np.log(prob + 1e-10)
                logits[token_id] = logit / temperature
            
            # Convert back to probabilities
            max_logit = max(logits.values())
            exp_logits = {k: np.exp(v - max_logit) for k, v in logits.items()}
            total = sum(exp_logits.values())
            
            distribution = {k: v / total for k, v in exp_logits.items()}
        
        return distribution
    
    def _apply_repetition_penalty(
        self,
        token_probs: Dict[int, float],
        context: str,
        penalty: float
    ) -> Dict[int, float]:
        """Apply repetition penalty to token probabilities.
        
        Args:
            token_probs: Token probabilities
            context: Current context
            penalty: Repetition penalty factor
            
        Returns:
            Modified probabilities
        """
        # Tokenize context to find recent tokens
        context_tokens = self.tokenizer.encode(context[-100:])  # Last 100 chars
        recent_tokens = set(context_tokens[-20:])  # Last 20 tokens
        
        # Apply penalty to repeated tokens
        penalized_probs = {}
        for token_id, prob in token_probs.items():
            if token_id in recent_tokens:
                penalized_probs[token_id] = prob / penalty
            else:
                penalized_probs[token_id] = prob
        
        # Renormalize
        total = sum(penalized_probs.values())
        if total > 0:
            penalized_probs = {k: v / total for k, v in penalized_probs.items()}
        
        return penalized_probs
    
    def _sample_token(
        self,
        token_probs: Dict[int, float],
        top_p: float = 0.95,
        top_k: int = 50
    ) -> int:
        """Sample next token with nucleus sampling.
        
        Args:
            token_probs: Token probabilities
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Sampled token ID
        """
        if not token_probs:
            return self.tokenizer.eos_token_id or 0
        
        # Sort by probability
        sorted_tokens = sorted(token_probs.items(), key=lambda x: x[1], reverse=True)
        
        # Apply top-k filtering
        if top_k > 0:
            sorted_tokens = sorted_tokens[:top_k]
        
        # Apply top-p filtering
        if top_p < 1.0:
            cumsum = 0
            filtered_tokens = []
            for token_id, prob in sorted_tokens:
                cumsum += prob
                filtered_tokens.append((token_id, prob))
                if cumsum >= top_p:
                    break
            sorted_tokens = filtered_tokens
        
        # Renormalize
        tokens, probs = zip(*sorted_tokens)
        total = sum(probs)
        if total == 0:
            return tokens[0]
        
        probs = [p / total for p in probs]
        
        # Sample
        return np.random.choice(tokens, p=probs)
    
    def batch_generate(
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
        results = []
        for prompt in prompts:
            generated = self.generate(prompt, **kwargs)
            results.append(generated)
        
        return results
    
    def compute_perplexity(
        self,
        text: str,
        context_window: int = 100
    ) -> float:
        """Compute perplexity of text under RETGEN model.
        
        Args:
            text: Text to evaluate
            context_window: Context window size
            
        Returns:
            Perplexity score
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) < 2:
            return float('inf')
        
        log_prob_sum = 0.0
        token_count = 0
        
        for i in range(1, len(tokens)):
            # Get context
            start_idx = max(0, i - context_window)
            context_tokens = tokens[start_idx:i]
            context = self.tokenizer.decode(context_tokens, skip_special_tokens=True)
            
            # Get next token probabilities
            token_probs = self.get_next_token_probs(context)
            
            # Get probability of actual next token
            target_token = tokens[i]
            prob = token_probs.get(target_token, 1e-10)
            
            log_prob_sum += np.log(prob)
            token_count += 1
        
        # Compute perplexity
        if token_count == 0:
            return float('inf')
        
        avg_log_prob = log_prob_sum / token_count
        perplexity = np.exp(-avg_log_prob)
        
        return float(perplexity)