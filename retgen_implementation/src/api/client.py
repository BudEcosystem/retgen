#!/usr/bin/env python3
"""Python client for RETGEN REST API."""

import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class RETGENClient:
    """Client for interacting with RETGEN REST API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """Initialize client.
        
        Args:
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        # Setup session with retries
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/", timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        response = self.session.get(f"{self.base_url}/model/info", timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def train_model(
        self,
        documents: List[str],
        validation_split: float = 0.1,
        config_overrides: Optional[Dict[str, Any]] = None,
        wait: bool = True,
        poll_interval: int = 2
    ) -> Dict[str, Any]:
        """Train a new model.
        
        Args:
            documents: Training documents
            validation_split: Fraction for validation
            config_overrides: Configuration overrides
            wait: Whether to wait for training to complete
            poll_interval: Polling interval in seconds if waiting
            
        Returns:
            Training response
        """
        payload = {
            "documents": documents,
            "validation_split": validation_split
        }
        
        if config_overrides:
            payload["config_overrides"] = config_overrides
        
        response = self.session.post(
            f"{self.base_url}/model/train",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        if wait:
            # Poll until model is trained
            while True:
                info = self.get_model_info()
                if info.get("is_trained", False):
                    break
                time.sleep(poll_interval)
        
        return result
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Load a pre-trained model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Load response
        """
        response = self.session.post(
            f"{self.base_url}/model/load",
            json={"model_path": model_path},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def save_model(self, model_path: str) -> Dict[str, Any]:
        """Save the current model.
        
        Args:
            model_path: Path to save model
            
        Returns:
            Save response
        """
        response = self.session.post(
            f"{self.base_url}/model/save",
            json={"model_path": model_path},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        num_generations: int = 1
    ) -> Dict[str, Any]:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            num_generations: Number of generations
            
        Returns:
            Generation response with text
        """
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k,
            "num_generations": num_generations
        }
        
        response = self.session.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def generate_batch(
        self,
        prompts: List[str],
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> Dict[str, Any]:
        """Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Batch generation response
        """
        params = {
            "max_length": max_length,
            "temperature": temperature,
            "top_k": top_k
        }
        
        response = self.session.post(
            f"{self.base_url}/generate/batch",
            json=prompts,
            params=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()


def demo_client():
    """Demonstrate client usage."""
    print("RETGEN API Client Demo")
    print("=" * 50)
    
    # Create client
    client = RETGENClient()
    
    # Check health
    print("\n1. Health Check:")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}")
    
    # Train model
    print("\n2. Training Model:")
    documents = [
        "Natural language processing is a field of artificial intelligence.",
        "Machine learning algorithms can learn patterns from data.",
        "Deep learning models use neural networks with multiple layers.",
        "Text generation is the task of producing human-like text.",
        "Retrieval-augmented generation combines retrieval and generation."
    ] * 10  # Repeat for more data
    
    try:
        train_response = client.train_model(documents, validation_split=0.2)
        print(f"   Success: {train_response['success']}")
        print(f"   Message: {train_response['message']}")
        print(f"   Metrics: {train_response['metrics']}")
    except Exception as e:
        print(f"   Training failed: {e}")
    
    # Get model info
    print("\n3. Model Information:")
    info = client.get_model_info()
    print(f"   Loaded: {info['is_loaded']}")
    print(f"   Trained: {info['is_trained']}")
    
    # Generate text
    print("\n4. Text Generation:")
    if info['is_trained']:
        response = client.generate(
            prompt="Natural language processing",
            max_length=30,
            temperature=0.8
        )
        print(f"   Prompt: {response['prompt']}")
        print(f"   Generated: {response['generations'][0]}")
    
    # Batch generation
    print("\n5. Batch Generation:")
    if info['is_trained']:
        prompts = [
            "Machine learning",
            "Deep learning",
            "Text generation"
        ]
        batch_response = client.generate_batch(prompts, max_length=20)
        for prompt, gen in zip(batch_response['prompts'], batch_response['generations']):
            print(f"   '{prompt}' -> '{gen}'")


if __name__ == "__main__":
    demo_client()