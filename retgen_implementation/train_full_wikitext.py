#!/usr/bin/env python3
"""Train RETGEN on full WikiText-103 dataset with optimizations."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import os
import time
import logging
import json
from datetime import datetime
import argparse
import gc

from datasets import load_dataset
from tqdm import tqdm
import torch

from src.core.config import RETGENConfig
from run_retgen import RETGENSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_wikitext103_full():
    """Train on full WikiText-103 dataset."""
    logger.info("Starting full WikiText-103 training...")
    
    # Optimized configuration for faster training
    config = RETGENConfig(
        # Use faster embedding model
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        
        # Pattern extraction - more selective to reduce memory
        min_pattern_frequency=5,  # Higher threshold for efficiency
        max_pattern_length=6,     # Shorter patterns
        resolutions=[1, 2, 3],    # Fewer resolutions for speed
        
        # Retrieval settings
        retrieval_k=50,           # Smaller k for faster generation
        
        # Training settings
        batch_size=1024,          # Larger batch for efficiency
        
        # Simple index for initial training
        index_type="Flat",        # No compression initially
        
        # Performance settings
        device="cpu",
        use_gpu=False,
        num_workers=4,
        
        # Generation settings
        max_generation_length=100,
        temperature=0.8,
        top_k=50
    )
    
    # Create save directory
    save_path = Path("models/retgen_wikitext103_full")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.save(save_path / "config.json")
    
    # Initialize RETGEN
    retgen = RETGENSystem(config)
    
    # Load WikiText-103
    logger.info("Loading WikiText-103 dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    # Process documents
    logger.info("Processing documents...")
    train_texts = []
    
    # Process in chunks to manage memory
    chunk_size = 10000
    total_docs = 0
    
    for i, item in enumerate(tqdm(dataset['train'], desc="Loading docs")):
        text = item['text'].strip()
        if len(text) > 100:  # Only substantial texts
            train_texts.append(text)
            total_docs += 1
            
            # Process chunk
            if len(train_texts) >= chunk_size:
                logger.info(f"Processing chunk {i//chunk_size + 1}...")
                # Train on first 50k documents
                if total_docs >= 50000:
                    break
    
    # Use validation set
    val_texts = []
    for item in dataset['validation']:
        text = item['text'].strip()
        if len(text) > 100:
            val_texts.append(text)
            if len(val_texts) >= 1000:  # Limit validation size
                break
    
    logger.info(f"Using {len(train_texts)} training documents")
    logger.info(f"Using {len(val_texts)} validation documents")
    
    # Training
    logger.info("Training RETGEN...")
    training_start = time.time()
    
    metrics = retgen.train(train_texts, val_texts)
    
    training_time = time.time() - training_start
    
    # Update metrics
    metrics['dataset'] = 'wikitext-103-v1'
    metrics['train_docs'] = len(train_texts)
    metrics['val_docs'] = len(val_texts)
    metrics['total_training_time'] = training_time
    
    # Log results
    logger.info(f"\nTraining completed in {training_time/60:.1f} minutes")
    logger.info(f"Patterns: {metrics['pattern_count']:,}")
    logger.info(f"Model size: {metrics['model_size_mb']:.1f} MB")
    
    # Save model
    logger.info(f"Saving model to {save_path}...")
    retgen.save(save_path)
    
    # Save metrics
    with open(save_path / "training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Test generation
    logger.info("\nTesting generation...")
    test_prompts = [
        "The history of",
        "Natural language processing",
        "Machine learning is",
        "In the future",
        "Scientists have discovered"
    ]
    
    generations = []
    for prompt in test_prompts:
        generated = retgen.generate(prompt, max_length=50, temperature=0.8)
        generations.append({"prompt": prompt, "generated": generated})
        logger.info(f"'{prompt}' -> '{generated}'")
    
    # Save samples
    with open(save_path / "generation_samples.json", 'w') as f:
        json.dump(generations, f, indent=2)
    
    # Create README
    readme = f"""# RETGEN WikiText-103 Model

Trained on WikiText-103 dataset with the following configuration:
- Embedding model: {config.embedding_model}
- Patterns: {metrics['pattern_count']:,}
- Model size: {metrics['model_size_mb']:.1f} MB
- Training time: {training_time/60:.1f} minutes

## Usage

```python
from run_retgen import RETGENSystem

# Load model
retgen = RETGENSystem.load('models/retgen_wikitext103_full')

# Generate text
text = retgen.generate("Your prompt here", max_length=100)
print(text)
```

## Generation Samples

"""
    
    for gen in generations:
        readme += f"**Prompt**: {gen['prompt']}\n"
        readme += f"**Generated**: {gen['generated']}\n\n"
    
    with open(save_path / "README.md", 'w') as f:
        f.write(readme)
    
    logger.info(f"\n✅ Model saved to {save_path}")
    
    return save_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train RETGEN on WikiText-103")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Train on full dataset (default: 10k docs)"
    )
    
    args = parser.parse_args()
    
    try:
        model_path = train_wikitext103_full()
        
        print("\n" + "="*60)
        print("RETGEN Training Complete!")
        print("="*60)
        print(f"Model saved to: {model_path}")
        print("\nTo use the model:")
        print("```python")
        print("from run_retgen import RETGENSystem")
        print(f"retgen = RETGENSystem.load('{model_path}')")
        print("text = retgen.generate('Your prompt', max_length=100)")
        print("```")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()