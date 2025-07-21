#!/usr/bin/env python3
"""Train a demo-ready RETGEN model on WikiText-103 subset."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import time
import logging
import json
from datasets import load_dataset
from tqdm import tqdm

from src.core.config import RETGENConfig
from run_retgen import RETGENSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_demo_model():
    """Train a demo model on 10k WikiText-103 documents."""
    logger.info("Starting demo model training...")
    
    # Optimized configuration for demo
    config = RETGENConfig(
        # Use fast embedding model
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        
        # Efficient pattern extraction
        min_pattern_frequency=3,
        max_pattern_length=5,
        resolutions=[1, 2, 3],
        
        # Retrieval settings
        retrieval_k=30,
        
        # Training settings
        batch_size=1024,
        
        # Simple flat index for demo
        index_type="Flat",
        
        # Performance settings
        device="cpu",
        use_gpu=False,
        num_workers=0,
        
        # Generation settings
        max_generation_length=100,
        temperature=0.8,
        top_k=40,
        top_p=0.9
    )
    
    # Create save directory
    save_path = Path("models/retgen_demo")
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
    
    # Process first 10k substantial documents
    for item in tqdm(dataset['train'], desc="Loading docs"):
        text = item['text'].strip()
        if len(text) > 100:  # Only substantial texts
            train_texts.append(text)
            if len(train_texts) >= 10000:
                break
    
    # Use validation set
    val_texts = []
    for item in dataset['validation']:
        text = item['text'].strip()
        if len(text) > 100:
            val_texts.append(text)
            if len(val_texts) >= 500:
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
        "Scientists have discovered",
        "The United States",
        "Computer science",
        "Artificial intelligence"
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
    readme = f"""# RETGEN Demo Model

A demonstration model trained on 10,000 WikiText-103 documents.

## Model Details
- Embedding model: {config.embedding_model}
- Patterns: {metrics['pattern_count']:,}
- Model size: {metrics['model_size_mb']:.1f} MB
- Training time: {training_time/60:.1f} minutes

## Usage

```python
from run_retgen import RETGENSystem

# Load model
retgen = RETGENSystem.load('models/retgen_demo')

# Generate text
text = retgen.generate("Your prompt here", max_length=100)
print(text)
```

## Generation Examples

"""
    
    for i, gen in enumerate(generations[:5]):
        readme += f"{i+1}. **{gen['prompt']}**\n   {gen['generated']}\n\n"
    
    readme += """## Performance

This is a demonstration model trained on a small subset of WikiText-103. 
For better quality, train on more documents or adjust the configuration parameters.
"""
    
    with open(save_path / "README.md", 'w') as f:
        f.write(readme)
    
    logger.info(f"\n✅ Demo model saved to {save_path}")
    
    return save_path


if __name__ == "__main__":
    try:
        model_path = train_demo_model()
        
        print("\n" + "="*60)
        print("RETGEN Demo Model Training Complete!")
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