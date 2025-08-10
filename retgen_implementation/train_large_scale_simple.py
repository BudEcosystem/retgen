"""
Simplified large-scale training script using WikiText and OpenWebText.
"""

import os
import sys
import time
import logging
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from datasets import load_dataset
from src.core.config import RETGENConfig
from src.core.retgen import RETGEN

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_large_dataset(max_samples=1000000):
    """Load WikiText-103 and sample from OpenWebText."""
    logger.info("Loading datasets...")
    
    all_texts = []
    
    # Load WikiText-103
    logger.info("Loading WikiText-103...")
    wikitext = load_dataset("wikitext", "wikitext-103-v1", split="train")
    
    # Filter and clean WikiText
    for example in tqdm(wikitext, desc="Processing WikiText"):
        text = example['text'].strip()
        if len(text.split()) >= 10:  # Min 10 words
            all_texts.append(text)
            if len(all_texts) >= max_samples // 2:
                break
    
    logger.info(f"Loaded {len(all_texts)} samples from WikiText-103")
    
    # Load OpenWebText samples
    if len(all_texts) < max_samples:
        logger.info("Loading OpenWebText samples...")
        try:
            # Load a subset of OpenWebText
            openwebtext = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
            
            count = 0
            for example in tqdm(openwebtext, desc="Processing OpenWebText", total=max_samples - len(all_texts)):
                text = example['text'].strip()
                if len(text.split()) >= 10 and len(text.split()) <= 512:
                    all_texts.append(text)
                    count += 1
                    if len(all_texts) >= max_samples:
                        break
            
            logger.info(f"Loaded {count} samples from OpenWebText")
        except Exception as e:
            logger.warning(f"Could not load OpenWebText: {e}")
    
    logger.info(f"Total samples loaded: {len(all_texts)}")
    return all_texts[:max_samples]


def train_retgen_gpu(texts, config, batch_size=32, checkpoint_interval=10000):
    """Train RetGen on GPU with batched processing."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Enable GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Initialize model
    model = RETGEN(config)
    
    # Training loop
    start_time = time.time()
    total_patterns = 0
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Training batches"):
        batch = texts[i:i+batch_size]
        
        # Process batch
        batch_patterns = 0
        for text in batch:
            try:
                patterns = model.pattern_extractor.extract_patterns(text)
                batch_patterns += len(patterns)
                
                # Add patterns to model
                for pattern in patterns:
                    model.add_pattern(pattern)
            except Exception as e:
                logger.warning(f"Error processing text: {e}")
                continue
        
        total_patterns += batch_patterns
        
        # Checkpoint
        if (i + batch_size) % checkpoint_interval == 0:
            checkpoint_path = f"models/checkpoints/retgen_checkpoint_{i+batch_size}.pkl"
            logger.info(f"Saving checkpoint at {i+batch_size} samples...")
            try:
                model.save(checkpoint_path)
                logger.info(f"Checkpoint saved. Total patterns: {total_patterns}")
            except Exception as e:
                logger.warning(f"Could not save checkpoint: {e}")
        
        # Clear GPU cache periodically
        if device.type == 'cuda' and i % 1000 == 0:
            torch.cuda.empty_cache()
    
    # Save final model
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed/3600:.2f} hours")
    logger.info(f"Total patterns extracted: {total_patterns}")
    
    final_path = f"models/retgen_large_scale_final.pkl"
    try:
        model.save(final_path)
        logger.info(f"Final model saved to {final_path}")
    except Exception as e:
        logger.warning(f"Could not save final model: {e}")
    
    return model


def main():
    """Main training function."""
    # Configuration
    MAX_SAMPLES = 1000000
    BATCH_SIZE = 64
    CHECKPOINT_INTERVAL = 50000
    
    # Create directories
    os.makedirs("models/checkpoints", exist_ok=True)
    
    # Setup RetGen config
    config = RETGENConfig(
        embedding_dim=768,
        resolutions=[1, 2, 3, 5, 8],
        retrieval_k=100,
        min_pattern_frequency=2,
        index_type='Flat',  # Start with Flat index for simplicity
        use_gpu=torch.cuda.is_available(),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    logger.info("=" * 80)
    logger.info("Starting Large-Scale RetGen Training")
    logger.info(f"Target samples: {MAX_SAMPLES:,}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Device: {config.device}")
    logger.info("=" * 80)
    
    # Load datasets
    texts = load_large_dataset(MAX_SAMPLES)
    
    # Train model
    model = train_retgen_gpu(
        texts, 
        config, 
        batch_size=BATCH_SIZE,
        checkpoint_interval=CHECKPOINT_INTERVAL
    )
    
    # Test generation
    logger.info("\nTesting generation...")
    test_prompt = "The future of artificial intelligence"
    try:
        generated = model.generate(test_prompt, max_length=50)
        logger.info(f"Prompt: {test_prompt}")
        logger.info(f"Generated: {generated}")
    except Exception as e:
        logger.warning(f"Could not test generation: {e}")
    
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()