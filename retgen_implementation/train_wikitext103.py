#!/usr/bin/env python3
"""Train RETGEN on full WikiText-103 dataset."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import os
import time
import logging
import json
from datetime import datetime
import argparse

from datasets import load_dataset
from tqdm import tqdm

from src.core.config import RETGENConfig
from run_retgen import RETGENSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WikiTextTrainer:
    """Train RETGEN on WikiText-103."""
    
    def __init__(self, config: RETGENConfig = None):
        """Initialize trainer."""
        self.config = config or RETGENConfig(
            # Model configuration
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            embedding_dim=768,
            
            # Pattern extraction
            min_pattern_frequency=2,
            max_pattern_length=10,
            resolutions=[1, 2, 3, 5, 8],
            
            # Retrieval settings
            retrieval_k=100,
            
            # Training settings
            batch_size=512,
            
            # Index configuration
            index_type="Flat",  # Use Flat index for small datasets
            nprobe=10,
            
            # Performance settings
            device="cpu",
            use_gpu=False,
            num_workers=0,  # Disable multiprocessing to avoid issues
            
            # Generation settings
            max_generation_length=200,
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )
        
        self.retgen = RETGENSystem(self.config)
        self.dataset_name = "wikitext-103-v1"
        self.max_documents = None
    
    def load_wikitext103(self):
        """Load WikiText-103 dataset."""
        logger.info("Loading WikiText-103 dataset...")
        
        try:
            # Load dataset
            dataset = load_dataset("wikitext", "wikitext-103-v1")
            
            # Extract documents
            train_texts = []
            val_texts = []
            test_texts = []
            
            # Process training set
            logger.info("Processing training documents...")
            for item in tqdm(dataset['train'], desc="Train docs"):
                text = item['text'].strip()
                if len(text) > 50:  # Filter short texts
                    train_texts.append(text)
            
            # Process validation set
            logger.info("Processing validation documents...")
            for item in tqdm(dataset['validation'], desc="Val docs"):
                text = item['text'].strip()
                if len(text) > 50:
                    val_texts.append(text)
            
            # Process test set
            logger.info("Processing test documents...")
            for item in tqdm(dataset['test'], desc="Test docs"):
                text = item['text'].strip()
                if len(text) > 50:
                    test_texts.append(text)
            
            logger.info(f"Loaded {len(train_texts):,} training documents")
            logger.info(f"Loaded {len(val_texts):,} validation documents")
            logger.info(f"Loaded {len(test_texts):,} test documents")
            
            return train_texts, val_texts, test_texts
            
        except Exception as e:
            logger.error(f"Failed to load WikiText-103: {e}")
            logger.info("Attempting to download...")
            
            # Try downloading with different method
            import subprocess
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "datasets"
            ])
            
            # Retry
            dataset = load_dataset("wikitext", "wikitext-103-v1")
            return self._process_dataset(dataset)
    
    def _process_dataset(self, dataset):
        """Process raw dataset."""
        train_texts = []
        val_texts = []
        test_texts = []
        
        for split, texts in [
            ('train', train_texts),
            ('validation', val_texts),
            ('test', test_texts)
        ]:
            for item in dataset[split]:
                text = item['text'].strip()
                if len(text) > 50:
                    texts.append(text)
        
        return train_texts, val_texts, test_texts
    
    def train(self, save_path: Path):
        """Train RETGEN on WikiText-103."""
        logger.info("Starting WikiText-103 training...")
        logger.info(f"Configuration: {self.config}")
        
        # Create save directory
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.config.save(save_path / "config.json")
        
        # Load dataset
        train_docs, val_docs, test_docs = self.load_wikitext103()
        
        # Limit documents if specified
        if self.max_documents:
            train_docs = train_docs[:self.max_documents]
            logger.info(f"Limited to {len(train_docs)} training documents")
        
        # Training metrics
        training_start = time.time()
        
        # Train model
        logger.info("\nStarting RETGEN training...")
        metrics = self.retgen.train(train_docs, val_docs)
        
        training_time = time.time() - training_start
        
        # Update metrics
        metrics['dataset'] = self.dataset_name
        metrics['train_docs'] = len(train_docs)
        metrics['val_docs'] = len(val_docs)
        metrics['test_docs'] = len(test_docs)
        metrics['total_training_time'] = training_time
        
        # Log results
        logger.info("\nTraining completed!")
        logger.info(f"Total time: {training_time/60:.1f} minutes")
        logger.info(f"Patterns extracted: {metrics['pattern_count']:,}")
        logger.info(f"Model size: {metrics['model_size_mb']:.1f} MB")
        
        # Save model
        logger.info(f"\nSaving model to {save_path}...")
        self.retgen.save(save_path)
        
        # Save extended metrics
        with open(save_path / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Test generation
        logger.info("\nTesting generation on sample prompts...")
        test_prompts = [
            "The history of artificial intelligence",
            "Natural language processing is",
            "Machine learning algorithms",
            "Deep neural networks have",
            "The future of technology"
        ]
        
        generations = []
        for prompt in test_prompts:
            generated = self.retgen.generate(prompt, max_length=50)
            generations.append({
                "prompt": prompt,
                "generated": generated
            })
            logger.info(f"'{prompt}' -> '{generated}'")
        
        # Save generation samples
        with open(save_path / "generation_samples.json", 'w') as f:
            json.dump(generations, f, indent=2)
        
        # Create model card
        self._create_model_card(save_path, metrics, generations)
        
        logger.info(f"\n✅ Model successfully trained and saved to {save_path}")
        
        return metrics
    
    def _create_model_card(self, save_path: Path, metrics: dict, generations: list):
        """Create model card with training details."""
        model_card = f"""# RETGEN WikiText-103 Model

## Model Details

- **Model Type**: RETGEN (Retrieval-Enhanced Text Generation)
- **Training Dataset**: WikiText-103
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Embedding Model**: {self.config.embedding_model}
- **Embedding Dimension**: {self.config.embedding_dim}

## Training Configuration

```json
{json.dumps(self.config.__dict__, indent=2)}
```

## Training Metrics

- **Training Documents**: {metrics['train_docs']:,}
- **Validation Documents**: {metrics.get('val_docs', 0):,}
- **Total Patterns**: {metrics['pattern_count']:,}
- **Model Size**: {metrics['model_size_mb']:.1f} MB
- **Training Time**: {metrics['total_training_time']/60:.1f} minutes

## Pattern Extraction Details

- **Resolutions**: {self.config.resolutions}
- **Min Pattern Frequency**: {self.config.min_pattern_frequency}
- **Max Pattern Length**: {self.config.max_pattern_length}

## Retrieval Configuration

- **Retrieval K**: {self.config.retrieval_k}
- **Similarity Metric**: {self.config.similarity_metric}
- **Index Type**: {self.config.index_type}

## Generation Samples

"""
        
        for gen in generations[:5]:
            model_card += f"**Prompt**: {gen['prompt']}\n"
            model_card += f"**Generated**: {gen['generated']}\n\n"
        
        model_card += """
## Usage

```python
from run_retgen import RETGENSystem

# Load model
retgen = RETGENSystem.load('path/to/model')

# Generate text
generated = retgen.generate(
    "Your prompt here",
    max_length=100,
    temperature=0.8
)
print(generated)
```

## Citation

If you use this model, please cite the RETGEN paper:
```
@article{retgen2024,
  title={RETGEN: Retrieval-Enhanced Text Generation},
  author={...},
  year={2024}
}
```
"""
        
        with open(save_path / "MODEL_CARD.md", 'w') as f:
            f.write(model_card)


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train RETGEN on WikiText-103")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("models/retgen_wikitext103"),
        help="Path to save trained model"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to use (for testing)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence transformer model to use"
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=100,
        help="Number of patterns to retrieve"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for embedding computation"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = RETGENConfig(
        embedding_model=args.embedding_model,
        retrieval_k=args.retrieval_k,
        batch_size=args.batch_size,
        index_type="Flat",  # Use Flat for small datasets
        num_workers=0  # Avoid multiprocessing issues
    )
    
    # Create trainer
    trainer = WikiTextTrainer(config)
    
    # Set max documents separately
    if args.max_docs:
        trainer.max_documents = args.max_docs
    
    # Train model
    try:
        metrics = trainer.train(args.save_path)
        
        # Print summary
        print("\n" + "="*60)
        print("RETGEN WikiText-103 Training Complete!")
        print("="*60)
        print(f"Model saved to: {args.save_path}")
        print(f"Total patterns: {metrics['pattern_count']:,}")
        print(f"Model size: {metrics['model_size_mb']:.1f} MB")
        print(f"Training time: {metrics['total_training_time']/60:.1f} minutes")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()