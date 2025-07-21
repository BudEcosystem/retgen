#!/usr/bin/env python3
"""Training script for RETGEN models."""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.retgen import RETGEN
from core.config import RETGENConfig
from training.dataset_loader import DatasetLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RETGEN model")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="sample",
                        choices=["sample", "wikitext103", "local"],
                        help="Dataset to use for training")
    parser.add_argument("--data-path", type=str,
                        help="Path to local data directory (for local dataset)")
    parser.add_argument("--num-docs", type=int, default=1000,
                        help="Number of documents to use (for sample/limited datasets)")
    
    # Model arguments
    parser.add_argument("--embedding-model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model to use")
    parser.add_argument("--embedding-dim", type=int, default=384,
                        help="Embedding dimension")
    parser.add_argument("--retrieval-k", type=int, default=50,
                        help="Number of patterns to retrieve")
    parser.add_argument("--min-pattern-freq", type=int, default=2,
                        help="Minimum pattern frequency")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for embedding computation")
    parser.add_argument("--validation-split", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./trained_model",
                        help="Directory to save trained model")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the trained model")
    
    # Other arguments
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to use for computation")
    
    return parser.parse_args()


def load_dataset(args):
    """Load dataset based on arguments."""
    if args.dataset == "sample":
        print(f"Creating sample dataset with {args.num_docs} documents...")
        docs = DatasetLoader.create_sample_dataset(args.num_docs)
        train_docs, val_docs, test_docs = DatasetLoader.split_dataset(
            docs, val_ratio=args.validation_split
        )
        
    elif args.dataset == "wikitext103":
        print("Loading WikiText-103 dataset...")
        train_docs, val_docs, test_docs = DatasetLoader.load_wikitext103()
        
        # Limit documents if specified
        if args.num_docs < len(train_docs):
            train_docs = train_docs[:args.num_docs]
            val_docs = val_docs[:args.num_docs // 10]
            test_docs = test_docs[:args.num_docs // 10]
    
    elif args.dataset == "local":
        if not args.data_path:
            raise ValueError("--data-path required for local dataset")
        
        print(f"Loading local dataset from {args.data_path}...")
        docs = DatasetLoader.load_local_texts(Path(args.data_path))
        
        if len(docs) == 0:
            raise ValueError(f"No documents found in {args.data_path}")
        
        train_docs, val_docs, test_docs = DatasetLoader.split_dataset(
            docs, val_ratio=args.validation_split
        )
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return train_docs, val_docs, test_docs


def create_config(args):
    """Create RETGEN configuration from arguments."""
    config = RETGENConfig(
        # Embedding settings
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
        device=args.device,
        
        # Pattern settings
        min_pattern_frequency=args.min_pattern_freq,
        retrieval_k=args.retrieval_k,
        
        # Training settings
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        
        # Logging
        log_level=args.log_level
    )
    
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting RETGEN training")
    
    try:
        # Load dataset
        train_docs, val_docs, test_docs = load_dataset(args)
        
        # Print dataset info
        train_info = DatasetLoader.get_dataset_info(train_docs)
        val_info = DatasetLoader.get_dataset_info(val_docs)
        
        print(f"\nDataset Statistics:")
        print(f"Training set: {train_info['num_docs']} docs, "
              f"avg length: {train_info['avg_doc_length']:.1f} chars")
        print(f"Validation set: {val_info['num_docs']} docs, "
              f"avg length: {val_info['avg_doc_length']:.1f} chars")
        
        # Create configuration
        config = create_config(args)
        
        print(f"\nModel Configuration:")
        print(f"Embedding model: {config.embedding_model}")
        print(f"Embedding dimension: {config.embedding_dim}")
        print(f"Retrieval k: {config.retrieval_k}")
        print(f"Min pattern frequency: {config.min_pattern_frequency}")
        print(f"Batch size: {config.batch_size}")
        
        # Create and train model
        print(f"\nCreating RETGEN model...")
        model = RETGEN(config)
        
        print(f"Training model...")
        save_path = Path(args.output_dir) if args.save_model else None
        
        metrics = model.train(
            corpus=train_docs,
            validation_corpus=val_docs,
            save_path=save_path
        )
        
        # Print training results
        print(f"\n🎉 Training completed successfully!")
        
        if metrics.training_time:
            training_time = metrics.training_time[-1]
            print(f"Training time: {training_time:.2f} seconds")
        
        if metrics.index_size:
            pattern_count = metrics.index_size[-1]
            print(f"Patterns extracted: {pattern_count:,}")
        
        model_size = model.get_size_mb()
        print(f"Model size: {model_size:.1f} MB")
        
        if metrics.coverage:
            coverage = metrics.coverage[-1]
            print(f"Validation coverage: {coverage:.3f}")
        
        if metrics.perplexity:
            perplexity = metrics.perplexity[-1]
            print(f"Validation perplexity: {perplexity:.2f}")
        
        # Test generation
        print(f"\nTesting generation:")
        test_prompts = [
            "The future of artificial intelligence",
            "Natural language processing is",
            "Machine learning algorithms"
        ]
        
        for prompt in test_prompts:
            try:
                generated = model.generate(prompt, max_length=50)
                print(f"  '{prompt}' -> '{generated}'")
            except Exception as e:
                print(f"  '{prompt}' -> ERROR: {e}")
        
        if args.save_model:
            print(f"\n💾 Model saved to: {args.output_dir}")
            print(f"To load: model = RETGEN.load('{args.output_dir}')")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()