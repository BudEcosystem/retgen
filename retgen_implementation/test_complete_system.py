#!/usr/bin/env python3
"""Complete system test for RETGEN implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging
logging.basicConfig(level=logging.INFO)

def test_complete_system():
    """Test the complete RETGEN system."""
    print("=" * 60)
    print("RETGEN Complete System Test")
    print("=" * 60)
    
    try:
        # Import main components
        from core.retgen import RETGEN
        from core.config import RETGENConfig
        from training.dataset_loader import DatasetLoader
        print("✓ All imports successful")
        
        # Create configuration
        config = RETGENConfig(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
            min_pattern_frequency=1,
            retrieval_k=10,
            max_generation_length=30
        )
        print("✓ Configuration created")
        
        # Create model
        model = RETGEN(config)
        print("✓ Model created")
        
        # Create small dataset
        docs = DatasetLoader.create_sample_dataset(20)
        train_docs, val_docs, _ = DatasetLoader.split_dataset(docs)
        print(f"✓ Dataset created ({len(train_docs)} train, {len(val_docs)} val docs)")
        
        # Train model
        print("Training model...")
        metrics = model.train(train_docs, val_docs)
        print("✓ Model trained successfully")
        
        if metrics.training_time:
            print(f"  Training time: {metrics.training_time[-1]:.2f}s")
        if metrics.index_size:
            print(f"  Patterns extracted: {metrics.index_size[-1]:,}")
        
        print(f"  Model size: {model.get_size_mb():.1f} MB")
        
        # Test generation
        test_prompts = [
            "The future of",
            "Natural language",
            "Machine learning"
        ]
        
        print("\nTesting text generation:")
        for prompt in test_prompts:
            try:
                generated = model.generate(prompt, max_length=20)
                print(f"  '{prompt}' -> '{generated}'")
            except Exception as e:
                print(f"  '{prompt}' -> ERROR: {e}")
        
        # Test model save/load
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            
            print("\nTesting model save/load:")
            model.save(save_path)
            print("✓ Model saved")
            
            loaded_model = RETGEN.load(save_path)
            print("✓ Model loaded")
            
            # Test loaded model
            test_gen = loaded_model.generate("Test prompt", max_length=10)
            print(f"✓ Loaded model generates: '{test_gen}'")
        
        print("\n🎉 Complete system test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_system()
    sys.exit(0 if success else 1)