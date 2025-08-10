#!/usr/bin/env python3
"""Test GPU setup and dependencies for RetGen training."""

import sys
import subprocess

def test_gpu_setup():
    """Test if GPU is properly set up for training."""
    
    print("=" * 60)
    print("Testing GPU Setup for RetGen Training")
    print("=" * 60)
    
    # Test PyTorch
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Test GPU computation
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            print("✓ GPU computation test passed")
        else:
            print("✗ CUDA not available - will use CPU (slower)")
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False
    
    # Test other dependencies
    dependencies = [
        'transformers',
        'datasets',
        'sentence_transformers',
        'numpy',
        'pandas',
        'tqdm',
        'sklearn',
        'psutil'
    ]
    
    print("\nChecking dependencies:")
    all_installed = True
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} installed")
        except ImportError:
            print(f"✗ {dep} not installed")
            all_installed = False
    
    # Test FAISS
    try:
        import faiss
        print(f"✓ FAISS installed")
        
        # Check if GPU version
        if hasattr(faiss, 'StandardGpuResources'):
            print("✓ FAISS GPU support available")
        else:
            print("○ FAISS CPU version (GPU version not available)")
    except ImportError:
        print("✗ FAISS not installed")
        all_installed = False
    
    print("\n" + "=" * 60)
    
    if all_installed and torch.cuda.is_available():
        print("✓ System ready for GPU-accelerated RetGen training!")
        return True
    elif all_installed:
        print("○ System ready for RetGen training (CPU mode)")
        return True
    else:
        print("✗ Some dependencies missing. Please install them first.")
        return False


if __name__ == "__main__":
    success = test_gpu_setup()
    sys.exit(0 if success else 1)