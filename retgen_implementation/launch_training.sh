#!/bin/bash
# Launch script for large-scale RetGen training with GPU acceleration

echo "=========================================="
echo "RetGen Large-Scale Training Launcher"
echo "=========================================="

# Set environment variables for optimal GPU performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=8

# Create necessary directories
mkdir -p models/checkpoints
mkdir -p models/stats
mkdir -p data/large_scale_cache
mkdir -p logs

# Training configuration
DATASET="both"                # Use both C4 and Wikipedia
MAX_SAMPLES=1000000           # Train on 1 million samples
BATCH_SIZE=128                # Optimal batch size for RTX 3080
CHECKPOINT_INTERVAL=50000     # Save checkpoint every 50k samples

# Model configuration
EMBEDDING_DIM=768             # Standard BERT dimension
RESOLUTIONS="1 2 3 5 8"       # Multi-resolution pattern extraction
RETRIEVAL_K=100               # Number of patterns to retrieve

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Max samples: $MAX_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  GPU: NVIDIA GeForce RTX 3080"
echo "  CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 test_gpu_setup.py

if [ $? -ne 0 ]; then
    echo "WARNING: Some dependencies might be missing"
    echo "Attempting to continue anyway..."
fi

echo ""
echo "Starting training..."
echo "Logs will be saved to: training_gpu.log"
echo ""

# Launch training with optimized settings
python3 train_large_scale_gpu.py \
    --dataset $DATASET \
    --max-samples $MAX_SAMPLES \
    --batch-size $BATCH_SIZE \
    --embedding-dim $EMBEDDING_DIM \
    --resolutions $RESOLUTIONS \
    --retrieval-k $RETRIEVAL_K \
    --checkpoint-interval $CHECKPOINT_INTERVAL \
    --num-workers 4 \
    --gpu-id 0 \
    2>&1 | tee -a logs/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=========================================="
echo "Training completed!"
echo "Check models/ directory for saved models"
echo "=========================================="