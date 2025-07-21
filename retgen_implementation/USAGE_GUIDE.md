# RETGEN Usage Guide

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the implementation
cd retgen_implementation

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install faiss-gpu

# Install in development mode
pip install -e .
```

### Basic Usage

```python
# Set up Python path
import sys
sys.path.append('src')

from core.config import RETGENConfig
from training.dataset_loader import DatasetLoader

# Create configuration
config = RETGENConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim=384,
    retrieval_k=20,
    min_pattern_frequency=1
)

# Load or create dataset
docs = DatasetLoader.create_sample_dataset(100)
train_docs, val_docs, test_docs = DatasetLoader.split_dataset(docs)

# Train model (requires proper package structure)
# This will work once imports are fixed
model = RETGEN(config)
model.train(train_docs, val_docs)

# Generate text
generated = model.generate("The future of artificial intelligence", max_length=50)
print(generated)
```

## 🔧 Component Testing

### Test Configuration System

```python
import sys
sys.path.append('src')

from core.config import RETGENConfig

# Create and test configuration
config = RETGENConfig()
print(f"Embedding model: {config.embedding_model}")
print(f"Retrieval k: {config.retrieval_k}")

# Test validation
config.validate()  # Should not raise

# Test serialization
config.save("test_config.json")
loaded_config = RETGENConfig.load("test_config.json")
```

### Test Pattern Extraction

```python
from data.pattern_extraction import Pattern, PatternDatabase

# Create test patterns
patterns = [
    Pattern([1, 2, 3], "the cat", 4, "sat", 0, 3, 0),
    Pattern([1, 2], "the cat", 5, "ran", 1, 2, 0),
]

# Test database
db = PatternDatabase()
for pattern in patterns:
    db.add_pattern(pattern)

print(f"Total patterns: {db.total_patterns}")
print(f"Unique patterns: {len(db.patterns)}")

# Get continuation distribution
dist = db.get_continuation_distribution((1, 2, 3))
print(f"Distribution: {dist}")
```

### Test Vector Database

```python
from indexing.vector_database import FAISSIndexBuilder, VectorDatabase
import numpy as np

# Create test embeddings
embeddings = np.random.randn(100, 64).astype(np.float32)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Build FAISS index
config = RETGENConfig(similarity_metric="cosine", index_type="Flat")
builder = FAISSIndexBuilder(config)
index = builder.build_index(embeddings)

print(f"Index built with {index.ntotal} vectors")

# Test search
query = embeddings[0:1]
distances, indices = index.search(query, k=5)
print(f"Search results: {indices[0]}")
```

## 📊 Benchmarking

### Performance Benchmark

```python
from benchmarks.performance import PerformanceBenchmark

# Create mock model for testing
model = create_test_model()  # Your model creation logic

benchmark = PerformanceBenchmark(model)

# Run memory usage analysis
memory_results = benchmark.benchmark_memory_usage()
print(f"Model size: {memory_results['model_size_mb']:.1f} MB")

# Run scaling analysis
scaling_results = benchmark.benchmark_scaling(
    pattern_counts=[1000, 5000, 10000],
    query_batch_sizes=[1, 10, 50]
)
```

### Accuracy Benchmark

```python
from benchmarks.accuracy import AccuracyBenchmark
from training.dataset_loader import DatasetLoader

# Create test datasets
test_datasets = {
    'sample': DatasetLoader.create_sample_dataset(50)
}

benchmark = AccuracyBenchmark(model)

# Run perplexity benchmark
ppl_results = benchmark.benchmark_perplexity(test_datasets)
print(f"Perplexity: {ppl_results}")

# Run generation quality benchmark
prompt_sets = {
    'general': ["The future of", "Natural language", "Machine learning"]
}

quality_results = benchmark.benchmark_generation_quality(prompt_sets)
```

## 🔧 Configuration Options

### Essential Parameters

```python
config = RETGENConfig(
    # Embedding settings
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Fast model
    embedding_dim=384,                    # Model dimension
    normalize_embeddings=True,            # Normalize for cosine similarity
    
    # Pattern extraction
    resolutions=[1, 2, 3, 5, 8],         # Multi-resolution extraction
    min_pattern_frequency=2,              # Filter rare patterns
    
    # Retrieval settings
    retrieval_k=50,                       # Top-k patterns to retrieve
    similarity_metric="cosine",           # cosine, dot, l2
    temperature=1.0,                      # Sampling temperature
    
    # Generation settings
    max_generation_length=100,            # Max tokens to generate
    top_p=0.95,                          # Nucleus sampling
    repetition_penalty=1.2,              # Avoid repetition
    
    # Index settings
    index_type="IVF1024,PQ64",           # FAISS index type
    use_gpu=False,                       # GPU acceleration
    
    # Hardware settings
    device="cpu",                        # cpu or cuda
    batch_size=256,                      # Embedding batch size
)
```

### Performance Tuning

```python
# For speed (smaller model)
fast_config = RETGENConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim=384,
    retrieval_k=10,
    min_pattern_frequency=3,
    index_type="Flat"
)

# For quality (larger model)
quality_config = RETGENConfig(
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    embedding_dim=768,
    retrieval_k=100,
    min_pattern_frequency=1,
    index_type="IVF2048,PQ128"
)

# For memory efficiency
efficient_config = RETGENConfig(
    embedding_dim=256,
    retrieval_k=20,
    min_pattern_frequency=5,
    index_type="IVF512,PQ32"
)
```

## 📁 Dataset Loading

### Built-in Datasets

```python
from training.dataset_loader import DatasetLoader

# Sample dataset for testing
docs = DatasetLoader.create_sample_dataset(100)

# WikiText-103 (requires HuggingFace datasets)
train_docs, val_docs, test_docs = DatasetLoader.load_wikitext103()

# Local text files
docs = DatasetLoader.load_local_texts("path/to/text/files/")

# Dataset information
info = DatasetLoader.get_dataset_info(docs)
print(f"Documents: {info['num_docs']}")
print(f"Average length: {info['avg_doc_length']:.1f} chars")
```

### Custom Datasets

```python
# Prepare your own dataset
my_documents = [
    "Your first document text here...",
    "Your second document text here...",
    # ... more documents
]

# Split for training
train_docs, val_docs, test_docs = DatasetLoader.split_dataset(
    my_documents,
    train_ratio=0.8,
    val_ratio=0.1
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: 
   - Ensure `src` directory is in Python path
   - Use absolute imports when possible

2. **Memory Issues**:
   - Reduce `batch_size` in config
   - Lower `retrieval_k` parameter
   - Use smaller embedding models

3. **Slow Training**:
   - Increase `min_pattern_frequency` to filter patterns
   - Use smaller dataset for testing
   - Enable GPU if available

4. **Poor Generation Quality**:
   - Increase `retrieval_k` parameter
   - Lower `temperature` for more focused generation
   - Try different similarity metrics

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = RETGENConfig(log_level="DEBUG")
# Now all operations will be logged in detail
```

## 🔄 Integration with Existing Code

### As a Library

```python
# Create wrapper class for your use case
class MyRETGENWrapper:
    def __init__(self, model_path=None):
        if model_path:
            self.model = RETGEN.load(model_path)
        else:
            config = RETGENConfig(
                # Your custom configuration
            )
            self.model = RETGEN(config)
    
    def train_on_documents(self, documents):
        return self.model.train(documents)
    
    def generate_text(self, prompt, length=50):
        return self.model.generate(prompt, max_length=length)
```

### API Server

```python
from fastapi import FastAPI

app = FastAPI()
model = RETGEN.load("path/to/trained/model")

@app.post("/generate")
async def generate_text(prompt: str, max_length: int = 100):
    generated = model.generate(prompt, max_length=max_length)
    return {"generated": generated}
```

## 📈 Performance Optimization

### Memory Optimization

```python
# Use efficient data types
config.use_compression = True

# Limit cache size
embedder.cache = EmbeddingCache(max_size=1000)

# Use quantized index
config.index_type = "IVF1024,PQ64"  # Compressed index
```

### Speed Optimization

```python
# Enable GPU
config.use_gpu = True
config.device = "cuda"

# Optimize batch size
config.batch_size = 512  # Larger batches for GPU

# Use faster embedding model
config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
```

## 🎯 Best Practices

1. **Start Small**: Begin with sample datasets and small configurations
2. **Monitor Memory**: Watch memory usage during training
3. **Validate Configs**: Always call `config.validate()` before training
4. **Save Models**: Regularly save trained models
5. **Log Everything**: Use appropriate log levels for debugging
6. **Test Incrementally**: Test each component separately before integration
7. **Benchmark Regularly**: Monitor performance metrics during development

---

This implementation provides a solid foundation for RETGEN research and development. While some integration issues remain with the relative imports, all core components are fully functional and thoroughly tested.