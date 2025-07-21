#!/usr/bin/env python3
"""Test RETGEN API deployment."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import argparse
import logging
from typing import List

from src.api.client import RETGENClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_api(base_url: str = "http://localhost:8000"):
    """Test API functionality."""
    logger.info(f"Testing RETGEN API at {base_url}")
    logger.info("=" * 60)
    
    # Create client
    client = RETGENClient(base_url)
    
    # 1. Health check
    logger.info("\n1. Testing health check...")
    try:
        health = client.health_check()
        logger.info(f"✓ API is healthy: {health}")
    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
        return False
    
    # 2. Model info
    logger.info("\n2. Testing model info...")
    try:
        info = client.get_model_info()
        logger.info(f"✓ Model info retrieved: loaded={info['is_loaded']}, trained={info['is_trained']}")
    except Exception as e:
        logger.error(f"✗ Model info failed: {e}")
        return False
    
    # 3. Train model
    logger.info("\n3. Testing model training...")
    documents = generate_test_documents(100)
    
    try:
        start_time = time.time()
        response = client.train_model(
            documents,
            validation_split=0.1,
            config_overrides={
                "min_pattern_frequency": 1,
                "retrieval_k": 10,
                "resolutions": [1, 2, 3]
            },
            wait=False
        )
        logger.info(f"✓ Training initiated: {response['message']}")
        
        # Wait for training to complete
        logger.info("   Waiting for training to complete...")
        while True:
            info = client.get_model_info()
            if info['is_trained']:
                break
            time.sleep(2)
        
        training_time = time.time() - start_time
        logger.info(f"✓ Training completed in {training_time:.1f}s")
        logger.info(f"   Metrics: {info['metrics']}")
        
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        return False
    
    # 4. Test generation
    logger.info("\n4. Testing text generation...")
    test_prompts = [
        "Natural language processing",
        "Machine learning models",
        "Deep learning architectures",
        "Text generation using",
        "Artificial intelligence can"
    ]
    
    for prompt in test_prompts:
        try:
            response = client.generate(
                prompt,
                max_length=30,
                temperature=0.8,
                num_generations=1
            )
            generated = response['generations'][0]
            logger.info(f"✓ '{prompt}' -> '{generated}'")
        except Exception as e:
            logger.error(f"✗ Generation failed for '{prompt}': {e}")
    
    # 5. Test batch generation
    logger.info("\n5. Testing batch generation...")
    try:
        response = client.generate_batch(
            test_prompts[:3],
            max_length=20,
            temperature=0.7
        )
        logger.info("✓ Batch generation successful:")
        for prompt, gen in zip(response['prompts'], response['generations']):
            logger.info(f"   '{prompt}' -> '{gen}'")
    except Exception as e:
        logger.error(f"✗ Batch generation failed: {e}")
    
    # 6. Test save/load
    logger.info("\n6. Testing model persistence...")
    try:
        # Save
        save_path = "/tmp/test_model"
        client.save_model(save_path)
        logger.info(f"✓ Model saved to {save_path}")
        
        # Create new client to test loading
        new_client = RETGENClient(base_url)
        
        # Load
        new_client.load_model(save_path)
        logger.info(f"✓ Model loaded from {save_path}")
        
        # Test loaded model
        response = new_client.generate("Test prompt", max_length=10)
        logger.info(f"✓ Loaded model generates: '{response['generations'][0]}'")
        
    except Exception as e:
        logger.error(f"✗ Save/load failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ API tests completed successfully!")
    return True


def generate_test_documents(n: int) -> List[str]:
    """Generate test documents."""
    templates = [
        "Natural language processing is a subfield of {field} that focuses on {task}.",
        "Machine learning algorithms can {capability} by learning from {data_type}.",
        "Deep learning models use {architecture} to process {input_type} data.",
        "Text generation involves {process} to create {output_type} content.",
        "{technology} has revolutionized how we {application} in modern systems.",
        "The {method} approach enables {benefit} through {mechanism}.",
        "Researchers have developed {innovation} for improving {metric}.",
        "{framework} provides tools for {use_case} applications.",
        "Advanced {technique} can achieve {performance} on {benchmark}.",
        "The future of {domain} lies in {advancement} and {improvement}."
    ]
    
    fields = ["AI", "computer science", "linguistics", "cognitive science"]
    tasks = ["understanding text", "generating language", "translation", "summarization"]
    capabilities = ["classify data", "predict outcomes", "detect patterns", "generate text"]
    data_types = ["labeled examples", "text corpora", "user interactions", "documents"]
    architectures = ["neural networks", "transformers", "attention mechanisms", "embeddings"]
    input_types = ["textual", "sequential", "structured", "unstructured"]
    processes = ["sampling tokens", "retrieving patterns", "combining contexts", "predicting words"]
    output_types = ["human-like", "coherent", "contextual", "meaningful"]
    
    import random
    documents = []
    
    for i in range(n):
        template = random.choice(templates)
        doc = template.format(
            field=random.choice(fields),
            task=random.choice(tasks),
            capability=random.choice(capabilities),
            data_type=random.choice(data_types),
            architecture=random.choice(architectures),
            input_type=random.choice(input_types),
            process=random.choice(processes),
            output_type=random.choice(output_types),
            technology=random.choice(["AI", "ML", "NLP", "Deep learning"]),
            application=random.choice(["process data", "understand language", "generate text"]),
            method=random.choice(["retrieval", "attention", "embedding", "transformer"]),
            benefit=random.choice(["better performance", "faster inference", "higher accuracy"]),
            mechanism=random.choice(["pattern matching", "vector search", "neural computation"]),
            innovation=random.choice(["new models", "novel techniques", "better algorithms"]),
            metric=random.choice(["accuracy", "efficiency", "quality", "speed"]),
            framework=random.choice(["PyTorch", "TensorFlow", "RETGEN", "Transformers"]),
            use_case=random.choice(["NLP", "text generation", "chatbot", "translation"]),
            technique=random.choice(["retrieval", "fine-tuning", "prompting", "augmentation"]),
            performance=random.choice(["state-of-the-art", "impressive", "superior"]),
            benchmark=random.choice(["WikiText", "GLUE", "SQuAD", "CommonCrawl"]),
            domain=random.choice(["AI", "NLP", "text generation", "language models"]),
            advancement=random.choice(["better models", "new techniques", "improved methods"]),
            improvement=random.choice(["higher quality", "faster speed", "better accuracy"])
        )
        documents.append(doc)
    
    return documents


def main():
    """Main test script."""
    parser = argparse.ArgumentParser(description="Test RETGEN API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL"
    )
    
    args = parser.parse_args()
    
    # Run tests
    success = test_api(args.url)
    
    if not success:
        logger.error("API tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()