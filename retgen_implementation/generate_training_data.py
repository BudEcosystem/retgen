#!/usr/bin/env python3
"""Generate training data for RETGEN from available sources."""

import json
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def generate_training_data(output_file="training_data.json", max_samples=1000000):
    """Generate training data from WikiText-103."""
    
    texts = []
    
    print("Loading WikiText-103...")
    try:
        # Load WikiText-103
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        
        for example in tqdm(dataset, desc="Processing WikiText"):
            text = example['text'].strip()
            if len(text.split()) >= 10 and len(text.split()) <= 512:
                texts.append(text)
                if len(texts) >= max_samples:
                    break
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Shuffle
    random.shuffle(texts)
    
    # Save to file
    print(f"Saving {len(texts)} samples to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(texts, f)
    
    print(f"Successfully saved {len(texts)} training samples!")
    return texts

if __name__ == "__main__":
    generate_training_data()