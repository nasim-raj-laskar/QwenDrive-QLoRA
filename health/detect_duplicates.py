#!/usr/bin/env python3
"""
Duplicate detection utility for the automotive dataset.
"""

import sys
sys.path.append('.')

import yaml
from src.analysis import detect_exact_duplicates
from src.data import load_and_prepare
from src.model import load_tokenizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Detect and report duplicates."""
    logger.info("Starting duplicate detection...")
    
    # Load configs
    with open("configs/model.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    
    with open("configs/training.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)
    
    # Load tokenizer and dataset
    tokenizer = load_tokenizer(model_cfg["model_name"], model_cfg["trust_remote_code"])
    datasets = load_and_prepare(train_cfg["data"], tokenizer)
    
    # Detect exact duplicates
    logger.info("Detecting exact duplicates...")
    duplicates = detect_exact_duplicates(datasets["train"])
    
    print(f"\nFound {len(duplicates)} exact duplicates:")
    print("="*60)
    
    for i, dup in enumerate(duplicates[:10]):  # Show first 10
        print(f"\nDuplicate #{i+1}:")
        print(f"  Index: {dup['index']} (duplicate of {dup['duplicate_of']})")
        print(f"  Prompt: {dup['prompt']}...")
    
    if len(duplicates) > 10:
        print(f"\n... and {len(duplicates) - 10} more duplicates")
    
    # Calculate duplicate rate
    total_samples = len(datasets["train"])
    duplicate_rate = len(duplicates) / total_samples
    
    print(f"\nSummary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Exact duplicates: {len(duplicates)}")
    print(f"  Duplicate rate: {duplicate_rate:.2%}")
    
    if duplicate_rate > 0.05:  # 5% threshold
        print(f" High duplicate rate detected!")
    else:
        print(f" Duplicate rate within acceptable range")

if __name__ == "__main__":
    main()