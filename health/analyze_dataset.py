#!/usr/bin/env python3
"""
Standalone dataset analysis script.
Run comprehensive analysis on the automotive dataset without training.
"""

import sys
sys.path.append('.')

import yaml
from src.analysis import DatasetAnalyzer, DatasetVersioner, detect_exact_duplicates
from src.model import load_tokenizer
from src.data import load_and_prepare
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Run dataset analysis."""
    logger.info("Starting standalone dataset analysis...")
    
    # Load configs
    with open("configs/model.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    
    with open("configs/training.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)
    
    # Load tokenizer only (no model needed for analysis)
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(model_cfg["model_name"], model_cfg["trust_remote_code"])
    
    # Load and prepare dataset
    logger.info("Loading dataset...")
    datasets = load_and_prepare(train_cfg["data"], tokenizer)
    
    # Run comprehensive analysis
    logger.info("Running dataset analysis...")
    analyzer = DatasetAnalyzer(
        dataset=datasets["train"],
        tokenizer=tokenizer,
        output_dir="output/data_analysis"
    )
    
    analysis_stats = analyzer.analyze_all()
    
    # Create dataset version
    logger.info("Creating dataset version...")
    versioner = DatasetVersioner()
    version_id, version_metadata = versioner.create_version(
        dataset=datasets["train"],
        config=train_cfg["data"],
        preprocessing_steps=[
            "load_jsonl",
            f"shuffle_seed_{train_cfg['data']['shuffle_seed']}",
            f"sample_{train_cfg['data']['sample_size']}",
            "format_chat_template",
            f"filter_length_{train_cfg['data']['min_seq_length']}_{train_cfg['data']['max_seq_length']}"
        ]
    )
    
    # Detect exact duplicates
    logger.info("Detecting exact duplicates...")
    duplicates = detect_exact_duplicates(datasets["train"])
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET ANALYSIS SUMMARY")
    print("="*60)
    print(f"Dataset Version: {version_id}")
    print(f"Total Samples: {analysis_stats['basic']['total_samples']}")
    print(f"Unique Prompts: {analysis_stats['basic']['unique_prompts']}")
    print(f"Mean Quality Score: {analysis_stats['quality_scores']['mean_score']:.1f}/100")
    print(f"Duplicate Rate: {analysis_stats['duplicates']['duplicate_rate']:.2%}")
    print(f"Low Quality Samples: {analysis_stats['quality_scores']['low_quality_count']}")
    print(f"Exact Duplicates Found: {len(duplicates)}")
    
    token_dist = analysis_stats['token_distribution']
    print(f"\nToken Statistics:")
    print(f"  Avg Prompt Tokens: {token_dist['prompt_tokens']['mean']:.1f}")
    print(f"  Avg Response Tokens: {token_dist['response_tokens']['mean']:.1f}")
    print(f"  Avg Total Tokens: {token_dist['total_tokens']['mean']:.1f}")
    
    print(f"\nQuality Flags:")
    for flag, count in analysis_stats['quality_flags'].items():
        if count > 0:
            print(f"  {flag}: {count}")
    
    print(f"\nResults saved to: output/data_analysis/")
    print(f"Version metadata: data/versions/version_{version_id}.json")
    
    # Check quality thresholds
    quality_cfg = train_cfg.get("data_quality", {})
    if quality_cfg:
        print(f"\nQuality Threshold Checks:")
        
        min_score = quality_cfg.get("min_quality_score", 70)
        if analysis_stats['quality_scores']['mean_score'] < min_score:
            print(f" Mean quality score ({analysis_stats['quality_scores']['mean_score']:.1f}) below threshold ({min_score})")
        else:
            print(f" Mean quality score ({analysis_stats['quality_scores']['mean_score']:.1f}) above threshold ({min_score})")
        
        max_dup_rate = quality_cfg.get("max_duplicate_rate", 0.05)
        if analysis_stats['duplicates']['duplicate_rate'] > max_dup_rate:
            print(f" Duplicate rate ({analysis_stats['duplicates']['duplicate_rate']:.2%}) above threshold ({max_dup_rate:.2%})")
        else:
            print(f" Duplicate rate ({analysis_stats['duplicates']['duplicate_rate']:.2%}) below threshold ({max_dup_rate:.2%})")

if __name__ == "__main__":
    main()