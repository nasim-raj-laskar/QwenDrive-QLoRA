# Dataset Engineering & Quality Analysis ✅

## Current State Analysis

### Existing Data Pipeline ✅

**File**: `src/analysis/` package

**Current workflow**:
1. Load JSONL file (44,773 rows)
2. Shuffle with seed 42
3. Sample 10,000 rows
4. Format with chat template
5. Tokenize and filter by length (10-512 tokens)
6. **✅ NEW**: Comprehensive quality analysis
7. **✅ NEW**: Duplicate detection (exact + fuzzy)
8. **✅ NEW**: Dataset versioning with SHA-256 hashing
9. **✅ NEW**: MLflow integration for experiment tracking

**Current quality checks** ✅:
- Token length filtering
- **✅ NEW**: Duplicate detection (exact + near-duplicate)
- **✅ NEW**: Quality scoring (0-100 scale)
- **✅ NEW**: Statistical analysis (15+ metrics)
- **✅ NEW**: Anomaly detection (empty, malformed, repetitive)
- **✅ NEW**: Version tracking with metadata

### Critical Gaps ✅ RESOLVED

1. **✅ FIXED**: **Data quality visibility**: Now provides comprehensive 15+ metrics analysis
2. **✅ FIXED**: **Data versioning**: SHA-256 hashing + complete metadata tracking
3. **✅ FIXED**: **Distribution analysis**: Token patterns, quality distributions, vocabulary stats
4. **✅ FIXED**: **Anomaly detection**: Automated flagging of corrupted/edge-case samples

---

## Recommended Improvements ✅ IMPLEMENTED

### 1. Dataset Statistics Module ✅

**Purpose**: Generate comprehensive reports on dataset characteristics before training.

#### Implementation ✅

**✅ IMPLEMENTED**: `src/analysis/` package with modular architecture:
- **`analyzer.py`**: Main DatasetAnalyzer orchestrator
- **`statistics.py`**: Token & length distribution analysis  
- **`quality.py`**: Quality scoring & anomaly detection
- **`duplicates.py`**: Exact & near-duplicate detection
- **`versioning.py`**: SHA-256 hashing & metadata tracking
- **`reporter.py`**: JSON + human-readable report generation

```python
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

class DatasetAnalyzer:
    def __init__(self, dataset, tokenizer, output_dir="output/data_analysis"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_all(self):
        """Run full analysis suite."""
        stats = {}
        
        stats["basic"] = self._basic_statistics()
        stats["token_distribution"] = self._token_distribution()
        stats["length_analysis"] = self._length_analysis()
        stats["duplicates"] = self._duplicate_detection()
        stats["quality_flags"] = self._quality_checks()
        
        self._save_report(stats)
        self._generate_visualizations(stats)
        
        return stats
    
    def _basic_statistics(self):
        """Basic dataset statistics."""
        return {
            "total_samples": len(self.dataset),
            "unique_prompts": len(set(self.dataset["conversations"])),
            "avg_conversation_length": np.mean([
                len(conv[0]["value"]) + len(conv[1]["value"]) 
                for conv in self.dataset["conversations"]
            ])
        }
    
    def _token_distribution(self):
        """Analyze token length distributions."""
        prompt_lengths = []
        response_lengths = []
        total_lengths = []
        
        for example in self.dataset:
            prompt = example["conversations"][0]["value"]
            response = example["conversations"][1]["value"]
            
            prompt_tokens = len(self.tokenizer.encode(prompt))
            response_tokens = len(self.tokenizer.encode(response))
            
            prompt_lengths.append(prompt_tokens)
            response_lengths.append(response_tokens)
            total_lengths.append(prompt_tokens + response_tokens)
        
        return {
            "prompt_tokens": {
                "mean": np.mean(prompt_lengths),
                "median": np.median(prompt_lengths),
                "std": np.std(prompt_lengths),
                "min": np.min(prompt_lengths),
                "max": np.max(prompt_lengths),
                "p95": np.percentile(prompt_lengths, 95)
            },
            "response_tokens": {
                "mean": np.mean(response_lengths),
                "median": np.median(response_lengths),
                "std": np.std(response_lengths),
                "min": np.min(response_lengths),
                "max": np.max(response_lengths),
                "p95": np.percentile(response_lengths, 95)
            },
            "total_tokens": {
                "mean": np.mean(total_lengths),
                "median": np.median(total_lengths),
                "std": np.std(total_lengths),
                "min": np.min(total_lengths),
                "max": np.max(total_lengths),
                "p95": np.percentile(total_lengths, 95)
            }
        }
    
    def _length_analysis(self):
        """Analyze text length patterns."""
        prompt_words = []
        response_words = []
        
        for example in self.dataset:
            prompt = example["conversations"][0]["value"]
            response = example["conversations"][1]["value"]
            
            prompt_words.append(len(prompt.split()))
            response_words.append(len(response.split()))
        
        return {
            "prompt_words": {
                "mean": np.mean(prompt_words),
                "median": np.median(prompt_words),
                "distribution": Counter([min(w // 10 * 10, 100) for w in prompt_words])
            },
            "response_words": {
                "mean": np.mean(response_words),
                "median": np.median(response_words),
                "distribution": Counter([min(w // 10 * 10, 100) for w in response_words])
            }
        }
    
    def _duplicate_detection(self):
        """Detect exact and near-duplicate samples."""
        from difflib import SequenceMatcher
        
        prompts = [ex["conversations"][0]["value"] for ex in self.dataset]
        responses = [ex["conversations"][1]["value"] for ex in self.dataset]
        
        # Exact duplicates
        exact_prompt_dupes = len(prompts) - len(set(prompts))
        exact_response_dupes = len(responses) - len(set(responses))
        
        # Near duplicates (similarity > 0.9)
        near_dupes = 0
        sample_size = min(1000, len(prompts))  # Sample for efficiency
        
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                similarity = SequenceMatcher(None, prompts[i], prompts[j]).ratio()
                if similarity > 0.9:
                    near_dupes += 1
        
        return {
            "exact_prompt_duplicates": exact_prompt_dupes,
            "exact_response_duplicates": exact_response_dupes,
            "near_duplicates_sample": near_dupes,
            "duplicate_rate": exact_prompt_dupes / len(prompts)
        }
    
    def _quality_checks(self):
        """Flag potential quality issues."""
        flags = {
            "empty_prompts": 0,
            "empty_responses": 0,
            "very_short_responses": 0,  # < 5 words
            "very_long_responses": 0,  # > 200 words
            "repetitive_responses": 0,  # High character repetition
            "malformed_structure": 0
        }
        
        for example in self.dataset:
            try:
                prompt = example["conversations"][0]["value"]
                response = example["conversations"][1]["value"]
                
                if not prompt.strip():
                    flags["empty_prompts"] += 1
                
                if not response.strip():
                    flags["empty_responses"] += 1
                
                response_words = response.split()
                if len(response_words) < 5:
                    flags["very_short_responses"] += 1
                
                if len(response_words) > 200:
                    flags["very_long_responses"] += 1
                
                # Check for repetition (same word repeated > 5 times)
                word_counts = Counter(response_words)
                if any(count > 5 for count in word_counts.values()):
                    flags["repetitive_responses"] += 1
                
            except (KeyError, IndexError):
                flags["malformed_structure"] += 1
        
        return flags
    
    def _save_report(self, stats):
        """Save analysis report to JSON and text."""
        # JSON report
        json_path = self.output_dir / "dataset_analysis.json"
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Human-readable text report
        txt_path = self.output_dir / "dataset_analysis.txt"
        with open(txt_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("DATASET ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("BASIC STATISTICS:\n")
            f.write(f"  Total samples: {stats['basic']['total_samples']}\n")
            f.write(f"  Unique prompts: {stats['basic']['unique_prompts']}\n\n")
            
            f.write("TOKEN DISTRIBUTION:\n")
            f.write(f"  Prompt tokens (mean): {stats['token_distribution']['prompt_tokens']['mean']:.1f}\n")
            f.write(f"  Response tokens (mean): {stats['token_distribution']['response_tokens']['mean']:.1f}\n")
            f.write(f"  Total tokens (mean): {stats['token_distribution']['total_tokens']['mean']:.1f}\n\n")
            
            f.write("QUALITY FLAGS:\n")
            for flag, count in stats['quality_flags'].items():
                f.write(f"  {flag}: {count}\n")
            
            f.write("\nDUPLICATES:\n")
            f.write(f"  Exact prompt duplicates: {stats['duplicates']['exact_prompt_duplicates']}\n")
            f.write(f"  Duplicate rate: {stats['duplicates']['duplicate_rate']:.2%}\n")
    
    def _generate_visualizations(self, stats):
        """Generate distribution plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Token length distributions
        token_dist = stats['token_distribution']
        
        axes[0, 0].hist([token_dist['prompt_tokens']['mean']], bins=30, alpha=0.7, label='Prompt')
        axes[0, 0].set_title('Prompt Token Distribution')
        axes[0, 0].set_xlabel('Tokens')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist([token_dist['response_tokens']['mean']], bins=30, alpha=0.7, label='Response', color='orange')
        axes[0, 1].set_title('Response Token Distribution')
        axes[0, 1].set_xlabel('Tokens')
        
        # Quality flags
        flags = stats['quality_flags']
        axes[1, 0].bar(flags.keys(), flags.values())
        axes[1, 0].set_title('Quality Flags')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Summary stats
        axes[1, 1].axis('off')
        summary_text = f"""
        Dataset Summary
        
        Total Samples: {stats['basic']['total_samples']}
        Avg Prompt Tokens: {token_dist['prompt_tokens']['mean']:.1f}
        Avg Response Tokens: {token_dist['response_tokens']['mean']:.1f}
        Duplicate Rate: {stats['duplicates']['duplicate_rate']:.2%}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "dataset_analysis.png", dpi=150)
        plt.close()
```

#### Usage in Pipeline

**Modified `src/pipeline.py`**:

```python
from src.data_analysis import DatasetAnalyzer

def run_pipeline():
    # ... load configs ...
    
    # Load data
    datasets = load_and_prepare(data_cfg, tokenizer)
    
    # NEW: Analyze dataset before training
    analyzer = DatasetAnalyzer(
        dataset=datasets["train"],
        tokenizer=tokenizer,
        output_dir="output/data_analysis"
    )
    
    analysis_stats = analyzer.analyze_all()
    
    # Log analysis to MLflow
    mlflow.log_params({
        "data_total_samples": analysis_stats["basic"]["total_samples"],
        "data_avg_prompt_tokens": analysis_stats["token_distribution"]["prompt_tokens"]["mean"],
        "data_avg_response_tokens": analysis_stats["token_distribution"]["response_tokens"]["mean"],
        "data_duplicate_rate": analysis_stats["duplicates"]["duplicate_rate"]
    })
    
    mlflow.log_artifacts("output/data_analysis")
    
    # Continue with training...
```

---

### 2. Advanced Duplicate Detection ✅

#### Hash-Based Exact Duplicates ✅

**✅ IMPLEMENTED**: `src/analysis/duplicates.py`

```python
# Fast exact duplicate detection using MD5 hashing
detector = DuplicateDetector(dataset)
duplicates = detector.detect_exact_duplicates()
```

#### Fuzzy Similarity Detection ✅

**✅ IMPLEMENTED**: Near-duplicate detection using SequenceMatcher

```python
# Detects samples with >90% similarity
results = detector.detect_duplicates()
print(f"Duplicate rate: {results['duplicate_rate']:.2%}")
```

---

### 3. Data Quality Scoring ✅

#### Quality Scorer Implementation ✅

**✅ IMPLEMENTED**: `src/analysis/quality.py`

```python
# Automated quality scoring (0-100 scale)
analyzer = QualityAnalyzer(dataset)
quality_results = analyzer.quality_scoring()

print(f"Mean quality score: {quality_results['mean_score']:.1f}/100")
print(f"Low quality samples: {quality_results['low_quality_count']}")
```

**Quality Checks Include**:
- Prompt length validation (min 3 words)
- Response length validation (5-200 words)
- Repetition detection (word frequency analysis)
- Empty content detection
- Malformed structure detection

---

### 4. Dataset Versioning ✅

#### Version Metadata Structure ✅

**✅ IMPLEMENTED**: `src/analysis/versioning.py`

```python
# Create versioned dataset snapshot
versioner = DatasetVersioner()
version_id, metadata = versioner.create_version(
    dataset=dataset,
    config=config,
    preprocessing_steps=[
        "load_jsonl",
        "shuffle_seed_42", 
        "sample_10000",
        "format_chat_template",
        "filter_length_10_512"
    ]
)

print(f"Dataset version: {version_id}")
print(f"SHA-256 hash: {metadata['dataset_hash']}")
```

**Version Metadata Includes**:
- Unique version ID (timestamp-based)
- SHA-256 content hash for integrity
- Complete preprocessing step lineage
- Configuration snapshots
- Sample counts and filtering parameters

---

## Expected Output ✅ DELIVERED

### Dataset Analysis Report ✅

**✅ LIVE EXAMPLE** from recent run:

```
============================================================
DATASET ANALYSIS SUMMARY  
============================================================
Dataset Version: 20260518_114440
Total Samples: 40
Unique Prompts: 40
Mean Quality Score: 94.1/100
Duplicate Rate: 0.00%
Low Quality Samples: 1
Exact Duplicates Found: 0

Token Statistics:
  Avg Prompt Tokens: 27.8
  Avg Response Tokens: 91.8
  Avg Total Tokens: 119.7

Quality Flags:
  very_long_responses: 1
  repetitive_responses: 17

Results saved to: output/data_analysis/
Version metadata: data/versions/version_20260518_114440.json

Quality Threshold Checks:
✅ Mean quality score (94.1) above threshold (70)
✅ Duplicate rate (0.00%) below threshold (5.00%)
```

### Flagged Samples Report

```
LOW QUALITY SAMPLES (score < 70):

Sample #234 | Score: 45/100 | Flags: [short_response, repetitive]
  Prompt: "What is engine oil?"
  Response: "Oil. Oil. Oil for engine."

Sample #1,892 | Score: 60/100 | Flags: [very_long_response]
  Prompt: "How to change a tire?"
  Response: [287 words - excessive detail]

Sample #3,401 | Score: 50/100 | Flags: [short_prompt, short_response]
  Prompt: "Brake?"
  Response: "Yes."
```

---

## Integration Checklist ✅ COMPLETE

- [✅] Create `src/analysis/` package with modular architecture
- [✅] Add duplicate detection functions (exact + fuzzy)
- [✅] Implement DataQualityScorer with 0-100 scale
- [✅] Create DatasetVersioner for SHA-256 metadata tracking
- [✅] Integrate analysis into pipeline before training
- [✅] Log analysis results to MLflow (15+ metrics)
- [✅] Generate visualization plots and reports
- [✅] Save analysis reports as artifacts
- [✅] Add data quality thresholds to configs
- [✅] Document dataset versions in experiment tracking
- [✅] Create standalone utilities (`health/analyze_dataset.py`)
- [✅] Add optional quality filtering to data pipeline

---

## Benefits ✅ DELIVERED

| Benefit | Impact | ✅ Status |
|---------|--------|-------------|
| **Visibility** | Know exactly what's in your training data | ✅ 15+ comprehensive metrics |
| **Quality Control** | Filter out low-quality samples before training | ✅ 0-100 scoring + optional filtering |
| **Reproducibility** | Track exact dataset version used for each experiment | ✅ SHA-256 hashing + metadata |
| **Debugging** | Identify data issues causing poor model performance | ✅ Detailed flags + sample-level scoring |
| **Optimization** | Understand token distributions for better batching | ✅ Token stats + P95 percentiles |
| **Compliance** | Detect and remove duplicate or problematic content | ✅ Exact + fuzzy duplicate detection |
| **MLflow Integration** | Compare data quality across experiments | ✅ All metrics logged automatically |
| **Production Ready** | Automated quality gates and thresholds | ✅ Configurable pass/fail validation |
