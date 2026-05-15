# Dataset Engineering & Quality Analysis

## Current State Analysis

### Existing Data Pipeline

**File**: `src/data.py`

**Current workflow**:
1. Load JSONL file (44,773 rows)
2. Shuffle with seed 42
3. Sample 10,000 rows
4. Format with chat template
5. Tokenize and filter by length (10-512 tokens)

**Current quality checks**:
- Token length filtering only
- No duplicate detection
- No quality scoring
- No statistical analysis

### Critical Gaps

1. **No visibility into data quality**: Unknown how many duplicates, malformed entries, or low-quality samples exist
2. **No data versioning**: Can't track which preprocessing was used for which experiment
3. **No distribution analysis**: Unknown token length patterns, response quality distribution
4. **No anomaly detection**: Corrupted or edge-case samples may slip through

---

## Recommended Improvements

### 1. Dataset Statistics Module

**Purpose**: Generate comprehensive reports on dataset characteristics before training.

#### Implementation

**New file**: `src/data_analysis.py`

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

### 2. Advanced Duplicate Detection

#### Hash-Based Exact Duplicates

```python
import hashlib

def detect_exact_duplicates(dataset):
    """Fast exact duplicate detection using hashing."""
    seen_hashes = {}
    duplicates = []
    
    for idx, example in enumerate(dataset):
        prompt = example["conversations"][0]["value"]
        response = example["conversations"][1]["value"]
        
        # Create hash of prompt+response
        content = f"{prompt}|||{response}"
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash in seen_hashes:
            duplicates.append({
                "index": idx,
                "duplicate_of": seen_hashes[content_hash],
                "prompt": prompt[:100]  # First 100 chars
            })
        else:
            seen_hashes[content_hash] = idx
    
    return duplicates
```

#### Fuzzy Similarity Detection

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def detect_near_duplicates(dataset, threshold=0.85):
    """Detect near-duplicates using TF-IDF + cosine similarity."""
    prompts = [ex["conversations"][0]["value"] for ex in dataset]
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(prompts)
    
    # Compute pairwise similarities
    similarities = cosine_similarity(tfidf_matrix)
    
    # Find pairs above threshold
    near_dupes = []
    for i in range(len(similarities)):
        for j in range(i + 1, len(similarities)):
            if similarities[i, j] > threshold:
                near_dupes.append({
                    "index_1": i,
                    "index_2": j,
                    "similarity": similarities[i, j],
                    "prompt_1": prompts[i][:100],
                    "prompt_2": prompts[j][:100]
                })
    
    return near_dupes
```

---

### 3. Data Quality Scoring

#### Quality Scorer Implementation

```python
class DataQualityScorer:
    def __init__(self):
        self.min_prompt_words = 3
        self.min_response_words = 5
        self.max_response_words = 200
        self.max_repetition_ratio = 0.3
    
    def score_sample(self, example):
        """Score a single sample (0-100)."""
        prompt = example["conversations"][0]["value"]
        response = example["conversations"][1]["value"]
        
        score = 100
        flags = []
        
        # Check prompt length
        prompt_words = prompt.split()
        if len(prompt_words) < self.min_prompt_words:
            score -= 20
            flags.append("short_prompt")
        
        # Check response length
        response_words = response.split()
        if len(response_words) < self.min_response_words:
            score -= 30
            flags.append("short_response")
        elif len(response_words) > self.max_response_words:
            score -= 10
            flags.append("long_response")
        
        # Check for repetition
        if response_words:
            unique_ratio = len(set(response_words)) / len(response_words)
            if unique_ratio < (1 - self.max_repetition_ratio):
                score -= 25
                flags.append("repetitive")
        
        # Check for empty content
        if not prompt.strip() or not response.strip():
            score = 0
            flags.append("empty_content")
        
        # Check for formatting issues
        if response.count("\n\n\n") > 2:  # Excessive newlines
            score -= 10
            flags.append("formatting_issue")
        
        return {
            "score": max(0, score),
            "flags": flags
        }
    
    def score_dataset(self, dataset):
        """Score entire dataset."""
        scores = []
        flagged_samples = []
        
        for idx, example in enumerate(dataset):
            result = self.score_sample(example)
            scores.append(result["score"])
            
            if result["score"] < 70:  # Flag low-quality samples
                flagged_samples.append({
                    "index": idx,
                    "score": result["score"],
                    "flags": result["flags"],
                    "prompt": example["conversations"][0]["value"][:100]
                })
        
        return {
            "mean_score": np.mean(scores),
            "median_score": np.median(scores),
            "low_quality_count": sum(1 for s in scores if s < 70),
            "flagged_samples": flagged_samples[:50]  # Top 50 worst
        }
```

---

### 4. Dataset Versioning

#### Version Metadata Structure

```python
import hashlib
import json
from datetime import datetime

class DatasetVersioner:
    def __init__(self, output_dir="data/versions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_version(self, dataset, config, preprocessing_steps):
        """Create a versioned snapshot of dataset metadata."""
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Compute dataset hash
        dataset_hash = self._compute_dataset_hash(dataset)
        
        # Create version metadata
        metadata = {
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "dataset_hash": dataset_hash,
            "total_samples": len(dataset),
            "config": config,
            "preprocessing_steps": preprocessing_steps,
            "source_file": config.get("file"),
            "sample_size": config.get("sample_size"),
            "shuffle_seed": config.get("shuffle_seed"),
            "filters": {
                "min_tokens": 10,
                "max_tokens": 512
            }
        }
        
        # Save metadata
        version_file = self.output_dir / f"version_{version_id}.json"
        with open(version_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return version_id, metadata
    
    def _compute_dataset_hash(self, dataset):
        """Compute hash of dataset content."""
        content = ""
        for example in dataset:
            prompt = example["conversations"][0]["value"]
            response = example["conversations"][1]["value"]
            content += f"{prompt}|||{response}\n"
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

#### Usage

```python
# In pipeline
versioner = DatasetVersioner()
version_id, metadata = versioner.create_version(
    dataset=datasets["train"],
    config=data_cfg,
    preprocessing_steps=[
        "load_jsonl",
        "shuffle_seed_42",
        "sample_10000",
        "format_chat_template",
        "filter_length_10_512"
    ]
)

# Log to MLflow
mlflow.log_param("dataset_version", version_id)
mlflow.log_param("dataset_hash", metadata["dataset_hash"])
mlflow.log_artifact(f"data/versions/version_{version_id}.json")
```

---

## Expected Output

### Dataset Analysis Report

```
==============================================================
DATASET ANALYSIS REPORT
==============================================================

BASIC STATISTICS:
  Total samples: 9,247
  Unique prompts: 9,103
  Avg conversation length: 342.5 characters

TOKEN DISTRIBUTION:
  Prompt tokens (mean): 28.3
  Prompt tokens (median): 24.0
  Prompt tokens (p95): 56.0
  
  Response tokens (mean): 87.6
  Response tokens (median): 76.0
  Response tokens (p95): 168.0
  
  Total tokens (mean): 115.9
  Total tokens (median): 102.0
  Total tokens (p95): 212.0

LENGTH ANALYSIS:
  Prompt words (mean): 18.7
  Response words (mean): 58.4

QUALITY FLAGS:
  empty_prompts: 0
  empty_responses: 2
  very_short_responses: 127
  very_long_responses: 34
  repetitive_responses: 18
  malformed_structure: 3

DUPLICATES:
  Exact prompt duplicates: 144
  Exact response duplicates: 89
  Duplicate rate: 1.56%
  Near duplicates (sample): 23

QUALITY SCORES:
  Mean quality score: 87.3/100
  Median quality score: 92.0/100
  Low quality samples: 156 (1.69%)

DATASET VERSION:
  Version ID: 20240115_143022
  Dataset hash: a3f7c9e2b1d4f8a6
  Preprocessing: shuffle_seed_42 → sample_10000 → filter_10_512
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

## Integration Checklist

- [ ] Create `src/data_analysis.py` with DatasetAnalyzer class
- [ ] Add duplicate detection functions (exact + fuzzy)
- [ ] Implement DataQualityScorer
- [ ] Create DatasetVersioner for metadata tracking
- [ ] Integrate analysis into pipeline before training
- [ ] Log analysis results to MLflow
- [ ] Generate visualization plots
- [ ] Save analysis reports as artifacts
- [ ] Add data quality thresholds to configs
- [ ] Document dataset versions in experiment tracking

---

## Benefits

| Benefit | Impact |
|---------|--------|
| **Visibility** | Know exactly what's in your training data |
| **Quality Control** | Filter out low-quality samples before training |
| **Reproducibility** | Track exact dataset version used for each experiment |
| **Debugging** | Identify data issues causing poor model performance |
| **Optimization** | Understand token distributions for better batching |
| **Compliance** | Detect and remove duplicate or problematic content |
