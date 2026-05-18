import json
import numpy as np
import hashlib
from collections import Counter
from pathlib import Path
from difflib import SequenceMatcher
from datetime import datetime
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DatasetAnalyzer:
    def __init__(self, dataset, tokenizer, output_dir="output/data_analysis"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_all(self):
        """Run full analysis suite."""
        logger.info("Starting comprehensive dataset analysis...")
        stats = {}
        
        stats["basic"] = self._basic_statistics()
        stats["token_distribution"] = self._token_distribution()
        stats["length_analysis"] = self._length_analysis()
        stats["duplicates"] = self._duplicate_detection()
        stats["quality_flags"] = self._quality_checks()
        stats["quality_scores"] = self._quality_scoring()
        
        self._save_report(stats)
        logger.info(f"Dataset analysis completed. Results saved to {self.output_dir}")
        
        return stats
    
    def _basic_statistics(self):
        """Basic dataset statistics."""
        prompts = [ex["conversations"][0]["value"] for ex in self.dataset]
        responses = [ex["conversations"][1]["value"] for ex in self.dataset]
        
        return {
            "total_samples": len(self.dataset),
            "unique_prompts": len(set(prompts)),
            "unique_responses": len(set(responses)),
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
                "mean": float(np.mean(prompt_lengths)),
                "median": float(np.median(prompt_lengths)),
                "std": float(np.std(prompt_lengths)),
                "min": int(np.min(prompt_lengths)),
                "max": int(np.max(prompt_lengths)),
                "p95": float(np.percentile(prompt_lengths, 95))
            },
            "response_tokens": {
                "mean": float(np.mean(response_lengths)),
                "median": float(np.median(response_lengths)),
                "std": float(np.std(response_lengths)),
                "min": int(np.min(response_lengths)),
                "max": int(np.max(response_lengths)),
                "p95": float(np.percentile(response_lengths, 95))
            },
            "total_tokens": {
                "mean": float(np.mean(total_lengths)),
                "median": float(np.median(total_lengths)),
                "std": float(np.std(total_lengths)),
                "min": int(np.min(total_lengths)),
                "max": int(np.max(total_lengths)),
                "p95": float(np.percentile(total_lengths, 95))
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
                "mean": float(np.mean(prompt_words)),
                "median": float(np.median(prompt_words)),
                "distribution": dict(Counter([min(w // 10 * 10, 100) for w in prompt_words]))
            },
            "response_words": {
                "mean": float(np.mean(response_words)),
                "median": float(np.median(response_words)),
                "distribution": dict(Counter([min(w // 10 * 10, 100) for w in response_words]))
            }
        }
    
    def _duplicate_detection(self):
        """Detect exact and near-duplicate samples."""
        prompts = [ex["conversations"][0]["value"] for ex in self.dataset]
        responses = [ex["conversations"][1]["value"] for ex in self.dataset]
        
        # Exact duplicates
        exact_prompt_dupes = len(prompts) - len(set(prompts))
        exact_response_dupes = len(responses) - len(set(responses))
        
        # Near duplicates (similarity > 0.9) - sample for efficiency
        near_dupes = 0
        sample_size = min(500, len(prompts))  # Reduced for performance
        
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                similarity = SequenceMatcher(None, prompts[i], prompts[j]).ratio()
                if similarity > 0.9:
                    near_dupes += 1
        
        return {
            "exact_prompt_duplicates": exact_prompt_dupes,
            "exact_response_duplicates": exact_response_dupes,
            "near_duplicates_sample": near_dupes,
            "duplicate_rate": exact_prompt_dupes / len(prompts) if prompts else 0
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
    
    def _quality_scoring(self):
        """Score dataset quality."""
        scores = []
        flagged_samples = []
        
        for idx, example in enumerate(self.dataset):
            score = self._score_sample(example)
            scores.append(score["score"])
            
            if score["score"] < 70:  # Flag low-quality samples
                flagged_samples.append({
                    "index": idx,
                    "score": score["score"],
                    "flags": score["flags"],
                    "prompt": example["conversations"][0]["value"][:100]
                })
        
        return {
            "mean_score": float(np.mean(scores)) if scores else 0,
            "median_score": float(np.median(scores)) if scores else 0,
            "low_quality_count": sum(1 for s in scores if s < 70),
            "flagged_samples": flagged_samples[:20]  # Top 20 worst
        }
    
    def _score_sample(self, example):
        """Score a single sample (0-100)."""
        try:
            prompt = example["conversations"][0]["value"]
            response = example["conversations"][1]["value"]
        except (KeyError, IndexError):
            return {"score": 0, "flags": ["malformed_structure"]}
        
        score = 100
        flags = []
        
        # Check prompt length
        prompt_words = prompt.split()
        if len(prompt_words) < 3:
            score -= 20
            flags.append("short_prompt")
        
        # Check response length
        response_words = response.split()
        if len(response_words) < 5:
            score -= 30
            flags.append("short_response")
        elif len(response_words) > 200:
            score -= 10
            flags.append("long_response")
        
        # Check for repetition
        if response_words:
            unique_ratio = len(set(response_words)) / len(response_words)
            if unique_ratio < 0.7:
                score -= 25
                flags.append("repetitive")
        
        # Check for empty content
        if not prompt.strip() or not response.strip():
            score = 0
            flags.append("empty_content")
        
        return {
            "score": max(0, score),
            "flags": flags
        }
    
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
            f.write(f"  Unique prompts: {stats['basic']['unique_prompts']}\n")
            f.write(f"  Unique responses: {stats['basic']['unique_responses']}\n\n")
            
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
            
            f.write("\nQUALITY SCORES:\n")
            f.write(f"  Mean quality score: {stats['quality_scores']['mean_score']:.1f}/100\n")
            f.write(f"  Low quality samples: {stats['quality_scores']['low_quality_count']}\n")


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
                "min_tokens": config.get("min_seq_length", 10),
                "max_tokens": config.get("max_seq_length", 512)
            }
        }
        
        # Save metadata
        version_file = self.output_dir / f"version_{version_id}.json"
        with open(version_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created dataset version {version_id} with hash {dataset_hash[:8]}")
        return version_id, metadata
    
    def _compute_dataset_hash(self, dataset):
        """Compute hash of dataset content."""
        content = ""
        for example in dataset:
            prompt = example["conversations"][0]["value"]
            response = example["conversations"][1]["value"]
            content += f"{prompt}|||{response}\n"
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]


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