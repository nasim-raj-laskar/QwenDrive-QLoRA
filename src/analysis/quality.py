import numpy as np
from collections import Counter

class QualityAnalyzer:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def quality_checks(self):
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
    
    def quality_scoring(self):
        """Score dataset quality."""
        scores = []
        flagged_samples = []
        
        for idx, example in enumerate(self.dataset):
            score = self.score_sample(example)
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
    
    def score_sample(self, example):
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