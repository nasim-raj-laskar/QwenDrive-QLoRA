import numpy as np
from collections import Counter

class StatisticsAnalyzer:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
    
    def basic_statistics(self):
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
    
    def token_distribution(self):
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
            "prompt_tokens": self._compute_stats(prompt_lengths),
            "response_tokens": self._compute_stats(response_lengths),
            "total_tokens": self._compute_stats(total_lengths)
        }
    
    def length_analysis(self):
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
    
    def _compute_stats(self, lengths):
        """Compute statistics for a list of lengths."""
        return {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "p95": float(np.percentile(lengths, 95))
        }