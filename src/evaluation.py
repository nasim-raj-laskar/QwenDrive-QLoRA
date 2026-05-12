import time
import mlflow
import warnings
import os
from transformers import pipeline
import numpy as np
from typing import Dict, List
from src.metrics.eval_metrics import calculate_perplexity, calculate_bleu_approx, calculate_exact_match
from src.metrics.eval_data import load_test_data

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

class ModelEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            clean_up_tokenization_spaces=False
        )
        
    def evaluate_model(self, eval_samples: int = 100) -> Dict[str, float]:
        """Run comprehensive evaluation."""
        print(f"🔍 Starting evaluation with {eval_samples} samples...")
        
        test_data = load_test_data(eval_samples)
        
        results = {}
        results["perplexity"] = self._evaluate_perplexity(test_data)
        results.update(self._evaluate_generation_quality(test_data))
        results.update(self._evaluate_performance(test_data))
        
        # Log to MLflow
        for metric, value in results.items():
            mlflow.log_metric(f"eval_{metric}", value)
        
        print("✅ Evaluation complete!")
        return results
    
    def _evaluate_perplexity(self, test_data: List[Dict]) -> float:
        """Calculate perplexity."""
        print("📊 Calculating perplexity...")
        return calculate_perplexity(self.model, self.tokenizer, test_data)
    
    def _evaluate_generation_quality(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate generation quality."""
        print("📝 Evaluating generation quality...")
        
        predictions, references = [], []
        
        for item in test_data[:5]:
            prompt = f"User: {item['input']}\nAssistant:"
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.pipe(
                    prompt,
                    max_new_tokens=20,
                    do_sample=False,
                    return_full_text=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            predictions.append(result[0]["generated_text"].strip())
            references.append(item["target"].strip())
        
        return {
            "exact_match": calculate_exact_match(predictions, references),
            "bleu_approx": calculate_bleu_approx(predictions, references)
        }
    
    def _evaluate_performance(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate latency and throughput."""
        print("⚡ Evaluating performance...")
        
        latencies, total_tokens = [], 0
        
        for item in test_data[:3]:
            prompt = f"User: {item['input']}\nAssistant:"
            
            start_time = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.pipe(
                    prompt,
                    max_new_tokens=10,
                    do_sample=False,
                    return_full_text=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            end_time = time.time()
            
            latencies.append(end_time - start_time)
            total_tokens += len(self.tokenizer.encode(result[0]["generated_text"]))
        
        avg_latency = np.mean(latencies)
        throughput = total_tokens / sum(latencies) if sum(latencies) > 0 else 0
        
        return {
            "avg_latency_ms": avg_latency * 1000,
            "token_throughput": throughput
        }

def run_evaluation(model, tokenizer, eval_samples: int = 100) -> Dict[str, float]:
    """Run model evaluation and return results."""
    evaluator = ModelEvaluator(model, tokenizer)
    return evaluator.evaluate_model(eval_samples)