import time
import mlflow
import warnings
import os
import yaml
from datetime import datetime
from transformers import pipeline
import numpy as np
from typing import Dict, List
from src.metrics.eval_metrics import calculate_perplexity, calculate_bleu_approx, calculate_similarity
from src.metrics.eval_data import load_test_data

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

def load_eval_config():
    """Load evaluation configuration."""
    with open("configs/eval.yaml") as f:
        return yaml.safe_load(f)["evaluation"]

class ModelEvaluator:
    def __init__(self, model, tokenizer, gpu_profiler=None):
        self.model = model
        self.tokenizer = tokenizer
        self.gpu_profiler = gpu_profiler
        self.config = load_eval_config()
        self.pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            clean_up_tokenization_spaces=False
        )
        
    def evaluate_model(self, eval_samples: int = 100) -> Dict[str, float]:
        """Run comprehensive evaluation."""
        print(f"Starting evaluation with {eval_samples} samples...")
        
        test_data = load_test_data(eval_samples, self.config)
        
        results = {}
        results["perplexity"] = self._evaluate_perplexity(test_data)
        results.update(self._evaluate_generation_quality(test_data))
        results.update(self._evaluate_performance(test_data))
        
        # Save results to file
        self._save_results_to_file(results, eval_samples)
        
        # Log to MLflow
        for metric, value in results.items():
            mlflow.log_metric(f"eval_{metric}", value)
        
        print("Evaluation complete!")
        return results
    
    def _evaluate_perplexity(self, test_data: List[Dict]) -> float:
        """Calculate perplexity."""
        print("Calculating perplexity...")
        return calculate_perplexity(self.model, self.tokenizer, test_data, self.config)
    
    def _evaluate_generation_quality(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate generation quality."""
        print("Evaluating generation quality...")
        
        predictions, references = [], []
        
        for item in test_data[:self.config["generation_samples"]]:
            prompt = f"User: {item['input']}\nAssistant:"
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.pipe(
                    prompt,
                    max_new_tokens=self.config["max_new_tokens_quality"],
                    max_length=None,
                    do_sample=self.config["do_sample"],
                    return_full_text=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            predictions.append(result[0]["generated_text"].strip())
            references.append(item["target"].strip())
        
        return {
            "bleu_approx": calculate_bleu_approx(predictions, references),
            "similarity": calculate_similarity(predictions, references)
        }
    
    def _evaluate_performance(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate latency and throughput."""
        print("⚡ Evaluating performance...")
        
        latencies, total_tokens = [], 0
        
        for item in test_data[:self.config["performance_samples"]]:
            prompt = f"User: {item['input']}\nAssistant:"
            
            start_time = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.pipe(
                    prompt,
                    max_new_tokens=self.config["max_new_tokens_performance"],
                    max_length=None,
                    do_sample=self.config["do_sample"],
                    return_full_text=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            end_time = time.time()
            
            duration = end_time - start_time
            latencies.append(duration)
            
            generated_tokens = len(self.tokenizer.encode(result[0]["generated_text"]))
            total_tokens += generated_tokens
            
            # Record tokens/sec for GPU profiler
            if self.gpu_profiler:
                self.gpu_profiler.record_tokens_per_sec(generated_tokens, duration)
        
        avg_latency = np.mean(latencies)
        throughput = total_tokens / sum(latencies) if sum(latencies) > 0 else 0
        
        return {
            "avg_latency_ms": avg_latency * 1000,
            "token_throughput": throughput
        }
    
    def _save_results_to_file(self, results: Dict[str, float], eval_samples: int):
        """Save evaluation results to text file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/eval_results_{timestamp}.txt"
        
        # Get GPU metrics if profiler is available
        gpu_metrics = {}
        if self.gpu_profiler:
            gpu_metrics = self.gpu_profiler.get_metrics()
        
        with open(filename, "w") as f:
            f.write(f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Samples: {eval_samples}\n")
            f.write("=" * 50 + "\n")
            
            # Write evaluation metrics
            f.write("EVALUATION METRICS:\n")
            f.write(f"Perplexity:           {results.get('perplexity', 0):.2f}\n")
            f.write(f"BLEU (approx):        {results.get('bleu_approx', 0):.1%}\n")
            f.write(f"Similarity:           {results.get('similarity', 0):.1%}\n")
            f.write(f"Avg Latency:          {results.get('avg_latency_ms', 0):.0f} ms\n")
            f.write(f"Token Throughput:     {results.get('token_throughput', 0):.1f} tokens/sec\n")
            
            # Write GPU metrics if available
            if gpu_metrics:
                f.write("\nGPU METRICS:\n")
                f.write(f"VRAM Peak:            {gpu_metrics.get('gpu_vram_peak_gb', 0):.2f} GB\n")
                f.write(f"GPU Utilization Avg:  {gpu_metrics.get('gpu_utilization_avg', 0):.1f}%\n")
                f.write(f"GPU Utilization Max:  {gpu_metrics.get('gpu_utilization_max', 0):.1f}%\n")
                f.write(f"Tokens/sec Avg:       {gpu_metrics.get('tokens_per_sec_avg', 0):.0f} tokens/sec\n")
                f.write(f"Tokens/sec Max:       {gpu_metrics.get('tokens_per_sec_max', 0):.0f} tokens/sec\n")
        
        # Log as MLflow artifact
        mlflow.log_artifact(filename)

def run_evaluation(model, tokenizer, eval_samples: int = 100, gpu_profiler=None) -> Dict[str, float]:
    """Run model evaluation and return results."""
    evaluator = ModelEvaluator(model, tokenizer, gpu_profiler)
    return evaluator.evaluate_model(eval_samples)