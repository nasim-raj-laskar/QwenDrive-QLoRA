import torch
import time
import mlflow
from transformers import pipeline
from datasets import load_dataset
from evaluate import load
import numpy as np
from typing import Dict, List, Any

class ModelEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
    def evaluate_model(self, eval_samples: int = 100) -> Dict[str, float]:
        """Run comprehensive evaluation."""
        print(f"🔍 Starting evaluation with {eval_samples} samples...")
        
        # Load test data
        test_data = self._load_test_data(eval_samples)
        
        # Run evaluations
        results = {}
        results.update(self._evaluate_perplexity(test_data))
        results.update(self._evaluate_generation_quality(test_data))
        results.update(self._evaluate_performance(test_data))
        
        # Log to MLflow
        for metric, value in results.items():
            mlflow.log_metric(f"eval_{metric}", value)
        
        print("✅ Evaluation complete!")
        return results
    
    def _load_test_data(self, n_samples: int) -> List[Dict]:
        """Load test dataset."""
        dataset = load_dataset("json", data_files="data/automotive_en_dataset.jsonl")
        test_data = dataset["train"].shuffle(seed=123).select(range(n_samples))
        
        formatted_data = []
        for example in test_data:
            human = example["conversations"][0]["value"]
            assistant = example["conversations"][1]["value"]
            formatted_data.append({"input": human, "target": assistant})
        
        return formatted_data
    
    def _evaluate_perplexity(self, test_data: List[Dict]) -> Dict[str, float]:
        """Calculate perplexity."""
        print("📊 Calculating perplexity...")
        
        perplexities = []
        for item in test_data[:20]:  # Sample for speed
            text = f"User: {item['input']}\nAssistant: {item['target']}"
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Move inputs to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
        
        return {"perplexity": np.mean(perplexities)}
    
    def _evaluate_generation_quality(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate generation quality with simple metrics."""
        print("📝 Evaluating generation quality...")
        
        predictions = []
        references = []
        exact_matches = 0
        
        for item in test_data[:5]:  # Only 5 samples for speed
            prompt = f"User: {item['input']}\nAssistant:"
            
            # Generate response with minimal tokens
            result = self.pipe(
                prompt,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated = result[0]["generated_text"].strip()
            target = item["target"].strip()
            
            predictions.append(generated)
            references.append(target)
            
            # Exact match
            if generated.lower() == target.lower():
                exact_matches += 1
        
        results = {"exact_match": exact_matches / len(predictions)}
        
        # Simple BLEU approximation (word overlap)
        bleu_approx = 0
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            if ref_words:
                bleu_approx += len(pred_words & ref_words) / len(ref_words)
        
        results["bleu_approx"] = bleu_approx / len(predictions)
        return results
    
    def _evaluate_performance(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate latency and throughput."""
        print("⚡ Evaluating performance...")
        
        latencies = []
        total_tokens = 0
        
        for item in test_data[:3]:  # Only 3 samples for speed
            prompt = f"User: {item['input']}\nAssistant:"
            
            start_time = time.time()
            result = self.pipe(
                prompt,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            end_time = time.time()
            
            latency = end_time - start_time
            latencies.append(latency)
            
            # Count tokens
            generated_tokens = len(self.tokenizer.encode(result[0]["generated_text"]))
            total_tokens += generated_tokens
        
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