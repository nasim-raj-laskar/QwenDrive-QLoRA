import torch
import numpy as np
from typing import Dict, List
from difflib import SequenceMatcher

def calculate_perplexity(model, tokenizer, test_data: List[Dict], config: Dict) -> float:
    """Calculate perplexity on test data."""
    perplexities = []
    for item in test_data[:config["perplexity_samples"]]:
        text = f"User: {item['input']}\nAssistant: {item['target']}"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=config["max_seq_length"])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
    
    return np.mean(perplexities)

def calculate_bleu_approx(predictions: List[str], references: List[str]) -> float:
    """Calculate simple BLEU approximation using word overlap."""
    bleu_approx = 0
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        if ref_words:
            bleu_approx += len(pred_words & ref_words) / len(ref_words)
    
    return bleu_approx / len(predictions) if predictions else 0

def calculate_similarity(predictions: List[str], references: List[str]) -> float:
    """Calculate average string similarity using SequenceMatcher."""
    similarities = []
    for pred, ref in zip(predictions, references):
        similarity = SequenceMatcher(None, pred.lower().strip(), ref.lower().strip()).ratio()
        similarities.append(similarity)
    return np.mean(similarities) if similarities else 0