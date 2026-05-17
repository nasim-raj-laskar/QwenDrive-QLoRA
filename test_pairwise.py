#!/usr/bin/env python3
import sys
sys.path.append('.')

from src.metrics.pairwise_eval import PairwiseEvaluator
from unsloth import FastLanguageModel
import yaml

def test_pairwise():
    # Load configs
    with open('configs/model.yaml') as f:
        model_cfg = yaml.safe_load(f)
    with open('configs/lora.yaml') as f:
        lora_cfg = yaml.safe_load(f)
    
    # Load base model
    print("Loading base model...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_name"],
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Load fine-tuned model from saved checkpoint
    print("Loading fine-tuned model from checkpoint...")
    ft_model, _ = FastLanguageModel.from_pretrained(
        model_name="./output/qwen3b-automotive",  # Load from saved checkpoint
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Initialize pairwise evaluator with LLM judge
    from src.metrics.llm_judge import LLMJudge
    import os
    
    llm_judge = None
    if os.getenv("GROQ_API"):
        judge_config = {
            "model": "llama-3.3-70b-versatile",
            "api_base_url": "https://api.groq.com/openai/v1/chat/completions",
            "max_new_tokens": 300,
            "temperature": 0.1,
            "timeout": 30
        }
        llm_judge = LLMJudge(config=judge_config, api_key=os.getenv("GROQ_API"))
        print("✅ LLM Judge enabled for better comparison")
    
    evaluator = PairwiseEvaluator(base_model, ft_model, tokenizer, llm_judge)
    
    # Test prompts
    prompts = [
        "What causes engine overheating?",
        "How do I check my brake pads?",
        "My car won't start. What should I check?"
    ]
    
    print("Running pairwise comparison...")
    
    # Debug: Show actual responses
    for i, prompt in enumerate(prompts):
        base_resp = evaluator._generate_response(evaluator.base_pipe, prompt, 50)
        ft_resp = evaluator._generate_response(evaluator.ft_pipe, prompt, 50)
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Base: {base_resp[:100]}...")
        print(f"FT:   {ft_resp[:100]}...")
    
    results = evaluator.compare_responses(prompts, max_new_tokens=50)
    
    print("\nResults:")
    for key, value in results.items():
        if key.endswith('_rate'):
            print(f"{key}: {value:.1%}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    test_pairwise()