#!/usr/bin/env python3
import sys
sys.path.append('.')

from src.metrics.pairwise_eval import PairwiseEvaluator
from unsloth import FastLanguageModel
import yaml

def test_pairwise():
    # Load all configs
    with open('configs/model.yaml') as f:
        model_cfg = yaml.safe_load(f)
    with open('configs/eval.yaml') as f:
        eval_cfg = yaml.safe_load(f)['evaluation']
    with open('configs/training.yaml') as f:
        train_cfg = yaml.safe_load(f)
    
    # Load base model using config
    print("Loading base model...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_name"],
        max_seq_length=eval_cfg.get("max_seq_length", 512),
        dtype=None,
        load_in_4bit=model_cfg.get("quantization", {}).get("load_in_4bit", True),
    )
    
    # Load fine-tuned model from config output dir
    print("Loading fine-tuned model from checkpoint...")
    output_dir = train_cfg["training"]["output_dir"]
    ft_model, _ = FastLanguageModel.from_pretrained(
        model_name=output_dir,
        max_seq_length=eval_cfg.get("max_seq_length", 512),
        dtype=None,
        load_in_4bit=model_cfg.get("quantization", {}).get("load_in_4bit", True),
    )
    
    # Initialize LLM judge from config
    from src.metrics.llm_judge import LLMJudge
    import os
    
    llm_judge = None
    llm_judge_config = eval_cfg.get("llm_judge", {})
    if llm_judge_config.get("enabled"):
        api_key = os.getenv(llm_judge_config.get("api_key_env", "GROQ_API"))
        if api_key:
            llm_judge = LLMJudge(config=llm_judge_config, api_key=api_key)
            print("LLM Judge enabled")
    
    evaluator = PairwiseEvaluator(base_model, ft_model, tokenizer, llm_judge)
    
    # Get pairwise config
    pairwise_config = eval_cfg.get("pairwise_eval", {})
    max_tokens = pairwise_config.get("max_new_tokens", 50)
    
    # Load test prompts from eval prompts file
    import json
    prompts_file = eval_cfg.get("category_eval", {}).get("prompts_file", "src/prompts/eval_prompts.jsonl")
    prompts = []
    
    try:
        with open(prompts_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                prompts.append(item.get("input", ""))
                if len(prompts) >= 3:  # Limit to 3 for testing
                    break
    except FileNotFoundError:
        # Fallback to default prompts if file not found
        prompts = [
            "What causes engine overheating?",
            "How do I check my brake pads?",
            "My car won't start. What should I check?"
        ]
    
    print("Running pairwise comparison...")
    
    # Debug: Show actual responses
    for i, prompt in enumerate(prompts):
        base_resp = evaluator._generate_response(evaluator.base_pipe, prompt, max_tokens)
        ft_resp = evaluator._generate_response(evaluator.ft_pipe, prompt, max_tokens)
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Base: {base_resp[:100]}...")
        print(f"FT:   {ft_resp[:100]}...")
    
    results = evaluator.compare_responses(prompts, max_new_tokens=max_tokens)
    
    print("\nResults:")
    for key, value in results.items():
        if key.endswith('_rate'):
            print(f"{key}: {value:.1%}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    test_pairwise()