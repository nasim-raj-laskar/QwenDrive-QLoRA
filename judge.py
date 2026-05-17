#!/usr/bin/env python3
"""
Test script for LLM-as-a-Judge evaluation
"""

import os
import sys
import yaml
sys.path.append('.')

from src.metrics.llm_judge import LLMJudge

def test_llm_judge():
    """Test the LLM judge with a sample automotive question."""
    
    # Load config from eval.yaml
    with open('configs/eval.yaml') as f:
        eval_cfg = yaml.safe_load(f)['evaluation']
    
    llm_judge_config = eval_cfg.get("llm_judge", {})
    
    # Check if LLM judge is enabled
    if not llm_judge_config.get("enabled"):
        print("ERROR: LLM Judge is disabled in config")
        return False
    
    # Check if API key is available
    api_key_env = llm_judge_config.get("api_key_env", "GROQ_API")
    api_key = os.getenv(api_key_env)
    if not api_key:
        print(f"ERROR: {api_key_env} environment variable not found")
        return False
    
    print(f"SUCCESS: {api_key_env} key found")
    
    # Initialize judge using config
    try:
        judge = LLMJudge(config=llm_judge_config, api_key=api_key)
        print("SUCCESS: LLM Judge initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM Judge: {e}")
        return False
    
    # Load test sample from file
    import json
    try:
        with open('src/prompts/test_sample.jsonl', 'r') as f:
            test_sample = json.loads(f.read().strip())
        prompt = test_sample["input"]
        response = test_sample["generated"]
        reference = test_sample["target"]
    except FileNotFoundError:
        # Fallback to default if file not found
        prompt = "What causes engine overheating?"
        response = "Engine overheating can be caused by low coolant levels, a faulty thermostat, a broken water pump, or a clogged radiator."
        reference = "Common causes include low coolant, faulty thermostat, water pump failure, or radiator blockage."
    
    print(f"\nTesting with:")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    # Evaluate
    try:
        scores = judge.evaluate_response(prompt, response, reference)
        print("\nSUCCESS: LLM Judge evaluation successful!")
        print("Scores:")
        for dimension, score in scores.items():
            print(f"  {dimension}: {score:.1f}/10")
        return True
    except Exception as e:
        print(f"ERROR: LLM Judge evaluation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing LLM-as-a-Judge implementation...")
    success = test_llm_judge()
    
    if success:
        print("\nAll tests passed! LLM Judge is ready to use.")
    else:
        print("\nTests failed. Check your API key and internet connection.")