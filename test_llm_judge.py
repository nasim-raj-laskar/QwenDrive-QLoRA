#!/usr/bin/env python3
"""
Test script for LLM-as-a-Judge evaluation
"""

import os
import sys
sys.path.append('.')

from src.metrics.llm_judge import LLMJudge

def test_llm_judge():
    """Test the LLM judge with a sample automotive question."""
    
    # Check if GROQ API key is available
    api_key = os.getenv("GROQ_API")
    if not api_key:
        print("❌ GROQ_API environment variable not found")
        return False
    
    print("✅ GROQ_API key found")
    
    # Initialize judge
    try:
        judge = LLMJudge(api_key=api_key)
        print("✅ LLM Judge initialized")
    except Exception as e:
        print(f"❌ Failed to initialize LLM Judge: {e}")
        return False
    
    # Test sample
    prompt = "What causes engine overheating?"
    response = "Engine overheating can be caused by low coolant levels, a faulty thermostat, a broken water pump, or a clogged radiator."
    reference = "Common causes include low coolant, faulty thermostat, water pump failure, or radiator blockage."
    
    print(f"\nTesting with:")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    # Evaluate
    try:
        scores = judge.evaluate_response(prompt, response, reference)
        print("\n✅ LLM Judge evaluation successful!")
        print("Scores:")
        for dimension, score in scores.items():
            print(f"  {dimension}: {score:.1f}/10")
        return True
    except Exception as e:
        print(f"❌ LLM Judge evaluation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing LLM-as-a-Judge implementation...")
    success = test_llm_judge()
    
    if success:
        print("\n🎉 All tests passed! LLM Judge is ready to use.")
    else:
        print("\n💥 Tests failed. Check your GROQ_API key and internet connection.")