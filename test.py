#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import yaml
from src.model import load_tokenizer, load_model
from src.inference import run_inference

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def test_model():
    # Load configs
    model_cfg = load_config("configs/model.yaml")
    lora_cfg = load_config("configs/lora.yaml")
    inference_cfg = load_config("configs/inference.yaml")
    
    print("Loading trained model...")
    
    # Load base model and tokenizer (without LoRA)
    tokenizer = load_tokenizer(model_cfg["model_name"], model_cfg["trust_remote_code"])
    
    # Load base model with quantization only
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    q = model_cfg["quantization"]
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=dtype_map[q["bnb_4bit_compute_dtype"]],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"],
        quantization_config=bnb_config,
        device_map=model_cfg["device_map"],
        trust_remote_code=model_cfg["trust_remote_code"],
        cache_dir="./models/hf_cache"
    )
    
    # Load trained LoRA adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, "output/qwen3b-automotive")
    
    print("Model loaded successfully!")
    print("=" * 50)
    
    # Test with config prompt
    print(f"Prompt: {inference_cfg['prompt']}")
    print("Response:")
    response = run_inference(model, tokenizer, inference_cfg)
    print(response)
    print("=" * 50)
    
    # Interactive testing
    while True:
        user_input = input("\nEnter your automotive question (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
            
        test_cfg = inference_cfg.copy()
        test_cfg["prompt"] = user_input
        
        print("Response:")
        response = run_inference(model, tokenizer, test_cfg)
        print(response)

if __name__ == "__main__":
    test_model()