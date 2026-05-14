#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import yaml
from transformers import TextStreamer
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
    
    # Load base model and tokenizer with Unsloth
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_name"],
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=model_cfg["trust_remote_code"],
        cache_dir="./models/hf_cache"
    )
    
    # Load trained LoRA adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, "output/qwen3b-automotive")
    
    print("Model loaded successfully!")
    print("=" * 50)
    
    # Setup streaming
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    def stream_response(prompt):
        messages = [{"role": "system", "content": "You are an automotive expert assistant."}, 
                   {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=inference_cfg.get("max_new_tokens", 256),
            temperature=inference_cfg.get("temperature", 0.7),
            top_p=inference_cfg.get("top_p", 0.9),
            do_sample=inference_cfg.get("do_sample", True),
            repetition_penalty=inference_cfg.get("repetition_penalty", 1.1),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Test with config prompt
    print(f"Prompt: {inference_cfg['prompt']}")
    print("Response:")
    stream_response(inference_cfg['prompt'])
    print("\n" + "=" * 50)
    
    # Interactive testing
    while True:
        user_input = input("\nEnter your automotive question (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
            
        print("Response:")
        stream_response(user_input)
        print()

if __name__ == "__main__":
    test_model()