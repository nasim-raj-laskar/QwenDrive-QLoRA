import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import os

def load_tokenizer(model_name, trust_remote_code):
    os.environ["HF_HOME"] = "./models/hf_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, cache_dir="./models/hf_cache")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(model_cfg, lora_cfg):
    os.environ["HF_HOME"] = "./models/hf_cache"
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

    lora_config = LoraConfig(**lora_cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
