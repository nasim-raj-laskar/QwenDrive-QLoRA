import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import os
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_tokenizer(model_name, trust_remote_code):
    logger.info(f"Loading tokenizer: {model_name}")
    os.environ["HF_HOME"] = "./models/hf_cache"
    
    # Check if model is already cached
    cache_dir = "./models/hf_cache"
    if os.path.exists(cache_dir):
        logger.info("Using cached model files")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=trust_remote_code, 
        cache_dir=cache_dir,
        local_files_only=os.path.exists(cache_dir)
    )
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
    return tokenizer

def load_model(model_cfg, lora_cfg):
    logger.info(f"Loading model: {model_cfg['model_name']}")
    os.environ["HF_HOME"] = "./models/hf_cache"
    cache_dir = "./models/hf_cache"
    
    # Check if model is already cached
    if os.path.exists(cache_dir):
        logger.info("Using cached model files")
    
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    q = model_cfg["quantization"]
    
    logger.info(f"Quantization config: {q}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=dtype_map[q["bnb_4bit_compute_dtype"]],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )

    logger.info("Loading base model with quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"],
        quantization_config=bnb_config,
        device_map=model_cfg["device_map"],
        trust_remote_code=model_cfg["trust_remote_code"],
        cache_dir=cache_dir,
        local_files_only=os.path.exists(cache_dir)
    )

    logger.info(f"Applying LoRA config: {lora_cfg}")
    lora_config = LoraConfig(**lora_cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("Model loaded successfully with LoRA adapters")
    return model
