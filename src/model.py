import torch
from unsloth import FastLanguageModel
import os
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_tokenizer(model_name, trust_remote_code):
    logger.info(f"Loading tokenizer: {model_name}")
    os.environ["HF_HOME"] = "./models/hf_cache"
    
    cache_dir = "./models/hf_cache"
    if os.path.exists(cache_dir):
        logger.info("Using cached model files")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir
    )
    
    logger.info(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
    return tokenizer

def load_model(model_cfg, lora_cfg):
    logger.info(f"Loading model: {model_cfg['model_name']}")
    os.environ["HF_HOME"] = "./models/hf_cache"
    cache_dir = "./models/hf_cache"
    
    if os.path.exists(cache_dir):
        logger.info("Using cached model files")
    
    logger.info("Loading model with Unsloth optimizations...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_name"],
        max_seq_length=512,
        dtype=None,  # Auto-detect optimal dtype
        load_in_4bit=True,
        trust_remote_code=model_cfg["trust_remote_code"],
        cache_dir=cache_dir
    )
    
    logger.info(f"Applying LoRA with Unsloth: {lora_cfg}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    logger.info("Model loaded successfully with Unsloth optimizations")
    return model, tokenizer
