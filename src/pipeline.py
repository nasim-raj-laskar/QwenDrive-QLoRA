import os
import mlflow
from src.data import load_and_prepare
from src.model import load_tokenizer, load_model
from src.trainer import build_trainer, train_and_save
from src.utils.mlflow import init_mlflow
from src.metrics.metrics import *

def setup_directories():
    """Create necessary directories."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)

def log_all_params(model_cfg, lora_cfg, train_cfg):
    """Log all training parameters to MLflow."""
    log_model_params(model_cfg)
    log_quantization_params(model_cfg.get("quantization", {}))
    log_lora_params(lora_cfg)
    log_training_params(train_cfg["training"])
    log_dataset_params(train_cfg["data"])

def run_training_pipeline(model_cfg, lora_cfg, train_cfg):
    """Execute the complete training pipeline."""
    init_mlflow()
    
    with mlflow.start_run(run_name="qwen-drive-lora-training"):
        log_all_params(model_cfg, lora_cfg, train_cfg)
        
        # Load components
        tokenizer = load_tokenizer(model_cfg["model_name"], model_cfg["trust_remote_code"])
        dataset = load_and_prepare(train_cfg["data"], tokenizer, save_sample_path="output/training_sample.jsonl")
        model = load_model(model_cfg, lora_cfg)
        
        # Log initial state
        log_model_summary(model, tokenizer)
        log_memory_usage()
        
        # Train
        trainer = build_trainer(model, dataset, train_cfg["training"])
        train_and_save(trainer, tokenizer, train_cfg["training"]["output_dir"])
        
        # Log final artifacts
        log_adapter_config(train_cfg["training"]["output_dir"])
        log_memory_usage()