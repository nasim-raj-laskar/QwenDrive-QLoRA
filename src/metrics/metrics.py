import os
import json
import mlflow
from typing import Dict, Any

def log_model_params(model_config: Dict[str, Any]):
    """Log model configuration parameters."""
    mlflow.log_params({
        "model_name": model_config.get("model_name"),
        "model_type": "qwen2.5",
        "parameters": "3.09B",
        "context_window": 32768
    })

def log_quantization_params(bnb_config: Dict[str, Any]):
    """Log quantization configuration."""
    mlflow.log_params({
        "load_in_4bit": bnb_config.get("load_in_4bit"),
        "bnb_4bit_quant_type": bnb_config.get("bnb_4bit_quant_type"),
        "bnb_4bit_compute_dtype": str(bnb_config.get("bnb_4bit_compute_dtype")),
        "bnb_4bit_use_double_quant": bnb_config.get("bnb_4bit_use_double_quant")
    })

def log_lora_params(lora_config: Dict[str, Any]):
    """Log LoRA configuration parameters."""
    mlflow.log_params({
        "lora_r": lora_config.get("r"),
        "lora_alpha": lora_config.get("lora_alpha"),
        "lora_dropout": lora_config.get("lora_dropout"),
        "lora_target_modules": str(lora_config.get("target_modules"))
    })

def log_training_params(training_args: Dict[str, Any]):
    """Log essential training configuration parameters only."""
    mlflow.log_params({
        "num_train_epochs": training_args.get("num_train_epochs"),
        "per_device_train_batch_size": training_args.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": training_args.get("gradient_accumulation_steps"),
        "learning_rate": training_args.get("learning_rate"),
        "lr_scheduler_type": training_args.get("lr_scheduler_type"),
        "optim": training_args.get("optim"),
        "max_seq_length": training_args.get("max_seq_length"),
        "eval_strategy": training_args.get("eval_strategy"),
        "eval_steps": training_args.get("eval_steps"),
        "save_steps": training_args.get("save_steps"),
        "warmup_steps": training_args.get("warmup_steps")
    })

def log_dataset_params(dataset_info: Dict[str, Any]):
    """Log dataset information."""
    mlflow.log_params({
        "dataset_name": "automotive_en_dataset",
        "total_samples": dataset_info.get("total_samples"),
        "sample_size": dataset_info.get("sample_size"),
        "shuffle_seed": dataset_info.get("shuffle_seed")
    })

def log_memory_usage():
    """Log GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            mlflow.log_metrics({
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3
            })
    except Exception:
        pass

def log_model_summary(model, tokenizer):
    """Log model summary as artifact."""
    try:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        summary = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": round((trainable_params / total_params) * 100, 2)
        }
        
        with open("model_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact("model_summary.json")
        os.remove("model_summary.json")
    except Exception:
        pass

def log_adapter_config(output_dir: str):
    """Log adapter config as artifact."""
    config_path = f"{output_dir}/adapter_config.json"
    if os.path.exists(config_path):
        mlflow.log_artifact(config_path)