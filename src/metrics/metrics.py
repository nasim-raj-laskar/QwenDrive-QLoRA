import os
import json
import mlflow
from typing import Dict, Any
from src.utils.git_utils import get_git_metadata
from src.utils.env_utils import get_environment_info

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

def log_git_metadata():
    """Log git metadata as artifact."""
    git_meta = get_git_metadata()
    
    # Only log essential git params
    mlflow.log_params({
        "git_commit": git_meta["git_commit"],
        "git_branch": git_meta["git_branch"],
        "git_is_dirty": git_meta["git_is_dirty"]
    })
    
    # Save full git info as artifact
    git_path = "output/git_metadata.json"
    os.makedirs("output", exist_ok=True)
    with open(git_path, "w") as f:
        json.dump(git_meta, f, indent=2)
    mlflow.log_artifact(git_path)
    
    if git_meta["git_is_dirty"]:
        from src.utils.logger import setup_logger
        logger = setup_logger(__name__)
        logger.warning(" Repository has uncommitted changes")

def log_environment_info():
    """Log environment info as artifact."""
    env_info = get_environment_info()
    
    # Only log essential env params
    mlflow.log_params({
        "env_python": env_info.get("env_python", "unknown"),
        "env_cuda": env_info.get("env_cuda", "N/A"),
        "env_gpu_name": env_info.get("env_gpu_name", "N/A")
    })
    
    # Save full environment info as artifact
    env_path = "output/environment.json"
    os.makedirs("output", exist_ok=True)
    with open(env_path, "w") as f:
        json.dump(env_info, f, indent=2)
    mlflow.log_artifact(env_path)

def save_config_snapshot(model_cfg, lora_cfg, training_cfg, data_cfg, output_dir="output"):
    """Save complete runtime configuration snapshot."""
    import yaml
    from datetime import datetime
    
    runtime_config = {
        "timestamp": datetime.now().isoformat(),
        "git": get_git_metadata(),
        "environment": get_environment_info(),
        "model": model_cfg,
        "lora": lora_cfg,
        "training": training_cfg,
        "data": data_cfg
    }
    
    config_path = f"{output_dir}/runtime_config.yaml"
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(runtime_config, f, default_flow_style=False)
    
    mlflow.log_artifact(config_path)
    return config_path