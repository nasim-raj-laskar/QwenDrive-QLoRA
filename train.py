import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import yaml
from src.pipeline import setup_directories, run_training_pipeline
from src.utils.logger import setup_logger

logger = setup_logger("main")

def load_config(path):
    logger.info(f"Loading config from {path}")
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    logger.info("Starting QwenDrive-QLoRA training...")
    model_cfg = load_config("configs/model.yaml")
    lora_cfg = load_config("configs/lora.yaml")
    train_cfg = load_config("configs/training.yaml")
    
    setup_directories()
    run_training_pipeline(model_cfg, lora_cfg, train_cfg)
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
