import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import yaml
from src.pipeline import setup_directories, run_training_pipeline

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    model_cfg = load_config("configs/model.yaml")
    lora_cfg = load_config("configs/lora.yaml")
    train_cfg = load_config("configs/training.yaml")
    
    setup_directories()
    run_training_pipeline(model_cfg, lora_cfg, train_cfg)

if __name__ == "__main__":
    main()
