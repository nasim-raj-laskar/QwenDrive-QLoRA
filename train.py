import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import yaml
from src.data import load_and_prepare
from src.model import load_tokenizer, load_model
from src.trainer import build_trainer, train_and_save

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    model_cfg = load_config("configs/model.yaml")
    lora_cfg = load_config("configs/lora.yaml")
    train_cfg = load_config("configs/training.yaml")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    tokenizer = load_tokenizer(model_cfg["model_name"], model_cfg["trust_remote_code"])
    dataset = load_and_prepare(train_cfg["data"], tokenizer, save_sample_path="output/training_sample.jsonl")
    model = load_model(model_cfg, lora_cfg)
    
    trainer = build_trainer(model, dataset, train_cfg["training"])
    train_and_save(trainer, tokenizer, train_cfg["training"]["output_dir"])

if __name__ == "__main__":
    main()
