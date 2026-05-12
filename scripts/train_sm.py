"""
SageMaker training entrypoint.
Runs inside the container; data is at /opt/ml/input/data/training/
and the model is saved to /opt/ml/model/.
"""
import os
import yaml
import sys

sys.path.insert(0, "/opt/ml/code")

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

    sm_data_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    data_file = os.path.join(sm_data_dir, "automotive_en_dataset.jsonl")
    train_cfg["data"]["file"] = data_file
    train_cfg["training"]["output_dir"] = sm_model_dir

    sample_save_path = os.path.join(sm_model_dir, "training_sample.jsonl")

    tokenizer = load_tokenizer(model_cfg["model_name"], model_cfg["trust_remote_code"])
    dataset = load_and_prepare(train_cfg["data"], tokenizer, save_sample_path=sample_save_path)
    model = load_model(model_cfg, lora_cfg)

    trainer = build_trainer(model, dataset, train_cfg["training"])
    train_and_save(trainer, tokenizer, sm_model_dir)


if __name__ == "__main__":
    main()
