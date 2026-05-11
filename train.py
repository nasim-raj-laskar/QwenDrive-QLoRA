import yaml

from src.data import load_and_prepare
from src.model import load_tokenizer, load_model
from src.trainer import build_trainer, train_and_save
from src.inference import run_inference


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    model_cfg = load_config("configs/model.yaml")
    lora_cfg = load_config("configs/lora.yaml")
    train_cfg = load_config("configs/training.yaml")
    inference_cfg = load_config("configs/inference.yaml")

    tokenizer = load_tokenizer(model_cfg["model_name"], model_cfg["trust_remote_code"])
    dataset = load_and_prepare(train_cfg["data"], tokenizer)
    model = load_model(model_cfg, lora_cfg)

    trainer = build_trainer(model, dataset, train_cfg["training"])
    train_and_save(trainer, tokenizer, train_cfg["training"]["output_dir"])

    print(run_inference(model, tokenizer, inference_cfg))


if __name__ == "__main__":
    main()
