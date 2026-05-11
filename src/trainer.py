from trl import SFTConfig, SFTTrainer


def build_trainer(model, dataset, training_cfg):
    args = SFTConfig(**training_cfg)
    return SFTTrainer(model=model, train_dataset=dataset, args=args)


def train_and_push(trainer, tokenizer, push_cfg):
    trainer.train()

    trainer.model.save_pretrained(push_cfg["lora_dir"])
    tokenizer.save_pretrained(push_cfg["lora_dir"])

    trainer.model.push_to_hub(push_cfg["lora_hub_repo"])
    tokenizer.push_to_hub(push_cfg["lora_hub_repo"])

    merged = trainer.model.merge_and_unload()
    merged.push_to_hub(push_cfg["merged_hub_repo"])
