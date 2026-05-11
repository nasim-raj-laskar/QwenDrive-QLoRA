from trl import SFTConfig, SFTTrainer


def build_trainer(model, dataset, training_cfg):
    args = SFTConfig(**training_cfg)
    return SFTTrainer(model=model, train_dataset=dataset, args=args)


def train_and_save(trainer, tokenizer, output_dir):
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
