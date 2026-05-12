from trl import SFTConfig, SFTTrainer
import mlflow


def build_trainer(model, dataset, training_cfg):
    args = SFTConfig(**training_cfg)
    return SFTTrainer(model=model, train_dataset=dataset, args=args)


def train_and_save(trainer, tokenizer, output_dir):
    # Train with automatic metric logging
    result = trainer.train()
    
    # Log final training metrics
    if hasattr(result, 'metrics'):
        for key, value in result.metrics.items():
            mlflow.log_metric(f"final_{key}", value)
    
    # Save model and tokenizer
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
