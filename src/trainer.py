from trl import SFTConfig, SFTTrainer
import mlflow
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_trainer(model, dataset, training_cfg):
    logger.info("Building SFT trainer...")
    logger.info(f"Training config: {training_cfg}")
    args = SFTConfig(**training_cfg)
    trainer = SFTTrainer(model=model, train_dataset=dataset, args=args)
    logger.info("SFT trainer built successfully")
    return trainer


def train_and_save(trainer, tokenizer, output_dir):
    logger.info("Starting model training...")
    # Train with automatic metric logging
    result = trainer.train()
    logger.info("Training completed")
    
    # Log final training metrics
    if hasattr(result, 'metrics'):
        logger.info("Logging final training metrics...")
        for key, value in result.metrics.items():
            mlflow.log_metric(f"final_{key}", value)
            logger.info(f"Final {key}: {value}")
    
    # Save model and tokenizer
    logger.info(f"Saving model and tokenizer to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model and tokenizer saved successfully")
