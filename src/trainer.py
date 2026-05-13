from unsloth import is_bfloat16_supported, FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import mlflow
import time
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def build_trainer(model, tokenizer, dataset, training_cfg):
    logger.info("Building Unsloth SFT trainer...")
    logger.info(f"Training config: {training_cfg}")
    
    # Patch model to use standard cross-entropy loss
    FastLanguageModel.for_training(model)
    
    # Use Unsloth's optimized trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 2),
            warmup_steps=training_cfg.get("warmup_steps", 5),
            num_train_epochs=training_cfg.get("num_train_epochs", 1),
            learning_rate=training_cfg.get("learning_rate", 5e-5),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=training_cfg.get("output_dir", "./output"),
            save_strategy="epoch",
            save_steps=training_cfg.get("save_steps", 500),
            logging_dir="./logs",
            report_to=None,
            dataloader_drop_last=True,  # Ensure consistent batch sizes
        ),
    )
    
    logger.info("Unsloth SFT trainer built successfully")
    return trainer


def train_and_save(trainer, tokenizer, output_dir, gpu_profiler=None):
    logger.info("Starting model training...")
    
    # Record training start time for tokens/sec calculation
    start_time = time.time()
    
    # Train with automatic metric logging
    result = trainer.train()
    
    # Calculate training duration and tokens/sec
    end_time = time.time()
    training_duration = end_time - start_time
    
    if gpu_profiler and hasattr(trainer, 'train_dataset'):
        # Estimate total tokens processed
        total_samples = len(trainer.train_dataset)
        avg_tokens_per_sample = 256  # Rough estimate
        total_tokens = total_samples * avg_tokens_per_sample
        gpu_profiler.record_tokens_per_sec(total_tokens, training_duration)
    
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
