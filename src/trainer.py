from unsloth import is_bfloat16_supported, FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import mlflow
import time
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def build_trainer(model, tokenizer, datasets, training_cfg):
    """
    Args:
        datasets: Dict with keys "train", "validation", "test"
    """
    logger.info("Building Unsloth SFT trainer with validation...")
    logger.info(f"Training config: {training_cfg}")
    
    # Patch model to use standard cross-entropy loss
    FastLanguageModel.for_training(model)
    
    # Use Unsloth's optimized trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        dataset_text_field="text",
        max_seq_length=training_cfg.get("max_seq_length", 512),
        dataset_num_proc=training_cfg.get("dataset_num_proc", 2),
        packing=training_cfg.get("packing", False),
        args=TrainingArguments(
            # Training params
            per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 2),
            num_train_epochs=training_cfg.get("num_train_epochs", 1),
            learning_rate=training_cfg.get("learning_rate", 5e-5),
            
            # Validation params
            eval_strategy=training_cfg.get("eval_strategy", "steps"),
            eval_steps=training_cfg.get("eval_steps", 100),
            per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 4),
            
            # Checkpoint management
            save_strategy=training_cfg.get("save_strategy", "steps"),
            save_steps=training_cfg.get("save_steps", 100),
            save_total_limit=training_cfg.get("save_total_limit", 3),
            load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
            metric_for_best_model=training_cfg.get("metric_for_best_model", "eval_loss"),
            greater_is_better=training_cfg.get("greater_is_better", False),
            
            # Logging
            logging_steps=training_cfg.get("logging_steps", 10),
            logging_strategy=training_cfg.get("logging_strategy", "steps"),
            
            # Optimizer
            optim=training_cfg.get("optim", "adamw_8bit"),
            weight_decay=training_cfg.get("weight_decay", 0.01),
            lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
            warmup_steps=training_cfg.get("warmup_steps", 5),
            
            # Precision
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            
            # Output
            output_dir=training_cfg.get("output_dir", "./output"),
            seed=training_cfg.get("seed", 3407),
            report_to=[],  
            dataloader_drop_last=True,
        ),
    )
    
    logger.info("Unsloth SFT trainer built successfully")
    return trainer


def train_and_save(trainer, tokenizer, output_dir, gpu_profiler=None, training_cfg=None):
    logger.info("Starting model training with validation...")
    
    # Custom callback for overfitting detection
    class ValidationCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                logger.info(f"Step {state.global_step} - Validation Loss: {metrics.get('eval_loss', 'N/A'):.4f}")
                
                # Check for overfitting using config threshold
                if state.log_history:
                    recent_train_loss = [h.get("loss") for h in state.log_history if "loss" in h][-5:]
                    if recent_train_loss and metrics.get("eval_loss"):
                        avg_train_loss = sum(recent_train_loss) / len(recent_train_loss)
                        threshold = 2.0  # Default threshold
                        if metrics["eval_loss"] > avg_train_loss * threshold:
                            logger.warning(f" Potential overfitting detected: eval_loss ({metrics['eval_loss']:.4f}) >> train_loss ({avg_train_loss:.4f})")
    
    trainer.add_callback(ValidationCallback())
    
    # Record training start time for tokens/sec calculation
    start_time = time.time()
    
    # Train with manual metric logging
    result = trainer.train()
    
    # Manually log training metrics since report_to is disabled
    if hasattr(result, 'metrics'):
        for key, value in result.metrics.items():
            mlflow.log_metric(key, value)
    
    # Calculate training duration and tokens/sec
    end_time = time.time()
    training_duration = end_time - start_time
    
    if gpu_profiler and hasattr(trainer, 'train_dataset'):
        # Estimate total tokens processed using config values
        total_samples = len(trainer.train_dataset)
        avg_tokens_per_sample = training_cfg.get("avg_tokens_per_sample", 256)
        total_tokens = total_samples * avg_tokens_per_sample
        gpu_profiler.record_tokens_per_sec(total_tokens, training_duration)
    
    logger.info("Training completed")
    logger.info(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    logger.info(f"Best eval loss: {trainer.state.best_metric:.4f}")
    
    # Log final training metrics
    if hasattr(result, 'metrics'):
        logger.info("Logging final training metrics...")
        for key, value in result.metrics.items():
            mlflow.log_metric(f"final_{key}", value)
            logger.info(f"Final {key}: {value}")
    
    # Save best model
    logger.info(f"Saving best model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model and tokenizer saved successfully")
