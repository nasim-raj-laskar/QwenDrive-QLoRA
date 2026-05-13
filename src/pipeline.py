import os
import mlflow
from src.data import load_and_prepare
from src.model import load_tokenizer, load_model
from src.trainer import build_trainer, train_and_save
from src.utils.mlflow import init_mlflow
from src.utils.logger import setup_logger
from src.metrics.metrics import *
from src.metrics.gpu_profiler import GPUProfiler
from src.evaluation import run_evaluation

logger = setup_logger(__name__)

def setup_directories():
    """Create necessary directories."""
    logger.info("Setting up directories...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    logger.info("Directories created successfully")

def log_all_params(model_cfg, lora_cfg, train_cfg):
    """Log all training parameters to MLflow."""
    logger.info("Logging training parameters to MLflow...")
    log_model_params(model_cfg)
    log_quantization_params(model_cfg.get("quantization", {}))
    log_lora_params(lora_cfg)
    log_training_params(train_cfg["training"])
    log_dataset_params(train_cfg["data"])
    logger.info("Parameters logged successfully")

def run_training_pipeline(model_cfg, lora_cfg, train_cfg):
    """Execute the complete training pipeline."""
    logger.info("Starting training pipeline...")
    init_mlflow()
    
    # Initialize GPU profiler
    gpu_profiler = GPUProfiler()
    
    try:
        with mlflow.start_run(run_name="qwen-drive-lora-training"):
            # Start GPU monitoring
            gpu_profiler.start_monitoring()
            
            # Load components
            logger.info("Loading components...")
            model, tokenizer = load_model(model_cfg, lora_cfg)
            dataset = load_and_prepare(train_cfg["data"], tokenizer, save_sample_path="output/training_sample.jsonl")
            
            # Log initial state
            logger.info("Logging initial state...")
            log_model_summary(model, tokenizer)
            log_memory_usage()
            
            # Train
            logger.info("Starting training...")
            trainer = build_trainer(model, tokenizer, dataset, train_cfg["training"])
            train_and_save(trainer, tokenizer, train_cfg["training"]["output_dir"], gpu_profiler)
            
            # Stop GPU monitoring
            gpu_profiler.stop_monitoring()
            
            # Log final artifacts
            logger.info("Logging final artifacts...")
            log_memory_usage()
            
            # Run evaluation
            logger.info("Starting post-training evaluation...")
            eval_results = run_evaluation(model, tokenizer, eval_samples=100, gpu_profiler=gpu_profiler)
            
            logger.info("Evaluation Results:")
            for metric, value in eval_results.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            logger.info("Training pipeline completed successfully")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")
        # Continue without MLflow if it fails
        logger.info("Continuing training without MLflow...")