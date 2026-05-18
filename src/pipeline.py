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
from src.analysis import DatasetAnalyzer, DatasetVersioner

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
            datasets = load_and_prepare(train_cfg["data"], tokenizer, save_sample_path="output/training_sample.jsonl")
            
            # NEW: Analyze dataset before training
            logger.info("Analyzing dataset quality and characteristics...")
            analyzer = DatasetAnalyzer(
                dataset=datasets["train"],
                tokenizer=tokenizer,
                output_dir="output/data_analysis"
            )
            
            analysis_stats = analyzer.analyze_all()
            
            # Create dataset version
            versioner = DatasetVersioner()
            version_id, version_metadata = versioner.create_version(
                dataset=datasets["train"],
                config=train_cfg["data"],
                preprocessing_steps=[
                    "load_jsonl",
                    f"shuffle_seed_{train_cfg['data']['shuffle_seed']}",
                    f"sample_{train_cfg['data']['sample_size']}",
                    "format_chat_template",
                    f"filter_length_{train_cfg['data']['min_seq_length']}_{train_cfg['data']['max_seq_length']}",
                    f"split_train_{train_cfg['data']['splits']['train']}_val_{train_cfg['data']['splits']['validation']}_test_{train_cfg['data']['splits']['test']}"
                ]
            )
            
            # Log initial state
            logger.info("Logging initial state...")
            log_all_params(model_cfg, lora_cfg, train_cfg)
            log_model_summary(model, tokenizer)
            log_memory_usage()
            
            # Log dataset analysis results
            logger.info("Logging dataset analysis results...")
            log_dataset_analysis(analysis_stats)
            log_dataset_version(version_metadata)
            
            # Log analysis artifacts
            mlflow.log_artifacts("output/data_analysis")
            mlflow.log_artifacts("data/versions")
            
            # Log git metadata and save config snapshot
            from src.metrics.metrics import log_git_metadata, save_config_snapshot, log_environment_info
            log_git_metadata()
            log_environment_info()
            save_config_snapshot(model_cfg, lora_cfg, train_cfg["training"], train_cfg["data"])
            
            # Train
            logger.info("Starting training...")
            trainer = build_trainer(model, tokenizer, datasets, train_cfg["training"])
            train_and_save(trainer, tokenizer, train_cfg["training"]["output_dir"], gpu_profiler, train_cfg["training"])
            
            # Stop GPU monitoring
            gpu_profiler.stop_monitoring()
            
            # Log final artifacts
            logger.info("Logging final artifacts...")
            log_adapter_config(train_cfg["training"]["output_dir"])
            log_memory_usage()
            
            # Run evaluation on held-out test set
            logger.info("Starting post-training evaluation on test set...")
            eval_samples = train_cfg.get("eval_samples", 100)
            
            # Load base model for pairwise comparison if enabled
            base_model = None
            if train_cfg.get("pairwise_eval", {}).get("enabled"):
                logger.info("Loading base model for pairwise comparison...")
                from unsloth import FastLanguageModel
                base_model, _ = FastLanguageModel.from_pretrained(
                    model_name=model_cfg["name"],
                    max_seq_length=model_cfg["max_seq_length"],
                    dtype=None,
                    load_in_4bit=True,
                )
            
            eval_results = run_evaluation(model, tokenizer, eval_samples=eval_samples, base_model=base_model, gpu_profiler=gpu_profiler)
            
            logger.info("Evaluation Results:")
            for metric, value in eval_results.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            logger.info("Training pipeline completed successfully")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")
        # Continue without MLflow if it fails
        logger.info("Continuing training without MLflow...")