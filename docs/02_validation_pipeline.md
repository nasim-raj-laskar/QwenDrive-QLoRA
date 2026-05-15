# Validation Pipeline Implementation

## Current State Analysis

### Existing Training Setup

**File**: `src/trainer.py`

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,  # Only training data
    # No eval_dataset parameter
    args=TrainingArguments(
        save_strategy="epoch",
        # No evaluation_strategy
        # No eval_steps
    )
)
```

**Current data split**: `src/data.py`
- Loads 10,000 samples from 44,773 total
- No train/validation/test split
- All samples used for training

### Critical Limitations

1. **No overfitting detection**: Without validation loss, impossible to know if model is memorizing training data
2. **No checkpoint quality estimation**: Can't identify best checkpoint during training
3. **No early stopping**: Training continues even if validation performance degrades
4. **No hyperparameter tuning feedback**: Can't compare learning rates, batch sizes, etc. objectively

---

## Recommended Implementation

### 1. Dataset Splitting Strategy

#### Proposed Split Ratios

| Split | Percentage | Samples (from 10K) | Purpose |
|-------|-----------|-------------------|---------|
| **Train** | 90% | 9,000 | Model training |
| **Validation** | 5% | 500 | Hyperparameter tuning, checkpoint selection |
| **Test** | 5% | 500 | Final evaluation (held-out, never seen during training) |

#### Why These Ratios?

- **90% train**: Sufficient data for fine-tuning (9K samples is substantial for LoRA)
- **5% validation**: Enough for reliable loss estimation without wasting training data
- **5% test**: Reserved for unbiased final evaluation

### 2. Implementation in `src/data.py`

#### Current Code

```python
def load_and_prepare(cfg, tokenizer, save_sample_path=None):
    dataset = load_dataset("json", data_files=cfg["file"])
    split = dataset["train"].shuffle(seed=cfg["shuffle_seed"]).select(range(cfg["sample_size"]))
    # ... formatting ...
    return dataset  # Single dataset
```

#### Proposed Modification

```python
def load_and_prepare(cfg, tokenizer, save_sample_path=None):
    logger.info(f"Loading dataset from {cfg['file']}")
    dataset = load_dataset("json", data_files=cfg["file"])
    
    # Sample and shuffle
    full_data = dataset["train"].shuffle(seed=cfg["shuffle_seed"]).select(range(cfg["sample_size"]))
    
    # Format with chat template
    formatted_data = full_data.map(format_chat)
    
    # Filter by length
    filtered_data = formatted_data.filter(lambda x: 10 <= len(x["input_ids"]) <= 512)
    
    # Split into train/val/test
    splits = filtered_data.train_test_split(
        test_size=0.10,  # 10% for val+test
        seed=cfg["shuffle_seed"]
    )
    
    # Further split the test portion into val and test
    val_test_splits = splits["test"].train_test_split(
        test_size=0.5,  # 50% of 10% = 5% each
        seed=cfg["shuffle_seed"]
    )
    
    train_dataset = splits["train"]  # 90%
    val_dataset = val_test_splits["train"]  # 5%
    test_dataset = val_test_splits["test"]  # 5%
    
    logger.info(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }
```

### 3. Trainer Configuration Updates

#### Modified `src/trainer.py`

```python
def build_trainer(model, tokenizer, datasets, training_cfg):
    """
    Args:
        datasets: Dict with keys "train", "validation", "test"
    """
    logger.info("Building Unsloth SFT trainer with validation...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],  # NEW: explicit train split
        eval_dataset=datasets["validation"],  # NEW: validation split
        dataset_text_field="text",
        max_seq_length=512,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            # Training params
            per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 2),
            num_train_epochs=training_cfg.get("num_train_epochs", 1),
            learning_rate=training_cfg.get("learning_rate", 5e-5),
            
            # NEW: Validation params
            evaluation_strategy="steps",  # Evaluate during training
            eval_steps=100,  # Evaluate every 100 steps
            per_device_eval_batch_size=4,  # Batch size for validation
            
            # NEW: Checkpoint management
            save_strategy="steps",
            save_steps=100,  # Save checkpoint every 100 steps
            save_total_limit=3,  # Keep only best 3 checkpoints
            load_best_model_at_end=True,  # Load best checkpoint after training
            metric_for_best_model="eval_loss",  # Use validation loss
            greater_is_better=False,  # Lower loss is better
            
            # Logging
            logging_steps=10,
            logging_strategy="steps",
            
            # Optimizer
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_steps=training_cfg.get("warmup_steps", 5),
            
            # Precision
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            
            # Output
            output_dir=training_cfg.get("output_dir", "./output"),
            seed=3407,
            report_to=["mlflow"],
        ),
    )
    
    return trainer
```

### 4. Configuration File Updates

#### `configs/training.yaml`

```yaml
data:
  file: "data/automotive_en_dataset.jsonl"
  sample_size: 10000
  shuffle_seed: 42
  system_prompt: "You are an automotive expert assistant."
  
  # NEW: Split configuration
  splits:
    train: 0.90
    validation: 0.05
    test: 0.05

training:
  output_dir: "./output/qwen3b-automotive"
  num_train_epochs: 1
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 0.00005
  
  # NEW: Validation configuration
  evaluation_strategy: "steps"
  eval_steps: 100
  per_device_eval_batch_size: 4
  
  # NEW: Checkpoint configuration
  save_strategy: "steps"
  save_steps: 100
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  
  # Logging
  logging_steps: 10
  
  # Optimizer
  optim: "adamw_8bit"
  lr_scheduler_type: "cosine"
  warmup_steps: 5
  weight_decay: 0.01
  
  # Precision
  bf16: true
  
  # Misc
  max_seq_length: 512
  packing: false
  seed: 3407
```

---

## Validation Loss Monitoring

### Metrics to Track

| Metric | Description | Logged As |
|--------|-------------|-----------|
| **train_loss** | Loss on training batch | `train/loss` |
| **eval_loss** | Loss on validation set | `eval/loss` |
| **learning_rate** | Current LR (cosine schedule) | `train/learning_rate` |
| **grad_norm** | Gradient norm (stability indicator) | `train/grad_norm` |
| **train_runtime** | Training time per epoch | `train/train_runtime` |
| **eval_runtime** | Validation time | `eval/eval_runtime` |

### Automatic MLflow Logging

With `report_to=["mlflow"]`, Transformers automatically logs:
- All training metrics every `logging_steps`
- All validation metrics every `eval_steps`
- Learning rate schedule
- Gradient norms

### Custom Logging Addition

**Modified `src/trainer.py`**:

```python
def train_and_save(trainer, tokenizer, output_dir, gpu_profiler=None):
    logger.info("Starting model training with validation...")
    
    # Custom callback for additional logging
    class ValidationCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                # Log validation metrics to console
                logger.info(f"Step {state.global_step} - Validation Loss: {metrics.get('eval_loss', 'N/A'):.4f}")
                
                # Check for overfitting
                if state.log_history:
                    recent_train_loss = [h.get("loss") for h in state.log_history if "loss" in h][-5:]
                    if recent_train_loss and metrics.get("eval_loss"):
                        avg_train_loss = sum(recent_train_loss) / len(recent_train_loss)
                        if metrics["eval_loss"] > avg_train_loss * 1.5:
                            logger.warning(f"⚠️ Potential overfitting detected: eval_loss ({metrics['eval_loss']:.4f}) >> train_loss ({avg_train_loss:.4f})")
    
    trainer.add_callback(ValidationCallback())
    
    # Train
    result = trainer.train()
    
    # Log final metrics
    logger.info("Training completed")
    logger.info(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    logger.info(f"Best eval loss: {trainer.state.best_metric:.4f}")
    
    # Save best model
    logger.info(f"Saving best model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
```

---

## Overfitting Detection

### Visual Indicators

#### Healthy Training
```
Step 100  | train_loss: 1.234 | eval_loss: 1.256 | ✓ Normal gap
Step 200  | train_loss: 1.102 | eval_loss: 1.118 | ✓ Both decreasing
Step 300  | train_loss: 1.045 | eval_loss: 1.067 | ✓ Converging
```

#### Overfitting Warning
```
Step 100  | train_loss: 1.234 | eval_loss: 1.256 | ✓ Normal gap
Step 200  | train_loss: 0.856 | eval_loss: 1.289 | ⚠️ Eval increasing
Step 300  | train_loss: 0.623 | eval_loss: 1.412 | 🚨 OVERFITTING
```

### Automated Detection Logic

```python
def detect_overfitting(train_losses, eval_losses, threshold=1.3):
    """
    Returns True if eval_loss is significantly higher than train_loss
    
    Args:
        threshold: Ratio above which overfitting is flagged (default 1.3 = 30% higher)
    """
    if not train_losses or not eval_losses:
        return False
    
    recent_train = np.mean(train_losses[-5:])
    recent_eval = np.mean(eval_losses[-5:])
    
    ratio = recent_eval / recent_train
    
    if ratio > threshold:
        logger.warning(f"Overfitting detected: eval/train ratio = {ratio:.2f}")
        return True
    
    return False
```

---

## Early Stopping (Optional)

### Implementation

```python
from transformers import EarlyStoppingCallback

trainer = SFTTrainer(
    # ... other params ...
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,  # Stop if no improvement for 3 evals
            early_stopping_threshold=0.01  # Minimum improvement threshold
        )
    ]
)
```

### Configuration

**`configs/training.yaml`**:

```yaml
training:
  # ... other params ...
  
  # Early stopping (optional)
  early_stopping:
    enabled: false  # Set to true to enable
    patience: 3  # Number of eval steps without improvement
    threshold: 0.01  # Minimum improvement to count as progress
```

---

## Checkpoint Management

### Directory Structure

```
output/qwen3b-automotive/
├── checkpoint-100/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── trainer_state.json
│   └── training_args.bin
├── checkpoint-200/
│   └── ...
├── checkpoint-300/  # Best checkpoint
│   └── ...
└── final_model/  # Copied from best checkpoint
    ├── adapter_model.safetensors
    ├── adapter_config.json
    └── tokenizer files
```

### Automatic Cleanup

With `save_total_limit=3`, only the 3 best checkpoints are retained:
- Checkpoint-100: eval_loss = 1.256
- Checkpoint-200: eval_loss = 1.118 ← kept
- Checkpoint-300: eval_loss = 1.067 ← kept (best)
- Checkpoint-400: eval_loss = 1.089 ← kept
- Checkpoint-500: eval_loss = 1.145 ← replaces checkpoint-100

### Checkpoint Metadata

Each checkpoint includes `trainer_state.json`:

```json
{
  "best_metric": 1.067,
  "best_model_checkpoint": "./output/qwen3b-automotive/checkpoint-300",
  "global_step": 300,
  "epoch": 0.75,
  "log_history": [
    {"loss": 1.234, "eval_loss": 1.256, "step": 100},
    {"loss": 1.102, "eval_loss": 1.118, "step": 200},
    {"loss": 1.045, "eval_loss": 1.067, "step": 300}
  ]
}
```

---

## Integration with Existing Pipeline

### Modified `src/pipeline.py`

```python
def run_pipeline():
    # Load configs
    model_cfg = load_config("configs/model.yaml")
    lora_cfg = load_config("configs/lora.yaml")
    training_cfg = load_config("configs/training.yaml")
    data_cfg = load_config("configs/training.yaml")["data"]
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_cfg, lora_cfg)
    
    # Load and split data
    datasets = load_and_prepare(data_cfg, tokenizer)  # Returns dict with train/val/test
    
    # Build trainer with validation
    trainer = build_trainer(model, tokenizer, datasets, training_cfg["training"])
    
    # Train (automatically uses validation)
    train_and_save(trainer, tokenizer, training_cfg["training"]["output_dir"])
    
    # Evaluate on held-out test set
    test_results = run_evaluation(
        model=trainer.model,
        tokenizer=tokenizer,
        test_dataset=datasets["test"]  # Use held-out test set
    )
    
    mlflow.log_metrics(test_results)
```

---

## Expected Training Output

### Before Implementation
```
Training started...
Step 10  | loss: 1.456
Step 20  | loss: 1.234
Step 30  | loss: 1.123
...
Training complete!
```

### After Implementation
```
Training started with validation...

Step 10  | train_loss: 1.456 | lr: 4.8e-5
Step 20  | train_loss: 1.234 | lr: 4.6e-5
...
Step 100 | train_loss: 1.102 | lr: 4.2e-5
         | 🔍 Evaluating...
         | eval_loss: 1.118 | eval_runtime: 12.3s
         | ✓ New best checkpoint saved!

Step 200 | train_loss: 0.987 | lr: 3.8e-5
         | 🔍 Evaluating...
         | eval_loss: 1.045 | eval_runtime: 12.1s
         | ✓ New best checkpoint saved!

Step 300 | train_loss: 0.923 | lr: 3.2e-5
         | 🔍 Evaluating...
         | eval_loss: 1.067 | eval_runtime: 12.4s
         | ⚠️ Eval loss increased (previous: 1.045)

Training complete!
Best checkpoint: checkpoint-200 (eval_loss: 1.045)
Loading best model for final save...
```

---

## Benefits Summary

| Benefit | Impact |
|---------|--------|
| **Overfitting Detection** | Prevents memorization, ensures generalization |
| **Best Checkpoint Selection** | Automatically saves optimal model state |
| **Hyperparameter Tuning** | Objective comparison of different configs |
| **Training Stability Monitoring** | Early detection of gradient issues |
| **Resource Efficiency** | Stop training when performance plateaus |
| **Reproducibility** | Deterministic splits with fixed seeds |

---

## Implementation Checklist

- [ ] Modify `src/data.py` to return train/val/test splits
- [ ] Update `src/trainer.py` to accept validation dataset
- [ ] Add evaluation_strategy and eval_steps to TrainingArguments
- [ ] Configure checkpoint management (save_total_limit, load_best_model_at_end)
- [ ] Update `configs/training.yaml` with validation params
- [ ] Add validation callback for overfitting detection
- [ ] Modify `src/pipeline.py` to use split datasets
- [ ] Update evaluation to use held-out test set
- [ ] Test with small sample to verify splits work correctly
- [ ] Document validation loss trends in MLflow
