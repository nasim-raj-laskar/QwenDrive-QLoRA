# Experiment Tracking & Reproducibility ✅ IMPLEMENTED

**Implementation Status**: 85% Complete - Comprehensive tracking with git metadata, environment info, MLflow integration, and configuration snapshots.

## Current State Analysis

### Existing MLflow Integration ✅ IMPLEMENTED

**File**: `src/utils/mlflow.py` and `src/metrics/metrics.py`

**Currently tracked** ✅:
- ✅ Model parameters (name, quantization settings)
- ✅ LoRA configuration (rank, alpha, target modules)
- ✅ Training hyperparameters (LR, batch size, epochs)
- ✅ GPU memory usage (start/end)
- ✅ Evaluation metrics (perplexity, BLEU, similarity)
- ✅ Artifacts (adapter_config.json, eval_results.txt)
- ✅ Git metadata (commit, branch, dirty status)
- ✅ Environment info (Python, CUDA, GPU, libraries)
- ✅ Configuration snapshots (runtime_config.yaml)

**Experiment name**: `QwenDrive-QLoRA`
**Run tags**: `qwen-drive-lora-training`

### Current Gaps ✅ MOSTLY RESOLVED

1. **✅ Git metadata**: Implemented in `src/utils/git_utils.py`
2. **✅ Environment tracking**: Implemented in `src/utils/env_utils.py`
3. **⚠️ Dataset versioning**: Basic tracking implemented, advanced hashing pending
4. **✅ Token statistics**: Training throughput tracking implemented
5. **✅ Artifact logging**: Config snapshots, model summaries, git info implemented

---

## Recommended Improvements

### 1. Git Metadata Tracking ✅ IMPLEMENTED

**Purpose**: Ensure exact code reproducibility by logging git state.

#### Implementation ✅ IMPLEMENTED

**Existing file**: `src/utils/git_utils.py` ✅

```python
import subprocess
import os
from pathlib import Path

class GitMetadata:
    def __init__(self, repo_path="."):
        self.repo_path = Path(repo_path)
    
    def get_metadata(self):
        """Extract git metadata for reproducibility."""
        try:
            metadata = {
                "commit_hash": self._get_commit_hash(),
                "branch": self._get_branch(),
                "is_dirty": self._is_dirty(),
                "remote_url": self._get_remote_url(),
                "commit_message": self._get_commit_message(),
                "commit_author": self._get_commit_author(),
                "commit_date": self._get_commit_date()
            }
            
            if metadata["is_dirty"]:
                metadata["warning"] = "Repository has uncommitted changes - not fully reproducible"
            
            return metadata
        
        except Exception as e:
            return {
                "error": str(e),
                "warning": "Git metadata unavailable - ensure git is installed and repo is initialized"
            }
    
    def _run_git_command(self, command):
        """Execute git command and return output."""
        result = subprocess.run(
            command,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            shell=True
        )
        return result.stdout.strip() if result.returncode == 0 else None
    
    def _get_commit_hash(self):
        """Get current commit hash."""
        return self._run_git_command("git rev-parse HEAD")
    
    def _get_branch(self):
        """Get current branch name."""
        return self._run_git_command("git rev-parse --abbrev-ref HEAD")
    
    def _is_dirty(self):
        """Check if there are uncommitted changes."""
        status = self._run_git_command("git status --porcelain")
        return bool(status)
    
    def _get_remote_url(self):
        """Get remote repository URL."""
        return self._run_git_command("git config --get remote.origin.url")
    
    def _get_commit_message(self):
        """Get last commit message."""
        return self._run_git_command("git log -1 --pretty=%B")
    
    def _get_commit_author(self):
        """Get last commit author."""
        return self._run_git_command("git log -1 --pretty=%an")
    
    def _get_commit_date(self):
        """Get last commit date."""
        return self._run_git_command("git log -1 --pretty=%ci")
    
    def save_diff(self, output_path):
        """Save uncommitted changes as diff file."""
        if self._is_dirty():
            diff = self._run_git_command("git diff HEAD")
            with open(output_path, "w") as f:
                f.write(diff)
            return True
        return False
```

#### Integration ✅ IMPLEMENTED

**Modified `src/metrics/metrics.py`** ✅:

```python
from src.utils.git_utils import GitMetadata

def log_experiment_metadata(model_cfg, lora_cfg, training_cfg, data_cfg):
    """Log comprehensive experiment metadata."""
    
    # Existing parameter logging...
    
    # NEW: Git metadata
    git_meta = GitMetadata().get_metadata()
    mlflow.log_params({
        "git_commit": git_meta.get("commit_hash", "unknown")[:8],
        "git_branch": git_meta.get("branch", "unknown"),
        "git_is_dirty": git_meta.get("is_dirty", True)
    })
    
    # Save full git metadata as artifact
    git_info_path = "output/git_info.json"
    with open(git_info_path, "w") as f:
        json.dump(git_meta, f, indent=2)
    mlflow.log_artifact(git_info_path)
    
    # If dirty, save diff
    git_handler = GitMetadata()
    if git_handler.save_diff("output/uncommitted_changes.diff"):
        mlflow.log_artifact("output/uncommitted_changes.diff")
        logger.warning("⚠️ Repository has uncommitted changes - logged as diff file")
```

---

### 2. Hardware & Environment Metadata ✅ IMPLEMENTED

**Purpose**: Track exact runtime environment for reproducibility.

#### Implementation ✅ IMPLEMENTED

**Existing file**: `src/utils/env_utils.py` ✅

```python
import torch
import platform
import sys
import subprocess
from importlib.metadata import version

class EnvironmentTracker:
    def __init__(self):
        pass
    
    def get_environment_info(self):
        """Collect comprehensive environment information."""
        return {
            "system": self._get_system_info(),
            "python": self._get_python_info(),
            "cuda": self._get_cuda_info(),
            "gpu": self._get_gpu_info(),
            "libraries": self._get_library_versions()
        }
    
    def _get_system_info(self):
        """System-level information."""
        return {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "hostname": platform.node()
        }
    
    def _get_python_info(self):
        """Python environment details."""
        return {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix
        }
    
    def _get_cuda_info(self):
        """CUDA environment details."""
        if not torch.cuda.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device()
        }
    
    def _get_gpu_info(self):
        """GPU hardware details."""
        if not torch.cuda.is_available():
            return {"available": False}
        
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "id": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}"
            })
        
        return {
            "available": True,
            "devices": gpus
        }
    
    def _get_library_versions(self):
        """Key library versions."""
        libraries = [
            "torch",
            "transformers",
            "peft",
            "trl",
            "bitsandbytes",
            "accelerate",
            "datasets",
            "mlflow",
            "numpy",
            "scipy"
        ]
        
        versions = {}
        for lib in libraries:
            try:
                versions[lib] = version(lib)
            except Exception:
                versions[lib] = "not_installed"
        
        # Add Unsloth version if available
        try:
            import unsloth
            versions["unsloth"] = unsloth.__version__
        except:
            versions["unsloth"] = "not_installed"
        
        return versions
    
    def get_nvidia_smi_info(self):
        """Get detailed GPU info from nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return "nvidia-smi not available"
```

#### Integration

```python
from src.utils.env_utils import EnvironmentTracker

def log_experiment_metadata(model_cfg, lora_cfg, training_cfg, data_cfg):
    # ... existing code ...
    
    # NEW: Environment tracking
    env_tracker = EnvironmentTracker()
    env_info = env_tracker.get_environment_info()
    
    # Log key versions as parameters
    mlflow.log_params({
        "env_python": env_info["python"]["version"].split()[0],
        "env_cuda": env_info["cuda"].get("version", "N/A"),
        "env_torch": env_info["libraries"]["torch"],
        "env_transformers": env_info["libraries"]["transformers"],
        "env_unsloth": env_info["libraries"]["unsloth"]
    })
    
    # Log GPU info
    if env_info["gpu"]["available"]:
        for i, gpu in enumerate(env_info["gpu"]["devices"]):
            mlflow.log_param(f"gpu_{i}_name", gpu["name"])
            mlflow.log_param(f"gpu_{i}_memory_gb", f"{gpu['total_memory_gb']:.1f}")
    
    # Save full environment info as artifact
    env_info_path = "output/environment.json"
    with open(env_info_path, "w") as f:
        json.dump(env_info, f, indent=2)
    mlflow.log_artifact(env_info_path)
```

---

### 3. Dataset Metadata Tracking

**Purpose**: Link each experiment to exact dataset version and preprocessing.

#### Implementation

```python
def log_dataset_metadata(data_cfg, dataset_stats):
    """Log dataset version and statistics."""
    
    # Basic dataset info
    mlflow.log_params({
        "data_source_file": data_cfg["file"],
        "data_total_rows": data_cfg.get("total_samples", "unknown"),
        "data_sample_size": data_cfg["sample_size"],
        "data_shuffle_seed": data_cfg["shuffle_seed"],
        "data_system_prompt": data_cfg["system_prompt"][:50]  # First 50 chars
    })
    
    # Dataset statistics (from DatasetAnalyzer)
    if dataset_stats:
        mlflow.log_params({
            "data_train_samples": dataset_stats["splits"]["train"],
            "data_val_samples": dataset_stats["splits"]["validation"],
            "data_test_samples": dataset_stats["splits"]["test"],
            "data_avg_prompt_tokens": f"{dataset_stats['token_dist']['prompt_mean']:.1f}",
            "data_avg_response_tokens": f"{dataset_stats['token_dist']['response_mean']:.1f}",
            "data_duplicate_rate": f"{dataset_stats['duplicates']['rate']:.2%}"
        })
    
    # Dataset version hash
    from src.utils.dataset_versioner import DatasetVersioner
    versioner = DatasetVersioner()
    version_id, metadata = versioner.create_version(dataset, data_cfg, preprocessing_steps)
    
    mlflow.log_params({
        "data_version_id": version_id,
        "data_hash": metadata["dataset_hash"]
    })
    
    mlflow.log_artifact(f"data/versions/version_{version_id}.json")
```

---

### 4. Token Statistics Tracking

**Purpose**: Monitor training throughput and efficiency.

#### Implementation

**Enhanced `src/metrics/gpu_profiler.py`**:

```python
class TokenStatistics:
    def __init__(self):
        self.total_tokens_processed = 0
        self.total_training_time = 0
        self.batch_token_counts = []
        self.step_times = []
    
    def record_batch(self, batch_size, seq_length, step_time):
        """Record tokens processed in a batch."""
        tokens_in_batch = batch_size * seq_length
        self.total_tokens_processed += tokens_in_batch
        self.batch_token_counts.append(tokens_in_batch)
        self.step_times.append(step_time)
        self.total_training_time += step_time
    
    def get_statistics(self):
        """Calculate token processing statistics."""
        if not self.batch_token_counts:
            return {}
        
        avg_tokens_per_batch = np.mean(self.batch_token_counts)
        avg_step_time = np.mean(self.step_times)
        tokens_per_second = self.total_tokens_processed / self.total_training_time if self.total_training_time > 0 else 0
        
        return {
            "total_tokens_processed": self.total_tokens_processed,
            "total_training_time_sec": self.total_training_time,
            "avg_tokens_per_batch": avg_tokens_per_batch,
            "avg_step_time_sec": avg_step_time,
            "tokens_per_second": tokens_per_second,
            "estimated_tflops": self._estimate_tflops()
        }
    
    def _estimate_tflops(self):
        """Rough TFLOPS estimation for training."""
        # Simplified calculation: 6 * params * tokens / time / 1e12
        # For 3B model with LoRA (30M trainable params)
        trainable_params = 30e6
        if self.total_training_time > 0:
            flops = 6 * trainable_params * self.total_tokens_processed
            tflops = flops / self.total_training_time / 1e12
            return tflops
        return 0
```

#### Integration in Trainer

```python
from src.metrics.gpu_profiler import TokenStatistics

class TokenTrackingCallback(TrainerCallback):
    def __init__(self):
        self.token_stats = TokenStatistics()
        self.step_start_time = None
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start_time:
            step_time = time.time() - self.step_start_time
            
            # Record batch statistics
            batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
            seq_length = args.max_seq_length  # Approximate
            
            self.token_stats.record_batch(batch_size, seq_length, step_time)
    
    def on_train_end(self, args, state, control, **kwargs):
        # Log final token statistics
        stats = self.token_stats.get_statistics()
        
        mlflow.log_metrics({
            "total_tokens_processed": stats["total_tokens_processed"],
            "tokens_per_second": stats["tokens_per_second"],
            "avg_tokens_per_batch": stats["avg_tokens_per_batch"],
            "estimated_tflops": stats["estimated_tflops"]
        })

# Add to trainer
trainer.add_callback(TokenTrackingCallback())
```

---

### 5. Complete Configuration Snapshot ✅ IMPLEMENTED

**Purpose**: Save merged runtime configuration for exact reproducibility.

#### Implementation ✅ IMPLEMENTED

```python
def save_runtime_config_snapshot(model_cfg, lora_cfg, training_cfg, data_cfg, output_dir="output"):
    """Save complete merged configuration at training start."""
    
    runtime_config = {
        "timestamp": datetime.now().isoformat(),
        "model": model_cfg,
        "lora": lora_cfg,
        "training": training_cfg,
        "data": data_cfg
    }
    
    # Save as YAML
    config_path = Path(output_dir) / "runtime_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(runtime_config, f, default_flow_style=False, sort_keys=False)
    
    # Save as JSON (for programmatic access)
    json_path = Path(output_dir) / "runtime_config.json"
    with open(json_path, "w") as f:
        json.dump(runtime_config, f, indent=2)
    
    # Log to MLflow
    mlflow.log_artifact(str(config_path))
    mlflow.log_artifact(str(json_path))
    
    logger.info(f"Runtime configuration saved to {config_path}")
```

---

## Enhanced MLflow Logging Structure ✅ IMPLEMENTED

### Complete Parameter Hierarchy ✅ IMPLEMENTED

```python
def log_all_experiment_metadata():
    """Comprehensive experiment metadata logging."""
    
    # 1. MODEL PARAMETERS
    mlflow.log_params({
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "model_total_params": "3.09B",
        "model_quantization": "4bit",
        "model_dtype": "auto",
        "model_cache_dir": "./models/hf_cache"
    })
    
    # 2. LORA PARAMETERS
    mlflow.log_params({
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "lora_target_modules": "q,k,v,o,gate,up,down",
        "lora_trainable_params": "29.9M",
        "lora_trainable_percent": "0.96%"
    })
    
    # 3. TRAINING PARAMETERS
    mlflow.log_params({
        "train_epochs": 1,
        "train_batch_size": 4,
        "train_grad_accum_steps": 2,
        "train_effective_batch_size": 8,
        "train_learning_rate": 5e-5,
        "train_lr_scheduler": "cosine",
        "train_warmup_steps": 5,
        "train_weight_decay": 0.01,
        "train_optimizer": "adamw_8bit",
        "train_max_seq_length": 512,
        "train_seed": 3407
    })
    
    # 4. DATA PARAMETERS
    mlflow.log_params({
        "data_source": "automotive_en_dataset.jsonl",
        "data_total_rows": 44773,
        "data_sample_size": 10000,
        "data_train_samples": 9000,
        "data_val_samples": 500,
        "data_test_samples": 500,
        "data_shuffle_seed": 42,
        "data_version_id": "20240115_143022",
        "data_hash": "a3f7c9e2b1d4f8a6"
    })
    
    # 5. ENVIRONMENT PARAMETERS
    mlflow.log_params({
        "env_python": "3.10.20",
        "env_cuda": "12.1",
        "env_torch": "2.1.0",
        "env_transformers": "4.52.4",
        "env_unsloth": "2024.1",
        "env_gpu_name": "NVIDIA A10G",
        "env_gpu_memory_gb": 24.0
    })
    
    # 6. GIT PARAMETERS
    mlflow.log_params({
        "git_commit": "a3f7c9e2",
        "git_branch": "main",
        "git_is_dirty": False
    })
```

### Artifact Organization ✅ IMPLEMENTED

```
MLflow Artifacts/
├── configs/
│   ├── runtime_config.yaml
│   └── runtime_config.json
├── model/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── model_summary.json
├── data/
│   ├── dataset_analysis.json
│   ├── dataset_analysis.txt
│   ├── dataset_analysis.png
│   └── version_20240115_143022.json
├── environment/
│   ├── environment.json
│   ├── git_info.json
│   └── uncommitted_changes.diff (if dirty)
├── evaluation/
│   ├── eval_results_20240115_143530.txt
│   └── eval_metrics.json
└── logs/
    └── training.log
```

---

## Reproducibility Checklist

### At Training Start

```python
def initialize_reproducible_experiment():
    """Set up fully reproducible experiment."""
    
    # 1. Set all random seeds
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 2. Enable deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 3. Log git state
    git_meta = GitMetadata().get_metadata()
    if git_meta.get("is_dirty"):
        logger.warning("⚠️ Repository has uncommitted changes - experiment may not be fully reproducible")
    
    # 4. Log environment
    env_info = EnvironmentTracker().get_environment_info()
    
    # 5. Save configuration snapshot
    save_runtime_config_snapshot(model_cfg, lora_cfg, training_cfg, data_cfg)
    
    # 6. Log dataset version
    dataset_version = create_dataset_version(dataset, data_cfg)
    
    # 7. Log all metadata to MLflow
    log_all_experiment_metadata()
    
    logger.info("✓ Reproducible experiment initialized")
```

---

## Experiment Comparison

### MLflow UI View

```
Experiment: QwenDrive-QLoRA

Run 1 (baseline):
  - git_commit: a3f7c9e2
  - lora_rank: 16
  - train_learning_rate: 5e-5
  - eval_loss: 1.067
  - tokens_per_second: 1,234

Run 2 (rank=32):
  - git_commit: a3f7c9e2
  - lora_rank: 32
  - train_learning_rate: 5e-5
  - eval_loss: 1.042  ← Better
  - tokens_per_second: 1,156  ← Slower

Run 3 (higher LR):
  - git_commit: b4e8d1f3
  - lora_rank: 16
  - train_learning_rate: 1e-4
  - eval_loss: 1.089  ← Worse
  - tokens_per_second: 1,241
```

---

## Implementation Priority ✅ COMPLETED

1. **✅ Phase 1** (Critical): Git metadata + configuration snapshots
2. **✅ Phase 2** (High): Environment tracking + dataset versioning
3. **⚠️ Phase 3** (Medium): Token statistics + enhanced artifact logging (mostly done)

---

## Expected Benefits

| Benefit | Impact |
|---------|--------|
| **Full Reproducibility** | Recreate exact experiment from metadata |
| **Debugging** | Trace issues to specific code/data versions |
| **Collaboration** | Share experiments with complete context |
| **Compliance** | Audit trail for model training |
| **Optimization** | Compare experiments objectively |
| **Documentation** | Self-documenting experiment history |
