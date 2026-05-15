# Implementation Roadmap

## Overview

This document provides a phased implementation plan for evolving the QwenDrive fine-tuning pipeline from a training system into a robust experimentation and evaluation architecture.

**Current State**: Functional training pipeline with basic evaluation
**Target State**: Production-grade LLM experimentation platform

---

## Phase 1: Foundation (Week 1-2)

### Priority: Critical
### Goal: Establish reproducibility and validation infrastructure

#### 1.1 Validation Pipeline

**Estimated Time**: 3-4 days

**Tasks**:
- [ ] Modify `src/data.py` to implement train/val/test splits (90/5/5)
- [ ] Update `src/trainer.py` to accept validation dataset
- [ ] Add `evaluation_strategy="steps"` and `eval_steps=100` to TrainingArguments
- [ ] Implement checkpoint management (`save_total_limit=3`, `load_best_model_at_end=True`)
- [ ] Update `configs/training.yaml` with validation parameters
- [ ] Test validation loss logging in MLflow

**Success Criteria**:
- Training logs show both train_loss and eval_loss
- Best checkpoint is automatically selected based on validation loss
- Overfitting is detectable from loss curves

**Files to Modify**:
- `src/data.py`
- `src/trainer.py`
- `configs/training.yaml`

**Testing**:
```bash
# Run training with small sample to verify splits
python train.py --sample-size 1000 --epochs 1
# Check MLflow for eval_loss metrics
# Verify checkpoint-* directories are created
```

---

#### 1.2 Git Metadata Tracking

**Estimated Time**: 1-2 days

**Tasks**:
- [ ] Create `src/utils/git_utils.py` with GitMetadata class
- [ ] Implement commit hash, branch, dirty state detection
- [ ] Add git metadata logging to `src/metrics/metrics.py`
- [ ] Save uncommitted changes as diff file if dirty
- [ ] Log git info as MLflow artifact

**Success Criteria**:
- Every MLflow run includes git_commit, git_branch, git_is_dirty parameters
- Uncommitted changes are saved as diff file
- Warning logged if repository is dirty

**Files to Create**:
- `src/utils/git_utils.py`

**Files to Modify**:
- `src/metrics/metrics.py`

**Testing**:
```bash
# Test with clean repo
git commit -am "test commit"
python train.py
# Verify git_commit in MLflow

# Test with dirty repo
echo "test" >> test.txt
python train.py
# Verify warning and diff file in artifacts
```

---

#### 1.3 Configuration Snapshots

**Estimated Time**: 1 day

**Tasks**:
- [ ] Create function to merge all config files into single snapshot
- [ ] Save runtime_config.yaml and runtime_config.json at training start
- [ ] Log config snapshots as MLflow artifacts
- [ ] Add timestamp to config snapshot

**Success Criteria**:
- `output/runtime_config.yaml` contains merged configuration
- Config snapshot logged to MLflow artifacts
- Can reproduce exact training setup from snapshot

**Files to Modify**:
- `src/pipeline.py`
- `src/metrics/metrics.py`

---

## Phase 2: Data Quality (Week 3-4)

### Priority: High
### Goal: Understand and improve dataset quality

#### 2.1 Dataset Statistics Module

**Estimated Time**: 3-4 days

**Tasks**:
- [ ] Create `src/data_analysis.py` with DatasetAnalyzer class
- [ ] Implement token distribution analysis
- [ ] Implement length distribution analysis
- [ ] Generate visualization plots (histograms, distributions)
- [ ] Save analysis reports as JSON and text
- [ ] Integrate into pipeline before training
- [ ] Log analysis results to MLflow

**Success Criteria**:
- `output/data_analysis/` contains reports and plots
- Know exact token length distributions
- Understand prompt/response length patterns
- Analysis logged to MLflow artifacts

**Files to Create**:
- `src/data_analysis.py`

**Files to Modify**:
- `src/pipeline.py`

**Testing**:
```bash
python train.py
# Check output/data_analysis/ for reports
# Verify plots are generated
# Check MLflow artifacts
```

---

#### 2.2 Duplicate Detection

**Estimated Time**: 2-3 days

**Tasks**:
- [ ] Implement exact duplicate detection (hash-based)
- [ ] Implement fuzzy duplicate detection (TF-IDF + cosine similarity)
- [ ] Add duplicate statistics to analysis report
- [ ] Flag duplicate samples for review
- [ ] Add option to filter duplicates in config

**Success Criteria**:
- Know exact duplicate count and rate
- Can identify near-duplicate samples
- Option to automatically filter duplicates

**Files to Modify**:
- `src/data_analysis.py`
- `configs/training.yaml`

---

#### 2.3 Data Quality Scoring

**Estimated Time**: 2 days

**Tasks**:
- [ ] Create DataQualityScorer class
- [ ] Implement quality checks (empty, short, repetitive, malformed)
- [ ] Score each sample (0-100)
- [ ] Generate flagged samples report
- [ ] Add quality threshold filtering option

**Success Criteria**:
- Each sample has quality score
- Low-quality samples are flagged
- Can filter by quality threshold

**Files to Modify**:
- `src/data_analysis.py`

---

#### 2.4 Dataset Versioning

**Estimated Time**: 2 days

**Tasks**:
- [ ] Create `src/utils/dataset_versioner.py`
- [ ] Implement dataset hash computation
- [ ] Create version metadata structure
- [ ] Save version snapshots to `data/versions/`
- [ ] Log dataset version to MLflow

**Success Criteria**:
- Each training run linked to dataset version
- Can reproduce exact dataset from version ID
- Dataset hash tracked in MLflow

**Files to Create**:
- `src/utils/dataset_versioner.py`

**Files to Modify**:
- `src/pipeline.py`
- `src/metrics/metrics.py`

---

## Phase 3: Advanced Evaluation (Week 5-6)

### Priority: High
### Goal: Implement robust evaluation methodology

#### 3.1 Structured Prompt Categories

**Estimated Time**: 2-3 days

**Tasks**:
- [ ] Create `data/eval_prompts.jsonl` with categorized prompts
- [ ] Define categories: factual_qa, troubleshooting, instruction_following, conversational, edge_cases, safety
- [ ] Create 20-30 prompts per category
- [ ] Create `src/metrics/category_eval.py`
- [ ] Implement per-category evaluation
- [ ] Log category-specific metrics to MLflow

**Success Criteria**:
- Evaluation broken down by prompt category
- Can identify model strengths/weaknesses per category
- Category metrics logged to MLflow

**Files to Create**:
- `data/eval_prompts.jsonl`
- `src/metrics/category_eval.py`

**Files to Modify**:
- `src/evaluation.py`
- `configs/eval.yaml`

---

#### 3.2 Pairwise Comparison Evaluation

**Estimated Time**: 3-4 days

**Tasks**:
- [ ] Create `src/metrics/pairwise_eval.py`
- [ ] Implement base model loading
- [ ] Generate responses from both base and fine-tuned models
- [ ] Implement comparison logic (manual or LLM-based)
- [ ] Calculate win/loss/tie rates
- [ ] Log pairwise metrics to MLflow

**Success Criteria**:
- Can compare base vs fine-tuned model
- Know win rate and regression rate
- Identify where fine-tuning helped/hurt

**Files to Create**:
- `src/metrics/pairwise_eval.py`

**Files to Modify**:
- `src/evaluation.py`
- `configs/eval.yaml`

---

#### 3.3 LLM-as-a-Judge (Optional)

**Estimated Time**: 3-5 days

**Tasks**:
- [ ] Create `src/metrics/llm_judge.py`
- [ ] Implement judge model integration (GPT-4, Claude, or local)
- [ ] Design scoring rubric (helpfulness, correctness, coherence, etc.)
- [ ] Implement structured scoring
- [ ] Log judge scores to MLflow
- [ ] Add cost tracking for API-based judges

**Success Criteria**:
- Responses scored on multiple dimensions
- Judge reasoning captured
- Scores logged to MLflow

**Files to Create**:
- `src/metrics/llm_judge.py`

**Files to Modify**:
- `src/evaluation.py`
- `configs/eval.yaml`

**Note**: This is optional and can be deferred if API costs are a concern. Start with local models (Qwen2.5-72B) or GPT-3.5-turbo for cost efficiency.

---

## Phase 4: Environment & Monitoring (Week 7)

### Priority: Medium
### Goal: Complete reproducibility and monitoring

#### 4.1 Environment Tracking

**Estimated Time**: 2 days

**Tasks**:
- [ ] Create `src/utils/env_utils.py` with EnvironmentTracker
- [ ] Collect system info (OS, CPU, RAM)
- [ ] Collect Python environment (version, packages)
- [ ] Collect CUDA/GPU info
- [ ] Collect library versions (torch, transformers, unsloth, etc.)
- [ ] Log environment metadata to MLflow
- [ ] Save environment.json as artifact

**Success Criteria**:
- Complete environment snapshot for each run
- Can identify environment-related issues
- Environment info logged to MLflow

**Files to Create**:
- `src/utils/env_utils.py`

**Files to Modify**:
- `src/metrics/metrics.py`

---

#### 4.2 Token Statistics Tracking

**Estimated Time**: 2 days

**Tasks**:
- [ ] Enhance `src/metrics/gpu_profiler.py` with TokenStatistics class
- [ ] Track tokens processed per batch
- [ ] Calculate tokens/second throughput
- [ ] Estimate TFLOPS
- [ ] Create TokenTrackingCallback for trainer
- [ ] Log token statistics to MLflow

**Success Criteria**:
- Know exact tokens processed during training
- Track training throughput
- Compare efficiency across runs

**Files to Modify**:
- `src/metrics/gpu_profiler.py`
- `src/trainer.py`

---

## Phase 5: Architecture Refactor (Week 8+)

### Priority: Low
### Goal: Improve code organization and scalability

#### 5.1 Module Reorganization

**Estimated Time**: 3-5 days

**Tasks**:
- [ ] Create `src/data/` module (data loading, preprocessing, analysis)
- [ ] Create `src/model/` module (model loading, LoRA setup)
- [ ] Create `src/training/` module (trainer, callbacks)
- [ ] Create `src/evaluation/` module (all evaluation logic)
- [ ] Create `src/tracking/` module (MLflow, versioning)
- [ ] Create `src/monitoring/` module (GPU profiler, token stats)
- [ ] Update imports across codebase

**Success Criteria**:
- Clear separation of concerns
- Easier to navigate codebase
- Improved maintainability

**New Structure**:
```
src/
├── data/
│   ├── loader.py
│   ├── preprocessor.py
│   ├── analyzer.py
│   └── versioner.py
├── model/
│   ├── loader.py
│   └── lora.py
├── training/
│   ├── trainer.py
│   └── callbacks.py
├── evaluation/
│   ├── evaluator.py
│   ├── category_eval.py
│   ├── pairwise_eval.py
│   └── llm_judge.py
├── tracking/
│   ├── mlflow_logger.py
│   ├── git_tracker.py
│   └── env_tracker.py
├── monitoring/
│   ├── gpu_profiler.py
│   └── token_stats.py
└── utils/
    └── logger.py
```

---

## Testing Strategy

### Unit Tests

Create `tests/` directory with unit tests for each module:

```
tests/
├── test_data_analysis.py
├── test_duplicate_detection.py
├── test_quality_scoring.py
├── test_git_utils.py
├── test_env_utils.py
└── test_evaluation.py
```

### Integration Tests

Test end-to-end pipeline with small sample:

```bash
# Quick integration test
python train.py --sample-size 100 --epochs 1 --eval-samples 10

# Verify:
# - Data splits created correctly
# - Validation loss logged
# - Git metadata captured
# - Config snapshot saved
# - Dataset analysis generated
# - Evaluation completed
# - All artifacts logged to MLflow
```

### Regression Tests

Before each phase, run baseline test:

```bash
# Baseline test
python train.py --sample-size 1000 --epochs 1
# Record metrics

# After changes
python train.py --sample-size 1000 --epochs 1
# Compare metrics - should be similar
```

---

## Rollout Strategy

### Incremental Deployment

1. **Implement in feature branch**: Each phase in separate branch
2. **Test thoroughly**: Run integration tests
3. **Merge to main**: Only after validation
4. **Document changes**: Update README and docs

### Backward Compatibility

- Keep existing functionality working
- Add new features as opt-in (config flags)
- Deprecate old features gradually

### Example Config Flags

```yaml
# configs/training.yaml
features:
  validation_split: true  # Phase 1
  data_analysis: true     # Phase 2
  category_eval: true     # Phase 3
  pairwise_eval: false    # Phase 3 (opt-in)
  llm_judge: false        # Phase 3 (opt-in, requires API key)
  token_tracking: true    # Phase 4
```

---

## Success Metrics

### Phase 1 Success
- [ ] Validation loss logged for every run
- [ ] Best checkpoint automatically selected
- [ ] Git commit tracked for every run
- [ ] Config snapshot saved for every run

### Phase 2 Success
- [ ] Dataset analysis report generated before training
- [ ] Duplicate rate known and logged
- [ ] Quality scores computed for all samples
- [ ] Dataset version tracked for every run

### Phase 3 Success
- [ ] Evaluation broken down by category
- [ ] Base vs fine-tuned comparison available
- [ ] Multiple evaluation dimensions tracked

### Phase 4 Success
- [ ] Complete environment snapshot for every run
- [ ] Token throughput tracked
- [ ] Full reproducibility achieved

---

## Risk Mitigation

### Risk 1: Breaking Existing Pipeline

**Mitigation**:
- Implement changes incrementally
- Test each phase thoroughly
- Keep feature flags for new functionality
- Maintain backward compatibility

### Risk 2: Increased Training Time

**Mitigation**:
- Make data analysis optional (run once, cache results)
- Optimize duplicate detection (sample-based for large datasets)
- Run evaluation on subset of data

### Risk 3: Storage Overhead

**Mitigation**:
- Compress artifacts before logging
- Set retention policies in MLflow
- Clean up old checkpoints automatically

### Risk 4: API Costs (LLM-as-a-Judge)

**Mitigation**:
- Start with local models (Qwen2.5-72B)
- Use GPT-3.5-turbo instead of GPT-4
- Limit evaluation samples
- Make LLM judge optional

---

## Resource Requirements

### Compute
- No additional GPU requirements (same as current)
- Slightly longer training time due to validation (~10% overhead)
- Data analysis runs on CPU (minimal impact)

### Storage
- Dataset analysis: ~10 MB per run
- Config snapshots: ~1 KB per run
- Environment metadata: ~5 KB per run
- Git metadata: ~2 KB per run
- Total overhead: ~15-20 MB per run

### Development Time
- Phase 1: 1-2 weeks
- Phase 2: 2-3 weeks
- Phase 3: 2-3 weeks
- Phase 4: 1 week
- Phase 5: 1-2 weeks (optional)
- **Total**: 7-11 weeks for full implementation

---

## Quick Start Guide

### Minimal Implementation (1 week)

If time is limited, implement these critical features first:

1. **Validation split** (Phase 1.1) - 3 days
2. **Git tracking** (Phase 1.2) - 1 day
3. **Config snapshots** (Phase 1.3) - 1 day
4. **Basic data analysis** (Phase 2.1, simplified) - 2 days

This gives you:
- Overfitting detection
- Reproducibility
- Data visibility

### Recommended Implementation (4 weeks)

For maximum impact with reasonable effort:

- Complete Phase 1 (Foundation)
- Complete Phase 2 (Data Quality)
- Phase 3.1 only (Category Evaluation)
- Phase 4.1 only (Environment Tracking)

This gives you a production-ready experimentation platform without the complexity of LLM-as-a-Judge or major refactoring.

---

## Next Steps

1. **Review this roadmap** with team
2. **Prioritize phases** based on needs
3. **Set up feature branch** for Phase 1
4. **Begin implementation** with validation pipeline
5. **Test incrementally** after each feature
6. **Document as you go** in this docs/ folder

---

## Questions to Answer Before Starting

- [ ] Do we need LLM-as-a-Judge evaluation? (API costs)
- [ ] Should we implement pairwise comparison? (requires loading base model)
- [ ] What's the priority: reproducibility or evaluation quality?
- [ ] Do we have time for full architecture refactor (Phase 5)?
- [ ] What's the target timeline for completion?

---

## Conclusion

This roadmap transforms the pipeline from a training script into a research-grade experimentation platform. The phased approach allows incremental progress while maintaining a working system at each stage.

**Key Philosophy**: Every experiment should answer a question. This infrastructure makes it possible to ask and answer those questions reliably.
