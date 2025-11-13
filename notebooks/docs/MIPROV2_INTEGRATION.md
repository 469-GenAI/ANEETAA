# MIPROv2 Integration Guide

## Overview

This guide explains the MIPROv2 (Multi-prompt Instruction Proposal and Revision Optimizer v2) integration into ANEETAA's evaluation framework.

## What is MIPROv2?

**MIPROv2** is an advanced DSPy optimizer that:

- ✅ **Optimizes both instructions AND demonstrations** (vs Bootstrap which only optimizes demonstrations)
- ✅ **Uses iterative refinement** to find the best prompt combinations
- ✅ **Generally produces 5-10% better accuracy** than Bootstrap
- ⚠️ **Takes 2-3x longer to train** (~90-135 min per model vs 30-45 min for Bootstrap)
- ⚠️ **Requires more training data** (recommended 300+ samples vs 200 for Bootstrap)

## Architecture: 3×4 Matrix

### Before (3×3 Matrix):
- **3 Models**: llama3.1:8b, gemma2:9b, mistral-nemo:12b
- **3 Agent Types**: Vanilla ANEETAA, DSPy Baseline, DSPy Bootstrap
- **Total**: 9 configurations

### After (3×4 Matrix):
- **3 Models**: llama3.1:8b, gemma2:9b, mistral-nemo:12b
- **4 Agent Types**: Vanilla ANEETAA, DSPy Baseline, DSPy Bootstrap, **DSPy MIPROv2** ✨
- **Total**: 12 configurations

## Files Created

### 1. Training Scripts

#### `train_all_models_miprov2.py`
Trains MIPROv2-optimized models for all 3 Ollama models.

**Configuration:**
```python
TRAINING_CONFIG = {
    'use_split_datasets': True,
    'train_samples': 300,      # More data for MIPRO
    'val_samples': 40,
    'seed': 42,
    'provider': 'ollama',      # FREE training with Ollama
    'method': 'mipro',
    'candidates': 7,           # Instruction candidates to try
}
```

**Output Models:**
- `models/dspy_mipro_llama3.1_8b.json`
- `models/dspy_mipro_gemma2_9b.json`
- `models/dspy_mipro_mistral_nemo_12b.json`

**Usage:**
```bash
python notebooks/scripts/train_all_models_miprov2.py
```

**Expected Time:** ~4.5-7 hours total (90-135 min per model)  
**Expected Cost:** $0 (using Ollama)

### 2. Comparison Scripts

#### `compare_3x4_matrix.py`
Evaluates all 12 configurations (3 models × 4 agent types).

**Features:**
- Compares Vanilla vs Baseline vs Bootstrap vs MIPROv2
- Uses centralized LLM judge for quality scoring
- Generates detailed and summary CSVs
- Logs to MLflow for visualization
- Direct Bootstrap vs MIPROv2 comparison output

**Usage:**
```bash
python notebooks/scripts/compare_3x4_matrix.py --test-samples 100
```

#### `compare_3x4_controller.py`
Easy-to-use controller for running 3×4 comparisons.

**Features:**
- Preset configurations (quick/medium/full)
- Automatic prerequisite checking
- Model availability validation
- Time estimation

**Usage:**
```bash
python notebooks/scripts/compare_3x4_controller.py
```

## Workflow

### Step 1: Train Bootstrap Models (if not done)
```bash
# Takes ~90-135 minutes total
python notebooks/scripts/train_all_models.py
```

**Outputs:**
- `models/dspy_bootstrap_llama3.1_8b.json`
- `models/dspy_bootstrap_gemma2_9b.json`
- `models/dspy_bootstrap_mistral_nemo_12b.json`

### Step 2: Train MIPROv2 Models
```bash
# Takes ~4.5-7 hours total
python notebooks/scripts/train_all_models_miprov2.py
```

**Outputs:**
- `models/dspy_mipro_llama3.1_8b.json`
- `models/dspy_mipro_gemma2_9b.json`
- `models/dspy_mipro_mistral_nemo_12b.json`

### Step 3: Run 3×4 Comparison
```bash
# Quick test (10 questions, ~8 minutes)
python notebooks/scripts/compare_3x4_controller.py

# Or directly with custom samples
python notebooks/scripts/compare_3x4_matrix.py --test-samples 50
```

### Step 4: View Results

**CSV Files:**
- `results/3x4_matrix_detailed_results.csv` - Per-question results
- `results/3x4_matrix_summary.csv` - Aggregated metrics

**MLflow UI:**
```bash
mlflow ui --port 8080
# Open: http://localhost:8080
```

## Expected Results

Based on DSPy research, typical improvements:

| Agent Type | Expected Accuracy | Notes |
|------------|------------------|-------|
| Vanilla ANEETAA | 65-75% | Baseline |
| DSPy Baseline | 70-75% | Unoptimized DSPy |
| DSPy Bootstrap | 80-87% | +10-15% over baseline |
| DSPy MIPROv2 | 85-92% | +5-10% over Bootstrap |

## Configuration Options

### Training Configuration

**For faster testing (lower quality):**
```python
TRAINING_CONFIG = {
    'train_samples': 200,      # Less data
    'candidates': 5,           # Fewer candidates
}
```

**For better quality (slower):**
```python
TRAINING_CONFIG = {
    'train_samples': 500,      # More data
    'candidates': 10,          # More candidates
}
```

### Comparison Configuration

**Quick test:**
```python
COMPARISON_CONFIG = {
    'test_samples': 10,
}
```

**Full evaluation:**
```python
COMPARISON_CONFIG = {
    'test_samples': 100,
}
```

## Model Files Structure

```
models/
├── dspy_bootstrap_llama3.1_8b.json      # Bootstrap optimized
├── dspy_bootstrap_gemma2_9b.json
├── dspy_bootstrap_mistral_nemo_12b.json
├── dspy_mipro_llama3.1_8b.json          # MIPROv2 optimized (NEW!)
├── dspy_mipro_gemma2_9b.json
└── dspy_mipro_mistral_nemo_12b.json
```

## Troubleshooting

### Issue: MIPROv2 training takes too long

**Solution:** Reduce training data or candidates:
```python
'train_samples': 200,  # Instead of 300
'candidates': 5,       # Instead of 7
```

### Issue: Out of memory during training

**Solution:** Train models one at a time:
```bash
# Edit train_all_models_miprov2.py to comment out 2 models
# Train one at a time
```

### Issue: MIPROv2 models not found

**Solution:** Check if training completed:
```bash
ls models/dspy_mipro*.json
```

If missing, run training script again.

### Issue: Comparison fails with "model not found"

**Solution:** The comparison will skip missing models. Train them first:
```bash
python notebooks/scripts/train_all_models.py          # Bootstrap
python notebooks/scripts/train_all_models_miprov2.py  # MIPROv2
```

## Performance Comparison

### Training Time

| Optimizer | Time per Model | Total (3 models) |
|-----------|---------------|------------------|
| Bootstrap | 30-45 min | 90-135 min |
| MIPROv2 | 90-135 min | 4.5-7 hours |

### Evaluation Time

| Matrix | Questions | Time |
|--------|-----------|------|
| 3×3 (9 configs) | 100 | ~60 min |
| 3×4 (12 configs) | 100 | ~80 min |

### Expected Accuracy Gains

- **Baseline → Bootstrap**: +10-15%
- **Bootstrap → MIPROv2**: +5-10%
- **Total (Baseline → MIPROv2)**: +15-25%

## References

- [DSPy MIPROv2 Documentation](https://dspy-docs.vercel.app/docs/building-blocks/optimizers#miprov2)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [MIPROv2 Paper](https://arxiv.org/abs/2310.03714)

## Next Steps

1. ✅ Train Bootstrap models (if not done)
2. ✅ Train MIPROv2 models
3. ✅ Run 3×4 comparison
4. 📊 Analyze results to see if MIPROv2 provides significant improvements
5. 🎯 Use best-performing configuration for production

---

**Last Updated:** November 12, 2025  
**Status:** Ready for testing  
**Author:** ANEETAA Team
