# Split Dataset Update - Documentation

**Date:** November 12, 2025  
**Change Type:** Dataset Loading Enhancement

---

## 🎯 Summary

Updated all training scripts to support the new **split train/validation dataset** approach instead of random splitting from a combined dataset.

### New Dataset Structure
- **Training Set:** `dspy_dataset_train.jsonl` - Used for model training
- **Validation Set:** `dspy_dataset_val.jsonl` - Used for model evaluation
- **Default Configuration:** 200 train samples + 40 val samples = 240 total questions

---

## 📝 Files Modified

### 1. `train_mcq_solver.py` (Core Training Script)

**New Function Added:**
```python
def load_split_datasets(
    train_samples: int = 200,
    val_samples: int = 40,
    seed: int = 42
) -> tuple[List[Dict], List[Dict]]
```

**New Arguments:**
- `--use-split-datasets` - Flag to enable split dataset loading
- `--train-samples` - Number of samples from train.jsonl (default: 200)
- `--val-samples` - Number of samples from val.jsonl (default: 40)

**Behavior:**
- When `--use-split-datasets` is used:
  - Loads from `aneeta_v2/Processed Data/dspy_dataset_train.jsonl`
  - Loads from `aneeta_v2/Processed Data/dspy_dataset_val.jsonl`
  - Uses train data for training, val data for testing (no random split)
  - Different random seeds for train/val sampling to avoid overlap

- When NOT using split datasets (backward compatible):
  - Uses old behavior with `--use-combined` or Gemini data
  - Performs random train/test split

### 2. `train_controller.py` (Training Controller)

**Configuration Changes:**
```python
TRAINING_CONFIG = {
    # NEW: Option A - Split datasets (RECOMMENDED)
    'use_split_datasets': True,
    'train_samples': 200,
    'val_samples': 40,
    
    # OLD: Option B - Combined dataset (still supported)
    # 'use_combined': True,
    # 'questions': 100,
    # 'test_split': 0.2,
    ...
}
```

**Updated Functions:**
- `build_command()` - Now detects and handles split dataset configuration
- `print_config()` - Shows different output based on dataset mode

### 3. `train_all_models.py` (Multi-Model Training)

**Configuration Changes:**
```python
TRAINING_CONFIG = {
    'use_split_datasets': True,
    'train_samples': 200,
    'val_samples': 40,
    ...
}
```

**Updated Functions:**
- `build_train_command()` - Passes split dataset arguments to training script
- Output display updated to show train/val split information

### 4. `run_full_3x3_evaluation.py` (Master Evaluation Script)

**Updated:**
- Documentation reflects new dataset sizes (200 train + 40 val)
- Comparison test samples updated to 40 (matching val_samples)
- Time estimates updated based on larger dataset

### 5. `compare_three_agents.py` (Three-Way Comparison) ✨ NEW

**Updated Function:**
```python
def load_test_questions(
    max_questions: int = 10,
    seed: int = 42,
    filter_visual: bool = True,
    use_validation_set: bool = True  # NEW parameter
) -> List[Dict]
```

**New Arguments:**
- `--use-validation-set` - Use validation dataset (default: True)
- `--use-gemini-data` - Override to use Gemini data instead

**Behavior:**
- **Default:** Loads from `dspy_dataset_val.jsonl` (same data used in training evaluation)
- **With `--use-gemini-data`:** Falls back to original Gemini data loading
- Ensures comparison uses the **same validation set** as training

### 6. `compare_controller.py` (Comparison Controller) ✨ NEW

**Configuration Changes:**
```python
COMPARISON_CONFIG = {
    'test_samples': 40,              # Match val_samples from training
    'use_validation_set': True,      # NEW - use val.jsonl
    ...
}
```

**Updated Functions:**
- `build_command()` - Passes validation set flag to comparison script
- `print_config()` - Shows which dataset is being used

### 7. `compare_3x3_simple.py` (DSPy-Only Comparison) ✨ NEW

**Updated Function:**
```python
def load_test_questions(
    max_questions: int,
    seed: int,
    use_validation_set: bool = True  # NEW parameter
) -> List[Dict]
```

**New Arguments:**
- `--use-validation-set` - Use validation dataset (default: True)
- `--use-combined` - Override to use combined dataset

**Behavior:**
- **Default:** Uses validation dataset for consistency with training
- Ensures reproducible comparisons on the same held-out data

---

## 🚀 Usage Examples

### Quick Training (Single Model)
```bash
# Using split datasets (NEW)
python notebooks/scripts/train_controller.py

# Using combined dataset (OLD - still works)
# Edit TRAINING_CONFIG to use 'use_combined': True instead
```

### Train All 3 Models
```bash
python notebooks/scripts/train_all_models.py
```

### Full Evaluation Pipeline
```bash
python notebooks/scripts/run_full_3x3_evaluation.py
```

### Manual Training (Advanced)
```bash
# New split dataset method
python notebooks/scripts/train_mcq_solver.py \
  --use-split-datasets \
  --train-samples 200 \
  --val-samples 40 \
  --provider ollama \
  --model gemma2:9b \
  --method bootstrap \
  --save-path models/dspy_bootstrap_gemma2_9b.json

# Old combined dataset method (still supported)
python notebooks/scripts/train_mcq_solver.py \
  --use-combined \
  --questions 100 \
  --test-split 0.2 \
  --provider ollama \
  --model llama3.1:8b
```

---

## 🔄 Migration Guide

### For Existing Scripts

**Before (using combined dataset):**
```python
TRAINING_CONFIG = {
    'use_combined': True,
    'questions': 100,
    'test_split': 0.2,
    'seed': 42,
    'provider': 'ollama',
    'model': 'gemma2:9b',
    'method': 'bootstrap',
    'max_demos': 4,
    'save_path': 'model.json',
}
```

**After (using split datasets):**
```python
TRAINING_CONFIG = {
    'use_split_datasets': True,
    'train_samples': 200,      # From train.jsonl
    'val_samples': 40,         # From val.jsonl
    'seed': 42,
    'provider': 'ollama',
    'model': 'gemma2:9b',
    'method': 'bootstrap',
    'max_demos': 4,
    'save_path': 'model.json',
}
```

### For Custom Scripts

If you have custom scripts calling `train_mcq_solver.py`, add these flags:

```bash
# Add these flags
--use-split-datasets \
--train-samples 200 \
--val-samples 40

# Remove these flags (no longer needed)
# --use-combined
# --questions 100
# --test-split 0.2
```

---

## ✅ Benefits of Split Datasets

### 1. **No Data Leakage**
- Training and validation data are completely separate
- No risk of same questions appearing in both sets

### 2. **Reproducible Splits**
- Same train/val split every time
- No need to rely on random seed for reproducibility

### 3. **Better Evaluation**
- Validation set is truly unseen data
- More realistic performance estimation

### 4. **Cleaner Code**
- No need for train_test_split
- Explicit separation of concerns

### 5. **Flexible Sampling**
- Control exactly how many questions from each set
- Different sampling strategies for train vs val

---

## 📊 Default Configuration

### Recommended Settings
- **Training:** 200 questions from train.jsonl
- **Validation:** 40 questions from val.jsonl
- **Ratio:** 83% train / 17% val
- **Seed:** 42 (train), 43 (val) - different seeds prevent overlap

### Dataset Statistics
- **train.jsonl:** ~7,000+ questions available
- **val.jsonl:** ~800+ questions available
- **Sampling:** Random sampling with seed for reproducibility

---

## 🔧 Backward Compatibility

All old scripts still work! The changes are **additive**:

✅ Old method (`--use-combined`) still supported  
✅ Old method (Gemini data loading) still supported  
✅ Existing saved models still compatible  
✅ No breaking changes to MLflow tracking  

---

## 📈 Performance Impact

### Training Time
- **Before:** 100 questions (80 train, 20 test) ~ 7-10 min per model
- **After:** 240 questions (200 train, 40 test) ~ 10-15 min per model
- **Increase:** ~40-50% longer training time

### Model Quality
- **More training data:** Better few-shot example selection
- **Larger validation set:** More reliable accuracy estimates
- **Expected improvement:** +5-10% accuracy from more training data

### Cost
- **Ollama:** Still $0 (no change)
- **OpenAI:** Proportional to question count (~2.4x cost if using OpenAI)

---

## 🐛 Troubleshooting

### Error: "Training dataset not found"
**Solution:** Ensure files exist:
- `d:\Git Projects\SMU\ANEETAA\aneeta_v2\Processed Data\dspy_dataset_train.jsonl`
- `d:\Git Projects\SMU\ANEETAA\aneeta_v2\Processed Data\dspy_dataset_val.jsonl`

### Error: "Not enough questions to sample"
**Solution:** Reduce `--train-samples` or `--val-samples` to match available data

### Mixed dataset usage
**Solution:** Choose ONE approach:
- Either use `use_split_datasets: True`
- OR use `use_combined: True`
- Don't use both at the same time!

---

## 📚 Related Documentation

- **Training Guide:** `notebooks/docs/3x3_EVALUATION_GUIDE.md`
- **Results:** `notebooks/docs/3X2_MATRIX_EVALUATION_RESULTS.md`
- **Dataset Info:** `docs/COMBINED_DATASET_USAGE.md`

---

## 🎓 Example Workflow

```bash
# 1. Train all 3 models with split datasets (200 train + 40 val)
python notebooks/scripts/train_all_models.py

# 2. Run comparison on validation set
python notebooks/scripts/compare_3x3_simple.py --test-samples 40

# 3. View results
mlflow ui --port 8080
```

**Expected Output:**
- 3 trained models in `models/` directory
- Results in `results/` directory
- MLflow tracking in `mlruns/` directory

---

**Status:** ✅ All controller files updated  
**Tested:** ✅ Backward compatible  
**Recommended:** Use split datasets for all new training runs
