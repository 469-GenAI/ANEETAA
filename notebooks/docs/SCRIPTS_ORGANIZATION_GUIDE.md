# Scripts Organization

This directory contains all training and evaluation scripts for ANEETAA, organized into three main folders for clarity and maintainability.

## 📁 Directory Structure

```
notebooks/scripts/
├── controllers/          # User-facing control scripts (START HERE!)
│   ├── train_controller.py
│   ├── compare_controller.py
│   └── compare_3x4_controller.py
│
├── runners/             # Actual execution scripts (called by controllers)
│   ├── train_mcq_solver.py
│   ├── train_all_models.py
│   ├── train_all_models_miprov2.py
│   ├── compare_three_agents.py
│   ├── compare_3x3_matrix.py
│   └── compare_3x4_matrix.py
│
├── config/              # Configuration and utilities
│   ├── llm_judge_config.py
│   └── judge_config_controller.py
│
└── [other utility scripts]
```

---

## 🎮 Controllers (User Interface)

**Location:** `notebooks/scripts/controllers/`

These are the **scripts you should run**. They provide easy-to-use interfaces with preset configurations.

### Training Controllers

#### `train_controller.py`
**Purpose:** Train a single DSPy MCQ solver model  
**Usage:**
```bash
python notebooks/scripts/controllers/train_controller.py
```

**What it does:**
- Trains DSPy models with Bootstrap or MIPRO optimizer
- Uses preset configuration (edit `TRAINING_CONFIG` in file)
- Saves trained model to `models/` directory

**Configuration:** Edit `TRAINING_CONFIG` dictionary at top of file

---

### Comparison Controllers

#### `compare_controller.py`
**Purpose:** Compare 3 agent types (Vanilla, Baseline, Optimized)  
**Usage:**
```bash
python notebooks/scripts/controllers/compare_controller.py
```

**What it does:**
- Tests Vanilla ANEETAA vs DSPy Baseline vs DSPy Optimized
- Currently set to 3 questions for quick testing
- Saves results to `results/three_way_comparison_results.csv`

**Configuration:** Edit `COMPARISON_CONFIG` dictionary

#### `compare_3x4_controller.py`
**Purpose:** Compare 3 models × 4 agent types (12 total configurations)  
**Usage:**
```bash
python notebooks/scripts/controllers/compare_3x4_controller.py
```

**What it does:**
- Compares 3 models (llama, gemma, mistral)
- Across 4 agent types (Vanilla, Baseline, Bootstrap, MIPROv2)
- Checks prerequisites before running
- Saves results to `results/3x4_matrix_*.csv`

**Configuration:** Edit `COMPARISON_CONFIG` dictionary

---

## ⚙️ Runners (Execution Logic)

**Location:** `notebooks/scripts/runners/`

These scripts contain the actual implementation logic. **You typically won't run these directly** - use controllers instead.

### Training Runners

#### `train_mcq_solver.py`
- Core training script
- Called by `train_controller.py`
- Supports Bootstrap and MIPRO optimizers
- Handles data loading, training, evaluation

#### `train_all_models.py`
- Trains Bootstrap models for all 3 Ollama models
- Can be run directly if needed
- Takes ~90-135 minutes

#### `train_all_models_miprov2.py`
- Trains MIPROv2 models for all 3 Ollama models
- Takes ~4.5-7 hours
- Higher quality than Bootstrap

### Comparison Runners

#### `compare_three_agents.py`
- Compares 3 agent types on test questions
- Called by `compare_controller.py`
- Logs to MLflow

#### `compare_3x3_matrix.py`
- 3 models × 3 agent types = 9 configurations
- Called by 3x3 controller (if created)

#### `compare_3x4_matrix.py`
- 3 models × 4 agent types = 12 configurations
- Called by `compare_3x4_controller.py`
- Includes MIPROv2 optimizer

---

## 🔧 Config (Configuration)

**Location:** `notebooks/scripts/config/`

Configuration utilities used by all scripts.

#### `llm_judge_config.py`
**Purpose:** Centralized LLM judge configuration

**Features:**
- Supports 4 providers: OpenAI, Groq, Anthropic, Ollama
- Cost estimation
- Easy provider switching
- Currently configured to Groq (FREE)

**Import in code:**
```python
from llm_judge_config import get_judge_llm, estimate_judge_cost
```

#### `judge_config_controller.py`
**Purpose:** CLI tool to change judge settings

**Usage:**
```bash
# Show current config
python notebooks/scripts/config/judge_config_controller.py --show

# Switch to Groq (FREE)
python notebooks/scripts/config/judge_config_controller.py --preset groq-default

# Switch to OpenAI
python notebooks/scripts/config/judge_config_controller.py --preset openai-mini

# Test current judge
python notebooks/scripts/config/judge_config_controller.py --test
```

---

## 🚀 Quick Start Guide

### 1. Configure LLM Judge (Optional)
```bash
# Check current judge (default: Groq - FREE)
python notebooks/scripts/config/judge_config_controller.py --show

# Change if needed
python notebooks/scripts/config/judge_config_controller.py --preset openai-mini
```

### 2. Train Models (First Time Only)

**Option A: Quick test (Bootstrap only)**
```bash
# Edit train_controller.py to set test_samples: 50
python notebooks/scripts/controllers/train_controller.py
```

**Option B: Full training (Bootstrap + MIPROv2)**
```bash
# Bootstrap (~2 hours for all 3 models)
python notebooks/scripts/runners/train_all_models.py

# MIPROv2 (~7 hours for all 3 models)
python notebooks/scripts/runners/train_all_models_miprov2.py
```

### 3. Run Comparisons

**Quick 3-way test:**
```bash
python notebooks/scripts/controllers/compare_controller.py
# Tests: Vanilla vs Baseline vs Optimized
# Default: 3 questions
```

**Full 3×4 matrix:**
```bash
python notebooks/scripts/controllers/compare_3x4_controller.py
# Tests: 3 models × 4 agent types
# Default: 10 questions
```

### 4. View Results

**CSV Files:**
```
results/
├── three_way_comparison_results.csv
├── 3x4_matrix_detailed_results.csv
└── 3x4_matrix_summary.csv
```

**MLflow UI:**
```bash
mlflow ui --port 8080
# Open: http://localhost:8080
```

---

## 📊 Typical Workflow

```
1. Configure judge (optional)
   ↓
2. Train models (one-time)
   ├─ Bootstrap: train_all_models.py
   └─ MIPROv2: train_all_models_miprov2.py
   ↓
3. Run comparisons
   ├─ Quick test: compare_controller.py
   └─ Full matrix: compare_3x4_controller.py
   ↓
4. Analyze results
   ├─ CSV files in results/
   └─ MLflow UI
```

---

## 🎯 When to Use What

### Use Controllers When:
- ✅ You want to run training or comparison
- ✅ You want preset configurations
- ✅ You want validation and error checking
- ✅ You're a user (not developing)

### Use Runners Directly When:
- ⚙️ You need custom command-line arguments
- ⚙️ You're debugging or developing
- ⚙️ Controllers don't provide needed flexibility

### Use Config When:
- 🔧 Switching LLM judge provider
- 🔧 Estimating costs
- 🔧 Testing judge functionality

---

## 📝 File Relationships

```
Controllers           →    Runners                  →    Config
─────────────────────────────────────────────────────────────────
train_controller.py   →    train_mcq_solver.py     →    [imports config]
compare_controller.py →    compare_three_agents.py →    llm_judge_config.py
compare_3x4_controller.py → compare_3x4_matrix.py  →    [imports config]
                                                   
                                                    judge_config_controller.py
                                                    (modifies llm_judge_config.py)
```

---

## 🔍 Finding the Right Script

**Want to:**
- **Train a model?** → `controllers/train_controller.py`
- **Compare 3 agents?** → `controllers/compare_controller.py`
- **Compare all optimizers?** → `controllers/compare_3x4_controller.py`
- **Change LLM judge?** → `config/judge_config_controller.py`
- **Understand training?** → Read `runners/train_mcq_solver.py`
- **Understand comparison?** → Read `runners/compare_three_agents.py`

---

## 🆘 Troubleshooting

### Import Errors
**Problem:** `Import "llm_judge_config" could not be resolved`  
**Solution:** This is a linting cosmetic issue. Code works at runtime because paths are added dynamically.

### File Not Found
**Problem:** Controller can't find runner script  
**Solution:** Make sure you're running from project root or using correct path

### Model Not Found
**Problem:** Comparison fails with "model not found"  
**Solution:** Train models first using `train_all_models.py` or `train_all_models_miprov2.py`

---

## 📚 Related Documentation

- **MIPROv2 Integration:** `notebooks/docs/MIPROV2_INTEGRATION.md`
- **LLM Judge Guide:** `notebooks/docs/LLM_JUDGE_GUIDE.md`
- **Checkpoint README:** `CHECKPOINT_README.md` (project root)

---

**Last Updated:** November 12, 2025  
**Organization Version:** 2.0 (Reorganized into controllers/runners/config)
