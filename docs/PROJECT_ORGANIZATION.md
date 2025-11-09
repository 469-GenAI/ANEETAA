# Project Organization Summary

This document describes the reorganized ANEETAA project structure (November 2025).

## 📂 Project Structure

```
ANEETAA/
├── .env                           # Environment variables (not in git)
├── .env.example                   # Example environment template
├── .gitignore                     # Git ignore rules
├── README.md                      # Main project README
├── LICENSE                        # Project license
├── requirements.txt               # Python dependencies
├── app.py                         # Main Streamlit application
│
├── src/                           # Source code
│   └── aneeta/                    # Main package
│       ├── core/                  # Core functionality
│       ├── nodes/                 # LangGraph nodes
│       ├── state/                 # State management
│       └── vectordb/              # Vector database
│
├── notebooks/                     # Jupyter notebooks and DSPy experiments
│   ├── scripts/                   # Production scripts
│   │   ├── compare_three_agents.py
│   │   └── train_mcq_solver.py
│   ├── tests/                     # Test scripts
│   ├── experiments/               # Experimental code
│   ├── docs/                      # Notebook documentation
│   ├── mlruns/                    # MLflow tracking (local)
│   ├── dspy_optimization.ipynb
│   ├── aneeta_quickstart.ipynb
│   ├── dspy_quickstart.ipynb
│   ├── test DSPy (1).ipynb
│   └── README.md
│
├── scripts/                       # Standalone utility scripts
│   ├── mcq_eval.py
│   ├── mlflow_mcq_solver.py
│   ├── test_mlflow.py
│   ├── verify_setup.py
│   └── README.md
│
├── models/                        # Trained models (excluded from git)
│   ├── dspy_bootstrap_optimized.json
│   ├── dspy_mipro_optimized.json
│   ├── dspy_training_data.json
│   ├── dspy_training_data_100.json
│   ├── .gitkeep
│   └── README.md
│
├── results/                       # Evaluation results (excluded from git)
│   ├── three_way_comparison_results.csv
│   ├── four_way_comparison_results.csv
│   ├── mlflow_mcq_detailed_results.csv
│   ├── mlflow_mcq_model_summary.csv
│   ├── evaluation_results.json
│   ├── .gitkeep
│   └── README.md
│
├── docs/                          # Project documentation
│   ├── DSPy_Integration_Guide.md
│   ├── RUN_GUIDE.md
│   ├── SETUP.md
│   ├── SUCCESS_SUMMARY.md
│   ├── COMPLETE_SUMMARY.md
│   ├── DEMO_GUIDE.md
│   └── ...
│
├── DataPrep-Notebooks/            # Data preparation notebooks
│   └── VectorDatabaseProcessing.ipynb
│
├── aneeta_v2/                     # ANEETAA v2 data and scripts
│   ├── Processed Data/
│   │   └── Gemini 2.5 Pro Data/  # 7,874 questions
│   └── scripts/
│       └── Data Extraction/
│
├── Raw Data/                      # Source data
│   ├── QuestionBank/              # PDF question papers
│   ├── NCRTBooks/                 # NCERT textbooks
│   └── MentorGuide/               # Mentor materials
│
├── Processed Data/                # Processed datasets
│   ├── mentor_data.json
│   ├── processed_biology_chunks.json
│   ├── processed_chemistry_chunks.json
│   ├── processed_physics_chunks.json
│   └── solved_question_papers.json
│
├── Images/                        # Project images and assets
├── mlruns/                        # MLflow experiment tracking (excluded from git)
└── .venv/                         # Python virtual environment (excluded from git)
```

## 🔄 What Changed

### Root Directory Cleanup

**Before**: Cluttered with 15+ loose files (models, results, scripts, notebooks, docs)

**After**: Clean structure with organized subdirectories

### New Directories Created

1. **`models/`** - All trained DSPy models and training data
2. **`results/`** - All evaluation results and comparison CSVs
3. **`scripts/`** - Standalone utility scripts
4. **`notebooks/scripts/`** - Production DSPy training/comparison scripts
5. **`notebooks/tests/`** - Test scripts
6. **`notebooks/experiments/`** - Experimental/deprecated code
7. **`notebooks/docs/`** - Notebook-specific documentation

### File Migrations

#### Models → `models/`
- dspy_bootstrap_optimized.json
- dspy_mipro_optimized.json
- dspy_training_data.json
- dspy_training_data_100.json

#### Results → `results/`
- three_way_comparison_results.csv
- four_way_comparison_results.csv
- evaluation_results.json
- mlflow_mcq_detailed_results.csv
- mlflow_mcq_model_summary.csv

#### Scripts → `scripts/`
- mcq_eval.py
- mlflow_mcq_solver.py
- test_mlflow.py
- verify_setup.py

#### Notebooks → `notebooks/`
- aneeta_quickstart.ipynb
- dspy_quickstart.ipynb
- test DSPy (1).ipynb

#### Documentation → `docs/`
- DSPy_Integration_Guide.md
- RUN_GUIDE.md
- SETUP.md
- SUCCESS_SUMMARY.md

### Notebooks Reorganization

**`notebooks/scripts/`** (Production)
- compare_three_agents.py
- train_mcq_solver.py

**`notebooks/tests/`** (Testing)
- test_cache_busting.py
- test_local_mlflow.py
- test_mlflow_connection.py
- test_vanilla_agent.py

**`notebooks/experiments/`** (Experimental)
- analyze_bootstrap.py
- check_optimizers.py
- compare_agents.py (old version)
- create_larger_training_set.py
- create_training_data.py
- dspy_optimization.py
- optimize_both_methods.py
- simple_evaluation.py

**`notebooks/docs/`** (Documentation)
- AGENT_COMPARISON_GUIDE.md
- CACHE_BUSTING_FIX.md
- FIXES_SUMMARY.md
- FOUR_WAY_COMPARISON_GUIDE.md
- MLFLOW_FREE_OPTIONS.md
- MLFLOW_GUIDE.md
- SETUP_THREE_WAY_COMPARISON.md
- TESTING_SUMMARY.md

## 📝 Updated Paths

### Scripts Updated

All scripts now use updated paths:

- `notebooks/scripts/train_mcq_solver.py` → saves to `models/`
- `notebooks/scripts/compare_three_agents.py` → saves to `results/`
- `scripts/mlflow_mcq_solver.py` → saves to `results/`
- `scripts/mcq_eval.py` → saves to `results/`

### Documentation Updated

- `notebooks/README.md` → Updated command examples
- `notebooks/docs/SETUP_THREE_WAY_COMPARISON.md` → Updated script paths
- `notebooks/docs/FOUR_WAY_COMPARISON_GUIDE.md` → Updated model paths

### Example Command Updates

**Old**:
```bash
python notebooks/compare_three_agents.py --optimized-model-path dspy_bootstrap_optimized.json
```

**New**:
```bash
python notebooks/scripts/compare_three_agents.py --optimized-model-path models/dspy_bootstrap_optimized.json
```

## 🔧 .gitignore Updates

Added explicit exclusions for:
- `models/` directory (with .gitkeep)
- `results/` directory (with .gitkeep)
- Cleaner JSON/CSV handling

## ✅ Benefits

1. **Cleaner Root** - Easy to navigate, find main files
2. **Logical Organization** - Related files grouped together
3. **Better Git Management** - Clear separation of tracked vs generated files
4. **Easier Onboarding** - New contributors can understand structure quickly
5. **Improved Documentation** - README files in each major directory
6. **Reproducibility** - Clear separation of source vs output

## 🚀 Quick Start After Reorganization

### Train Models
```bash
python notebooks/scripts/train_mcq_solver.py --questions 100 --method bootstrap
```

### Compare Agents
```bash
python notebooks/scripts/compare_three_agents.py --test-samples 10 --optimized-model-path models/dspy_bootstrap_optimized.json
```

### Verify Setup
```bash
python scripts/verify_setup.py
```

### View Results
```bash
mlflow ui --port 8080
```

## 📅 Migration Date

November 9, 2025

## 🔄 Backwards Compatibility

**Breaking Changes**:
- Model file paths changed (add `models/` prefix)
- Script paths changed (add `scripts/` or `notebooks/scripts/` prefix)
- Result file paths changed (add `results/` prefix)

**Migration Path**:
- Update any custom scripts that reference old paths
- Update CI/CD pipelines if applicable
- Clear any cached paths in local environments
