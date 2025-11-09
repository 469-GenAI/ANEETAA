# Notebooks Directory

This directory contains DSPy optimization experiments, training scripts, and comparison tools for the ANEETAA project.

## 📁 Directory Structure

```
notebooks/
├── scripts/                       # Main production scripts
│   ├── compare_three_agents.py   # Compare Vanilla vs DSPy Baseline vs DSPy Optimized
│   └── train_mcq_solver.py       # Train DSPy models with Bootstrap/MIPRO optimization
│
├── tests/                         # Test scripts
│   ├── test_cache_busting.py     # Test DSPy cache busting mechanisms
│   ├── test_local_mlflow.py      # Verify local MLflow setup
│   ├── test_mlflow_connection.py # Test Databricks MLflow connection
│   └── test_vanilla_agent.py     # Test vanilla ANEETAA agent
│
├── experiments/                   # Experimental/deprecated scripts
│   ├── analyze_bootstrap.py      # Analyze bootstrap optimization results
│   ├── check_optimizers.py       # Check available DSPy optimizers
│   ├── compare_agents.py         # Old version of agent comparison
│   ├── create_larger_training_set.py
│   ├── create_training_data.py
│   ├── dspy_optimization.py      # Old optimization script
│   ├── optimize_both_methods.py  # Compare Bootstrap vs MIPRO
│   └── simple_evaluation.py      # Simple evaluation script
│
├── docs/                          # Documentation
│   ├── AGENT_COMPARISON_GUIDE.md
│   ├── CACHE_BUSTING_FIX.md
│   ├── FIXES_SUMMARY.md
│   ├── FOUR_WAY_COMPARISON_GUIDE.md
│   ├── MLFLOW_FREE_OPTIONS.md
│   ├── MLFLOW_GUIDE.md
│   ├── SETUP_THREE_WAY_COMPARISON.md
│   └── TESTING_SUMMARY.md
│
├── mlruns/                        # MLflow experiment tracking data
├── dspy_optimization.ipynb        # Interactive optimization notebook
├── optimization_log.txt           # Training logs
└── requirements_dspy.txt          # DSPy-specific dependencies
```

## 🚀 Quick Start

### Train a DSPy Model

```bash
python notebooks/scripts/train_mcq_solver.py --questions 100 --method bootstrap --provider openai --model gpt-4o-mini
```

### Compare Agents

```bash
python notebooks/scripts/compare_three_agents.py --test-samples 10 --dspy-provider openai --dspy-model gpt-4o-mini --optimized-model-path models/dspy_bootstrap_optimized.json
```

### Run Tests

```bash
# Test MLflow setup
python notebooks/tests/test_local_mlflow.py

# Test vanilla ANEETAA agent
python notebooks/tests/test_vanilla_agent.py
```

## 📊 MLflow Tracking

All experiments are logged to MLflow. View results:

```bash
mlflow ui --port 8080
```

Then open: http://localhost:8080

## 📚 Documentation

See `docs/` folder for detailed guides:
- **AGENT_COMPARISON_GUIDE.md** - How to compare different agent implementations
- **MLFLOW_GUIDE.md** - MLflow setup and usage
- **CACHE_BUSTING_FIX.md** - Solutions for DSPy caching issues
- **SETUP_THREE_WAY_COMPARISON.md** - Three-way comparison setup

## 🔧 Key Files

- **scripts/train_mcq_solver.py** - Main training script supporting Bootstrap and MIPRO optimization
- **scripts/compare_three_agents.py** - Three-way agent comparison with MLflow logging
- **dspy_optimization.ipynb** - Interactive notebook for DSPy experiments
- **requirements_dspy.txt** - Python dependencies for DSPy work

## 💡 Notes

- The `experiments/` folder contains older scripts kept for reference
- Main production scripts are in `scripts/`
- All test scripts use local MLflow tracking
- Training data is loaded from `aneeta_v2/Processed Data/Gemini 2.5 Pro Data/`
