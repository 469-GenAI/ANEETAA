# Scripts Directory

This directory contains standalone utility and evaluation scripts for the ANEETAA project.

## 📁 Contents

### Evaluation Scripts
- `mcq_eval.py` - Standard MCQ evaluation script
- `mlflow_mcq_solver.py` - MLflow-integrated MCQ solver evaluation
- `test_mlflow.py` - Test MLflow connection and setup

### Setup Scripts
- `verify_setup.py` - Verify ANEETAA installation and dependencies

## 🚀 Usage

### Run MCQ Evaluation
```bash
python scripts/mcq_eval.py
```

### Run MLflow Evaluation
```bash
python scripts/mlflow_mcq_solver.py
```

### Test MLflow Connection
```bash
python scripts/test_mlflow.py
```

### Verify Setup
```bash
python scripts/verify_setup.py
```

## 📝 Notes

- These are standalone scripts (can be run directly)
- For DSPy training/comparison, use `notebooks/scripts/` instead
- Results are saved to `results/` directory
- All scripts use paths relative to project root
