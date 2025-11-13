# Results Directory

This directory contains evaluation results and comparison outputs from various experiments.

## 📁 Contents

### Agent Comparison Results
- `three_way_comparison_results.csv` - Vanilla vs DSPy Baseline vs DSPy Optimized
- `four_way_comparison_results.csv` - Four-way agent comparison (if exists)

### MLflow Evaluation Results
- `mlflow_mcq_detailed_results.csv` - Detailed question-by-question results
- `mlflow_mcq_model_summary.csv` - Model performance summary

### Other Evaluations
- `evaluation_results.json` - General evaluation metrics
- `mcq_eval_results.csv` - MCQ evaluation detailed results
- `mcq_eval_model_summary.csv` - MCQ evaluation summary

## 🔄 Regenerating Results

### Three-Way Comparison
```bash
python notebooks/scripts/compare_three_agents.py --test-samples 10 --optimized-model-path models/dspy_bootstrap_optimized.json
```

### MLflow Evaluation
```bash
python scripts/mlflow_mcq_solver.py
```

### Standard MCQ Evaluation
```bash
python scripts/mcq_eval.py
```

## 📊 Viewing Results

Results are also logged to MLflow. View them with:

```bash
mlflow ui --port 8080
```

Then open: http://localhost:8080

## 📝 Notes

- Results are regenerable and excluded from git (see `.gitignore`)
- All evaluation scripts save to this directory by default
- CSV files can be opened in Excel or analyzed with pandas
- MLflow artifacts include copies of these results
