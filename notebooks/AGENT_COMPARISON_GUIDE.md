# Agent Comparison Guide

This guide shows how to compare Vanilla ANEETAA, DSPy Baseline, and Optimized DSPy agents.

## Quick Start

### 1. Basic Comparison (All Three Agents)
```cmd
python notebooks\compare_agents.py --provider openai --model gpt-4o-mini --test-samples 10
```

This will evaluate:
- ✅ Vanilla ANEETAA MCQ Solver
- ✅ DSPy MCQ Solver Baseline (unoptimized)
- ✅ DSPy MCQ Solver Optimized (if available)

### 2. Compare with Your Optimized Model

If you have an optimized MCQ solver from a previous run:
```cmd
python notebooks\compare_agents.py --provider openai --model gpt-4o-mini --test-samples 10 --optimized-model-uri "runs:/YOUR_RUN_ID/mcq_agent"
```

### 3. Skip Certain Evaluations

Skip vanilla ANEETAA (faster):
```cmd
python notebooks\compare_agents.py --skip-vanilla --test-samples 10
```

Only evaluate vanilla vs optimized:
```cmd
python notebooks\compare_agents.py --skip-baseline --test-samples 10
```

## How to Find Your Optimized Model URI

1. Go to MLflow UI: http://localhost:8080
2. Click on your experiment
3. Click on the run with your optimized model
4. Look for "Logged models" in the right panel
5. Copy the model URI (looks like `runs:/abc123def456/teacher_agent`)

## What Gets Logged to MLflow

For each comparison run, MLflow tracks:

### Metrics:
- `vanilla_mcq_score` - Vanilla ANEETAA accuracy
- `dspy_baseline_score` - Unoptimized DSPy accuracy
- `dspy_optimized_score` - Optimized DSPy accuracy
- `improvement_vs_vanilla` - Percentage improvement over vanilla
- `improvement_vs_baseline` - Percentage improvement over baseline

### Parameters:
- `test_samples` - Number of questions tested
- `provider` - LLM provider (openai/ollama)
- `model` - Model name used

## Viewing Results in MLflow

1. Open MLflow UI: http://localhost:8080
2. Find experiment: **aneeta-agent-comparison**
3. Click on the latest run
4. Compare metrics side-by-side

### Compare Multiple Runs:
1. Select multiple runs (checkboxes)
2. Click "Compare" button
3. View metrics chart showing all three agents

## Advanced Options

### Use More Test Samples (More Accurate)
```cmd
python notebooks\compare_agents.py --test-samples 30
```

### Use Ollama (Free, Local)
```cmd
python notebooks\compare_agents.py --provider ollama --model llama3.1:8b --test-samples 10
```

## Expected Output

```
==============================================================
ANEETAA Agent Comparison
==============================================================

✓ Added to path: ...
✓ Using OpenAI: openai/gpt-4o-mini
✓ Using LOCAL MLflow
✓ Loaded 10 MCQ test questions

==============================================================
Evaluating Vanilla ANEETAA MCQ Solver
==============================================================
  [1/10] Score: 1.00
  [2/10] Score: 1.00
  ...
✓ Vanilla Agent Average Score: 85.00%

==============================================================
Evaluating DSPy MCQ Solver Baseline (Unoptimized)
==============================================================
  [1/10] Score: 1.00
  ...
✓ DSPy Baseline Average Score: 80.00%

==============================================================
Evaluating DSPy MCQ Solver Optimized
==============================================================
  [1/10] Score: 1.00
  ...
✓ DSPy Optimized Average Score: 92.00%

🎯 DSPy Optimized vs Vanilla: +8.2%
🎯 DSPy Optimized vs Baseline: +15.0%

==============================================================
📊 COMPARISON RESULTS
==============================================================
Vanilla Score                 : 85.00%
Dspy Baseline Score          : 80.00%
Dspy Optimized Score         : 92.00%

✓ Results logged to MLflow!
  View at: http://localhost:8080
  Experiment: aneeta-agent-comparison
```

## Troubleshooting

### "No test data loaded"
- Check that `Processed Data/solved_question_papers.json` exists
- Try with a smaller `--test-samples` number

### "Agent error"
- Make sure all dependencies are installed
- Check that vanilla agents are working: `python app.py`

### "Model not found"
- Verify the model URI is correct
- Try without `--optimized-model-uri` to use baseline

## Cost Estimation

With OpenAI GPT-4o-mini:
- 10 samples: ~$0.01-0.02
- 30 samples: ~$0.03-0.06

(Each question requires ~2-3 LLM calls for evaluation + agent calls)
