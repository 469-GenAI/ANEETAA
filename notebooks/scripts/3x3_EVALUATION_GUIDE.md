# 3×3 Matrix Evaluation Guide

## Overview

This evaluation compares **9 different configurations**:

### Models (3):
1. **llama3.1:8b** - Meta's Llama 3.1 (8 billion parameters)
2. **gemma2:9b** - Google's Gemma 2 (9 billion parameters)
3. **mistral-nemo:12b** - Mistral AI's Nemo (12 billion parameters)

### Agent Types (3):
1. **Vanilla ANEETAA** - Original LangChain-based implementation
2. **DSPy Baseline** - Unoptimized DSPy with Chain-of-Thought
3. **DSPy Optimized** - Bootstrap-optimized DSPy

### Matrix (3×3 = 9 configurations):
```
                    llama3.1:8b    gemma2:9b    mistral-nemo:12b
Vanilla ANEETAA         (1)           (2)             (3)
DSPy Baseline           (4)           (5)             (6)
DSPy Optimized          (7)           (8)             (9)
```

## Configuration

- **Questions:** 100 per evaluation
- **Training:** 80 train, 20 test per model
- **Optimizer:** Bootstrap FewShot
- **LLM Judge:** OpenAI GPT-4o-mini (for quality scoring)
- **Cost:** ~$0.18 (only for LLM judge)
- **Time:** ~2-3 hours total

## Quick Start

### Option 1: Run Everything (Recommended)
```bash
python notebooks/scripts/run_full_3x3_evaluation.py
```

This will:
1. Train all 3 DSPy optimized models (~90-135 min)
2. Run 3×3 matrix comparison (~30-45 min)
3. Generate results and MLflow logs

### Option 2: Step-by-Step

**Step 1: Train All Models**
```bash
python notebooks/scripts/train_all_models.py
```

Output:
- `models/dspy_bootstrap_llama3.1_8b.json`
- `models/dspy_bootstrap_gemma2_9b.json`
- `models/dspy_bootstrap_mistral_nemo_12b.json`

**Step 2: Run Comparison**
```bash
python notebooks/scripts/compare_3x3_matrix.py --test-samples 100 --seed 42
```

Output:
- `results/3x3_matrix_summary.csv`
- `results/3x3_matrix_detailed_results.csv`

## Results

### Summary CSV
Contains aggregated metrics for each of the 9 configurations:
- Accuracy (%)
- Correct/Wrong counts
- Average Quality Score (1-10 from LLM judge)
- Average Latency (ms)

### Detailed CSV
Contains individual results for each question × configuration:
- Full question text
- Agent response
- Correct answer
- Is correct (True/False)
- Quality score
- Judge reasoning
- Latency

### MLflow
View interactive results:
```bash
mlflow ui --port 8080
```
Then open: http://localhost:8080

Experiment: `aneetaa-3x3-matrix-comparison`

## What Questions to Ask

After running the evaluation, analyze:

1. **Which model works best overall?**
   - Compare accuracy across all 3 models

2. **Does optimization help?**
   - Compare DSPy Optimized vs DSPy Baseline for each model
   - Expected: 5-15% improvement

3. **Is DSPy worth it vs Vanilla?**
   - Compare DSPy Optimized vs Vanilla for each model
   - Look at both accuracy AND latency

4. **Does model size matter?**
   - Compare 8B vs 9B vs 12B parameter models
   - Larger models should be more accurate but slower

5. **Which configuration is best?**
   - Find the sweet spot: high accuracy, good quality, reasonable latency

## Expected Results

### Accuracy (rough estimates):
- Vanilla ANEETAA: 65-75%
- DSPy Baseline: 70-75%
- DSPy Optimized: 75-85%

### Quality Scores:
- All should be 6-8/10 (LLM judge is strict)

### Latency:
- Vanilla: 3,000-5,000ms (uses RAG, slower)
- DSPy Baseline: 2,000-3,500ms (no RAG)
- DSPy Optimized: 2,000-3,500ms (similar to baseline)

## Cost Breakdown

### Training (FREE):
- Using Ollama models locally
- 3 models × 100 questions = 300 training questions
- Cost: $0

### Evaluation (~$0.18):
- LLM Judge calls: 9 configs × 100 questions = 900 calls
- GPT-4o-mini cost: ~$0.0002 per call
- Cost: 900 × $0.0002 = ~$0.18

### Total: ~$0.18

## Troubleshooting

### Training fails
- **Ollama not running:** Start Ollama first
- **Model not found:** Pull models with `ollama pull <model>`
- **Out of memory:** Reduce questions to 50

### Comparison fails
- **Optimized models missing:** Run training step first
- **OPENAI_API_KEY not set:** Add to `.env` file for LLM judge
- **Vanilla agent errors:** Check if ANEETA components are properly installed

### Slow performance
- **Training too slow:** Normal, each model takes 30-45 min
- **Comparison too slow:** Normal with LLM judge, ~30-45 min for 100 questions

## Files Created

```
notebooks/scripts/
├── train_all_models.py              # Train all 3 DSPy models
├── compare_3x3_matrix.py            # Run 3×3 comparison
├── run_full_3x3_evaluation.py       # Master script (runs both)
└── 3x3_EVALUATION_GUIDE.md          # This file

models/
├── dspy_bootstrap_llama3.1_8b.json
├── dspy_bootstrap_gemma2_9b.json
└── dspy_bootstrap_mistral_nemo_12b.json

results/
├── 3x3_matrix_summary.csv           # Aggregated results
└── 3x3_matrix_detailed_results.csv  # Question-level results

mlruns/
└── <experiment_folders>             # MLflow tracking data
```

## Next Steps After Results

1. **Identify best configuration** from summary
2. **Scale up testing** to 200-500 questions for statistical significance
3. **Deploy best model** to production
4. **Document findings** for team
5. **Consider hybrid approaches** if results suggest it

## Questions?

Check the detailed results CSVs and MLflow UI for insights!
