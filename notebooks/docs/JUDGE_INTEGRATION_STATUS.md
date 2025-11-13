# LLM Judge Integration Status

**Last Updated:** November 12, 2025

## Overview

All comparison and evaluation scripts now use the centralized LLM judge configuration system from `notebooks/scripts/llm_judge_config.py`. This allows you to switch judges across all scripts by modifying a single configuration file or using the CLI controller.

## ✅ Fully Integrated Files

### DSPy Comparison Scripts (notebooks/scripts/)

1. **compare_3x3_matrix.py**
   - Status: ✅ Integrated
   - Judge Function: `judge_answer_quality()`
   - Import: `from llm_judge_config import get_judge_llm, estimate_judge_cost, JUDGE_CONFIG`
   - Cost Display: Yes (shows before evaluation)
   - Use Case: 3×3 matrix comparison with quality scoring

### Root Evaluation Scripts (scripts/)

2. **mlflow_mcq_solver.py**
   - Status: ✅ Integrated (Updated Nov 12, 2025)
   - Judge Function: `judge_answer_quality()`
   - Import: `from llm_judge_config import get_judge_llm, estimate_judge_cost, JUDGE_CONFIG`
   - Cost Display: Yes (shows before evaluation)
   - Use Case: MLflow-tracked multi-model comparison with quality scoring

3. **mcq_eval.py**
   - Status: ✅ Integrated (Updated Nov 12, 2025)
   - Judge Function: `judge_answer_quality()`
   - Import: `from llm_judge_config import get_judge_llm, estimate_judge_cost, JUDGE_CONFIG`
   - Cost Display: Yes (shows before evaluation)
   - Use Case: Basic MCQ evaluation with quality scoring

## ❌ No Judge Integration (By Design)

These files don't use LLM judges because they only perform correctness evaluation (binary right/wrong):

1. **compare_three_agents.py**
   - Status: No judge needed
   - Evaluation: Binary correctness only (fact_check_answer)
   - Use Case: Three-way agent comparison (Vanilla, DSPy Baseline, DSPy Optimized)

2. **compare_3x3_simple.py**
   - Status: No judge needed
   - Evaluation: Binary correctness only
   - Use Case: DSPy-only 3×2 matrix comparison (avoids vanilla ANEETAA hanging)

## How to Switch Judges

### Method 1: CLI Controller (Recommended)

```powershell
# Switch to Groq (FREE)
python notebooks/scripts/judge_config_controller.py --preset groq-default

# Switch to OpenAI GPT-4o (stronger evaluation)
python notebooks/scripts/judge_config_controller.py --preset openai-strong

# Switch to OpenAI GPT-4o-mini (balanced)
python notebooks/scripts/judge_config_controller.py --preset openai-mini

# Test the current judge
python notebooks/scripts/judge_config_controller.py --test

# Show current configuration
python notebooks/scripts/judge_config_controller.py --show
```

### Method 2: Manual Configuration

Edit `notebooks/scripts/llm_judge_config.py`:

```python
JUDGE_CONFIG = {
    'provider': 'groq',  # 'openai', 'groq', 'anthropic', 'ollama'
    'model': 'default',   # or specific model name
    'temperature': 0
}
```

## Available Presets

| Preset | Provider | Model | Cost/40 Questions | Use Case |
|--------|----------|-------|-------------------|----------|
| `groq-default` | Groq | llama-3.1-70b | **$0.00** | Fast, free comparisons |
| `groq-strong` | Groq | llama-3.2-90b | **$0.00** | Free high-quality evaluation |
| `openai-mini` | OpenAI | gpt-4o-mini | ~$0.03 | Balanced quality/cost |
| `openai-strong` | OpenAI | gpt-4o | ~$0.44 | Highest quality evaluation |
| `anthropic-default` | Anthropic | claude-3-5-sonnet | ~$0.48 | Alternative strong judge |
| `ollama-local` | Ollama | llama3.1:8b | **$0.00** | Local, no API needed |

## Cost Estimation

All integrated scripts display estimated costs before running:

```
LLM JUDGE CONFIGURATION
======================================================================
Provider: groq
Model: llama-3.1-70b-versatile
Temperature: 0
Estimated Cost: $0.0000
======================================================================
```

## Judge Configuration Details

### OpenAI
- Models: gpt-4o-mini ($0.15/$0.60 per 1M tokens), gpt-4o ($2.50/$10.00 per 1M tokens)
- API Key: Required in `.env` as `OPENAI_API_KEY`
- Best for: High-quality evaluation when budget allows

### Groq (Recommended for Cost-Effective Evaluation)
- Models: llama-3.1-70b-versatile, llama-3.2-90b-text-preview
- API Key: Required in `.env` as `GROQ_API_KEY`
- Cost: **FREE** (with rate limits)
- Best for: Most comparisons and experiments

### Anthropic
- Models: claude-3-5-sonnet-20241022
- API Key: Required in `.env` as `ANTHROPIC_API_KEY`
- Best for: Alternative perspective to OpenAI

### Ollama
- Models: Any locally installed model (llama3.1:8b, gemma2:9b, etc.)
- API Key: Not required
- Cost: **FREE** (runs locally)
- Best for: Offline evaluation, no API costs

## Migration Notes

### Changes Made (Nov 12, 2025)

**mlflow_mcq_solver.py:**
- Removed: `from langchain_openai import ChatOpenAI`
- Added: `from llm_judge_config import get_judge_llm, estimate_judge_cost, JUDGE_CONFIG`
- Updated: `judge_answer_quality()` to use `get_judge_llm()` instead of hardcoded ChatOpenAI
- Added: Judge configuration display in main()

**mcq_eval.py:**
- Removed: `from langchain_openai import ChatOpenAI`
- Added: `from llm_judge_config import get_judge_llm, estimate_judge_cost, JUDGE_CONFIG`
- Updated: `judge_answer_quality()` to use `get_judge_llm()` instead of hardcoded ChatOpenAI
- Added: Judge configuration display before evaluation

### Backward Compatibility

All scripts maintain the same command-line interface. No changes to how you run them:

```powershell
# MLflow evaluation (works as before)
python scripts/mlflow_mcq_solver.py --test-samples 40

# MCQ evaluation (works as before)
python scripts/mcq_eval.py

# 3x3 matrix comparison (works as before)
python notebooks/scripts/compare_3x3_matrix.py --test-samples 40
```

The only difference: they now respect the centralized judge configuration!

## Recommended Workflow

### 1. Development/Experimentation (Use Groq)
```powershell
# Set to Groq for free evaluation
python notebooks/scripts/judge_config_controller.py --preset groq-default

# Run comparisons without cost
python notebooks/scripts/compare_3x3_matrix.py --test-samples 40
python scripts/mlflow_mcq_solver.py --test-samples 40
```

### 2. Final Quality Check (Use OpenAI GPT-4o)
```powershell
# Switch to strongest judge
python notebooks/scripts/judge_config_controller.py --preset openai-strong

# Run final evaluation with best judge
python notebooks/scripts/compare_3x3_matrix.py --test-samples 40
```

### 3. Budget-Conscious Evaluation (Use GPT-4o-mini)
```powershell
# Balanced quality and cost
python notebooks/scripts/judge_config_controller.py --preset openai-mini

# Good quality at low cost
python scripts/mlflow_mcq_solver.py --test-samples 40
```

## Troubleshooting

### Import Error: "llm_judge_config could not be resolved"
This is a linting error only. The scripts add the path at runtime:
```python
NOTEBOOKS_SCRIPTS = ROOT / "notebooks" / "scripts"
sys.path.insert(0, str(NOTEBOOKS_SCRIPTS))
```

### Judge Returns Error
Check that the appropriate API key is set in `.env`:
```bash
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
```

### Ollama Judge Fails
Ensure the model is installed:
```powershell
ollama list
ollama pull llama3.1:8b  # if needed
```

## Next Steps

1. **Test Judge Controller:** Run `python notebooks/scripts/judge_config_controller.py --test`
2. **Try Groq:** Switch to free Groq judge and run a comparison
3. **Compare Judges:** Run same evaluation with different judges to see quality differences
4. **Document Findings:** Note which judge works best for your use case

## Related Documentation

- [LLM_JUDGE_GUIDE.md](LLM_JUDGE_GUIDE.md) - Complete judge configuration guide
- [SPLIT_DATASET_UPDATE.md](SPLIT_DATASET_UPDATE.md) - Dataset split documentation
- [RUN_GUIDE.md](../docs/RUN_GUIDE.md) - General usage guide
