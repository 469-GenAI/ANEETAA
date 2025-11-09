# Four-Way Agent Comparison Setup Guide

## Overview
This setup compares **4 different agent architectures** for ANEETAA MCQ evaluation:

1. **Vanilla ANEETAA** - Original graph-based agent with LangGraph
2. **DSPy Baseline** - Unoptimized DSPy CoT module  
3. **DSPy + MIPROv2** - Optimized with MIPROv2 (prompt + demo optimization)
4. **DSPy + BootstrapFewShot** - Optimized with BootstrapFewShot (example selection)

## Why This Comparison?

### MIPROv2 vs BootstrapFewShot
- **MIPROv2**: Optimizes both instruction prompts AND demonstration selection
  - Best for: Complex reasoning tasks requiring good prompts
  - Optimization time: 5-10 minutes
  - Expected improvement: 5-15% over baseline

- **BootstrapFewShot**: Optimizes only demonstration selection
  - Best for: Tasks where examples matter more than prompts
  - Optimization time: 3-5 minutes  
  - Expected improvement: 3-10% over baseline

### Why Not SIMBA?
SIMBA is not available in DSPy 2.6.5. However:
- **MIPROv2 is actually newer and often better than SIMBA**
- MIPROv2 uses more advanced optimization techniques
- For MCQ tasks, MIPROv2 typically outperforms SIMBA

## Step-by-Step Process

### Step 1: Create Training Data ✅ COMPLETED
```bash
python notebooks/create_training_data.py --num-examples 30 --output-file dspy_training_data.json
```

**Result:** Created 30 training examples from Gemini 2.5 Pro Data

### Step 2: Run Dual Optimization ⏳ IN PROGRESS
```bash
python notebooks/optimize_both_methods.py --train-size 20 --test-size 10 --training-data dspy_training_data.json
```

**What's happening:**
- Training MIPROv2 on 20 examples (using OpenAI GPT-4o-mini)
- Training BootstrapFewShot on 20 examples
- Testing both on 10 validation examples
- Saving optimized models:
  - `dspy_mipro_optimized.json`
  - `models/dspy_bootstrap_optimized.json`

**Expected time:** 8-15 minutes total
- MIPROv2: 5-10 minutes
- BootstrapFewShot: 3-5 minutes

### Step 3: Run Four-Way Comparison (PENDING)
```bash
python notebooks/compare_four_agents.py --test-samples 10 \
  --mipro-model dspy_mipro_optimized.json \
  --bootstrap-model models/dspy_bootstrap_optimized.json
```

**This will:**
1. Load 10 test questions (different from training data)
2. Evaluate all 4 agents on the same questions
3. Calculate accuracy and latency for each
4. Save results to `four_way_comparison_results.csv`
5. Log metrics to MLflow experiment `aneetaa-four-way-comparison`

## Expected Results

### Baseline (from previous runs)
- Vanilla ANEETAA: ~80% accuracy, ~4,500ms latency
- DSPy Baseline: ~80% accuracy, ~3,650ms latency

### Optimized (predictions)
- DSPy + MIPROv2: 85-95% accuracy (5-15% improvement)
- DSPy + BootstrapFewShot: 83-90% accuracy (3-10% improvement)

## Files Created

### Optimization Scripts
- `notebooks/optimize_both_methods.py` - Dual optimization script
- `notebooks/compare_four_agents.py` - Four-way comparison script
- `notebooks/check_optimizers.py` - Check available DSPy optimizers
- `notebooks/create_training_data.py` - Generate training data

### Data Files
- `dspy_training_data.json` - 30 training examples
- `dspy_mipro_optimized.json` - MIPROv2 optimized model (pending)
- `models/dspy_bootstrap_optimized.json` - BootstrapFewShot optimized model (pending)

### Results Files
- `four_way_comparison_results.csv` - Detailed results (pending)
- MLflow experiments:
  - `aneetaa-dspy-dual-optimization` - Optimization metrics
  - `aneetaa-four-way-comparison` - Comparison results

## Interpretation Guide

### What to Look For

1. **Does optimization help?**
   - Compare DSPy Baseline vs DSPy + MIPROv2/BootstrapFewShot
   - If optimized agents > baseline by 5%+, optimization is working

2. **Which optimizer is better?**
   - Compare MIPROv2 vs BootstrapFewShot
   - MIPROv2 should be better for reasoning-heavy questions
   - BootstrapFewShot might be better for pattern-matching

3. **Is DSPy worth it?**
   - Compare Vanilla ANEETAA vs DSPy agents
   - DSPy should be faster (lower latency)
   - If optimized DSPy > Vanilla, DSPy is worth adopting

4. **Speed vs Accuracy Trade-off**
   - Vanilla: Slower but potentially more accurate (uses RAG)
   - DSPy: Faster but relies on model reasoning
   - Optimized DSPy: Best of both worlds?

### Success Criteria

✅ **Excellent:** Optimized DSPy achieves 90%+ accuracy with <3000ms latency
✅ **Good:** Optimized DSPy achieves 85%+ accuracy, beats Vanilla
✅ **Acceptable:** Optimized DSPy achieves 80%+ accuracy, faster than Vanilla
⚠️ **Needs Work:** Optimized DSPy < 80% or slower than baseline

## Next Steps After Comparison

### If Optimization Works Well (>85% accuracy)
1. Scale up training data to 50-100 examples
2. Run optimization with larger dataset
3. Test on 50+ questions for statistical significance
4. Deploy best-performing agent to production

### If Optimization Needs Improvement
1. Analyze failed questions (which subjects/types)
2. Add more diverse training examples
3. Try different optimization parameters:
   - Increase `num_candidates` for MIPROv2
   - Increase `max_bootstrapped_demos` for BootstrapFewShot
4. Experiment with different base models (llama3.1:8b vs gemma2:9b)

### If DSPy Underperforms Vanilla
1. Investigate why RAG helps (which questions benefit)
2. Consider hybrid approach: DSPy + RAG
3. Analyze latency breakdown (where time is spent)
4. Optimize Vanilla ANEETAA instead

## MLflow Tracking

### View Results
```bash
python -m mlflow ui --port 8080
```
Then open: http://localhost:8080

### Experiments to Check
1. **aneetaa-dspy-dual-optimization**
   - Optimization progress metrics
   - Baseline vs optimized accuracy
   - Training/validation losses

2. **aneetaa-four-way-comparison**
   - Per-agent accuracy
   - Per-agent latency
   - Question-level results

## Troubleshooting

### Optimization Takes Too Long
- Reduce `--train-size` to 15 or 10
- Reduce `num_candidates` in MIPROv2 to 5
- Use Ollama instead of OpenAI (slower but free)

### Low Accuracy After Optimization
- Check training data quality (are answers correct?)
- Increase training size to 30-40 examples
- Try different metric function (more lenient)
- Check if model is overfitting (test on different questions)

### Memory Issues
- Reduce batch sizes
- Use smaller base model (gemma2:2b)
- Close other applications

## Current Status

- ✅ Training data created (30 examples)
- ⏳ Dual optimization running (MIPROv2 + BootstrapFewShot)
- ⏳ Waiting for optimized models to be saved
- 📋 Next: Run four-way comparison

**ETA for completion:** ~10-15 minutes from now
