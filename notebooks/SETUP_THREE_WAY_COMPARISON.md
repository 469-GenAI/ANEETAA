# Three-Way Agent Comparison Setup Guide

## 🎯 Goal
Compare three agent implementations:
1. **Vanilla ANEETAA** - Original LangChain implementation
2. **DSPy Baseline** - Unoptimized DSPy module
3. **DSPy Optimized** - After SIMBA/COPRO optimization

---

## ✅ Prerequisites

### 1. Check DSPy Installation
```powershell
python -m pip show dspy-ai
```
**Expected**: Version 2.6.x or higher

### 2. Verify OpenAI API Key
```powershell
# Check .env file has:
OPENAI_API_KEY=sk-...
```

### 3. Verify Ollama Running
```powershell
ollama list
```
**Expected**: gemma2:9b, llama3.1:8b models available

---

## 📋 Setup Steps

### Step 1: Run Baseline Comparison (No Optimization Yet)

This compares **Vanilla ANEETAA** vs **DSPy Baseline** (unoptimized):

```powershell
python notebooks/compare_three_agents.py --test-samples 5
```

**What this does:**
- Vanilla ANEETAA uses Ollama `gemma2:9b`
- DSPy Baseline uses OpenAI `gpt-4o-mini` (unoptimized)
- DSPy Optimized falls back to baseline (no optimized model yet)

**Expected Output:**
```
Vanilla ANEETAA          | Accuracy: XX.X% | Latency: XXXXms
DSPy Baseline            | Accuracy: XX.X% | Latency: XXXXms
DSPy Optimized           | Accuracy: XX.X% | Latency: XXXXms (same as baseline)
```

---

### Step 2: Create Training Data for DSPy Optimization

DSPy optimization needs training examples. Create a training set:

```powershell
python notebooks/create_training_data.py --output-file dspy_training_data.json --num-examples 50
```

*(We'll create this script if it doesn't exist)*

---

### Step 3: Run DSPy SIMBA Optimization

Optimize the DSPy module using SIMBA:

```powershell
python notebooks/dspy_optimization.py --training-data dspy_training_data.json --optimizer simba --output optimized_mcq_solver.json
```

**What this does:**
- Loads 50 training examples
- Runs SIMBA optimizer to improve prompts
- Saves optimized model to `optimized_mcq_solver.json`

**Time**: ~10-15 minutes depending on training set size

---

### Step 4: Compare All Three Agents

Now run the full comparison with the optimized model:

```powershell
python notebooks/compare_three_agents.py --test-samples 10 --optimized-model-path optimized_mcq_solver.json
```

**Expected Output:**
```
Vanilla ANEETAA          | Accuracy: XX.X% | Latency: XXXXms
DSPy Baseline            | Accuracy: XX.X% | Latency: XXXXms
DSPy Optimized           | Accuracy: YY.Y% | Latency: XXXXms (should be higher accuracy)
```

---

## 🎛️ Command Options

### Basic Run (5 questions)
```powershell
python notebooks/compare_three_agents.py --test-samples 5
```

### With Optimized Model
```powershell
python notebooks/compare_three_agents.py --test-samples 10 --optimized-model-path optimized_mcq_solver.json
```

### Use Ollama for DSPy (instead of OpenAI)
```powershell
python notebooks/compare_three_agents.py --test-samples 5 --dspy-provider ollama --dspy-model llama3.1:8b
```

### Different Vanilla Model
```powershell
python notebooks/compare_three_agents.py --test-samples 5 --vanilla-model llama3.1:8b
```

### Reproducible Seed
```powershell
python notebooks/compare_three_agents.py --test-samples 10 --seed 123
```

---

## 📊 View Results in MLflow

### 1. Start MLflow UI (if not running)
```powershell
mlflow ui --port 8080
```

### 2. Open Browser
```
http://localhost:8080
```

### 3. Navigate to Experiment
- Experiment name: `aneetaa-three-way-comparison`

### 4. Compare Runs
- Click "Compare" to see side-by-side metrics
- View artifacts to see detailed CSV results

---

## 🎯 What You're Testing

### Vanilla ANEETAA
- **Pros**: Battle-tested, uses RAG, works with local models
- **Cons**: Not optimized, manual prompt engineering

### DSPy Baseline (Unoptimized)
- **Pros**: Structured signatures, composable modules
- **Cons**: No optimization, baseline prompts may be suboptimal

### DSPy Optimized (After SIMBA)
- **Pros**: Optimized prompts via SIMBA, better accuracy expected
- **Cons**: Requires training data, optimization time

---

## 🚀 Quick Start (No Optimization)

If you just want to see the comparison without optimization:

```powershell
# 1. Run baseline comparison
python notebooks/compare_three_agents.py --test-samples 5

# 2. View in MLflow
# Open http://localhost:8080 and check experiment "aneetaa-three-way-comparison"
```

This will compare Vanilla vs DSPy Baseline. DSPy Optimized will be the same as Baseline until you run optimization.

---

## ⚠️ Troubleshooting

### "OPENAI_API_KEY not set"
```powershell
# Add to .env file:
OPENAI_API_KEY=sk-your-key-here
```

### "No optimized model found, using baseline"
This is normal if you haven't run optimization yet. DSPy Optimized will fall back to baseline.

### Slow performance
- Reduce `--test-samples` to 3-5 for quick tests
- Use Ollama instead of OpenAI: `--dspy-provider ollama --dspy-model llama3.1:8b`

---

## 📈 Expected Results

Based on typical DSPy optimization:

| Agent | Expected Accuracy | Notes |
|-------|------------------|-------|
| Vanilla ANEETAA | 70-90% | Depends on model (gemma2 vs llama3.1) |
| DSPy Baseline | 60-80% | Unoptimized prompts |
| DSPy Optimized | 75-95% | **5-15% improvement** expected |

**Goal**: DSPy Optimized should outperform both Vanilla and Baseline!

---

## 🎓 Next Steps After Comparison

1. **Analyze MLflow results** - Which agent performs best?
2. **Run larger evaluation** - 50-100 questions for statistical significance
3. **Try different optimizers** - COPRO, MIPROv2, Bootstrap
4. **Tune hyperparameters** - Temperature, max_tokens, etc.
5. **Combine strengths** - Use DSPy-optimized prompts in Vanilla ANEETAA

---

**Ready to start?** Run:
```powershell
python notebooks/compare_three_agents.py --test-samples 5
```
