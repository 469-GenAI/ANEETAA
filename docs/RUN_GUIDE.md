# ANEETAA DSPy Evaluation - Quick Start Guide

## 🚀 Step-by-Step Execution Guide

### Prerequisites Check ✓
- [x] Ollama running with models: llama3.1:8b, gemma2:9b, mistral-nemo:12b
- [x] Python 3.14 with all dependencies installed
- [x] .env file configured for local MLflow
- [x] 250 test questions ready in aneeta_v2/Processed Data/Gemini 2.5 Pro Data/

---

## STEP 1: Start MLflow UI Server

**Open a NEW PowerShell window** and run:
```powershell
cd "d:\Git Projects\SMU\ANEETAA"
python -m mlflow ui --port 8080
```

**Or use this one-liner:**
```powershell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\Git Projects\SMU\ANEETAA'; python -m mlflow ui --port 8080"
```

✅ **Verify**: Open http://localhost:8080 in your browser
- You should see the MLflow UI
- Experiment name: `aneetaa-vanilla-vs-dspy`

**Keep this window running!** Don't close it.

---

## STEP 2: Test Vanilla Agent (Quick Smoke Test)

In your **main PowerShell window**:
```powershell
python notebooks/test_vanilla_agent.py
```

**What this does:**
- Tests that vanilla ANEETAA agents work
- Verifies Ollama connection
- Checks vector database access
- Quick sanity check (should complete in < 1 minute)

**Expected output:**
- Agent successfully loads
- Answers a sample NEET question
- No errors about missing dependencies

---

## STEP 3: Run Full Agent Comparison

### Option A: Quick Test (10 samples)
```powershell
python notebooks/compare_agents.py --provider ollama --model gemma2:9b --test-samples 10
```

### Option B: Medium Test (50 samples)
```powershell
python notebooks/compare_agents.py --provider ollama --model gemma2:9b --test-samples 50
```

### Option C: Full Evaluation (all questions)
```powershell
python notebooks/compare_agents.py --provider ollama --model gemma2:9b
```

**What this does:**
- Loads test questions from Gemini 2.5 Pro Data
- Runs vanilla ANEETAA agent on each question
- Runs DSPy-optimized agent on each question
- Uses GPT-4o-mini as LLM judge to compare answers
- Logs all results to MLflow

**Expected runtime:**
- 10 samples: ~5-10 minutes
- 50 samples: ~20-30 minutes
- Full set (250): ~1-2 hours

**Monitor progress:**
- Watch the terminal for progress updates
- Check MLflow UI (http://localhost:8080) to see metrics in real-time

---

## STEP 4: View Results in MLflow

1. Open http://localhost:8080
2. Click on experiment: `aneetaa-vanilla-vs-dspy`
3. You'll see runs for:
   - Vanilla agent evaluation
   - DSPy-optimized agent evaluation
   - Comparison metrics

**Key metrics to look for:**
- Accuracy scores
- Average response time
- Answer quality ratings (from GPT-4o-mini judge)
- Comparison charts

---

## STEP 5 (Optional): Optimize DSPy Agents

If you want to create NEW optimized agents:

```powershell
python notebooks/dspy_optimization.py --provider ollama --model gemma2:9b --train-size 30
```

**What this does:**
- Trains DSPy agents using SIMBA optimizer
- Uses 30 training examples from your processed data
- Creates optimized prompts/signatures
- Logs optimization results to MLflow

**Runtime:** ~15-30 minutes

---

## 📊 Quick Commands Reference

| Task | Command |
|------|---------|
| Start MLflow | `python -m mlflow ui --port 8080` |
| Quick test | `python notebooks/test_vanilla_agent.py` |
| Compare (10) | `python notebooks/compare_agents.py --provider ollama --model gemma2:9b --test-samples 10` |
| Compare (50) | `python notebooks/compare_agents.py --provider ollama --model gemma2:9b --test-samples 50` |
| Full compare | `python notebooks/compare_agents.py --provider ollama --model gemma2:9b` |
| Optimize | `python notebooks/dspy_optimization.py --provider ollama --model gemma2:9b` |
| View results | Open http://localhost:8080 |

---

## 🔧 Troubleshooting

### MLflow UI won't start
```powershell
# Check if port 8080 is in use
netstat -ano | findstr :8080

# If something's using it, kill it or use a different port
python -m mlflow ui --port 8081
```

### Ollama not responding
```powershell
# Check if Ollama is running
ollama list

# If not, start it (it should auto-start on Windows)
# Or restart: taskkill /F /IM ollama.exe && ollama serve
```

### Python module errors
```powershell
# Reinstall dependencies
python -m pip install -r requirements.txt
```

---

## 🎯 Recommended First Run

**For your first time, I recommend:**

1. **Start MLflow** (keep it running)
   ```powershell
   Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\Git Projects\SMU\ANEETAA'; python -m mlflow ui --port 8080"
   ```

2. **Quick test** (verify setup)
   ```powershell
   python notebooks/test_vanilla_agent.py
   ```

3. **Small comparison** (10 samples to see how it works)
   ```powershell
   python notebooks/compare_agents.py --provider ollama --model gemma2:9b --test-samples 10
   ```

4. **Check results** at http://localhost:8080

5. **If all looks good, run full comparison** (250 samples)
   ```powershell
   python notebooks/compare_agents.py --provider ollama --model gemma2:9b
   ```

---

## 📈 What to Expect

**Vanilla Agent:**
- Uses standard LangChain prompts
- Retrieval-Augmented Generation (RAG) from NCERT books
- Subject-specific agents (Biology, Chemistry, Physics)

**DSPy-Optimized Agent:**
- Optimized prompts via SIMBA
- Better structured reasoning
- Improved answer accuracy

**LLM Judge (GPT-4o-mini):**
- Evaluates answer quality
- Scores on correctness, completeness, clarity
- Provides comparison metrics

**Results in MLflow:**
- Side-by-side metrics
- Performance charts
- Individual question analysis
- Exportable data

---

**Ready to start? Let me know if you want me to run the first test for you!** 🚀
