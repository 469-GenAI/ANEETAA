# Using Your Databricks MLflow Server for DSPy Evaluation

## 🎯 Quick Answer

Your Databricks MLflow server is **already configured** in the script! When you run the optimization, everything gets automatically logged to:

**Your Databricks Workspace**: https://dbc-847d0b13-8d35.cloud.databricks.com  
**Your Experiment**: `/Users/benjaminloh.2023@smu.edu.sg/MLflow Tracing Tutorial` (ID: 3840367859585746)

## 🚀 Step-by-Step: Running Evaluation

### Step 1: Test MLflow Connection (Optional but Recommended)

```cmd
cd "c:\SMU_work\Year 3 Sem 1\Gen AI\Project\ANEETAA"
python notebooks\test_mlflow_connection.py
```

This will:

- ✅ Verify your `.env` credentials
- ✅ Test Databricks connection
- ✅ Create a test run in MLflow
- ✅ Show you the URL to view results

**Expected Output:**

```
============================================================
Testing Databricks MLflow Connection
============================================================

1. Checking environment variables...
   ✓ DATABRICKS_TOKEN: ******************** (hidden)
   ✓ DATABRICKS_HOST: https://dbc-847d0b13-8d35.cloud.databricks.com
   ✓ MLFLOW_TRACKING_URI: databricks
   ✓ MLFLOW_EXPERIMENT_ID: 3840367859585746

2. Setting up MLflow tracking...
   ✓ Tracking URI: databricks

3. Testing experiment access...
   ✓ Experiment found!
      Name: /Users/benjaminloh.2023@smu.edu.sg/MLflow Tracing Tutorial
      ID: 3840367859585746

4. Testing run creation...
   ✓ Run created successfully!

✅ All tests passed! Your MLflow connection is working.
```

### Step 2: Run DSPy Optimization (Logs to Your MLflow Server)

**Option A: Using OpenAI** (if you have API key)

```cmd
# Add your OpenAI API key to .env first:
# OPENAI_API_KEY=sk-your-key-here

python notebooks\dspy_optimization.py --provider openai --model gpt-4o-mini --use-rag --train-size 20 --max-demos 3
```

**Option B: Using Ollama** (free, no API key needed)

```cmd
# Make sure Ollama is running first (ollama serve)
python notebooks\dspy_optimization.py --provider ollama --model llama3.1:8b --use-rag --train-size 20 --max-demos 3
```

### Step 3: View Results in Databricks MLflow

1. **Open your Databricks workspace**:

   - Go to: https://dbc-847d0b13-8d35.cloud.databricks.com

2. **Navigate to your experiment**:

   - Click "Experiments" in the left sidebar
   - Find: `/Users/benjaminloh.2023@smu.edu.sg/MLflow Tracing Tutorial`

3. **View the optimization run**:
   - You'll see runs like: `teacher_agent_optimization_2025-11-07_14-30-45`
   - Click on it to see details

## 📊 What Gets Logged to MLflow

### Metrics (Evaluation Results)

```
baseline_score: 0.90          # Original agent performance
optimized_score: 0.95         # After SIMBA optimization
improvement: 5.0              # Percentage improvement
```

### Parameters (Configuration Used)

```
provider: openai
model: gpt-4o-mini
training_mode: rag
max_demos: 3
batch_size: 12
train_size: 20
test_size: 10
```

### Artifacts (Downloadable Files)

- 📦 **Optimized Model**: The improved TeacherAgentDSPy with optimized prompts
- 📝 **Prompts**: Before/after prompt templates
- 🎯 **Demonstrations**: Few-shot examples selected by SIMBA

### Traces (If DSPy Autolog Works)

- 🔍 Every LLM call during optimization
- 📋 Input prompts and output responses
- ⏱️ Latency and token usage

## 🔍 Comparing Runs in MLflow

**Compare different configurations:**

1. Run optimization with different settings:

   ```cmd
   # Run 1: Small training set
   python notebooks\dspy_optimization.py --use-rag --train-size 10 --max-demos 2

   # Run 2: Larger training set
   python notebooks\dspy_optimization.py --use-rag --train-size 30 --max-demos 5

   # Run 3: Different model
   python notebooks\dspy_optimization.py --provider ollama --model gemma2:9b --use-rag --train-size 20
   ```

2. In Databricks MLflow UI:
   - Select multiple runs (checkbox on left)
   - Click "Compare" button
   - View metrics/parameters side-by-side
   - See which configuration performed best!

## 🎓 Understanding the Evaluation Process

### What Happens During Optimization:

```
┌─────────────────────────────────────────────────┐
│ 1. Load RAG Training Data (60 NEET questions)  │
│    - Retrieve context from vector DBs (k=3)    │
│    - Create examples with Q+Context+Answer     │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 2. Split: 80% train (48 examples)              │
│           20% test (12 examples)                │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 3. Baseline Evaluation                         │
│    - Run vanilla TeacherAgentDSPy on test set  │
│    - Measure quality with validate_explanation │
│    - Log baseline_score to MLflow              │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 4. SIMBA Optimization                          │
│    - Bootstrap prompts with max_demos examples │
│    - Evaluate on training subset              │
│    - Iteratively improve prompts              │
│    - Each iteration logged to MLflow          │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 5. Optimized Evaluation                        │
│    - Run optimized TeacherAgentDSPy on test   │
│    - Measure quality again                     │
│    - Log optimized_score to MLflow             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 6. Log Model & Artifacts                       │
│    - Save optimized model to MLflow            │
│    - Calculate improvement percentage          │
│    - All tracked in your Databricks experiment │
└─────────────────────────────────────────────────┘
```

## 🛠️ Troubleshooting

### "Can't connect to Databricks"

- Check your `.env` file has all 4 variables:
  - `DATABRICKS_TOKEN`
  - `DATABRICKS_HOST`
  - `MLFLOW_TRACKING_URI=databricks`
  - `MLFLOW_EXPERIMENT_ID`
- Run `python notebooks\test_mlflow_connection.py` to diagnose

### "Experiment not found"

- Verify experiment ID is correct: `3840367859585746`
- Check you have access to this experiment in Databricks

### "No runs appearing in MLflow"

- Check the script completed without errors
- Refresh the Databricks UI
- Make sure you're looking at the correct experiment

## 📝 Example: Full Workflow

```cmd
# 1. Test connection
python notebooks\test_mlflow_connection.py

# 2. Run small test optimization (fast, ~5 min)
python notebooks\dspy_optimization.py --provider ollama --model llama3.1:8b --use-rag --train-size 10 --max-demos 2

# 3. Open Databricks and verify run appears

# 4. Run full optimization (slower, ~20 min)
python notebooks\dspy_optimization.py --provider ollama --model llama3.1:8b --use-rag --train-size 30 --max-demos 5

# 5. Compare the two runs in MLflow UI
```

## 🎯 Key Takeaway

**You don't need to do anything special!** Your MLflow server is already configured via `.env`. Just run the script and all evaluation metrics will automatically appear in your Databricks MLflow experiment. 🚀

---

_Need help? Check `notebooks/TESTING_SUMMARY.md` for more details._
