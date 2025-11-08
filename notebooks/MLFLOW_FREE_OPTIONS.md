# Free MLflow Options for ANEETAA DSPy Optimization

Since you've run out of Databricks credits, here are your **100% FREE** alternatives for MLflow tracking:

## ✅ Option 1: Local MLflow (RECOMMENDED - Already Configured!)

**What I just set up for you:**
- MLflow runs **locally on your machine** - completely free!
- All experiment data saved to `mlruns/` folder in your project
- No external servers, no credits needed, no internet required for tracking

**How to use:**

1. **Your `.env` is already configured** to use local MLflow:
   ```properties
   USE_DATABRICKS_MLFLOW=false
   ```

2. **Run your optimization normally:**
   ```cmd
   python notebooks\dspy_optimization.py --provider openai --model gpt-4o-mini --use-json --train-size 15 --test-samples 5
   ```

3. **View results in MLflow UI:**
   ```cmd
   mlflow ui
   ```
   Then open: http://localhost:5000

**What you'll see in the UI:**
- ✅ All experiment runs with timestamps
- ✅ Baseline vs Optimized scores
- ✅ Model parameters and hyperparameters
- ✅ Metrics (improvement %, scores)
- ✅ Logged models (if enabled)
- ✅ Full trace of DSPy optimization

**Pros:**
- 100% FREE forever
- Fast - no network latency
- Works offline
- Full MLflow features
- Your data stays on your machine

**Cons:**
- No team collaboration (unless you share the mlruns/ folder)
- Need to run `mlflow ui` locally to view

---

## Option 2: MLflow on Free Cloud Platforms

If you need team collaboration, you can deploy MLflow to free cloud platforms:

### A. **Dagshub** (Free tier)
- Website: https://dagshub.com
- Free tier: Unlimited public repos, 10GB storage
- Auto-syncs with GitHub
- Built-in MLflow tracking server

**Setup:**
```bash
pip install dagshub
```

In your code:
```python
import dagshub
dagshub.init(repo_owner="YOUR_USERNAME", repo_name="ANEETAA", mlflow=True)
```

### B. **Railway.app** (Free $5/month credit)
- Deploy your own MLflow server
- Free $5 monthly credit (enough for small projects)
- Auto-sleeps when not in use

### C. **Render.com** (Free tier)
- Deploy MLflow server
- Free tier includes 750 hours/month
- Auto-sleeps after 15 min inactivity

---

## Option 3: Google Colab + Google Drive

Store MLflow artifacts in Google Drive (15GB free):

```python
from google.colab import drive
drive.mount('/content/drive')

mlflow.set_tracking_uri('file:///content/drive/MyDrive/mlflow')
```

---

## Comparison Table

| Option | Cost | Team Sharing | Setup Difficulty | Storage |
|--------|------|--------------|------------------|---------|
| **Local MLflow** | **FREE** | Manual (share folder) | **Easy** | Unlimited (your disk) |
| Dagshub | FREE | Yes | Easy | 10GB |
| Railway | $5/mo credit | Yes | Medium | Limited by credit |
| Render | FREE | Yes | Medium | Limited |
| Google Drive | FREE | Yes (via Drive) | Medium | 15GB |

---

## Current Status

✅ **Your project is now using Local MLflow (FREE)**
- Just run your optimization script normally
- Use `mlflow ui` to view results
- All data saved to `mlruns/` folder

---

## To Switch Back to Databricks Later

If you get more credits, just change in `.env`:
```properties
USE_DATABRICKS_MLFLOW=true
```

That's it! 🎉
