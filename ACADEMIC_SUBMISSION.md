# ANEETA: DSPy Optimization Reproduction Guide

## Prerequisites

### Environment Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# Key packages: dspy-ai==2.6.5, mlflow==3.6.0, langchain, chromadb==0.4.24

# 2. Install Ollama (https://ollama.ai) and pull models
ollama pull llama3.1:8b
ollama pull gemma2:9b
ollama pull mistral-nemo:12b
ollama pull nomic-embed-text

# 3. Configure environment variables (.env file)
OPENAI_API_KEY=your_key_here        # For DSPy optimization
GROQ_API_KEY=your_key_here          # For LLM judge (optional, FREE)
OLLAMA_URL=http://localhost:11434
```

---

## Step 1: Data Processing

### 1.1 Dataset Structure

Location: `aneeta_v2/Processed Data/Gemini 2.5 Pro Data/`

- 7,874 NEET MCQs (Physics, Chemistry, Biology)
- Format: 250 JSON files (01.json - 250.json)
- Each question: text, 4 options, correct answer, explanation

### 1.2 Split into Train/Validation Sets

```bash
# Script creates:
# - dspy_dataset_train.jsonl (6,300 questions, seed=42)
# - dspy_dataset_val.jsonl (1,575 questions, seed=43)
# - dspy_dataset_combined.jsonl (all 7,874 questions)

python scripts/prepare_dspy_dataset.py
```

**Output**: Three JSONL files in `aneeta_v2/Processed Data/`

---

## Step 2: DSPy Training

### 2.1 Train All Models with MIPROv2

```bash
# Trains 6 models (3 LLMs × 2 optimizers: Bootstrap + MIPROv2)
# Uses 200 training samples, 40 validation samples
# Training time: ~2-3 hours total

python notebooks/scripts/train_all_models_miprov2.py
```

**What happens:**

1. Loads training data from `dspy_dataset_train.jsonl`
2. For each model (Llama 3.1 8B, Gemma2 9B, Mistral Nemo 12B):
   - Trains Bootstrap optimizer (4 few-shot demos)
   - Trains MIPROv2 optimizer (5 instruction candidates)
3. Saves optimized models to `models/` directory
4. Logs all runs to MLflow

**Output files** (in `models/` directory):

```
dspy_bootstrap_llama3.1_8b.json
dspy_mipro_llama3.1_8b.json
dspy_bootstrap_gemma2_9b.json
dspy_mipro_gemma2_9b.json
dspy_bootstrap_mistral-nemo_12b.json
dspy_mipro_mistral-nemo_12b.json
```

### 2.2 Training Configuration

**BootstrapFewShot**:

- `max_bootstrapped_demos=4`: Selects 4 best few-shot examples
- `max_rounds=1`: Single optimization pass
- Metric: Binary accuracy (correct answer match)

**MIPROv2**:

- `num_candidates=5`: Tests 5 instruction variants
- `init_temperature=1.0`: Exploration parameter
- Optimizes both instructions and demonstrations

---

## Step 3: MLflow Tracking

### 3.1 Start MLflow UI

```bash
# Open in separate terminal/window
mlflow ui --port 8080

# Access at: http://localhost:8080
```

### 3.2 View Training Runs

1. Navigate to experiment: `mcq-solver-training`
2. Compare runs by:
   - Model name (llama3.1:8b, gemma2:9b, mistral-nemo:12b)
   - Optimizer (bootstrap, mipro)
   - Metrics: train_accuracy, val_accuracy
3. Download model artifacts from run details

---

## Step 4: Evaluation (3×4 Matrix)

### 4.1 Configure LLM Judge

```bash
# Option 1: Use Groq (FREE)
python notebooks/scripts/judge_config_controller.py --preset groq-default

# Option 2: Use OpenAI GPT-4o (higher quality)
python notebooks/scripts/judge_config_controller.py --preset openai-strong

# Test judge configuration
python notebooks/scripts/judge_config_controller.py --test
```

### 4.2 Run Full 3×4 Matrix Comparison

```bash
# Evaluates all 12 configurations (3 models × 4 agent types)
# Uses 40 validation questions (seed=123)
# Evaluation time: ~1-2 hours

python notebooks/scripts/compare_3x4_matrix.py
```

**What happens:**

1. Loads 40 validation questions from `dspy_dataset_val.jsonl`
2. For each model (Llama, Gemma2, Mistral):
   - Runs Vanilla ANEETAA agent (RAG-based)
   - Runs DSPy Baseline (unoptimized signatures)
   - Runs DSPy Bootstrap (loads trained model)
   - Runs DSPy MIPROv2 (loads trained model)
3. For each answer:
   - Calculates Fact Score (0/10 binary: correct/wrong)
   - Calculates Quality Score (1-10 via LLM judge)
4. Logs all results to MLflow
5. Saves CSV outputs

**Output files** (in `results/` directory):

```
3x4_matrix_detailed_results.csv    # 480 rows (12 configs × 40 questions)
3x4_matrix_summary.csv             # 12 rows (aggregated metrics)
```

### 4.3 Quick Test (10 Questions)

```bash
# For faster testing, use subset
python notebooks/scripts/compare_3x4_matrix.py --test-samples 10
```

---

## Step 5: Results Analysis

### 5.1 View Results in MLflow

1. Go to: http://localhost:8080
2. Experiment: `aneetaa-3x4-comparison`
3. Select runs to compare
4. View charts: accuracy, fact_score, quality_score, latency

### 5.2 CSV Results Format

**Summary CSV columns**:

- `model`: Llama 3.1 8B / Gemma2 9B / Mistral Nemo 12B
- `agent_type`: vanilla / dspy_baseline / dspy_bootstrap / dspy_mipro
- `accuracy`: % correct answers
- `avg_fact_score`: Average 0-10 correctness score
- `avg_quality_score`: Average 1-10 explanation quality
- `avg_latency_ms`: Average response time

**Detailed CSV**: Individual question results with responses and judge reasoning

---

## Expected Results

### Key Findings

**Gemma2 9B** (Best DSPy synergy):

- Vanilla: 62.5% accuracy
- Bootstrap: 65.0% (+2.5%)
- **MIPROv2: 67.5% (+5.0%)**

**Llama 3.1 8B** (RAG-preferred):

- **Vanilla: 72.5% (highest overall)**
- Bootstrap: 52.5% (-20.0%)
- MIPROv2: 67.5% (-5.0%)

**Mistral Nemo 12B** (Negative response):

- Vanilla: 55.0%
- Bootstrap: 47.5% (-7.5%)
- MIPROv2: 37.5% (-17.5%)

### Quality vs Accuracy Trade-off

- Vanilla: Higher quality scores (6.8-7.4/10) but variable accuracy
- Optimized: Higher accuracy but lower quality (3.6-4.3/10)

---

## Troubleshooting

### Ollama Connection Issues

```bash
# Check Ollama status
ollama list

# Restart if needed
ollama serve
```

### MLflow Port Conflicts

```bash
# Use different port
mlflow ui --port 8081
```

### Memory Issues

```bash
# Process models sequentially instead of parallel
# Edit train_all_models_miprov2.py: set one model at a time
```

### Missing Dependencies

```bash
# Reinstall specific packages
pip install dspy-ai==2.6.5 mlflow==3.6.0
```

---

## File Locations Summary

```
ANEETAA/
├── aneeta_v2/Processed Data/
│   ├── Gemini 2.5 Pro Data/       # Source: 7,874 questions
│   ├── dspy_dataset_train.jsonl   # Step 1 output: 6,300 questions
│   ├── dspy_dataset_val.jsonl     # Step 1 output: 1,575 questions
│   └── dspy_dataset_combined.jsonl
│
├── models/                         # Step 2 output: 6 trained models
│   ├── dspy_bootstrap_*.json
│   └── dspy_mipro_*.json
│
├── results/                        # Step 4 output: CSV results
│   ├── 3x4_matrix_detailed_results.csv
│   └── 3x4_matrix_summary.csv
│
├── mlruns/                         # MLflow tracking data
│
├── notebooks/scripts/
│   ├── train_all_models_miprov2.py    # Step 2: Training
│   ├── compare_3x4_matrix.py          # Step 4: Evaluation
│   └── config/llm_judge_config.py     # Judge configuration
│
└── scripts/
    └── prepare_dspy_dataset.py        # Step 1: Data processing
```

---

## Cost Estimation

- **Training**: $0 (Ollama local inference)
- **Evaluation with Groq judge**: $0 (FREE API)
- **Evaluation with OpenAI GPT-4o-mini judge**: ~$0.96 per 40-question run
- **Full 3×4 matrix (12 configs × 40 questions)**: ~$11.52 with OpenAI

---

**Last Updated**: November 17, 2025
**Contact**: See README.md for project team information
