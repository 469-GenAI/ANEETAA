# Screenshot Checklist for Project Presentation

## 📸 Essential Screenshots (Priority Order)

### 🏆 #1: MLflow UI Dashboard (MUST HAVE)
**Location:** Browser at `http://localhost:8080`

**How to capture:**
```bash
# In terminal:
mlflow ui --port 8080
# Open browser: http://localhost:8080
```

**What to show:**
- Left panel: List of experiment runs
- Main panel: Metrics comparison chart (accuracy over runs)
- Highlight: Multiple runs showing Bootstrap vs MIPROv2 performance
- Click on a run to show: Parameters, Metrics, Artifacts

**Why important:** Shows professional ML experiment tracking, reproducibility

**Screenshot tips:**
- Capture full browser window
- Show experiment name: "aneetaa-3x4-matrix-comparison"
- Highlight metrics: accuracy, quality_score, fact_score
- Show date/time stamps proving you ran experiments

---

### 🏆 #2: 3×4 Matrix Results (MUST HAVE)
**Location:** `results/3x4_matrix_summary.csv`

**How to capture:**
```bash
# Open in Excel or VSCode
code results/3x4_matrix_summary.csv
```

**What to show:**
- All 12 rows (3 models × 4 agent types)
- Columns: Model, Agent_Type, Accuracy_%, Avg_Fact_Score, Avg_Quality_Score, Avg_Latency_ms
- Highlight best performer (likely Gemma2 9B + MIPROv2)
- Color code or conditional format for easy visualization

**Why important:** Core evidence of DSPy optimization improvements

**Screenshot tips:**
- Sort by Accuracy_% descending to show best performers at top
- Add Excel conditional formatting (green for high, red for low)
- Circle or highlight the MIPROv2 rows showing improvement
- Show row count (12 configurations)

---

### 🏆 #3: Training Progress Terminal (HIGH PRIORITY)
**Location:** Terminal output during training

**What to show:**
- MIPROv2 training progress with trial scores
- Final message: "✅ All models trained successfully"
- Model save confirmations showing file paths
- Time elapsed and completion messages

**Why important:** Proves you actually trained the models, not just used pre-trained

**Screenshot tips:**
- Capture output showing:
  - "Training Model 1/3: Llama 3.1 8B"
  - "Optimizer: MIPROv2"
  - Trial scores improving
  - "✅ Llama 3.1 8B MIPROv2 training completed!"
  - Final summary showing all 3 models successful

**Alternative:** If training already complete, capture the model files:
```bash
dir models\dspy_mipro_*.json
```

---

### #4: Comparison Completion Output
**Location:** Terminal showing comparison results

**What to show:**
- "🚀 Starting 3×4 matrix comparison..."
- Progress: "[Question 1/40]", "[Question 20/40]", etc.
- Per-question results: "[OK] Fact: 10/10 | Quality: 8/10 | 2500ms"
- Final summary table in terminal
- "✅ COMPARISON COMPLETED SUCCESSFULLY!"

**Why important:** Shows evaluation in action with dual metrics

**Screenshot tips:**
- Capture middle section showing actual evaluations
- Show variety: some [OK], some [X] to show realistic testing
- Capture final summary showing all 12 configurations

---

### #5: Project Structure (VSCode Explorer)
**Location:** VSCode file explorer

**What to show:**
```
ANEETAA/
├── notebooks/scripts/
│   ├── config/               # Judge configuration
│   ├── controllers/          # Easy-to-use runners
│   └── runners/              # Core comparison scripts
├── models/                   # 6 trained models
├── results/                  # CSV outputs
├── mlruns/                   # MLflow tracking
└── aneeta_v2/Processed Data/ # Datasets
```

**Why important:** Shows organized, professional code structure

**Screenshot tips:**
- Expand key folders to show organization
- Highlight the separation: controllers vs runners vs config
- Show model files with timestamps proving recent training
- Show results folder with CSV files

---

### #6: Dataset Files Evidence
**Location:** File explorer at `aneeta_v2/Processed Data/`

**What to show:**
- `dspy_dataset_combined.jsonl` - 7,874 questions (large file size)
- `dspy_dataset_train.jsonl` - Training split
- `dspy_dataset_val.jsonl` - Validation split
- File sizes and date modified

**Why important:** Proves substantial dataset curation work

**Screenshot tips:**
- Windows Explorer with "Details" view showing file sizes
- Highlight the combined dataset size (should be multiple MB)
- Show all three split files together
- Include date modified to show recent work

---

### #7: LLM Judge Configuration Code
**Location:** `notebooks/scripts/config/llm_judge_config.py`

**What to show:**
```python
JUDGE_CONFIG = {
    'provider': 'openai',
    'models': {
        'openai': {...},
        'groq': {...},      # FREE alternative
        'anthropic': {...}
    },
    'costs': {
        # Cost tracking per provider
    }
}

def get_judge_llm(provider=None, model=None):
    """Centralized LLM judge creation"""
    ...
```

**Why important:** Shows technical innovation - centralized, multi-provider judge

**Screenshot tips:**
- Capture JUDGE_CONFIG dictionary showing multiple providers
- Show get_judge_llm() function
- Highlight the 'costs' section showing FREE Groq option

---

### #8: Evaluation Metrics Code
**Location:** `notebooks/scripts/runners/compare_3x4_matrix.py` (lines 340-400)

**What to show:**
```python
def judge_answer_quality(question: str, answer: str) -> dict:
    """Subject-specific evaluation criteria"""
    
    # Step 1: Identify subject
    subject = judge_llm.invoke(subject_identification_prompt)
    
    # Step 2: Subject-specific criteria
    if "Physics" in subject:
        criteria = "equations, units, calculations"
    elif "Chemistry" in subject:
        criteria = "IUPAC, balanced equations"
    elif "Biology" in subject:
        criteria = "terminology, mechanisms"
```

**Why important:** Shows sophisticated evaluation design

**Screenshot tips:**
- Capture the subject identification logic
- Show the different criteria for each subject
- Highlight the dual-metric approach (fact vs quality)

---

## 📊 Optional But Impressive Screenshots

### #9: Model Comparison in MLflow
**Location:** MLflow UI - Compare Runs

**How to access:**
1. Open MLflow UI (http://localhost:8080)
2. Select multiple runs (checkbox on left)
3. Click "Compare" button
4. View parallel coordinates plot or metric comparison table

**What to show:**
- Multiple runs side-by-side
- Accuracy comparison across different optimizers
- Parameter differences (Bootstrap vs MIPROv2)

---

### #10: Detailed Results CSV (Snippet)
**Location:** `results/3x4_matrix_detailed_results.csv` (first 10 rows)

**What to show:**
- Columns: question_id, model, agent_type, response, is_correct, fact_score, quality_score, judge_reasoning
- Show diverse results (some correct, some wrong)
- Highlight the judge_reasoning column showing LLM explanations

---

### #11: Training Configuration Code
**Location:** `notebooks/scripts/controllers/compare_3x4_controller.py`

**What to show:**
```python
COMPARISON_CONFIG = {
    'test_samples': 40,
    'seed': 123,
    'use_validation_set': True,
}

MODELS = [
    {'name': 'llama3.1:8b', ...},
    {'name': 'gemma2:9b', ...},
    {'name': 'mistral-nemo:12b', ...}
]
```

**Why important:** Shows systematic, reproducible configuration

---

### #12: Judge Prompt in Action
**Location:** Terminal showing judge evaluation output

**What to show:**
- Question text
- Model response
- Judge reasoning: "Overall Quality Score: 8/10"
- Judge reasoning: "Brief Reasoning: Clear explanation with proper terminology..."

---

## 🎬 Screenshot Workflow Recommendation

### Before Presentation:
1. ✅ Start MLflow UI and capture dashboard (#1)
2. ✅ Open results CSV in Excel, format, and capture (#2)
3. ✅ Find or re-run training to capture terminal output (#3)
4. ✅ If comparison still running, capture progress (#4)
5. ✅ Open VSCode, show project structure (#5)
6. ✅ File explorer for datasets (#6)

### During Presentation Flow:
1. **Slide 1:** Project overview (no screenshot needed)
2. **Slide 2:** Dataset evidence (#6)
3. **Slide 3:** Code structure (#5)
4. **Slide 4:** Training process (#3)
5. **Slide 5:** MLflow tracking (#1) ⭐
6. **Slide 6:** Evaluation in progress (#4)
7. **Slide 7:** Results summary (#2) ⭐⭐⭐
8. **Slide 8:** Technical innovations (#7, #8)
9. **Slide 9:** Conclusion with key metrics from #2

---

## 💡 Pro Tips

### Making Screenshots Pop:
1. **High resolution:** Use Windows Snipping Tool or ShareX
2. **Annotations:** Add arrows, circles, text boxes in PowerPoint
3. **Crop tightly:** Remove unnecessary UI elements
4. **Contrast:** Use dark mode or light mode consistently
5. **Highlights:** Yellow highlighter in Excel for key numbers

### For MLflow Screenshots:
- Use the "Chart" view for visual comparison
- Show at least 5-10 experiment runs to prove iteration
- Capture both the overview and a detailed run view

### For Results CSV:
- Add Excel formulas showing improvement percentages
- Use conditional formatting (Data Bars or Color Scales)
- Sort by Agent_Type to group Bootstrap vs MIPROv2
- Add a summary row with averages

### For Code Screenshots:
- Use VSCode with a clean theme (Dark+ or Light+)
- Enable minimap for context
- Zoom in slightly for readability (125-150%)
- Use syntax highlighting

---

## 📋 Checklist Summary

Priority Level | Screenshot | File/Location | Status
---|---|---|---
🔴 CRITICAL | MLflow Dashboard | Browser: localhost:8080 | ⬜
🔴 CRITICAL | 3×4 Results Table | results/3x4_matrix_summary.csv | ⬜
🟡 HIGH | Training Progress | Terminal output | ⬜
🟡 HIGH | Comparison Output | Terminal output | ⬜
🟡 HIGH | Project Structure | VSCode explorer | ⬜
🟢 NICE TO HAVE | Dataset Files | File explorer | ⬜
🟢 NICE TO HAVE | Judge Config Code | llm_judge_config.py | ⬜
🟢 NICE TO HAVE | Evaluation Code | compare_3x4_matrix.py | ⬜

---

**Total Screenshots Needed:** 5-8 (3 critical, 2-3 high priority, 0-3 optional)

**Estimated Time:** 15-20 minutes to capture all

**Best Timing:** After 3×4 comparison completes (can capture everything in one session)
