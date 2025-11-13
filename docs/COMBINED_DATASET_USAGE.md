# Using the Combined Dataset (dspy_dataset_combined.jsonl)

## 📊 Dataset Overview

**File**: `aneeta_v2/Processed Data/dspy_dataset_combined.jsonl`  
**Total Questions**: **7,874 NEET MCQs**  
**Format**: JSONL (JSON Lines - one question per line)

### Data Structure

Each question contains:
- ✅ **Question metadata**: ID, text, type, subject, topic, difficulty
- ✅ **Answer options**: A, B, C, D with full text
- ✅ **Correct answer**: The right option letter
- ✅ **Rich explanations**: 
  - Summary and key concepts
  - Step-by-step solutions with formulas
  - Scientific principles
  - Reasoning for correct/incorrect options
  - Common mistakes to avoid

---

## 🚀 How to Use with Your DSPy Training

### Method 1: Command-Line Flag (Recommended)

I've updated `notebooks/scripts/train_mcq_solver.py` to support the combined dataset via a simple flag:

```bash
# Use the combined dataset (7,874 questions)
python notebooks/scripts/train_mcq_solver.py --use-combined --questions 500

# Compare: original Gemini data
python notebooks/scripts/train_mcq_solver.py --questions 100
```

**Key Arguments:**
- `--use-combined` - Load from `dspy_dataset_combined.jsonl` instead of Gemini data
- `--questions 500` - Sample 500 questions (can go up to 7,874!)
- `--method bootstrap` - Use Bootstrap optimizer (default)
- `--max-demos 4` - Number of few-shot examples (default)

### Method 2: Load Directly in Python

```python
import json
from pathlib import Path

# Load all questions
combined_file = Path("aneeta_v2/Processed Data/dspy_dataset_combined.jsonl")
questions = []

with open(combined_file, 'r', encoding='utf-8') as f:
    for line in f:
        questions.append(json.loads(line))

print(f"Loaded {len(questions)} questions")

# Example: Filter by subject
physics_questions = [q for q in questions if q['subject'] == 'Physics']
print(f"Physics questions: {len(physics_questions)}")

# Example: Filter by difficulty
easy_questions = [q for q in questions if q['difficulty'] == 'Easy']
print(f"Easy questions: {len(easy_questions)}")
```

---

## 🎯 Recommended Training Workflows

### Workflow 1: Quick Experiment (100-200 questions)
```bash
# Fast iteration for testing
python notebooks/scripts/train_mcq_solver.py \
  --use-combined \
  --questions 150 \
  --method bootstrap \
  --max-demos 4
```

### Workflow 2: Medium Training (500-1000 questions)
```bash
# Better model performance
python notebooks/scripts/train_mcq_solver.py \
  --use-combined \
  --questions 750 \
  --method bootstrap \
  --max-demos 6
```

### Workflow 3: Full Dataset Training (7,874 questions)
```bash
# Maximum performance (may take longer)
python notebooks/scripts/train_mcq_solver.py \
  --use-combined \
  --questions 7874 \
  --method mipro \
  --candidates 10
```

---

## 💡 Advantages Over Original Gemini Data

| Feature | Gemini Data | Combined Dataset |
|---------|-------------|------------------|
| **Size** | ~1,200 questions | **7,874 questions** |
| **Structure** | Variable | **Consistent JSONL** |
| **Explanations** | Basic | **Rich (step-by-step, formulas, reasoning)** |
| **Easy Loading** | Multiple JSON files | **Single JSONL file** |
| **DSPy Ready** | Needs processing | **Pre-formatted** |

---

## 🔍 Data Quality Examples

### Example Question Structure:
```json
{
  "question_id": "Question_Paper_1_p3_q5",
  "question_text": "A biconvex lens has radii of curvature, 20 cm each...",
  "question_type": "single_correct",
  "options": {
    "A": "+2 D",
    "B": "+20 D",
    "C": "+5 D",
    "D": "Infinity"
  },
  "correct_answer": "C",
  "explanation": {
    "summary": "This is a direct application of the Lens Maker's formula...",
    "key_concepts": ["Lens Maker's Formula", "Power of a lens", ...],
    "step_by_step": [
      {
        "step_number": 1,
        "description": "Identify the given values: Refractive index (μ) = 1.5...",
        "formula": "\\(R_1 = +20 \\text{ cm}, R_2 = -20 \\text{ cm}, \\mu = 1.5\\)"
      },
      ...
    ],
    "correct_option_reasoning": "Using the Lens Maker's formula...",
    "common_mistakes": ["Using the wrong sign for R_2...", ...]
  },
  "subject": "Physics",
  "topic": "Ray Optics and Optical Instruments",
  "difficulty": "Easy"
}
```

---

## 📈 Expected Performance Improvements

Based on DSPy principles, training with more data should yield:

| Training Size | Expected Accuracy | Training Time |
|---------------|-------------------|---------------|
| 100 questions | 70-80% | 2-5 min |
| 500 questions | 80-85% | 10-15 min |
| 1000+ questions | 85-90%+ | 20-30 min |

*Note: Actual results depend on LLM, optimizer, and question difficulty*

---

## 🛠️ Troubleshooting

### Issue: File not found
```bash
# Check if file exists
ls "aneeta_v2/Processed Data/dspy_dataset_combined.jsonl"

# If missing, it's in the wrong location - should be created by scripts/prepare_dspy_dataset.py
```

### Issue: Memory errors with full dataset
```python
# Solution: Use streaming or chunking
import json

def stream_questions(file_path, chunk_size=1000):
    chunk = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            chunk.append(json.loads(line))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk
```

---

## 🎓 Next Steps

1. **Start small**: Test with 200 questions to validate the pipeline
2. **Compare results**: Run A/B test between original and combined dataset
3. **Scale up**: Gradually increase to 500, 1000, then full 7,874 questions
4. **Track metrics**: Use MLflow to compare accuracy across different dataset sizes

**Ready to train?** Run this command to get started:
```bash
python notebooks/scripts/train_mcq_solver.py --use-combined --questions 200
```
