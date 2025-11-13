# Evaluation Metrics Standardization

**Updated:** November 12, 2025

## Overview

All evaluation scripts now use **consistent evaluation metrics** across vanilla ANEETAA and DSPy comparison files. This ensures fair, comparable results across all agent types.

## Standardized Metrics

### 1. Fact-Check Score (0/10 Binary)
**Function:** `fact_check_answer(model_answer, correct_answer, options)`

**Scoring:**
- **10 points:** Correct option identified (anywhere in response)
- **0 points:** Wrong option stated as answer OR no identifiable answer

**Pattern Matching:**
- Detects: `B`, `(B)`, `**B:`, `option B`, `answer is B`, `correct answer is B`
- Case-insensitive
- Checks for wrong answers explicitly stated

### 2. Quality Score (1-10 Scale)
**Function:** `judge_answer_quality(question, answer)`

**Uses centralized LLM judge with:**
- Subject identification (Physics/Chemistry/Biology)
- Subject-specific evaluation criteria
- Weighted scoring:
  - **Clarity (30%):** Proper terminology, clear explanations
  - **Logical Reasoning (40%):** Step-by-step approach, showing work
  - **Correctness (30%):** Proper application of concepts

**Returns:**
- `score`: 1-10 quality score
- `reasoning`: Judge's explanation
- `subject`: Detected subject

## Files Updated

### ✅ scripts/mcq_eval.py (Vanilla ANEETAA)
**Already had both metrics** - used as reference

**Metrics:**
- ✅ `fact_check_answer()` - Detailed 0/10 scoring
- ✅ `judge_answer_quality()` - Subject-specific LLM judge with full rubrics

### ✅ notebooks/scripts/compare_3x3_matrix.py (DSPy 3×3 Comparison)
**Updated from simplified to detailed metrics**

**Before:**
- `validate_answer()` - Simple True/False
- `judge_answer_quality()` - Generic prompt without subject specificity

**After:**
- ✅ `fact_check_answer()` - Full 0/10 scoring (NEW)
- ✅ `judge_answer_quality()` - Subject-specific with criteria (ENHANCED)
- ✅ Added `fact_score` to results (NEW)
- ✅ Added `detected_subject` to results (NEW)
- Kept `validate_answer()` for backward compatibility

### ✅ notebooks/scripts/compare_three_agents.py (Three-Way Comparison)
**Updated from basic validation to full metrics**

**Before:**
- `validate_answer()` - True/False only
- ❌ No LLM judge at all

**After:**
- ✅ `fact_check_answer()` - Full 0/10 scoring (NEW)
- ✅ `judge_answer_quality()` - Subject-specific LLM judge (NEW)
- ✅ Centralized judge integration (NEW)
- ✅ Added `fact_score` to results (NEW)
- ✅ Added `quality_score` to results (NEW)
- ✅ Added `judge_reasoning` to results (NEW)
- ✅ Added `detected_subject` to results (NEW)
- ✅ Judge config display before evaluation (NEW)
- Kept `validate_answer()` for backward compatibility

## New Result Schema

All comparison files now return:

```python
{
    'question_id': str,
    'subject': str,
    'agent_type': str,
    'response': str,
    'correct_answer': str,
    'is_correct': bool,              # Derived from fact_score == 10
    'fact_score': int,               # NEW: 0 or 10
    'quality_score': int,            # NEW: 1-10
    'judge_reasoning': str,          # NEW: Judge explanation
    'detected_subject': str,         # NEW: Subject identified by judge
    'latency_ms': float
}
```

## Output Changes

### Console Output (per question)

**Before:**
```
✓ CORRECT - 2500ms
```

**After:**
```
✓ CORRECT | Fact: 10/10 | Quality: 8/10 | 2500ms
```

### Summary Output

**Before:**
```
Vanilla ANEETAA              | Accuracy: 80.0% | Latency: 2500.0ms
```

**After:**
```
Vanilla ANEETAA              | Acc: 80.0% | Fact: 8.5/10 | Quality: 7.2/10 | 2500.0ms
```

### CSV Output

**New columns added:**
- `fact_score` - Binary correctness (0 or 10)
- `quality_score` - LLM judge quality (1-10)
- `judge_reasoning` - Judge's explanation
- `detected_subject` - Subject identified by judge

### MLflow Metrics

**New metrics logged per agent:**
- `{agent_type}_avg_fact_score` - Average fact-check score
- `{agent_type}_avg_quality_score` - Average quality score
- Existing: `{agent_type}_accuracy`, `{agent_type}_correct`, `{agent_type}_avg_latency_ms`

## Benefits

### 1. Fair Comparison
- Same scoring rubric for vanilla ANEETAA and all DSPy variants
- No bias from different evaluation methods

### 2. Richer Insights
- **Fact Score:** Did the model get the right answer?
- **Quality Score:** Was the explanation good?
- Can identify models that are correct but poorly explained (or vice versa)

### 3. Subject-Aware Evaluation
- Physics questions judged by physics criteria (equations, units)
- Chemistry questions judged by chemistry criteria (balanced equations, stoichiometry)
- Biology questions judged by biology criteria (terminology, mechanisms)

### 4. Debugging Power
- `judge_reasoning` explains WHY a score was given
- `detected_subject` shows if question was properly categorized
- Helps identify systematic issues

## Example Use Cases

### Case 1: High Fact Score, Low Quality Score
```
fact_score: 10/10
quality_score: 4/10
judge_reasoning: "Correct answer B but no explanation provided"
```
**Diagnosis:** Model guesses correctly but doesn't show work

### Case 2: Low Fact Score, High Quality Score
```
fact_score: 0/10
quality_score: 8/10
judge_reasoning: "Excellent step-by-step reasoning but made calculation error in Step 3"
```
**Diagnosis:** Model has good approach but execution issues

### Case 3: Subject Mismatch
```
subject: "Physics"
detected_subject: "Biology"
```
**Diagnosis:** Question metadata may be incorrect or question is interdisciplinary

## Judge Configuration

All comparison files now display judge config before running:

```
======================================================================
LLM JUDGE CONFIGURATION
======================================================================
Provider: groq
Model: llama-3.1-70b-versatile
Temperature: 0
Estimated Cost: $0.0000
======================================================================
```

## Backward Compatibility

- ✅ `is_correct` field still present (derived from `fact_score == 10`)
- ✅ `validate_answer()` function kept for compatibility
- ✅ Existing CSV columns unchanged
- ✅ New columns added to end of CSV

## Related Documentation

- [JUDGE_INTEGRATION_STATUS.md](JUDGE_INTEGRATION_STATUS.md) - LLM judge setup and switching
- [LLM_JUDGE_GUIDE.md](LLM_JUDGE_GUIDE.md) - Judge configuration guide
- [SPLIT_DATASET_UPDATE.md](SPLIT_DATASET_UPDATE.md) - Dataset configuration

## Testing Recommendations

1. **Run a small comparison** (5 questions) to verify metrics:
   ```powershell
   python notebooks/scripts/compare_three_agents.py --test-samples 5
   ```

2. **Check CSV output** has new columns:
   - `fact_score`
   - `quality_score`
   - `judge_reasoning`
   - `detected_subject`

3. **Verify judge is working** by checking reasoning makes sense

4. **Compare with vanilla results** from `scripts/mcq_eval.py` to ensure consistency

## Next Steps

1. ✅ Metrics standardized across all files
2. ✅ Centralized judge integration complete
3. ⏭️ Run baseline comparisons with new metrics
4. ⏭️ Document insights from dual-metric evaluation
5. ⏭️ Consider adding subject-specific accuracy breakdown
