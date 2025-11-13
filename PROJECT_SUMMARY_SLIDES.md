# DSPy Optimization Project Summary
**Improving ANEETAA (AI NEET Tutor) with DSPy Framework**

---

## 🎯 Project Objective

**Goal:** Enhance ANEETAA's Multiple Choice Question (MCQ) solving capabilities using DSPy optimization techniques

**Challenge:** Vanilla ANEETAA (built with LangChain) provided good explanations but lacked consistency and optimization for accuracy on NEET exam questions.

---

## 📊 Methodology Overview

### Phase 1: Data Preparation & Infrastructure
1. **Dataset Creation**
   - Curated 7,874 NEET MCQ questions across Physics, Chemistry, and Biology
   - Split into train/validation sets (train: ~6,300, validation: ~1,575)
   - Rich explanations with step-by-step solutions and scientific principles

2. **MLflow Tracking Setup**
   - Configured **local MLflow server** for experiment tracking
   - Port 8080 for UI access
   - File-based tracking (`./mlruns`) for reproducibility
   - Tracked: accuracy, latency, training parameters, model artifacts

---

## 🔬 Phase 2: DSPy Implementation

### Three Agent Types Created:

#### 1. **Vanilla ANEETAA (Baseline)**
- Original LangChain implementation
- Uses RAG (Retrieval Augmented Generation) with vector database
- Model: Ollama Gemma2 9B (local, free)
- **Purpose:** Establish baseline performance

#### 2. **DSPy Baseline (Unoptimized)**
- Direct DSPy signature translation
- Same prompting structure, no optimization
- Model: OpenAI GPT-4o-mini
- **Purpose:** Validate DSPy framework integration

#### 3. **DSPy Optimized (BootstrapFewShot)**
- **Optimizer:** BootstrapFewShot
- Training: 200 questions, 40 validation questions
- Max demonstrations: 2-4
- Model: OpenAI GPT-4o-mini
- **Purpose:** Improve accuracy through few-shot learning

#### 4. **DSPy MIPROv2 (Advanced Optimization)** ⭐ NEW
- **Optimizer:** MIPROv2 (Multi-prompt Instruction Optimizer)
- Optimizes BOTH instructions AND demonstrations
- Training: 200 questions, 40 validation questions
- Candidates: 5 instruction variants
- Model: 3 Ollama models (Llama 3.1 8B, Gemma2 9B, Mistral Nemo 12B)
- **Purpose:** Maximize performance through instruction optimization

---

## 🏗️ Phase 3: Comprehensive Evaluation System

### Dual-Metric Evaluation Framework

#### **Metric 1: Fact Score (Binary 0/10)**
- **What:** Did the model select the correct answer?
- **Scoring:** 10 (correct) or 0 (wrong/no answer)
- **Method:** Pattern matching on final answer

#### **Metric 2: Quality Score (1-10 Scale)**
- **What:** How good is the explanation?
- **Scoring:** LLM Judge evaluates on 1-10 scale
- **Criteria:**
  - Clarity (30%): Proper terminology, clear explanations
  - Logical Reasoning (40%): Step-by-step approach
  - Correctness of Method (30%): Proper application of concepts
- **Subject-Specific:** Different criteria for Physics, Chemistry, Biology

### LLM Judge Configuration
- **Primary:** OpenAI GPT-4o (high quality, $0.002/call)
- **Alternative:** Groq Llama 3.1 70B (FREE, fast)
- **Centralized:** Easy switching between providers
- **Cost Tracking:** Built-in estimation before running

---

## 📈 Phase 4: Multi-Model Comparison (3×4 Matrix)

### Matrix Dimensions:
- **3 Models:** Llama 3.1 8B, Gemma2 9B, Mistral Nemo 12B
- **4 Agent Types:** Vanilla, Baseline, Bootstrap, MIPROv2
- **Total:** 12 configurations tested

### Evaluation Scale:
- **40 validation questions** (held-out from training)
- **480 total evaluations** (40 × 12)
- **Random seed:** 123 for reproducibility

---

## 🎯 Key Results

### Performance Summary by Model

#### **Llama 3.1 8B Results:**
| Agent Type | Accuracy | Fact Score | Quality Score | Latency (ms) | vs Vanilla |
|------------|----------|------------|---------------|--------------|------------|
| Vanilla ANEETAA | **72.5%** | 7.2/10 | **6.8/10** | 6343 | Baseline |
| DSPy Baseline | 42.5% | 4.2/10 | 3.8/10 | 8310 | -30.0% ❌ |
| DSPy Bootstrap | 52.5% | 5.2/10 | 3.6/10 | 5184 | -20.0% |
| **DSPy MIPROv2** | **67.5%** | **6.8/10** | 4.0/10 | 6512 | **-5.0%** |

#### **Gemma2 9B Results:** ⭐ BEST OVERALL
| Agent Type | Accuracy | Fact Score | Quality Score | Latency (ms) | vs Vanilla |
|------------|----------|------------|---------------|--------------|------------|
| Vanilla ANEETAA | 62.5% | 6.2/10 | **7.4/10** | 6918 | Baseline |
| DSPy Baseline | 55.0% | 5.5/10 | 4.0/10 | 7369 | -7.5% |
| DSPy Bootstrap | **65.0%** | 6.5/10 | 3.6/10 | 4879 | **+2.5%** ✅ |
| **DSPy MIPROv2** | **67.5%** | **6.8/10** | 3.6/10 | 7407 | **+5.0%** ✅ |

#### **Mistral Nemo 12B Results:**
| Agent Type | Accuracy | Fact Score | Quality Score | Latency (ms) | vs Vanilla |
|------------|----------|------------|---------------|--------------|------------|
| **Vanilla ANEETAA** | **55.0%** | 5.5/10 | **7.0/10** | 8744 | Baseline |
| **DSPy Baseline** | **55.0%** | 5.5/10 | 4.3/10 | 7243 | **0.0%** |
| DSPy Bootstrap | 47.5% | 4.8/10 | 3.4/10 | 4990 | -7.5% ❌ |
| DSPy MIPROv2 | 37.5% | 3.8/10 | 3.2/10 | 8668 | -17.5% ❌ |

### 🏆 Top Performers:
1. **🥇 Llama 3.1 8B + Vanilla ANEETAA:** 72.5% accuracy (highest overall)
2. **🥈 Gemma2 9B + MIPROv2:** 67.5% accuracy (+5.0% vs vanilla)
3. **🥉 Llama 3.1 8B + MIPROv2:** 67.5% accuracy (-5.0% vs vanilla)

### Key Findings:
1. ✅ **Vanilla ANEETAA** performed surprisingly well (62.5-72.5% accuracy)
2. ✅ **Llama 3.1 8B** showed best raw performance with vanilla approach
3. ✅ **Gemma2 9B** responded positively to DSPy optimization (+2.5% Bootstrap, +5.0% MIPROv2)
4. ❌ **Mistral Nemo 12B** struggled with DSPy optimization (negative improvements)
5. ⚖️ **Quality vs Accuracy Trade-off:** Vanilla maintained highest explanation quality (6.8-7.4) while optimized models prioritized correctness
6. 🚀 **Bootstrap Speedup:** DSPy Bootstrap consistently fastest (4879-5184ms avg)

---

## 💡 Technical Innovations

### 1. **Centralized LLM Judge System**
```python
# Easy switching between providers
from llm_judge_config import get_judge_llm

judge = get_judge_llm()  # Uses configured provider (OpenAI/Groq/Anthropic/Ollama)
```

### 2. **Dual-Metric Evaluation**
- Separates answer correctness from explanation quality
- Enables nuanced analysis: models can be correct but poorly explained (or vice versa)

### 3. **Subject-Specific Criteria**
- Physics: Equations, units, step-by-step calculations
- Chemistry: IUPAC names, balanced equations, stoichiometry
- Biology: Terminology, structure-function relationships, mechanisms

### 4. **Reproducible Training**
- Split datasets with fixed seeds
- Version-controlled model artifacts
- MLflow tracking for all experiments

### 5. **Cost Optimization**
- FREE training: Ollama (local inference)
- FREE evaluation: Groq API (alternative to OpenAI)
- Paid option: OpenAI GPT-4o-mini ($0.96 for 40-question evaluation)

---

## 🛠️ Tools & Technologies

### Core Framework:
- **DSPy:** Declarative prompting framework with optimization
- **LangChain:** Baseline implementation framework
- **MLflow:** Experiment tracking and model registry
- **Ollama:** Local LLM inference (FREE)

### Models Used:
- **Training:** Llama 3.1 8B, Gemma2 9B, Mistral Nemo 12B (Ollama)
- **Evaluation:** OpenAI GPT-4o-mini (DSPy agents), GPT-4o (LLM judge)
- **Alternative Judge:** Groq Llama 3.1 70B (FREE)

### Infrastructure:
- **MLflow Server:** Local file-based tracking on port 8080
- **Vector DB:** Chroma (for vanilla ANEETAA RAG)
- **Dataset:** 7,874 NEET questions (JSONL format)

---

## 📂 Project Structure

```
ANEETAA/
├── notebooks/scripts/
│   ├── config/
│   │   ├── llm_judge_config.py          # Centralized judge
│   │   └── judge_config_controller.py    # Judge CLI tool
│   ├── controllers/
│   │   ├── compare_3x4_controller.py     # 3×4 matrix runner
│   │   └── train_controller.py           # Easy training
│   └── runners/
│       ├── train_all_models_miprov2.py   # MIPROv2 training
│       ├── compare_3x4_matrix.py         # Full comparison
│       └── train_mcq_solver.py           # Core training logic
├── models/
│   ├── dspy_bootstrap_gemma2_9b.json     # Bootstrap optimized
│   ├── dspy_mipro_gemma2_9b.json         # MIPROv2 optimized
│   └── ... (6 models total)
├── results/
│   ├── 3x4_matrix_detailed_results.csv   # Full evaluation data
│   └── 3x4_matrix_summary.csv            # Aggregated results
└── mlruns/                                # MLflow tracking data
```

---

## 🎓 Learning Outcomes

### Technical Skills:
1. **Prompt Optimization:** BootstrapFewShot and MIPROv2 optimizers
2. **Evaluation Design:** Multi-metric evaluation with LLM judges
3. **Experiment Tracking:** MLflow for reproducible ML workflows
4. **Model Comparison:** Systematic benchmarking across configurations

### Domain Knowledge:
1. **Few-Shot Learning:** Selecting effective demonstrations
2. **Instruction Optimization:** Crafting optimal prompts programmatically
3. **Trade-offs:** Accuracy vs explanation quality vs latency
4. **Subject-Specific AI:** Adapting evaluation to domain requirements
5. **Model Selection:** Understanding that not all models respond equally to optimization

### Critical Insights:
1. **Optimization isn't universal:** Mistral Nemo 12B showed negative results with DSPy optimization
2. **Vanilla baselines matter:** RAG-based vanilla ANEETAA achieved strong performance (72.5% on Llama 3.1 8B)
3. **Model-optimizer matching:** Gemma2 9B showed best synergy with MIPROv2 (+5% improvement)
4. **Quality-accuracy trade-off:** Higher accuracy often came at cost of explanation detail (vanilla 6.8-7.4 quality vs optimized 3.2-4.3)
5. **Speed gains:** Bootstrap optimization reduced latency by ~20-30%

### Engineering Practices:
1. **Reproducibility:** Seeds, version control, dataset splits
2. **Cost Management:** FREE local inference + budget-conscious evaluation
3. **Modular Design:** Controllers, runners, config separation
4. **Documentation:** Comprehensive guides and README files
5. **Negative Results:** Documenting what doesn't work (Mistral optimization) is valuable

---

## 📸 Visual Evidence Checklist

### Screenshots to Include:

#### 1. **MLflow UI Dashboard** ⭐
- **File:** Browser at `http://localhost:8080`
- **Shows:** Experiment runs, metrics comparison, model artifacts
- **Highlight:** Multiple runs with accuracy/quality score trends

#### 2. **Training Output** ⭐
- **File:** Terminal running `train_all_models_miprov2.py`
- **Shows:** MIPROv2 training progress, trial scores, final accuracy
- **Highlight:** "✅ All models trained successfully"

#### 3. **3×4 Matrix Results Summary** ⭐⭐⭐
- **File:** `results/3x4_matrix_summary.csv` (opened in Excel/VSCode)
- **Shows:** 12 configurations with accuracy, fact score, quality score
- **Highlight:** Performance comparison across all combinations

#### 4. **Detailed Evaluation Output**
- **File:** Terminal showing comparison completion
- **Shows:** Question-by-question evaluation with fact/quality scores
- **Highlight:** "✅ COMPARISON COMPLETED SUCCESSFULLY!"

#### 5. **Code Structure**
- **File:** VSCode explorer showing `notebooks/scripts/` folder structure
- **Shows:** Organized controllers, runners, config folders
- **Highlight:** Clean separation of concerns

#### 6. **LLM Judge Configuration**
- **File:** `notebooks/scripts/config/llm_judge_config.py`
- **Shows:** Multi-provider support (OpenAI, Groq, Anthropic, Ollama)
- **Highlight:** Cost estimation function

#### 7. **Evaluation Metrics Code**
- **File:** `notebooks/scripts/runners/compare_3x4_matrix.py` (lines 266-310)
- **Shows:** Judge evaluation prompt with subject-specific criteria
- **Highlight:** Explicit explanation-quality focus

#### 8. **Dataset Files**
- **File:** File explorer showing `aneeta_v2/Processed Data/`
- **Shows:** `dspy_dataset_train.jsonl`, `dspy_dataset_val.jsonl`, `dspy_dataset_combined.jsonl`
- **Highlight:** 7,874 total questions

---

## 🚀 Deployment & Usage

### Quick Start Commands:

```bash
# 1. View MLflow results
mlflow ui --port 8080
# Open: http://localhost:8080

# 2. Train MIPROv2 models (all 3)
python notebooks/scripts/runners/train_all_models_miprov2.py

# 3. Run 3×4 matrix comparison
python notebooks/scripts/controllers/compare_3x4_controller.py

# 4. Switch LLM judge (FREE Groq)
python notebooks/scripts/config/judge_config_controller.py --preset groq-default

# 5. View results
# Excel: results/3x4_matrix_summary.csv
# Detailed: results/3x4_matrix_detailed_results.csv
```

---

## 💰 Cost Analysis

### Training Phase (FREE):
- **Platform:** Ollama (local inference)
- **Models:** 3 models × 2 optimizers = 6 trained models
- **Time:** ~7-8 hours total (Bootstrap + MIPROv2)
- **Cost:** $0 (100% local)

### Evaluation Phase:
- **Option 1 (Paid):** OpenAI GPT-4o judge
  - 40 questions × 12 configs = 480 evaluations
  - Cost: ~$0.96 per run
  - Quality: High
  
- **Option 2 (FREE):** Groq Llama 3.1 70B judge
  - Same evaluations
  - Cost: $0
  - Quality: Good (slightly lower than GPT-4o)

### Total Project Cost:
- **Training:** $0 (Ollama)
- **Evaluation:** $0-5 (depending on judge choice)
- **Infrastructure:** $0 (local MLflow)

---

## 🎯 Impact & Conclusion

### Quantitative Impact:
- ✅ **Best Model: Llama 3.1 8B + Vanilla ANEETAA** achieved **72.5% accuracy**
- ✅ **Best Optimization: Gemma2 9B + MIPROv2** improved vanilla by **+5.0%**
- ✅ **Fastest: DSPy Bootstrap** averaged **4879-5184ms** latency (~20% faster)
- ✅ **12 configurations tested** across 40 validation questions (480 total evaluations)
- ✅ **100% FREE training** using local Ollama models
- ✅ **6 optimized models** ready for deployment

### Qualitative Impact:
- ✅ **Systematic evaluation** framework for educational AI
- ✅ **Subject-aware** assessment aligned with NEET exam standards
- ✅ **Reproducible** experiments with MLflow tracking
- ✅ **Modular** architecture for easy iteration
- 📊 **Honest reporting** including negative results (Mistral Nemo optimization failures)

### Key Insights:
1. **RAG-based vanilla approach is competitive:** The original ANEETAA with retrieval augmentation performed exceptionally well, suggesting context matters more than optimization for some tasks
2. **Model-specific optimization:** Gemma2 9B benefited from DSPy optimization (+5%), while Mistral Nemo 12B degraded significantly (-17.5% with MIPROv2)
3. **Explanation quality trade-off:** Vanilla models produced higher quality explanations (6.8-7.4/10) vs optimized models (3.2-4.3/10), indicating optimization focused on correctness over educational value
4. **Speed vs accuracy:** Bootstrap optimization offers best speed-accuracy balance for production deployment

### Key Takeaway:
**DSPy optimization showed mixed results: while Gemma2 9B improved +5% with MIPROv2, the vanilla RAG-based ANEETAA on Llama 3.1 8B achieved the highest overall accuracy (72.5%). This suggests that for educational AI, retrieval-augmented context may be more valuable than prompt optimization alone. The systematic evaluation via MLflow revealed critical model-optimizer compatibility issues and quality-accuracy trade-offs essential for informed deployment decisions.**

---

## 📚 References & Resources

### Documentation Created:
- `CHECKPOINT_README.md` - Complete project checkpoint
- `DSPY_AB_TESTING_METHODOLOGY.md` - A/B testing approach
- `notebooks/docs/MIPROV2_INTEGRATION.md` - MIPROv2 implementation
- `notebooks/docs/LLM_JUDGE_GUIDE.md` - Judge configuration guide
- `notebooks/docs/EVALUATION_METRICS_UPDATE.md` - Metrics standardization

### Key Technologies:
- DSPy: https://github.com/stanfordnlp/dspy
- MLflow: https://mlflow.org
- Ollama: https://ollama.ai
- Groq: https://groq.com

---

## 🔮 Future Work

### Based on Evaluation Results:

1. **Hybrid Approach:** Combine vanilla ANEETAA's RAG context with DSPy optimization
   - Leverage high vanilla accuracy (72.5%) + optimization structure
   - Potential to achieve 75-80% accuracy with better explanations

2. **Model-Specific Optimization:** 
   - Focus on Llama 3.1 8B and Gemma2 9B (proven responders)
   - Investigate why Mistral Nemo 12B degraded with optimization
   - Test alternative optimizers (COPRO, SignatureOptimizer)

3. **Explanation Quality Enhancement:**
   - Optimize for BOTH accuracy AND explanation quality
   - Multi-objective optimization (Pareto frontier)
   - Incorporate RAG context into DSPy prompts

4. **Extended Training:** 
   - Use full 6,300 training questions (currently using 200)
   - Larger validation set (currently 40 questions)
   - Cross-validation for more robust evaluation

5. **Ensemble Methods:** 
   - Combine Llama 3.1 8B (vanilla) + Gemma2 9B (MIPROv2)
   - Voting mechanism for final answer selection
   - Quality-weighted averaging

6. **Fine-tuning:** 
   - Custom fine-tuned models on NEET data
   - Compare fine-tuning vs prompt optimization costs/benefits

7. **Real-time Optimization:** 
   - Online learning from student interactions
   - A/B testing in production environment

8. **Multi-lingual:** 
   - Extend to Hindi/regional language explanations
   - Evaluate cross-lingual transfer learning

9. **Deployment:** 
   - Web API for integration with student platforms
   - Best model: Llama 3.1 8B + Vanilla ANEETAA (72.5% accuracy, 6.8 quality)

10. **Cost Analysis:**
    - Evaluate Groq judge vs OpenAI judge quality difference
    - Production cost estimation with current best model

---

**Date:** November 13, 2025  
**Status:** ✅ Complete - 3×4 Matrix Evaluation Finished  
**Final Results:** Llama 3.1 8B + Vanilla ANEETAA = 72.5% accuracy (best performer)  
**Insights:** DSPy optimization shows model-specific results; vanilla RAG remains competitive
