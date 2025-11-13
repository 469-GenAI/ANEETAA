"""
3×4 Matrix Comparison: All Models × All Agent Types (Including MIPROv2)

Compares:
- 3 Models: llama3.1:8b, gemma2:9b, mistral-nemo:12b
- 4 Agent Types: Vanilla ANEETAA, DSPy Baseline, DSPy Bootstrap, DSPy MIPROv2
- Total: 12 configurations

Uses centralized LLM judge configuration for quality evaluation.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List
import mlflow
from dotenv import load_dotenv

# Import centralized judge config
CONFIG_DIR = Path(__file__).parent.parent / "config"
sys.path.insert(0, str(CONFIG_DIR))
from llm_judge_config import get_judge_llm, estimate_judge_cost, JUDGE_CONFIG, GROQ_API_KEYS

# Setup
ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))
load_dotenv()

# Import ANEETA
from aneeta.state.models import State
from aneeta.core import resources
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
import dspy
import importlib

# MLflow setup
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns'))
mlflow.set_experiment('aneetaa-3x4-matrix-comparison')

print("="*70)
print("3×4 MATRIX COMPARISON (INCLUDING MIPROv2)")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS = [
    {
        'name': 'llama3.1:8b',
        'display': 'Llama 3.1 8B',
        'bootstrap_path': 'models/dspy_bootstrap_llama3.1_8b.json',
        'mipro_path': 'models/dspy_mipro_llama3.1_8b.json'
    },
    {
        'name': 'gemma2:9b',
        'display': 'Gemma2 9B',
        'bootstrap_path': 'models/dspy_bootstrap_gemma2_9b.json',
        'mipro_path': 'models/dspy_mipro_gemma2_9b.json'
    },
    {
        'name': 'mistral-nemo:12b',
        'display': 'Mistral Nemo 12B',
        'bootstrap_path': 'models/dspy_bootstrap_mistral_nemo_12b.json',
        'mipro_path': 'models/dspy_mipro_mistral_nemo_12b.json'
    },
]

AGENT_TYPES = [
    'vanilla',
    'dspy_baseline',
    'dspy_bootstrap',     # Bootstrap optimizer
    'dspy_mipro'          # MIPROv2 optimizer (NEW!)
]

AGENT_DISPLAY_NAMES = {
    'vanilla': 'Vanilla ANEETAA',
    'dspy_baseline': 'DSPy Baseline',
    'dspy_bootstrap': 'DSPy Bootstrap',
    'dspy_mipro': 'DSPy MIPROv2'
}

# ============================================================================
# DSPy Components
# ============================================================================

class MCQSolverSignature(dspy.Signature):
    """Solve NEET MCQ questions with explanations."""
    question = dspy.InputField(desc="The MCQ question with options A, B, C, D")
    subject = dspy.InputField(desc="Subject area: Physics, Chemistry, or Biology")
    reasoning = dspy.OutputField(desc="Step-by-step explanation")
    answer = dspy.OutputField(desc="Final answer: A, B, C, or D")

class MCQSolverModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(MCQSolverSignature)
    
    def forward(self, question: str, subject: str = "Biology") -> dspy.Prediction:
        return self.predictor(question=question, subject=subject)

# ============================================================================
# Data Loading
# ============================================================================

def load_test_questions(max_questions: int = 100, use_validation_set: bool = True, start_index: int = 0) -> List[Dict]:
    """Load test questions from validation dataset or combined dataset.
    
    Args:
        max_questions: Maximum number of questions to load
        use_validation_set: Whether to use validation set (default: True)
        start_index: Starting index for deterministic slicing (default: 0)
    
    Returns:
        List of question dictionaries
    
    Note:
        Questions are loaded deterministically in order (no random sampling).
        This ensures reproducibility and allows for batch-based API key rotation.
    """
    if use_validation_set:
        val_file = ROOT / "aneeta_v2" / "Processed Data" / "dspy_dataset_val.jsonl"
        if val_file.exists():
            print(f"✓ Loading validation dataset: {val_file}")
            all_questions = []
            with open(val_file, 'r', encoding='utf-8') as f:
                for line in f:
                    all_questions.append(json.loads(line))
            
            print(f"✓ Loaded {len(all_questions)} total validation questions")
            
            # Deterministic slicing (no random sampling)
            end_index = min(start_index + max_questions, len(all_questions))
            questions = all_questions[start_index:end_index]
            print(f"✓ Selected questions [{start_index}:{end_index}] = {len(questions)} questions")
            
            return questions
    
    # Fallback to combined dataset
    combined_file = ROOT / "aneeta_v2" / "Processed Data" / "dspy_dataset_combined.jsonl"
    print(f"✓ Loading combined dataset: {combined_file}")
    
    all_questions = []
    with open(combined_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_questions.append(json.loads(line))
    
    print(f"✓ Loaded {len(all_questions)} total questions")
    
    # Deterministic slicing (no random sampling)
    end_index = min(start_index + max_questions, len(all_questions))
    questions = all_questions[start_index:end_index]
    print(f"✓ Selected questions [{start_index}:{end_index}] = {len(questions)} questions")
    
    return questions

def format_question(q_data: Dict) -> str:
    """Format question with options."""
    q_text = q_data['question_text']
    options = q_data['options']
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return f"{q_text}\n\nOptions:\n{options_text}"

# ============================================================================
# Agent Functions
# ============================================================================

def vanilla_aneetaa_agent(question: str, model_name: str) -> str:
    """Vanilla ANEETAA using LangChain."""
    try:
        # Update environment
        os.environ["LLM_MODEL"] = model_name
        os.environ["CREATIVE_LLM_MODEL"] = model_name
        
        # Create new LLM
        new_llm = ChatOllama(model=model_name, temperature=0, max_tokens=700)
        resources.llm = new_llm
        
        # Reload agents
        import aneeta.nodes.agents
        importlib.reload(aneeta.nodes.agents)
        from aneeta.nodes.agents import mcq_question_solver_agent
        
        # Build state
        state = State(
            messages=[HumanMessage(content=question)],
            user_explanation_language="English",
            agent_routing="mcq_question_solver",
            teacher_vectordb_routing="biology",
            response_stream=None
        )
        
        # Get result
        result = mcq_question_solver_agent(state)
        
        if isinstance(result, dict) and 'response_stream' in result:
            chunks = list(result['response_stream'])
            return ''.join(chunks).strip()
        return str(result)
        
    except Exception as e:
        import traceback
        return f"[Error: {str(e)}]\n{traceback.format_exc()}"

def configure_dspy_ollama(model_name: str):
    """Configure DSPy to use Ollama model WITHOUT caching."""
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    lm = dspy.LM(
        model=f"ollama_chat/{model_name}",
        api_base=ollama_url,
        max_tokens=500,
        temperature=0.02,
        cache=False  # ⚠️ DISABLE CACHING for fair evaluation
    )
    dspy.settings.configure(lm=lm)

def dspy_baseline_agent(question: str, subject: str, model_name: str) -> str:
    """DSPy baseline (unoptimized)."""
    try:
        configure_dspy_ollama(model_name)
        solver = MCQSolverModule()
        prediction = solver(question=question, subject=subject)
        # Return both answer and reasoning for LLM judge
        answer = getattr(prediction, 'answer', None)
        reasoning = getattr(prediction, 'reasoning', None)
        if answer and reasoning:
            return f"{answer}\n\nExplanation: {reasoning}"
        elif answer:
            return str(answer)
        else:
            return str(prediction)
    except Exception as e:
        return f"[Error: {str(e)}]"

def dspy_optimized_agent(question: str, subject: str, model_name: str, optimized_path: str, optimizer_name: str = "Unknown") -> str:
    """DSPy optimized (Bootstrap or MIPROv2)."""
    try:
        configure_dspy_ollama(model_name)
        solver = MCQSolverModule()
        
        if Path(optimized_path).exists():
            solver.load(optimized_path)
        else:
            print(f"  ⚠️  {optimizer_name} model not found: {optimized_path}, using baseline")
        
        prediction = solver(question=question, subject=subject)
        # Return both answer and reasoning for LLM judge
        answer = getattr(prediction, 'answer', None)
        reasoning = getattr(prediction, 'reasoning', None)
        if answer and reasoning:
            return f"{answer}\n\nExplanation: {reasoning}"
        elif answer:
            return str(answer)
        else:
            return str(prediction)
    except Exception as e:
        return f"[Error: {str(e)}]"

# ============================================================================
# LLM Judge
# ============================================================================

def judge_answer_quality(question: str, answer: str, api_key_index: int = 0) -> dict:
    """Use configured LLM judge to evaluate answer quality with subject-specific criteria.
    
    Args:
        question: The question text
        answer: The model's answer/response
        api_key_index: Index for API key rotation (for Groq rate limit management)
    
    Returns:
        Dictionary with score, reasoning, and detected subject
    """
    try:
        # Get judge from centralized config with API key rotation
        judge_llm = get_judge_llm(api_key_index=api_key_index)
        
        # Step 1: Identify the subject
        subject_identification_prompt = f"""Analyze this NEET exam question and identify which subject it belongs to.

Question: {question}

Respond with ONLY ONE WORD: Physics, Chemistry, or Biology"""
        
        subject_response = judge_llm.invoke(subject_identification_prompt)
        subject = subject_response.content.strip() if hasattr(subject_response, 'content') else "Unknown"
        
        # Step 2: Subject-specific evaluation criteria
        if "Physics" in subject:
            criteria_desc = "Physics: Clarity (proper terminology, variable definitions), Logical Reasoning (step-by-step with equations), Correctness (proper physics principles)"
        elif "Chemistry" in subject:
            criteria_desc = "Chemistry: Clarity (IUPAC names, balanced equations), Logical Reasoning (stoichiometry, calculations), Correctness (chemistry concepts)"
        elif "Biology" in subject:
            criteria_desc = "Biology: Clarity (biological terminology, structure-function), Logical Reasoning (mechanisms, processes), Correctness (biological principles)"
        else:
            criteria_desc = "General: Clarity, Logical Reasoning, Correctness"
        
        # Explanation quality evaluation (answer correctness is evaluated separately)
        evaluation_prompt = f"""You are evaluating ONLY the explanation quality of an AI MCQ solver for NEET exam questions.
Do NOT evaluate whether the final answer is correct - that is assessed separately.

Subject: {subject}
Evaluation Criteria: {criteria_desc}

Question: {question}

MCQ Solver's Response:
{answer}

Rate ONLY the explanation quality on a scale of 1-10:
- Clarity (30%): Use of proper terminology, clear explanations, well-structured response
- Logical Reasoning (40%): Step-by-step approach, showing work, justification for each step
- Correctness of Method (30%): Proper application of scientific principles and methodology

IMPORTANT: Focus on the quality of reasoning and explanation, NOT whether the final answer is correct.
A response can have excellent reasoning but arrive at the wrong answer, or vice versa.

Provide your evaluation in this format:
Overall Quality Score: [number 1-10]
Brief Reasoning: [2-3 sentences explaining the explanation quality]
"""
        
        response = judge_llm.invoke(evaluation_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract score
        import re
        score_match = re.search(r"Overall Quality Score:\s*(\d+)", response_text)
        score = int(score_match.group(1)) if score_match else 5
        score = max(1, min(10, score))
        
        # Extract reasoning
        reasoning_match = re.search(r"Brief Reasoning:\s*(.+?)(?:\n\n|$)", response_text, re.S)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text[:200]
        
        return {"score": score, "reasoning": reasoning, "subject": subject}
        
    except Exception as e:
        return {"score": 5, "reasoning": f"Judge error: {str(e)[:100]}", "subject": "Unknown"}

# ============================================================================
# Answer Validation & Scoring
# ============================================================================

def fact_check_answer(model_answer: str, correct_answer: str, options: dict) -> int:
    """Score answer correctness - binary scoring: correct or wrong
    
    Scoring rubric (purely binary):
    - 10: Correct option identified (explicitly, prominently, or anywhere in response)
    - 0: Wrong option stated OR no correct option identified
    
    Note: Reasoning quality is evaluated separately by the LLM judge (Quality Score)
    """
    import re
    
    # Normalize inputs
    model_answer_lower = model_answer.lower()
    correct_answer_upper = correct_answer.upper()
    correct_answer_lower = correct_answer.lower()
    
    # Check if correct option letter appears anywhere in the answer
    # Match patterns like "B", "(B)", "**B:", "B:", "option B", "answer is B"
    correct_patterns = [
        rf"\b{correct_answer_lower}\b",  # standalone letter
        rf"\({correct_answer_lower}\)",   # (B)
        rf"\*\*{correct_answer_lower}:",  # **B:
        rf"option\s+{correct_answer_lower}\b",  # option B
        rf"answer\s+is\s+{correct_answer_lower}\b",  # answer is B
        rf"correct\s+answer\s+is\s+{correct_answer_lower}\b",  # correct answer is B
    ]
    correct_option_mentioned = any(re.search(pattern, model_answer_lower) for pattern in correct_patterns)
    
    # Check for wrong options explicitly stated as the answer
    # Only flag if clearly stated as THE answer (not just mentioned)
    wrong_options = [opt for opt in options.keys() if opt.upper() != correct_answer_upper]
    wrong_answer_found = False
    
    for wrong_opt in wrong_options:
        wrong_opt_lower = wrong_opt.lower()
        # Check if wrong option is explicitly stated as the answer
        wrong_patterns = [
            rf"(?:correct\s+answer\s+is|answer\s+is|therefore.*option)\s+{wrong_opt_lower}\b",
        ]
        if any(re.search(pattern, model_answer_lower) for pattern in wrong_patterns):
            wrong_answer_found = True
            break
    
    # Binary scoring logic
    if wrong_answer_found:
        # Wrong answer explicitly stated as THE answer
        return 0
    elif correct_option_mentioned:
        # Correct option mentioned
        return 10
    else:
        # No identifiable answer
        return 0

# ============================================================================
# Evaluation
# ============================================================================

def evaluate_configuration(model_config: Dict, agent_type: str, question: Dict, api_key_index: int = 0) -> Dict:
    """Evaluate one configuration on one question.
    
    Args:
        model_config: Model configuration dictionary
        agent_type: Type of agent (vanilla, dspy_baseline, etc.)
        question: Question dictionary
        api_key_index: Index for API key rotation (for Groq rate limit management)
    
    Returns:
        Dictionary with evaluation results
    """
    model_name = model_config['name']
    model_display = model_config['display']
    question_text = format_question(question)
    correct_answer = question['correct_answer']
    subject = question.get('subject', 'Biology')
    
    print(f"  Testing: {model_display} + {AGENT_DISPLAY_NAMES[agent_type]}")
    
    try:
        t0 = time.time()
        
        if agent_type == 'vanilla':
            response = vanilla_aneetaa_agent(question_text, model_name)
        elif agent_type == 'dspy_baseline':
            response = dspy_baseline_agent(question_text, subject, model_name)
        elif agent_type == 'dspy_bootstrap':
            response = dspy_optimized_agent(question_text, subject, model_name, 
                                           model_config['bootstrap_path'], "Bootstrap")
        elif agent_type == 'dspy_mipro':
            response = dspy_optimized_agent(question_text, subject, model_name, 
                                           model_config['mipro_path'], "MIPROv2")
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        latency_ms = (time.time() - t0) * 1000
        
        # Score answer with both metrics
        fact_score = fact_check_answer(response, correct_answer, question.get('options', {}))
        is_correct = fact_score == 10  # Backward compatibility
        
        # Judge quality with API key rotation
        quality_result = judge_answer_quality(question_text, response, api_key_index=api_key_index)
        
        return {
            'question_id': question.get('question_id', ''),
            'subject': subject,
            'model': model_display,
            'agent_type': agent_type,
            'agent_display': AGENT_DISPLAY_NAMES[agent_type],
            'response': response,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'fact_score': fact_score,
            'quality_score': quality_result['score'],
            'judge_reasoning': quality_result['reasoning'],
            'detected_subject': quality_result.get('subject', 'Unknown'),
            'latency_ms': round(latency_ms, 1)
        }
        
    except Exception as e:
        import traceback
        return {
            'question_id': question.get('question_id', ''),
            'subject': subject,
            'model': model_display,
            'agent_type': agent_type,
            'agent_display': AGENT_DISPLAY_NAMES[agent_type],
            'response': f"[Error: {str(e)}]",
            'correct_answer': correct_answer,
            'is_correct': False,
            'fact_score': 0,
            'quality_score': 0,
            'judge_reasoning': 'Error occurred',
            'detected_subject': 'Unknown',
            'latency_ms': 0,
            'error': traceback.format_exc()
        }

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='3×4 Matrix Comparison (with MIPROv2)')
    parser.add_argument('--test-samples', type=int, default=100, help='Number of test questions')
    parser.add_argument('--batch-size', type=int, default=20, 
                        help='Batch size for API key rotation (default: 20)')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Starting index in dataset (default: 0)')
    parser.add_argument('--use-validation-set', action='store_true', default=True,
                        help='Use validation dataset (default: True)')
    args = parser.parse_args()
    
    # Print LLM judge configuration
    print("\n" + "="*70)
    print("LLM JUDGE CONFIGURATION")
    print("="*70)
    print(f"Provider: {JUDGE_CONFIG['provider'].upper()}")
    model_name = JUDGE_CONFIG['models'][JUDGE_CONFIG['provider']].get(
        JUDGE_CONFIG['model'], JUDGE_CONFIG['model'])
    print(f"Model: {model_name}")
    print(f"Temperature: {JUDGE_CONFIG['temperature']}")
    
    # Show API key rotation info
    if JUDGE_CONFIG['provider'] == 'groq':
        print(f"\n🔑 API Key Rotation:")
        print(f"   Available Keys: {len(GROQ_API_KEYS)}")
        print(f"   Batch Size: {args.batch_size} questions per key")
        total_batches = (args.test_samples + args.batch_size - 1) // args.batch_size
        print(f"   Total Batches: {total_batches}")
        print(f"   Rate Limit: 30 req/min per key, 1000 req/day per key")
    
    # Estimate cost (12 configurations = 3 models × 4 agent types)
    cost_est = estimate_judge_cost(args.test_samples * len(MODELS) * len(AGENT_TYPES))
    print(f"\nEstimated cost: ${cost_est['estimated_cost_usd']:.4f}")
    print("="*70)
    
    # Load questions deterministically
    print("\nLoading test questions...")
    questions = load_test_questions(
        max_questions=args.test_samples,
        use_validation_set=args.use_validation_set,
        start_index=args.start_index
    )
    
    print(f"\n📊 Evaluating {len(questions)} questions across 3×4 matrix")
    print(f"Total evaluations: {len(questions) * len(MODELS) * len(AGENT_TYPES)} = {len(questions)} × 3 models × 4 agent types")
    print("\nAgent Types:")
    for agent_type in AGENT_TYPES:
        print(f"  - {AGENT_DISPLAY_NAMES[agent_type]}")
    print("="*70 + "\n")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"3x4_matrix_{len(questions)}q"):
        mlflow.log_param("num_questions", len(questions))
        mlflow.log_param("num_models", len(MODELS))
        mlflow.log_param("num_agent_types", len(AGENT_TYPES))
        mlflow.log_param("start_index", args.start_index)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("deterministic", True)
        
        all_results = []
        
        # Evaluate each configuration with batch-based API key rotation
        for i, question in enumerate(questions, 1):
            # Calculate which API key to use based on batch size
            batch_index = (i - 1) // args.batch_size
            api_key_index = batch_index % len(GROQ_API_KEYS) if GROQ_API_KEYS else 0
            
            if (i - 1) % args.batch_size == 0 and JUDGE_CONFIG['provider'] == 'groq':
                print(f"\n🔑 Switching to API Key #{api_key_index + 1}/{len(GROQ_API_KEYS)} for batch {batch_index + 1}")
            
            print(f"\n[Question {i}/{len(questions)}] {question.get('subject', 'Unknown')} (Batch {batch_index + 1}, Key #{api_key_index + 1})")
            print(f"Question: {question['question_text'][:80]}...")
            print("-"*70)
            
            for model_config in MODELS:
                for agent_type in AGENT_TYPES:
                    result = evaluate_configuration(model_config, agent_type, question, api_key_index=api_key_index)
                    all_results.append(result)
                    
                    status = "[OK]" if result['is_correct'] else "[X]"
                    print(f"    {status} Fact: {result['fact_score']}/10, Quality: {result['quality_score']}/10, {result['latency_ms']:.0f}ms")
        
        # Create summary
        print("\n" + "="*70)
        print("3×4 MATRIX SUMMARY")
        print("="*70 + "\n")
        
        # Summary by configuration
        summary_data = []
        for model_config in MODELS:
            for agent_type in AGENT_TYPES:
                config_results = [r for r in all_results 
                                 if r['model'] == model_config['display'] and r['agent_type'] == agent_type]
                
                if config_results:
                    correct = sum(1 for r in config_results if r['is_correct'])
                    accuracy = (correct / len(config_results)) * 100
                    avg_fact = sum(r['fact_score'] for r in config_results) / len(config_results)
                    avg_quality = sum(r['quality_score'] for r in config_results) / len(config_results)
                    avg_latency = sum(r['latency_ms'] for r in config_results) / len(config_results)
                    
                    summary_data.append({
                        'Model': model_config['display'],
                        'Agent_Type': AGENT_DISPLAY_NAMES[agent_type],
                        'Accuracy_%': round(accuracy, 1),
                        'Correct': correct,
                        'Total': len(config_results),
                        'Avg_Fact_Score': round(avg_fact, 1),
                        'Avg_Quality_Score': round(avg_quality, 1),
                        'Avg_Latency_ms': round(avg_latency, 1)
                    })
        
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        
        # Save results
        df_detailed = pd.DataFrame(all_results)
        
        detailed_output = ROOT / "results" / "3x4_matrix_detailed_results.csv"
        summary_output = ROOT / "results" / "3x4_matrix_summary.csv"

        df_detailed.to_csv(detailed_output, index=False)
        df_summary.to_csv(summary_output, index=False)

        mlflow.log_artifact(str(detailed_output))
        mlflow.log_artifact(str(summary_output))

    # Log metrics
    for row in summary_data:
        prefix = f"{row['Model'].replace(' ', '_')}_{row['Agent_Type'].replace(' ', '_')}"
        mlflow.log_metric(f"{prefix}_accuracy", row['Accuracy_%'])
        mlflow.log_metric(f"{prefix}_fact_score", row['Avg_Fact_Score'])
        mlflow.log_metric(f"{prefix}_quality", row['Avg_Quality_Score'])
        mlflow.log_metric(f"{prefix}_latency", row['Avg_Latency_ms'])

    print(f"\n✓ Detailed results: {detailed_output}")
    print(f"✓ Summary: {summary_output}")
    print(f"✓ MLflow: {mlflow.get_tracking_uri()}")

    # Find best configuration
    best_idx = df_summary['Accuracy_%'].idxmax()
    best_config = df_summary.iloc[best_idx]
    print(f"\n🏆 Best Configuration:")
    print(f"   {best_config['Model']} + {best_config['Agent_Type']}")
    print(f"   Accuracy: {best_config['Accuracy_%']}%")
    print(f"   Quality: {best_config['Avg_Quality_Score']}/10")

    # Compare Bootstrap vs MIPROv2
    print(f"\n📊 Bootstrap vs MIPROv2 Comparison:")
    for model_config in MODELS:
        bootstrap_results = [r for r in summary_data 
                            if r['Model'] == model_config['display'] and 'Bootstrap' in r['Agent_Type']]
        mipro_results = [r for r in summary_data 
                       if r['Model'] == model_config['display'] and 'MIPROv2' in r['Agent_Type']]
        if bootstrap_results and mipro_results:
            bootstrap_acc = bootstrap_results[0]['Accuracy_%']
            mipro_acc = mipro_results[0]['Accuracy_%']
            diff = mipro_acc - bootstrap_acc
            print(f"   {model_config['display']}:")
            print(f"      Bootstrap: {bootstrap_acc}% | MIPROv2: {mipro_acc}% | Δ {diff:+.1f}%")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
