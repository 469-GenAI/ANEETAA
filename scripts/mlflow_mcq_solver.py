"""
MLflow MCQ Solver Evaluation - Combined Best of Both Worlds
Features:
- Multi-model comparison (llama3.1:8b, gemma2:9b, mistral-nemo:12b)
- Centralized LLM judge configuration (supports OpenAI, Groq, Anthropic, Ollama)
- MLflow experiment tracking
- Seed-based reproducible sampling
- Advanced answer validation with regex patterns
"""

import os
import sys
import json
import time
import random
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List
import mlflow
import importlib

# Setup paths
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

# Add notebooks/scripts to path for centralized judge config
NOTEBOOKS_SCRIPTS = ROOT / "notebooks" / "scripts"
sys.path.insert(0, str(NOTEBOOKS_SCRIPTS))
from llm_judge_config import get_judge_llm, estimate_judge_cost, JUDGE_CONFIG

# Configure base environment
os.environ["EMBEDDING_MODEL"] = "nomic-embed-text"
os.environ["VECTORDB_BASE_PATH"] = str(ROOT / "src" / "aneeta" / "vectordb")

# Load .env file for OpenAI API key and other configs
from dotenv import load_dotenv
load_dotenv()

# Setup MLflow
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns'))
mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME', 'aneetaa-mcq-model-comparison'))

# Import ANEETA components
from aneeta.state.models import State
from aneeta.core import resources
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

print("="*70)
print("MLFLOW MCQ SOLVER EVALUATION - MULTI-MODEL COMPARISON")
print("="*70)
print(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"✓ MLflow experiment: {mlflow.get_experiment_by_name(os.getenv('MLFLOW_EXPERIMENT_NAME', 'aneetaa-mcq-model-comparison')).name if mlflow.get_experiment_by_name(os.getenv('MLFLOW_EXPERIMENT_NAME', 'aneetaa-mcq-model-comparison')) else 'Will be created'}")
print("="*70 + "\n")


def load_test_questions(max_questions: int = 10, seed: int = 42, filter_visual: bool = True) -> List[Dict]:
    """Load test questions with optional filtering and random sampling."""
    data_dir = ROOT / "aneeta_v2" / "Processed Data" / "Gemini 2.5 Pro Data"
    
    all_questions = []
    json_files = sorted(data_dir.glob("*.json"))
    
    # Load all questions
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                file_questions = json.load(f)
                all_questions.extend(file_questions)
        except Exception as e:
            print(f"⚠ Warning: Could not load {json_file.name}: {e}")
    
    print(f"✓ Loaded {len(all_questions)} total questions")
    
    # Filter non-visual questions if requested
    if filter_visual:
        all_questions = [q for q in all_questions if not q.get('metadata', {}).get('requires_visual', False)]
        print(f"✓ Filtered to {len(all_questions)} non-visual questions")
    
    # Random sampling with seed for reproducibility
    if len(all_questions) > max_questions:
        random.seed(seed)
        all_questions = random.sample(all_questions, max_questions)
        print(f"✓ Randomly sampled {max_questions} questions (seed={seed})")
    
    # Convert to simplified format
    test_questions = []
    for item in all_questions:
        if 'question_text' in item and 'correct_answer' in item:
            test_questions.append({
                'id': item.get('question_id', ''),
                'question': item['question_text'],
                'options': item.get('options', {}),
                'correct': item['correct_answer'],
                'subject': item.get('metadata', {}).get('subject', 'unknown')
            })
    
    return test_questions


def format_question(q_data: Dict) -> str:
    """Format question with options for MCQ solver."""
    q_text = q_data['question']
    options = q_data['options']
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return f"{q_text}\n\nOptions:\n{options_text}"


def mcq_question_solver(question: str, model_name: str, language: str = "English", timeout: int = 300) -> str:
    """Call ANEETA's MCQ solver with specified model using dynamic reload."""
    try:
        # Update environment variables
        os.environ["LLM_MODEL"] = model_name
        os.environ["CREATIVE_LLM_MODEL"] = model_name
        
        # Create new LLM instance
        new_llm = ChatOllama(
            model=model_name,
            temperature=0,
            max_tokens=700,
            timeout=timeout
        )
        
        # Test the LLM
        _ = new_llm.invoke("Test")
        
        # Update the global resources.llm
        resources.llm = new_llm
        
        # Reload the agents module to pick up the new llm
        import aneeta.nodes.agents
        importlib.reload(aneeta.nodes.agents)
        from aneeta.nodes.agents import mcq_question_solver_agent as reloaded_agent
        
        # Build State
        state = State(
            messages=[HumanMessage(content=question)],
            user_explanation_language=language,
            agent_routing="mcq_question_solver",
            teacher_vectordb_routing="biology",
            response_stream=None
        )
        
        # Call agent
        result_state = reloaded_agent(state)
        
        # Collect response
        response_chunks = []
        if isinstance(result_state, dict) and 'response_stream' in result_state:
            for chunk in result_state['response_stream']:
                response_chunks.append(str(chunk))
        
        return "".join(response_chunks).strip()
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return f"[Error: {str(e)}]\n\n{error_trace}"


def fact_check_answer(model_answer: str, correct_answer: str, options: dict) -> int:
    """Score answer correctness - binary scoring: 10 for correct, 0 for wrong.
    
    Uses regex pattern matching to detect if the correct answer is mentioned.
    """
    import re
    
    model_answer_lower = model_answer.lower()
    correct_answer_lower = correct_answer.lower()
    correct_answer_upper = correct_answer.upper()
    
    # Check if correct option appears in the answer
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
    wrong_options = [opt for opt in options.keys() if opt.upper() != correct_answer_upper]
    wrong_answer_found = False
    
    for wrong_opt in wrong_options:
        wrong_opt_lower = wrong_opt.lower()
        wrong_patterns = [
            rf"(?:correct\s+answer\s+is|answer\s+is|therefore.*option)\s+{wrong_opt_lower}\b",
        ]
        if any(re.search(pattern, model_answer_lower) for pattern in wrong_patterns):
            wrong_answer_found = True
            break
    
    # Binary scoring
    if wrong_answer_found:
        return 0
    elif correct_option_mentioned:
        return 10
    else:
        return 0


def judge_answer_quality(question: str, answer: str) -> dict:
    """Use configured LLM judge to evaluate answer quality with subject-specific criteria."""
    try:
        # Get judge from centralized config
        judge_llm = get_judge_llm()
        
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
        
        # Simplified evaluation prompt
        evaluation_prompt = f"""You are evaluating an AI MCQ solver for NEET exam questions. 

Subject: {subject}
Evaluation Criteria: {criteria_desc}

Question: {question}

MCQ Solver's Response:
{answer}

Rate the response on a scale of 1-10 considering:
- Clarity (30%): Use of proper terminology, clear explanations
- Logical Reasoning (40%): Step-by-step approach, showing work
- Correctness (30%): Proper application of concepts

Provide your evaluation in this format:
Overall Quality Score: [number 1-10]
Brief Reasoning: [2-3 sentences explaining the score]
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


def evaluate_question_with_model(question_data: Dict, model_name: str) -> Dict:
    """Evaluate a single question with a specific model."""
    question_text = format_question(question_data)
    correct_answer = question_data['correct']
    
    # Get answer with timing
    t0 = time.time()
    answer = mcq_question_solver(question_text, model_name)
    latency_ms = (time.time() - t0) * 1000
    
    # Score answer
    fact_score = fact_check_answer(answer, correct_answer, question_data['options'])
    quality_result = judge_answer_quality(question_text, answer)
    quality_score = quality_result["score"]
    
    return {
        "question_id": question_data['id'],
        "subject": question_data['subject'],
        "model": model_name,
        "full_answer": answer,
        "answer_preview": answer[:150] + "..." if len(answer) > 150 else answer,
        "correct_answer": correct_answer,
        "fact_score": fact_score,
        "quality_score": quality_score,
        "judge_reasoning": quality_result['reasoning'],
        "detected_subject": quality_result.get("subject", "Unknown"),
        "latency_ms": round(latency_ms, 1),
        "is_correct": fact_score == 10
    }


def main():
    parser = argparse.ArgumentParser(description='MLflow MCQ Evaluation with Multi-Model Comparison')
    parser.add_argument('--test-samples', type=int, default=3, help='Number of test questions (default: 3)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling (default: 42)')
    parser.add_argument('--models', nargs='+', default=["llama3.1:8b", "gemma2:9b", "mistral-nemo:12b"],
                        help='Models to test (default: llama3.1:8b gemma2:9b mistral-nemo:12b)')
    parser.add_argument('--filter-visual', action='store_true', default=True,
                        help='Filter out questions requiring visual aids (default: True)')
    args = parser.parse_args()
    
    # Print LLM judge configuration
    print("\n" + "="*70)
    print("LLM JUDGE CONFIGURATION")
    print("="*70)
    print(f"Provider: {JUDGE_CONFIG['provider']}")
    print(f"Model: {JUDGE_CONFIG['model']}")
    print(f"Temperature: {JUDGE_CONFIG['temperature']}")
    if JUDGE_CONFIG['provider'] in ['openai', 'groq', 'anthropic']:
        est_cost = estimate_judge_cost(args.test_samples * len(args.models))
        print(f"Estimated Cost: ${est_cost:.4f}")
    print("="*70 + "\n")
    
    # Load test questions
    print("Loading test questions...")
    questions = load_test_questions(
        max_questions=args.test_samples,
        seed=args.seed,
        filter_visual=args.filter_visual
    )
    
    if not questions:
        print("❌ No test questions loaded!")
        return
    
    print(f"\nEvaluating {len(questions)} questions with {len(args.models)} models")
    print(f"Models: {', '.join(args.models)}")
    print("="*70 + "\n")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"mcq_comparison_{len(questions)}q_{len(args.models)}models"):
        # Log parameters
        mlflow.log_param("num_questions", len(questions))
        mlflow.log_param("num_models", len(args.models))
        mlflow.log_param("models", ",".join(args.models))
        mlflow.log_param("random_seed", args.seed)
        mlflow.log_param("filter_visual", args.filter_visual)
        
        # Evaluate each model on each question
        all_results = []
        
        for model in args.models:
            print(f"\n{'#'*70}")
            print(f"# TESTING MODEL: {model}")
            print(f"{'#'*70}\n")
            
            model_results = []
            
            for i, q in enumerate(questions, 1):
                print(f"\n[{i}/{len(questions)}] {q['subject']} - {q['id']}")
                print(f"Question: {q['question'][:80]}...")
                print(f"Model: {model}")
                print("-"*70)
                
                result = evaluate_question_with_model(q, model)
                model_results.append(result)
                all_results.append(result)
                
                # Print result
                status = "✓ CORRECT" if result['is_correct'] else "✗ WRONG"
                print(f"{status} - Fact: {result['fact_score']}/10, Quality: {result['quality_score']}/10, Latency: {result['latency_ms']:.0f}ms")
                print(f"Answer: {result['answer_preview']}")
            
            # Log model-specific metrics to MLflow
            model_correct = sum(1 for r in model_results if r['is_correct'])
            model_accuracy = (model_correct / len(model_results)) * 100
            model_avg_fact = sum(r['fact_score'] for r in model_results) / len(model_results)
            model_avg_quality = sum(r['quality_score'] for r in model_results) / len(model_results)
            model_avg_latency = sum(r['latency_ms'] for r in model_results) / len(model_results)
            
            # Use model name as prefix for metrics
            model_prefix = model.replace(":", "_").replace(".", "_")
            mlflow.log_metric(f"{model_prefix}_accuracy", model_accuracy)
            mlflow.log_metric(f"{model_prefix}_avg_fact_score", model_avg_fact)
            mlflow.log_metric(f"{model_prefix}_avg_quality_score", model_avg_quality)
            mlflow.log_metric(f"{model_prefix}_avg_latency_ms", model_avg_latency)
            mlflow.log_metric(f"{model_prefix}_correct", model_correct)
            mlflow.log_metric(f"{model_prefix}_wrong", len(model_results) - model_correct)
        
        # Create summary report
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70 + "\n")
        
        # Model summary
        summary_data = []
        for model in args.models:
            model_results = [r for r in all_results if r['model'] == model]
            model_correct = sum(1 for r in model_results if r['is_correct'])
            summary_data.append({
                "Model": model,
                "Accuracy_%": round((model_correct / len(model_results)) * 100, 1),
                "Correct": model_correct,
                "Wrong": len(model_results) - model_correct,
                "Avg_Fact_Score": round(sum(r['fact_score'] for r in model_results) / len(model_results), 1),
                "Avg_Quality_Score": round(sum(r['quality_score'] for r in model_results) / len(model_results), 1),
                "Avg_Latency_ms": round(sum(r['latency_ms'] for r in model_results) / len(model_results), 1)
            })
        
        df_summary = pd.DataFrame(summary_data)
        print("MODEL SUMMARY")
        print("="*70)
        print(df_summary.to_string(index=False))
        
        # Save detailed results (side-by-side comparison format)
        restructured_results = []
        for q in questions:
            row = {
                "Subject": q['subject'],
                "Question_ID": q['id'],
                "Question": format_question(q),
                "Correct_Answer": q['correct']
            }
            
            # Add each model's results
            for model in args.models:
                model_result = next((r for r in all_results if r['question_id'] == q['id'] and r['model'] == model), None)
                if model_result:
                    model_prefix = model.replace(":", "_").replace(".", "_")
                    row[f"{model_prefix}_Output"] = model_result["full_answer"]
                    row[f"{model_prefix}_Fact_Score"] = model_result["fact_score"]
                    row[f"{model_prefix}_Quality_Score"] = model_result["quality_score"]
                    row[f"{model_prefix}_Judge_Reasoning"] = model_result["judge_reasoning"]
                    row[f"{model_prefix}_Latency_ms"] = model_result["latency_ms"]
                    row[f"{model_prefix}_Correct"] = model_result["is_correct"]
            
            restructured_results.append(row)
        
        df_detailed = pd.DataFrame(restructured_results)
        
        # Save CSVs
        detailed_output = ROOT / "results" / "mlflow_mcq_detailed_results.csv"
        summary_output = ROOT / "results" / "mlflow_mcq_model_summary.csv"
        
        df_detailed.to_csv(detailed_output, index=False)
        df_summary.to_csv(summary_output, index=False)
        
        # Log as MLflow artifacts
        mlflow.log_artifact(str(detailed_output))
        mlflow.log_artifact(str(summary_output))
        
        print(f"\n✓ Detailed results saved: {detailed_output}")
        print(f"✓ Model summary saved: {summary_output}")
        print(f"✓ Results logged to MLflow experiment: {mlflow.get_experiment_by_name(os.getenv('MLFLOW_EXPERIMENT_NAME', 'aneetaa-mcq-model-comparison')).name}")
        print(f"✓ View in MLflow UI: {mlflow.get_tracking_uri()}")
        
        # Log overall comparison metrics
        best_model = df_summary.loc[df_summary['Accuracy_%'].idxmax(), 'Model']
        best_accuracy = df_summary['Accuracy_%'].max()
        
        mlflow.log_metric("best_accuracy", best_accuracy)
        mlflow.log_param("best_model", best_model)
        
        print(f"\n🏆 Best Model: {best_model} ({best_accuracy:.1f}% accuracy)")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
