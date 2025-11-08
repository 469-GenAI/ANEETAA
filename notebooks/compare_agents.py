"""
Compare Vanilla ANEETAA, DSPy Baseline, and Optimized DSPy Agents

This script evaluates and compares:
1. Vanilla ANEETAA MCQ Solver (original implementation)
2. DSPy MCQ Solver Baseline (unoptimized)
3. DSPy MCQ Solver Optimized (after SIMBA optimization)

All results are logged to MLflow for easy comparison.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

import dspy
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient


def setup_paths():
    """Add src to Python path for ANEETAA imports."""
    src_path = Path(__file__).parent.parent / 'src'
    sys.path.insert(0, str(src_path))
    print(f"✓ Added to path: {src_path}")


def setup_mlflow():
    """Setup MLflow tracking."""
    load_dotenv()
    
    # Use local MLflow
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment('aneeta-agent-comparison')
    
    print(f"✓ Using LOCAL MLflow")
    print(f"  🌐 View results at: http://localhost:8080")


def configure_llm(provider: str = "openai", model: str = None):
    """Configure DSPy with the chosen LLM provider."""
    if provider == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")
        
        model_name = model or "openai/gpt-4o-mini"
        lm = dspy.LM(
            model=model_name,
            max_tokens=500,
            temperature=0.1
        )
        print(f"✓ Using OpenAI: {model_name}")
        
    elif provider == "ollama":
        model_name = model or "llama3.1:8b"
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        
        lm = dspy.LM(
            model=f"ollama_chat/{model_name}",
            api_base=ollama_url,
            max_tokens=500,
            temperature=0.1
        )
        print(f"✓ Using Ollama: {model_name}")
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    dspy.settings.configure(lm=lm, cache_turn_on=False)
    print(f"✓ DSPy configured")
    return lm


def load_mcq_test_data(data_dir: Path = None, max_questions: int = 20) -> List[Dict]:
    """
    Load MCQ test data from solved question papers.
    
    Returns:
        List of MCQ questions with answers
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'Processed Data'
    
    mcq_path = data_dir / 'solved_question_papers.json'
    
    if not mcq_path.exists():
        print(f"⚠ Warning: {mcq_path} not found")
        return []
    
    with open(mcq_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract MCQ questions
    test_questions = []
    for item in data[:max_questions]:
        if 'question' in item and 'correct_answer' in item:
            test_questions.append({
                'question': item['question'],
                'options': item.get('options', []),
                'correct_answer': item['correct_answer'],
                'subject': item.get('subject', 'unknown'),
                'context': item.get('context', '')
            })
    
    print(f"✓ Loaded {len(test_questions)} MCQ test questions")
    return test_questions


def validate_mcq_answer(question_data: Dict, prediction: Any) -> float:
    """
    Validate MCQ answer using GPT-4o-mini as judge.
    
    Args:
        question_data: Dict with question, options, correct_answer
        prediction: Agent's prediction/response
    
    Returns:
        Score between 0.0 and 1.0
    """
    from openai import OpenAI
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        # Fallback to simple exact match
        return validate_mcq_answer_simple(question_data, prediction)
    
    # Extract answer from prediction
    if hasattr(prediction, 'answer'):
        response = prediction.answer
    elif hasattr(prediction, 'response'):
        response = prediction.response
    else:
        response = str(prediction)
    
    # Quick check for empty response
    if len(response.strip()) < 1:
        return 0.0
    
    try:
        client = OpenAI(api_key=api_key)
        
        options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(question_data.get('options', []))])
        
        judge_prompt = f"""You are evaluating an MCQ answer for a NEET exam question.

Question: {question_data['question']}

Options:
{options_str}

Correct Answer: {question_data['correct_answer']}

Student's Answer: {response}

Evaluate if the student's answer matches the correct answer. Consider:
1. Direct matches (e.g., "A", "Option A", "The answer is A")
2. The student may have provided the full text of the correct option
3. Minor formatting differences should be ignored

Return ONLY a number:
- 1.0 if the answer is correct
- 0.5 if the answer is partially correct or unclear
- 0.0 if the answer is wrong

Just return the number (1.0, 0.5, or 0.0), nothing else."""

        response_obj = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=10
        )
        
        score_text = response_obj.choices[0].message.content.strip()
        score = float(score_text)
        
        # Clamp to valid range
        score = max(0.0, min(1.0, score))
        return score
        
    except Exception as e:
        print(f"⚠ Error using LLM judge: {e}, falling back to simple validation")
        return validate_mcq_answer_simple(question_data, prediction)


def validate_mcq_answer_simple(question_data: Dict, prediction: Any) -> float:
    """
    Simple rule-based MCQ validation (fallback).
    
    Returns:
        1.0 if correct, 0.0 if wrong
    """
    if hasattr(prediction, 'answer'):
        response = prediction.answer
    elif hasattr(prediction, 'response'):
        response = prediction.response
    else:
        response = str(prediction)
    
    response = response.strip().upper()
    correct = question_data['correct_answer'].strip().upper()
    
    # Check for exact match or if correct answer is in response
    if correct in response or response in correct:
        return 1.0
    
    return 0.0


def evaluate_vanilla_agent(test_questions: List[Dict], max_samples: int = 10) -> float:
    """
    Evaluate vanilla ANEETAA MCQ Solver agent.
    
    Returns:
        Average accuracy score
    """
    print("\n" + "="*60)
    print("Evaluating Vanilla ANEETAA MCQ Solver")
    print("="*60)
    
    from aneeta.nodes.agents import MCQSolverAgent
    from aneeta.state.models import State
    
    agent = MCQSolverAgent()
    scores = []
    
    for i, q_data in enumerate(test_questions[:max_samples]):
        try:
            # Prepare state
            state = State(
                question=q_data['question'],
                language="English"
            )
            
            # Run agent
            result = agent.solve(state)
            
            # Validate answer
            score = validate_mcq_answer(q_data, result)
            scores.append(score)
            
            print(f"  [{i+1}/{max_samples}] Score: {score:.2f}")
            
        except Exception as e:
            print(f"  [{i+1}/{max_samples}] Error: {e}")
            scores.append(0.0)
    
    avg_score = np.mean(scores) if scores else 0.0
    print(f"\n✓ Vanilla Agent Average Score: {avg_score:.2%}")
    return avg_score


def evaluate_dspy_baseline(test_questions: List[Dict], max_samples: int = 10) -> float:
    """
    Evaluate DSPy MCQ Solver baseline (unoptimized).
    
    Returns:
        Average accuracy score
    """
    print("\n" + "="*60)
    print("Evaluating DSPy MCQ Solver Baseline (Unoptimized)")
    print("="*60)
    
    from aneeta.nodes.agents_dspy import MCQSolverAgentDSPy
    
    agent = MCQSolverAgentDSPy()
    scores = []
    
    for i, q_data in enumerate(test_questions[:max_samples]):
        try:
            # Run agent with required parameters
            result = agent(
                question=q_data['question'],
                context=q_data.get('context', ''),
                language="English"
            )
            
            # Validate answer
            score = validate_mcq_answer(q_data, result)
            scores.append(score)
            
            print(f"  [{i+1}/{max_samples}] Score: {score:.2f}")
            
        except Exception as e:
            print(f"  [{i+1}/{max_samples}] Error: {e}")
            scores.append(0.0)
    
    avg_score = np.mean(scores) if scores else 0.0
    print(f"\n✓ DSPy Baseline Average Score: {avg_score:.2%}")
    return avg_score


def evaluate_dspy_optimized(test_questions: List[Dict], model_uri: str = None, max_samples: int = 10) -> float:
    """
    Evaluate optimized DSPy MCQ Solver (loaded from MLflow).
    
    Args:
        model_uri: MLflow model URI (e.g., 'runs:/run_id/model_path')
        test_questions: Test questions
        max_samples: Number of samples to evaluate
    
    Returns:
        Average accuracy score
    """
    print("\n" + "="*60)
    print("Evaluating DSPy MCQ Solver Optimized")
    print("="*60)
    
    if model_uri:
        print(f"Loading optimized model from: {model_uri}")
        try:
            agent = mlflow.dspy.load_model(model_uri)
            print("✓ Loaded optimized model from MLflow")
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            print("  Using baseline DSPy agent instead")
            from aneeta.nodes.agents_dspy import MCQSolverAgentDSPy
            agent = MCQSolverAgentDSPy()
    else:
        print("⚠ No model URI provided, using baseline")
        from aneeta.nodes.agents_dspy import MCQSolverAgentDSPy
        agent = MCQSolverAgentDSPy()
    
    scores = []
    
    for i, q_data in enumerate(test_questions[:max_samples]):
        try:
            # Run agent
            result = agent(
                question=q_data['question'],
                context=q_data.get('context', ''),
                language="English"
            )
            
            # Validate answer
            score = validate_mcq_answer(q_data, result)
            scores.append(score)
            
            print(f"  [{i+1}/{max_samples}] Score: {score:.2f}")
            
        except Exception as e:
            print(f"  [{i+1}/{max_samples}] Error: {e}")
            scores.append(0.0)
    
    avg_score = np.mean(scores) if scores else 0.0
    print(f"\n✓ DSPy Optimized Average Score: {avg_score:.2%}")
    return avg_score


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Compare ANEETAA Agents')
    parser.add_argument('--provider', type=str, default='openai', choices=['openai', 'ollama'],
                        help='LLM provider')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name')
    parser.add_argument('--test-samples', type=int, default=10,
                        help='Number of test samples per agent')
    parser.add_argument('--optimized-model-uri', type=str, default=None,
                        help='MLflow URI of optimized model (e.g., runs:/run_id/model)')
    parser.add_argument('--skip-vanilla', action='store_true',
                        help='Skip vanilla ANEETAA evaluation')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip DSPy baseline evaluation')
    parser.add_argument('--skip-optimized', action='store_true',
                        help='Skip DSPy optimized evaluation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ANEETAA Agent Comparison")
    print("="*60 + "\n")
    
    # Setup
    setup_paths()
    load_dotenv()
    
    # Configure LLM
    print(f"\nConfiguring LLM ({args.provider})...")
    configure_llm(args.provider, args.model)
    
    # Setup MLflow
    print("\nSetting up MLflow...")
    setup_mlflow()
    
    # Load test data
    print("\nLoading test data...")
    test_questions = load_mcq_test_data(max_questions=args.test_samples)
    
    if not test_questions:
        print("❌ No test data loaded. Exiting.")
        return
    
    # Run evaluations and log to MLflow
    with mlflow.start_run(run_name="agent_comparison"):
        results = {}
        
        # Evaluate Vanilla ANEETAA
        if not args.skip_vanilla:
            vanilla_score = evaluate_vanilla_agent(test_questions, args.test_samples)
            results['vanilla_score'] = vanilla_score
            mlflow.log_metric("vanilla_mcq_score", vanilla_score)
        
        # Evaluate DSPy Baseline
        if not args.skip_baseline:
            baseline_score = evaluate_dspy_baseline(test_questions, args.test_samples)
            results['dspy_baseline_score'] = baseline_score
            mlflow.log_metric("dspy_baseline_score", baseline_score)
        
        # Evaluate DSPy Optimized
        if not args.skip_optimized:
            optimized_score = evaluate_dspy_optimized(
                test_questions, 
                args.optimized_model_uri,
                args.test_samples
            )
            results['dspy_optimized_score'] = optimized_score
            mlflow.log_metric("dspy_optimized_score", optimized_score)
        
        # Log parameters
        mlflow.log_param("test_samples", args.test_samples)
        mlflow.log_param("provider", args.provider)
        mlflow.log_param("model", args.model or "default")
        
        # Calculate improvements
        if 'vanilla_score' in results and 'dspy_optimized_score' in results:
            improvement = ((results['dspy_optimized_score'] - results['vanilla_score']) / 
                          results['vanilla_score'] * 100) if results['vanilla_score'] > 0 else 0
            mlflow.log_metric("improvement_vs_vanilla", improvement)
            print(f"\n🎯 DSPy Optimized vs Vanilla: {improvement:+.1f}%")
        
        if 'dspy_baseline_score' in results and 'dspy_optimized_score' in results:
            improvement = ((results['dspy_optimized_score'] - results['dspy_baseline_score']) / 
                          results['dspy_baseline_score'] * 100) if results['dspy_baseline_score'] > 0 else 0
            mlflow.log_metric("improvement_vs_baseline", improvement)
            print(f"🎯 DSPy Optimized vs Baseline: {improvement:+.1f}%")
        
        # Summary
        print("\n" + "="*60)
        print("📊 COMPARISON RESULTS")
        print("="*60)
        for key, value in results.items():
            agent_name = key.replace('_score', '').replace('_', ' ').title()
            print(f"{agent_name:30s}: {value:.2%}")
        
        print("\n✓ Results logged to MLflow!")
        print("  View at: http://localhost:8080")
        print("  Experiment: aneeta-agent-comparison")


if __name__ == "__main__":
    main()
