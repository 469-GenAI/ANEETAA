"""
Three-Way Agent Comparison: Vanilla ANEETAA vs DSPy Baseline vs DSPy Optimized

This script compares three agent implementations:
1. Vanilla ANEETAA - Original LangChain-based implementation
2. DSPy Baseline - Unoptimized DSPy signature/module
3. DSPy Optimized - After SIMBA/COPRO optimization

All results are logged to MLflow for comprehensive comparison.
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
import mlflow
import dspy
from dotenv import load_dotenv

# Setup paths
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

load_dotenv()

# Configure MLflow
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns'))
mlflow.set_experiment('aneetaa-three-way-comparison')

print("="*70)
print("THREE-WAY AGENT COMPARISON: Vanilla vs DSPy Baseline vs DSPy Optimized")
print("="*70)
print(f"✓ MLflow experiment: aneetaa-three-way-comparison")
print("="*70 + "\n")


# ============================================================================
# PART 1: DSPy MCQ Solver Signature & Modules
# ============================================================================

class MCQSolverSignature(dspy.Signature):
    """Solve NEET MCQ questions with explanations."""
    
    question = dspy.InputField(desc="The MCQ question with options A, B, C, D")
    subject = dspy.InputField(desc="Subject area: Physics, Chemistry, or Biology")
    reasoning = dspy.OutputField(desc="Step-by-step explanation of how to solve this question")
    answer = dspy.OutputField(desc="Final answer: A, B, C, or D")


class MCQSolverModule(dspy.Module):
    """DSPy Module for MCQ solving."""
    
    def __init__(self):
        super().__init__()
        # Use 'predictor' to match optimized model naming
        self.predictor = dspy.ChainOfThought(MCQSolverSignature)
    
    def forward(self, question: str, subject: str = "Biology") -> dspy.Prediction:
        """Solve MCQ question."""
        return self.predictor(question=question, subject=subject)


# ============================================================================
# PART 2: Agent Implementations
# ============================================================================

def vanilla_aneetaa_agent(question: str, model_name: str = "gemma2:9b") -> str:
    """Vanilla ANEETAA agent using LangChain."""
    from aneeta.state.models import State
    from aneeta.core import resources
    from langchain_core.messages import HumanMessage
    from langchain_ollama import ChatOllama
    import importlib
    
    # Update model
    os.environ["LLM_MODEL"] = model_name
    os.environ["CREATIVE_LLM_MODEL"] = model_name
    
    # Create new LLM
    new_llm = ChatOllama(model=model_name, temperature=0, max_tokens=700)
    _ = new_llm.invoke("Test")
    resources.llm = new_llm
    
    # Reload agents module
    import aneeta.nodes.agents
    importlib.reload(aneeta.nodes.agents)
    from aneeta.nodes.agents import mcq_question_solver_agent
    
    # Create state
    state = State(
        messages=[HumanMessage(content=question)],
        user_explanation_language="English",
        agent_routing="mcq_question_solver",
        teacher_vectordb_routing="biology",
        response_stream=None
    )
    
    # Get result
    result = mcq_question_solver_agent(state)
    
    # Extract answer
    if isinstance(result, dict) and 'response_stream' in result:
        chunks = list(result['response_stream'])
        return ''.join(chunks).strip()
    return str(result)


def dspy_baseline_agent(question: str, subject: str = "Biology") -> dspy.Prediction:
    """Unoptimized DSPy agent (baseline)."""
    solver = MCQSolverModule()
    return solver(question=question, subject=subject)


def dspy_optimized_agent(question: str, subject: str = "Biology", optimized_model_path: str = None) -> dspy.Prediction:
    """Optimized DSPy agent (after SIMBA/COPRO)."""
    if optimized_model_path and Path(optimized_model_path).exists():
        # Load optimized model
        solver = MCQSolverModule()
        solver.load(optimized_model_path)
        print(f"  ✓ Loaded optimized model from: {optimized_model_path}")
    else:
        # Fallback to baseline if no optimized model
        print(f"  ⚠ No optimized model found, using baseline")
        solver = MCQSolverModule()
    
    return solver(question=question, subject=subject)


# ============================================================================
# PART 3: Data Loading & Formatting
# ============================================================================

def load_test_questions(max_questions: int = 10, seed: int = 42, filter_visual: bool = True) -> List[Dict]:
    """Load test questions from Gemini 2.5 Pro Data."""
    data_dir = ROOT / "aneeta_v2" / "Processed Data" / "Gemini 2.5 Pro Data"
    
    all_questions = []
    for json_file in sorted(data_dir.glob("*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                all_questions.extend(json.load(f))
        except Exception as e:
            print(f"⚠ Warning: {json_file.name}: {e}")
    
    print(f"✓ Loaded {len(all_questions)} total questions")
    
    if filter_visual:
        all_questions = [q for q in all_questions if not q.get('metadata', {}).get('requires_visual', False)]
        print(f"✓ Filtered to {len(all_questions)} non-visual questions")
    
    if len(all_questions) > max_questions:
        random.seed(seed)
        all_questions = random.sample(all_questions, max_questions)
        print(f"✓ Sampled {max_questions} questions (seed={seed})")
    
    return [
        {
            'id': q.get('question_id', ''),
            'question': q['question_text'],
            'options': q.get('options', {}),
            'correct': q['correct_answer'],
            'subject': q.get('metadata', {}).get('subject', 'unknown')
        }
        for q in all_questions
        if 'question_text' in q and 'correct_answer' in q
    ]


def format_question(q_data: Dict, add_cache_buster: bool = False) -> str:
    """Format question with options."""
    q_text = q_data['question']
    options = q_data['options']
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    formatted = f"{q_text}\n\nOptions:\n{options_text}"
    
    # Add cache buster to ensure unique prompts
    if add_cache_buster:
        import uuid
        formatted += f"\n\n[Request ID: {uuid.uuid4()}]"
    
    return formatted


# ============================================================================
# PART 4: Answer Validation
# ============================================================================

def extract_answer_letter(response: str) -> str:
    """Extract answer letter (A/B/C/D) from response."""
    import re
    
    # Look for patterns like "A:", "Answer: B", "(C)", etc.
    patterns = [
        r'\b([A-D])\b',  # Standalone letter
        r'\(([A-D])\)',  # (A)
        r'\*\*([A-D]):',  # **A:
        r'answer\s+is\s+([A-D])',  # answer is A
        r'correct\s+answer\s+is\s+([A-D])',  # correct answer is A
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None


def validate_answer(response: str, correct_answer: str) -> bool:
    """Check if response contains correct answer."""
    extracted = extract_answer_letter(response)
    return extracted == correct_answer.upper() if extracted else False


# ============================================================================
# PART 5: Evaluation Function
# ============================================================================

def evaluate_agent(agent_type: str, question_data: Dict, **kwargs) -> Dict:
    """Evaluate a single agent on a single question."""
    # Add cache buster for DSPy agents
    add_cache_buster = agent_type.startswith('dspy')
    question_text = format_question(question_data, add_cache_buster=add_cache_buster)
    correct_answer = question_data['correct']
    subject = question_data.get('subject', 'Biology')
    
    try:
        t0 = time.time()
        
        if agent_type == "vanilla":
            model_name = kwargs.get('model_name', 'gemma2:9b')
            response = vanilla_aneetaa_agent(question_text, model_name)
        
        elif agent_type == "dspy_baseline":
            # Configure DSPy with cache busting
            provider = kwargs.get('provider', 'openai')
            model = kwargs.get('model', 'gpt-4o-mini')
            configure_dspy(provider, model, force_new=True)
            
            prediction = dspy_baseline_agent(question_text, subject=subject)
            response = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
        
        elif agent_type == "dspy_optimized":
            # Configure DSPy with cache busting
            provider = kwargs.get('provider', 'openai')
            model = kwargs.get('model', 'gpt-4o-mini')
            configure_dspy(provider, model, force_new=True)
            
            optimized_path = kwargs.get('optimized_model_path')
            prediction = dspy_optimized_agent(question_text, subject=subject, optimized_model_path=optimized_path)
            response = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        latency_ms = (time.time() - t0) * 1000
        is_correct = validate_answer(response, correct_answer)
        
        return {
            'question_id': question_data['id'],
            'subject': question_data['subject'],
            'agent_type': agent_type,
            'response': response,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'latency_ms': round(latency_ms, 1)
        }
    
    except Exception as e:
        import traceback
        return {
            'question_id': question_data['id'],
            'subject': question_data['subject'],
            'agent_type': agent_type,
            'response': f"[Error: {str(e)}]",
            'correct_answer': correct_answer,
            'is_correct': False,
            'latency_ms': 0,
            'error': traceback.format_exc()
        }


def configure_dspy(provider: str, model: str, force_new: bool = False):
    """Configure DSPy LM with proper cache busting."""
    
    # Vary temperature slightly to prevent caching
    # Use timestamp to generate slightly different temp each time (0.3-0.5 range)
    temp_variation = (hash(str(time.time())) % 20) / 100  # 0.00 to 0.19
    base_temp = 0.3
    temperature = base_temp + temp_variation if force_new else 0.3
    
    if provider == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        lm = dspy.LM(
            model=f"openai/{model}",
            max_tokens=500,
            temperature=temperature
        )
    
    elif provider == "ollama":
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        
        lm = dspy.LM(
            model=f"ollama_chat/{model}",
            api_base=ollama_url,
            max_tokens=500,
            temperature=temperature
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    # Configure with explicit cache disable
    dspy.settings.configure(lm=lm)
    
    # Force disable any caching mechanisms
    if hasattr(dspy.settings, 'cache_turn_on'):
        dspy.settings.cache_turn_on = False
    if hasattr(lm, '_cache'):
        lm._cache = None


# ============================================================================
# PART 6: Main Comparison Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compare Vanilla, DSPy Baseline, and DSPy Optimized agents')
    parser.add_argument('--test-samples', type=int, default=5, help='Number of test questions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--vanilla-model', default='gemma2:9b', help='Ollama model for vanilla agent')
    parser.add_argument('--dspy-provider', default='openai', choices=['openai', 'ollama'], help='DSPy LLM provider')
    parser.add_argument('--dspy-model', default='gpt-4o-mini', help='Model for DSPy agents')
    parser.add_argument('--optimized-model-path', help='Path to optimized DSPy model')
    args = parser.parse_args()
    
    # Load questions
    print("Loading test questions...")
    questions = load_test_questions(max_questions=args.test_samples, seed=args.seed)
    
    if not questions:
        print("❌ No questions loaded!")
        return
    
    print(f"\nEvaluating {len(questions)} questions with 3 agent types")
    print("="*70 + "\n")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"three_way_comparison_{len(questions)}q"):
        # Log parameters
        mlflow.log_param("num_questions", len(questions))
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("vanilla_model", args.vanilla_model)
        mlflow.log_param("dspy_provider", args.dspy_provider)
        mlflow.log_param("dspy_model", args.dspy_model)
        
        all_results = []
        
        # Agent configurations
        agent_configs = [
            {
                'type': 'vanilla',
                'name': 'Vanilla ANEETAA',
                'kwargs': {'model_name': args.vanilla_model}
            },
            {
                'type': 'dspy_baseline',
                'name': 'DSPy Baseline (Unoptimized)',
                'kwargs': {'provider': args.dspy_provider, 'model': args.dspy_model}
            },
            {
                'type': 'dspy_optimized',
                'name': 'DSPy Optimized',
                'kwargs': {
                    'provider': args.dspy_provider,
                    'model': args.dspy_model,
                    'optimized_model_path': args.optimized_model_path
                }
            }
        ]
        
        # Evaluate each agent
        for agent_config in agent_configs:
            agent_type = agent_config['type']
            agent_name = agent_config['name']
            
            print(f"\n{'#'*70}")
            print(f"# {agent_name}")
            print(f"{'#'*70}\n")
            
            agent_results = []
            
            for i, q in enumerate(questions, 1):
                print(f"[{i}/{len(questions)}] {q['subject']} - {q['id']}")
                
                result = evaluate_agent(agent_type, q, **agent_config['kwargs'])
                agent_results.append(result)
                all_results.append(result)
                
                status = "✓ CORRECT" if result['is_correct'] else "✗ WRONG"
                print(f"{status} - {result['latency_ms']:.0f}ms")
            
            # Log agent-specific metrics
            correct = sum(1 for r in agent_results if r['is_correct'])
            accuracy = (correct / len(agent_results)) * 100
            avg_latency = sum(r['latency_ms'] for r in agent_results) / len(agent_results)
            
            mlflow.log_metric(f"{agent_type}_accuracy", accuracy)
            mlflow.log_metric(f"{agent_type}_correct", correct)
            mlflow.log_metric(f"{agent_type}_avg_latency_ms", avg_latency)
        
        # Summary
        print("\n" + "="*70)
        print("FINAL COMPARISON SUMMARY")
        print("="*70 + "\n")
        
        for agent_config in agent_configs:
            agent_type = agent_config['type']
            agent_results = [r for r in all_results if r['agent_type'] == agent_type]
            correct = sum(1 for r in agent_results if r['is_correct'])
            accuracy = (correct / len(agent_results)) * 100
            avg_latency = sum(r['latency_ms'] for r in agent_results) / len(agent_results)
            
            print(f"{agent_config['name']:30} | Accuracy: {accuracy:5.1f}% | Latency: {avg_latency:6.1f}ms")
        
        # Save results
        import pandas as pd
        df = pd.DataFrame(all_results)
        output_file = ROOT / "three_way_comparison_results.csv"
        df.to_csv(output_file, index=False)
        
        mlflow.log_artifact(str(output_file))
        print(f"\n✓ Results saved: {output_file}")
        print(f"✓ Logged to MLflow: aneetaa-three-way-comparison")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
