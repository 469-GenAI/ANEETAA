"""
Three-Way Agent Comparison: Vanilla ANEETAA vs DSPy Baseline vs DSPy Optimized

This script compares three agent implementations:
1. Vanilla ANEETAA - Original LangChain-based implementation
2. DSPy Baseline - Unoptimized DSPy signature/module
3. DSPy Optimized - After SIMBA/COPRO optimization

All results are logged to MLflow for comprehensive comparison.
Uses centralized LLM judge configuration (supports OpenAI, Groq, Anthropic, Ollama)
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
ROOT = Path(__file__).parent.parent.parent.parent.resolve()  # Go up to project root
sys.path.insert(0, str(ROOT / "src"))

# Add config folder to path for centralized judge config
CONFIG_DIR = Path(__file__).parent.parent / "config"
sys.path.insert(0, str(CONFIG_DIR))
from llm_judge_config import get_judge_llm, estimate_judge_cost, JUDGE_CONFIG

load_dotenv()

# Configure MLflow
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns'))
mlflow.set_experiment('aneetaa-three-way-comparison')

print("="*70)
print("THREE-WAY AGENT COMPARISON: Vanilla vs DSPy Baseline vs DSPy Optimized")
print("="*70)
print(f"[OK] MLflow experiment: aneetaa-three-way-comparison")
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

def load_test_questions(
    max_questions: int = 10, 
    seed: int = 42, 
    filter_visual: bool = True,
    use_validation_set: bool = True
) -> List[Dict]:
    """
    Load test questions for comparison.
    
    Args:
        max_questions: Maximum number of questions to load
        seed: Random seed for sampling
        filter_visual: Filter out questions requiring visuals (only for Gemini data)
        use_validation_set: If True, use dspy_dataset_val.jsonl; else use Gemini data
    
    Returns:
        List of formatted question dictionaries
    """
    if use_validation_set:
        # Use the validation dataset (RECOMMENDED - matches training evaluation)
        val_file = ROOT / "aneeta_v2" / "Processed Data" / "dspy_dataset_val.jsonl"
        
        if not val_file.exists():
            print(f"⚠ Validation dataset not found: {val_file}")
            print(f"  Falling back to Gemini data...")
            use_validation_set = False
        else:
            print(f"✓ Loading validation dataset: {val_file}")
            all_questions = []
            with open(val_file, 'r', encoding='utf-8') as f:
                for line in f:
                    all_questions.append(json.loads(line))
            
            print(f"✓ Loaded {len(all_questions)} validation questions")
            
            # Sample if needed
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
                    'subject': q.get('subject', 'unknown')
                }
                for q in all_questions
            ]
    
    # Fall back to Gemini data (original method)
    data_dir = ROOT / "aneeta_v2" / "Processed Data" / "Gemini 2.5 Pro Data"
    
    all_questions = []
    for json_file in sorted(data_dir.glob("*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                all_questions.extend(json.load(f))
        except Exception as e:
            print(f"⚠ Warning: {json_file.name}: {e}")
    
    print(f"✓ Loaded {len(all_questions)} total questions from Gemini data")
    
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
# PART 4: Answer Validation & Scoring
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

def validate_answer(response: str, correct_answer: str) -> bool:
    """Check if response contains correct answer (for backward compatibility)."""
    import re
    patterns = [
        rf"\b{correct_answer.lower()}\b",
        rf"\({correct_answer.lower()}\)",
        rf"answer\s+is\s+{correct_answer.lower()}\b",
    ]
    return any(re.search(pattern, response.lower()) for pattern in patterns)

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
        
        # Simplified evaluation prompt with subject-specific criteria
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
        
        # Score answer with both metrics
        fact_score = fact_check_answer(response, correct_answer, question_data['options'])
        is_correct = fact_score == 10  # Backward compatibility
        
        # Judge quality
        quality_result = judge_answer_quality(question_text, response)
        
        return {
            'question_id': question_data['id'],
            'subject': question_data['subject'],
            'agent_type': agent_type,
            'response': response,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'fact_score': fact_score,  # NEW: 0/10 score
            'quality_score': quality_result['score'],  # NEW: LLM judge score
            'judge_reasoning': quality_result['reasoning'],  # NEW: Judge feedback
            'detected_subject': quality_result.get('subject', 'Unknown'),  # NEW: Detected subject
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
            'fact_score': 0,  # NEW: 0/10 score
            'quality_score': 0,  # NEW: LLM judge score
            'judge_reasoning': 'Error occurred',  # NEW: Judge feedback
            'detected_subject': 'Unknown',  # NEW: Detected subject
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
    parser.add_argument('--use-validation-set', action='store_true', default=True,
                        help='Use validation dataset (default: True)')
    parser.add_argument('--use-gemini-data', action='store_true',
                        help='Use Gemini data instead of validation set')
    args = parser.parse_args()
    
    # Print LLM judge configuration
    print("\n" + "="*70)
    print("LLM JUDGE CONFIGURATION")
    print("="*70)
    print(f"Provider: {JUDGE_CONFIG['provider']}")
    print(f"Model: {JUDGE_CONFIG['model']}")
    print(f"Temperature: {JUDGE_CONFIG['temperature']}")
    if JUDGE_CONFIG['provider'] in ['openai', 'groq', 'anthropic']:
        # Estimate cost: 3 agent types × test samples
        est_cost = estimate_judge_cost(3 * args.test_samples)
        print(f"Estimated Cost: ${est_cost['estimated_cost_usd']:.4f}")
    print("="*70 + "\n")
    
    # Determine which dataset to use
    use_val_set = args.use_validation_set and not args.use_gemini_data
    
    # Load questions
    print("Loading test questions...")
    questions = load_test_questions(
        max_questions=args.test_samples, 
        seed=args.seed,
        use_validation_set=use_val_set
    )
    
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
                print(f"{status} | Fact: {result['fact_score']}/10 | Quality: {result['quality_score']}/10 | {result['latency_ms']:.0f}ms")
            
            # Log agent-specific metrics
            correct = sum(1 for r in agent_results if r['is_correct'])
            accuracy = (correct / len(agent_results)) * 100
            avg_fact_score = sum(r['fact_score'] for r in agent_results) / len(agent_results)
            avg_quality_score = sum(r['quality_score'] for r in agent_results) / len(agent_results)
            avg_latency = sum(r['latency_ms'] for r in agent_results) / len(agent_results)
            
            mlflow.log_metric(f"{agent_type}_accuracy", accuracy)
            mlflow.log_metric(f"{agent_type}_correct", correct)
            mlflow.log_metric(f"{agent_type}_avg_fact_score", avg_fact_score)
            mlflow.log_metric(f"{agent_type}_avg_quality_score", avg_quality_score)
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
            avg_fact_score = sum(r['fact_score'] for r in agent_results) / len(agent_results)
            avg_quality_score = sum(r['quality_score'] for r in agent_results) / len(agent_results)
            avg_latency = sum(r['latency_ms'] for r in agent_results) / len(agent_results)
            
            print(f"{agent_config['name']:30} | Acc: {accuracy:5.1f}% | Fact: {avg_fact_score:4.1f}/10 | Quality: {avg_quality_score:4.1f}/10 | {avg_latency:6.1f}ms")
        
        # Save results
        import pandas as pd
        df = pd.DataFrame(all_results)
        output_file = ROOT / "results" / "three_way_comparison_results.csv"
        df.to_csv(output_file, index=False)
        
        mlflow.log_artifact(str(output_file))
        print(f"\n✓ Results saved: {output_file}")
        print(f"✓ Logged to MLflow: aneetaa-three-way-comparison")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
