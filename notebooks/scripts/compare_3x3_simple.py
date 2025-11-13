"""
Simplified 3×3 Matrix Comparison - No vanilla ANEETAA (avoids hanging)

Compares DSPy only:
- 3 Models: llama3.1:8b, gemma2:9b, mistral-nemo:12b
- 2 Agent Types: DSPy Baseline, DSPy Optimized
- Total: 6 configurations (instead of 9)

This avoids the vanilla ANEETAA hanging issue.
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
from dotenv import load_dotenv
import dspy

# Setup
ROOT = Path(__file__).parent.parent.parent.resolve()
load_dotenv()

# MLflow setup
mlflow.set_tracking_uri('file:./mlruns')
mlflow.set_experiment('aneetaa-dspy-only-comparison')

print("="*70)
print("SIMPLIFIED 3×2 MATRIX COMPARISON (DSPy Only)")
print("="*70)

# ============================================================================
# Configuration
# ============================================================================

MODELS = [
    {'name': 'llama3.1:8b', 'display': 'Llama 3.1 8B', 'optimized_path': 'models/dspy_bootstrap_llama3.1_8b.json'},
    {'name': 'gemma2:9b', 'display': 'Gemma2 9B', 'optimized_path': 'models/dspy_bootstrap_gemma2_9b.json'},
    {'name': 'mistral-nemo:12b', 'display': 'Mistral Nemo 12B', 'optimized_path': 'models/dspy_bootstrap_mistral_nemo_12b.json'},
]

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
# Helper Functions
# ============================================================================

def load_test_questions(max_questions: int, seed: int, use_validation_set: bool = True) -> List[Dict]:
    """
    Load test questions for comparison.
    
    Args:
        max_questions: Maximum number of questions
        seed: Random seed for sampling
        use_validation_set: If True, use val.jsonl; else use combined.jsonl
    
    Returns:
        List of question dictionaries
    """
    if use_validation_set:
        # Use validation dataset (RECOMMENDED)
        val_file = ROOT / "aneeta_v2" / "Processed Data" / "dspy_dataset_val.jsonl"
        
        if not val_file.exists():
            print(f"⚠ Validation dataset not found: {val_file}")
            print(f"  Falling back to combined dataset...")
            use_validation_set = False
        else:
            print(f"✓ Loading validation dataset: {val_file}")
            all_questions = []
            with open(val_file, 'r', encoding='utf-8') as f:
                for line in f:
                    all_questions.append(json.loads(line))
            
            print(f"✓ Loaded {len(all_questions)} validation questions")
            
            if len(all_questions) > max_questions:
                random.seed(seed)
                all_questions = random.sample(all_questions, max_questions)
                print(f"✓ Sampled {max_questions} questions (seed={seed})")
            
            return all_questions
    
    # Fall back to combined dataset
    combined_file = ROOT / "aneeta_v2" / "Processed Data" / "dspy_dataset_combined.jsonl"
    
    print(f"✓ Loading combined dataset: {combined_file}")
    all_questions = []
    with open(combined_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_questions.append(json.loads(line))
    
    print(f"✓ Loaded {len(all_questions)} questions")
    
    if len(all_questions) > max_questions:
        random.seed(seed)
        all_questions = random.sample(all_questions, max_questions)
        print(f"✓ Sampled {max_questions} questions (seed={seed})")
    
    return all_questions

def format_question(q_data: Dict) -> str:
    """Format question with options."""
    q_text = q_data['question_text']
    options = q_data['options']
    
    # Handle both dict and list formats
    if isinstance(options, dict):
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    elif isinstance(options, list):
        # If it's a list, assume it's [A, B, C, D]
        option_letters = ['A', 'B', 'C', 'D']
        options_text = "\n".join([f"{letter}: {opt}" for letter, opt in zip(option_letters, options)])
    else:
        options_text = str(options)
    
    return f"{q_text}\n\nOptions:\n{options_text}"

def configure_dspy_ollama(model_name: str):
    """Configure DSPy to use Ollama model."""
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    lm = dspy.LM(
        model=f"ollama_chat/{model_name}",
        api_base=ollama_url,
        max_tokens=500,
        temperature=0.3
    )
    dspy.settings.configure(lm=lm)

def extract_answer_letter(response: str) -> str:
    """Extract answer letter from response."""
    import re
    patterns = [
        r'\b([A-D])\b',
        r'\(([A-D])\)',
        r'\*\*([A-D]):',
        r'answer\s+is\s+([A-D])',
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

def evaluate_config(model_config: Dict, is_optimized: bool, question: Dict) -> Dict:
    """Evaluate one configuration."""
    model_name = model_config['name']
    model_display = model_config['display']
    agent_type = "DSPy Optimized" if is_optimized else "DSPy Baseline"
    
    question_text = format_question(question)
    correct_answer = question['correct_answer']
    subject = question.get('subject', 'Biology')
    
    print(f"  Testing: {model_display} + {agent_type}... ", end='', flush=True)
    
    try:
        # ALWAYS reconfigure DSPy for each test to avoid caching
        configure_dspy_ollama(model_name)
        
        # Create NEW solver instance each time
        solver = MCQSolverModule()
        
        # Load optimized model if requested (AFTER creating solver)
        if is_optimized:
            opt_path = Path(model_config['optimized_path'])
            if opt_path.exists():
                solver.load(str(opt_path))
            else:
                print(f"⚠️ Model not found: {opt_path}")
        
        # Get prediction with timing
        t0 = time.time()
        prediction = solver(question=question_text, subject=subject)
        latency_ms = (time.time() - t0) * 1000
        
        # Ensure we actually got a response
        if latency_ms < 10 and is_optimized:
            # Suspicious - might be cached or error
            print(f"⚠️ Suspiciously fast ({latency_ms:.0f}ms) ", end='')
        
        response = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
        is_correct = validate_answer(response, correct_answer)
        
        status = "✓" if is_correct else "✗"
        print(f"{status} {latency_ms:.0f}ms")
        
        return {
            'question_id': question.get('question_id', ''),
            'subject': subject,
            'model': model_display,
            'agent_type': agent_type,
            'response': response,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'latency_ms': round(latency_ms, 1)
        }
        
    except Exception as e:
        import traceback
        error_msg = str(e)[:50]
        print(f"ERROR: {error_msg}")
        
        return {
            'question_id': question.get('question_id', ''),
            'subject': subject,
            'model': model_display,
            'agent_type': agent_type,
            'response': f"[Error: {str(e)}]",
            'correct_answer': correct_answer,
            'is_correct': False,
            'latency_ms': 0,
            'error_trace': traceback.format_exc()
        }

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-validation-set', action='store_true', default=True,
                        help='Use validation dataset (default: True)')
    parser.add_argument('--use-combined', action='store_true',
                        help='Use combined dataset instead of validation set')
    args = parser.parse_args()
    
    # Determine which dataset to use
    use_val_set = args.use_validation_set and not args.use_combined
    
    # Load questions
    print("\nLoading test questions...")
    questions = load_test_questions(args.test_samples, args.seed, use_validation_set=use_val_set)
    print(f"✓ Loaded {len(questions)} questions\n")
    
    print(f"Evaluating 3 models × 2 agent types = 6 configurations")
    print("="*70 + "\n")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"dspy_only_{len(questions)}q"):
        mlflow.log_param("num_questions", len(questions))
        mlflow.log_param("seed", args.seed)
        
        all_results = []
        
        # Evaluate each configuration
        for i, question in enumerate(questions, 1):
            print(f"\n[Question {i}/{len(questions)}] {question.get('subject', 'Unknown')}")
            print(f"{question['question_text'][:80]}...")
            print("-"*70)
            
            for model_config in MODELS:
                # Baseline
                result = evaluate_config(model_config, False, question)
                all_results.append(result)
                
                # Optimized
                result = evaluate_config(model_config, True, question)
                all_results.append(result)
        
        # Create summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70 + "\n")
        
        summary_data = []
        for model_config in MODELS:
            for agent_type in ["DSPy Baseline", "DSPy Optimized"]:
                config_results = [r for r in all_results 
                                 if r['model'] == model_config['display'] and r['agent_type'] == agent_type]
                
                if config_results:
                    correct = sum(1 for r in config_results if r['is_correct'])
                    accuracy = (correct / len(config_results)) * 100
                    avg_latency = sum(r['latency_ms'] for r in config_results) / len(config_results)
                    
                    summary_data.append({
                        'Model': model_config['display'],
                        'Agent_Type': agent_type,
                        'Accuracy_%': round(accuracy, 1),
                        'Correct': correct,
                        'Wrong': len(config_results) - correct,
                        'Avg_Latency_ms': round(avg_latency, 1)
                    })
        
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        
        # Save results
        df_detailed = pd.DataFrame(all_results)
        
        detailed_output = ROOT / "results" / "dspy_only_detailed_results.csv"
        summary_output = ROOT / "results" / "dspy_only_summary.csv"
        
        df_detailed.to_csv(detailed_output, index=False)
        df_summary.to_csv(summary_output, index=False)
        
        mlflow.log_artifact(str(detailed_output))
        mlflow.log_artifact(str(summary_output))
        
        # Log metrics
        for row in summary_data:
            prefix = f"{row['Model'].replace(' ', '_')}_{row['Agent_Type'].replace(' ', '_')}"
            mlflow.log_metric(f"{prefix}_accuracy", row['Accuracy_%'])
            mlflow.log_metric(f"{prefix}_latency", row['Avg_Latency_ms'])
        
        print(f"\n✓ Results saved:")
        print(f"  - {detailed_output}")
        print(f"  - {summary_output}")
        
        # Best config
        best_idx = df_summary['Accuracy_%'].idxmax()
        best_config = df_summary.iloc[best_idx]
        print(f"\n🏆 Best: {best_config['Model']} + {best_config['Agent_Type']}")
        print(f"   Accuracy: {best_config['Accuracy_%']}%")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
