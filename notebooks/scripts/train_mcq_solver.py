"""
Train DSPy MCQ Solver with ANEETAA Processed Data

This script trains DSPy MCQ solvers (baseline and optimized) using 100+ non-visual 
questions from the Gemini 2.5 Pro processed data.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any

import dspy
import mlflow
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Setup paths
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

load_dotenv()


# ============================================================================
# PART 1: MCQ Solver Signature & Module (matches optimized models)
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
        # Use 'predictor' to match saved model naming
        self.predictor = dspy.ChainOfThought(MCQSolverSignature)
    
    def forward(self, question: str, subject: str = "Biology") -> dspy.Prediction:
        """Solve MCQ question."""
        return self.predictor(question=question, subject=subject)


# ============================================================================
# PART 2: Data Loading
# ============================================================================

def load_gemini_processed_questions(
    data_dir: Path = None,
    max_questions: int = 100,
    filter_visual: bool = True,
    seed: int = 42
) -> List[Dict]:
    """
    Load questions from Gemini 2.5 Pro processed data.
    
    Args:
        data_dir: Directory containing Gemini data (01.json, 02.json, etc.)
        max_questions: Maximum number of questions to load
        filter_visual: If True, exclude questions with require_visuals=True
        seed: Random seed for reproducibility
        
    Returns:
        List of question dictionaries
    """
    if data_dir is None:
        data_dir = ROOT / "aneeta_v2" / "Processed Data" / "Gemini 2.5 Pro Data"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Loading questions from: {data_dir}")
    
    all_questions = []
    json_files = sorted(data_dir.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            # Filter if needed
            if filter_visual:
                questions = [q for q in questions if not q.get('require_visuals', False)]
            
            all_questions.extend(questions)
            print(f"  ✓ Loaded {len(questions)} questions from {json_file.name}")
            
        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")
    
    print(f"\n✓ Total questions loaded: {len(all_questions)}")
    
    if filter_visual:
        print(f"  (Filtered out visual questions)")
    
    # Sample if needed
    if len(all_questions) > max_questions:
        random.seed(seed)
        all_questions = random.sample(all_questions, max_questions)
        print(f"✓ Sampled {max_questions} questions (seed={seed})")
    
    return all_questions


def format_question_for_dspy(q_data: Dict) -> str:
    """Format question with options for DSPy input."""
    q_text = q_data['question_text']
    options = q_data.get('options', {})
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return f"{q_text}\n\nOptions:\n{options_text}"


def create_dspy_examples(questions: List[Dict]) -> List[dspy.Example]:
    """
    Convert question dictionaries to DSPy examples.
    
    Args:
        questions: List of question dictionaries
        
    Returns:
        List of DSPy examples with inputs/outputs
    """
    examples = []
    
    for q in questions:
        # Skip if missing essential fields
        if 'question_text' not in q or 'correct_answer' not in q:
            continue
        
        # Format question
        question_text = format_question_for_dspy(q)
        
        # Extract subject from metadata
        subject = q.get('metadata', {}).get('subject', 'Unknown')
        
        # Get explanation if available
        explanation = q.get('explanation', {})
        reasoning = explanation.get('step_by_step', '') or explanation.get('summary', '')
        
        # Create example
        example = dspy.Example(
            question=question_text,
            subject=subject,
            reasoning=reasoning[:500] if reasoning else "Solve step by step.",
            answer=q['correct_answer']
        ).with_inputs('question', 'subject')
        
        examples.append(example)
    
    print(f"\n✓ Created {len(examples)} DSPy training examples")
    return examples


# ============================================================================
# PART 3: Evaluation Metric
# ============================================================================

def mcq_accuracy_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    Evaluate if the predicted answer matches the correct answer.
    
    Args:
        example: DSPy example with ground truth
        prediction: Model prediction
        trace: Optional trace (not used)
        
    Returns:
        1.0 if correct, 0.0 if wrong
    """
    # Extract answer from prediction
    pred_answer = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
    
    # Extract just the letter (A/B/C/D)
    import re
    match = re.search(r'\b([A-D])\b', pred_answer.upper())
    if match:
        pred_letter = match.group(1)
    else:
        pred_letter = pred_answer.strip()[:1].upper()
    
    # Check if correct
    correct = (pred_letter == example.answer.upper())
    
    return 1.0 if correct else 0.0


def evaluate_model(model: MCQSolverModule, testset: List[dspy.Example], name: str = "Model") -> float:
    """
    Evaluate model on test set.
    
    Args:
        model: MCQ solver module
        testset: Test examples
        name: Name for logging
        
    Returns:
        Accuracy (0.0 to 1.0)
    """
    correct = 0
    total = 0
    
    print(f"\nEvaluating {name} on {len(testset)} questions...")
    
    for i, example in enumerate(testset):
        try:
            prediction = model(question=example.question, subject=example.subject)
            score = mcq_accuracy_metric(example, prediction)
            correct += score
            total += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(testset)} - Accuracy: {correct/total:.1%}")
                
        except Exception as e:
            print(f"  ✗ Error on question {i+1}: {e}")
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n✓ {name} Final Accuracy: {accuracy:.1%} ({int(correct)}/{total})")
    
    return accuracy


# ============================================================================
# PART 4: DSPy Configuration
# ============================================================================

def configure_dspy(provider: str = "openai", model: str = "gpt-4o-mini"):
    """Configure DSPy with LLM provider."""
    
    if provider == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        lm = dspy.LM(
            model=f"openai/{model}",
            max_tokens=500,
            temperature=0.1
        )
        print(f"✓ Using OpenAI: {model}")
        
    elif provider == "ollama":
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        lm = dspy.LM(
            model=f"ollama_chat/{model}",
            api_base=ollama_url,
            max_tokens=500,
            temperature=0.1
        )
        print(f"✓ Using Ollama: {model} at {ollama_url}")
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    dspy.settings.configure(lm=lm)
    print(f"✓ DSPy configured")
    
    return lm


# ============================================================================
# PART 5: Optimization
# ============================================================================

def train_baseline_model(trainset: List[dspy.Example], testset: List[dspy.Example]) -> MCQSolverModule:
    """
    Train baseline model (simple ChainOfThought, no optimization).
    
    Args:
        trainset: Training examples
        testset: Test examples
        
    Returns:
        Baseline MCQ solver
    """
    print("\n" + "="*70)
    print("BASELINE MODEL (No Optimization)")
    print("="*70)
    
    baseline = MCQSolverModule()
    accuracy = evaluate_model(baseline, testset, "Baseline")
    
    return baseline, accuracy


def train_bootstrap_model(
    trainset: List[dspy.Example],
    testset: List[dspy.Example],
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4
) -> MCQSolverModule:
    """
    Train using BootstrapFewShot optimizer.
    
    Args:
        trainset: Training examples
        testset: Test examples
        max_bootstrapped_demos: Max number of bootstrapped demos
        max_labeled_demos: Max number of labeled demos
        
    Returns:
        Optimized MCQ solver and accuracy
    """
    print("\n" + "="*70)
    print("BOOTSTRAP FEW-SHOT OPTIMIZATION")
    print("="*70)
    
    from dspy.teleprompt import BootstrapFewShot
    
    optimizer = BootstrapFewShot(
        metric=mcq_accuracy_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=1
    )
    
    print(f"\nOptimizing with {len(trainset)} training examples...")
    print("This may take several minutes...")
    
    baseline = MCQSolverModule()
    optimized = optimizer.compile(baseline, trainset=trainset)
    
    print("\n✓ Optimization complete!")
    
    accuracy = evaluate_model(optimized, testset, "Bootstrap Optimized")
    
    return optimized, accuracy


def train_mipro_model(
    trainset: List[dspy.Example],
    testset: List[dspy.Example],
    num_candidates: int = 7,
    init_temperature: float = 1.0
) -> MCQSolverModule:
    """
    Train using MIPRO optimizer (best results, slower).
    
    Args:
        trainset: Training examples
        testset: Test examples
        num_candidates: Number of instruction candidates
        init_temperature: Initial temperature for sampling
        
    Returns:
        Optimized MCQ solver and accuracy
    """
    print("\n" + "="*70)
    print("MIPRO OPTIMIZATION")
    print("="*70)
    
    from dspy.teleprompt import MIPROv2
    
    optimizer = MIPROv2(
        metric=mcq_accuracy_metric,
        num_candidates=num_candidates,
        init_temperature=init_temperature,
        verbose=True
    )
    
    print(f"\nOptimizing with {len(trainset)} training examples...")
    print("This may take 10-20 minutes...")
    
    baseline = MCQSolverModule()
    optimized = optimizer.compile(baseline, trainset=trainset)
    
    print("\n✓ Optimization complete!")
    
    accuracy = evaluate_model(optimized, testset, "MIPRO Optimized")
    
    return optimized, accuracy


# ============================================================================
# PART 6: MLflow Setup
# ============================================================================

def setup_mlflow():
    """Setup MLflow tracking (local file-based)."""
    mlflow_dir = Path.cwd() / 'mlruns'
    mlflow_dir.mkdir(exist_ok=True)
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment('mcq-solver-training')
    
    print(f"\n✓ MLflow tracking: {mlflow_dir}")
    print(f"  To view UI: mlflow ui --port 5000")
    print(f"  Then open: http://localhost:5000\n")


# ============================================================================
# PART 7: Main Training Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train DSPy MCQ Solver')
    parser.add_argument('--questions', type=int, default=100, help='Number of questions to load')
    parser.add_argument('--test-split', type=float, default=0.2, help='Test set split ratio')
    parser.add_argument('--provider', default='openai', choices=['openai', 'ollama'])
    parser.add_argument('--model', default='gpt-4o-mini', help='Model name')
    parser.add_argument('--method', default='bootstrap', choices=['baseline', 'bootstrap', 'mipro', 'all'],
                        help='Optimization method to use')
    parser.add_argument('--max-demos', type=int, default=4, help='Max demos for Bootstrap')
    parser.add_argument('--candidates', type=int, default=7, help='Candidates for MIPRO')
    parser.add_argument('--save-path', default='dspy_mcq_optimized.json', help='Path to save optimized model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("="*70)
    print("DSPy MCQ SOLVER TRAINING")
    print("="*70)
    
    # Load environment
    load_dotenv()
    
    # Setup MLflow
    setup_mlflow()
    
    # Configure DSPy
    print("\nConfiguring DSPy...")
    configure_dspy(args.provider, args.model)
    
    # Load data
    print("\n" + "="*70)
    print("LOADING TRAINING DATA")
    print("="*70)
    
    questions = load_gemini_processed_questions(
        max_questions=args.questions,
        filter_visual=True,
        seed=args.seed
    )
    
    # Convert to DSPy examples
    examples = create_dspy_examples(questions)
    
    # Split train/test
    trainset, testset = train_test_split(
        examples,
        test_size=args.test_split,
        random_state=args.seed
    )
    
    print(f"\n✓ Data split: Train={len(trainset)} | Test={len(testset)}")
    
    # Train models based on method
    results = {}
    
    if args.method in ['baseline', 'all']:
        with mlflow.start_run(run_name="baseline"):
            baseline, baseline_acc = train_baseline_model(trainset, testset)
            results['baseline'] = {'model': baseline, 'accuracy': baseline_acc}
            mlflow.log_metric("accuracy", baseline_acc)
            mlflow.log_param("method", "baseline")
            mlflow.log_param("train_size", len(trainset))
    
    if args.method in ['bootstrap', 'all']:
        with mlflow.start_run(run_name="bootstrap"):
            bootstrap, bootstrap_acc = train_bootstrap_model(
                trainset, testset,
                max_bootstrapped_demos=args.max_demos,
                max_labeled_demos=args.max_demos
            )
            results['bootstrap'] = {'model': bootstrap, 'accuracy': bootstrap_acc}
            mlflow.log_metric("accuracy", bootstrap_acc)
            mlflow.log_param("method", "bootstrap")
            mlflow.log_param("max_demos", args.max_demos)
            mlflow.log_param("train_size", len(trainset))
            
            # Save model
            save_path = ROOT / "models" / "dspy_bootstrap_optimized.json"
            bootstrap.save(str(save_path))
            print(f"\n✓ Saved Bootstrap model: {save_path}")
    
    if args.method in ['mipro', 'all']:
        with mlflow.start_run(run_name="mipro"):
            mipro, mipro_acc = train_mipro_model(
                trainset, testset,
                num_candidates=args.candidates
            )
            results['mipro'] = {'model': mipro, 'accuracy': mipro_acc}
            mlflow.log_metric("accuracy", mipro_acc)
            mlflow.log_param("method", "mipro")
            mlflow.log_param("num_candidates", args.candidates)
            mlflow.log_param("train_size", len(trainset))
            
            # Save model
            save_path = ROOT / "models" / "dspy_mipro_optimized.json"
            mipro.save(str(save_path))
            print(f"\n✓ Saved MIPRO model: {save_path}")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    for name, data in results.items():
        print(f"{name.upper():20s} Accuracy: {data['accuracy']:.1%}")
    
    if 'baseline' in results and len(results) > 1:
        baseline_acc = results['baseline']['accuracy']
        for name, data in results.items():
            if name != 'baseline':
                improvement = (data['accuracy'] - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
                print(f"\n{name.upper()} Improvement over Baseline: {improvement:+.1f}%")
    
    print("\n✓ View results in MLflow UI: mlflow ui --port 5000")
    print("="*70)


if __name__ == "__main__":
    main()
