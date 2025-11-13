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
ROOT = Path(__file__).parent.parent.parent.parent.resolve()  # Go up to project root from runners/
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

def load_split_datasets(
    train_samples: int = 200,
    val_samples: int = 40,
    seed: int = 42
) -> tuple[List[Dict], List[Dict]]:
    """
    Load questions from the split train/val datasets.
    
    Args:
        train_samples: Number of questions to sample from train.jsonl
        val_samples: Number of questions to sample from val.jsonl
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_questions, val_questions)
    """
    train_file = ROOT / "aneeta_v2" / "Processed Data" / "dspy_dataset_train.jsonl"
    val_file = ROOT / "aneeta_v2" / "Processed Data" / "dspy_dataset_val.jsonl"
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_file}")
    
    print(f"Loading split datasets:")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")
    
    # Load training data
    train_questions = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            train_questions.append(json.loads(line))
    print(f"✓ Loaded {len(train_questions)} training questions")
    
    # Load validation data
    val_questions = []
    with open(val_file, 'r', encoding='utf-8') as f:
        for line in f:
            val_questions.append(json.loads(line))
    print(f"✓ Loaded {len(val_questions)} validation questions")
    
    # Sample from train
    if len(train_questions) > train_samples:
        random.seed(seed)
        train_questions = random.sample(train_questions, train_samples)
        print(f"✓ Sampled {train_samples} training questions (seed={seed})")
    
    # Sample from val
    if len(val_questions) > val_samples:
        random.seed(seed + 1)  # Different seed for val to avoid overlap
        val_questions = random.sample(val_questions, val_samples)
        print(f"✓ Sampled {val_samples} validation questions (seed={seed + 1})")
    
    return train_questions, val_questions


def load_gemini_processed_questions(
    data_dir: Path = None,
    max_questions: int = 100,
    filter_visual: bool = True,
    seed: int = 42,
    use_combined_dataset: bool = False
) -> List[Dict]:
    """
    Load questions from Gemini 2.5 Pro processed data.
    
    Args:
        data_dir: Directory containing Gemini data (01.json, 02.json, etc.)
        max_questions: Maximum number of questions to load
        filter_visual: If True, exclude questions with require_visuals=True
        seed: Random seed for reproducibility
        use_combined_dataset: If True, use dspy_dataset_combined.jsonl instead
        
    Returns:
        List of question dictionaries
    """
    # Option to use the combined dataset (7,874 questions)
    if use_combined_dataset:
        combined_file = ROOT / "aneeta_v2" / "Processed Data" / "dspy_dataset_combined.jsonl"
        if combined_file.exists():
            print(f"Loading combined dataset from: {combined_file}")
            all_questions = []
            with open(combined_file, 'r', encoding='utf-8') as f:
                for line in f:
                    all_questions.append(json.loads(line))
            print(f"✓ Total questions loaded: {len(all_questions)}")
            
            # Sample if needed
            if len(all_questions) > max_questions:
                random.seed(seed)
                all_questions = random.sample(all_questions, max_questions)
                print(f"✓ Sampled {max_questions} questions (seed={seed})")
            
            return all_questions
        else:
            print(f"⚠ Combined dataset not found at {combined_file}, falling back to original method")
    
    # Original loading method
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
    
    # Handle both dict and list formats for options
    if isinstance(options, dict):
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    elif isinstance(options, list):
        options_text = "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])
    else:
        options_text = ""
    
    return f"{q_text}\n\nOptions:\n{options_text}"


def create_dspy_examples(questions: List[Dict]) -> List[dspy.Example]:
    """
    Convert question dictionaries to DSPy examples.
    Works with both Gemini format and combined dataset format.
    
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
        
        # Extract subject - works for both formats
        # Combined dataset has 'subject' at top level
        # Gemini format has it in 'metadata'
        subject = q.get('subject') or q.get('metadata', {}).get('subject', 'Unknown')
        
        # Get explanation - combined dataset has richer structure
        explanation = q.get('explanation', {})
        
        # Handle both dict and string formats for explanation
        if isinstance(explanation, str):
            # If explanation is a string, use it directly as reasoning
            reasoning = explanation[:500] if explanation else "Solve step by step."
        elif isinstance(explanation, dict):
            # Try to get step-by-step reasoning from dict
            step_by_step = explanation.get('step_by_step', [])
            if isinstance(step_by_step, list) and step_by_step:
                # Combine steps into reasoning
                reasoning = "\n".join([
                    f"Step {s.get('step_number', i+1)}: {s.get('description', '')}"
                    for i, s in enumerate(step_by_step)
                ])
            else:
                # Fallback to summary or correct_option_reasoning
                reasoning = explanation.get('summary', '') or explanation.get('correct_option_reasoning', '')
        else:
            reasoning = "Solve step by step."
        
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
    optimized = optimizer.compile(
        baseline, 
        trainset=trainset,
        requires_permission_to_run=False  # Disable interactive confirmation
    )
    
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
    print(f"  To view UI: mlflow ui --port 8080")
    print(f"  Then open: http://localhost:8080\n")


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
    parser.add_argument('--use-combined', action='store_true', 
                        help='Use dspy_dataset_combined.jsonl (7,874 questions) instead of Gemini data')
    parser.add_argument('--use-split-datasets', action='store_true',
                        help='Use split train/val datasets instead of combined dataset')
    parser.add_argument('--train-samples', type=int, default=200, 
                        help='Number of samples from train.jsonl (only with --use-split-datasets)')
    parser.add_argument('--val-samples', type=int, default=40,
                        help='Number of samples from val.jsonl (only with --use-split-datasets)')
    
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
    
    if args.use_split_datasets:
        # Use the split train/val datasets
        train_questions, val_questions = load_split_datasets(
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            seed=args.seed
        )
        
        # Combine for training (all from train + all from val)
        all_questions = train_questions + val_questions
        print(f"\n✓ Total questions: {len(all_questions)} ({len(train_questions)} train + {len(val_questions)} val)")
        
        # Convert to DSPy examples
        examples = create_dspy_examples(all_questions)
        
        # Split: Use train_questions for training, val_questions for testing
        train_examples = create_dspy_examples(train_questions)
        test_examples = create_dspy_examples(val_questions)
        
        trainset = train_examples
        testset = test_examples
        
        print(f"✓ Data split: Train={len(trainset)} | Test={len(testset)}")
    else:
        # Original behavior
        questions = load_gemini_processed_questions(
            max_questions=args.questions,
            filter_visual=True,
            seed=args.seed,
            use_combined_dataset=args.use_combined  # Use the combined dataset if flag is set
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
            
            # Save model using the provided save_path argument
            # If save_path already includes 'models/', use it as is, otherwise prepend it
            if args.save_path.startswith('models/') or args.save_path.startswith('models\\'):
                save_path = ROOT / args.save_path
            else:
                save_path = ROOT / "models" / args.save_path
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
            
            # Save model using the provided save_path argument
            mipro_save_path = args.save_path.replace('bootstrap', 'mipro')
            if mipro_save_path.startswith('models/') or mipro_save_path.startswith('models\\'):
                save_path = ROOT / mipro_save_path
            else:
                save_path = ROOT / "models" / mipro_save_path
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
    
    print("\n✓ View results in MLflow UI: mlflow ui --port 8080")
    print("="*70)


if __name__ == "__main__":
    main()
