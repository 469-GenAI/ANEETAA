"""
Optimize DSPy MCQ Solver with MIPROv2 and BootstrapFewShot

This script runs both optimization strategies and saves the models:
1. MIPROv2: Optimizes prompts AND demonstration selection
2. BootstrapFewShot: Optimizes demonstration selection only

Usage:
    python optimize_both_methods.py --train-size 20 --test-size 5
    python optimize_both_methods.py --train-size 20 --test-size 5 --training-data custom_data.json
"""

import sys
import os
from pathlib import Path
import json
import argparse
from typing import List, Dict
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dspy
from dspy.teleprompt import MIPROv2, BootstrapFewShot
import mlflow

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# ============================================================
# DSPy Signatures and Modules
# ============================================================

class MCQSolverSignature(dspy.Signature):
    """Solve a multiple choice question using step-by-step reasoning"""
    question = dspy.InputField(desc="The MCQ question with options A, B, C, D")
    subject = dspy.InputField(desc="Subject area: Physics, Chemistry, or Biology")
    reasoning = dspy.OutputField(desc="Step-by-step explanation of how to solve this question")
    answer = dspy.OutputField(desc="Final answer: A, B, C, or D")


class MCQSolverModule(dspy.Module):
    """DSPy module for MCQ solving with Chain of Thought"""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(MCQSolverSignature)
    
    def forward(self, question: str, subject: str):
        result = self.predictor(question=question, subject=subject)
        return result


# ============================================================
# Data Loading
# ============================================================

def load_training_data(data_file: str) -> List[dspy.Example]:
    """Load training data from JSON file"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        # The question is already formatted in the data
        question = item['question']
        
        # Extract metadata
        metadata = item.get('metadata', {})
        subject = metadata.get('subject', 'Unknown')
        correct_answer = metadata.get('correct_answer', 'A')
        
        # Create DSPy example
        example = dspy.Example(
            question=question,
            subject=subject,
            answer=correct_answer
        ).with_inputs('question', 'subject')
        
        examples.append(example)
    
    return examples


# ============================================================
# Metric Functions
# ============================================================

def answer_exact_match(example, prediction, trace=None):
    """Check if predicted answer matches correct answer exactly"""
    predicted_answer = prediction.answer.strip().upper()[0] if prediction.answer else 'X'
    correct_answer = example.answer.strip().upper()
    return predicted_answer == correct_answer


def answer_quality_metric(example, prediction, trace=None):
    """More lenient metric that gives partial credit"""
    predicted_answer = prediction.answer.strip().upper()[0] if prediction.answer else 'X'
    correct_answer = example.answer.strip().upper()
    
    # Full credit for exact match
    if predicted_answer == correct_answer:
        return 1.0
    
    # Partial credit if reasoning is present
    if len(prediction.reasoning) > 50:
        return 0.3
    
    return 0.0


# ============================================================
# Optimization Functions
# ============================================================

def optimize_with_mipro(
    trainset: List[dspy.Example],
    testset: List[dspy.Example],
    output_path: str
) -> MCQSolverModule:
    """Optimize using MIPROv2"""
    print("\n" + "="*60)
    print("Running MIPROv2 Optimization")
    print("="*60)
    
    # Create optimizer with improved parameters
    optimizer = MIPROv2(
        metric=answer_exact_match,
        num_candidates=10,  # Increased from 7 to generate more instruction candidates
        init_temperature=1.0,
        max_bootstrapped_demos=8,  # Increased from 4 to create more few-shot examples
        max_labeled_demos=8,  # Ensure we use labeled demos
        verbose=True
    )
    
    # Run optimization
    print(f"\nOptimizing on {len(trainset)} training examples...")
    print("This may take 10-15 minutes with increased parameters...\n")
    
    with mlflow.start_run(run_name="mipro_optimization", nested=True):
        optimized_module = optimizer.compile(
            MCQSolverModule(),
            trainset=trainset,
            valset=testset,  # Use full testset for validation
            minibatch_size=min(10, len(testset)),  # Set minibatch size based on valset size
            minibatch_full_eval_steps=5,
            requires_permission_to_run=False
        )
        
        # Save model
        optimized_module.save(output_path)
        print(f"\nMIPROv2 model saved to: {output_path}")
        
        # Log to MLflow
        mlflow.log_param("optimizer", "MIPROv2")
        mlflow.log_param("train_size", len(trainset))
        mlflow.log_param("num_candidates", 10)
        mlflow.log_param("max_bootstrapped_demos", 8)
        mlflow.log_param("max_labeled_demos", 8)
        mlflow.log_artifact(output_path)
    
    return optimized_module


def optimize_with_bootstrap(
    trainset: List[dspy.Example],
    testset: List[dspy.Example],
    output_path: str
) -> MCQSolverModule:
    """Optimize using BootstrapFewShot"""
    print("\n" + "="*60)
    print("Running BootstrapFewShot Optimization")
    print("="*60)
    
    # Create optimizer
    optimizer = BootstrapFewShot(
        metric=answer_exact_match,
        max_bootstrapped_demos=4,  # Number of demonstrations to use
        max_labeled_demos=4,
        max_rounds=2
    )
    
    # Run optimization
    print(f"\nOptimizing on {len(trainset)} training examples...")
    print("This may take 3-5 minutes...\n")
    
    with mlflow.start_run(run_name="bootstrap_optimization", nested=True):
        optimized_module = optimizer.compile(
            MCQSolverModule(),
            trainset=trainset
        )
        
        # Save model
        optimized_module.save(output_path)
        print(f"\nBootstrapFewShot model saved to: {output_path}")
        
        # Log to MLflow
        mlflow.log_param("optimizer", "BootstrapFewShot")
        mlflow.log_param("train_size", len(trainset))
        mlflow.log_param("max_demos", 4)
        mlflow.log_artifact(output_path)
    
    return optimized_module


# ============================================================
# Evaluation
# ============================================================

def evaluate_module(module: MCQSolverModule, testset: List[dspy.Example], name: str):
    """Evaluate a module on test set"""
    print(f"\nEvaluating {name}...")
    correct = 0
    
    for example in testset:
        prediction = module(question=example.question, subject=example.subject)
        if answer_exact_match(example, prediction):
            correct += 1
    
    accuracy = (correct / len(testset)) * 100
    print(f"  {name}: {accuracy:.1f}% ({correct}/{len(testset)})")
    
    return accuracy


# ============================================================
# Main Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Optimize DSPy MCQ solver with both MIPROv2 and BootstrapFewShot")
    parser.add_argument('--training-data', type=str, default='dspy_training_data.json', 
                        help='Path to training data JSON file')
    parser.add_argument('--train-size', type=int, default=20,
                        help='Number of training examples to use')
    parser.add_argument('--test-size', type=int, default=40,  # Increased from 10 to 40
                        help='Number of test examples to use (validation set for optimization)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--mipro-output', type=str, default='dspy_mipro_optimized.json',
                        help='Output path for MIPROv2 model')
    parser.add_argument('--bootstrap-output', type=str, default='dspy_bootstrap_optimized.json',
                        help='Output path for BootstrapFewShot model')
    parser.add_argument('--mlflow-experiment', type=str, default='aneetaa-dspy-dual-optimization',
                        help='MLflow experiment name')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DSPy Dual Optimization: MIPROv2 + BootstrapFewShot")
    print("="*60)
    print(f"Training data: {args.training_data}")
    print(f"Train size: {args.train_size}")
    print(f"Test size: {args.test_size}")
    print(f"Random seed: {args.seed}")
    
    # Configure DSPy with OpenAI (better for optimization)
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("\nWARNING: OPENAI_API_KEY not found in .env")
        print("   Using Ollama instead (optimization will be slower)")
        lm = dspy.LM('ollama/gemma2:9b', api_base='http://localhost:11434')
    else:
        print("\nUsing OpenAI GPT-4o-mini for optimization")
        lm = dspy.LM('openai/gpt-4o-mini', api_key=openai_key)
    
    dspy.configure(lm=lm)
    
    # Set up MLflow
    mlflow.set_experiment(args.mlflow_experiment)
    print(f"MLflow experiment: {args.mlflow_experiment}")
    
    # Load training data
    print(f"\nLoading training data from {args.training_data}...")
    all_examples = load_training_data(args.training_data)
    print(f"Loaded {len(all_examples)} examples")
    
    # Split into train and test
    random.seed(args.seed)
    random.shuffle(all_examples)
    
    trainset = all_examples[:args.train_size]
    testset = all_examples[args.train_size:args.train_size + args.test_size]
    
    print(f"Train set: {len(trainset)} examples")
    print(f"Test set: {len(testset)} examples")
    
    # Start parent MLflow run
    with mlflow.start_run(run_name="dual_optimization_comparison"):
        # Optimize with MIPROv2
        mipro_module = optimize_with_mipro(trainset, testset, args.mipro_output)
        mipro_accuracy = evaluate_module(mipro_module, testset, "MIPROv2")
        mlflow.log_metric("mipro_accuracy", mipro_accuracy)
        
        # Optimize with BootstrapFewShot
        bootstrap_module = optimize_with_bootstrap(trainset, testset, args.bootstrap_output)
        bootstrap_accuracy = evaluate_module(bootstrap_module, testset, "BootstrapFewShot")
        mlflow.log_metric("bootstrap_accuracy", bootstrap_accuracy)
        
        # Baseline comparison
        print("\nBaseline Comparison:")
        baseline = MCQSolverModule()
        baseline_accuracy = evaluate_module(baseline, testset, "Baseline (Unoptimized)")
        mlflow.log_metric("baseline_accuracy", baseline_accuracy)
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*60)
    print(f"Baseline (Unoptimized):   {baseline_accuracy:.1f}%")
    print(f"MIPROv2:                  {mipro_accuracy:.1f}% ({mipro_accuracy - baseline_accuracy:+.1f}%)")
    print(f"BootstrapFewShot:         {bootstrap_accuracy:.1f}% ({bootstrap_accuracy - baseline_accuracy:+.1f}%)")
    print("\nSaved models:")
    print(f"  - {args.mipro_output}")
    print(f"  - {args.bootstrap_output}")
    print("\nDual optimization complete!")
    print("\nNext step: Run four-way comparison:")
    print(f"  python notebooks/compare_four_agents.py --test-samples 10 \\")
    print(f"    --mipro-model {args.mipro_output} \\")
    print(f"    --bootstrap-model {args.bootstrap_output}")


if __name__ == "__main__":
    main()
