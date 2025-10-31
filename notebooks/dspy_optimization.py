"""
DSPy Optimization for ANEETAA Agents

This script demonstrates how to optimize ANEETAA agents using DSPy's SIMBA optimizer.
It can be run from the command line with various configuration options.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import dspy
import mlflow
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split


def setup_paths():
    """Add src to Python path for ANEETAA imports."""
    src_path = Path(__file__).parent.parent / 'src'
    sys.path.insert(0, str(src_path))
    print(f"✓ Added to path: {src_path}")


def configure_llm(provider: str = "openai", model: str = None):
    """
    Configure DSPy with the chosen LLM provider.
    
    Args:
        provider: 'openai' or 'ollama'
        model: Model name (e.g., 'gpt-4o-mini', 'llama3.1:8b')
    """
    if provider == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment. Set it in .env file.")
        
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
        print(f"✓ Using Ollama: {model_name} at {ollama_url}")
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    dspy.settings.configure(lm=lm)
    print(f"✓ DSPy configured with {lm.model}")
    return lm


def setup_mlflow():
    """Setup MLflow tracking with Databricks or local fallback."""
    load_dotenv()
    
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 
                                     f"file://{str(Path.cwd() / 'mlruns')}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Try to use Databricks experiment if configured
    exp_id = os.getenv('MLFLOW_EXPERIMENT_ID')
    if exp_id:
        try:
            client = MlflowClient()
            exp = client.get_experiment(exp_id)
            if exp is not None:
                mlflow.set_experiment(experiment_id=exp_id)
                print(f"✓ Using Databricks experiment: {exp.name} (ID: {exp_id})")
            else:
                print(f"Experiment ID {exp_id} not found, creating new experiment")
                mlflow.set_experiment('aneeta-dspy-optimization')
        except Exception as e:
            print(f"Error accessing experiment ID: {e}")
            mlflow.set_experiment('aneeta-dspy-optimization')
    else:
        mlflow.set_experiment('aneeta-dspy-optimization')
    
    # Enable DSPy autologging if available
    try:
        if hasattr(mlflow, 'dspy'):
            mlflow.dspy.autolog()
            print("✓ DSPy autolog enabled")
    except Exception as e:
        print(f"⚠ MLflow DSPy autolog not available: {e}")
    
    print(f"✓ MLflow configured - Tracking URI: {mlflow.get_tracking_uri()}")


def load_neet_training_data(data_dir: Path = None, max_chunks_per_subject: int = 20) -> List[dspy.Example]:
    """
    Load training data from processed NEET materials.
    
    Args:
        data_dir: Directory containing processed data files
        max_chunks_per_subject: Maximum number of chunks to load per subject
    
    Returns:
        List of DSPy examples
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'Processed Data'
    
    bio_path = data_dir / 'processed_biology_chunks.json'
    chem_path = data_dir / 'processed_chemistry_chunks.json'
    physics_path = data_dir / 'processed_physics_chunks.json'
    
    examples = []
    
    for path, subject in [(bio_path, 'biology'), (chem_path, 'chemistry'), (physics_path, 'physics')]:
        if not path.exists():
            print(f"⚠ Warning: {path} not found, skipping {subject}")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Convert chunks to Q&A format
        for chunk in chunks[:max_chunks_per_subject]:
            content = chunk.get('page_content', '')
            if len(content) > 100:
                sentences = content.split('.')
                if len(sentences) >= 2:
                    question = f"Explain: {sentences[0].strip()}"
                    answer = content
                    
                    examples.append(dspy.Example(
                        question=question,
                        context=content,
                        answer=answer[:500],
                        subject=subject
                    ).with_inputs('question', 'context'))
    
    print(f"✓ Loaded {len(examples)} training examples")
    return examples


def validate_explanation(example, prediction, trace=None) -> float:
    """
    Validate if explanation is good quality.
    
    Returns:
        Score between 0.0 and 1.0
    """
    response = prediction.response if hasattr(prediction, 'response') else str(prediction)
    
    if len(response) < 50:
        return 0.0
    
    # Check if it contains key terms from context
    context_words = set(example.context.lower().split())
    response_words = set(response.lower().split())
    overlap = len(context_words & response_words) / len(context_words) if context_words else 0
    
    return 1.0 if overlap > 0.1 else 0.0


def evaluate_agent(agent, testset: List[dspy.Example], name: str = "Agent", max_samples: int = 10) -> float:
    """
    Evaluate agent on test set.
    
    Args:
        agent: DSPy agent to evaluate
        testset: Test examples
        name: Name for logging
        max_samples: Maximum number of samples to evaluate
    
    Returns:
        Average score
    """
    scores = []
    
    for example in testset[:max_samples]:
        try:
            prediction = agent(
                question=example.question,
                context=example.context,
                language="English"
            )
            score = validate_explanation(example, prediction)
            scores.append(score)
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            scores.append(0)
    
    avg_score = np.mean(scores)
    print(f"{name} - Average Score: {avg_score:.2%}")
    return avg_score


def optimize_teacher_agent(
    trainset: List[dspy.Example],
    testset: List[dspy.Example],
    max_demos: int = 3,
    batch_size: int = 12,
    num_threads: int = 1,
    train_subset_size: int = 30
):
    """
    Optimize teacher agent using SIMBA optimizer.
    
    Args:
        trainset: Training examples
        testset: Test examples
        max_demos: Number of demonstrations to bootstrap
        batch_size: Batch size for evaluation
        num_threads: Number of parallel threads
        train_subset_size: Number of training examples to use
    
    Returns:
        Tuple of (optimized_agent, baseline_score, optimized_score, improvement)
    """
    # Import ANEETAA agents
    from aneeta.nodes.agents_dspy import TeacherAgentDSPy
    
    # Initialize baseline agent
    teacher_agent = TeacherAgentDSPy()
    print("✓ DSPy teacher agent initialized")
    
    # Configure SIMBA optimizer
    from dspy import SIMBA
    optimizer = SIMBA(
        metric=validate_explanation,
        max_demos=max_demos,
        bsize=batch_size,
        num_threads=num_threads
    )
    print("✓ SIMBA optimizer configured")
    
    # Run optimization
    print(f"Starting optimization on {train_subset_size} training examples...")
    print("This may take several minutes...")
    
    with mlflow.start_run(run_name="teacher_agent_optimization"):
        optimized_teacher = optimizer.compile(
            teacher_agent,
            trainset=trainset[:train_subset_size],
        )
        print("✓ Optimization complete!")
        
        # Evaluate both versions
        print("\nEvaluating agents...")
        baseline_score = evaluate_agent(TeacherAgentDSPy(), testset, "Baseline")
        optimized_score = evaluate_agent(optimized_teacher, testset, "Optimized")
        
        improvement = ((optimized_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
        print(f"\n🎉 Improvement: {improvement:.1f}%")
        
        # Log metrics to MLflow
        mlflow.log_metric("baseline_score", baseline_score)
        mlflow.log_metric("optimized_score", optimized_score)
        mlflow.log_metric("improvement_percent", improvement)
        mlflow.log_param("optimizer", "SIMBA")
        mlflow.log_param("max_demos", max_demos)
        mlflow.log_param("training_size", len(trainset))
        mlflow.log_param("train_subset_size", train_subset_size)
    
    return optimized_teacher, baseline_score, optimized_score, improvement


def test_optimized_agent(agent, question: str, context: str, language: str = "English"):
    """Test the optimized agent with a sample question."""
    print("\n" + "="*60)
    print("Testing Optimized Agent")
    print("="*60)
    print(f"Question: {question}")
    print(f"Language: {language}")
    
    result = agent(
        question=question,
        context=context,
        language=language
    )
    
    print(f"\nResponse:\n{result.response}")
    print("="*60)
    return result


def log_model_to_mlflow(optimized_teacher, baseline_score: float, optimized_score: float, improvement: float):
    """Log the optimized model to MLflow."""
    print("\nLogging model to MLflow...")
    
    with mlflow.start_run(run_name="teacher_agent_v1"):
        try:
            model_info = mlflow.dspy.log_model(
                optimized_teacher,
                artifact_path="teacher_agent",
                input_example="What is mitosis?"
            )
            
            # Log metrics
            mlflow.log_metric("baseline_score", baseline_score)
            mlflow.log_metric("optimized_score", optimized_score)
            mlflow.log_metric("improvement_percent", improvement)
            
            # Log parameters
            mlflow.log_param("optimizer", "SIMBA")
            mlflow.log_param("max_demos", 3)
            
            print("✓ Model logged to MLflow")
            print(f"  Model URI: {model_info.model_uri}")
            return model_info
        except Exception as e:
            print(f"⚠ Error logging model: {e}")
            return None


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='DSPy Optimization for ANEETAA Agents')
    parser.add_argument('--provider', type=str, default='openai', choices=['openai', 'ollama'],
                        help='LLM provider (openai or ollama)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (e.g., gpt-4o-mini, llama3.1:8b)')
    parser.add_argument('--max-chunks', type=int, default=20,
                        help='Maximum chunks per subject to load')
    parser.add_argument('--train-size', type=int, default=30,
                        help='Number of training examples to use for optimization')
    parser.add_argument('--max-demos', type=int, default=3,
                        help='Number of demonstrations to bootstrap')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='Batch size for evaluation')
    parser.add_argument('--test-samples', type=int, default=10,
                        help='Number of test samples to evaluate')
    parser.add_argument('--skip-model-log', action='store_true',
                        help='Skip logging model to MLflow')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DSPy Optimization for ANEETAA Agents")
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
    
    # Load training data
    print("\nLoading NEET training data...")
    training_data = load_neet_training_data(max_chunks_per_subject=args.max_chunks)
    
    if not training_data:
        print("❌ No training data loaded. Exiting.")
        return
    
    # Split into train/test
    trainset, testset = train_test_split(training_data, test_size=0.2, random_state=42)
    print(f"Train: {len(trainset)} | Test: {len(testset)}")
    
    # Optimize agent
    print("\n" + "="*60)
    print("Running SIMBA Optimization")
    print("="*60)
    optimized_teacher, baseline_score, optimized_score, improvement = optimize_teacher_agent(
        trainset=trainset,
        testset=testset,
        max_demos=args.max_demos,
        batch_size=args.batch_size,
        train_subset_size=args.train_size
    )
    
    # Test optimized agent
    test_optimized_agent(
        optimized_teacher,
        question="Explain the process of photosynthesis",
        context="Photosynthesis is the process by which green plants use sunlight to synthesize foods from carbon dioxide and water.",
        language="English"
    )
    
    # Log model to MLflow
    if not args.skip_model_log:
        log_model_to_mlflow(optimized_teacher, baseline_score, optimized_score, improvement)
    
    print("\n" + "="*60)
    print("✓ Optimization Complete!")
    print("="*60)
    print(f"Baseline Score: {baseline_score:.2%}")
    print(f"Optimized Score: {optimized_score:.2%}")
    print(f"Improvement: {improvement:+.1f}%")
    print("\nNext steps:")
    print("1. View results in MLflow UI")
    print("2. Test the optimized agent with different questions")
    print("3. Optimize other agents (MCQ Solver, Mentor)")
    print("4. Expand training data for better results")


if __name__ == "__main__":
    main()
