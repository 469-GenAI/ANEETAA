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
from langchain_community.vectorstores import Chroma
from openai import OpenAI


def setup_paths():
    """Add src to Python path for ANEETAA imports."""
    src_path = Path(__file__).parent.parent / 'src'
    sys.path.insert(0, str(src_path))
    print(f"✓ Added to path: {src_path}")


def load_vector_stores():
    """
    Load ANEETAA vector stores for realistic RAG-based training.
    
    Returns:
        Dictionary of vector stores by subject
    """
    from aneeta.embeddingmodel.embedding import get_embeddings
    
    # Get embeddings model
    embeddings = get_embeddings()
    
    # Vector DB configuration
    VECTOR_DB_CONFIGS = {
        'biology': 'chroma_vector_db_biology_nomic',
        'chemistry': 'chroma_vector_db_chemistry_nomic',
        'physics': 'chroma_vector_db_physics_nomic',
        'question_bank': 'chroma_vector_db_questionbank_nomic',
        'mentor': 'chroma_vector_db_mentor_nomic'
    }
    
    base_path = Path(__file__).parent.parent / 'src' / 'aneeta' / 'vectordb'
    vector_stores = {}
    
    print("\nLoading vector databases...")
    for subject, db_name in VECTOR_DB_CONFIGS.items():
        persist_dir = base_path / db_name
        if not persist_dir.exists():
            print(f"⚠ Warning: {subject} vector DB not found at {persist_dir}")
            continue
        
        try:
            vector_stores[subject] = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings
            )
            # Test the store
            count = vector_stores[subject]._collection.count()
            print(f"✓ Loaded {subject}: {count} documents")
        except Exception as e:
            print(f"❌ Error loading {subject}: {e}")
    
    if not vector_stores:
        raise RuntimeError("No vector stores loaded! Check your vectordb path.")
    
    return vector_stores


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
    
    print(f"✓ Loaded {len(examples)} training examples from JSON files")
    return examples


def load_neet_training_data_with_rag(
    vector_stores: Dict,
    max_examples_per_subject: int = 20,
    retrieval_k: int = 3
) -> List[dspy.Example]:
    """
    Load training data using RAG retrieval from vector databases.
    This creates more realistic training examples that match production usage.
    
    Args:
        vector_stores: Dictionary of loaded ChromaDB vector stores
        max_examples_per_subject: Maximum examples to generate per subject
        retrieval_k: Number of documents to retrieve for context
    
    Returns:
        List of DSPy examples with RAG-retrieved context
    """
    examples = []
    
    # Sample NEET-style questions by subject
    sample_questions = {
        'biology': [
            "What is the process of photosynthesis?",
            "Explain the structure and function of mitochondria",
            "What is DNA replication?",
            "Describe the process of cell division",
            "What are enzymes and how do they work?",
            "Explain the human circulatory system",
            "What is the role of ribosomes in protein synthesis?",
            "Describe the process of respiration in plants",
            "What are the differences between prokaryotic and eukaryotic cells?",
            "Explain the concept of natural selection",
            "What is the nitrogen cycle?",
            "Describe the structure of the human heart",
            "What are antibodies and their role in immunity?",
            "Explain the process of digestion",
            "What is the role of hormones in the human body?",
            "Describe the process of photosynthesis in detail",
            "What are chromosomes and genes?",
            "Explain the mechanism of muscle contraction",
            "What is the role of the nervous system?",
            "Describe the process of cellular respiration"
        ],
        'chemistry': [
            "What is atomic structure?",
            "Explain the periodic table organization",
            "What are chemical bonds?",
            "Describe the properties of acids and bases",
            "What is the mole concept?",
            "Explain oxidation and reduction reactions",
            "What are hydrocarbons?",
            "Describe the states of matter",
            "What is electrochemistry?",
            "Explain the concept of chemical equilibrium",
            "What are organic compounds?",
            "Describe the properties of metals and non-metals",
            "What is stoichiometry?",
            "Explain the gas laws",
            "What are coordination compounds?",
            "Describe the structure of benzene",
            "What is thermodynamics in chemistry?",
            "Explain the concept of pH",
            "What are polymers?",
            "Describe the process of electrolysis"
        ],
        'physics': [
            "What is Newton's first law of motion?",
            "Explain the concept of energy conservation",
            "What is electromagnetic induction?",
            "Describe the properties of waves",
            "What is the photoelectric effect?",
            "Explain Ohm's law",
            "What is gravitational force?",
            "Describe the concept of work and power",
            "What is the theory of relativity?",
            "Explain the concept of electric current",
            "What are the laws of thermodynamics?",
            "Describe the phenomenon of refraction",
            "What is the Doppler effect?",
            "Explain the concept of momentum",
            "What is magnetic field?",
            "Describe the structure of an atom",
            "What is radioactivity?",
            "Explain the concept of potential energy",
            "What are the properties of light?",
            "Describe the working of a transformer"
        ]
    }
    
    print("\nGenerating RAG-based training examples...")
    for subject, questions in sample_questions.items():
        if subject not in vector_stores:
            print(f"⚠ Skipping {subject} - vector store not available")
            continue
        
        vectorstore = vector_stores[subject]
        retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})
        
        # Generate examples using the first N questions
        for question in questions[:max_examples_per_subject]:
            try:
                # Retrieve relevant context from vector DB (like production agents do)
                retrieved_docs = retriever.invoke(question)
                
                if not retrieved_docs:
                    continue
                
                # Combine retrieved documents into context
                context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)
                
                # Create a reference answer from the first retrieved doc
                answer = retrieved_docs[0].page_content[:500] if retrieved_docs else ""
                
                examples.append(dspy.Example(
                    question=question,
                    context=context_str[:2000],  # Limit context length
                    answer=answer,
                    subject=subject
                ).with_inputs('question', 'context'))
                
            except Exception as e:
                print(f"⚠ Error creating example for '{question[:50]}...': {e}")
                continue
        
        print(f"✓ Generated {min(max_examples_per_subject, len(questions))} examples for {subject}")
    
    print(f"✓ Total RAG-based examples: {len(examples)}")
    return examples


def validate_explanation(example, prediction, trace=None) -> float:
    """
    Validate explanation quality using GPT-4o-mini as an LLM judge.
    
    Args:
        example: The test example with question, context, answer
        prediction: The agent's prediction/response
        trace: Optional trace information (not used)
    
    Returns:
        Score between 0.0 and 1.0
    """
    # Load environment to get OpenAI API key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        # Fallback to simple rule-based metric if no API key
        print("⚠ No OpenAI API key found, using rule-based metric")
        return validate_explanation_simple(example, prediction)
    
    # Get the response text
    response = prediction.response if hasattr(prediction, 'response') else str(prediction)
    
    # Quick sanity check
    if len(response) < 20:
        return 0.0
    
    # Use GPT-4o-mini as judge
    try:
        client = OpenAI(api_key=api_key)
        
        judge_prompt = f"""You are an expert evaluator for NEET (medical entrance exam) tutoring responses.

Question: {example.question}

Reference Context: {example.context[:1000]}

Student Level: NEET exam preparation (high school biology/chemistry/physics)

AI Tutor's Response: {response}

Please evaluate this response on a scale of 0.0 to 1.0 based on:
1. ACCURACY (30%): Uses correct information from the reference context
2. CLARITY (25%): Clear, easy to understand for NEET students
3. COMPLETENESS (25%): Fully addresses the question
4. PEDAGOGICAL QUALITY (20%): Good teaching approach, helps student learn

Return ONLY a single number between 0.0 and 1.0 (e.g., 0.85).
Do not include any explanation, just the number."""

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
        print(f"⚠ Error using LLM judge: {e}, falling back to rule-based metric")
        return validate_explanation_simple(example, prediction)


def validate_explanation_simple(example, prediction, trace=None) -> float:
    """
    Simple rule-based validation (fallback when LLM judge unavailable).
    
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
    parser.add_argument('--use-rag', action='store_true', default=True,
                        help='Use RAG-based training data from vector stores (recommended)')
    parser.add_argument('--use-json', action='store_true',
                        help='Use JSON file-based training data instead of RAG')
    parser.add_argument('--max-chunks', type=int, default=20,
                        help='Maximum chunks per subject to load (for JSON mode)')
    parser.add_argument('--max-examples', type=int, default=20,
                        help='Maximum examples per subject to generate (for RAG mode)')
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
    
    # Override use-rag if use-json is specified
    if args.use_json:
        args.use_rag = False
    
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
    if args.use_rag:
        print("\n" + "="*60)
        print("Loading Training Data with RAG (Production-like)")
        print("="*60)
        vector_stores = load_vector_stores()
        training_data = load_neet_training_data_with_rag(
            vector_stores,
            max_examples_per_subject=args.max_examples
        )
    else:
        print("\n" + "="*60)
        print("Loading Training Data from JSON Files")
        print("="*60)
        training_data = load_neet_training_data(max_chunks_per_subject=args.max_chunks)
    
    if not training_data:
        print("❌ No training data loaded. Exiting.")
        return
    
    # Split into train/test
    trainset, testset = train_test_split(training_data, test_size=0.2, random_state=42)
    print(f"\n✓ Data split: Train={len(trainset)} | Test={len(testset)}")
    
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
    print(f"Training Mode: {'RAG-based (Production-like)' if args.use_rag else 'JSON file-based'}")
    print(f"Baseline Score: {baseline_score:.2%}")
    print(f"Optimized Score: {optimized_score:.2%}")
    print(f"Improvement: {improvement:+.1f}%")
    print("\nNext steps:")
    print("1. View results in MLflow UI")
    print("2. Test the optimized agent with different questions")
    print("3. Optimize other agents (MCQ Solver, Mentor)")
    print("4. Expand training data for better results")
    if args.use_rag:
        print("\n💡 Tip: This optimization used RAG-retrieved context like production!")
        print("   The prompts are now optimized for your actual vector DB setup.")
    else:
        print("\n💡 Tip: Consider using --use-rag for more production-like training.")


if __name__ == "__main__":
    main()
