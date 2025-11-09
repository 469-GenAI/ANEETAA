"""
Simple MCQ Evaluation Script for ANEETAA
Tests vanilla ANEETAA MCQ solver on Gemini 2.5 Pro test questions
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List
import time
import mlflow

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
load_dotenv()

# Import ANEETAA components
from aneeta.nodes.agents import mcq_question_solver_agent
from aneeta.state.models import State

# Setup MLflow
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns'))
mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME', 'aneetaa-vanilla-vs-dspy'))

def load_test_questions(max_questions: int = 10) -> List[Dict]:
    """Load test questions from Gemini 2.5 Pro Data."""
    data_dir = Path(__file__).parent.parent / 'aneeta_v2' / 'Processed Data' / 'Gemini 2.5 Pro Data'
    
    test_questions = []
    json_files = sorted(data_dir.glob('*.json'))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                
                for item in file_data:
                    if 'question_text' in item and 'correct_answer' in item:
                        test_questions.append({
                            'id': item.get('question_id', ''),
                            'question': item['question_text'],
                            'options': item.get('options', []),
                            'correct': item['correct_answer'],
                            'subject': item.get('metadata', {}).get('subject', 'unknown')
                        })
                        
                        if len(test_questions) >= max_questions:
                            break
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
        
        if len(test_questions) >= max_questions:
            break
    
    print(f"✓ Loaded {len(test_questions)} test questions")
    return test_questions


def evaluate_question(question_data: Dict) -> Dict:
    """Evaluate ANEETAA on a single question."""
    from langchain_core.messages import HumanMessage
    
    # Format question with options
    question_text = question_data['question']
    options = question_data['options']
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    full_question = f"{question_text}\n\nOptions:\n{options_text}"
    
    # Create state - must include messages field (required by State TypedDict)
    state = State(
        messages=[HumanMessage(content=full_question)],
        user_explanation_language="English",
        agent_routing="mcq_question_solver",
        teacher_vectordb_routing="physics",  # default, will be determined by agent
        response_stream=None
    )
    
    # Run MCQ solver
    try:
        start_time = time.time()
        result = mcq_question_solver_agent(state)
        elapsed_time = time.time() - start_time
        
        # Extract answer from result
        # The agent returns a State dict with 'response_stream' generator
        if isinstance(result, dict) and 'response_stream' in result:
            # Consume the generator to get the full response
            response_parts = list(result['response_stream'])
            agent_answer = ''.join(response_parts)
        elif hasattr(result, 'selected_answer'):
            agent_answer = result.selected_answer
        elif hasattr(result, 'response'):
            agent_answer = str(result.response)
        else:
            agent_answer = str(result)
        
        # Check if correct - look for answer letter in the response
        correct_answer = question_data['correct']
        is_correct = correct_answer.upper() in agent_answer.upper()
        
        return {
            'question_id': question_data['id'],
            'subject': question_data['subject'],
            'agent_answer': agent_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'time_taken': elapsed_time
        }
        
    except Exception as e:
        print(f"❌ Error on {question_data['id']}: {e}")
        return {
            'question_id': question_data['id'],
            'subject': question_data['subject'],
            'agent_answer': 'ERROR',
            'correct_answer': question_data['correct'],
            'is_correct': False,
            'time_taken': 0,
            'error': str(e)
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ANEETAA MCQ Solver')
    parser.add_argument('--test-samples', type=int, default=10, help='Number of test questions')
    args = parser.parse_args()
    
    print("="*70)
    print("ANEETAA MCQ SOLVER EVALUATION")
    print("="*70)
    print()
    
    # Load test data
    print("Loading test questions...")
    questions = load_test_questions(max_questions=args.test_samples)
    
    if not questions:
        print("❌ No test questions loaded!")
        return
    
    print(f"\nEvaluating on {len(questions)} questions...")
    print("-"*70)
    
    # Evaluate each question
    results = []
    correct_count = 0
    
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q['subject']} - {q['id']}")
        print(f"Question: {q['question'][:80]}...")
        
        result = evaluate_question(q)
        results.append(result)
        
        if result['is_correct']:
            correct_count += 1
            print(f"✓ CORRECT - Answer: {result['correct_answer']} ({result['time_taken']:.1f}s)")
        else:
            print(f"✗ WRONG - Got: {result['agent_answer'][:50]}... | Expected: {result['correct_answer']}")
    
    # Calculate metrics
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    accuracy = (correct_count / len(questions)) * 100
    avg_time = sum(r['time_taken'] for r in results) / len(results)
    
    print(f"\nTotal Questions: {len(questions)}")
    print(f"Correct: {correct_count}")
    print(f"Wrong: {len(questions) - correct_count}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Average Time: {avg_time:.2f}s per question")
    
    # By subject
    print("\nBreakdown by Subject:")
    subjects = {}
    for r in results:
        subj = r['subject']
        if subj not in subjects:
            subjects[subj] = {'correct': 0, 'total': 0}
        subjects[subj]['total'] += 1
        if r['is_correct']:
            subjects[subj]['correct'] += 1
    
    for subj, stats in subjects.items():
        acc = (stats['correct'] / stats['total']) * 100
        print(f"  {subj}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    # Log to MLflow
    with mlflow.start_run(run_name=f"vanilla_mcq_eval_{len(questions)}_samples"):
        # Log parameters
        mlflow.log_param("agent_type", "vanilla")
        mlflow.log_param("num_questions", len(questions))
        mlflow.log_param("model", os.getenv("MODEL", "unknown"))
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("correct", correct_count)
        mlflow.log_metric("wrong", len(questions) - correct_count)
        mlflow.log_metric("avg_time_seconds", avg_time)
        
        # Log per-subject metrics
        for subj, stats in subjects.items():
            subj_acc = (stats['correct'] / stats['total']) * 100
            mlflow.log_metric(f"{subj.lower()}_accuracy", subj_acc)
            mlflow.log_metric(f"{subj.lower()}_correct", stats['correct'])
            mlflow.log_metric(f"{subj.lower()}_total", stats['total'])
        
        # Save and log detailed results
        output_file = Path(__file__).parent.parent / 'evaluation_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total': len(questions),
                    'correct': correct_count,
                    'accuracy': accuracy,
                    'avg_time': avg_time
                },
                'by_subject': subjects,
                'detailed_results': results
            }, f, indent=2)
        
        mlflow.log_artifact(str(output_file))
        print(f"\n✓ Results saved to: {output_file}")
        print(f"✓ Results logged to MLflow experiment: {os.getenv('MLFLOW_EXPERIMENT_NAME', 'aneetaa-vanilla-vs-dspy')}")
    
    print("="*70)


if __name__ == "__main__":
    main()
