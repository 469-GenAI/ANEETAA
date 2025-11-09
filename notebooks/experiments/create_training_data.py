"""
Create Training Data for DSPy Optimization

This script extracts training examples from test questions for DSPy optimization.
It formats questions in DSPy-compatible format with input/output pairs.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()


def load_all_questions(filter_visual: bool = True) -> List[Dict]:
    """Load all questions from Gemini 2.5 Pro Data."""
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
    
    return all_questions


def format_question(q_data: Dict) -> str:
    """Format question with options."""
    q_text = q_data['question_text']
    options = q_data.get('options', {})
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return f"{q_text}\n\nOptions:\n{options_text}"


def create_training_example(q_data: Dict) -> Dict:
    """Create a DSPy training example from a question."""
    question = format_question(q_data)
    correct_answer = q_data['correct_answer']
    
    # Create a good answer format (what we want DSPy to learn)
    answer = f"**Answer: {correct_answer}**\n\nThis is the correct answer to the MCQ question."
    
    return {
        'question': question,
        'answer': answer,
        'metadata': {
            'question_id': q_data.get('question_id', ''),
            'subject': q_data.get('metadata', {}).get('subject', 'unknown'),
            'correct_answer': correct_answer
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Create training data for DSPy optimization')
    parser.add_argument('--num-examples', type=int, default=50, help='Number of training examples')
    parser.add_argument('--output-file', default='dspy_training_data.json', help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    print("="*70)
    print("CREATING DSPy TRAINING DATA")
    print("="*70 + "\n")
    
    # Load all questions
    all_questions = load_all_questions(filter_visual=True)
    
    # Filter questions with correct_answer field
    valid_questions = [q for q in all_questions if 'question_text' in q and 'correct_answer' in q]
    print(f"✓ Found {len(valid_questions)} valid questions for training")
    
    # Sample training examples
    if len(valid_questions) < args.num_examples:
        print(f"⚠ Only {len(valid_questions)} questions available, using all")
        training_questions = valid_questions
    else:
        random.seed(args.seed)
        training_questions = random.sample(valid_questions, args.num_examples)
        print(f"✓ Sampled {args.num_examples} training questions (seed={args.seed})")
    
    # Create training examples
    training_examples = [create_training_example(q) for q in training_questions]
    
    # Save to file
    output_path = ROOT / args.output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, indent=2)
    
    print(f"\n✓ Training data saved to: {output_path}")
    print(f"✓ Total examples: {len(training_examples)}")
    
    # Show sample
    print(f"\nSample training example:")
    print("-"*70)
    sample = training_examples[0]
    print(f"Question: {sample['question'][:150]}...")
    print(f"Answer: {sample['answer']}")
    print(f"Subject: {sample['metadata']['subject']}")
    
    print("\n" + "="*70)
    print("TRAINING DATA CREATION COMPLETE")
    print("="*70)
    print(f"\nNext step: Run DSPy optimization")
    print(f"python notebooks/dspy_optimization.py --training-data {args.output_file}")


if __name__ == "__main__":
    main()
