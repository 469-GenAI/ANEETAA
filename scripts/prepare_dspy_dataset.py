"""
Script to combine Gemini 2.5 Pro Data JSON files into a single dataset,
filter out questions requiring visual input, preprocess LaTeX notation,
and split into training/validation sets.
"""

import json
import os
from pathlib import Path
import random
import re
from typing import List, Dict

def preprocess_latex(text: str) -> str:
    """
    Clean and simplify LaTeX notation in text to make it more LLM-friendly.
    
    Args:
        text: Text containing LaTeX notation
        
    Returns:
        Text with simplified LaTeX
    """
    if not text or not isinstance(text, str):
        return text
    
    # Remove LaTeX delimiters but keep the content
    text = re.sub(r'\\?\\\(', '', text)  # Remove \( or \\(
    text = re.sub(r'\\?\\\)', '', text)  # Remove \) or \\)
    text = re.sub(r'\\?\\\[', '', text)  # Remove \[ or \\[
    text = re.sub(r'\\?\\\]', '', text)  # Remove \] or \\]
    
    # Remove dollar signs but keep content
    text = text.replace('$$', '')
    text = text.replace('$', '')
    
    # Simplify common LaTeX commands
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)  # \text{abc} -> abc
    text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)  # \mathrm{abc} -> abc
    text = re.sub(r'\\rm\{([^}]+)\}', r'\1', text)  # \rm{abc} -> abc
    
    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def preprocess_question(question: Dict, keep_all_fields: bool = False) -> Dict:
    """
    Preprocess a single question to clean LaTeX notation.
    
    Args:
        question: Question dictionary
        keep_all_fields: If True, keep all original fields. If False, only keep DSPy-relevant fields.
        
    Returns:
        Preprocessed question dictionary
    """
    # Create a copy to avoid modifying original
    processed = question.copy()
    
    # Clean question text
    if 'question_text' in processed:
        processed['question_text'] = preprocess_latex(processed['question_text'])
    
    # Clean options
    if 'options' in processed and isinstance(processed['options'], dict):
        processed['options'] = {
            key: preprocess_latex(val) 
            for key, val in processed['options'].items()
        }
    
    # Clean explanation fields if present
    if 'explanation' in processed and isinstance(processed['explanation'], dict):
        explanation = processed['explanation'].copy()
        
        # Clean summary
        if 'summary' in explanation:
            explanation['summary'] = preprocess_latex(explanation['summary'])
        
        # Clean step-by-step descriptions
        if 'step_by_step' in explanation and isinstance(explanation['step_by_step'], list):
            explanation['step_by_step'] = [
                {
                    **step,
                    'description': preprocess_latex(step.get('description', ''))
                }
                for step in explanation['step_by_step']
            ]
        
        # Clean reasoning fields
        if 'correct_option_reasoning' in explanation:
            explanation['correct_option_reasoning'] = preprocess_latex(
                explanation['correct_option_reasoning']
            )
        
        # Clean incorrect options analysis
        if 'incorrect_options_analysis' in explanation and isinstance(
            explanation['incorrect_options_analysis'], dict
        ):
            explanation['incorrect_options_analysis'] = {
                key: preprocess_latex(val)
                for key, val in explanation['incorrect_options_analysis'].items()
            }
        
        processed['explanation'] = explanation
    
    # If keeping only DSPy-relevant fields, filter out metadata
    if not keep_all_fields:
        # Keep only essential fields for DSPy
        dspy_question = {
            'question_id': processed.get('question_id'),
            'question_text': processed.get('question_text'),
            'question_type': processed.get('question_type'),
            'options': processed.get('options'),
            'correct_answer': processed.get('correct_answer'),
            'explanation': processed.get('explanation'),
        }
        
        # Optionally keep subject/topic for filtering during training
        if 'metadata' in processed:
            dspy_question['subject'] = processed['metadata'].get('subject')
            dspy_question['topic'] = processed['metadata'].get('topic')
            dspy_question['difficulty'] = processed['metadata'].get('difficulty')
        
        return dspy_question
    
    return processed

def load_all_json_files(data_dir: str) -> List[Dict]:
    """
    Load all JSON files from the Gemini 2.5 Pro Data directory.
    
    Args:
        data_dir: Path to the directory containing JSON files
        
    Returns:
        List of all questions from all files
    """
    all_questions = []
    data_path = Path(data_dir)
    
    # Get all JSON files and sort them numerically
    json_files = sorted(
        data_path.glob("*.json"),
        key=lambda x: int(x.stem) if x.stem.isdigit() else float('inf')
    )
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_questions.extend(data)
                else:
                    all_questions.append(data)
            print(f"Loaded {json_file.name}: {len(data) if isinstance(data, list) else 1} questions")
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
    
    return all_questions

def preprocess_all_questions(questions: List[Dict], keep_all_fields: bool = False) -> List[Dict]:
    """
    Preprocess all questions to clean LaTeX notation.
    
    Args:
        questions: List of all questions
        keep_all_fields: If True, keep all original fields. If False, only keep DSPy-relevant fields.
        
    Returns:
        List of preprocessed questions
    """
    print(f"Preprocessing {len(questions)} questions...")
    if not keep_all_fields:
        print(f"  → Keeping only DSPy-relevant fields (question, options, answer, explanation, subject/topic/difficulty)")
    processed = [preprocess_question(q, keep_all_fields) for q in questions]
    print(f"✓ Preprocessing complete")
    return processed

def filter_visual_questions(questions: List[Dict]) -> List[Dict]:
    """
    Filter out questions that require visual input.
    
    Args:
        questions: List of all questions
        
    Returns:
        List of questions with requires_visual = False
    """
    filtered = [
        q for q in questions 
        if q.get('metadata', {}).get('requires_visual', False) is False
    ]
    
    print(f"\nTotal questions: {len(questions)}")
    print(f"Questions requiring visual: {len(questions) - len(filtered)}")
    print(f"Questions without visual: {len(filtered)}")
    
    return filtered

def split_train_val(questions: List[Dict], train_ratio: float = 0.8, random_seed: int = 42) -> tuple:
    """
    Split questions into training and validation sets.
    
    Args:
        questions: List of all questions
        train_ratio: Proportion for training set (default 0.8 for 80/20 split)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_questions, val_questions)
    """
    # Shuffle with fixed seed for reproducibility
    random.seed(random_seed)
    shuffled = questions.copy()
    random.shuffle(shuffled)
    
    # Calculate split point
    split_idx = int(len(shuffled) * train_ratio)
    
    train_questions = shuffled[:split_idx]
    val_questions = shuffled[split_idx:]
    
    print(f"\nSplit into:")
    print(f"Training set: {len(train_questions)} questions ({train_ratio*100:.0f}%)")
    print(f"Validation set: {len(val_questions)} questions ({(1-train_ratio)*100:.0f}%)")
    
    return train_questions, val_questions

def save_jsonl(questions: List[Dict], output_path: str):
    """
    Save questions to a JSONL file (one JSON object per line).
    
    Args:
        questions: List of questions to save
        output_path: Path to output JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(questions)} questions to {output_path}")

def main():
    """Main execution function."""
    # Define paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "aneeta_v2" / "Processed Data" / "Gemini 2.5 Pro Data"
    output_dir = base_dir / "aneeta_v2" / "Processed Data"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DSPy Dataset Preparation")
    print("=" * 60)
    print("\nConfiguration:")
    print("  • LaTeX preprocessing: Enabled")
    print("  • Visual question filtering: Enabled")
    print("  • Field filtering: Enabled (DSPy-optimized)")
    print("  • Train/Val split: 80/20")
    
    # Step 1: Load all JSON files
    print("\n[1/5] Loading JSON files...")
    all_questions = load_all_json_files(str(data_dir))
    
    # Step 2: Preprocess LaTeX notation (keep only DSPy-relevant fields)
    print("\n[2/5] Preprocessing LaTeX notation...")
    preprocessed_questions = preprocess_all_questions(all_questions, keep_all_fields=False)
    
    # Step 3: Filter out visual questions
    print("\n[3/5] Filtering out visual questions...")
    filtered_questions = filter_visual_questions(preprocessed_questions)
    
    if len(filtered_questions) == 0:
        print("ERROR: No questions remaining after filtering!")
        return
    
    # Step 4: Split into train/val
    print("\n[4/5] Splitting into train/validation sets...")
    train_questions, val_questions = split_train_val(filtered_questions)
    
    # Step 5: Save to JSONL files
    print("\n[5/5] Saving to JSONL files...")
    
    # Save combined dataset (all non-visual questions)
    combined_path = output_dir / "dspy_dataset_combined.jsonl"
    save_jsonl(filtered_questions, str(combined_path))
    
    # Save training set
    train_path = output_dir / "dspy_dataset_train.jsonl"
    save_jsonl(train_questions, str(train_path))
    
    # Save validation set
    val_path = output_dir / "dspy_dataset_val.jsonl"
    save_jsonl(val_questions, str(val_path))
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  • Combined: {combined_path}")
    print(f"  • Training: {train_path}")
    print(f"  • Validation: {val_path}")
    print("\nDataset optimization:")
    print("  ✓ LaTeX cleaned (removed delimiters)")
    print("  ✓ Visual questions filtered out")
    print("  ✓ Only DSPy-relevant fields kept")
    print("  ✓ Metadata noise removed (estimated_time, source, tags, etc.)")
    print("\nReady for DSPy optimization!")

if __name__ == "__main__":
    main()
