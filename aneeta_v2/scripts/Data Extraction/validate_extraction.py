"""
Filter and validate extracted questions from process_questions_simple.py
Separates valid questions from image-based and incomplete extractions
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def validate_question(question: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate a single question
    
    Returns:
        (is_valid, reason) - True if valid, False with reason if invalid
    """
    metadata = question.get('metadata', {})
    
    # Check for image-based questions
    if metadata.get('is_image_based', False):
        return False, "image_based"
    
    # Check for incomplete extraction
    if metadata.get('has_incomplete_extraction', False):
        return False, "incomplete_extraction"
    
    # Check for required fields
    if not question.get('extracted_text'):
        return False, "missing_question_text"
    
    if not question.get('answer'):
        return False, "missing_answer"
    
    options = question.get('options', [])
    if len(options) < 4:
        return False, "insufficient_options"
    
    # Check if options have actual text (not placeholders)
    incomplete_count = 0
    empty_option_count = 0
    
    for opt in options:
        text = opt.get('text', '')
        if '[Image-based option' in text:
            return False, "placeholder_options"
        if not text.strip() or len(text.strip()) < 2:
            empty_option_count += 1
        if 'extraction incomplete' in text.lower():
            incomplete_count += 1
    
    # If 3+ options are empty or very short, it's likely image-based
    if empty_option_count >= 3:
        return False, "image_based"
    
    # If 2+ options are incomplete, likely image-based
    if incomplete_count >= 2:
        return False, "image_based"
    
    # Check for image-related keywords in question text
    question_text = question.get('extracted_text', '').lower()
    image_keywords = ['diagram', 'graph', 'figure', 'curve', 'plot', 'structure', 'following reaction']
    
    if any(keyword in question_text for keyword in image_keywords):
        # If question mentions images AND has 2+ incomplete options, it's image-based
        if incomplete_count >= 2 or empty_option_count >= 2:
            return False, "image_based"
    
    return True, "valid"


def filter_extraction_data(input_path: Path, output_dir: Path = None) -> Dict[str, Any]:
    """
    Filter extraction data into valid and invalid/image-based questions
    
    Args:
        input_path: Path to the raw extraction file (JSON or JSONL)
        output_dir: Directory to save filtered files (defaults to input file directory)
    
    Returns:
        Dict with statistics and output paths
    """
    if output_dir is None:
        output_dir = input_path.parent
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read input file
    print(f"Reading extraction data from: {input_path}")
    questions = []
    
    # Handle both JSON array and JSONL formats
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON array format
            questions = json.loads(content)
        else:
            # JSONL format (one JSON per line)
            f.seek(0)
            questions = [json.loads(line) for line in f if line.strip()]
    
    print(f"Total questions loaded: {len(questions)}")
    
    # Categorize questions
    valid_questions = []
    invalid_questions = []
    image_based_questions = []
    
    stats = {
        'total': len(questions),
        'valid': 0,
        'image_based': 0,
        'incomplete': 0,
        'missing_data': 0,
        'other_invalid': 0
    }
    
    for question in questions:
        is_valid, reason = validate_question(question)
        
        if is_valid:
            valid_questions.append(question)
            stats['valid'] += 1
        else:
            if reason == 'image_based':
                image_based_questions.append(question)
                stats['image_based'] += 1
            elif reason == 'incomplete_extraction':
                invalid_questions.append(question)
                stats['incomplete'] += 1
            elif reason in ['missing_question_text', 'missing_answer', 'insufficient_options']:
                invalid_questions.append(question)
                stats['missing_data'] += 1
            else:
                invalid_questions.append(question)
                stats['other_invalid'] += 1
    
    # Save filtered data
    base_name = input_path.stem
    
    # Save valid questions as JSONL
    valid_output = output_dir / f"{base_name}_valid.jsonl"
    print(f"\nSaving {len(valid_questions)} valid questions to: {valid_output}")
    with open(valid_output, 'w', encoding='utf-8') as f:
        for question in valid_questions:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')
    
    # Save invalid and image-based questions as JSONL
    invalid_output = output_dir / f"{base_name}_invalid_and_image.jsonl"
    combined_invalid = invalid_questions + image_based_questions
    print(f"Saving {len(combined_invalid)} invalid/image questions to: {invalid_output}")
    with open(invalid_output, 'w', encoding='utf-8') as f:
        for question in combined_invalid:
            f.write(json.dumps(question, ensure_ascii=False) + '\n')
    
    # Print statistics
    print("\n" + "=" * 70)
    print("VALIDATION STATISTICS")
    print("=" * 70)
    print(f"Total questions:        {stats['total']}")
    print(f"Valid questions:        {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"Image-based:            {stats['image_based']}")
    print(f"Incomplete extraction:  {stats['incomplete']}")
    print(f"Missing data:           {stats['missing_data']}")
    print(f"Other invalid:          {stats['other_invalid']}")
    print("=" * 70)
    
    return {
        'stats': stats,
        'valid_output': str(valid_output),
        'invalid_output': str(invalid_output)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Filter and validate extracted questions'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the raw extraction file (JSON or JSONL)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (defaults to input file directory)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    result = filter_extraction_data(input_path, output_dir)
    
    print(f"\n✓ Filtering complete!")
    print(f"Valid questions: {result['valid_output']}")
    print(f"Invalid questions: {result['invalid_output']}")
    
    return 0


if __name__ == '__main__':
    exit(main())
