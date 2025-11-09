"""
Create a larger training set from Gemini 2.5 Pro Data for DSPy optimization

This extracts 100 examples (balanced across subjects) from the processed data.
"""

import json
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)

# Load data from Gemini 2.5 Pro Data folder
gemini_dir = Path(__file__).parent.parent / "aneeta_v2" / "Processed Data" / "Gemini 2.5 Pro Data"

print(f"Loading questions from: {gemini_dir}")

all_questions = []
for json_file in gemini_dir.glob("*.json"):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for q in data:
            # Skip visual questions
            if q.get('metadata', {}).get('isVisualQuestion', False):
                continue
            all_questions.append(q)

print(f"Total non-visual questions: {len(all_questions)}")

# Balance by subject
physics_qs = [q for q in all_questions if q.get('metadata', {}).get('subject') == 'Physics']
chemistry_qs = [q for q in all_questions if q.get('metadata', {}).get('subject') == 'Chemistry']
biology_qs = [q for q in all_questions if q.get('metadata', {}).get('subject') == 'Biology']

print(f"Physics: {len(physics_qs)}")
print(f"Chemistry: {len(chemistry_qs)}")
print(f"Biology: {len(biology_qs)}")

# Sample 34 from each subject (102 total)
selected = []
selected.extend(random.sample(physics_qs, min(34, len(physics_qs))))
selected.extend(random.sample(chemistry_qs, min(34, len(chemistry_qs))))
selected.extend(random.sample(biology_qs, min(34, len(biology_qs))))

random.shuffle(selected)

print(f"\nSelected {len(selected)} examples")

# Convert to DSPy training format
training_examples = []
for q in selected:
    options = q.get('options', {})
    metadata = q.get('metadata', {})
    
    # Format question with options
    question_text = f"""Question: {q['question_text']}

A) {options.get('A', 'N/A')}
B) {options.get('B', 'N/A')}
C) {options.get('C', 'N/A')}
D) {options.get('D', 'N/A')}"""
    
    training_examples.append({
        'question': question_text,
        'metadata': {
            'subject': metadata.get('subject', 'Unknown'),
            'correct_answer': metadata.get('correctAnswer', 'A'),
            'question_id': q.get('questionId', 'unknown'),
            'difficulty': metadata.get('difficulty', 'medium')
        }
    })

# Save to file
output_file = Path(__file__).parent.parent / "dspy_training_data_100.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(training_examples, f, indent=2, ensure_ascii=False)

print(f"\n✓ Saved {len(training_examples)} examples to: {output_file}")
print("\nNow run optimization with:")
print("  python optimize_both_methods.py --train-size 60 --test-size 40 --training-data ../dspy_training_data_100.json")
