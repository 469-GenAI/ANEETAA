"""
Compare (OLD) Test_Extraction_Fixed_valid.jsonl with 
Test_Extraction_Fixed_valid.jsonl (NEW) to find which questions 
are in the NEW file that weren't in the OLD file.
These are questions that were added thanks to parser fixes and reprocessing.
"""

import json

# Load both files
print("Loading files...")
old_file = [json.loads(line) for line in open('aneeta_v2/Processed Data/(OLD) Test_Extraction_Fixed_valid.jsonl', 'r', encoding='utf-8')]
new_file = [json.loads(line) for line in open('aneeta_v2/Processed Data/Test_Extraction_Fixed_valid.jsonl', 'r', encoding='utf-8')]

print(f"OLD Test_Extraction_Fixed_valid: {len(old_file)} questions")
print(f"NEW Test_Extraction_Fixed_valid: {len(new_file)} questions")
print(f"Difference: {len(new_file) - len(old_file)} MORE questions in NEW file")

# Get IDs
old_ids = set(q['id'] for q in old_file)
new_ids = set(q['id'] for q in new_file)

# Find new questions (in new but not in old)
added_ids = new_ids - old_ids
added_questions = [q for q in new_file if q['id'] in added_ids]

# Also find questions that are in old but not in new (should be 0 ideally)
removed_ids = old_ids - new_ids
removed_questions = [q for q in old_file if q['id'] in removed_ids]

# Sort by source and question number for easier review
added_questions.sort(key=lambda x: (x['source'], int(x['question_number'])))

# Analyze the added questions
reprocessed_count = sum(1 for q in added_questions if q.get('metadata', {}).get('reprocessed', False))
parser_fixed_count = len(added_questions) - reprocessed_count

# Save to file
output_path = 'aneeta_v2/Processed Data/New_Questions_Added.jsonl'
with open(output_path, 'w', encoding='utf-8') as f:
    for q in added_questions:
        f.write(json.dumps(q, ensure_ascii=False) + '\n')

print(f"\n{'='*70}")
print(f"COMPARISON ANALYSIS")
print(f"{'='*70}")
print(f"Questions in OLD file:  {len(old_file)}")
print(f"Questions in NEW file:  {len(new_file)}")
print(f"Common questions:       {len(old_ids & new_ids)}")
print(f"New questions added:    {len(added_questions)}")
print(f"Questions removed:      {len(removed_questions)}")

if len(added_questions) > 0:
    print(f"\nNew Questions Breakdown:")
    print(f"  Reprocessed (Step 3): {reprocessed_count}")
    print(f"  Parser fixes:         {parser_fixed_count}")
    
    # Show sample of added questions
    print(f"\nSample of added questions (first 5):")
    for i, q in enumerate(added_questions[:5], 1):
        reprocessed = " [REPROCESSED]" if q.get('metadata', {}).get('reprocessed', False) else ""
        print(f"  {i}. {q['id']} - Q{q['question_number']} from {q['source']}{reprocessed}")

if len(removed_questions) > 0:
    print(f"\n⚠️  WARNING: {len(removed_questions)} questions were in OLD but NOT in NEW!")
    print(f"Sample of removed questions (first 5):")
    for i, q in enumerate(removed_questions[:5], 1):
        print(f"  {i}. {q['id']} - Q{q['question_number']} from {q['source']}")

print(f"\n✓ Saved {len(added_questions)} new questions to:")
print(f"  {output_path}")
print(f"{'='*70}")
