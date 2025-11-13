"""
Re-judge all responses from CSV using GPT-4o with same criteria.

This script reads the existing CSV file with responses and re-evaluates 
all of them using GPT-4o as the judge model, applying the same criteria
as the original evaluation.
"""

import os
import sys
import csv
import json
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Setup
ROOT = Path(__file__).parent.parent.resolve()
load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ============================================================================
# Configuration
# ============================================================================

INPUT_CSV = ROOT / "3x4_matrix_detailed_results_Ben.csv"
OUTPUT_CSV = ROOT / "results" / "rejudged_gpt4o.csv"
JUDGE_MODEL = "gpt-4o"  # GPT-4o model

# Rate limiting
REQUESTS_PER_MINUTE = 30
DELAY_BETWEEN_REQUESTS = 60 / REQUESTS_PER_MINUTE  # ~2 seconds

# ============================================================================
# Judge Function
# ============================================================================

def analyze_response_quality(response: str) -> dict:
    """Comprehensive analysis of response quality including language, length, structure.
    
    Args:
        response: The model's answer/response
    
    Returns:
        Dictionary with comprehensive quality metrics
    """
    metrics = {}
    
    # ========== LANGUAGE DETECTION ==========
    # Detect NON-ENGLISH scripts by checking Unicode ranges
    # If ANY non-English script is found, response is not English-only
    
    non_english_scripts = []
    
    # Indian scripts (Brahmic family)
    script_ranges = {
        'Hindi (Devanagari)': r'[\u0900-\u097F]',
        'Bengali': r'[\u0980-\u09FF]',
        'Punjabi (Gurmukhi)': r'[\u0A00-\u0A7F]',
        'Gujarati': r'[\u0A80-\u0AFF]',
        'Oriya': r'[\u0B00-\u0B7F]',
        'Tamil': r'[\u0B80-\u0BFF]',
        'Telugu': r'[\u0C00-\u0C7F]',
        'Kannada': r'[\u0C80-\u0CFF]',
        'Malayalam': r'[\u0D00-\u0D7F]',
        'Sinhala': r'[\u0D80-\u0DFF]',
        'Thai': r'[\u0E00-\u0E7F]',
        'Lao': r'[\u0E80-\u0EFF]',
        'Tibetan': r'[\u0F00-\u0FFF]',
        'Myanmar': r'[\u1000-\u109F]',
        'Khmer': r'[\u1780-\u17FF]',
        
        # East Asian
        'Chinese (CJK)': r'[\u4E00-\u9FFF]',
        'Japanese (Hiragana)': r'[\u3040-\u309F]',
        'Japanese (Katakana)': r'[\u30A0-\u30FF]',
        'Korean (Hangul)': r'[\uAC00-\uD7AF]',
        
        # Middle Eastern & African
        'Arabic': r'[\u0600-\u06FF]',
        'Hebrew': r'[\u0590-\u05FF]',
        'Syriac': r'[\u0700-\u074F]',
        'Thaana': r'[\u0780-\u07BF]',
        'Ethiopic': r'[\u1200-\u137F]',
        
        # European & Slavic
        'Cyrillic (Russian, etc.)': r'[\u0400-\u04FF]',
        'Greek': r'[\u0370-\u03FF]',
        'Armenian': r'[\u0530-\u058F]',
        'Georgian': r'[\u10A0-\u10FF]',
        
        # Other
        'Mongolian': r'[\u1800-\u18AF]',
    }
    
    # Check each script
    for script_name, pattern in script_ranges.items():
        if re.search(pattern, response):
            non_english_scripts.append(script_name)
    
    # Check for explicit language markers in text
    language_markers = [
        'Explanation in Hindi',
        'Explanation in Tamil',
        'Explanation in Chinese',
        'Explanation in Japanese',
        'Объяснение',  # Russian: Explanation
        '解释',  # Chinese: Explanation
        '説明',  # Japanese: Explanation
    ]
    
    for marker in language_markers:
        if marker in response and not any(marker in s for s in non_english_scripts):
            non_english_scripts.append(f"{marker.split()[-1] if ' ' in marker else 'Foreign'} (marked)")
    
    # Determine if English-only
    is_english_only = len(non_english_scripts) == 0
    
    metrics["is_english_only"] = is_english_only
    metrics["languages_detected"] = ", ".join(non_english_scripts) if non_english_scripts else "English"
    metrics["multilingual_penalty"] = len(non_english_scripts) * 1.0  # 1 point penalty per non-English script
    
    # ========== RESPONSE LENGTH & COMPLETENESS ==========
    response_clean = response.strip()
    word_count = len(response_clean.split())
    char_count = len(response_clean)
    
    # Categorize completeness by word count
    if word_count < 5:
        completeness = "MINIMAL"  # Just answer letter
    elif word_count < 30:
        completeness = "BRIEF"  # Short explanation
    elif word_count < 100:
        completeness = "MODERATE"  # Decent explanation
    elif word_count < 300:
        completeness = "COMPLETE"  # Full explanation
    else:
        completeness = "VERBOSE"  # Very long (might be multilingual)
    
    metrics["word_count"] = word_count
    metrics["char_count"] = char_count
    metrics["completeness"] = completeness
    
    # ========== STRUCTURAL ANALYSIS ==========
    # Check for step-by-step reasoning
    has_steps = bool(re.search(r'(step\s*\d+|firstly|secondly|finally|Step\s*\d+)', response, re.I))
    
    # Check for mathematical equations/formulas
    has_equations = bool(re.search(r'[=\+\-\*/\\]|\\frac|\\sum|\^|_{', response))
    
    # Check for proper formatting (bullet points, numbered lists)
    has_formatting = bool(re.search(r'(^\s*[-*•]\s)|(^\s*\d+\.)', response, re.M))
    
    # Check for scientific terminology (basic heuristic)
    scientific_words = ['therefore', 'hence', 'thus', 'equation', 'formula', 'reaction', 
                       'mechanism', 'principle', 'theory', 'law', 'hypothesis']
    has_scientific_terms = any(word in response.lower() for word in scientific_words)
    
    metrics["has_step_by_step"] = has_steps
    metrics["has_equations"] = has_equations
    metrics["has_formatting"] = has_formatting
    metrics["has_scientific_terminology"] = has_scientific_terms
    
    # Structural quality score (0-10)
    structural_score = 0
    if has_steps: structural_score += 3
    if has_equations: structural_score += 2
    if has_formatting: structural_score += 2
    if has_scientific_terms: structural_score += 3
    metrics["structural_quality"] = min(10, structural_score)
    
    # ========== ERROR DETECTION ==========
    # Check for error messages
    has_errors = bool(re.search(r'\[Error|\[error|Error:|Exception:|Failed', response, re.I))
    is_truncated = response.endswith('...') or response.endswith('…')
    
    metrics["has_errors"] = has_errors
    metrics["is_truncated"] = is_truncated
    
    # ========== ANSWER EXTRACTION ==========
    # Try to extract answer letter (A, B, C, D)
    answer_pattern = r'\b([A-D])\b'
    answer_matches = re.findall(answer_pattern, response)
    
    # Check if response starts with just answer letter
    starts_with_answer = bool(re.match(r'^[A-D]\s*$', response_clean.split('\n')[0]))
    
    metrics["extracted_answers"] = list(set(answer_matches))  # Unique answers found
    metrics["answer_count"] = len(answer_matches)
    metrics["starts_with_answer_only"] = starts_with_answer
    
    return metrics

def judge_answer_quality_gpt4o(question_text: str, response: str, subject: str) -> dict:
    """Use GPT-4o to evaluate answer quality with subject-specific criteria.
    
    Args:
        question_text: The question text
        response: The model's answer/response
        subject: Detected subject (Physics, Chemistry, or Biology)
    
    Returns:
        Dictionary with score, reasoning, and comprehensive metrics
    """
    try:
        # Comprehensive response analysis
        metrics = analyze_response_quality(response)
        
        # Subject-specific evaluation criteria
        if "Physics" in subject:
            criteria_desc = "Physics: Clarity (proper terminology, variable definitions), Logical Reasoning (step-by-step with equations), Correctness (proper physics principles)"
        elif "Chemistry" in subject:
            criteria_desc = "Chemistry: Clarity (IUPAC names, balanced equations), Logical Reasoning (stoichiometry, calculations), Correctness (chemistry concepts)"
        elif "Biology" in subject:
            criteria_desc = "Biology: Clarity (biological terminology, structure-function), Logical Reasoning (mechanisms, processes), Correctness (biological principles)"
        else:
            criteria_desc = "General: Clarity, Logical Reasoning, Correctness"
        
        # Explanation quality evaluation with comprehensive metrics awareness
        evaluation_prompt = f"""You are evaluating ONLY the explanation quality of an AI MCQ solver for NEET exam questions.
Do NOT evaluate whether the final answer is correct - that is assessed separately.

Subject: {subject}
Evaluation Criteria: {criteria_desc}

Response Statistics:
- Word Count: {metrics['word_count']}
- Completeness: {metrics['completeness']}
- Languages: {metrics['languages_detected']}
- Has Step-by-Step: {metrics['has_step_by_step']}
- Has Equations: {metrics['has_equations']}
- Has Formatting: {metrics['has_formatting']}

Question: {question_text}

MCQ Solver's Response:
{response}

Rate ONLY the explanation quality on a scale of 1-10:
- Clarity (30%): Use of proper terminology, clear explanations, well-structured response
- Logical Reasoning (40%): Step-by-step approach, showing work, justification for each step
- Correctness of Method (30%): Proper application of scientific principles and methodology

CRITICAL PENALTY GUIDELINES:
1. **Non-English Content**: If response contains Hindi, Tamil, or other non-English text, apply -2 point penalty for violating NEET English requirement
2. **Minimal Response**: If only answer letter ("A", "B", "C", "D") with no explanation, score 1-2 maximum
3. **No Reasoning**: If brief response (<30 words) without logical steps, score 2-4 maximum  
4. **Translation Errors**: If non-English portions contain factual errors, apply additional -1 to -2 penalty
5. **Structural Quality**: Reward step-by-step reasoning (+2), equations (+1), proper formatting (+1)
6. **Verbosity**: If excessively long (>300 words) due to unnecessary repetition, apply -1 penalty

Provide your evaluation in this format:
Overall Quality Score: [number 1-10]
Brief Reasoning: [2-3 sentences explaining the explanation quality and any penalties applied]
Language Issues: [YES if non-English content found, NO if English-only]
Completeness Assessment: [COMPLETE/MODERATE/BRIEF/MINIMAL]
"""
        
        # Call GPT-4o
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert evaluator of scientific explanations for NEET exam questions. NEET requires responses in English."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0,
            max_tokens=600
        )
        
        response_text = completion.choices[0].message.content
        
        # Handle None response_text
        if not response_text:
            response_text = ""
        
        # Extract score
        score_match = re.search(r"Overall Quality Score:\s*(\d+)", response_text)
        score = int(score_match.group(1)) if score_match else 5
        score = max(1, min(10, score))
        
        # Extract reasoning
        reasoning_match = re.search(r"Brief Reasoning:\s*(.+?)(?:\n|$)", response_text, re.S)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text[:200]
        
        # Extract language issues
        lang_issues_match = re.search(r"Language Issues:\s*(YES|NO)", response_text, re.I)
        judge_lang_flag = lang_issues_match.group(1).upper() if lang_issues_match else "UNKNOWN"
        
        # Extract completeness assessment
        completeness_match = re.search(r"Completeness Assessment:\s*(COMPLETE|MODERATE|BRIEF|MINIMAL)", response_text, re.I)
        judge_completeness = completeness_match.group(1).upper() if completeness_match else metrics["completeness"]
        
        # Return comprehensive metrics
        return {
            "score": score, 
            "reasoning": reasoning,
            # Language metrics
            "is_english_only": metrics["is_english_only"],
            "languages_detected": metrics["languages_detected"],
            "multilingual_penalty": metrics["multilingual_penalty"],
            "judge_lang_flag": judge_lang_flag,
            # Completeness metrics
            "completeness": metrics["completeness"],
            "judge_completeness": judge_completeness,
            "word_count": metrics["word_count"],
            "char_count": metrics["char_count"],
            # Structural metrics
            "has_step_by_step": metrics["has_step_by_step"],
            "has_equations": metrics["has_equations"],
            "has_formatting": metrics["has_formatting"],
            "has_scientific_terminology": metrics["has_scientific_terminology"],
            "structural_quality": metrics["structural_quality"],
            # Error detection
            "has_errors": metrics["has_errors"],
            "is_truncated": metrics["is_truncated"],
            # Answer extraction
            "extracted_answers": "|".join(metrics["extracted_answers"]),
            "answer_count": metrics["answer_count"],
            "starts_with_answer_only": metrics["starts_with_answer_only"]
        }
        
    except Exception as e:
        print(f"  ⚠️  Judge error: {str(e)[:100]}")
        # Return error metrics with defaults
        return {
            "score": 0, 
            "reasoning": f"Judge error: {str(e)[:100]}",
            "is_english_only": True,
            "languages_detected": "Error",
            "multilingual_penalty": 0,
            "judge_lang_flag": "ERROR",
            "completeness": "ERROR",
            "judge_completeness": "ERROR",
            "word_count": 0,
            "char_count": 0,
            "has_step_by_step": False,
            "has_equations": False,
            "has_formatting": False,
            "has_scientific_terminology": False,
            "structural_quality": 0,
            "has_errors": True,
            "is_truncated": False,
            "extracted_answers": "",
            "answer_count": 0,
            "starts_with_answer_only": False
        }

# ============================================================================
# Load and Process Questions
# ============================================================================

def load_question_text(question_id: str) -> str:
    """Load the actual question text from the dataset.
    
    Args:
        question_id: Question identifier (e.g., 'Question_Paper_21_p21_q51')
    
    Returns:
        Formatted question text with options
    """
    try:
        # Try validation set first
        val_file = ROOT / "aneeta_v2" / "Processed Data" / "dspy_dataset_val.jsonl"
        if val_file.exists():
            with open(val_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get('question_id') == question_id:
                        q_text = data['question_text']
                        options = data['options']
                        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
                        return f"{q_text}\n\nOptions:\n{options_text}"
        
        # Fallback to combined dataset
        combined_file = ROOT / "aneeta_v2" / "Processed Data" / "dspy_dataset_combined.jsonl"
        if combined_file.exists():
            with open(combined_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get('question_id') == question_id:
                        q_text = data['question_text']
                        options = data['options']
                        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
                        return f"{q_text}\n\nOptions:\n{options_text}"
        
        return f"[Question text not found for {question_id}]"
        
    except Exception as e:
        return f"[Error loading question: {str(e)}]"

def rejudge_csv():
    """Re-judge all responses in the CSV file using GPT-4o."""
    
    print("="*70)
    print("RE-JUDGING RESPONSES WITH GPT-4o")
    print("="*70)
    print(f"Input: {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Judge Model: {JUDGE_MODEL}")
    print(f"Rate Limit: {REQUESTS_PER_MINUTE} requests/minute")
    print("="*70)
    
    # Check if input file exists
    if not INPUT_CSV.exists():
        print(f"❌ Error: Input file not found: {INPUT_CSV}")
        return
    
    # Load CSV
    print(f"\n📖 Loading CSV file...")
    rows = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print(f"✓ Loaded {len(rows)} rows")
    
    # Add new columns for GPT-4o judgments and comprehensive metrics
    new_fieldnames = list(fieldnames or []) + [
        # Core judge outputs
        'gpt4o_quality_score', 
        'gpt4o_judge_reasoning',
        # Language metrics
        'is_english_only',
        'languages_detected',
        'multilingual_penalty',
        'judge_lang_flag',
        # Completeness metrics
        'completeness',
        'judge_completeness',
        'word_count',
        'char_count',
        # Structural metrics
        'has_step_by_step',
        'has_equations',
        'has_formatting',
        'has_scientific_terminology',
        'structural_quality',
        # Error detection
        'has_errors',
        'is_truncated',
        # Answer extraction
        'extracted_answers',
        'answer_count',
        'starts_with_answer_only'
    ]
    
    # Create output directory
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Open output file
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=new_fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        
        # Process each row
        total_rows = len(rows)
        for idx, row in enumerate(rows, 1):
            question_id = row['question_id']
            subject = row.get('detected_subject', row.get('subject', 'Unknown'))
            response = row['response']
            
            print(f"\n[{idx}/{total_rows}] {question_id} ({subject})")
            print(f"  Model: {row['model']} | Agent: {row['agent_display']}")
            
            # Skip if response is empty or error
            if not response or response.startswith('[Error'):
                print(f"  ⏭️  Skipping (empty or error response)")
                row['gpt4o_quality_score'] = 0
                row['gpt4o_judge_reasoning'] = 'Skipped: empty or error response'
                writer.writerow(row)
                continue
            
            # Load full question text
            question_text = load_question_text(question_id)
            if question_text.startswith('['):
                print(f"  ⚠️  Warning: Could not load question text")
            
            # Judge with GPT-4o
            print(f"  🧠 Judging with {JUDGE_MODEL}...")
            judge_result = judge_answer_quality_gpt4o(question_text, response, subject)
            
            # Add all metrics to row
            row['gpt4o_quality_score'] = judge_result['score']
            row['gpt4o_judge_reasoning'] = judge_result['reasoning']
            # Language metrics
            row['is_english_only'] = judge_result['is_english_only']
            row['languages_detected'] = judge_result['languages_detected']
            row['multilingual_penalty'] = judge_result['multilingual_penalty']
            row['judge_lang_flag'] = judge_result['judge_lang_flag']
            # Completeness metrics
            row['completeness'] = judge_result['completeness']
            row['judge_completeness'] = judge_result['judge_completeness']
            row['word_count'] = judge_result['word_count']
            row['char_count'] = judge_result['char_count']
            # Structural metrics
            row['has_step_by_step'] = judge_result['has_step_by_step']
            row['has_equations'] = judge_result['has_equations']
            row['has_formatting'] = judge_result['has_formatting']
            row['has_scientific_terminology'] = judge_result['has_scientific_terminology']
            row['structural_quality'] = judge_result['structural_quality']
            # Error detection
            row['has_errors'] = judge_result['has_errors']
            row['is_truncated'] = judge_result['is_truncated']
            # Answer extraction
            row['extracted_answers'] = judge_result['extracted_answers']
            row['answer_count'] = judge_result['answer_count']
            row['starts_with_answer_only'] = judge_result['starts_with_answer_only']
            
            # Print key metrics
            print(f"  ✓ Score: {judge_result['score']}/10 | Words: {judge_result['word_count']} | Structural: {judge_result['structural_quality']}/10")
            print(f"  📝 {judge_result['reasoning'][:80]}...")
            if not judge_result['is_english_only']:
                print(f"  ⚠️  Language Issue: {judge_result['languages_detected']} (penalty: {judge_result['multilingual_penalty']})")
            if judge_result['completeness'] in ['MINIMAL', 'BRIEF']:
                print(f"  ⚠️  Completeness: {judge_result['completeness']} (low quality expected)")
            if judge_result['has_errors'] or judge_result['is_truncated']:
                print(f"  ❌ Issues: Errors={judge_result['has_errors']}, Truncated={judge_result['is_truncated']}")
            
            # Write row
            writer.writerow(row)
            
            # Rate limiting
            if idx < total_rows:
                time.sleep(DELAY_BETWEEN_REQUESTS)
    
    print("\n" + "="*70)
    print("✅ RE-JUDGING COMPLETE!")
    print("="*70)
    print(f"Output saved to: {OUTPUT_CSV}")
    
    # Summary statistics
    print("\n📊 SUMMARY STATISTICS")
    print("="*70)
    
    with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    
    # Calculate averages by model and agent
    from collections import defaultdict
    stats = defaultdict(lambda: {'scores': [], 'original_scores': []})
    
    for row in all_rows:
        key = f"{row['model']} - {row['agent_display']}"
        
        gpt4o_score = row.get('gpt4o_quality_score', '0')
        if gpt4o_score and gpt4o_score != '0':
            stats[key]['scores'].append(int(gpt4o_score))
        
        original_score = row.get('quality_score', '0')
        if original_score and original_score != '0':
            stats[key]['original_scores'].append(int(original_score))
    
    print("\nComparison: Original Judge vs GPT-4o")
    print("-" * 70)
    print(f"{'Configuration':<50} {'Original':<12} {'GPT-4o':<12}")
    print("-" * 70)
    
    for config in sorted(stats.keys()):
        original_avg = sum(stats[config]['original_scores']) / len(stats[config]['original_scores']) if stats[config]['original_scores'] else 0
        gpt4o_avg = sum(stats[config]['scores']) / len(stats[config]['scores']) if stats[config]['scores'] else 0
        print(f"{config:<50} {original_avg:>6.2f}/10    {gpt4o_avg:>6.2f}/10")
    
    print("="*70)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    rejudge_csv()
