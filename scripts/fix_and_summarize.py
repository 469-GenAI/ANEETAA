"""
Fix failed entry in rejudged CSV by actually calling GPT-4o to rejudge it, then generate summary statistics.
"""

import os
import csv
import re
import pandas as pd
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Paths
ROOT = Path(__file__).parent.parent
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

INPUT_CSV = ROOT / "results" / "rejudged_gpt4o.csv"
OUTPUT_CSV = ROOT / "results" / "rejudged_gpt4o_fixed.csv"
SUMMARY_CSV = ROOT / "results" / "rejudged_gpt4o_summary.csv"
DETAILED_SUMMARY = ROOT / "results" / "rejudged_gpt4o_detailed_summary.txt"
JUDGE_MODEL = "gpt-4o"

def analyze_response_quality(response: str) -> dict:
    """Comprehensive analysis of response quality (copied from rejudge_with_gpt4o.py)"""
    metrics = {}
    
    # Language detection
    non_english_scripts = []
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
        'Chinese (CJK)': r'[\u4E00-\u9FFF]',
        'Japanese (Hiragana)': r'[\u3040-\u309F]',
        'Japanese (Katakana)': r'[\u30A0-\u30FF]',
        'Korean (Hangul)': r'[\uAC00-\uD7AF]',
        'Arabic': r'[\u0600-\u06FF]',
        'Hebrew': r'[\u0590-\u05FF]',
        'Cyrillic (Russian, etc.)': r'[\u0400-\u04FF]',
        'Greek': r'[\u0370-\u03FF]',
    }
    
    for script_name, pattern in script_ranges.items():
        if re.search(pattern, response):
            non_english_scripts.append(script_name)
    
    metrics['is_english_only'] = len(non_english_scripts) == 0
    metrics['languages_detected'] = ', '.join(non_english_scripts) if non_english_scripts else 'English'
    metrics['multilingual_penalty'] = len(non_english_scripts) * 2
    
    # Word count
    words = response.split()
    metrics['word_count'] = len(words)
    metrics['char_count'] = len(response)
    
    # Completeness
    if metrics['word_count'] < 10:
        metrics['completeness'] = 'MINIMAL'
    elif metrics['word_count'] < 50:
        metrics['completeness'] = 'BRIEF'
    elif metrics['word_count'] < 150:
        metrics['completeness'] = 'MODERATE'
    elif metrics['word_count'] < 300:
        metrics['completeness'] = 'COMPLETE'
    else:
        metrics['completeness'] = 'VERBOSE'
    
    # Structural features
    metrics['has_step_by_step'] = bool(re.search(r'(step\s*\d+|firstly|secondly|finally|\d+\.|Step\s*:)', response, re.I))
    metrics['has_equations'] = bool(re.search(r'[=+\-*/^]|\\[a-zA-Z]+|_\{|_{', response))
    metrics['has_formatting'] = bool(re.search(r'(\*\*|\n-|\n\d+\.|\n•)', response))
    
    science_terms = r'(oxidation|reduction|equilibrium|reaction|electron|proton|cell|DNA|enzyme|gene|velocity|force|energy|momentum|wavelength|frequency)'
    metrics['has_scientific_terminology'] = bool(re.search(science_terms, response, re.I))
    
    structural_score = 0
    if metrics['has_step_by_step']: structural_score += 3
    if metrics['has_equations']: structural_score += 2
    if metrics['has_formatting']: structural_score += 2
    if metrics['has_scientific_terminology']: structural_score += 3
    metrics['structural_quality'] = min(10, structural_score)
    
    # Error detection
    metrics['has_errors'] = bool(re.search(r'(error|exception|failed|invalid|malformed)', response.lower()))
    metrics['is_truncated'] = response.endswith('...') or response.endswith('…')
    
    # Answer extraction
    answer_pattern = r'\b([A-D])\b(?=\s|$|\.|\))'
    extracted = list(set(re.findall(answer_pattern, response)))
    metrics['extracted_answers'] = extracted
    metrics['answer_count'] = len(extracted)
    metrics['starts_with_answer_only'] = bool(re.match(r'^\s*[A-D]\s*$', response))
    
    return metrics

def judge_answer_quality_gpt4o(question_text: str, response: str, subject: str) -> dict:
    """Call GPT-4o to judge response quality (copied from rejudge_with_gpt4o.py)"""
    try:
        metrics = analyze_response_quality(response)
        
        if "Physics" in subject:
            criteria_desc = "Physics: Clarity (proper terminology, variable definitions), Logical Reasoning (step-by-step with equations), Correctness (proper physics principles)"
        elif "Chemistry" in subject:
            criteria_desc = "Chemistry: Clarity (IUPAC names, balanced equations), Logical Reasoning (stoichiometry, calculations), Correctness (chemistry concepts)"
        elif "Biology" in subject:
            criteria_desc = "Biology: Clarity (biological terminology, structure-function), Logical Reasoning (mechanisms, processes), Correctness (biological principles)"
        else:
            criteria_desc = "General: Clarity, Logical Reasoning, Correctness"
        
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
        
        completion = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert evaluator of scientific explanations for NEET exam questions. NEET requires responses in English."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0,
            max_tokens=600
        )
        
        response_text = completion.choices[0].message.content or ""
        
        score_match = re.search(r"Overall Quality Score:\s*(\d+)", response_text)
        score = int(score_match.group(1)) if score_match else 5
        score = max(1, min(10, score))
        
        reasoning_match = re.search(r"Brief Reasoning:\s*(.+?)(?:\n|$)", response_text, re.S)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response_text[:200]
        
        lang_issues_match = re.search(r"Language Issues:\s*(YES|NO)", response_text, re.I)
        judge_lang_flag = lang_issues_match.group(1).upper() if lang_issues_match else "UNKNOWN"
        
        completeness_match = re.search(r"Completeness Assessment:\s*(COMPLETE|MODERATE|BRIEF|MINIMAL)", response_text, re.I)
        judge_completeness = completeness_match.group(1).upper() if completeness_match else metrics["completeness"]
        
        return {
            "score": score, 
            "reasoning": reasoning,
            "is_english_only": metrics["is_english_only"],
            "languages_detected": metrics["languages_detected"],
            "multilingual_penalty": metrics["multilingual_penalty"],
            "judge_lang_flag": judge_lang_flag,
            "completeness": metrics["completeness"],
            "judge_completeness": judge_completeness,
            "word_count": metrics["word_count"],
            "char_count": metrics["char_count"],
            "has_step_by_step": metrics["has_step_by_step"],
            "has_equations": metrics["has_equations"],
            "has_formatting": metrics["has_formatting"],
            "has_scientific_terminology": metrics["has_scientific_terminology"],
            "structural_quality": metrics["structural_quality"],
            "has_errors": metrics["has_errors"],
            "is_truncated": metrics["is_truncated"],
            "extracted_answers": "|".join(metrics["extracted_answers"]),
            "answer_count": metrics["answer_count"],
            "starts_with_answer_only": metrics["starts_with_answer_only"]
        }
        
    except Exception as e:
        print(f"  ⚠️  Judge error: {str(e)[:100]}")
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

def fix_failed_entry():
    """Find ALL failed/error entries and call GPT-4o to rejudge them."""
    
    print("="*70)
    print("SCANNING FOR ALL ERROR ENTRIES")
    print("="*70)
    
    # Read CSV
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    
    # Find ALL error patterns
    error_patterns = [
        (lambda r: r['response'].startswith('[Error:'), "Error prefix"),
        (lambda r: r.get('gpt4o_quality_score') == '0' and 'Skipped' in r.get('gpt4o_judge_reasoning', ''), "Skipped with score 0"),
        (lambda r: r.get('gpt4o_judge_reasoning', '').startswith('Skipped'), "Skipped reasoning"),
        (lambda r: r.get('is_english_only') == '' or r.get('word_count') == '', "Missing metrics"),
        (lambda r: 'malformed' in r.get('response', '').lower(), "Malformed response"),
        (lambda r: 'dict_keys' in r.get('response', ''), "Dict keys error"),
    ]
    
    # Scan for all errors
    error_indices = set()
    print(f"\n📋 Scanning {len(rows)} rows for errors...")
    
    for i, row in enumerate(rows):
        for pattern_func, pattern_name in error_patterns:
            try:
                if pattern_func(row):
                    error_indices.add(i)
                    print(f"  Row {i+2}: {pattern_name} - {row['question_id']} ({row['model']}, {row['agent_display']})")
                    break
            except Exception as e:
                continue
    
    print(f"\n✓ Found {len(error_indices)} error entries to fix")
    
    # Fix all error entries
    fixed_count = 0
    failed_count = 0
    
    for i in sorted(error_indices):
        row = rows[i]
        print(f"\n{'='*70}")
        print(f"Processing row {i+2}/{len(rows)}: {row['question_id']}")
        print(f"Model: {row['model']} | Agent: {row['agent_display']} | Subject: {row['subject']}")
        print(f"Current response: {row['response'][:120]}...")
        
        try:
            # Check if response is truly malformed (no actual content to judge)
            is_truly_malformed = any([
                'The response should contain' in row['response'],
                len(row['response'].strip()) < 10,
                row['response'].strip() == '',
                row['response'].startswith('[Error:') and 'dict_keys' in row['response'] and len(row['response']) < 150
            ])
            
            if is_truly_malformed:
                # Mark with error values - can't judge this
                print("  ⚠️  Response is truly malformed (no content to judge)")
                judge_result = {
                    "score": 0,
                    "reasoning": "System error: Model failed to generate valid response with required keys",
                    "is_english_only": True,
                    "languages_detected": "Error",
                    "multilingual_penalty": 0,
                    "judge_lang_flag": "ERROR",
                    "completeness": "ERROR",
                    "judge_completeness": "ERROR",
                    "word_count": 0,
                    "char_count": len(row['response']),
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
                failed_count += 1
            else:
                # Try to actually judge it with GPT-4o
                print("  🤖 Calling GPT-4o to rejudge...")
                judge_result = judge_answer_quality_gpt4o(
                    question_text=row['question'],
                    response=row['response'],
                    subject=row['subject']
                )
                print(f"  ✓ GPT-4o Score: {judge_result['score']}/10")
            
            # Update row with judge results
            row['gpt4o_quality_score'] = str(judge_result['score'])
            row['gpt4o_judge_reasoning'] = judge_result['reasoning']
            row['is_english_only'] = str(judge_result['is_english_only'])
            row['languages_detected'] = judge_result['languages_detected']
            row['multilingual_penalty'] = str(judge_result['multilingual_penalty'])
            row['judge_lang_flag'] = judge_result['judge_lang_flag']
            row['completeness'] = judge_result['completeness']
            row['judge_completeness'] = judge_result['judge_completeness']
            row['word_count'] = str(judge_result['word_count'])
            row['char_count'] = str(judge_result['char_count'])
            row['has_step_by_step'] = str(judge_result['has_step_by_step'])
            row['has_equations'] = str(judge_result['has_equations'])
            row['has_formatting'] = str(judge_result['has_formatting'])
            row['has_scientific_terminology'] = str(judge_result['has_scientific_terminology'])
            row['structural_quality'] = str(judge_result['structural_quality'])
            row['has_errors'] = str(judge_result['has_errors'])
            row['is_truncated'] = str(judge_result['is_truncated'])
            row['extracted_answers'] = judge_result['extracted_answers']
            row['answer_count'] = str(judge_result['answer_count'])
            row['starts_with_answer_only'] = str(judge_result['starts_with_answer_only'])
            
            fixed_count += 1
            print(f"  ✓ Fixed!")
            
        except Exception as e:
            print(f"  ❌ Failed to fix: {str(e)[:100]}")
            # Still mark with error values
            row['gpt4o_quality_score'] = '0'
            row['gpt4o_judge_reasoning'] = f"Fix attempt failed: {str(e)[:100]}"
            row['is_english_only'] = 'True'
            row['languages_detected'] = 'Error'
            row['multilingual_penalty'] = '0'
            row['judge_lang_flag'] = 'ERROR'
            row['completeness'] = 'ERROR'
            row['judge_completeness'] = 'ERROR'
            row['word_count'] = '0'
            row['char_count'] = str(len(row.get('response', '')))
            row['has_step_by_step'] = 'False'
            row['has_equations'] = 'False'
            row['has_formatting'] = 'False'
            row['has_scientific_terminology'] = 'False'
            row['structural_quality'] = '0'
            row['has_errors'] = 'True'
            row['is_truncated'] = 'False'
            row['extracted_answers'] = ''
            row['answer_count'] = '0'
            row['starts_with_answer_only'] = 'False'
            failed_count += 1
            failed_count += 1
    
    # Write fixed CSV
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n{'='*70}")
    print(f"✓ Successfully fixed: {fixed_count} entries")
    print(f"⚠️  Failed to fix (marked as errors): {failed_count} entries")
    print(f"✓ Total processed: {len(error_indices)} entries")
    print(f"✓ Saved to: {OUTPUT_CSV}")
    print(f"{'='*70}")
    
    return rows

def generate_summary_statistics(rows):
    """Generate comprehensive summary statistics."""
    
    print("\n" + "="*70)
    print("GENERATING SUMMARY STATISTICS")
    print("="*70)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(rows)
    
    # Convert numeric columns
    numeric_cols = ['gpt4o_quality_score', 'quality_score', 'fact_score', 'word_count', 
                   'char_count', 'structural_quality', 'multilingual_penalty', 'answer_count']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Boolean columns
    bool_cols = ['is_english_only', 'has_step_by_step', 'has_equations', 'has_formatting',
                'has_scientific_terminology', 'has_errors', 'is_truncated', 'starts_with_answer_only']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({'True': True, 'False': False, True: True, False: False}).fillna(False)
    
    # ====== SUMMARY BY CONFIGURATION ======
    summary_stats = []
    
    for (model, agent), group in df.groupby(['model', 'agent_display']):
        stats = {
            'Model': model,
            'Agent': agent,
            'Total_Questions': len(group),
            
            # Accuracy metrics
            'Correct_Count': (group['is_correct'] == 'True').sum(),
            'Accuracy_%': round((group['is_correct'] == 'True').sum() / len(group) * 100, 2),
            
            # Original judge scores
            'Avg_Original_Quality': round(group['quality_score'].mean(), 2),
            'Avg_Original_Fact': round(group['fact_score'].mean(), 2),
            
            # GPT-4o judge scores
            'Avg_GPT4o_Quality': round(group['gpt4o_quality_score'].mean(), 2),
            'Score_Difference': round(group['gpt4o_quality_score'].mean() - group['quality_score'].mean(), 2),
            
            # Language metrics
            'English_Only_%': round(group['is_english_only'].sum() / len(group) * 100, 2),
            'Multilingual_Count': (~group['is_english_only']).sum(),
            'Avg_Multilingual_Penalty': round(group['multilingual_penalty'].mean(), 2),
            
            # Completeness metrics
            'Avg_Word_Count': round(group['word_count'].mean(), 1),
            'Minimal_Count': (group['completeness'] == 'MINIMAL').sum(),
            'Brief_Count': (group['completeness'] == 'BRIEF').sum(),
            'Complete_Count': (group['completeness'] == 'COMPLETE').sum(),
            'Verbose_Count': (group['completeness'] == 'VERBOSE').sum(),
            
            # Structural metrics
            'Avg_Structural_Quality': round(group['structural_quality'].mean(), 2),
            'Has_Steps_%': round(group['has_step_by_step'].sum() / len(group) * 100, 2),
            'Has_Equations_%': round(group['has_equations'].sum() / len(group) * 100, 2),
            'Has_Formatting_%': round(group['has_formatting'].sum() / len(group) * 100, 2),
            
            # Error metrics
            'Error_Count': group['has_errors'].sum(),
            'Truncated_Count': group['is_truncated'].sum(),
        }
        summary_stats.append(stats)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_stats)
    
    # Sort by model and agent
    summary_df = summary_df.sort_values(['Model', 'Agent'])
    
    # Save to CSV
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n✓ Summary saved to: {SUMMARY_CSV}")
    
    # ====== PRINT DETAILED SUMMARY ======
    output_lines = []
    
    output_lines.append("="*100)
    output_lines.append("COMPREHENSIVE SUMMARY STATISTICS - GPT-4o Re-judging Results")
    output_lines.append("="*100)
    
    # Overall statistics
    output_lines.append("\n📊 OVERALL STATISTICS")
    output_lines.append("-"*100)
    output_lines.append(f"Total Responses Analyzed: {len(df)}")
    output_lines.append(f"Total Configurations: {len(summary_df)} (3 models × 4 agent types)")
    output_lines.append(f"Average Original Quality Score: {df['quality_score'].mean():.2f}/10")
    output_lines.append(f"Average GPT-4o Quality Score: {df['gpt4o_quality_score'].mean():.2f}/10")
    output_lines.append(f"Score Difference (GPT-4o - Original): {(df['gpt4o_quality_score'] - df['quality_score']).mean():.2f}")
    
    # Language statistics
    output_lines.append("\n🌍 LANGUAGE COMPLIANCE")
    output_lines.append("-"*100)
    english_only_pct = (df['is_english_only'].sum() / len(df)) * 100
    output_lines.append(f"English-Only Responses: {df['is_english_only'].sum()} ({english_only_pct:.1f}%)")
    output_lines.append(f"Multilingual Responses: {(~df['is_english_only']).sum()} ({100-english_only_pct:.1f}%)")
    output_lines.append(f"Total Multilingual Penalty: {df['multilingual_penalty'].sum():.0f} points")
    
    # Top languages detected
    lang_counts = df[~df['is_english_only']]['languages_detected'].value_counts().head(5)
    if len(lang_counts) > 0:
        output_lines.append("\nTop Non-English Languages Detected:")
        for lang, count in lang_counts.items():
            output_lines.append(f"  • {lang}: {count} responses")
    
    # Completeness distribution
    output_lines.append("\n📝 COMPLETENESS DISTRIBUTION")
    output_lines.append("-"*100)
    completeness_counts = df['completeness'].value_counts()
    for comp_type in ['MINIMAL', 'BRIEF', 'MODERATE', 'COMPLETE', 'VERBOSE']:
        count = completeness_counts.get(comp_type, 0)
        pct = (count / len(df)) * 100
        output_lines.append(f"{comp_type:12}: {count:5} ({pct:5.1f}%)")
    
    output_lines.append(f"\nAverage Word Count: {df['word_count'].mean():.1f} words")
    
    # Structural quality
    output_lines.append("\n🏗️ STRUCTURAL QUALITY")
    output_lines.append("-"*100)
    output_lines.append(f"Average Structural Quality: {df['structural_quality'].mean():.2f}/10")
    output_lines.append(f"Responses with Step-by-Step: {df['has_step_by_step'].sum()} ({df['has_step_by_step'].sum()/len(df)*100:.1f}%)")
    output_lines.append(f"Responses with Equations: {df['has_equations'].sum()} ({df['has_equations'].sum()/len(df)*100:.1f}%)")
    output_lines.append(f"Responses with Formatting: {df['has_formatting'].sum()} ({df['has_formatting'].sum()/len(df)*100:.1f}%)")
    output_lines.append(f"Responses with Scientific Terms: {df['has_scientific_terminology'].sum()} ({df['has_scientific_terminology'].sum()/len(df)*100:.1f}%)")
    
    # Error analysis
    output_lines.append("\n❌ ERROR ANALYSIS")
    output_lines.append("-"*100)
    output_lines.append(f"Responses with Errors: {df['has_errors'].sum()}")
    output_lines.append(f"Truncated Responses: {df['is_truncated'].sum()}")
    output_lines.append(f"Minimal Responses (just answer): {(df['starts_with_answer_only']).sum()}")
    
    # Configuration comparison
    output_lines.append("\n📈 CONFIGURATION COMPARISON (Original vs GPT-4o)")
    output_lines.append("="*100)
    output_lines.append(f"{'Configuration':<50} {'Orig':<8} {'GPT-4o':<8} {'Diff':<8} {'English%':<10} {'Words':<8} {'Struct':<8}")
    output_lines.append("-"*100)
    
    for _, row in summary_df.iterrows():
        config = f"{row['Model']} - {row['Agent']}"
        output_lines.append(f"{config:<50} {row['Avg_Original_Quality']:>6.2f}  {row['Avg_GPT4o_Quality']:>6.2f}  {row['Score_Difference']:>+6.2f}  {row['English_Only_%']:>8.1f}%  {row['Avg_Word_Count']:>6.1f}  {row['Avg_Structural_Quality']:>6.2f}")
    
    # Best and worst configurations
    output_lines.append("\n🏆 TOP 3 CONFIGURATIONS (by GPT-4o Quality Score)")
    output_lines.append("-"*100)
    top3 = summary_df.nlargest(3, 'Avg_GPT4o_Quality')
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        output_lines.append(f"{i}. {row['Model']} - {row['Agent']}: {row['Avg_GPT4o_Quality']:.2f}/10")
        output_lines.append(f"   English: {row['English_Only_%']:.1f}% | Words: {row['Avg_Word_Count']:.0f} | Structural: {row['Avg_Structural_Quality']:.2f}/10")
    
    output_lines.append("\n⚠️ BOTTOM 3 CONFIGURATIONS (by GPT-4o Quality Score)")
    output_lines.append("-"*100)
    bottom3 = summary_df.nsmallest(3, 'Avg_GPT4o_Quality')
    for i, (_, row) in enumerate(bottom3.iterrows(), 1):
        output_lines.append(f"{i}. {row['Model']} - {row['Agent']}: {row['Avg_GPT4o_Quality']:.2f}/10")
        output_lines.append(f"   Multilingual: {row['Multilingual_Count']} | Errors: {row['Error_Count']} | Minimal: {row['Minimal_Count']}")
    
    # Subject breakdown
    output_lines.append("\n📚 SUBJECT BREAKDOWN")
    output_lines.append("="*100)
    for subject in ['Physics', 'Chemistry', 'Biology']:
        subject_df = df[df['subject'] == subject]
        if len(subject_df) > 0:
            output_lines.append(f"\n{subject}:")
            output_lines.append(f"  Questions: {len(subject_df)}")
            output_lines.append(f"  Avg GPT-4o Quality: {subject_df['gpt4o_quality_score'].mean():.2f}/10")
            output_lines.append(f"  Avg Word Count: {subject_df['word_count'].mean():.1f}")
            output_lines.append(f"  English-Only: {(subject_df['is_english_only'].sum()/len(subject_df)*100):.1f}%")
            output_lines.append(f"  Has Equations: {(subject_df['has_equations'].sum()/len(subject_df)*100):.1f}%")
    
    output_lines.append("\n" + "="*100)
    output_lines.append("END OF SUMMARY")
    output_lines.append("="*100)
    
    # Print and save
    summary_text = "\n".join(output_lines)
    print(summary_text)
    
    with open(DETAILED_SUMMARY, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"\n✓ Detailed summary saved to: {DETAILED_SUMMARY}")
    
    return summary_df

if __name__ == "__main__":
    # Fix failed entry
    rows = fix_failed_entry()
    
    # Generate statistics
    summary_df = generate_summary_statistics(rows)
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"Fixed CSV: {OUTPUT_CSV}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Detailed Summary: {DETAILED_SUMMARY}")
