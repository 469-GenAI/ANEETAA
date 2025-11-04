"""
Reprocess incomplete extractions identified in the extraction report.
Uses process_questions_simple.py logic to re-extract specific questions.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF


def load_extraction_report(report_path: Path) -> Dict:
    """Load the extraction report JSON."""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_existing_extraction(extraction_path: Path) -> Dict[str, Dict]:
    """Load existing extraction file and index by ID."""
    questions = {}
    
    with open(extraction_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                questions[q['id']] = q
    
    return questions


def reprocess_question(pdf_path: Path, page_num: int, question_num: str, source: str) -> Optional[Dict]:
    """
    Reprocess a single question from a PDF.
    Uses the same logic as process_questions_simple.py
    """
    try:
        doc = fitz.open(pdf_path)
        
        # Get text from the specific page and surrounding pages for context
        start_page = max(0, page_num - 1)
        end_page = min(len(doc) - 1, page_num + 1)
        
        full_text = ""
        for p in range(start_page, end_page + 1):
            page = doc[p]
            full_text += f"[PAGE {p + 1}]\n"
            full_text += page.get_text("text")
            full_text += "\n\n"
        
        doc.close()
        
        # Find the specific question
        # Pattern: question_num followed by a period and content
        pattern = rf'\b{question_num}\.\s+(.*?)(?=\b\d+\.|$)'
        match = re.search(pattern, full_text, re.DOTALL)
        
        if not match:
            print(f"  ✗ Could not find question {question_num} in PDF")
            return None
        
        content = match.group(1)
        
        # Parse the question content
        parsed = parse_question_content(content, question_num, page_num, source)
        
        if parsed:
            print(f"  ✓ Successfully reprocessed Q{question_num}")
            return parsed
        else:
            print(f"  ✗ Failed to parse Q{question_num}")
            return None
            
    except Exception as e:
        print(f"  ✗ Error reprocessing Q{question_num}: {e}")
        return None


def parse_question_content(content: str, q_num: str, page_num: int, source: str) -> Optional[Dict]:
    """
    Parse a single question's content.
    Same logic as in process_questions_simple.py
    """
    # Find Answer marker
    answer_match = re.search(r'Answer\s*\(([1-4/]+|NA)\*?\)', content, re.IGNORECASE)
    if not answer_match:
        return None
    
    answer_text = answer_match.group(1)
    if '/' in answer_text:
        answer_value = answer_text
    elif answer_text.upper() == 'NA':
        answer_value = "NA"
    else:
        answer_value = answer_text
    
    # Extract question and options
    question_and_options = content[:answer_match.start()]
    after_answer = content[answer_match.end():]
    
    # Extract solution
    sol_match = re.search(r'Sol\.\s*(.*?)$', after_answer, re.DOTALL)
    explanation = sol_match.group(1).strip() if sol_match else ""
    
    # Extract options
    options = []
    option_pattern = r'\(([1-4])\)\s*(.*?)(?=\s*\([1-4]\)|Answer|$)'
    
    for opt_match in re.finditer(option_pattern, question_and_options, re.DOTALL):
        opt_num = opt_match.group(1)
        opt_text = opt_match.group(2).strip()
        opt_text = re.sub(r'\s+', ' ', opt_text)
        
        if opt_text:
            options.append({
                'label': opt_num,
                'text': opt_text
            })
    
    # Handle missing option 3 specifically (common issue based on your history)
    if len(options) == 3:
        existing_labels = {opt['label'] for opt in options}
        if '3' not in existing_labels and '2' in existing_labels and '4' in existing_labels:
            # Try to find unmarked text between (2) and (4)
            pattern = r'\(2\)\s*(.*?)\(4\)'
            match = re.search(pattern, question_and_options, re.DOTALL)
            
            if match:
                raw_between = match.group(1)
                lines = [line.strip() for line in raw_between.split('\n') if line.strip()]
                
                if len(lines) >= 2:
                    # First line is option 2, second line is the missing option 3
                    prev_opt_idx = next(i for i, opt in enumerate(options) if opt['label'] == '2')
                    options[prev_opt_idx]['text'] = re.sub(r'\s+', ' ', lines[0]).strip()
                    
                    unmarked_text = re.sub(r'\s+', ' ', lines[1]).strip()
                    if unmarked_text and len(unmarked_text) > 1:
                        options.append({
                            'label': '3',
                            'text': unmarked_text
                        })
                        options = sorted(options, key=lambda x: x['label'])
    
    # Check for image-based questions
    image_option_pattern = r'\(1\)\s*\n\s*\n\s*\(2\)\s*\n\s*\n\s*\(3\)\s*\n\s*\n\s*\(4\)'
    is_image_based = bool(re.search(image_option_pattern, question_and_options))
    
    # Fill in missing options
    if len(options) < 4:
        if is_image_based:
            options = [
                {'label': '1', 'text': '[Image-based option - see original PDF]'},
                {'label': '2', 'text': '[Image-based option - see original PDF]'},
                {'label': '3', 'text': '[Image-based option - see original PDF]'},
                {'label': '4', 'text': '[Image-based option - see original PDF]'}
            ]
        else:
            existing_labels = {opt['label'] for opt in options}
            for i in range(1, 5):
                if str(i) not in existing_labels:
                    options.append({
                        'label': str(i),
                        'text': f'[Option {i} - extraction incomplete]'
                    })
            options = sorted(options, key=lambda x: x['label'])
    
    # Extract question text
    first_option_match = re.search(r'\(1\)', question_and_options)
    if first_option_match:
        question_text = question_and_options[:first_option_match.start()].strip()
    else:
        question_text = question_and_options.strip()
    
    if not question_text:
        question_text = f"[Question {q_num} - text extraction incomplete]"
    
    if not explanation:
        explanation = ""
    
    return {
        'id': f"{source}_p{page_num}_q{q_num}",
        'question_number': q_num,
        'page': page_num,
        'source': source,
        'type': 'mcq',
        'extracted_text': question_text,
        'options': options,
        'answer': answer_value,
        'explanation': explanation,
        'metadata': {
            'option_count': len(options),
            'is_image_based': is_image_based,
            'has_incomplete_extraction': len(options) < 4 and not is_image_based,
            'reprocessed': True
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Reprocess incomplete extractions from PDF files'
    )
    parser.add_argument(
        '--pdf-dir',
        required=True,
        help='Directory containing PDF files'
    )
    parser.add_argument(
        '--extraction',
        required=True,
        help='Path to existing extraction JSONL file'
    )
    parser.add_argument(
        '--report',
        required=True,
        help='Path to extraction report JSON (from check_extraction_incomplete.py)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output path for fixed extraction JSONL file'
    )
    
    args = parser.parse_args()
    
    pdf_dir = Path(args.pdf_dir)
    extraction_path = Path(args.extraction)
    report_path = Path(args.report)
    output_path = Path(args.output)
    
    # Validate inputs
    if not pdf_dir.exists():
        print(f"Error: PDF directory not found: {pdf_dir}")
        return 1
    
    if not extraction_path.exists():
        print(f"Error: Extraction file not found: {extraction_path}")
        return 1
    
    if not report_path.exists():
        print(f"Error: Report file not found: {report_path}")
        return 1
    
    print("=" * 70)
    print("REPROCESSING INCOMPLETE EXTRACTIONS")
    print("=" * 70)
    
    # Load extraction report
    print(f"\nLoading extraction report: {report_path.name}")
    report = load_extraction_report(report_path)
    
    if 'matches' not in report:
        print("Error: Invalid report format")
        return 1
    
    total_incomplete = sum(item['count'] for item in report['matches'])
    print(f"Found {total_incomplete} incomplete extractions across {len(report['matches'])} files")
    
    # Load existing extraction
    print(f"\nLoading existing extraction: {extraction_path.name}")
    questions = load_existing_extraction(extraction_path)
    print(f"Loaded {len(questions)} existing questions")
    
    # Reprocess incomplete questions
    print(f"\nReprocessing incomplete questions...")
    reprocessed_count = 0
    failed_count = 0
    
    for item in report['matches']:
        source = item['file']
        pdf_name = source + '.pdf'
        pdf_path = pdf_dir / pdf_name
        
        if not pdf_path.exists():
            print(f"\n✗ PDF not found: {pdf_name}")
            failed_count += item['count']
            continue
        
        print(f"\nProcessing {pdf_name} ({item['count']} incomplete)")
        
        for q_info in item['questions']:
            q_id = q_info['id']
            q_num = q_info['question_number']
            
            # Get page number from existing extraction
            if q_id in questions:
                page_num = questions[q_id]['page']
                
                # Reprocess the question
                fixed_q = reprocess_question(pdf_path, page_num, q_num, source)
                
                if fixed_q:
                    # Replace the old question with fixed version
                    questions[q_id] = fixed_q
                    reprocessed_count += 1
                else:
                    failed_count += 1
            else:
                print(f"  ✗ Question {q_id} not found in extraction")
                failed_count += 1
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n\nWriting fixed extraction to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for q in questions.values():
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    
    print("\n" + "=" * 70)
    print("REPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total questions:        {len(questions)}")
    print(f"Successfully fixed:     {reprocessed_count}")
    print(f"Failed to fix:          {failed_count}")
    print(f"Fixed percentage:       {reprocessed_count/total_incomplete*100:.1f}%")
    print("=" * 70)
    print(f"\n✓ Output written to: {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
