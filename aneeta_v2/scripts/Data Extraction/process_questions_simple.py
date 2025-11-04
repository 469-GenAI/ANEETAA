#!/usr/bin/env python3
"""
NEET Question PDF Processor - SIMPLE VERSION
Following the actual structure:
1. Question starts with "XXX."
2. Options start with "(1)" "(2)" "(3)" "(4)"
3. Answer starts with "Answer"
4. Solution starts with "Sol."
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF


class NEETQuestionProcessorSimple:
    """Simple processor following the actual PDF structure."""
    
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.questions = []
        self.stats = {
            'files_processed': 0,
            'questions_extracted': 0,
            'errors': 0
        }
    
    def process(self):
        """Main processing pipeline."""
        pdf_files = sorted(self.input_path.glob('*.pdf')) if self.input_path.is_dir() else [self.input_path]
        
        print(f"\nFound {len(pdf_files)} PDF files\n")
        
        for idx, pdf_file in enumerate(pdf_files, 1):
            try:
                print(f"[{idx}/{len(pdf_files)}] Processing: {pdf_file.name}")
                self.process_pdf(pdf_file)
                self.stats['files_processed'] += 1
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                self.stats['errors'] += 1
        
        self.write_jsonl()
        self.print_summary()
    
    def process_pdf(self, pdf_path: Path):
        """Process a single PDF file."""
        doc = fitz.open(pdf_path)
        source = pdf_path.stem
        
        # Step 1: Extract all content from PDF
        full_text = ""
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text")
            full_text += f"[PAGE {page_num + 1}]\n{text}\n"
        
        # Step 2: Parse text content - find question numbers as markers
        questions = self.parse_questions(full_text, source)
        
        self.questions.extend(questions)
        print(f"  ✓ Extracted {len(questions)} questions")
        
        doc.close()
    
    def parse_questions(self, text: str, source: str) -> List[Dict]:
        """
        Parse questions using the simple structure:
        1. Find question number markers (1., 2., 3., ...)
        2. Extract content between question N and question N+1
        3. Within that content:
           - Question text = from start to first option "(1)"
           - Options = (1), (2), (3), (4)
           - Answer = line starting with "Answer"
           - Solution = text after "Sol." until next question
        """
        questions = []
        
        # Find all question number markers (1. 2. 3. ... 200.)
        # Most questions: \n(number). 
        # SECTION-B format variations (some papers have questions without periods):
        #   - Paper 37: Q86-100 use \n(number) \n (space before newline)
        #   - Paper 38: Q51-100 use \n(number) \n (space before newline)
        #   - Paper 47: Q59 uses \n59\n \n (newline, space, newline)
        
        # Pattern 1: Standard format with period
        standard_pattern = r'\n(\d{1,3})\.\s+'
        
        # Pattern 2: SECTION-B format - VERY specific to avoid false matches
        # Papers 37 & 38 style: \n(51-100) \n (note the space before newline)
        section_b_paper37_38 = r'\n(5[1-9]|[6-9][0-9]|100) \n'
        # Paper 47 style: \n59\n \n (note: newline, then space, then newline)
        section_b_paper47 = r'\n(59)\n \n'
        
        # Find all matches
        standard_matches = [(m, m.group(1)) for m in re.finditer(standard_pattern, text)]
        section_b37_38_matches = [(m, m.group(1)) for m in re.finditer(section_b_paper37_38, text)]
        section_b47_matches = [(m, m.group(1)) for m in re.finditer(section_b_paper47, text)]
        
        # Combine and sort by position
        all_matches = standard_matches + section_b37_38_matches + section_b47_matches
        all_matches.sort(key=lambda x: x[0].start())
        
        matches = [m[0] for m in all_matches]
        print(f"    Found {len(matches)} question markers")
        
        # Track which question numbers we've seen (to avoid duplicates from Solutions section)
        seen_numbers = set()
        skipped_duplicates = []
        stopped_at = None
        
        for idx, match in enumerate(matches):
            q_num = match.group(1)
            
            # Stop after Q200 (don't process Solutions section)
            if int(q_num) > 200:
                stopped_at = q_num
                break
            
            # Skip duplicates (Solutions section references same question numbers)
            if q_num in seen_numbers:
                skipped_duplicates.append(q_num)
                continue
            seen_numbers.add(q_num)
            
            # Extract content from this question to the next one
            q_start = match.end()
            if idx + 1 < len(matches):
                q_end = matches[idx + 1].start()
            else:
                q_end = len(text)
            
            content = text[q_start:q_end]
            
            # Find page number
            page_search = text.rfind('[PAGE', 0, match.start())
            if page_search >= 0:
                page_match = re.search(r'\[PAGE (\d+)\]', text[page_search:page_search+20])
                page_num = int(page_match.group(1)) if page_match else 1
            else:
                page_num = 1
            
            # Parse this question's content
            parsed = self.parse_question_content(content, q_num, page_num, source)
            if parsed:
                questions.append(parsed)
                self.stats['questions_extracted'] += 1
        
        # Debug output
        if skipped_duplicates:
            print(f"    Skipped {len(skipped_duplicates)} duplicates: {sorted([int(x) for x in skipped_duplicates])}")
        if stopped_at:
            print(f"    Stopped at question {stopped_at}")
        
        return questions
    
    def parse_question_content(self, content: str, q_num: str, page_num: int, source: str) -> Optional[Dict]:
        """
        Parse a single question's content.
        Structure: Question text → Options (1)(2)(3)(4) → Answer → Sol.
        """
        
        # Find Answer marker (allow single answer 1-4, multiple answers like 3/4, or NA)
        answer_match = re.search(r'Answer\s*\(([1-4/]+|NA)\*?\)', content, re.IGNORECASE)
        if not answer_match:
            return None  # No valid answer found
        
        answer_text = answer_match.group(1)
        # For multiple answers (e.g., "3/4"), store as string "3/4"
        if '/' in answer_text:
            answer_value = answer_text  # Keep as "3/4" for training
        elif answer_text.upper() == 'NA':
            answer_value = "NA"  # Store NA as string, not 0
        else:
            answer_value = answer_text  # Store as "1", "2", "3", or "4"
        
        # Everything before "Answer" contains question + options
        question_and_options = content[:answer_match.start()]
        
        # Everything after "Answer" may contain solution
        after_answer = content[answer_match.end():]
        
        # Extract solution (starts with "Sol.")
        sol_match = re.search(r'Sol\.\s*(.*?)$', after_answer, re.DOTALL)
        explanation = sol_match.group(1).strip() if sol_match else ""
        
        # Extract options: (1) ... (2) ... (3) ... (4) ...
        # Note: \s* allows zero or more spaces after option number (handles cases like "(2)More" without space)
        options = []
        option_pattern = r'\(([1-4])\)\s*(.*?)(?=\s*\([1-4]\)|Answer|$)'
        
        for opt_match in re.finditer(option_pattern, question_and_options, re.DOTALL):
            opt_num = opt_match.group(1)
            opt_text = opt_match.group(2).strip()
            # Clean up whitespace
            opt_text = re.sub(r'\s+', ' ', opt_text)
            
            if opt_text:
                options.append({
                    'label': opt_num,
                    'text': opt_text
                })
        
        # If more than 4 options found (parsing error with matching questions), keep only last 4
        # This happens when table labels like (a), (i), etc. are confused with option markers
        if len(options) > 4:
            # Keep the last occurrence of each option label (1, 2, 3, 4)
            options_dict = {}
            for opt in options:
                options_dict[opt['label']] = opt
            options = [options_dict[label] for label in ['1', '2', '3', '4'] if label in options_dict]
        
        # FIX: Handle missing option markers (e.g., option 3 has no (3) marker)
        # If we're missing exactly one option, try to find unmarked text between adjacent options
        if len(options) == 3:
            existing_labels = {opt['label'] for opt in options}
            missing_label = None
            for i in range(1, 5):
                if str(i) not in existing_labels:
                    missing_label = str(i)
                    break
            
            if missing_label:
                # Find the options before and after the missing one
                prev_label = str(int(missing_label) - 1) if int(missing_label) > 1 else None
                next_label = str(int(missing_label) + 1) if int(missing_label) < 4 else None
                
                # Try to extract unmarked text between prev and next options
                if prev_label and next_label and prev_label in existing_labels and next_label in existing_labels:
                    # Find the RAW text between (prev_label) and (next_label) markers
                    pattern = rf'\({prev_label}\)\s*(.*?)\({next_label}\)'
                    match = re.search(pattern, question_and_options, re.DOTALL)
                    
                    if match:
                        raw_between = match.group(1)
                        
                        # Split by newlines to separate options
                        lines = [line.strip() for line in raw_between.split('\n') if line.strip()]
                        
                        if len(lines) >= 2:
                            # First line is prev option, second line is the missing option
                            prev_option_text = lines[0]
                            unmarked_option_text = lines[1]
                            
                            # Update the previous option to have only its text (without unmarked option)
                            prev_opt_idx = next(i for i, opt in enumerate(options) if opt['label'] == prev_label)
                            options[prev_opt_idx]['text'] = re.sub(r'\s+', ' ', prev_option_text).strip()
                            
                            # Add the missing option
                            unmarked_text = re.sub(r'\s+', ' ', unmarked_option_text).strip()
                            if unmarked_text and len(unmarked_text) > 1:
                                options.append({
                                    'label': missing_label,
                                    'text': unmarked_text
                                })
                                # Sort options by label
                                options = sorted(options, key=lambda x: x['label'])
        
        # Check if this is an image-based question (pattern: (1) \n \n(2) \n \n(3))
        # This happens when options are graphs/diagrams in the PDF
        image_option_pattern = r'\(1\)\s*\n\s*\n\s*\(2\)\s*\n\s*\n\s*\(3\)\s*\n\s*\n\s*\(4\)'
        is_image_based = bool(re.search(image_option_pattern, question_and_options))
        
        # If no options were extracted OR if it's an image-based question, create placeholders
        if len(options) < 4:
            if is_image_based:
                # Image-based question - create placeholder options
                options = [
                    {'label': '1', 'text': '[Image-based option - see original PDF]'},
                    {'label': '2', 'text': '[Image-based option - see original PDF]'},
                    {'label': '3', 'text': '[Image-based option - see original PDF]'},
                    {'label': '4', 'text': '[Image-based option - see original PDF]'}
                ]
            else:
                # Missing options - create placeholders for incomplete extraction
                existing_labels = {opt['label'] for opt in options}
                for i in range(1, 5):
                    if str(i) not in existing_labels:
                        options.append({
                            'label': str(i),
                            'text': f'[Option {i} - extraction incomplete]'
                        })
                # Sort options by label
                options = sorted(options, key=lambda x: x['label'])
        
        # Extract question text (everything before first option)
        first_option_match = re.search(r'\(1\)', question_and_options)
        if first_option_match:
            question_text = question_and_options[:first_option_match.start()].strip()
        else:
            question_text = question_and_options.strip()
        
        # Ensure question text is not empty
        if not question_text:
            question_text = f"[Question {q_num} - text extraction incomplete]"
        
        # Ensure explanation is not None (empty string is better for training)
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
                'has_incomplete_extraction': len(options) < 4 and not is_image_based
            }
        }
    
    def write_jsonl(self):
        """Write questions to JSONL format."""
        print(f"\nWriting {len(self.questions)} questions to {self.output_path}...")
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for q in self.questions:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')
        
        print(f"✓ Written to {self.output_path}")
    
    def print_summary(self):
        """Print processing summary."""
        avg_per_file = self.stats['questions_extracted'] / max(1, self.stats['files_processed'])
        
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Files processed:      {self.stats['files_processed']}")
        print(f"Questions extracted:  {self.stats['questions_extracted']}")
        print(f"Average per file:     {avg_per_file:.1f}")
        print(f"Errors:               {self.stats['errors']}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process NEET question PDFs - SIMPLE VERSION')
    parser.add_argument('--input', required=True, help='Input PDF or directory')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    
    args = parser.parse_args()
    
    processor = NEETQuestionProcessorSimple(args.input, args.output)
    processor.process()
