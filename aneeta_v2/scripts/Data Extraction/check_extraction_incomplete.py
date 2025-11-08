"""
Check extraction file for incomplete extractions.
Generates a report of questions with placeholder options or incomplete data.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def check_extraction_file(extraction_path: Path) -> Dict:
    """
    Check extraction file for incomplete extractions.
    
    Returns report with all incomplete questions.
    """
    report = {
        'total_questions': 0,
        'incomplete_count': 0,
        'matches': []
    }
    
    # Group questions by source file
    by_source = {}
    
    with open(extraction_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            report['total_questions'] += 1
            q = json.loads(line)
            
            # Check for incomplete extraction markers
            is_incomplete = False
            metadata = q.get('metadata', {})
            
            # Check metadata flags
            if metadata.get('has_incomplete_extraction', False):
                is_incomplete = True
            
            # Check for placeholder text in options
            options = q.get('options', [])
            for opt in options:
                text = opt.get('text', '')
                if 'extraction incomplete' in text.lower() or 'see original pdf' in text.lower():
                    is_incomplete = True
                    break
            
            # Check for empty/missing critical fields
            if not q.get('extracted_text') or not q.get('answer'):
                is_incomplete = True
            
            if is_incomplete:
                report['incomplete_count'] += 1
                source = q.get('source', 'unknown')
                
                if source not in by_source:
                    by_source[source] = []
                
                by_source[source].append({
                    'id': q['id'],
                    'question_number': q.get('question_number', '?'),
                    'page': q.get('page', 0),
                    'reason': 'incomplete_extraction'
                })
    
    # Format matches for output
    for source, questions in sorted(by_source.items()):
        report['matches'].append({
            'file': source,
            'count': len(questions),
            'questions': questions
        })
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Check extraction file for incomplete extractions'
    )
    parser.add_argument(
        'extraction_file',
        help='Path to extraction JSONL file'
    )
    parser.add_argument(
        '--report',
        help='Output path for report JSON (optional, defaults to extraction_report.json)'
    )
    
    args = parser.parse_args()
    
    extraction_path = Path(args.extraction_file)
    
    if not extraction_path.exists():
        print(f"Error: Extraction file not found: {extraction_path}")
        return 1
    
    print("=" * 70)
    print("CHECKING FOR INCOMPLETE EXTRACTIONS")
    print("=" * 70)
    print(f"\nAnalyzing: {extraction_path.name}")
    
    # Check the file
    report = check_extraction_file(extraction_path)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("REPORT SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total questions:        {report['total_questions']}")
    print(f"Incomplete extractions: {report['incomplete_count']}")
    print(f"Percentage incomplete:  {report['incomplete_count']/report['total_questions']*100:.1f}%")
    print(f"Affected files:         {len(report['matches'])}")
    
    if report['matches']:
        print(f"\nIncomplete questions by file:")
        for match in report['matches'][:10]:  # Show first 10
            print(f"  {match['file']}: {match['count']} questions")
        
        if len(report['matches']) > 10:
            print(f"  ... and {len(report['matches']) - 10} more files")
    
    # Save report if requested
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Report saved to: {report_path}")
    
    print(f"{'=' * 70}\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
