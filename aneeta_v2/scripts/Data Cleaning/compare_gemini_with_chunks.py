"""
Compare questions in Gemini 2.5 Pro Data folder with source chunks to find missing questions.
Assumes chunks are organized by file order (e.g., processed_biology_chunks.json).
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

def load_json_file(file_path, is_jsonl=False):
    """Load a JSON file, handling errors gracefully."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if is_jsonl:
                # Load JSONL format (one JSON object per line)
                data = []
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
                return data
            else:
                return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  ⚠ Warning: Could not parse {file_path.name}: {e.msg} at line {e.lineno}")
        return None
    except Exception as e:
        print(f"  ⚠ Warning: Error reading {file_path.name}: {e}")
        return None

def load_chunks(chunks_dir):
    """
    Load all chunk files from the directory.
    Returns dict with chunk_number as key and list of questions as value.
    """
    chunks_dir = Path(chunks_dir)
    
    # Look for chunk files (Test_Extraction_Fixed_valid_chunk_*.json)
    chunk_files = sorted(chunks_dir.glob("Test_Extraction_Fixed_valid_chunk_*.json"),
                         key=lambda x: int(x.stem.split('_')[-1]))
    
    if not chunk_files:
        print(f"No chunk files found in {chunks_dir}")
        return {}
    
    print(f"Found {len(chunk_files)} chunk files")
    print()
    
    all_chunks = {}
    total_questions = 0
    
    for chunk_file in chunk_files:
        # Extract chunk number from filename (e.g., chunk_01.json -> 1)
        chunk_num = int(chunk_file.stem.split('_')[-1])
        
        data = load_json_file(chunk_file, is_jsonl=True)  # Chunks are JSONL format
        if data:
            all_chunks[chunk_num] = data
            total_questions += len(data)
            if chunk_num <= 10 or chunk_num % 20 == 0:  # Show progress
                print(f"  ✓ Loaded chunk {chunk_num:3d}: {len(data)} questions")
    
    print(f"\nTotal questions in chunks: {total_questions}")
    print(f"Chunk files loaded: {len(all_chunks)}")
    print()
    
    return all_chunks

def load_gemini_data(gemini_dir):
    """
    Load all Gemini 2.5 Pro Data files.
    Returns dict with file_number as key and list of questions as value.
    """
    gemini_dir = Path(gemini_dir)
    
    # Sort numerically by filename (01.json, 02.json, etc.)
    json_files = sorted(gemini_dir.glob("*.json"), 
                       key=lambda x: int(x.stem) if x.stem.isdigit() else int(x.stem.lstrip('0')) if x.stem.lstrip('0').isdigit() else 0)
    
    if not json_files:
        print(f"No JSON files found in {gemini_dir}")
        return {}, []
    
    print(f"Loading {len(json_files)} Gemini data files...")
    
    all_questions = {}
    total_questions = 0
    failed_files = []
    
    for json_file in json_files:
        # Extract file number (01.json -> 1, 100.json -> 100)
        file_num = int(json_file.stem.lstrip('0')) if json_file.stem.lstrip('0') else 0
        
        data = load_json_file(json_file)
        
        if data is None:
            failed_files.append(json_file.name)
            continue
        
        # Store questions by file number
        if isinstance(data, list):
            all_questions[file_num] = data
            total_questions += len(data)
            if file_num <= 10 or file_num % 20 == 0:  # Show progress
                print(f"  ✓ Loaded file {file_num:3d}: {len(data)} questions")
    
    print(f"\n✓ Successfully loaded: {len(json_files) - len(failed_files)} files")
    print(f"✗ Failed to load: {len(failed_files)} files")
    if failed_files:
        print(f"  Failed files: {', '.join(failed_files[:10])}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    print(f"\nTotal questions in Gemini data: {total_questions}")
    print(f"Files loaded: {len(all_questions)}")
    print()
    
    return all_questions, failed_files

def extract_question_text(question):
    """Extract the question text for comparison from Gemini data."""
    if isinstance(question, dict):
        return question.get('question_text', question.get('question', question.get('text', '')))
    return str(question)

def extract_chunk_question(chunk):
    """Extract question text from a chunk."""
    if isinstance(chunk, dict):
        # Check different possible fields
        text = chunk.get('extracted_text', chunk.get('question', chunk.get('text', chunk.get('content', ''))))
        return text.strip() if text else ''
    return str(chunk).strip()

def compare_questions(chunks, gemini_data):
    """
    Compare chunks with Gemini data and find missing questions.
    Compares by file order (chunk 1 -> file 1, chunk 2 -> file 2, etc.)
    Returns dict with file_number and list of missing question indices.
    """
    missing_by_file = {}
    stats = {
        'total_chunks': 0,
        'total_gemini': 0,
        'matched': 0,
        'missing': 0,
        'extra_in_gemini': 0
    }
    
    # Get all file numbers from both sources
    all_file_nums = sorted(set(chunks.keys()) | set(gemini_data.keys()))
    
    print(f"Comparing {len(all_file_nums)} files...")
    print()
    
    for file_num in all_file_nums:
        chunk_list = chunks.get(file_num, [])
        gemini_list = gemini_data.get(file_num, [])
        
        stats['total_chunks'] += len(chunk_list)
        stats['total_gemini'] += len(gemini_list)
        
        if not chunk_list and not gemini_list:
            continue
        
        # Extract question texts
        chunk_questions = [extract_chunk_question(c) for c in chunk_list]
        gemini_questions = [extract_question_text(q) for q in gemini_list]
        
        # Also get question IDs for better matching
        chunk_ids = [c.get('id', '') if isinstance(c, dict) else '' for c in chunk_list]
        gemini_ids = [q.get('question_id', '') if isinstance(q, dict) else '' for q in gemini_list]
        
        # Normalize IDs: extract just the paper number and question number
        # Formats seen:
        #   - Question_Paper_9_p51_q138
        #   - Question_Paper_9_138
        #   - NEET_Q_8_P23_56
        #   - Question_Paper_07_123 (with leading zeros)
        # Should all normalize to: 9_138, 9_138, 8_56, 7_123 (no leading zeros)
        def normalize_id(qid):
            """Extract paper number and question number, ignoring prefix and page info, strip leading zeros"""
            if not qid:
                return None
            parts = qid.split('_')
            
            # Extract numbers from the ID - we want the last pure number (question)
            # and the second-to-last paper-related number
            numbers = []
            for part in parts:
                # Check if part is a pure number or starts with p/P/q/Q followed by number
                if part.isdigit():
                    # Remove leading zeros from the number
                    numbers.append(str(int(part)))
                elif len(part) > 1 and part[0].lower() in ['p', 'q'] and part[1:].isdigit():
                    # Extract the number part (removing leading zeros), mark if it's a Q (question number)
                    numbers.append((str(int(part[1:])), part[0].lower()))
            
            # The last number should be the question number
            # The paper number should be the last pure number before the question
            if not numbers:
                return None
            
            question_num = None
            paper_num = None
            
            # Work backwards to find question number and paper number
            for item in reversed(numbers):
                if isinstance(item, tuple):
                    num, prefix = item
                    if prefix == 'q' and question_num is None:
                        question_num = num
                    # Skip 'p' (page numbers)
                elif question_num is None:
                    # First pure number from the end is the question number
                    question_num = item
                elif paper_num is None:
                    # Second number from the end is the paper number
                    paper_num = item
                    break
            
            if paper_num and question_num:
                return f"{paper_num}_{question_num}"
            return None
        
        # Build normalized gemini ID set
        gemini_id_normalized = {normalize_id(gid): gid for gid in gemini_ids if gid}
        
        # Find missing chunks (questions in chunks but not in gemini)
        missing_indices = []
        for i, (chunk_q, chunk_id) in enumerate(zip(chunk_questions, chunk_ids)):
            if not chunk_q:  # Skip empty questions
                continue
            
            # Normalize and compare IDs
            found = False
            if chunk_id:
                normalized_chunk_id = normalize_id(chunk_id)
                if normalized_chunk_id and normalized_chunk_id in gemini_id_normalized:
                    found = True
                    stats['matched'] += 1
            
            if not found:
                # Fall back to text comparison (normalize whitespace and punctuation)
                chunk_q_norm = ' '.join(chunk_q.split()).lower()
                # Remove extra spaces, convert to lowercase
                chunk_q_norm = chunk_q_norm.replace('\n', ' ').strip()
                
                # Try partial match (first 50 chars minimum)
                chunk_start = chunk_q_norm[:50] if len(chunk_q_norm) > 50 else chunk_q_norm
                
                for gemini_q in gemini_questions:
                    gemini_q_norm = ' '.join(gemini_q.split()).lower()
                    gemini_q_norm = gemini_q_norm.replace('\n', ' ').strip()
                    
                    # Check if the beginning matches (allowing for formatting differences)
                    if gemini_q_norm.startswith(chunk_start) or chunk_start in gemini_q_norm[:100]:
                        found = True
                        stats['matched'] += 1
                        break
            
            if not found:
                missing_indices.append(i + 1)  # 1-indexed for readability
                stats['missing'] += 1
        
        # Check for extra questions in Gemini (not in chunks)
        if len(gemini_list) > len(chunk_list):
            stats['extra_in_gemini'] += len(gemini_list) - len(chunk_list)
        
        if missing_indices:
            missing_by_file[file_num] = {
                'total_chunks': len(chunk_list),
                'total_gemini': len(gemini_list),
                'missing_indices': missing_indices,
                'missing_count': len(missing_indices)
            }
    
    return missing_by_file, stats

def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare Gemini 2.5 Pro Data with source chunks to find missing questions'
    )
    parser.add_argument('--chunks-dir', 
                       default=r'c:\Users\quekd\OneDrive\Documents\GitHub\ANEETAA\aneeta_v2\Processed Data\Chunks',
                       help='Directory containing chunk files')
    parser.add_argument('--gemini-dir',
                       default=r'c:\Users\quekd\OneDrive\Documents\GitHub\ANEETAA\aneeta_v2\Processed Data\Gemini 2.5 Pro Data',
                       help='Directory containing Gemini 2.5 Pro Data files')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Show detailed missing question indices')
    parser.add_argument('--export', '-e', type=str,
                       help='Export results to JSON file')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPARING GEMINI DATA WITH SOURCE CHUNKS")
    print("=" * 80)
    print()
    
    # Load chunks
    print("STEP 1: Loading source chunks...")
    print("-" * 80)
    chunks = load_chunks(args.chunks_dir)
    
    if not chunks:
        print("Error: No chunks loaded. Cannot continue.")
        sys.exit(1)
    
    # Load Gemini data
    print("STEP 2: Loading Gemini 2.5 Pro Data...")
    print("-" * 80)
    gemini_data, failed_files = load_gemini_data(args.gemini_dir)
    
    if not gemini_data:
        print("Error: No Gemini data loaded. Cannot continue.")
        sys.exit(1)
    
    # Compare
    print("STEP 3: Comparing questions...")
    print("-" * 80)
    missing_by_file, stats = compare_questions(chunks, gemini_data)
    
    # Print results
    print()
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"Total questions in chunks:     {stats['total_chunks']}")
    print(f"Total questions in Gemini:     {stats['total_gemini']}")
    print(f"Matched questions:             {stats['matched']}")
    print(f"Missing questions:             {stats['missing']}")
    print(f"Extra questions in Gemini:     {stats['extra_in_gemini']}")
    print()
    
    if missing_by_file:
        print("=" * 80)
        print(f"FILES WITH MISSING QUESTIONS ({len(missing_by_file)} files)")
        print("=" * 80)
        
        total_missing = 0
        for file_num in sorted(missing_by_file.keys()):
            info = missing_by_file[file_num]
            total_missing += info['missing_count']
            print(f"\n📁 File {file_num:3d}")
            print(f"   Chunks: {info['total_chunks']} | Gemini: {info['total_gemini']} | Missing: {info['missing_count']}")
            
            if args.detailed:
                missing_str = ', '.join(map(str, info['missing_indices'][:30]))
                if len(info['missing_indices']) > 30:
                    missing_str += f" ... and {len(info['missing_indices']) - 30} more"
                print(f"   Missing question indices: {missing_str}")
        
        print()
        print("=" * 80)
        print(f"TOTAL MISSING QUESTIONS: {total_missing}")
        print("=" * 80)
    else:
        print("✅ No missing questions found! All chunks are present in Gemini data.")
    
    # Export if requested
    if args.export:
        export_data = {
            'stats': stats,
            'missing_by_file': missing_by_file,
            'failed_files': failed_files
        }
        
        with open(args.export, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✓ Results exported to: {args.export}")
    
    print()

if __name__ == '__main__':
    main()
