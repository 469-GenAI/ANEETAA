"""
Split a JSONL file into multiple chunks.
"""

import json
import sys
from pathlib import Path

def split_jsonl_file(input_file, num_chunks=10, output_dir=None):
    """
    Split a JSONL file into multiple chunks.
    
    Args:
        input_file: Path to input JSONL file
        num_chunks: Number of chunks to create
        output_dir: Output directory (default: same as input file)
    """
    input_path = Path(input_file)
    
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = input_path.parent
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read all questions
    print(f"Reading questions from: {input_file}")
    questions = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    
    total_questions = len(questions)
    print(f"Total questions: {total_questions}")
    
    # Calculate chunk size
    chunk_size = total_questions // num_chunks
    remainder = total_questions % num_chunks
    
    print(f"Chunk size: ~{chunk_size} questions per chunk")
    print(f"\nSplitting into {num_chunks} chunks...")
    
    # Split into chunks
    base_name = input_path.stem
    start_idx = 0
    
    for i in range(num_chunks):
        # Calculate this chunk's size (distribute remainder across first chunks)
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        # Get chunk
        chunk = questions[start_idx:end_idx]
        
        # Write chunk to file
        chunk_file = output_path / f"{base_name}_chunk_{i+1:02d}.json"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            for q in chunk:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')
        
        print(f"  Chunk {i+1:2d}: {len(chunk):4d} questions → {chunk_file.name}")
        
        start_idx = end_idx
    
    print(f"\n✓ Successfully split into {num_chunks} chunks")
    print(f"  Output directory: {output_path}")

def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Split a JSONL file into multiple chunks'
    )
    parser.add_argument('input_file', 
                       help='Path to input JSONL file')
    parser.add_argument('--chunks', '-n', type=int, default=10,
                       help='Number of chunks to create (default: 10)')
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Output directory (default: same as input file)')
    
    args = parser.parse_args()
    
    split_jsonl_file(args.input_file, args.chunks, args.output_dir)

if __name__ == '__main__':
    main()
