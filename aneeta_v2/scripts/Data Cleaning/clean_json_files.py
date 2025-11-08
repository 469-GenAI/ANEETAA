"""
Clean JSON files by removing markdown code block markers.
Removes first line if it starts with 'json' and last line if it ends with '```'.
"""

import os
import sys
from pathlib import Path

def clean_json_file(file_path):
    """
    Clean a JSON file by removing markdown code block markers.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return False
        
        modified = False
        
        # Check if first line starts with 'json' (case insensitive, with optional whitespace/backticks)
        first_line = lines[0].strip()
        if first_line.startswith('```json') or first_line.startswith('json') or first_line == '```':
            lines = lines[1:]
            modified = True
        
        # Check if last line ends with '```'
        if lines and lines[-1].strip().endswith('```'):
            lines = lines[:-1]
            modified = True
        
        # Write back if modified
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def clean_directory(directory):
    """
    Clean all JSON files in a directory.
    
    Args:
        directory: Path to directory
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Error: Directory does not exist: {directory}")
        sys.exit(1)
    
    if not dir_path.is_dir():
        print(f"Error: Path is not a directory: {directory}")
        sys.exit(1)
    
    # Find all JSON files
    json_files = sorted(dir_path.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in: {directory}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    print("Processing...\n")
    
    modified_count = 0
    skipped_count = 0
    
    for json_file in json_files:
        was_modified = clean_json_file(json_file)
        
        if was_modified:
            print(f"✓ Cleaned: {json_file.name}")
            modified_count += 1
        else:
            skipped_count += 1
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {len(json_files)}")
    print(f"Files cleaned:         {modified_count}")
    print(f"Files unchanged:       {skipped_count}")
    print("=" * 70)

def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Clean JSON files by removing markdown code block markers'
    )
    parser.add_argument('directory', nargs='?',
                       default=r'c:\Users\quekd\OneDrive\Documents\GitHub\ANEETAA\aneeta_v2\Processed Data\Gemini 2.5 Pro Data',
                       help='Directory containing JSON files to clean')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CLEANING JSON FILES")
    print("=" * 70)
    print(f"Directory: {args.directory}")
    print()
    
    clean_directory(args.directory)

if __name__ == '__main__':
    main()
