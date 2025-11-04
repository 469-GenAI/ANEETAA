"""
Analyze JSON file sizes and flag outliers.
Calculates statistics and identifies files that are significantly larger or smaller than average.
"""

import os
import sys
from pathlib import Path
import statistics

def count_lines(file_path):
    """Count lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def analyze_directory(directory):
    """
    Analyze all JSON files in a directory for size outliers.
    
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
    
    print(f"Analyzing {len(json_files)} JSON files...")
    print()
    
    # Count lines for each file
    file_stats = []
    for json_file in json_files:
        line_count = count_lines(json_file)
        if line_count is not None:
            file_stats.append({
                'name': json_file.name,
                'lines': line_count
            })
    
    if not file_stats:
        print("No files could be analyzed")
        return
    
    # Calculate statistics
    line_counts = [f['lines'] for f in file_stats]
    mean_lines = statistics.mean(line_counts)
    median_lines = statistics.median(line_counts)
    stdev_lines = statistics.stdev(line_counts) if len(line_counts) > 1 else 0
    min_lines = min(line_counts)
    max_lines = max(line_counts)
    
    # Define outlier thresholds (2 standard deviations)
    lower_threshold = mean_lines - (2 * stdev_lines)
    upper_threshold = mean_lines + (2 * stdev_lines)
    
    # Also flag files that are less than 50% or more than 200% of mean
    lower_threshold_pct = mean_lines * 0.5
    upper_threshold_pct = mean_lines * 2.0
    
    # Use the more conservative threshold
    lower_threshold = max(lower_threshold, lower_threshold_pct)
    upper_threshold = min(upper_threshold, upper_threshold_pct)
    
    # Find outliers
    outliers_low = [f for f in file_stats if f['lines'] < lower_threshold]
    outliers_high = [f for f in file_stats if f['lines'] > upper_threshold]
    
    # Sort outliers by line count
    outliers_low.sort(key=lambda x: x['lines'])
    outliers_high.sort(key=lambda x: x['lines'], reverse=True)
    
    # Print statistics
    print("=" * 80)
    print("FILE SIZE STATISTICS")
    print("=" * 80)
    print(f"Total files analyzed:  {len(file_stats)}")
    print(f"Average lines:         {mean_lines:.1f}")
    print(f"Median lines:          {median_lines:.1f}")
    print(f"Standard deviation:    {stdev_lines:.1f}")
    print(f"Minimum lines:         {min_lines}")
    print(f"Maximum lines:         {max_lines}")
    print()
    print(f"Lower threshold:       {lower_threshold:.1f} lines (files below this are flagged)")
    print(f"Upper threshold:       {upper_threshold:.1f} lines (files above this are flagged)")
    print("=" * 80)
    print()
    
    # Print outliers
    if outliers_low:
        print("=" * 80)
        print(f"SUSPICIOUSLY SMALL FILES ({len(outliers_low)} files)")
        print("=" * 80)
        for f in outliers_low:
            deviation = ((f['lines'] - mean_lines) / mean_lines) * 100
            print(f"⚠ {f['name']:20s} - {f['lines']:6d} lines ({deviation:+.1f}% from average)")
        print()
    
    if outliers_high:
        print("=" * 80)
        print(f"SUSPICIOUSLY LARGE FILES ({len(outliers_high)} files)")
        print("=" * 80)
        for f in outliers_high:
            deviation = ((f['lines'] - mean_lines) / mean_lines) * 100
            print(f"⚠ {f['name']:20s} - {f['lines']:6d} lines ({deviation:+.1f}% from average)")
        print()
    
    if not outliers_low and not outliers_high:
        print("✓ No outliers detected - all files are within normal size range")
        print()
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Normal files:          {len(file_stats) - len(outliers_low) - len(outliers_high)}")
    print(f"Suspiciously small:    {len(outliers_low)}")
    print(f"Suspiciously large:    {len(outliers_high)}")
    print("=" * 80)
    
    # Show distribution
    print()
    print("LINE COUNT DISTRIBUTION:")
    print("-" * 80)
    
    # Create bins
    bins = [
        (0, 500, "0-500"),
        (501, 1000, "501-1000"),
        (1001, 1500, "1001-1500"),
        (1501, 2000, "1501-2000"),
        (2001, 2500, "2001-2500"),
        (2501, 3000, "2501-3000"),
        (3001, float('inf'), "3000+")
    ]
    
    for min_val, max_val, label in bins:
        count = sum(1 for f in file_stats if min_val <= f['lines'] <= max_val)
        if count > 0:
            bar = "█" * count
            print(f"{label:12s}: {bar} ({count} files)")

def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze JSON file sizes and flag outliers'
    )
    parser.add_argument('directory', nargs='?',
                       default=r'c:\Users\quekd\OneDrive\Documents\GitHub\ANEETAA\aneeta_v2\Processed Data\Gemini 2.5 Pro Data',
                       help='Directory containing JSON files to analyze')
    
    args = parser.parse_args()
    
    analyze_directory(args.directory)

if __name__ == '__main__':
    main()
