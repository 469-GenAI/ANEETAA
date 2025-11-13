"""
Training Controller - Run DSPy training with predefined configurations

This controller allows you to run training with predefined parameter sets.
Just run: python notebooks/scripts/controllers/train_controller.py

Modify the TRAINING_CONFIG below to change parameters.
"""

import subprocess
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================

TRAINING_CONFIG = {
    # Data Configuration - Choose ONE option:
    # Option A: Use split train/val datasets (RECOMMENDED)
    'use_split_datasets': True,     # Use separate train.jsonl and val.jsonl
    'train_samples': 200,           # Questions from train.jsonl
    'val_samples': 40,              # Questions from val.jsonl
    
    # Option B: Use combined dataset (OLD METHOD - comment out if using Option A)
    # 'use_combined': True,         # Use dspy_dataset_combined.jsonl (7,874 questions)
    # 'questions': 100,             # Number of questions to use
    # 'test_split': 0.2,            # Test set ratio (0.2 = 20% test, 80% train)
    
    'seed': 42,                      # Random seed for reproducibility
    
    # Model Configuration
    'provider': 'ollama',           # 'openai' or 'ollama' (using ollama for FREE training)
    'model': 'gemma2:9b',           # Model name (e.g., 'gpt-4o-mini', 'gpt-4', 'gemma2:9b')
    
    # Training Configuration
    'method': 'bootstrap',          # 'baseline', 'bootstrap', 'mipro', or 'all'
    'max_demos': 4,                 # Max few-shot examples for Bootstrap
    'candidates': 7,                # Number of candidates for MIPRO
    
    # Output Configuration
    'save_path': 'dspy_mcq_optimized.json',  # Model save filename
}

# ============================================================================
# PRESET CONFIGURATIONS - Uncomment to use
# ============================================================================

# Quick test (fast, for testing)
# TRAINING_CONFIG = {
#     'use_combined': True,
#     'questions': 50,
#     'test_split': 0.2,
#     'seed': 42,
#     'provider': 'openai',
#     'model': 'gpt-4o-mini',
#     'method': 'bootstrap',
#     'max_demos': 2,
#     'candidates': 5,
#     'save_path': 'dspy_mcq_optimized.json',
# }

# Medium training (balanced)
# TRAINING_CONFIG = {
#     'use_combined': True,
#     'questions': 500,
#     'test_split': 0.2,
#     'seed': 42,
#     'provider': 'openai',
#     'model': 'gpt-4o-mini',
#     'method': 'bootstrap',
#     'max_demos': 6,
#     'candidates': 7,
#     'save_path': 'dspy_mcq_optimized.json',
# }

# Full dataset training (slow, best results)
# TRAINING_CONFIG = {
#     'use_combined': True,
#     'questions': 7874,
#     'test_split': 0.15,
#     'seed': 42,
#     'provider': 'openai',
#     'model': 'gpt-4o-mini',
#     'method': 'bootstrap',
#     'max_demos': 8,
#     'candidates': 10,
#     'save_path': 'dspy_mcq_optimized.json',
# }

# Compare all methods
# TRAINING_CONFIG = {
#     'use_combined': True,
#     'questions': 200,
#     'test_split': 0.2,
#     'seed': 42,
#     'provider': 'openai',
#     'model': 'gpt-4o-mini',
#     'method': 'all',  # Trains baseline, bootstrap, AND mipro
#     'max_demos': 4,
#     'candidates': 7,
#     'save_path': 'dspy_mcq_optimized.json',
# }

# ============================================================================
# Controller Logic - Don't modify below this line
# ============================================================================

def build_command(config):
    """Build command from configuration."""
    script_path = Path(__file__).parent.parent / "runners" / "train_mcq_solver.py"
    
    cmd = [sys.executable, str(script_path)]
    
    # Check if using split datasets or combined dataset
    if config.get('use_split_datasets', False):
        # New split dataset approach
        cmd.append('--use-split-datasets')
        cmd.extend(['--train-samples', str(config.get('train_samples', 200))])
        cmd.extend(['--val-samples', str(config.get('val_samples', 40))])
    else:
        # Old combined dataset approach
        cmd.extend(['--questions', str(config.get('questions', 100))])
        cmd.extend(['--test-split', str(config.get('test_split', 0.2))])
        if config.get('use_combined', False):
            cmd.append('--use-combined')
    
    # Add common arguments
    cmd.extend(['--provider', config['provider']])
    cmd.extend(['--model', config['model']])
    cmd.extend(['--method', config['method']])
    cmd.extend(['--max-demos', str(config['max_demos'])])
    cmd.extend(['--candidates', str(config['candidates'])])
    cmd.extend(['--save-path', config['save_path']])
    cmd.extend(['--seed', str(config['seed'])])
    
    return cmd


def print_config(config):
    """Pretty print configuration."""
    print("="*70)
    print("DSPy TRAINING CONTROLLER")
    print("="*70)
    print("\nTraining Configuration:")
    print("-" * 70)
    
    # Data settings
    print("\n📊 Data Settings:")
    if config.get('use_split_datasets', False):
        print(f"  Dataset: Split train/val datasets")
        print(f"  Train Samples: {config.get('train_samples', 200)} (from train.jsonl)")
        print(f"  Val Samples: {config.get('val_samples', 40)} (from val.jsonl)")
        print(f"  Total: {config.get('train_samples', 200) + config.get('val_samples', 40)} questions")
    else:
        print(f"  Dataset: {'Combined (7,874 questions)' if config.get('use_combined') else 'Gemini Data'}")
        print(f"  Questions: {config.get('questions', 100)}")
        test_split = config.get('test_split', 0.2)
        questions = config.get('questions', 100)
        print(f"  Test Split: {test_split*100:.0f}% ({int(questions*test_split)} test, {int(questions*(1-test_split))} train)")
    print(f"  Random Seed: {config['seed']}")
    
    # Model settings
    print("\n🤖 Model Settings:")
    print(f"  Provider: {config['provider']}")
    print(f"  Model: {config['model']}")
    
    # Training settings
    print("\n🏋️ Training Settings:")
    print(f"  Method: {config['method'].upper()}")
    if config['method'] in ['bootstrap', 'all']:
        print(f"  Max Few-Shot Demos: {config['max_demos']}")
    if config['method'] in ['mipro', 'all']:
        print(f"  MIPRO Candidates: {config['candidates']}")
    
    # Output
    print("\n💾 Output:")
    print(f"  Save Path: models/{config['save_path']}")
    
    print("\n" + "="*70 + "\n")


def main():
    """Run training with configured parameters."""
    # Print configuration
    print_config(TRAINING_CONFIG)
    
    # Build command
    cmd = build_command(TRAINING_CONFIG)
    
    # Confirm
    print("Command that will be executed:")
    print(" ".join(cmd))
    print("\n" + "="*70)
    
    response = input("\n▶ Start training? [Y/n]: ").strip().lower()
    
    if response in ['', 'y', 'yes']:
        print("\n🚀 Starting training...\n")
        print("="*70 + "\n")
        
        # Run training
        try:
            subprocess.run(cmd, check=True)
            
            print("\n" + "="*70)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print("="*70)
            print("\n📈 To view results, run:")
            print("   mlflow ui --port 8080")
            print("\nThen open: http://localhost:8080")
            print("="*70 + "\n")
            
        except subprocess.CalledProcessError as e:
            print("\n" + "="*70)
            print("❌ TRAINING FAILED!")
            print("="*70)
            print(f"\nError: {e}")
            sys.exit(1)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            sys.exit(1)
    else:
        print("\n❌ Training cancelled by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
