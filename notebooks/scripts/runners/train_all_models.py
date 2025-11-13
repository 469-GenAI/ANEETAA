"""
Train DSPy Optimized Models for All 3 Ollama Models

This script trains Bootstrap-optimized DSPy models for:
- llama3.1:8b
- gemma2:9b
- mistral-nemo:12b

Each model will be trained and saved separately for use in the 3×3 comparison.
"""

import subprocess
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_TO_TRAIN = [
    {
        'name': 'llama3.1:8b',
        'save_path': 'models/dspy_bootstrap_llama3.1_8b.json',
        'display_name': 'Llama 3.1 8B'
    },
    {
        'name': 'gemma2:9b',
        'save_path': 'models/dspy_bootstrap_gemma2_9b.json',
        'display_name': 'Gemma2 9B'
    },
    {
        'name': 'mistral-nemo:12b',
        'save_path': 'models/dspy_bootstrap_mistral_nemo_12b.json',
        'display_name': 'Mistral Nemo 12B'
    }
]

TRAINING_CONFIG = {
    # Data Configuration - Choose ONE option:
    # Option A: Use split train/val datasets (RECOMMENDED)
    'use_split_datasets': True,     # Use separate train.jsonl and val.jsonl
    'train_samples': 200,           # Questions from train.jsonl
    'val_samples': 40,              # Questions from val.jsonl
    
    # Option B: Use combined dataset (OLD METHOD - comment out if using Option A)
    # 'use_combined': True,
    # 'questions': 20,              # Number of questions total
    # 'test_split': 0.2,
    
    'seed': 42,
    'provider': 'ollama',           # Use Ollama (FREE!)
    'method': 'bootstrap',
    'max_demos': 2,                 # Reduced demos for faster training
    'candidates': 7,
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def build_train_command(model_config, training_config):
    """Build training command for a specific model."""
    script_path = Path(__file__).parent / "train_mcq_solver.py"
    
    cmd = [sys.executable, str(script_path)]
    
    # Check if using split datasets or combined dataset
    if training_config.get('use_split_datasets', False):
        # New split dataset approach
        cmd.append('--use-split-datasets')
        cmd.extend(['--train-samples', str(training_config.get('train_samples', 200))])
        cmd.extend(['--val-samples', str(training_config.get('val_samples', 40))])
    else:
        # Old combined dataset approach
        cmd.extend(['--questions', str(training_config.get('questions', 20))])
        cmd.extend(['--test-split', str(training_config.get('test_split', 0.2))])
        if training_config.get('use_combined'):
            cmd.append('--use-combined')
    
    # Add common arguments
    cmd.extend(['--provider', training_config['provider']])
    cmd.extend(['--model', model_config['name']])
    cmd.extend(['--method', training_config['method']])
    cmd.extend(['--max-demos', str(training_config['max_demos'])])
    cmd.extend(['--candidates', str(training_config['candidates'])])
    cmd.extend(['--save-path', model_config['save_path']])
    cmd.extend(['--seed', str(training_config['seed'])])
    
    return cmd


def main():
    print("="*70)
    print("TRAIN ALL OLLAMA MODELS FOR 3×3 MATRIX COMPARISON")
    print("="*70)
    print("\nThis will train DSPy Bootstrap-optimized models for:")
    for i, model in enumerate(MODELS_TO_TRAIN, 1):
        print(f"  {i}. {model['display_name']} ({model['name']})")
    
    print(f"\nTraining Configuration:")
    if TRAINING_CONFIG.get('use_split_datasets', False):
        train_samples = TRAINING_CONFIG.get('train_samples', 200)
        val_samples = TRAINING_CONFIG.get('val_samples', 40)
        print(f"  Dataset: Split train/val datasets")
        print(f"  Train: {train_samples} questions (from train.jsonl)")
        print(f"  Val: {val_samples} questions (from val.jsonl)")
        print(f"  Total per model: {train_samples + val_samples} questions")
    else:
        questions = TRAINING_CONFIG.get('questions', 20)
        test_split = TRAINING_CONFIG.get('test_split', 0.2)
        print(f"  Questions per model: {questions} ({int(questions*(1-test_split))} train, {int(questions*test_split)} test)")
    print(f"  Provider: {TRAINING_CONFIG['provider']} (FREE!)")
    print(f"  Method: {TRAINING_CONFIG['method']}")
    print(f"  Seed: {TRAINING_CONFIG['seed']}")
    
    print(f"\nExpected time: ~30-45 minutes per model (~90-135 min total)")
    print(f"Expected cost: $0 (using Ollama)")
    print("="*70)
    
    response = input("\n▶ Start training all models? [Y/n]: ").strip().lower()
    
    if response not in ['', 'y', 'yes']:
        print("\n❌ Training cancelled")
        return
    
    print("\n🚀 Starting training for all models...\n")
    
    # Train each model
    results = []
    
    for i, model_config in enumerate(MODELS_TO_TRAIN, 1):
        print("\n" + "="*70)
        print(f"TRAINING MODEL {i}/{len(MODELS_TO_TRAIN)}: {model_config['display_name']}")
        print("="*70)
        
        cmd = build_train_command(model_config, TRAINING_CONFIG)
        
        print(f"Command: {' '.join(cmd)}")
        print("\n" + "-"*70 + "\n")
        
        try:
            # Run training
            result = subprocess.run(cmd, check=True)
            
            print(f"\n✅ {model_config['display_name']} training completed!")
            print(f"   Model saved to: {model_config['save_path']}")
            
            results.append({
                'model': model_config['display_name'],
                'status': 'SUCCESS',
                'path': model_config['save_path']
            })
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {model_config['display_name']} training FAILED!")
            print(f"   Error: {e}")
            
            results.append({
                'model': model_config['display_name'],
                'status': 'FAILED',
                'error': str(e)
            })
            
            response = input("\n⚠️  Continue with remaining models? [Y/n]: ").strip().lower()
            if response not in ['', 'y', 'yes']:
                print("\n❌ Training stopped by user")
                break
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    for result in results:
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_icon} {result['model']}: {result['status']}")
        if result['status'] == 'SUCCESS':
            print(f"   Saved to: {result['path']}")
    
    successful = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\n✅ {successful}/{len(MODELS_TO_TRAIN)} models trained successfully")
    
    if successful == len(MODELS_TO_TRAIN):
        print("\n🎉 All models trained! Ready for 3×3 matrix comparison!")
        print("\nNext step: Run the 3×3 comparison script")
        print("   python notebooks/scripts/compare_3x3_matrix.py")
    
    print("="*70)


if __name__ == "__main__":
    main()
