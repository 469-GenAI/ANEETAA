"""
Train DSPy MIPROv2 Optimized Models for All 3 Ollama Models

This script trains MIPROv2-optimized DSPy models for:
- llama3.1:8b
- gemma2:9b
- mistral-nemo:12b

MIPROv2 optimizes both instructions AND demonstrations (vs Bootstrap which only optimizes demos).
Expected to take 2-3x longer than Bootstrap but should produce better results.

Each model will be trained and saved separately for use in the 3×4 comparison.
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
        'save_path': 'models/dspy_mipro_llama3.1_8b.json',
        'display_name': 'Llama 3.1 8B'
    },
    {
        'name': 'gemma2:9b',
        'save_path': 'models/dspy_mipro_gemma2_9b.json',
        'display_name': 'Gemma2 9B'
    },
    {
        'name': 'mistral-nemo:12b',
        'save_path': 'models/dspy_mipro_mistral_nemo_12b.json',
        'display_name': 'Mistral Nemo 12B'
    }
]

TRAINING_CONFIG = {
    # Data Configuration - Use split train/val datasets
    'use_split_datasets': True,
    'train_samples': 200,           # FASTER: Reduced from 300 (saves ~20% time)
    'val_samples': 40,              # Keep validation set consistent
    
    'seed': 42,
    'provider': 'ollama',           # Use Ollama (FREE!) - or 'openai' for potentially better quality
    'method': 'mipro',              # MIPROv2 optimizer
    'max_demos': 3,                 # FASTER: Reduced from 4 (fewer demos to optimize)
    'candidates': 5,                # FASTER: Reduced from 7 (fewer candidates = ~30% faster)
    
    # PERFORMANCE TUNING NOTES:
    # - 'candidates': 5 instead of 7 saves ~30% time with minimal quality loss
    # - 'train_samples': 200 instead of 300 saves ~20% time
    # - 'max_demos': 3 instead of 4 reduces prompt optimization overhead
    # - Total time reduction: ~40-50% (from 7h → 3.5-4h for all 3 models)
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def build_train_command(model_config, training_config):
    """Build training command for a specific model."""
    script_path = Path(__file__).parent / "train_mcq_solver.py"
    
    cmd = [sys.executable, str(script_path)]
    
    # Add split dataset arguments
    if training_config.get('use_split_datasets', False):
        cmd.append('--use-split-datasets')
        cmd.extend(['--train-samples', str(training_config.get('train_samples', 300))])
        cmd.extend(['--val-samples', str(training_config.get('val_samples', 40))])
    else:
        # Fallback to old method
        cmd.extend(['--questions', str(training_config.get('questions', 300))])
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
    print("TRAIN ALL OLLAMA MODELS WITH MIPROv2 OPTIMIZER")
    print("="*70)
    print("\nThis will train DSPy MIPROv2-optimized models for:")
    for i, model in enumerate(MODELS_TO_TRAIN, 1):
        print(f"  {i}. {model['display_name']} ({model['name']})")
    
    print(f"\n📊 Training Configuration:")
    if TRAINING_CONFIG.get('use_split_datasets', False):
        train_samples = TRAINING_CONFIG.get('train_samples', 300)
        val_samples = TRAINING_CONFIG.get('val_samples', 40)
        print(f"  Dataset: Split train/val datasets")
        print(f"  Train: {train_samples} questions (from train.jsonl)")
        print(f"  Val: {val_samples} questions (from val.jsonl)")
        print(f"  Total per model: {train_samples + val_samples} questions")
    else:
        questions = TRAINING_CONFIG.get('questions', 300)
        test_split = TRAINING_CONFIG.get('test_split', 0.2)
        print(f"  Questions per model: {questions} ({int(questions*(1-test_split))} train, {int(questions*test_split)} test)")
    
    print(f"  Provider: {TRAINING_CONFIG['provider']}")
    print(f"  Method: MIPROv2")
    print(f"  Instruction Candidates: {TRAINING_CONFIG['candidates']}")
    print(f"  Seed: {TRAINING_CONFIG['seed']}")
    
    print(f"\n⏱️  Expected time: ~90-135 minutes per model (~4.5-7 hours total)")
    print(f"💰 Expected cost: $0 (using Ollama)")
    
    print("\n⚠️  NOTE: MIPROv2 takes 2-3x longer than Bootstrap but produces better results")
    print("   It optimizes both instructions AND demonstrations (vs Bootstrap = demos only)")
    print("="*70)
    
    response = input("\n▶ Start MIPROv2 training for all models? [Y/n]: ").strip().lower()
    
    if response not in ['', 'y', 'yes']:
        print("\n❌ Training cancelled")
        return
    
    print("\n🚀 Starting MIPROv2 training for all models...\n")
    
    # Train each model
    results = []
    
    for i, model_config in enumerate(MODELS_TO_TRAIN, 1):
        print("\n" + "="*70)
        print(f"TRAINING MODEL {i}/{len(MODELS_TO_TRAIN)}: {model_config['display_name']}")
        print(f"Optimizer: MIPROv2")
        print("="*70)
        
        cmd = build_train_command(model_config, TRAINING_CONFIG)
        
        print(f"Command: {' '.join(cmd)}")
        print("\n" + "-"*70 + "\n")
        
        try:
            # Run training
            result = subprocess.run(cmd, check=True)
            
            print(f"\n✅ {model_config['display_name']} MIPROv2 training completed!")
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
    print("MIPROv2 TRAINING SUMMARY")
    print("="*70)
    
    for result in results:
        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_icon} {result['model']}: {result['status']}")
        if result['status'] == 'SUCCESS':
            print(f"   Saved to: {result['path']}")
    
    successful = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\n✅ {successful}/{len(MODELS_TO_TRAIN)} MIPROv2 models trained successfully")
    
    if successful == len(MODELS_TO_TRAIN):
        print("\n🎉 All MIPROv2 models trained! Ready for 3×4 matrix comparison!")
        print("\n📊 You now have:")
        print("   - 3 Bootstrap-optimized models (from train_all_models.py)")
        print("   - 3 MIPROv2-optimized models (from this script)")
        print("\nNext step: Run the 3×4 comparison script")
        print("   python notebooks/scripts/compare_3x4_matrix.py")
    
    print("="*70)


if __name__ == "__main__":
    main()
