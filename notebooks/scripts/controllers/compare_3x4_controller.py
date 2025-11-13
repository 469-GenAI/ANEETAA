"""
3×4 Matrix Comparison Controller

Easy-to-use controller for running the 3×4 matrix comparison
(3 models × 4 agent types = 12 configurations including MIPROv2)

Just run: python notebooks/scripts/controllers/compare_3x4_controller.py
"""

import subprocess
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

COMPARISON_CONFIG = {
    # Test Configuration
    'test_samples': 40,              # Number of test questions
    'seed': 123,                     # Random seed (changed from 42)
    'use_validation_set': True,      # Use validation dataset
}

# ============================================================================
# PRESET CONFIGURATIONS - Uncomment to use
# ============================================================================

# Quick test (10 questions)
# COMPARISON_CONFIG = {
#     'test_samples': 10,
#     'seed': 42,
#     'use_validation_set': True,
# }

# Medium test (30 questions)
# COMPARISON_CONFIG = {
#     'test_samples': 30,
#     'seed': 42,
#     'use_validation_set': True,
# }

# Full test (100 questions)
# COMPARISON_CONFIG = {
#     'test_samples': 100,
#     'seed': 42,
#     'use_validation_set': True,
# }

# ============================================================================
# Functions
# ============================================================================

def check_prerequisites():
    """Check if all required model files exist."""
    ROOT = Path(__file__).parent.parent.parent.parent.resolve()  # Up to project root
    
    required_models = [
        'models/dspy_bootstrap_llama3.1_8b.json',
        'models/dspy_bootstrap_gemma2_9b.json',
        'models/dspy_bootstrap_mistral_nemo_12b.json',
        'models/dspy_mipro_llama3.1_8b.json',
        'models/dspy_mipro_gemma2_9b.json',
        'models/dspy_mipro_mistral_nemo_12b.json',
    ]
    
    missing = []
    existing = []
    
    for model_path in required_models:
        full_path = ROOT / model_path
        if full_path.exists():
            existing.append(model_path)
        else:
            missing.append(model_path)
    
    return existing, missing


def build_command(config):
    """Build command from configuration."""
    script_path = Path(__file__).parent.parent / "runners" / "compare_3x4_matrix.py"
    
    cmd = [sys.executable, str(script_path)]
    cmd.extend(['--test-samples', str(config['test_samples'])])
    cmd.extend(['--seed', str(config['seed'])])
    
    if config.get('use_validation_set', True):
        cmd.append('--use-validation-set')
    
    return cmd


def print_config(config, existing_models, missing_models):
    """Pretty print configuration."""
    print("="*70)
    print("3×4 MATRIX COMPARISON CONTROLLER")
    print("="*70)
    print("\n📊 Comparison Configuration:")
    print("-" * 70)
    
    # Test settings
    print("\nTest Settings:")
    print(f"  Test Samples: {config['test_samples']} questions")
    dataset_source = "Validation dataset (val.jsonl)" if config.get('use_validation_set', True) else "Combined dataset"
    print(f"  Dataset Source: {dataset_source}")
    print(f"  Random Seed: {config['seed']}")
    
    # Matrix dimensions
    print("\n🔢 Matrix Dimensions:")
    print(f"  Models: 3 (llama3.1:8b, gemma2:9b, mistral-nemo:12b)")
    print(f"  Agent Types: 4 (Vanilla, Baseline, Bootstrap, MIPROv2)")
    print(f"  Total Configurations: 12")
    print(f"  Total Evaluations: {config['test_samples']} × 12 = {config['test_samples'] * 12}")
    
    # Agent types
    print("\n🤖 Agent Types:")
    print("  1️⃣  Vanilla ANEETAA - Original LangChain implementation")
    print("  2️⃣  DSPy Baseline - Unoptimized DSPy")
    print("  3️⃣  DSPy Bootstrap - BootstrapFewShot optimizer")
    print("  4️⃣  DSPy MIPROv2 - Multi-prompt instruction optimizer (NEW!)")
    
    # Model status
    print("\n📦 Model Status:")
    print(f"  Bootstrap models: {sum(1 for m in existing_models if 'bootstrap' in m)}/3")
    print(f"  MIPROv2 models: {sum(1 for m in existing_models if 'mipro' in m)}/3")
    
    if missing_models:
        print("\n⚠️  Missing Models:")
        for model in missing_models:
            print(f"     ❌ {model}")
        
        if any('bootstrap' in m for m in missing_models):
            print("\n  To train Bootstrap models:")
            print("     python notebooks/scripts/train_all_models.py")
        
        if any('mipro' in m for m in missing_models):
            print("\n  To train MIPROv2 models:")
            print("     python notebooks/scripts/train_all_models_miprov2.py")
    else:
        print("  ✅ All 6 models found!")
    
    # Output
    print("\n💾 Output:")
    print(f"  Detailed Results: results/3x4_matrix_detailed_results.csv")
    print(f"  Summary: results/3x4_matrix_summary.csv")
    print(f"  MLflow Experiment: aneetaa-3x4-matrix-comparison")
    
    # Time estimate
    avg_time_per_eval = 4  # seconds
    total_time_sec = config['test_samples'] * 12 * avg_time_per_eval
    total_time_min = total_time_sec / 60
    print(f"\n⏱️  Estimated Time: ~{total_time_min:.1f} minutes")
    
    print("\n" + "="*70)


def main():
    """Run 3×4 matrix comparison."""
    # Check prerequisites
    existing_models, missing_models = check_prerequisites()
    
    # Print configuration
    print_config(COMPARISON_CONFIG, existing_models, missing_models)
    
    # Check if we can proceed
    if missing_models:
        print("\n⚠️  WARNING: Some models are missing!")
        print("   The comparison will still run but may show errors for missing models.")
        print("\n   Continue anyway?")
        response = input("\n▶ Start comparison? [y/N]: ").strip().lower()
        
        if response not in ['y', 'yes']:
            print("\n❌ Comparison cancelled")
            print("\nNext steps:")
            if any('bootstrap' in m for m in missing_models):
                print("  1. Train Bootstrap models: python notebooks/scripts/train_all_models.py")
            if any('mipro' in m for m in missing_models):
                print("  2. Train MIPROv2 models: python notebooks/scripts/train_all_models_miprov2.py")
            print("  3. Run comparison again")
            return
    else:
        response = input("\n▶ Start comparison? [Y/n]: ").strip().lower()
        
        if response not in ['', 'y', 'yes']:
            print("\n❌ Comparison cancelled")
            return
    
    # Build command
    cmd = build_command(COMPARISON_CONFIG)
    
    print("\n🚀 Starting 3×4 matrix comparison...")
    print("\nCommand:")
    print(" ".join(cmd))
    print("\n" + "="*70 + "\n")
    
    # Run comparison
    try:
        subprocess.run(cmd, check=True)
        
        print("\n" + "="*70)
        print("✅ COMPARISON COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n📊 Results saved to:")
        print("   results/3x4_matrix_detailed_results.csv")
        print("   results/3x4_matrix_summary.csv")
        print("\n📈 To view MLflow results, run:")
        print("   mlflow ui --port 8080")
        print("\nThen open: http://localhost:8080")
        print("="*70 + "\n")
        
    except subprocess.CalledProcessError as e:
        print("\n" + "="*70)
        print("❌ COMPARISON FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        print("\nCommon issues:")
        print("  • Ollama not running")
        print("  • Model files not found")
        print("  • Insufficient memory")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Comparison interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
