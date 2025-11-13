"""
Comparison Controller - Run three-way agent comparison with predefined configurations

This controller allows you to run comparisons with predefined parameter sets.
Just run: python notebooks/scripts/controllers/compare_controller.py

Modify the COMPARISON_CONFIG below to change parameters.
"""

import subprocess
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================

# ============================================================================
# CONFIGURATION - Edit these parameters
# ============================================================================

# Quick test (3 questions) - ACTIVE FOR TESTING
COMPARISON_CONFIG = {
    'test_samples': 3,
    'seed': 42,
    'use_validation_set': True,
    'vanilla_model': 'gemma2:9b',
    'dspy_provider': 'openai',
    'dspy_model': 'gpt-4o-mini',
    'optimized_model_path': 'models/dspy_bootstrap_optimized.json',
}

# Full evaluation (40 questions) - COMMENTED OUT
# COMPARISON_CONFIG = {
#     # Test Configuration
#     'test_samples': 40,              # Number of test questions (should match val_samples from training)
#     'seed': 42,                      # Random seed for reproducibility
#     'use_validation_set': True,      # Use validation dataset (RECOMMENDED - matches training)
#     # Set to False to use Gemini data instead
#     
#     # Vanilla ANEETAA Configuration
#     'vanilla_model': 'gemma2:9b',   # Ollama model for vanilla agent
#     
#     # DSPy Configuration (for both baseline and optimized)
#     'dspy_provider': 'openai',      # 'openai' or 'ollama'
#     'dspy_model': 'gpt-4o-mini',    # Model for DSPy agents
#     
#     # Optimized Model Path (leave None to use baseline only)
#     'optimized_model_path': 'models/dspy_bootstrap_optimized.json',
#     # Or use: None  (if you haven't trained yet)
#     # Or use: 'models/dspy_mipro_optimized.json'  (if using MIPRO)
# }

# ============================================================================
# PRESET CONFIGURATIONS - Uncomment to use
# ============================================================================

# Quick comparison (5 questions, fast)
# COMPARISON_CONFIG = {
#     'test_samples': 5,
#     'seed': 42,
#     'vanilla_model': 'gemma2:9b',
#     'dspy_provider': 'openai',
#     'dspy_model': 'gpt-4o-mini',
#     'optimized_model_path': 'models/dspy_bootstrap_optimized.json',
# }

# Standard comparison (20 questions, good balance)
# COMPARISON_CONFIG = {
#     'test_samples': 20,
#     'seed': 42,
#     'vanilla_model': 'gemma2:9b',
#     'dspy_provider': 'openai',
#     'dspy_model': 'gpt-4o-mini',
#     'optimized_model_path': 'models/dspy_bootstrap_optimized.json',
# }

# Comprehensive comparison (50 questions, thorough)
# COMPARISON_CONFIG = {
#     'test_samples': 50,
#     'seed': 42,
#     'vanilla_model': 'gemma2:9b',
#     'dspy_provider': 'openai',
#     'dspy_model': 'gpt-4o-mini',
#     'optimized_model_path': 'models/dspy_bootstrap_optimized.json',
# }

# Test without trained model (just baseline vs vanilla)
# COMPARISON_CONFIG = {
#     'test_samples': 10,
#     'seed': 42,
#     'vanilla_model': 'gemma2:9b',
#     'dspy_provider': 'openai',
#     'dspy_model': 'gpt-4o-mini',
#     'optimized_model_path': None,  # No optimized model yet
# }

# Compare MIPRO optimized model
# COMPARISON_CONFIG = {
#     'test_samples': 20,
#     'seed': 42,
#     'vanilla_model': 'gemma2:9b',
#     'dspy_provider': 'openai',
#     'dspy_model': 'gpt-4o-mini',
#     'optimized_model_path': 'models/dspy_mipro_optimized.json',
# }

# All Ollama (no OpenAI costs)
# COMPARISON_CONFIG = {
#     'test_samples': 15,
#     'seed': 42,
#     'vanilla_model': 'gemma2:9b',
#     'dspy_provider': 'ollama',
#     'dspy_model': 'gemma2:9b',  # Use same model for fair comparison
#     'optimized_model_path': 'models/dspy_bootstrap_optimized.json',
# }

# ============================================================================
# Controller Logic - Don't modify below this line
# ============================================================================

def build_command(config):
    """Build command from configuration."""
    script_path = Path(__file__).parent.parent / "runners" / "compare_three_agents.py"
    
    cmd = [sys.executable, str(script_path)]
    
    # Add arguments
    cmd.extend(['--test-samples', str(config['test_samples'])])
    cmd.extend(['--seed', str(config['seed'])])
    cmd.extend(['--vanilla-model', config['vanilla_model']])
    cmd.extend(['--dspy-provider', config['dspy_provider']])
    cmd.extend(['--dspy-model', config['dspy_model']])
    
    # Dataset selection
    if not config.get('use_validation_set', True):
        cmd.append('--use-gemini-data')
    
    if config.get('optimized_model_path'):
        cmd.extend(['--optimized-model-path', config['optimized_model_path']])
    
    return cmd


def check_prerequisites(config):
    """Check if required models/files exist."""
    warnings = []
    
    # Check if optimized model exists
    if config.get('optimized_model_path'):
        model_path = Path(config['optimized_model_path'])
        if not model_path.exists():
            warnings.append(f"⚠️  Optimized model not found: {model_path}")
            warnings.append(f"    Run training first or set 'optimized_model_path': None")
    
    # Check if vanilla model might be available (can't check Ollama directly)
    if config['vanilla_model'] and 'ollama' in config['vanilla_model'].lower():
        warnings.append(f"ℹ️  Make sure Ollama is running with model: {config['vanilla_model']}")
    
    # Check DSPy provider
    if config['dspy_provider'] == 'ollama':
        warnings.append(f"ℹ️  Make sure Ollama is running for DSPy agents")
    elif config['dspy_provider'] == 'openai':
        warnings.append(f"ℹ️  Make sure OPENAI_API_KEY is set in .env")
    
    return warnings


def print_config(config):
    """Pretty print configuration."""
    print("="*70)
    print("THREE-WAY COMPARISON CONTROLLER")
    print("="*70)
    print("\nComparison Configuration:")
    print("-" * 70)
    
    # Test settings
    print("\n📊 Test Settings:")
    print(f"  Test Samples: {config['test_samples']} questions")
    dataset_source = "Validation dataset (val.jsonl)" if config.get('use_validation_set', True) else "Gemini data"
    print(f"  Dataset Source: {dataset_source}")
    print(f"  Random Seed: {config['seed']}")
    
    # Agent configurations
    print("\n🤖 Agent Configurations:")
    print(f"  1️⃣  Vanilla ANEETAA:")
    print(f"      Model: {config['vanilla_model']} (Ollama)")
    print(f"  2️⃣  DSPy Baseline:")
    print(f"      Provider: {config['dspy_provider']}")
    print(f"      Model: {config['dspy_model']}")
    print(f"      Optimization: None (fresh/unoptimized)")
    print(f"  3️⃣  DSPy Optimized:")
    print(f"      Provider: {config['dspy_provider']}")
    print(f"      Model: {config['dspy_model']}")
    if config.get('optimized_model_path'):
        print(f"      Optimization: {config['optimized_model_path']}")
    else:
        print(f"      Optimization: None (will use baseline)")
    
    # Output
    print("\n💾 Output:")
    print(f"  Results CSV: results/three_way_comparison_results.csv")
    print(f"  MLflow Experiment: aneetaa-three-way-comparison")
    
    print("\n" + "="*70 + "\n")
    
    # Check prerequisites
    warnings = check_prerequisites(config)
    if warnings:
        print("⚠️  Prerequisites Check:")
        print("-" * 70)
        for warning in warnings:
            print(warning)
        print("="*70 + "\n")


def main():
    """Run comparison with configured parameters."""
    # Print configuration
    print_config(COMPARISON_CONFIG)
    
    # Build command
    cmd = build_command(COMPARISON_CONFIG)
    
    # Show command
    print("Command that will be executed:")
    print(" ".join(cmd))
    print("\n" + "="*70)
    
    response = input("\n▶ Start comparison? [Y/n]: ").strip().lower()
    
    if response in ['', 'y', 'yes']:
        print("\n🚀 Starting comparison...\n")
        print("="*70 + "\n")
        
        # Run comparison
        try:
            subprocess.run(cmd, check=True)
            
            print("\n" + "="*70)
            print("✅ COMPARISON COMPLETED SUCCESSFULLY!")
            print("="*70)
            print("\n📊 Results saved to:")
            print("   results/three_way_comparison_results.csv")
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
            print("  • Ollama not running (if using Ollama)")
            print("  • OPENAI_API_KEY not set (if using OpenAI)")
            print("  • Optimized model file not found")
            sys.exit(1)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Comparison interrupted by user")
            sys.exit(1)
    else:
        print("\n❌ Comparison cancelled by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
