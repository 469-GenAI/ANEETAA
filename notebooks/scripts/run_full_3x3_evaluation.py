"""
Master Script: Full 3×3 Matrix Evaluation

This script orchestrates the complete evaluation:
1. Train DSPy optimized models for all 3 Ollama models
2. Run 3×3 matrix comparison with LLM judge
3. Generate comprehensive results

Total time: ~2-3 hours
Total cost: ~$0.18 (only for LLM judge)
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("="*70)
    print("FULL 3×3 MATRIX EVALUATION - MASTER SCRIPT")
    print("="*70)
    print("\nThis will:")
    print("  1. Train 3 DSPy optimized models (llama3.1:8b, gemma2:9b, mistral-nemo:12b)")
    print("     - 200 train questions + 40 val questions per model")
    print("     - Bootstrap optimization")
    print("     - Using Ollama (FREE)")
    print("     - ~10-15 min per model (~30-45 min total)")
    print()
    print("  2. Run 3×3 matrix comparison")
    print("     - 3 models × 3 agent types = 9 configurations")
    print("     - 40 test questions")
    print("     - OpenAI GPT-4o-mini LLM judge (~$0.08)")
    print("     - ~10-15 minutes")
    print()
    print("Total time: ~40-60 minutes")
    print("Total cost: ~$0.08")
    print("="*70)
    
    response = input("\n▶ Start full evaluation? [Y/n]: ").strip().lower()
    
    if response not in ['', 'y', 'yes']:
        print("\n❌ Evaluation cancelled")
        return
    
    scripts_dir = Path(__file__).parent
    
    # Step 1: Train all models
    print("\n" + "="*70)
    print("STEP 1: TRAINING ALL MODELS")
    print("="*70 + "\n")
    
    train_script = scripts_dir / "train_all_models.py"
    
    try:
        subprocess.run([sys.executable, str(train_script)], check=True)
        print("\n✅ All models trained successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        response = input("\n⚠️  Continue with comparison anyway? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("\n❌ Evaluation stopped")
            return
    
    # Step 2: Run 3×3 comparison
    print("\n" + "="*70)
    print("STEP 2: RUNNING 3×3 MATRIX COMPARISON")
    print("="*70 + "\n")
    
    compare_script = scripts_dir / "compare_3x3_matrix.py"
    
    try:
        subprocess.run([sys.executable, str(compare_script), 
                       '--test-samples', '40',  # Updated to match val_samples
                       '--seed', '42'], check=True)
        print("\n✅ Comparison completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Comparison failed: {e}")
        return
    
    # Summary
    print("\n" + "="*70)
    print("FULL EVALUATION COMPLETE!")
    print("="*70)
    print("\n📊 Results available:")
    print("  - results/3x3_matrix_summary.csv")
    print("  - results/3x3_matrix_detailed_results.csv")
    print("\n📈 View in MLflow:")
    print("  mlflow ui --port 8080")
    print("  Then open: http://localhost:8080")
    print("="*70)

if __name__ == "__main__":
    main()
