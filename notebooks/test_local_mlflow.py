"""
Quick test script for local MLflow setup.
This verifies that MLflow is working correctly on your machine.
"""

import mlflow
import urllib.parse
from pathlib import Path

def test_local_mlflow():
    """Test local MLflow configuration and logging."""
    
    print("=" * 60)
    print("Testing Local MLflow Setup")
    print("=" * 60)
    
    # Setup MLflow tracking
    mlflow_dir = Path.cwd() / 'mlruns'
    mlflow_dir.mkdir(exist_ok=True)
    
    # Use relative path - MLflow prefers this for local storage
    mlflow.set_tracking_uri("./mlruns")
    
    print(f"✓ MLflow directory: {mlflow_dir}")
    print(f"✓ Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Create experiment
    experiment_name = "test-local-mlflow"
    mlflow.set_experiment(experiment_name)
    print(f"✓ Experiment set: {experiment_name}")
    
    # Start a test run
    with mlflow.start_run(run_name="test_run") as run:
        print(f"\n✓ Started run: {run.info.run_id}")
        
        # Log some test parameters
        mlflow.log_param("test_param", "hello_mlflow")
        mlflow.log_param("model", "gpt-4o-mini")
        print("✓ Logged parameters")
        
        # Log some test metrics
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_metric("f1_score", 0.92)
        mlflow.log_metric("baseline_score", 0.71)
        mlflow.log_metric("optimized_score", 0.73)
        print("✓ Logged metrics")
        
        # Log a test artifact (text file)
        test_file = mlflow_dir / "test_artifact.txt"
        test_file.write_text("This is a test artifact from local MLflow!")
        mlflow.log_artifact(str(test_file))
        print("✓ Logged artifact")
        
        print(f"\n✓ Run completed successfully!")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("✅ LOCAL MLFLOW TEST PASSED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. View your results in MLflow UI:")
    print("   Run: python -m mlflow ui --port 8080")
    print("   Then open: http://localhost:8080")
    print()
    print("2. You should see:")
    print(f"   - Experiment: '{experiment_name}'")
    print(f"   - Run: 'test_run'")
    print("   - Metrics: accuracy, f1_score, baseline_score, optimized_score")
    print("   - Parameters: test_param, model")
    print("   - 1 artifact")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        test_local_mlflow()
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        print("\nPlease check your MLflow installation:")
        print("  pip install mlflow")
