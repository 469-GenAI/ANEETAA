"""
Quick test to verify Databricks MLflow connection.
Run this to ensure your .env configuration works before running full optimization.
"""

import os
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

def test_mlflow_connection():
    """Test connection to Databricks MLflow."""
    print("="*60)
    print("Testing Databricks MLflow Connection")
    print("="*60 + "\n")
    
    # Load environment variables
    load_dotenv()
    
    # Check required variables
    print("1. Checking environment variables...")
    required_vars = ['DATABRICKS_TOKEN', 'DATABRICKS_HOST', 'MLFLOW_TRACKING_URI', 'MLFLOW_EXPERIMENT_ID']
    for var in required_vars:
        value = os.getenv(var)
        if value:
            if 'TOKEN' in var:
                print(f"   ✓ {var}: {'*' * 20} (hidden)")
            else:
                print(f"   ✓ {var}: {value}")
        else:
            print(f"   ✗ {var}: NOT SET")
            return False
    
    # Set tracking URI
    print("\n2. Setting up MLflow tracking...")
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"   ✓ Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Test experiment access
    print("\n3. Testing experiment access...")
    exp_id = os.getenv('MLFLOW_EXPERIMENT_ID')
    try:
        client = MlflowClient()
        exp = client.get_experiment(exp_id)
        if exp is not None:
            print(f"   ✓ Experiment found!")
            print(f"      Name: {exp.name}")
            print(f"      ID: {exp.experiment_id}")
            print(f"      Location: {exp.artifact_location}")
        else:
            print(f"   ✗ Experiment {exp_id} not found")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test creating a run
    print("\n4. Testing run creation...")
    try:
        mlflow.set_experiment(experiment_id=exp_id)
        with mlflow.start_run(run_name="connection_test") as run:
            # Log some test data
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.99)
            mlflow.set_tag("test", "connection_verification")
            
            print(f"   ✓ Run created successfully!")
            print(f"      Run ID: {run.info.run_id}")
            print(f"      Run Name: connection_test")
            print(f"      View at: {os.getenv('DATABRICKS_HOST')}")
    except Exception as e:
        print(f"   ✗ Error creating run: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ All tests passed! Your MLflow connection is working.")
    print("="*60)
    print(f"\nYou can view the test run at:")
    print(f"{os.getenv('DATABRICKS_HOST')}/ml/experiments/{exp_id}")
    return True


if __name__ == "__main__":
    success = test_mlflow_connection()
    if not success:
        print("\n❌ Connection test failed. Check your .env file configuration.")
        exit(1)
