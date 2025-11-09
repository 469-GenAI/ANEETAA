"""
ANEETAA Setup Verification Script
Verifies all dependencies and configurations for DSPy agent evaluation project
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 12):
        print("  ⚠️  Warning: Python 3.12+ recommended")
    return True

def check_packages():
    """Check required Python packages"""
    packages = {
        'dspy': 'DSPy',
        'mlflow': 'MLflow',
        'langchain_community': 'LangChain Community',
        'chromadb': 'ChromaDB',
        'openai': 'OpenAI',
        'langchain_ollama': 'LangChain Ollama',
        'datasets': 'Datasets',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: NOT INSTALLED")
            all_ok = False
    return all_ok

def check_env_file():
    """Check .env file configuration"""
    env_path = Path('.env')
    if not env_path.exists():
        print("✗ .env file not found")
        return False
    
    print("✓ .env file exists")
    
    # Check key variables
    required_vars = [
        'OPENAI_API_KEY',
        'OLLAMA_URL',
        'LLM_MODEL',
        'USE_DATABRICKS_MLFLOW',
        'MLFLOW_TRACKING_URI'
    ]
    
    with open(env_path, 'r') as f:
        content = f.read()
    
    for var in required_vars:
        if var in content:
            print(f"  ✓ {var} configured")
        else:
            print(f"  ⚠️  {var} missing")
    
    return True

def check_ollama():
    """Check Ollama installation and models"""
    import subprocess
    
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✓ Ollama installed: {version}")
            
            # Check models
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                models = result.stdout
                required_models = ['llama3.1:8b', 'gemma2:9b', 'mistral-nemo:12b']
                for model in required_models:
                    if model in models:
                        print(f"  ✓ {model}")
                    else:
                        print(f"  ✗ {model} NOT FOUND")
            return True
        else:
            print("✗ Ollama not running")
            return False
    except Exception as e:
        print(f"✗ Ollama check failed: {e}")
        return False

def check_project_structure():
    """Check key project files and directories"""
    paths = {
        'notebooks/compare_agents.py': 'Agent comparison script',
        'notebooks/dspy_optimization.py': 'DSPy optimization script',
        'notebooks/test_vanilla_agent.py': 'Vanilla agent test script',
        'aneeta_v2/Processed Data/Gemini 2.5 Pro Data': 'Test questions directory',
        'Processed Data': 'Training data directory',
        'src/aneeta': 'Source code directory',
    }
    
    all_ok = True
    for path, desc in paths.items():
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                count = len(list(path_obj.glob('*')))
                print(f"✓ {desc}: {count} items")
            else:
                print(f"✓ {desc}")
        else:
            print(f"✗ {desc}: NOT FOUND")
            all_ok = False
    
    return all_ok

def main():
    print("="*60)
    print("ANEETAA DSPy Evaluation - Setup Verification")
    print("="*60)
    print()
    
    print("1. Python Environment")
    print("-" * 40)
    check_python_version()
    print()
    
    print("2. Python Packages")
    print("-" * 40)
    packages_ok = check_packages()
    print()
    
    print("3. Environment Configuration")
    print("-" * 40)
    env_ok = check_env_file()
    print()
    
    print("4. Ollama Setup")
    print("-" * 40)
    ollama_ok = check_ollama()
    print()
    
    print("5. Project Structure")
    print("-" * 40)
    structure_ok = check_project_structure()
    print()
    
    print("="*60)
    if packages_ok and env_ok and ollama_ok and structure_ok:
        print("✅ SETUP COMPLETE - Ready to run evaluations!")
        print()
        print("Next steps:")
        print("1. Start MLflow UI: python -m mlflow ui --port 8080")
        print("2. Test setup: python notebooks/test_vanilla_agent.py")
        print("3. Run comparison: python notebooks/compare_agents.py --provider ollama --model gemma2:9b")
    else:
        print("⚠️  SETUP INCOMPLETE - Please fix the issues above")
    print("="*60)

if __name__ == "__main__":
    main()
