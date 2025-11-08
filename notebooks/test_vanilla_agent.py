"""
Quick Test: Verify Vanilla ANEETAA MCQ Solver Works

This script tests if your vanilla ANEETAA MCQ solver agent is working properly
before running full evaluations.
"""

import os
import sys
import json
from pathlib import Path

# Setup
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

print("="*70)
print("VANILLA ANEETAA MCQ SOLVER - SETUP TEST")
print("="*70 + "\n")

# Step 1: Check Python path
print("Step 1: Checking Python path...")
print(f"✓ Root: {ROOT}")
print(f"✓ Src path: {ROOT / 'src'}")

# Step 2: Load environment variables
print("\nStep 2: Loading environment variables...")
from dotenv import load_dotenv
load_dotenv()

required_env_vars = ['OPENAI_API_KEY', 'OLLAMA_URL']
missing_vars = []
for var in required_env_vars:
    value = os.getenv(var)
    if value:
        masked_value = value[:10] + "..." if len(value) > 10 else value
        print(f"✓ {var}: {masked_value}")
    else:
        missing_vars.append(var)
        print(f"⚠ {var}: Not set")

if missing_vars:
    print(f"\n⚠ Warning: Some environment variables are missing: {missing_vars}")
    print("  This is OK if you're using defaults, but may cause issues.")

# Step 3: Check test data exists
print("\nStep 3: Checking test data...")
gemini_data_folder = ROOT / "aneeta_v2" / "Processed Data" / "Gemini 2.5 Pro Data"
if gemini_data_folder.exists():
    json_files = list(gemini_data_folder.glob("*.json"))
    print(f"✓ Gemini data folder found: {gemini_data_folder}")
    print(f"✓ Found {len(json_files)} JSON files")
    
    # Load one sample file
    if json_files:
        sample_file = json_files[0]
        with open(sample_file, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        print(f"✓ Sample file loaded: {sample_file.name} ({len(sample_data)} questions)")
        
        # Show first question
        if sample_data:
            first_q = sample_data[0]
            print(f"\nSample question structure:")
            print(f"  - question_id: {first_q.get('question_id', 'N/A')}")
            print(f"  - question_text: {first_q.get('question_text', 'N/A')[:80]}...")
            print(f"  - options: {list(first_q.get('options', {}).keys())}")
            print(f"  - correct_answer: {first_q.get('correct_answer', 'N/A')}")
            print(f"  - metadata.subject: {first_q.get('metadata', {}).get('subject', 'N/A')}")
else:
    print(f"❌ Gemini data folder NOT found: {gemini_data_folder}")
    print("   Please check your data directory.")

# Step 4: Try importing ANEETAA components
print("\nStep 4: Importing ANEETAA components...")
try:
    from aneeta.state.models import State
    print("✓ Imported State model")
except ImportError as e:
    print(f"❌ Failed to import State: {e}")
    sys.exit(1)

try:
    from aneeta.nodes.agents import mcq_question_solver_agent
    print("✓ Imported mcq_question_solver_agent")
except ImportError as e:
    print(f"❌ Failed to import mcq_question_solver_agent: {e}")
    sys.exit(1)

# Step 5: Check the agent
print("\nStep 5: Verifying MCQ Solver Agent...")
try:
    print(f"✓ mcq_question_solver_agent is a: {type(mcq_question_solver_agent)}")
    print("✓ Agent function ready to use")
except Exception as e:
    print(f"❌ Failed to verify agent: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test with a simple question (if data available)
print("\nStep 6: Testing with a sample question...")
if gemini_data_folder.exists() and json_files and sample_data:
    test_q = sample_data[0]
    
    # Format question
    question_text = test_q['question_text']
    options = test_q['options']
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    full_question = f"{question_text}\n\nOptions:\n{options_text}"
    
    print(f"\nTest Question:")
    print(f"{full_question[:200]}...")
    print(f"\nCorrect Answer: {test_q['correct_answer']}")
    
    try:
        # Create state with proper structure
        from langchain_core.messages import HumanMessage
        
        state = State(
            messages=[HumanMessage(content=full_question)],
            question=full_question,
            language="English",
            user_explanation_language="English"
        )
        
        print("\nCalling mcq_question_solver_agent()...")
        result = mcq_question_solver_agent(state)
        
        print("\n✓ Agent responded!")
        print(f"\nAgent's Answer:")
        if hasattr(result, 'answer'):
            print(result.answer)
        elif hasattr(result, 'response'):
            print(result.response)
        else:
            print(result)
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nYour vanilla ANEETAA MCQ Solver is working correctly.")
        print("You can now run full evaluations with compare_agents.py")
        
    except Exception as e:
        print(f"\n❌ Error calling agent: {e}")
        import traceback
        traceback.print_exc()
        print("\nThis needs to be fixed before running evaluations.")
        sys.exit(1)
else:
    print("\n⚠ Skipping agent test (no test data available)")
    print("But imports are working, which is a good sign!")
    
print("\n" + "="*70)
print("Setup verification complete!")
print("="*70)
