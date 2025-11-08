"""
MCQ Solver Agent Evaluation Script
Tests 3 models (llama3.1:8b, gemma2:9b, mistral-nemo:12b) on 3 NEET questions
"""

import os
import sys
import json
import time
import pandas as pd
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

# Configure environment
os.environ["LLM_MODEL"] = "llama3.1:8b"
os.environ["CREATIVE_LLM_MODEL"] = "llama3.1:8b"
os.environ["EMBEDDING_MODEL"] = "nomic-embed-text"
os.environ["VECTORDB_BASE_PATH"] = str(ROOT / "src" / "aneeta" / "vectordb")

# Load .env file for OpenAI API key
env_path = ROOT / ".env"
if env_path.exists():
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

# Import ANEETA components
from aneeta.state.models import State
from aneeta.nodes.agents import mcq_question_solver_agent as aneeta_mcq_agent
from aneeta.core import resources
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

print("✓ Loaded ANEETA MCQ solver agent")

# Load questions from all files in Gemini 2.5 Pro data folder
import random
import argparse

gemini_data_folder = ROOT / "aneeta_v2" / "Processed Data" / "Gemini 2.5 Pro Data"
all_questions = []

# Load all JSON files
for json_file in gemini_data_folder.glob("*.json"):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            file_questions = json.load(f)
            all_questions.extend(file_questions)
    except Exception as e:
        print(f"Warning: Could not load {json_file.name}: {e}")

print(f"✓ Loaded {len(all_questions)} total questions from Gemini 2.5 Pro data folder")

# Filter questions that don't require visual
non_visual_questions = [q for q in all_questions if not q.get('metadata', {}).get('requires_visual', False)]
print(f"✓ Filtered to {len(non_visual_questions)} non-visual questions")

# Parse seed for reproducible sampling (required)
parser = argparse.ArgumentParser(description='MCQ evaluator (deterministic question sampling with seed)')
parser.add_argument('--seed', type=int, default=None, help='Integer seed for random sampling (defaults to RANDOM_SEED from .env)')
args, unknown = parser.parse_known_args()

# Get seed from CLI arg or env var (default to 42 if neither provided)
seed = args.seed if args.seed is not None else (int(os.getenv('RANDOM_SEED')) if os.getenv('RANDOM_SEED') and os.getenv('RANDOM_SEED').isdigit() else 42)
random.seed(seed)
print(f"✓ Using random seed: {seed} (deterministic sampling)")

# Randomly select 3 questions (deterministic if seed set)
if len(non_visual_questions) < 3:
    print(f"Warning: Only {len(non_visual_questions)} non-visual questions available")
    questions = non_visual_questions
else:
    questions = random.sample(non_visual_questions, 3)

print(f"✓ Randomly selected {len(questions)} questions for evaluation\n")

# Format question for agent
def format_question(q_data):
    """Format question with options for MCQ solver"""
    q_text = q_data['question_text']
    options = q_data['options']
    options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return f"{q_text}\n\nOptions:\n{options_text}"

# MCQ Solver wrapper with timeout
def mcq_question_solver(question: str, model_name: str, language: str = "English", timeout: int = 300):
    """Call ANEETA's MCQ solver with specified model"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Model took longer than {timeout} seconds to respond")
    
    try:
        # Update model with full name including size tag
        os.environ["LLM_MODEL"] = model_name
        os.environ["CREATIVE_LLM_MODEL"] = model_name
        
        print(f"  → Loading model: {model_name}")
        
        # Create new LLM instance directly (bypass Streamlit cache)
        new_llm = ChatOllama(
            model=model_name,
            temperature=0,
            max_tokens=700,
            timeout=timeout  # Add timeout to ChatOllama
        )
        
        # Test the LLM before using it
        test_result = new_llm.invoke("Test")
        print(f"  → Model loaded successfully")
        
        # Update the global resources.llm
        resources.llm = new_llm
        
        # Reload the agents module to pick up the new llm
        import importlib
        import aneeta.nodes.agents
        importlib.reload(aneeta.nodes.agents)
        from aneeta.nodes.agents import mcq_question_solver_agent as reloaded_agent
        
        # Print debug info
        print(f"  → resources.llm.model = {resources.llm.model}")
        print(f"  → Generating answer (timeout: {timeout}s)...")
        
        # Build State
        state = State(
            messages=[HumanMessage(content=question)],
            user_explanation_language=language,
            agent_routing="mcq_question_solver",
            teacher_vectordb_routing="biology",
            response_stream=None
        )
        
        # Call reloaded agent with streaming
        result_state = reloaded_agent(state)
        
        # Collect response with progress indicator
        response_chunks = []
        chunk_count = 0
        for chunk in result_state.get("response_stream", []):
            response_chunks.append(str(chunk))
            chunk_count += 1
            if chunk_count % 10 == 0:
                print(".", end="", flush=True)
        
        if chunk_count > 0:
            print()  # New line after dots
        
        return "".join(response_chunks).strip()
        
    except TimeoutError as e:
        return f"[Timeout Error: {str(e)}]"
    except Exception as e:
        import traceback
        full_trace = traceback.format_exc()
        return f"[Error: {str(e)}]\n\nFull traceback:\n{full_trace}"

# Evaluation metrics
def fact_check_answer(model_answer: str, correct_answer: str, options: dict) -> int:
    """Score answer correctness - binary scoring: correct or wrong
    
    Scoring rubric (purely binary):
    - 10: Correct option identified (explicitly, prominently, or anywhere in response)
    - 0: Wrong option stated OR no correct option identified
    
    Note: Reasoning quality is evaluated separately by the LLM judge (Quality Score)
    """
    import re
    
    # Normalize inputs
    model_answer_lower = model_answer.lower()
    correct_answer_upper = correct_answer.upper()
    
    # Check if correct option letter appears anywhere in the answer
    correct_option_mentioned = re.search(rf"\(?{correct_answer_upper}\)?", model_answer)
    
    # Check for wrong options explicitly stated
    wrong_options = [opt for opt in options.keys() if opt != correct_answer]
    wrong_option_pattern = re.compile(
        rf"(?:answer|correct|option)(?:\s+is)?[\s:]*\(?({'|'.join(wrong_options)})\)?",
        re.IGNORECASE
    )
    wrong_option_stated = wrong_option_pattern.search(model_answer)
    
    # Binary scoring logic
    if wrong_option_stated:
        # Wrong answer explicitly stated
        return 0
    elif correct_option_mentioned:
        # Correct option mentioned anywhere
        return 10
    else:
        # No identifiable answer
        return 0

def judge_answer_quality(question: str, answer: str) -> dict:
    """Use GPT-4o-mini to judge answer quality with subject-specific criteria"""
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "your-api-key-here":
            return {"score": 5, "reasoning": "OpenAI API key not configured", "subject": "Unknown"}
        
        judge_llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0)
        
        # Step 1: Identify the subject
        subject_identification_prompt = f"""Analyze this NEET exam question and identify which subject it belongs to.

Question: {question}

Respond with ONLY ONE WORD: Physics, Chemistry, or Biology"""
        
        subject_response = judge_llm.invoke(subject_identification_prompt)
        subject = subject_response.content.strip() if hasattr(subject_response, 'content') else "Unknown"
        
        # Step 2: Subject-specific evaluation criteria
        if "Physics" in subject:
            evaluation_prompt = f"""You are evaluating an AI MCQ solver agent designed for NEET Physics questions. Assess the agent's response with RIGOROUS standards.

===================================================================================
PHYSICS-SPECIFIC EVALUATION CRITERIA FOR MCQ SOLVER AGENT
===================================================================================

CRITERION 1: CLARITY (30%)
---------------------------
EXCELLENT (9-10):
- Uses proper physics terminology (force, velocity, acceleration, field, potential, etc.)
- Defines variables clearly (e.g., "Let v = velocity in m/s, F = applied force in N")
- Uses analogies that connect to real-world physics phenomena
- Example: "The voltage across a capacitor is like water pressure in a tank - higher charge creates higher 'pressure' (voltage) that pushes current through the circuit."

GOOD (7-8):
- Clear physics terms but minimal variable definitions
- Some physical intuition provided
- Example: "The capacitor stores charge and the voltage increases as more charge accumulates."

POOR (4-6):
- Vague terms like "thing" or "stuff" instead of proper physics terminology
- No variable definitions
- Example: "The thing stores energy and makes voltage go up."

UNACCEPTABLE (1-3):
- Wrong terminology or no physics language
- Incomprehensible explanation
- Example: "It just works that way."

CRITERION 2: LOGICAL REASONING & STEPS (40%)
---------------------------------------------
EXCELLENT (9-10):
- Shows ALL intermediate steps with proper equations
- Explains WHY each step follows from the previous
- Checks units at each step
- Example: "Step 1: Apply Newton's 2nd Law F=ma. Step 2: Substitute F=10N, m=2kg. Step 3: Solve for a: a=F/m=10/2=5 m/s². Step 4: Verify units: N/kg = (kg⋅m/s²)/kg = m/s² ✓"

GOOD (7-8):
- Shows main steps with equations
- Some justification for each step
- Example: "Using F=ma, we get a=F/m=10/2=5 m/s²"

POOR (4-6):
- Jumps to answer without showing work
- Missing critical steps
- Example: "The answer is 5 m/s²"

UNACCEPTABLE (1-3):
- No logical flow or completely wrong approach
- Example: "I think it's 5"

CRITERION 3: CORRECTNESS OF APPROACH (30%)
-------------------------------------------
EXCELLENT (9-10):
- Uses correct physics principles (laws, conservation principles, equations)
- Applies proper sign conventions and vector directions
- Recognizes special cases or limiting conditions
- Example: For projectile motion, correctly identifies x and y components, applies g=-9.8m/s² downward

GOOD (7-8):
- Correct main principle but minor formula errors
- Example: Uses energy conservation but forgets a term

POOR (4-6):
- Wrong principle or formula but logical structure exists
- Example: Uses F=mv instead of F=ma

UNACCEPTABLE (1-3):
- Completely wrong physics approach
- Example: Uses chemical reaction equation for a mechanics problem

===================================================================================

Question: {question}

MCQ Solver Agent's Response:
{answer}

Provide your evaluation in this format:
Subject: Physics
Score: [number 1-10]
Clarity: [score/10] - [specific feedback]
Reasoning: [score/10] - [specific feedback]
Approach: [score/10] - [specific feedback]
Overall Reasoning: [2-3 sentences with specific examples from the agent's response]
"""
        
        elif "Chemistry" in subject:
            evaluation_prompt = f"""You are evaluating an AI MCQ solver agent designed for NEET Chemistry questions. Assess the agent's response with RIGOROUS standards.

===================================================================================
CHEMISTRY-SPECIFIC EVALUATION CRITERIA FOR MCQ SOLVER AGENT
===================================================================================

CRITERION 1: CLARITY (30%)
---------------------------
EXCELLENT (9-10):
- Uses proper chemical terminology (oxidation, reduction, molarity, equilibrium, etc.)
- Writes balanced chemical equations with states (s, l, g, aq)
- Clearly names compounds using IUPAC nomenclature
- Example: "In this redox reaction, Fe²⁺ (ferrous ion) is oxidized to Fe³⁺ (ferric ion) by losing one electron: Fe²⁺(aq) → Fe³⁺(aq) + e⁻"

GOOD (7-8):
- Good chemistry terminology with chemical formulas
- Equations mostly balanced
- Example: "Fe²⁺ loses an electron to become Fe³⁺ in this oxidation reaction"

POOR (4-6):
- Vague terms, missing chemical formulas
- Unbalanced equations
- Example: "Iron changes its charge and becomes different iron"

UNACCEPTABLE (1-3):
- No chemical terminology or formulas
- Completely unclear
- Example: "The stuff changes"

CRITERION 2: LOGICAL REASONING & STEPS (40%)
---------------------------------------------
EXCELLENT (9-10):
- Shows all steps: balanced equation, mole calculations, stoichiometry, conversions
- Explains electron transfer in redox or bond breaking/forming
- Tracks units through calculations (moles, grams, M, etc.)
- Example: "Step 1: Balance equation: 2Fe²⁺ + Cl₂ → 2Fe³⁺ + 2Cl⁻. Step 2: Calculate moles: n=mass/MW = 5.6g/56g/mol = 0.1 mol Fe²⁺. Step 3: Use stoichiometry: 0.1 mol Fe²⁺ × (1 mol Cl₂/2 mol Fe²⁺) = 0.05 mol Cl₂ needed"

GOOD (7-8):
- Shows main calculation steps
- Some chemical reasoning
- Example: "0.1 mol Fe²⁺ reacts with 0.05 mol Cl₂ based on the balanced equation"

POOR (4-6):
- Missing steps or incorrect stoichiometry
- No balanced equation
- Example: "Need 0.05 mol Cl₂"

UNACCEPTABLE (1-3):
- No logical steps or completely wrong
- Example: "The answer is 0.05"

CRITERION 3: CORRECTNESS OF APPROACH (30%)
-------------------------------------------
EXCELLENT (9-10):
- Uses correct chemistry concepts (Le Chatelier, electronegativity, hybridization, etc.)
- Applies appropriate formulas (pH = -log[H⁺], ΔG = ΔH - TΔS, etc.)
- Recognizes limiting reagent, spectator ions, or functional groups correctly
- Example: Correctly identifies which species is oxidized vs reduced using oxidation numbers

GOOD (7-8):
- Correct main concept but minor errors
- Example: Correct oxidation numbers but wrong final balancing

POOR (4-6):
- Wrong concept but shows some chemical knowledge
- Example: Confuses oxidation with reduction

UNACCEPTABLE (1-3):
- Completely wrong chemistry
- Example: Uses physics equations for a chemistry problem

===================================================================================

Question: {question}

MCQ Solver Agent's Response:
{answer}

Provide your evaluation in this format:
Subject: Chemistry
Score: [number 1-10]
Clarity: [score/10] - [specific feedback]
Reasoning: [score/10] - [specific feedback]
Approach: [score/10] - [specific feedback]
Overall Reasoning: [2-3 sentences with specific examples from the agent's response]
"""
        
        elif "Biology" in subject:
            evaluation_prompt = f"""You are evaluating an AI MCQ solver agent designed for NEET Biology questions. Assess the agent's response with RIGOROUS standards.

===================================================================================
BIOLOGY-SPECIFIC EVALUATION CRITERIA FOR MCQ SOLVER AGENT
===================================================================================

CRITERION 1: CLARITY (30%)
---------------------------
EXCELLENT (9-10):
- Uses precise biological terminology (mitochondria, ATP synthase, Krebs cycle, photosynthesis, etc.)
- Provides specific names (genus/species, organ systems, cell organelles)
- Relates structure to function clearly
- Example: "The mitochondrion's cristae (folded inner membrane) provide a large surface area for embedding ATP synthase complexes, which synthesize ATP through oxidative phosphorylation during cellular respiration."

GOOD (7-8):
- Good biology terms with some specificity
- Basic structure-function relationships
- Example: "Mitochondria have a folded inner membrane that helps produce ATP"

POOR (4-6):
- Generic terms without specificity
- Vague descriptions
- Example: "The powerhouse has folds that make energy"

UNACCEPTABLE (1-3):
- No biological terminology
- Incorrect names or completely unclear
- Example: "The cell thing does stuff"

CRITERION 2: LOGICAL REASONING & STEPS (40%)
---------------------------------------------
EXCELLENT (9-10):
- Shows process flow (stimulus → receptor → pathway → response)
- Explains biological mechanisms step-by-step
- Connects cause and effect with biological reasoning
- Example: "Step 1: Glucose enters the cell via GLUT4 transporters. Step 2: Glycolysis breaks glucose into 2 pyruvate molecules, yielding 2 ATP. Step 3: Pyruvate enters mitochondria. Step 4: Krebs cycle oxidizes pyruvate, producing NADH and FADH₂. Step 5: Electron transport chain uses NADH/FADH₂ to create proton gradient. Step 6: ATP synthase uses gradient to produce ~32 ATP."

GOOD (7-8):
- Shows main biological steps
- Some mechanistic explanation
- Example: "Glucose is broken down through glycolysis and Krebs cycle to produce ATP in the mitochondria"

POOR (4-6):
- Lists steps without explanation or missing key steps
- No mechanistic connection
- Example: "Glucose makes ATP somehow"

UNACCEPTABLE (1-3):
- No logical biological reasoning
- Random statements
- Example: "Cells just make energy"

CRITERION 3: CORRECTNESS OF APPROACH (30%)
-------------------------------------------
EXCELLENT (9-10):
- Uses correct biological concepts (enzyme kinetics, genetics, evolution, homeostasis, etc.)
- Applies proper biological principles (natural selection, cell theory, energy flow, etc.)
- Correctly identifies systems, organs, tissues, cells involved
- Example: For genetics, correctly uses Punnett square, identifies dominant/recessive alleles, calculates genotype ratios

GOOD (7-8):
- Correct main biological concept with minor errors
- Example: Right inheritance pattern but small calculation error

POOR (4-6):
- Wrong biological concept but shows some knowledge
- Example: Confuses mitosis with meiosis

UNACCEPTABLE (1-3):
- Completely wrong biology
- Example: Uses chemical reactions for a genetics problem

===================================================================================

Question: {question}

MCQ Solver Agent's Response:
{answer}

Provide your evaluation in this format:
Subject: Biology
Score: [number 1-10]
Clarity: [score/10] - [specific feedback]
Reasoning: [score/10] - [specific feedback]
Approach: [score/10] - [specific feedback]
Overall Reasoning: [2-3 sentences with specific examples from the agent's response]
"""
        
        else:
            # Fallback for unknown subject
            evaluation_prompt = f"""You are evaluating an AI MCQ solver agent designed for NEET exam questions. Assess the agent's response.

Question: {question}

MCQ Solver Agent's Response:
{answer}

Provide your evaluation in this format:
Subject: Unknown
Score: [number 1-10]
Reasoning: [2-3 sentences explaining the score and quality of the agent's response]
"""
        
        response = judge_llm.invoke(evaluation_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract score
        import re
        score_match = re.search(r"Score:\s*(\d+)", response_text)
        score = int(score_match.group(1)) if score_match else 5
        score = max(1, min(10, score))
        
        # Extract reasoning (everything after "Overall Reasoning:" or "Reasoning:")
        reasoning_match = re.search(r"(?:Overall Reasoning|Reasoning):\s*(.+)", response_text, re.S)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        return {"score": score, "reasoning": reasoning, "subject": subject}
    except Exception as e:
        import traceback
        return {"score": 5, "reasoning": f"Judge error: {str(e)[:100]}", "subject": "Unknown"}

# Models to test (exact names from ollama list)
models = ["llama3.1:8b", "gemma2:9b", "mistral-nemo:12b"]

# Run evaluation
print("="*70)
print("STARTING EVALUATION")
print("="*70)
print(f"Models: {', '.join(models)}")
print(f"Questions: {len(questions)}")
print("="*70 + "\n")

results = []

for model in models:
    print(f"\n{'#'*70}")
    print(f"# TESTING MODEL: {model}")
    print(f"{'#'*70}\n")
    
    for i, q_data in enumerate(questions, 1):
        question_text = format_question(q_data)
        correct_answer = q_data['correct_answer']
        subject = q_data['metadata']['subject']
        
        print(f"\n{'='*70}")
        print(f"QUESTION {i}/{len(questions)}: {subject}")
        print(f"{'='*70}")
        print(f"{question_text}")
        print(f"\nCorrect Answer: ({correct_answer})")
        print(f"\nModel: {model}")
        print("-"*70)
        
        # Get answer with timing
        t0 = time.time()
        answer = mcq_question_solver(question_text, model)
        latency_ms = (time.time() - t0) * 1000
        
        # Show answer
        print("MODEL OUTPUT:")
        print(answer if answer else "[no response]")
        print("-"*70)
        
        # Score answer
        fact_score = fact_check_answer(answer, correct_answer, q_data['options'])
        quality_result = judge_answer_quality(question_text, answer)
        quality_score = quality_result["score"]
        detected_subject = quality_result.get("subject", "Unknown")
        
        print(f"METRICS:")
        print(f"  ✓ Detected Subject: {detected_subject}")
        print(f"  ✓ Fact-Check Score: {fact_score}/10")
        print(f"  ✓ Quality Score: {quality_score}/10")
        print(f"  ✓ Latency: {latency_ms:.0f}ms")
        print(f"\nQuality Reasoning:")
        print(f"  {quality_result['reasoning']}")
        
        # Store result
        results.append({
            "Model": model,
            "Question": f"Q{i} ({subject})",
            "Question_ID": q_data['question_id'],
            "Detected_Subject": detected_subject,
            "Correct_Answer": correct_answer,
            "Full_Answer": answer,  # Store full answer without truncation
            "Answer_Preview": answer[:150] + "..." if len(answer) > 150 else answer,
            "Fact_Score": fact_score,
            "Quality_Score": quality_score,
            "Latency_ms": round(latency_ms, 1)
        })

# Create summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70 + "\n")

# Restructure data: one row per question with all models side-by-side
restructured_results = []
for i, q_data in enumerate(questions, 1):
    # Include full question with options (not truncated)
    formatted_question = format_question(q_data)
    row = {
        "Subject": q_data['metadata']['subject'],
        "Question": f"{formatted_question}\n\nCorrect Answer: {q_data['correct_answer']}"
    }
    
    # Add each model's results
    for model in models:
        model_result = next((r for r in results if r["Model"] == model and r["Question"] == f"Q{i} ({q_data['metadata']['subject']})"), None)
        if model_result:
            # Use Full_Answer instead of Answer_Preview for complete output
            row[f"{model}_Output"] = model_result["Full_Answer"]
            row[f"{model}_Fact_Score"] = model_result["Fact_Score"]
            row[f"{model}_Quality_Score"] = model_result["Quality_Score"]
            row[f"{model}_Latency_ms"] = model_result["Latency_ms"]
    
    restructured_results.append(row)

df_restructured = pd.DataFrame(restructured_results)

# Model summary (without Overall_Score)
print("MODEL SUMMARY")
print("="*70)
summary_data = []
df_original = pd.DataFrame(results)
for model in models:
    model_results = df_original[df_original["Model"] == model]
    summary_data.append({
        "Model": model,
        "Avg_Fact_Score": round(model_results["Fact_Score"].mean(), 1),
        "Avg_Quality_Score": round(model_results["Quality_Score"].mean(), 1),
        "Avg_Latency_ms": round(model_results["Latency_ms"].mean(), 1)
    })

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# Save restructured results
output_path = ROOT / "mcq_eval_results.csv"
df_restructured.to_csv(output_path, index=False)
print(f"\n✓ Detailed results saved to: {output_path}")

# Also save model summary as a separate CSV
summary_output_path = ROOT / "mcq_eval_model_summary.csv"
df_summary.to_csv(summary_output_path, index=False)
print(f"✓ Model summary saved to: {summary_output_path}")

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)
