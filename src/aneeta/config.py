# --- Page Configuration Constants ---
PAGE_TITLE = "ANEETA: Agents for National Eligibility Entrance Test Assistance"
TAG_LINE = "In India, many dream of becoming doctors but can't afford costly NEET coaching. Some, like Late Ms. Anitha, tragically gave up after years of struggle. ANEETA uses local AI (Gemma and Ollama) to help underprivileged students prepare offline for NEET UG in their own language, making education accessible and affordable. ANEETA bridges gaps to ensure no dream is limited by money or location."

# --- Language Options ---
ASSISTIVE_LANGUAGE_OPTIONS = ["Tamil", "Hindi", "Bengali", "Telugu", "Marathi"]
DEFAULT_LANGUAGE = "Tamil"

# --- Vector DB Names ---
VECTOR_DB_CONFIGS = {
    'biology': 'chroma_vector_db_biology_nomic',
    'chemistry': 'chroma_vector_db_chemistry_nomic',
    'physics': 'chroma_vector_db_physics_nomic',
    'question_bank': 'chroma_vector_db_questionbank_nomic',
    'mentor': 'chroma_vector_db_mentor_nomic'
}

# --- DSPy Configuration ---
import os
USE_DSPY_AGENTS = os.getenv("USE_DSPY_AGENTS", "false").lower() == "true"
DSPY_LM_MODEL = os.getenv("DSPY_LM_MODEL", "openai/gpt-4o-mini")  # Can also use Ollama models

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "aneeta-production")
MLFLOW_ENABLE_TRACING = os.getenv("MLFLOW_ENABLE_TRACING", "true").lower() == "true"