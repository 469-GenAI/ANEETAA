"""
Centralized LLM Judge Configuration

This module provides a unified interface for configuring the LLM judge used in comparisons.
Supports multiple providers: OpenAI, Groq, Anthropic, etc.

Usage:
    from llm_judge_config import get_judge_llm, JUDGE_CONFIG
    
    # Get configured judge
    judge_llm = get_judge_llm()
    
    # Or customize
    judge_llm = get_judge_llm(provider='groq', model='llama-3.1-70b-versatile')
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION - Edit this to change judge settings
# ============================================================================

JUDGE_CONFIG = {
    # Provider Options: 'openai', 'groq', 'anthropic', 'ollama'
    'provider': 'openai',
    
    # Model configurations per provider
    'models': {
        'openai': {
            'default': 'gpt-4o-mini',      # Cheap, fast
            'strong': 'gpt-4o',            # More accurate
            'legacy': 'gpt-3.5-turbo',     # Cheapest
        },
        'groq': {
            'default': 'llama-3.1-70b-versatile',  # Fast, good quality
            'fast': 'llama-3.1-8b-instant',        # Fastest
            'strong': 'llama-3.2-90b-vision-preview',  # Strongest
        },
        'anthropic': {
            'default': 'claude-3-haiku-20240307',   # Fast, cheap
            'strong': 'claude-3-5-sonnet-20241022',  # Most accurate
        },
        'ollama': {
            'default': 'llama3.1:8b',
            'strong': 'llama3.1:70b',
        }
    },
    
    # Active model (or use 'default', 'strong', etc.)
    'model': 'gpt-4o',
    
    # LLM parameters
    'temperature': 0,      # 0 for deterministic evaluation
    'max_tokens': 500,     # Enough for score + reasoning
    
    # Cost estimates (USD per 1M tokens)
    'costs': {
        'openai': {
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
            'gpt-4o': {'input': 2.50, 'output': 10.00},
            'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
        },
        'groq': {
            'llama-3.1-70b-versatile': {'input': 0.59, 'output': 0.79},
            'llama-3.1-8b-instant': {'input': 0.05, 'output': 0.08},
            'llama-3.2-90b-vision-preview': {'input': 0.90, 'output': 0.90},
        },
        'anthropic': {
            'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
            'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00},
        },
        'ollama': {
            'llama3.1:8b': {'input': 0.0, 'output': 0.0},
            'llama3.1:70b': {'input': 0.0, 'output': 0.0},
        }
    }
}

# ============================================================================
# PRESET CONFIGURATIONS - Uncomment to use
# ============================================================================

# High quality OpenAI judge
# JUDGE_CONFIG['provider'] = 'openai'
# JUDGE_CONFIG['model'] = 'gpt-4o'  # or 'strong'

# Fast and free with Groq
# JUDGE_CONFIG['provider'] = 'groq'
# JUDGE_CONFIG['model'] = 'llama-3.1-70b-versatile'

# Local with Ollama (no API costs)
# JUDGE_CONFIG['provider'] = 'ollama'
# JUDGE_CONFIG['model'] = 'llama3.1:70b'

# Anthropic Claude
# JUDGE_CONFIG['provider'] = 'anthropic'
# JUDGE_CONFIG['model'] = 'claude-3-5-sonnet-20241022'


# ============================================================================
# Helper Functions
# ============================================================================

def get_model_name(provider: str = None, model: str = None) -> str:
    """
    Get the actual model name from provider and model key.
    
    Args:
        provider: Provider name (defaults to JUDGE_CONFIG['provider'])
        model: Model key or actual name (defaults to JUDGE_CONFIG['model'])
    
    Returns:
        Actual model name string
    """
    provider = provider or JUDGE_CONFIG['provider']
    model = model or JUDGE_CONFIG['model']
    
    # If model is a preset key (default, strong, etc.), resolve it
    if model in JUDGE_CONFIG['models'][provider]:
        return JUDGE_CONFIG['models'][provider][model]
    
    # Otherwise assume it's an actual model name
    return model


def get_judge_llm(provider: str = None, model: str = None, temperature: float = None):
    """
    Create and return configured LLM judge.
    
    Args:
        provider: Provider name (overrides JUDGE_CONFIG)
        model: Model name or preset key (overrides JUDGE_CONFIG)
        temperature: Temperature setting (overrides JUDGE_CONFIG)
    
    Returns:
        Configured LLM instance from LangChain
    """
    provider = provider or JUDGE_CONFIG['provider']
    model_name = get_model_name(provider, model)
    temperature = temperature if temperature is not None else JUDGE_CONFIG['temperature']
    max_tokens = JUDGE_CONFIG['max_tokens']
    
    if provider == 'openai':
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    elif provider == 'groq':
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment. Get one free at https://console.groq.com")
        
        return ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    elif provider == 'anthropic':
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        return ChatAnthropic(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    elif provider == 'ollama':
        from langchain_ollama import ChatOllama
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        
        return ChatOllama(
            model=model_name,
            base_url=ollama_url,
            temperature=temperature,
            num_predict=max_tokens
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, groq, anthropic, ollama")


def estimate_judge_cost(num_questions: int, provider: str = None, model: str = None) -> dict:
    """
    Estimate cost for judging a number of questions.
    
    Args:
        num_questions: Number of questions to judge
        provider: Provider name (defaults to JUDGE_CONFIG)
        model: Model name (defaults to JUDGE_CONFIG)
    
    Returns:
        Dict with cost estimate details
    """
    provider = provider or JUDGE_CONFIG['provider']
    model_name = get_model_name(provider, model)
    
    # Average tokens per question (empirical estimate)
    avg_input_tokens_per_q = 400   # Question + answer + prompt
    avg_output_tokens_per_q = 100  # Score + reasoning
    
    total_input_tokens = num_questions * avg_input_tokens_per_q
    total_output_tokens = num_questions * avg_output_tokens_per_q
    
    # Get costs
    costs = JUDGE_CONFIG['costs'].get(provider, {}).get(model_name, {'input': 0, 'output': 0})
    
    input_cost = (total_input_tokens / 1_000_000) * costs['input']
    output_cost = (total_output_tokens / 1_000_000) * costs['output']
    total_cost = input_cost + output_cost
    
    return {
        'provider': provider,
        'model': model_name,
        'num_questions': num_questions,
        'estimated_input_tokens': total_input_tokens,
        'estimated_output_tokens': total_output_tokens,
        'estimated_cost_usd': round(total_cost, 4),
        'cost_breakdown': {
            'input': round(input_cost, 4),
            'output': round(output_cost, 4)
        }
    }


def print_judge_config():
    """Print current judge configuration."""
    provider = JUDGE_CONFIG['provider']
    model_name = get_model_name()
    
    print("="*70)
    print("LLM JUDGE CONFIGURATION")
    print("="*70)
    print(f"Provider: {provider.upper()}")
    print(f"Model: {model_name}")
    print(f"Temperature: {JUDGE_CONFIG['temperature']}")
    print(f"Max Tokens: {JUDGE_CONFIG['max_tokens']}")
    
    # Cost estimate for 40 questions
    cost_est = estimate_judge_cost(40, provider, model_name)
    print(f"\nEstimated cost for 40 questions: ${cost_est['estimated_cost_usd']:.4f}")
    print("="*70)


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    print_judge_config()
    
    print("\nTesting LLM judge creation...")
    try:
        judge = get_judge_llm()
        print(f"✅ Successfully created {JUDGE_CONFIG['provider']} judge")
        
        # Test with a simple prompt
        test_response = judge.invoke("Rate this on 1-10: The mitochondria is the powerhouse of the cell.")
        print(f"✅ Judge response: {test_response.content[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\nCost estimates:")
    for num_q in [10, 40, 100, 200]:
        est = estimate_judge_cost(num_q)
        print(f"  {num_q} questions: ${est['estimated_cost_usd']:.4f}")
