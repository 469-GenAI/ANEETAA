"""
LLM Judge Configuration Controller

Quick script to change LLM judge settings without editing the config file directly.
Run this to switch between OpenAI, Groq, or other providers.

Usage:
    python notebooks/scripts/config/judge_config_controller.py --provider openai --model gpt-4o
    python notebooks/scripts/config/judge_config_controller.py --provider groq --model default
    python notebooks/scripts/config/judge_config_controller.py --show
"""

import argparse
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_judge_config import JUDGE_CONFIG, print_judge_config, estimate_judge_cost, get_judge_llm

# ============================================================================
# Preset Configurations
# ============================================================================

PRESETS = {
    'openai-mini': {
        'provider': 'openai',
        'model': 'gpt-4o-mini',
        'description': 'OpenAI GPT-4o-mini (cheap, fast)'
    },
    'openai-strong': {
        'provider': 'openai',
        'model': 'gpt-4o',
        'description': 'OpenAI GPT-4o (more accurate, expensive)'
    },
    'groq-fast': {
        'provider': 'groq',
        'model': 'llama-3.1-8b-instant',
        'description': 'Groq Llama 3.1 8B (fastest, FREE)'
    },
    'groq-default': {
        'provider': 'groq',
        'model': 'llama-3.1-70b-versatile',
        'description': 'Groq Llama 3.1 70B (good quality, FREE)'
    },
    'groq-strong': {
        'provider': 'groq',
        'model': 'llama-3.2-90b-vision-preview',
        'description': 'Groq Llama 3.2 90B (strongest, FREE)'
    },
    'ollama-local': {
        'provider': 'ollama',
        'model': 'llama3.1:70b',
        'description': 'Ollama Llama 3.1 70B (local, FREE, requires local model)'
    }
}


def apply_config(provider: str, model: str):
    """
    Apply configuration by modifying llm_judge_config.py file.
    """
    config_file = Path(__file__).parent / "llm_judge_config.py"
    
    # Read current file
    with open(config_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Update JUDGE_CONFIG lines
    new_lines = []
    in_judge_config = False
    updated_provider = False
    updated_model = False
    
    for line in lines:
        if 'JUDGE_CONFIG = {' in line:
            in_judge_config = True
            new_lines.append(line)
            continue
        
        if in_judge_config:
            if "'provider':" in line and not updated_provider:
                new_lines.append(f"    'provider': '{provider}',\n")
                updated_provider = True
                continue
            
            if "'model':" in line and not line.strip().startswith('#'):
                # Found the active model line
                if "'models':" in line:
                    # Skip the models dict
                    new_lines.append(line)
                else:
                    new_lines.append(f"    'model': '{model}',\n")
                    updated_model = True
                continue
            
            if line.strip() == '}' and updated_provider and updated_model:
                in_judge_config = False
        
        new_lines.append(line)
    
    # Write back
    with open(config_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✅ Updated llm_judge_config.py")
    print(f"   Provider: {provider}")
    print(f"   Model: {model}")


def test_judge():
    """Test if judge can be created and works."""
    print("\nTesting LLM judge...")
    try:
        judge = get_judge_llm()
        print(f"✅ Successfully created {JUDGE_CONFIG['provider']} judge")
        
        # Quick test
        test_prompt = "Rate this answer on 1-10: Photosynthesis converts light energy to chemical energy."
        response = judge.invoke(test_prompt)
        print(f"✅ Judge response: {response.content[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error testing judge: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Configure LLM Judge')
    parser.add_argument('--provider', choices=['openai', 'groq', 'anthropic', 'ollama'],
                        help='LLM provider')
    parser.add_argument('--model', help='Model name or preset (default, strong, fast)')
    parser.add_argument('--preset', choices=list(PRESETS.keys()),
                        help='Use a preset configuration')
    parser.add_argument('--show', action='store_true',
                        help='Show current configuration')
    parser.add_argument('--list-presets', action='store_true',
                        help='List available presets')
    parser.add_argument('--test', action='store_true',
                        help='Test current configuration')
    parser.add_argument('--estimate-cost', type=int, metavar='NUM_QUESTIONS',
                        help='Estimate cost for N questions')
    
    args = parser.parse_args()
    
    # Show current config
    if args.show or (not any([args.provider, args.preset, args.list_presets, args.test, args.estimate_cost])):
        print_judge_config()
        return
    
    # List presets
    if args.list_presets:
        print("="*70)
        print("AVAILABLE PRESETS")
        print("="*70)
        for preset_name, preset_config in PRESETS.items():
            print(f"\n{preset_name}:")
            print(f"  Provider: {preset_config['provider']}")
            print(f"  Model: {preset_config['model']}")
            print(f"  Description: {preset_config['description']}")
        print("\n" + "="*70)
        print("\nUsage: python judge_config_controller.py --preset <preset_name>")
        return
    
    # Test current config
    if args.test:
        print_judge_config()
        test_judge()
        return
    
    # Estimate cost
    if args.estimate_cost:
        print_judge_config()
        print(f"\nCost estimates for {args.estimate_cost} questions:")
        est = estimate_judge_cost(args.estimate_cost)
        print(f"  Input tokens: ~{est['estimated_input_tokens']:,}")
        print(f"  Output tokens: ~{est['estimated_output_tokens']:,}")
        print(f"  Total cost: ${est['estimated_cost_usd']:.4f}")
        print(f"    Input: ${est['cost_breakdown']['input']:.4f}")
        print(f"    Output: ${est['cost_breakdown']['output']:.4f}")
        return
    
    # Apply preset
    if args.preset:
        preset = PRESETS[args.preset]
        print(f"Applying preset: {args.preset}")
        print(f"  {preset['description']}")
        apply_config(preset['provider'], preset['model'])
        
        # Show new config and test
        print("\nNew configuration:")
        print_judge_config()
        
        response = input("\nTest the judge? [y/N]: ").strip().lower()
        if response == 'y':
            test_judge()
        
        return
    
    # Apply custom config
    if args.provider and args.model:
        print(f"Setting custom configuration:")
        apply_config(args.provider, args.model)
        
        print("\nNew configuration:")
        print_judge_config()
        
        response = input("\nTest the judge? [y/N]: ").strip().lower()
        if response == 'y':
            test_judge()
        
        return
    
    # Missing arguments
    if args.provider and not args.model:
        print("❌ Error: --model is required when using --provider")
    elif args.model and not args.provider:
        print("❌ Error: --provider is required when using --model")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
