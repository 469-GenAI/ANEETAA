# LLM Judge Configuration Guide

**Created:** November 12, 2025  
**Purpose:** Centralized configuration for LLM judge used in agent comparisons

---

## 🎯 Overview

The LLM judge evaluates the quality of MCQ solver responses. This centralized configuration allows you to easily switch between different providers and models.

### Supported Providers:
- ✅ **OpenAI** - GPT-4o, GPT-4o-mini (requires API key)
- ✅ **Groq** - Llama 3.1/3.2 models (FREE with API key)
- ✅ **Anthropic** - Claude models (requires API key)
- ✅ **Ollama** - Local models (FREE, requires local setup)

---

## 🚀 Quick Start

### View Current Configuration
```bash
python notebooks/scripts/judge_config_controller.py --show
```

### Change to Groq (FREE!)
```bash
python notebooks/scripts/judge_config_controller.py --preset groq-default
```

### Change to OpenAI GPT-4o (Stronger)
```bash
python notebooks/scripts/judge_config_controller.py --preset openai-strong
```

### List All Presets
```bash
python notebooks/scripts/judge_config_controller.py --list-presets
```

---

## 📋 Available Presets

### OpenAI Options

| Preset | Model | Cost/1M tokens (in+out) | Speed | Quality |
|--------|-------|-------------------------|-------|---------|
| `openai-mini` | gpt-4o-mini | $0.15 + $0.60 | Fast | Good |
| `openai-strong` | gpt-4o | $2.50 + $10.00 | Medium | Excellent |

### Groq Options (FREE!)

| Preset | Model | Cost | Speed | Quality |
|--------|-------|------|-------|---------|
| `groq-fast` | llama-3.1-8b-instant | FREE | Fastest | Good |
| `groq-default` | llama-3.1-70b-versatile | FREE | Fast | Very Good |
| `groq-strong` | llama-3.2-90b-vision-preview | FREE | Medium | Excellent |

### Ollama (Local)

| Preset | Model | Cost | Speed | Quality |
|--------|-------|------|-------|---------|
| `ollama-local` | llama3.1:70b | FREE | Depends | Very Good |

---

## 💰 Cost Estimation

### Estimate Cost Before Running
```bash
python notebooks/scripts/judge_config_controller.py --estimate-cost 40
```

**Output:**
```
Cost estimates for 40 questions:
  Input tokens: ~16,000
  Output tokens: ~4,000
  Total cost: $0.0264
    Input: $0.0024
    Output: $0.0240
```

### Cost Comparison (40 questions)

| Provider | Model | Estimated Cost |
|----------|-------|----------------|
| OpenAI | gpt-4o-mini | ~$0.03 |
| OpenAI | gpt-4o | ~$0.44 |
| Groq | llama-3.1-70b | $0.00 |
| Groq | llama-3.2-90b | $0.00 |
| Ollama | local | $0.00 |

---

## 🔧 Manual Configuration

### Edit Configuration File Directly

Edit `notebooks/scripts/llm_judge_config.py`:

```python
JUDGE_CONFIG = {
    'provider': 'groq',                    # Change this
    'model': 'llama-3.1-70b-versatile',   # And this
    'temperature': 0,
    'max_tokens': 500,
}
```

### Use Custom Provider/Model
```bash
python notebooks/scripts/judge_config_controller.py \
  --provider groq \
  --model llama-3.1-70b-versatile
```

---

## 🧪 Testing Configuration

### Test Current Judge
```bash
python notebooks/scripts/judge_config_controller.py --test
```

**Output:**
```
LLM JUDGE CONFIGURATION
======================================================================
Provider: GROQ
Model: llama-3.1-70b-versatile
Temperature: 0
Max Tokens: 500

Estimated cost for 40 questions: $0.0000
======================================================================

Testing LLM judge...
✅ Successfully created groq judge
✅ Judge response: Overall Quality Score: 9

Brief Reasoning: The answer is scientifically accur...
```

---

## 🔑 API Key Setup

### Required Environment Variables

Add to your `.env` file:

```bash
# For OpenAI
OPENAI_API_KEY=sk-...

# For Groq (FREE - get at https://console.groq.com)
GROQ_API_KEY=gsk_...

# For Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# For Ollama (if not using default localhost)
OLLAMA_URL=http://localhost:11434
```

### Get Free Groq API Key
1. Visit https://console.groq.com
2. Sign up (free)
3. Generate API key
4. Add to `.env`: `GROQ_API_KEY=gsk_...`

---

## 📊 Usage in Comparison Scripts

All comparison scripts automatically use the centralized judge configuration:

```python
# In compare_3x3_matrix.py, compare_three_agents.py, etc.
from llm_judge_config import get_judge_llm, estimate_judge_cost

# Get configured judge
judge_llm = get_judge_llm()

# Estimate cost
cost = estimate_judge_cost(num_questions=40)
print(f"Estimated cost: ${cost['estimated_cost_usd']:.4f}")
```

---

## 🎬 Complete Workflow Example

### Scenario: Compare 3 models with Groq judge (FREE)

```bash
# 1. Switch to Groq
python notebooks/scripts/judge_config_controller.py --preset groq-default

# 2. Verify configuration
python notebooks/scripts/judge_config_controller.py --show

# 3. Test it works
python notebooks/scripts/judge_config_controller.py --test

# 4. Estimate cost (should be $0.00)
python notebooks/scripts/judge_config_controller.py --estimate-cost 40

# 5. Run comparison
python notebooks/scripts/compare_3x3_simple.py --test-samples 40
```

### Scenario: Use stronger OpenAI model

```bash
# 1. Switch to GPT-4o
python notebooks/scripts/judge_config_controller.py --preset openai-strong

# 2. Check cost (will be higher)
python notebooks/scripts/judge_config_controller.py --estimate-cost 40

# 3. Run comparison
python notebooks/scripts/compare_3x3_simple.py --test-samples 40
```

---

## 🔍 Comparison: OpenAI vs Groq

### Quality Comparison

| Aspect | OpenAI gpt-4o-mini | OpenAI gpt-4o | Groq llama-3.1-70b | Groq llama-3.2-90b |
|--------|-------------------|---------------|--------------------|--------------------|
| Accuracy | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | ⚡⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡⚡⚡⚡ | ⚡⚡⚡⚡ |
| Cost | 💰 | 💰💰💰 | FREE | FREE |
| Consistency | High | Very High | High | Very High |

### Recommendations

**For Most Users:**
- Use **Groq llama-3.1-70b** (FREE, fast, good quality)

**For Best Quality:**
- Use **OpenAI gpt-4o** or **Groq llama-3.2-90b**

**For Speed:**
- Use **Groq llama-3.1-8b-instant** (fastest)

**For Local/Offline:**
- Use **Ollama llama3.1:70b** (requires local setup)

---

## 🐛 Troubleshooting

### Error: "GROQ_API_KEY not found"
**Solution:**
1. Get free API key: https://console.groq.com
2. Add to `.env`: `GROQ_API_KEY=gsk_...`
3. Restart terminal/reload environment

### Error: "OPENAI_API_KEY not found"
**Solution:**
1. Get API key: https://platform.openai.com/api-keys
2. Add to `.env`: `OPENAI_API_KEY=sk-...`
3. Ensure you have credits on your account

### Error: "Connection refused" (Ollama)
**Solution:**
1. Start Ollama: `ollama serve`
2. Ensure model is pulled: `ollama pull llama3.1:70b`
3. Check `OLLAMA_URL` in `.env`

### Judge returns generic scores (5/10)
**Possible causes:**
- API key invalid
- Rate limit exceeded
- Model not responding
- Check logs for detailed error

---

## 📚 Advanced Usage

### Custom Temperature
Edit `llm_judge_config.py`:
```python
JUDGE_CONFIG['temperature'] = 0.3  # For more varied evaluations
```

### Add New Model
Edit `llm_judge_config.py`:
```python
JUDGE_CONFIG['models']['groq']['custom'] = 'mixtral-8x7b-32768'
```

Then use:
```bash
python judge_config_controller.py --provider groq --model mixtral-8x7b-32768
```

### Programmatic Usage
```python
from llm_judge_config import get_judge_llm, JUDGE_CONFIG

# Get judge with custom settings
judge = get_judge_llm(provider='groq', model='llama-3.2-90b-vision-preview')

# Evaluate response
result = judge.invoke("Rate this answer...")
print(result.content)
```

---

## 🎯 Best Practices

1. **Test before running full comparison**
   ```bash
   python judge_config_controller.py --test
   ```

2. **Always estimate cost first**
   ```bash
   python judge_config_controller.py --estimate-cost <num_questions>
   ```

3. **Use Groq for experimentation** (FREE!)

4. **Use OpenAI GPT-4o for final/production comparisons**

5. **Keep API keys in `.env`**, never commit them

6. **Document which judge was used** in results

---

## 📝 Files Modified

### New Files Created:
1. `notebooks/scripts/llm_judge_config.py` - Centralized configuration
2. `notebooks/scripts/judge_config_controller.py` - CLI controller
3. `notebooks/docs/LLM_JUDGE_GUIDE.md` - This guide

### Updated Files:
1. `notebooks/scripts/compare_3x3_matrix.py` - Uses centralized config
2. Other comparison scripts (as needed)

---

**Status:** ✅ Ready to use  
**Recommended:** Start with `groq-default` preset (FREE)
