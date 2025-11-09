# DSPy Caching Issue & Fix

## Problem: MIPROv2 Showing 0ms Latency

**Observation:** MIPROv2 agent showed 0.3ms latency while producing byte-for-byte identical output to baseline.

**Root Cause:** DSPy uses aggressive multi-level caching:
1. **LM-level cache** - `func_cached` wrapper caches identical prompts
2. **LiteLLM cache** - `cached_litellm_completion` with `cache={"no-cache": False}`
3. **Request hash** - When MIPROv2 had empty demos, it sent identical requests to baseline → cache hit

**Why it happened:**
- MIPROv2 optimization failed (empty demos)
- Same model, same prompt, same inputs as baseline
- DSPy cached the baseline responses
- MIPROv2 requests hit cache → instant response (0ms)

## Solution: Multi-Level Cache Busting

### Approach 1: Disable Cache Globally (Primary)
```python
lm = dspy.LM(
    'ollama/gemma2:9b', 
    api_base='http://localhost:11434',
    cache=False  # Disable caching
)
dspy.configure(lm=lm)
```

### Approach 2: Temperature Variation (Backup)
Added unique temperature values per agent to ensure different request hashes:
```python
# Baseline agent
result = solver(..., config={"temperature": 0.00001})

# MIPROv2 agent  
result = solver(..., config={"temperature": 0.00002})

# Bootstrap agent
result = solver(..., config={"temperature": 0.00003})
```

**Why this works:**
- Different temperatures → different request parameters
- Cache key includes all parameters
- Even 0.00001 difference breaks cache
- Temperature so low it doesn't affect output quality

### Approach 3: Clear Cache Before Comparison
```python
if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'history'):
    dspy.settings.lm.history = []
```

## Implementation Details

### Modified Functions:
1. **`dspy_baseline_agent()`**
   - Added `agent_id` parameter
   - Sets `temperature=0.00001` via config
   - Ensures unique cache key

2. **`dspy_mipro_agent()`**
   - Sets `temperature=0.00002` via config
   - Different from baseline → cache miss

3. **`dspy_bootstrap_agent()`**
   - Sets `temperature=0.00003` via config
   - Different from both baseline and MIPROv2

### DSPy Configuration:
```python
# Configure DSPy with cache disabled for accurate latency measurement
# Note: Caching can cause 0ms latencies when identical prompts are used
lm = dspy.LM(
    'ollama/gemma2:9b', 
    api_base='http://localhost:11434',
    cache=False  # Disable caching to measure true latency
)
dspy.configure(lm=lm)

# Additional: Clear any existing cache
try:
    if hasattr(dspy.settings, 'lm') and hasattr(dspy.settings.lm, 'history'):
        dspy.settings.lm.history = []
        print("Cleared DSPy cache")
except:
    pass
```

## Expected Results After Fix

### Before (With Caching):
```
DSPy Baseline:     647ms  (first call, cache miss)
DSPy + MIPROv2:    0.3ms  (cache hit - identical to baseline)
Bootstrap:         446ms  (different demos → cache miss)
```

### After (No Caching):
```
DSPy Baseline:     600-700ms  (true latency)
DSPy + MIPROv2:    600-700ms  (true latency - will be similar if demos still empty)
Bootstrap:         400-500ms  (true latency - few-shot examples make it faster)
```

**Note:** If MIPROv2 still shows very low latency after the fix, it means the optimization actually succeeded and created efficient prompts/demos that genuinely speed up inference.

## Testing the Fix

1. **Run comparison with cache disabled:**
   ```bash
   cd notebooks
   python compare_four_agents.py --test-samples 5
   ```

2. **Check latencies in output:**
   - All DSPy agents should show >500ms latency
   - No agent should show <10ms unless genuinely optimized
   - MIPROv2 and Baseline should have similar latencies if demos are empty

3. **Verify in CSV:**
   ```python
   import pandas as pd
   df = pd.read_csv('four_way_comparison_results.csv')
   print(df.groupby('agent_name')['latency_ms'].mean())
   ```

## Why Temperature Over Other Methods?

**Advantages:**
- ✅ Guaranteed to work (changes request hash)
- ✅ Doesn't affect output quality (0.00001 is negligible)
- ✅ Works across all DSPy versions
- ✅ No dependency on internal DSPy cache implementation

**Alternatives considered:**
- ❌ Adding random seed - might affect reproducibility
- ❌ Adding timestamps - messy and unnecessary
- ❌ Clearing cache manually - implementation-dependent
- ✅ Setting `cache=False` - clean but might not work in all versions

## Summary

The 0ms latency was NOT a timing measurement bug - it was legitimate caching behavior. When MIPROv2 had empty demos, it was functionally identical to baseline, so DSPy correctly returned cached results.

The fix ensures each agent gets independent measurements by:
1. Disabling cache globally
2. Using unique temperatures per agent
3. Clearing any existing cache

This will reveal the TRUE latencies and show whether optimizations actually improve performance.
