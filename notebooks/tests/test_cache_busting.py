"""
Quick test to verify DSPy cache busting works
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import dspy
import time

# Configure DSPy with cache disabled
lm = dspy.LM(
    'ollama/gemma2:9b', 
    api_base='http://localhost:11434',
    cache=False
)
dspy.configure(lm=lm)

print("Testing DSPy cache busting...")

# Define a simple signature
class SimpleQA(dspy.Signature):
    """Answer a simple question"""
    question = dspy.InputField()
    answer = dspy.OutputField()

# Create module
predictor = dspy.ChainOfThought(SimpleQA)

# Test 1: Same prompt, no temperature
print("\n=== Test 1: Same prompt, no temperature ===")
q = "What is 2+2?"

start = time.time()
result1 = predictor(question=q)
latency1 = (time.time() - start) * 1000
print(f"First call: {latency1:.2f}ms")
print(f"Answer: {result1.answer}")

start = time.time()
result2 = predictor(question=q)
latency2 = (time.time() - start) * 1000
print(f"Second call (should be fast if cached): {latency2:.2f}ms")
print(f"Answer: {result2.answer}")

if latency2 < latency1 * 0.1:
    print("❌ CACHE IS WORKING - Second call was >90% faster")
else:
    print("✅ CACHE DISABLED - Latencies similar")

# Test 2: Same prompt, different temperatures
print("\n=== Test 2: Same prompt, different temperatures ===")

start = time.time()
result3 = predictor(question=q, config={"temperature": 0.00001})
latency3 = (time.time() - start) * 1000
print(f"Temperature=0.00001: {latency3:.2f}ms")

start = time.time()
result4 = predictor(question=q, config={"temperature": 0.00002})
latency4 = (time.time() - start) * 1000
print(f"Temperature=0.00002: {latency4:.2f}ms")

if latency4 < latency3 * 0.1:
    print("❌ CACHE STILL WORKING - Different temperatures didn't help")
else:
    print("✅ CACHE BUSTED - Different temperatures force new requests")

print("\n" + "="*60)
print("CONCLUSION:")
if latency2 < latency1 * 0.1 or latency4 < latency3 * 0.1:
    print("Cache is still active. cache=False might not work in this DSPy version.")
    print("Use temperature variation as primary cache-busting method.")
else:
    print("Cache successfully disabled! Both methods work.")
