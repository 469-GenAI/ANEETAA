"""Check available DSPy optimizers"""
import dspy

print("=" * 60)
print("DSPy Optimizer Availability Check")
print("=" * 60)

# Check core DSPy version
print(f"\nDSPy version: {dspy.__version__}")

# Try importing various optimizers
optimizers_status = {}

# 1. BootstrapFewShot (most basic, always available)
try:
    from dspy.teleprompt import BootstrapFewShot
    optimizers_status['BootstrapFewShot'] = '✓ AVAILABLE (basic few-shot learning)'
except ImportError as e:
    optimizers_status['BootstrapFewShot'] = f'✗ NOT AVAILABLE: {e}'

# 2. MIPROv2 (better prompt optimization)
try:
    from dspy.teleprompt import MIPROv2
    optimizers_status['MIPROv2'] = '✓ AVAILABLE (prompt + demo optimization)'
except ImportError as e:
    optimizers_status['MIPROv2'] = f'✗ NOT AVAILABLE: {e}'

# 3. COPRO (older optimizer)
try:
    from dspy.teleprompt import COPRO
    optimizers_status['COPRO'] = '✓ AVAILABLE (prompt optimization)'
except ImportError as e:
    optimizers_status['COPRO'] = f'✗ NOT AVAILABLE: {e}'

# 4. SIMBA (newer optimizer, may not be in older versions)
try:
    from dspy import SIMBA
    optimizers_status['SIMBA'] = '✓ AVAILABLE (instruction + demo optimization)'
except ImportError:
    try:
        from dspy.teleprompt import SIMBA
        optimizers_status['SIMBA'] = '✓ AVAILABLE (instruction + demo optimization)'
    except ImportError as e:
        optimizers_status['SIMBA'] = f'✗ NOT AVAILABLE (requires DSPy >= 2.5.0)'

# Print results
print("\nOptimizer Status:")
print("-" * 60)
for name, status in optimizers_status.items():
    print(f"{name:20s}: {status}")

print("\n" + "=" * 60)
print("Recommendation:")
print("=" * 60)

if '✓' in optimizers_status.get('SIMBA', ''):
    print("✓ SIMBA available - Best for instruction + example optimization")
    print("  Use SIMBA for the most comprehensive optimization")
elif '✓' in optimizers_status.get('MIPROv2', ''):
    print("✓ MIPROv2 available - Great alternative to SIMBA")
    print("  Use MIPROv2 for prompt + demo optimization")
else:
    print("✓ BootstrapFewShot available - Basic but effective")
    print("  Use BootstrapFewShot for few-shot learning")

print("\nFor 4-way comparison, suggested agents:")
print("1. Vanilla ANEETAA (baseline)")
print("2. DSPy Baseline (unoptimized)")
if '✓' in optimizers_status.get('MIPROv2', ''):
    print("3. DSPy + MIPROv2 (prompt optimization)")
if '✓' in optimizers_status.get('BootstrapFewShot', ''):
    print("4. DSPy + BootstrapFewShot (few-shot learning)")
if '✓' in optimizers_status.get('SIMBA', ''):
    print("   OR DSPy + SIMBA (comprehensive optimization)")
