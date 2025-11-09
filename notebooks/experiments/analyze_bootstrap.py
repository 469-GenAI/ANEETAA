"""
Analyze BootstrapFewShot quality degradation
"""

import pandas as pd

# Load results
df = pd.read_csv('../four_way_comparison_results.csv')

# Filter for BootstrapFewShot
bs = df[df['agent_name'] == 'DSPy + BootstrapFewShot'].copy()
baseline = df[df['agent_name'] == 'DSPy Baseline'].copy()

print("="*60)
print("BOOTSTRAP FEWSHOT QUALITY ANALYSIS")
print("="*60)

print(f"\nOverall Metrics:")
print(f"  BootstrapFewShot: {bs['quality_score'].mean():.2f}/10 quality, {bs['is_correct'].mean()*100:.0f}% accuracy")
print(f"  Baseline:         {baseline['quality_score'].mean():.2f}/10 quality, {baseline['is_correct'].mean()*100:.0f}% accuracy")
print(f"  Difference:       {bs['quality_score'].mean() - baseline['quality_score'].mean():.2f} quality points")

# Compare question by question
print("\n" + "="*60)
print("QUESTION-BY-QUESTION COMPARISON")
print("="*60)

merged = bs.merge(baseline, on='question_id', suffixes=('_bs', '_base'))

print(f"\n{'Question ID':<30} {'Bootstrap':<15} {'Baseline':<15} {'Diff':<10}")
print("-"*70)

for idx, row in merged.iterrows():
    q_id = row['question_id'][-15:]
    bs_quality = row['quality_score_bs']
    base_quality = row['quality_score_base']
    diff = bs_quality - base_quality
    
    bs_correct = '✓' if row['is_correct_bs'] else '✗'
    base_correct = '✓' if row['is_correct_base'] else '✗'
    
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
    
    print(f"{q_id:<30} {bs_quality}/10 {bs_correct:<8} {base_quality}/10 {base_correct:<8} {arrow} {diff:+.1f}")

# Identify where Bootstrap did worse
print("\n" + "="*60)
print("QUESTIONS WHERE BOOTSTRAP DID WORSE (quality)")
print("="*60)

worse = merged[merged['quality_score_bs'] < merged['quality_score_base']].copy()
worse['diff'] = worse['quality_score_bs'] - worse['quality_score_base']
worse = worse.sort_values('diff')

for idx, row in worse.iterrows():
    print(f"\n{row['question_id']}")
    print(f"  Bootstrap: {row['quality_score_bs']}/10 | Baseline: {row['quality_score_base']}/10 | Diff: {row['diff']:.1f}")
    print(f"  Subject: {row['subject_bs']}")
    print(f"  Bootstrap Judge: {row['judge_feedback_bs'][:150]}...")
    print(f"  Baseline Judge:  {row['judge_feedback_base'][:150]}...")

# Check the few-shot examples in Bootstrap model
print("\n" + "="*60)
print("BOOTSTRAP FEW-SHOT EXAMPLES")
print("="*60)

import json
with open('../dspy_bootstrap_optimized.json', 'r') as f:
    bootstrap_model = json.load(f)

demos = bootstrap_model['predictor.predict']['demos']
print(f"\nNumber of few-shot examples: {len(demos)}")

for i, demo in enumerate(demos):
    print(f"\nExample {i+1}:")
    print(f"  Subject: {demo['subject']}")
    print(f"  Answer: {demo['answer']}")
    print(f"  Question preview: {demo['question'][:100]}...")
    print(f"  Reasoning preview: {demo['reasoning'][:150]}...")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("\nPossible reasons for quality degradation:")
print("1. Few-shot examples may be from different difficulty levels")
print("2. Examples all from Biology - may not help with Physics/Chemistry")
print("3. Bootstrap examples might use different reasoning style")
print("4. Model might be overfitting to the example style")
