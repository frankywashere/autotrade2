#!/usr/bin/env python3
"""Test v4 directional evaluator."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class FakeResult:
    def __init__(self, metrics=None):
        self.metrics = metrics or {}
sys.modules['openevolve'] = types.ModuleType('openevolve')
fake_mod = types.ModuleType('openevolve.evaluation_result')
fake_mod.EvaluationResult = FakeResult
sys.modules['openevolve.evaluation_result'] = fake_mod

from v15.validation.openevolve_bounce.evaluator_v4 import evaluate
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'openevolve_bounce')

print("="*70)
print("EVALUATOR V4 (directional: long + short)")
print("="*70)

path = os.path.join(base, 'initial_program_v4.py')
result = evaluate(path)
m = result.metrics
print(f"\nSEED v4:")
print(f"  combined_score: {m.get('combined_score', 0):>15,.0f}")
print(f"  n_trades:       {m.get('n_trades', 0):>15.0f}  (long={m.get('n_long',0):.0f} short={m.get('n_short',0):.0f})")
print(f"  total_pnl:     ${m.get('total_pnl', 0):>14,.0f}")
print(f"  win_rate:       {m.get('win_rate', 0):>14.1%}  (long={m.get('long_wr',0):.1%} short={m.get('short_wr',0):.1%})")
print(f"  sharpe:         {m.get('sharpe', 0):>15.3f}")
print(f"  avg_adverse:    {m.get('avg_adverse', 0):>14.1%}")
print(f"  fire_rate:      {m.get('fire_rate', 0):>14.1%}")
print(f"  selectivity:    {m.get('selectivity', 0):>15.3f}")
