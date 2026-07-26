#!/usr/bin/env python3
"""Paired statistics for the regime-vs-static isolation study.

Reads per-seed results from results/regime_vs_static_crn/seed*/results.csv
(3 seeds x 3 comfort targets x 4 conditions, 1,440 steps per episode, common
random numbers across conditions) and computes the paired comparisons reported
in the response: RegimeAware - Static, Oracle - Static, RandomSwitch - Static,
each paired per (seed, comfort_target) cell.

Usage:  python scripts/regime_paired_stats.py [root_dir]
"""
import glob
import sys

import pandas as pd
from scipy import stats

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'results/regime_vs_static_crn'

frames = [pd.read_csv(f) for f in sorted(glob.glob(f'{ROOT}/seed*/results.csv'))]
if not frames:
    sys.exit(f'no results.csv under {ROOT}')
df = pd.concat(frames, ignore_index=True)
print(f'{ROOT}: {len(df)} rows, seeds={sorted(df.seed.unique())}, '
      f'targets={sorted(df.comfort_target.unique())}, '
      f'conditions={sorted(df.condition.unique())}\n')

wide = df.pivot_table(index=['seed', 'comfort_target'], columns='condition',
                      values=['MO', 'kWh'], aggfunc='first')

for cond in ('RegimeAware', 'Oracle', 'RandomSwitch'):
    for metric in ('MO', 'kWh'):
        a = wide[(metric, cond)]
        b = wide[(metric, 'Static')]
        d = a - b
        t, p = stats.ttest_rel(a, b)
        print(f'{cond:12s} - Static  {metric:8s}: '
              f'Δ = {d.mean():+.4f} ± {d.std(ddof=1):.4f}  '
              f'(t = {t:.2f}, p = {p:.4f}, favor = {(d > 0).sum()}/{len(d)} cells)')

gap = (wide[('MO', 'Oracle')] - wide[('MO', 'Static')]).mean()
ra = (wide[('MO', 'RegimeAware')] - wide[('MO', 'Static')]).mean()
if abs(gap) > 1e-12:
    print(f'\nOracle-gap capture: RegimeAware achieves {100 * ra / gap:.1f}% '
          f'of the Oracle-Static MO gap ({gap:+.4f})')