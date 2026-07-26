#!/usr/bin/env python
"""Quick: regenerate Pareto plot + HV from existing CSV."""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from run_exp_suite_B_prelim import (
    generate_b1_plots, compute_hypervolume_from_rows
)

RAW = 'neurips_results_regime_aware/b1_open_window_raw.csv'
OUT = 'neurips_results_regime_aware'

raw_df = pd.read_csv(RAW)

# Build summary_df that generate_b1_plots expects
summary_df = (raw_df.groupby('arm')
              .agg(sat_mean=('satisfaction', 'mean'),
                   sat_std=('satisfaction', 'std'),
                   eng_mean=('energy', 'mean'),
                   eng_std=('energy', 'std'),
                   mo_mean=('mo_score', 'mean'),
                   mo_std=('mo_score', 'std'))
              .reset_index())

# Print HV (paper A.3.4 / A.3.5 formulation)
hv = compute_hypervolume_from_rows(raw_df.to_dict('records'))
print(f"=== Combined HV (cross-policy Pareto) = {hv['combined']:.4f} ===")
print(f"\nPareto frontier points: {hv['frontier']}")
print(f"\n--- Per-arm HV (rectangle to nadir) ---")
for arm, val in sorted(hv['per_arm'].items(), key=lambda x: -x[1]):
    mk = '*' if any(abs(hv['arm_means'][arm][0] - fk) < 0.5
                     and abs(hv['arm_means'][arm][1] - fd) < 0.5
                     for fk, fd in hv['frontier']) else ' '
    mean_k, mean_d = hv['arm_means'][arm]
    print(f" {mk} {arm:25s}  HV={val:.4f}  mean=({mean_k:.1f} kWh, {mean_d:.1f} DH)")

# Regenerate all 4 plots
generate_b1_plots(raw_df, summary_df, 'open_window', output_dir=OUT)
print(f"\nPlots saved to {OUT}/b1_prelim_open_window_*.png")
