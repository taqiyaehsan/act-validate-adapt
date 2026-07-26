#!/usr/bin/env python
"""Quick re-run: 3 comfort targets × 3 episodes × key arms with λ=0.1."""
import sys, os, csv, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from quick_smoke import VALIDATED_EDGES, INTERVENTION_DIRECTIONS, FakePipeline, load_scaling
from experiments_neurips import NeurIPSExperiments
import pandas as pd

SIM = 'open_window'
CFG = {
    'sim_path': f'js/{SIM}.js',
    'obs_data_path': f'data/challenge1_data_10k_processed.csv',
    'scaling_path': f'data/challenge1_data_10k_processed_scaling.csv',
    'dataset_type': SIM,
    'data_in_physical_units': False,
}

obs_df = pd.read_csv(CFG['obs_data_path'])
scaling = load_scaling(CFG['scaling_path'])
COMFORT_TARGETS = [0.5, 1.0, 2.0]
N_EPISODES = 3
# Only test arms that matter for the Pareto comparison
ARMS = ['ASHRAE', 'GRID', 'GRID_ObsOnly', 'DDPC_Subspace', 'DDPC_Behavioral']

rows = []
t0 = time.time()

for ct in COMFORT_TARGETS:
    print(f"\n--- comfort_target={ct} ---")
    exp = NeurIPSExperiments(
        pipeline=FakePipeline(VALIDATED_EDGES),
        dataset=obs_df,
        smart_room_path=CFG['sim_path'],
        comfort_target=ct,
        obs_only_edges=VALIDATED_EDGES,
        h0_edges=None, h1_edges=None,
        dataset_type=CFG['dataset_type'],
        scaling_ranges=scaling,
        data_in_physical_units=CFG['data_in_physical_units'],
    )
    exp.set_comfort_target(ct)  # builds policy arms

    for arm in ARMS:
        if arm not in exp.policy_arms:
            print(f"  {arm}: skipped (not available)")
            continue
        policy_fn = exp.policy_arms[arm]
        for ep in range(N_EPISODES):
            sat, eng, cv, cvr, kwh, dh = exp._evaluate_policy_episode_with_cv(
                policy_fn, arm, seed=ep)
            mo = 0.6 * (sat / 100.0) + 0.4 * (1.0 - eng / 100.0)
            rows.append({
                'arm': arm, 'discovery_seed': 0, 'episode_seed': ep,
                'comfort_target': ct, 'satisfaction': sat, 'energy': eng,
                'mo_score': mo, 'cv_mean': cv, 'cvr': cvr,
                'kwh': kwh, 'dh': dh, 'intervention_count': 216,
                'simulator': SIM,
            })
            print(f"  {arm} ep={ep}: sat={sat:.1f}% eng={eng:.1f}% kwh={kwh:.1f} dh={dh:.2f}")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s")

# Save CSV
OUT = 'neurips_results_regime_aware/b1_open_window_raw_lam01.csv'
keys = rows[0].keys()
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(f"Saved {len(rows)} rows to {OUT}")

# Generate Pareto plot
from run_exp_suite_B_prelim import generate_b1_plots, compute_hypervolume_from_rows
raw_df = pd.DataFrame(rows)
summary_df = (raw_df.groupby('arm')
              .agg(sat_mean=('satisfaction', 'mean'),
                   sat_std=('satisfaction', 'std'),
                   eng_mean=('energy', 'mean'),
                   eng_std=('energy', 'std'),
                   mo_mean=('mo_score', 'mean'),
                   mo_std=('mo_score', 'std'))
              .reset_index())

hv = compute_hypervolume_from_rows(raw_df.to_dict('records'))
print(f"\nCombined HV = {hv['combined']:.4f}")
print("Per-arm HV:")
for arm, val in sorted(hv['per_arm'].items(), key=lambda x: -x[1]):
    mk, md = hv['arm_means'][arm]
    print(f"  {arm:25s}  HV={val:.4f}  mean=({mk:.1f} kWh, {md:.1f} DH)")

generate_b1_plots(raw_df, summary_df, 'open_window',
                  output_dir='neurips_results_regime_aware')
print(f"\nPlot saved to neurips_results_regime_aware/b1_prelim_open_window_pareto.png")
