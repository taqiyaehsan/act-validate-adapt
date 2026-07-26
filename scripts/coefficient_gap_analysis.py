#!/usr/bin/env python3
"""
Post-hoc analysis: Do validated edges have larger regime-dependent
coefficient gaps than obs-only (unvalidated) edges?

Reads saved edges from results/neurips_final/discovered_edges.json,
rebuilds monitors from training data, extracts per-edge coefficients
under occ_on/occ_off and win_on/win_off, and compares the gaps.

No pipeline re-run needed.
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks_new'))

from neurips_final_experiments import (
    OBSERVABLE_VARS, DATA_PATH, SCALING_PATH,
    build_observable_graph, build_predictive_model,
)

# ── Load data ────────────────────────────────────────────────────────────────
data = pd.read_csv(DATA_PATH)
data.columns = [c.lower() for c in data.columns]

# ── Load saved edges ─────────────────────────────────────────────────────────
with open('results/neurips_final/discovered_edges.json') as f:
    edges_data = json.load(f)

validated = {(s.lower(), t.lower()) for s, t in edges_data['consensus_validated']}
obs_only_all = {(s.lower(), t.lower()) for s, t in edges_data['consensus_obs_only']}
obs_only_not_validated = obs_only_all - validated  # edges that FAILED validation

print(f"Validated edges: {len(validated)}")
print(f"Obs-only total: {len(obs_only_all)}")
print(f"Obs-only NOT validated (rejected): {len(obs_only_not_validated)}")

# ── Filter to observable-only edges ──────────────────────────────────────────
val_obs = build_observable_graph(validated)
rej_obs = build_observable_graph(obs_only_not_validated)
# Use union of both for building models so we can extract coefficients for all
all_edges = val_obs | rej_obs

print(f"\nObservable validated: {len(val_obs)}")
print(f"Observable rejected: {len(rej_obs)}")

# ── Build regime-specific models ─────────────────────────────────────────────
occ_vals = data['occupancy'].values
win_vals = data['windowposition'].values

print("\nTraining occ_on / occ_off models...")
model_occ_on  = build_predictive_model(all_edges, data, occ_vals >= 0.1)
model_occ_off = build_predictive_model(all_edges, data, occ_vals < 0.1)

print("Training win_on / win_off models...")
model_win_on  = build_predictive_model(all_edges, data, win_vals >= 0.15)
model_win_off = build_predictive_model(all_edges, data, win_vals < 0.15)


def get_coefficient(model, source, target):
    """Extract learned weight for source→target from model."""
    params = model.parameters.get(target, {})
    return params.get('weights', {}).get(source, 0.0)


# ── Compute coefficient gaps per edge ────────────────────────────────────────
results = []
for source, target in all_edges:
    occ_on_coeff  = get_coefficient(model_occ_on, source, target)
    occ_off_coeff = get_coefficient(model_occ_off, source, target)
    win_on_coeff  = get_coefficient(model_win_on, source, target)
    win_off_coeff = get_coefficient(model_win_off, source, target)

    occ_gap = abs(occ_on_coeff - occ_off_coeff)
    win_gap = abs(win_on_coeff - win_off_coeff)
    max_gap = max(occ_gap, win_gap)

    is_validated = (source, target) in val_obs
    results.append({
        'source': source,
        'target': target,
        'validated': is_validated,
        'occ_on_coeff': round(occ_on_coeff, 4),
        'occ_off_coeff': round(occ_off_coeff, 4),
        'occ_gap': round(occ_gap, 4),
        'win_on_coeff': round(win_on_coeff, 4),
        'win_off_coeff': round(win_off_coeff, 4),
        'win_gap': round(win_gap, 4),
        'max_gap': round(max_gap, 4),
    })

df = pd.DataFrame(results)
df.to_csv('results/neurips_final/coefficient_gap_analysis.csv', index=False)

# ── Summary statistics ───────────────────────────────────────────────────────
val_df = df[df['validated']]
rej_df = df[~df['validated']]

print("\n" + "=" * 60)
print("COEFFICIENT GAP ANALYSIS")
print("=" * 60)
print(f"\n{'':30s} {'Validated':>12s} {'Rejected':>12s}")
print(f"{'':30s} {'(n=' + str(len(val_df)) + ')':>12s} {'(n=' + str(len(rej_df)) + ')':>12s}")
print("-" * 56)
print(f"{'Mean occ gap':30s} {val_df['occ_gap'].mean():12.4f} {rej_df['occ_gap'].mean():12.4f}")
print(f"{'Mean win gap':30s} {val_df['win_gap'].mean():12.4f} {rej_df['win_gap'].mean():12.4f}")
print(f"{'Mean max gap':30s} {val_df['max_gap'].mean():12.4f} {rej_df['max_gap'].mean():12.4f}")
print(f"{'Median max gap':30s} {val_df['max_gap'].median():12.4f} {rej_df['max_gap'].median():12.4f}")

# Mann-Whitney U test (non-parametric, doesn't assume normality)
from scipy.stats import mannwhitneyu
if len(val_df) > 0 and len(rej_df) > 0:
    stat, pval = mannwhitneyu(val_df['max_gap'], rej_df['max_gap'], alternative='greater')
    print(f"\nMann-Whitney U (validated > rejected): U={stat:.1f}, p={pval:.4f}")
    if pval < 0.05:
        print("  → Significant (p < 0.05): validated edges have larger regime gaps")
    else:
        print("  → Not significant (p >= 0.05)")

# ── Top edges by gap ─────────────────────────────────────────────────────────
print("\n\nTop 10 edges by max regime gap:")
print(f"{'Edge':35s} {'Validated':>10s} {'Occ Gap':>10s} {'Win Gap':>10s} {'Max Gap':>10s}")
print("-" * 77)
for _, row in df.nlargest(10, 'max_gap').iterrows():
    tag = "YES" if row['validated'] else "no"
    print(f"{row['source']:>15s} → {row['target']:<15s} {tag:>10s} "
          f"{row['occ_gap']:10.4f} {row['win_gap']:10.4f} {row['max_gap']:10.4f}")

# ── Box plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, col, title in zip(axes,
                           ['occ_gap', 'win_gap', 'max_gap'],
                           ['Occupancy Coeff Gap', 'Window Coeff Gap', 'Max Coeff Gap']):
    val_vals = val_df[col].values
    rej_vals = rej_df[col].values

    bp = ax.boxplot([val_vals, rej_vals],
                    labels=['Validated', 'Rejected'],
                    patch_artist=True,
                    widths=0.5)
    bp['boxes'][0].set_facecolor('#009E73')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#D55E00')
    bp['boxes'][1].set_alpha(0.7)

    # Overlay individual points
    for i, (vals, x) in enumerate([(val_vals, 1), (rej_vals, 2)]):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
        ax.scatter(x + jitter, vals, alpha=0.4, s=20, color='black', zorder=3)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel('|coeff_on - coeff_off|')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Regime Sensitivity: Validated vs Rejected Edges', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/neurips_final/fig_coefficient_gap_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved: results/neurips_final/fig_coefficient_gap_analysis.png")
print("Saved: results/neurips_final/coefficient_gap_analysis.csv")
