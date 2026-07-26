#!/usr/bin/env python3
"""
Comprehensive Post-Hoc Analyses for PolicyGRID NeurIPS Submission
=================================================================
All analyses run on saved results — no pipeline re-run needed.

Outputs:
  results/neurips_final/posthoc/
    1_edge_overlap_gt.png          — Per-method GT edge recovery heatmap
    2_prediction_discriminability.png — Cumulative error gap: correct vs wrong model
    3_coefficient_gap.png          — Validated vs rejected edge regime sensitivity
    4_seed_stability.png           — Per-seed Jaccard + edge agreement heatmap
    5_monitoring_confusion.png     — 4-way regime confusion matrix (PolicyGRID)
    6_monitoring_calibration.png   — P(occ) and P(win) calibration plots
    7_monitoring_temporal.png      — Accuracy by time-of-day and regime transition
    8_discovery_method_contribution.png — What each of PC/SAM/LLM/VARLiNGAM contributes
    9_policy_variance.png          — Per-run policy variance (not just means)
    10_ablation_critical_sensor.png — Which sensor matters most per method
    posthoc_summary.txt            — Text summary of all analyses
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks_new'))

from neurips_final_experiments import (
    OBSERVABLE_VARS, SENSOR_VARS, LATENT_VARS, DATA_PATH, SCALING_PATH, SIM_PATH,
    GT_PATH, OUT_DIR,
    build_observable_graph, build_predictive_model, construct_factored_monitors,
    run_monitoring, compute_monitoring_metrics, collect_sim_data, get_scaling_map,
)

OUT = 'results/neurips_final/posthoc'
os.makedirs(OUT, exist_ok=True)

# ── Load all saved data ──────────────────────────────────────────────────────
data = pd.read_csv(DATA_PATH)
data.columns = [c.lower() for c in data.columns]
scaling = pd.read_csv(SCALING_PATH)
smap = get_scaling_map(scaling)

with open('results/neurips_final/discovered_edges.json') as f:
    edges_data = json.load(f)

with open(GT_PATH) as f:
    gt_all = json.load(f)
gt_edges = {(s.lower(), t.lower()) for s, t in gt_all['smart_building_rich']['edges']}

disc_df = pd.read_csv('results/neurips_final/exp1_benchmarks/discovery_metrics.csv')
mon_df = pd.read_csv('results/neurips_final/exp1_benchmarks/monitoring_metrics.csv')
pol_df = pd.read_csv('results/neurips_final/exp1_benchmarks/policy_metrics.csv')
abl_df = pd.read_csv('results/neurips_final/exp2_ablation/ablation_metrics.csv')
abl_raw = pd.read_csv('results/neurips_final/exp2_ablation/ablation_metrics_raw.csv')

validated = {(s.lower(), t.lower()) for s, t in edges_data['consensus_validated']}
obs_only_all = {(s.lower(), t.lower()) for s, t in edges_data['consensus_obs_only']}

# Load per-seed edges
per_seed_val = [
    {(s.lower(), t.lower()) for s, t in seed_edges}
    for seed_edges in edges_data['per_seed_validated']
]
per_seed_obs = [
    {(s.lower(), t.lower()) for s, t in seed_edges}
    for seed_edges in edges_data['per_seed_obs_only']
]

# Load benchmark edges (re-run benchmarks from saved data)
benchmark_edges = {}
try:
    for name, module_name, func_name in [
        ('PC', 'pc_baseline', 'run_pc_baseline'),
        ('SAM', 'sam_baseline', 'run_sam_baseline'),
        ('GIES', 'gies_comparison', 'run_gies'),
        ('ICP', 'icp_comparison', 'run_icp'),
        ('NOTEARS-I', 'notears_i', 'run_notears_i'),
        ('ABCD', 'abcd_comparison', 'run_abcd'),
        ('JCI', 'jci_comparison', 'run_jci'),
        ('Causal Bandits', 'causal_bandits', 'run_causal_bandits'),
        ('IID', 'iid_comparison', 'run_iid'),
    ]:
        try:
            mod = __import__(module_name)
            func = getattr(mod, func_name)
            edges = func(data)
            benchmark_edges[name] = {(s.lower(), t.lower()) for s, t in edges}
        except Exception as e:
            print(f"  {name} benchmark reload failed: {e}")
            benchmark_edges[name] = set()
except Exception as e:
    print(f"Benchmark reload failed: {e}")

summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append("POST-HOC ANALYSES — PolicyGRID NeurIPS Submission")
summary_lines.append("=" * 70)

# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Per-method GT edge recovery heatmap
# ═════════════════════════════════════════════════════════════════════════════
print("\n[1/10] GT Edge Recovery Heatmap...")

all_methods = {
    'PolicyGRID-O': obs_only_all,
    'PolicyGRID': validated,
}
all_methods.update(benchmark_edges)

gt_list = sorted(gt_edges)
method_names = list(all_methods.keys())

# Build binary matrix: which method found which GT edge
matrix = np.zeros((len(gt_list), len(method_names)))
for j, name in enumerate(method_names):
    for i, edge in enumerate(gt_list):
        if edge in all_methods[name]:
            matrix[i, j] = 1

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(matrix, aspect='auto', cmap='YlGn', interpolation='nearest')
ax.set_xticks(range(len(method_names)))
ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(gt_list)))
ax.set_yticklabels([f"{s}→{t}" for s, t in gt_list], fontsize=7)
ax.set_xlabel('Method')
ax.set_ylabel('Ground Truth Edge')
ax.set_title('GT Edge Recovery: Which Methods Find Which True Edges', fontsize=13)

# Add recovery count per method at top
for j, name in enumerate(method_names):
    count = int(matrix[:, j].sum())
    ax.text(j, -1.2, f"{count}/{len(gt_list)}", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
fig.savefig(f'{OUT}/1_edge_overlap_gt.png', dpi=150, bbox_inches='tight')
plt.close()

# Summary stats
summary_lines.append("\n\nANALYSIS 1: GT Edge Recovery")
summary_lines.append("-" * 40)
for name in method_names:
    edges = all_methods[name]
    tp = len(edges & gt_edges)
    summary_lines.append(f"  {name:25s}: {tp}/{len(gt_edges)} GT edges recovered")

# Which GT edges NO method finds?
found_by_any = set()
for edges in all_methods.values():
    found_by_any.update(edges & gt_edges)
never_found = gt_edges - found_by_any
summary_lines.append(f"\n  GT edges found by NO method: {len(never_found)}")
for e in sorted(never_found):
    summary_lines.append(f"    {e[0]} → {e[1]}")

# GT edges found ONLY by PolicyGRID (validated)
only_pgrid = (validated & gt_edges) - set().union(*[
    benchmark_edges.get(n, set()) & gt_edges for n in benchmark_edges
])
summary_lines.append(f"\n  GT edges found ONLY by PolicyGRID (validated): {len(only_pgrid)}")
for e in sorted(only_pgrid):
    summary_lines.append(f"    {e[0]} → {e[1]}")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Prediction Error Discriminability
# ═════════════════════════════════════════════════════════════════════════════
print("[2/10] Prediction Error Discriminability...")

# Build monitors for PolicyGRID and a few benchmarks, then measure
# how well the correct model predicts vs the wrong model
methods_to_test = {
    'PolicyGRID': validated,
    'PolicyGRID-O': obs_only_all,
}
# Add top benchmarks
for name in ['PC', 'GIES', 'ICP', 'Causal Bandits']:
    if name in benchmark_edges and len(benchmark_edges[name]) > 0:
        methods_to_test[name] = benchmark_edges[name]

# Collect sim data (1 seed, reuse for all analyses needing sim)
print("  Collecting sim data for post-hoc analyses (1 seed, 5760 steps / 4 days)...")
sim_records = collect_sim_data(smap, duration_steps=5760)

discrim_results = {}
for name, edges in methods_to_test.items():
    monitors = construct_factored_monitors(edges, data)
    if monitors is None:
        continue

    # 4-way monitors: compute per-regime prediction errors
    regime_names = ['base', 'occ', 'win', 'full']
    occ_err_gaps = []
    win_err_gaps = []
    for rec in sim_records:
        obs = rec['state_obs']
        occ_gt = rec['occ_active']
        win_gt = rec['win_active']

        errs = {}
        for rname in regime_names:
            pred = monitors[rname].predict(obs)
            errs[rname] = sum(abs(obs.get(v, 0.5) - pred.get(v, 0.5))
                              for v in SENSOR_VARS)

        # Occ gap: avg error of occ-OFF regimes minus avg error of occ-ON regimes
        err_occ_off = (errs['base'] + errs['win']) / 2
        err_occ_on = (errs['occ'] + errs['full']) / 2
        if occ_gt:
            occ_err_gaps.append(err_occ_off - err_occ_on)
        else:
            occ_err_gaps.append(err_occ_on - err_occ_off)

        # Win gap: avg error of win-OFF regimes minus avg error of win-ON regimes
        err_win_off = (errs['base'] + errs['occ']) / 2
        err_win_on = (errs['win'] + errs['full']) / 2
        if win_gt:
            win_err_gaps.append(err_win_off - err_win_on)
        else:
            win_err_gaps.append(err_win_on - err_win_off)

    discrim_results[name] = {
        'occ_gap_mean': np.mean(occ_err_gaps),
        'occ_gap_std': np.std(occ_err_gaps),
        'occ_gap_positive_frac': np.mean(np.array(occ_err_gaps) > 0),
        'win_gap_mean': np.mean(win_err_gaps),
        'win_gap_std': np.std(win_err_gaps),
        'win_gap_positive_frac': np.mean(np.array(win_err_gaps) > 0),
        'occ_gaps': occ_err_gaps,
        'win_gaps': win_err_gaps,
    }

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
method_order = ['PolicyGRID', 'PolicyGRID-O'] + [n for n in discrim_results if n not in ('PolicyGRID', 'PolicyGRID-O')]
colors_map = {'PolicyGRID': '#000000', 'PolicyGRID-O': '#0072B2',
              'PC': '#D55E00', 'GIES': '#E69F00', 'ICP': '#56B4E9', 'Causal Bandits': '#999999'}

for ax, dim, title in zip(axes, ['occ', 'win'], ['Occupancy', 'Window']):
    names = []
    means = []
    stds = []
    fracs = []
    cols = []
    for name in method_order:
        if name not in discrim_results:
            continue
        r = discrim_results[name]
        names.append(name)
        means.append(r[f'{dim}_gap_mean'])
        stds.append(r[f'{dim}_gap_std'])
        fracs.append(r[f'{dim}_gap_positive_frac'])
        cols.append(colors_map.get(name, '#999999'))

    x = range(len(names))
    bars = ax.bar(x, means, yerr=stds, color=cols, alpha=0.7, edgecolor='black',
                  capsize=4)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5, label='No discriminability')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Mean Error Gap\n(err_wrong − err_correct)')
    ax.set_title(f'{title} Monitor Discriminability', fontsize=12)

    # Annotate with fraction where correct model wins
    for i, (bar, frac) in enumerate(zip(bars, fracs)):
        ax.text(i, bar.get_height() + stds[i] + 0.01,
                f'{frac:.0%}', ha='center', va='bottom', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Prediction Error Discriminability: Correct vs Wrong Regime Model',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/2_prediction_discriminability.png', dpi=150, bbox_inches='tight')
plt.close()

summary_lines.append("\n\nANALYSIS 2: Prediction Error Discriminability")
summary_lines.append("-" * 40)
summary_lines.append(f"  {'Method':25s} {'Occ Gap':>10s} {'Occ Win%':>10s} {'Win Gap':>10s} {'Win Win%':>10s}")
for name in method_order:
    if name not in discrim_results:
        continue
    r = discrim_results[name]
    summary_lines.append(f"  {name:25s} {r['occ_gap_mean']:10.4f} {r['occ_gap_positive_frac']:10.1%} "
                         f"{r['win_gap_mean']:10.4f} {r['win_gap_positive_frac']:10.1%}")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Coefficient Gap (already have this, redo cleaner)
# ═════════════════════════════════════════════════════════════════════════════
print("[3/10] Coefficient Gap Analysis...")

val_obs = build_observable_graph(validated)
rej_edges = obs_only_all - validated
rej_obs = build_observable_graph(rej_edges)
all_edges_union = val_obs | rej_obs

occ_vals = data['occupancy'].values
win_vals = data['windowposition'].values

model_occ_on = build_predictive_model(all_edges_union, data, occ_vals >= 0.1)
model_occ_off = build_predictive_model(all_edges_union, data, occ_vals < 0.1)
model_win_on = build_predictive_model(all_edges_union, data, win_vals >= 0.15)
model_win_off = build_predictive_model(all_edges_union, data, win_vals < 0.15)

def get_coeff(model, source, target):
    return model.parameters.get(target, {}).get('weights', {}).get(source, 0.0)

coeff_rows = []
for source, target in all_edges_union:
    occ_gap = abs(get_coeff(model_occ_on, source, target) - get_coeff(model_occ_off, source, target))
    win_gap = abs(get_coeff(model_win_on, source, target) - get_coeff(model_win_off, source, target))
    coeff_rows.append({
        'edge': f"{source}→{target}",
        'validated': (source, target) in val_obs,
        'in_gt': (source, target) in gt_edges,
        'occ_gap': occ_gap,
        'win_gap': win_gap,
        'max_gap': max(occ_gap, win_gap),
    })

coeff_df = pd.DataFrame(coeff_rows)
coeff_df.to_csv(f'{OUT}/coefficient_gaps.csv', index=False)

# Box plot + strip
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, title in zip(axes, ['occ_gap', 'win_gap', 'max_gap'],
                           ['Occupancy Gap', 'Window Gap', 'Max Gap']):
    val_vals = coeff_df[coeff_df['validated']][col].values
    rej_vals = coeff_df[~coeff_df['validated']][col].values
    gt_vals = coeff_df[coeff_df['in_gt']][col].values
    non_gt_vals = coeff_df[~coeff_df['in_gt']][col].values

    bp = ax.boxplot([val_vals, rej_vals, gt_vals, non_gt_vals],
                    labels=['Validated', 'Rejected', 'In GT', 'Not in GT'],
                    patch_artist=True, widths=0.5)
    colors = ['#009E73', '#D55E00', '#0072B2', '#CC79A7']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    rng = np.random.default_rng(42)
    for i, vals in enumerate([val_vals, rej_vals, gt_vals, non_gt_vals]):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(i + 1 + jitter, vals, alpha=0.4, s=15, color='black', zorder=3)

    ax.set_title(title, fontsize=11)
    ax.set_ylabel('|coeff_on − coeff_off|')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Regime Sensitivity: Validated vs Rejected & GT vs Non-GT Edges',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/3_coefficient_gap.png', dpi=150, bbox_inches='tight')
plt.close()

from scipy.stats import mannwhitneyu
summary_lines.append("\n\nANALYSIS 3: Coefficient Gap (Regime Sensitivity)")
summary_lines.append("-" * 40)
for label, group_a, group_b, a_name, b_name in [
    ('Validated vs Rejected', coeff_df[coeff_df['validated']]['max_gap'],
     coeff_df[~coeff_df['validated']]['max_gap'], 'Validated', 'Rejected'),
    ('In GT vs Not in GT', coeff_df[coeff_df['in_gt']]['max_gap'],
     coeff_df[~coeff_df['in_gt']]['max_gap'], 'In GT', 'Not in GT'),
]:
    if len(group_a) > 0 and len(group_b) > 0:
        stat, pval = mannwhitneyu(group_a, group_b, alternative='greater')
        summary_lines.append(f"  {label}: {a_name} mean={group_a.mean():.4f}, "
                             f"{b_name} mean={group_b.mean():.4f}, U={stat:.0f}, p={pval:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Seed Stability
# ═════════════════════════════════════════════════════════════════════════════
print("[4/10] Seed Stability Analysis...")

seeds = edges_data['seeds']
n_seeds = len(seeds)

# Jaccard similarity between all seed pairs
def jaccard(a, b):
    if len(a | b) == 0:
        return 1.0
    return len(a & b) / len(a | b)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, edge_type, per_seed in zip(axes, ['Validated', 'Obs-Only'],
                                    [per_seed_val, per_seed_obs]):
    jac_matrix = np.zeros((n_seeds, n_seeds))
    for i in range(n_seeds):
        for j in range(n_seeds):
            jac_matrix[i, j] = jaccard(per_seed[i], per_seed[j])

    im = ax.imshow(jac_matrix, cmap='YlGn', vmin=0, vmax=1)
    for i in range(n_seeds):
        for j in range(n_seeds):
            ax.text(j, i, f'{jac_matrix[i,j]:.3f}', ha='center', va='center', fontsize=10)
    ax.set_xticks(range(n_seeds))
    ax.set_yticks(range(n_seeds))
    ax.set_xticklabels([f'Seed {s}' for s in seeds])
    ax.set_yticklabels([f'Seed {s}' for s in seeds])
    ax.set_title(f'{edge_type} Edge Jaccard Similarity', fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Edge counts
    for i, s in enumerate(seeds):
        ax.text(n_seeds + 0.3, i, f'n={len(per_seed[i])}', va='center', fontsize=9)

plt.suptitle('Discovery Stability Across Seeds', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/4_seed_stability.png', dpi=150, bbox_inches='tight')
plt.close()

# Which edges appear in all 3 seeds vs only 1?
val_in_all = per_seed_val[0] & per_seed_val[1] & per_seed_val[2]
val_in_any = per_seed_val[0] | per_seed_val[1] | per_seed_val[2]
val_in_one = val_in_any - (per_seed_val[0] & per_seed_val[1]) - \
             (per_seed_val[0] & per_seed_val[2]) - \
             (per_seed_val[1] & per_seed_val[2]) | val_in_all
# Simpler: count per edge
val_edge_counts = defaultdict(int)
for seed_set in per_seed_val:
    for e in seed_set:
        val_edge_counts[e] += 1

summary_lines.append("\n\nANALYSIS 4: Seed Stability")
summary_lines.append("-" * 40)
summary_lines.append(f"  Validated edge counts across seeds:")
summary_lines.append(f"    In all 3 seeds: {sum(1 for c in val_edge_counts.values() if c == 3)}")
summary_lines.append(f"    In 2 seeds:     {sum(1 for c in val_edge_counts.values() if c == 2)}")
summary_lines.append(f"    In 1 seed only: {sum(1 for c in val_edge_counts.values() if c == 1)}")
summary_lines.append(f"  Consensus (majority ≥2): {len(validated)} edges")

# Which GT edges are stable?
gt_stable = {e for e in gt_edges if val_edge_counts.get(e, 0) >= 2}
gt_unstable = {e for e in (validated & gt_edges) if val_edge_counts.get(e, 0) < 2}
summary_lines.append(f"  GT edges in consensus: {len(gt_stable)}/{len(gt_edges)}")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Monitoring Confusion Matrix
# ═════════════════════════════════════════════════════════════════════════════
print("[5/10] Monitoring Confusion Matrix...")

pgrid_monitors = construct_factored_monitors(validated, data)
pgrid_recs = run_monitoring(pgrid_monitors, sim_records)
pgrid_recs_df = pd.DataFrame(pgrid_recs)

regime_map = {0: 'base', 1: 'occ', 2: 'win', 3: 'full'}
labels = ['base', 'occ', 'win', 'full']
gt_labels = pgrid_recs_df['regime_gt'].map(regime_map)
pred_labels = pgrid_recs_df['winner']

# Build confusion matrix
conf = np.zeros((4, 4), dtype=int)
for gt, pred in zip(gt_labels, pred_labels):
    i = labels.index(gt)
    j = labels.index(pred)
    conf[i, j] += 1

# Normalize per row (recall per class)
conf_norm = conf.astype(float) / conf.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, mat, title, fmt in zip(axes,
                                [conf, conf_norm],
                                ['Counts', 'Recall (row-normalized)'],
                                ['d', '.2f']):
    im = ax.imshow(mat, cmap='Blues', interpolation='nearest')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{mat[i,j]:{fmt}}', ha='center', va='center',
                    fontsize=11, color='white' if mat[i,j] > mat.max()*0.6 else 'black')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted Regime')
    ax.set_ylabel('True Regime')
    ax.set_title(f'PolicyGRID — {title}', fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('4-Way Regime Detection Confusion Matrix (PolicyGRID)', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/5_monitoring_confusion.png', dpi=150, bbox_inches='tight')
plt.close()

summary_lines.append("\n\nANALYSIS 5: Confusion Matrix (PolicyGRID)")
summary_lines.append("-" * 40)
for i, gt in enumerate(labels):
    row_total = conf[i].sum()
    correct = conf[i, i]
    summary_lines.append(f"  True={gt:5s}: {correct}/{row_total} correct ({conf_norm[i,i]:.1%})")
    errors = [(labels[j], conf[i,j]) for j in range(4) if j != i and conf[i,j] > 0]
    if errors:
        summary_lines.append(f"    Errors: {', '.join(f'{l}={c}' for l, c in errors)}")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 6: Calibration Plot
# ═════════════════════════════════════════════════════════════════════════════
print("[6/10] Monitoring Calibration...")

# Bin P(occ) and P(win), compute actual frequency in each bin
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, dim, gt_col in zip(axes, ['P_occ', 'P_win'], ['occ_active', 'win_active']):
    probs = pgrid_recs_df[dim].values
    gts = pgrid_recs_df[gt_col].values.astype(float)

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_freqs = []
    bin_counts = []

    for b in range(n_bins):
        mask = (probs >= bin_edges[b]) & (probs < bin_edges[b+1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[b] + bin_edges[b+1]) / 2)
            bin_freqs.append(gts[mask].mean())
            bin_counts.append(mask.sum())

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.scatter(bin_centers, bin_freqs, s=[max(20, c/5) for c in bin_counts],
               color='#0072B2', alpha=0.7, edgecolor='black', zorder=3)
    ax.plot(bin_centers, bin_freqs, color='#0072B2', alpha=0.5)

    # Annotate bin counts
    for bc, bf, cnt in zip(bin_centers, bin_freqs, bin_counts):
        ax.annotate(f'n={cnt}', (bc, bf), textcoords='offset points',
                    xytext=(5, 5), fontsize=7, alpha=0.7)

    ax.set_xlabel(f'Predicted {dim}')
    ax.set_ylabel('Actual Frequency')
    ax.set_title(f'{dim} Calibration', fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    # Brier score
    brier = np.mean((probs - gts) ** 2)
    ax.text(0.05, 0.9, f'Brier={brier:.4f}', transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('PolicyGRID Monitoring Calibration', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/6_monitoring_calibration.png', dpi=150, bbox_inches='tight')
plt.close()

brier_occ = np.mean((pgrid_recs_df['P_occ'].values - pgrid_recs_df['occ_active'].values.astype(float)) ** 2)
brier_win = np.mean((pgrid_recs_df['P_win'].values - pgrid_recs_df['win_active'].values.astype(float)) ** 2)
summary_lines.append("\n\nANALYSIS 6: Calibration")
summary_lines.append("-" * 40)
summary_lines.append(f"  PolicyGRID Brier score (occ): {brier_occ:.4f}")
summary_lines.append(f"  PolicyGRID Brier score (win): {brier_win:.4f}")
summary_lines.append(f"  (Lower is better. Random baseline = 0.25)")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 7: Temporal Accuracy (by time-of-day and near transitions)
# ═════════════════════════════════════════════════════════════════════════════
print("[7/10] Temporal Accuracy Analysis...")

pgrid_recs_df['hour_mod'] = pgrid_recs_df['hour'] % 24
pgrid_recs_df['gt_label'] = pgrid_recs_df['regime_gt'].map(regime_map)
pgrid_recs_df['correct'] = (pgrid_recs_df['winner'] == pgrid_recs_df['gt_label']).astype(int)

# Detect regime transitions
pgrid_recs_df['transition'] = (pgrid_recs_df['regime_gt'] != pgrid_recs_df['regime_gt'].shift(1)).astype(int)
pgrid_recs_df['near_transition'] = pgrid_recs_df['transition'].rolling(window=20, center=True, min_periods=1).max()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 7a: Accuracy by hour of day
hourly = pgrid_recs_df.groupby(pgrid_recs_df['hour_mod'].astype(int))['correct'].mean()
axes[0].bar(hourly.index, hourly.values, color='#0072B2', alpha=0.7, edgecolor='black')
axes[0].axhline(0.9, color='green', linestyle='--', alpha=0.5, label='90% target')
axes[0].axhline(0.25, color='red', linestyle='--', alpha=0.5, label='Random (25%)')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('4-Way Accuracy')
axes[0].set_title('Accuracy by Time of Day', fontsize=12)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# 7b: Accuracy near transitions vs stable
near_trans = pgrid_recs_df[pgrid_recs_df['near_transition'] == 1]['correct'].mean()
stable = pgrid_recs_df[pgrid_recs_df['near_transition'] == 0]['correct'].mean()
bars = axes[1].bar(['Near Transition\n(±10 steps)', 'Stable'],
                    [near_trans, stable],
                    color=['#D55E00', '#009E73'], alpha=0.7, edgecolor='black')
for bar, val in zip(bars, [near_trans, stable]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.1%}', ha='center', fontsize=11)
axes[1].set_ylabel('4-Way Accuracy')
axes[1].set_title('Accuracy: Transitions vs Stable Periods', fontsize=12)
axes[1].set_ylim(0, 1.1)
axes[1].grid(axis='y', alpha=0.3)

# 7c: Accuracy per regime
per_regime = pgrid_recs_df.groupby('gt_label')['correct'].agg(['mean', 'count'])
regime_colors = {'base': '#56B4E9', 'occ': '#E69F00', 'win': '#009E73', 'full': '#D55E00'}
for i, (regime, row) in enumerate(per_regime.iterrows()):
    axes[2].bar(i, row['mean'], color=regime_colors.get(regime, '#999999'),
                alpha=0.7, edgecolor='black')
    axes[2].text(i, row['mean'] + 0.01, f"{row['mean']:.1%}\nn={int(row['count'])}",
                 ha='center', fontsize=9)
axes[2].set_xticks(range(len(per_regime)))
axes[2].set_xticklabels(per_regime.index)
axes[2].set_ylabel('4-Way Accuracy')
axes[2].set_title('Accuracy per True Regime', fontsize=12)
axes[2].set_ylim(0, 1.1)
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('PolicyGRID Monitoring: When Does It Succeed and Fail?', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/7_monitoring_temporal.png', dpi=150, bbox_inches='tight')
plt.close()

summary_lines.append("\n\nANALYSIS 7: Temporal Accuracy")
summary_lines.append("-" * 40)
summary_lines.append(f"  Near transitions (±10 steps): {near_trans:.1%}")
summary_lines.append(f"  Stable periods:               {stable:.1%}")
summary_lines.append(f"  Per regime:")
for regime, row in per_regime.iterrows():
    summary_lines.append(f"    {regime:5s}: {row['mean']:.1%} (n={int(row['count'])})")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 8: Discovery Method Contribution (PC/SAM/LLM/VARLiNGAM)
# ═════════════════════════════════════════════════════════════════════════════
print("[8/10] Discovery Method Contribution...")

# We can get per-method DAGs from re-running benchmarks, but we don't have
# the internal PC/SAM/LLM/VARLiNGAM breakdown saved. However, we can analyze
# what the obs-only union contains that each benchmark finds.

# What we CAN do: analyze which benchmark methods' edges overlap with
# PolicyGRID's validated edges (what does validation rescue vs reject)
fig, ax = plt.subplots(figsize=(12, 6))

method_names_bench = [n for n in benchmark_edges if len(benchmark_edges[n]) > 0]
overlap_data = []
for name in method_names_bench:
    edges = benchmark_edges[name]
    shared_with_val = len(edges & validated)
    shared_with_gt = len(edges & gt_edges)
    only_in_bench = len(edges - validated - gt_edges)
    overlap_data.append({
        'method': name,
        'shared_val': shared_with_val,
        'shared_gt': shared_with_gt,
        'unique_wrong': only_in_bench,
        'total': len(edges),
    })

overlap_df = pd.DataFrame(overlap_data)
x = range(len(overlap_df))
w = 0.25
ax.bar([i - w for i in x], overlap_df['shared_gt'], w, label='Shared with GT', color='#009E73', alpha=0.7, edgecolor='black')
ax.bar(x, overlap_df['shared_val'], w, label='Shared with PolicyGRID-Val', color='#0072B2', alpha=0.7, edgecolor='black')
ax.bar([i + w for i in x], overlap_df['unique_wrong'], w, label='Unique to benchmark (not in GT)', color='#D55E00', alpha=0.7, edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(overlap_df['method'], rotation=30, ha='right')
ax.set_ylabel('Number of Edges')
ax.set_title('Benchmark Edge Overlap with GT and PolicyGRID Validated', fontsize=12)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(f'{OUT}/8_discovery_method_contribution.png', dpi=150, bbox_inches='tight')
plt.close()

summary_lines.append("\n\nANALYSIS 8: Benchmark Edge Overlap")
summary_lines.append("-" * 40)
summary_lines.append(f"  {'Method':20s} {'Total':>6s} {'∩ GT':>6s} {'∩ Val':>6s} {'Unique wrong':>13s}")
for _, row in overlap_df.iterrows():
    summary_lines.append(f"  {row['method']:20s} {row['total']:6d} {row['shared_gt']:6d} "
                         f"{row['shared_val']:6d} {row['unique_wrong']:13d}")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 9: Policy Variance (per-run, not just means)
# ═════════════════════════════════════════════════════════════════════════════
print("[9/10] Policy Variance Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics_to_plot = [('kwh', 'kWh'), ('dh', 'Degree-Hours'), ('mo_score', 'MO Score')]

policy_methods = pol_df['method'].unique()
ct_default = 0.8  # focus on default comfort target

for ax, (metric, label) in zip(axes, metrics_to_plot):
    sub = pol_df[pol_df['comfort_target'] == ct_default]
    method_data = []
    method_labels = []
    for m in policy_methods:
        vals = sub[sub['method'] == m][metric].values
        if len(vals) > 0:
            method_data.append(vals)
            method_labels.append(m)

    bp = ax.boxplot(method_data, labels=method_labels, patch_artist=True, widths=0.5)
    for i, (patch, m) in enumerate(zip(bp['boxes'], method_labels)):
        c = {'PolicyGRID-O': '#0072B2', 'PolicyGRID': '#000000', 'PolicyGRID-R': '#009E73',
             'DDPC_Behavioral': '#D55E00', 'DDPC_Subspace': '#E69F00',
             'DDPC_Neural': '#CC79A7', 'PID': '#999999'}.get(m, '#999999')
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    # Overlay individual runs
    rng = np.random.default_rng(42)
    for i, vals in enumerate(method_data):
        jitter = rng.uniform(-0.1, 0.1, len(vals))
        ax.scatter(i + 1 + jitter, vals, alpha=0.6, s=30, color='black', zorder=3)

    ax.set_ylabel(label)
    ax.set_title(f'{label} Distribution (ε=0.8)', fontsize=11)
    ax.tick_params(axis='x', rotation=35)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Policy Performance Variance Across Runs (ε=0.8)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT}/9_policy_variance.png', dpi=150, bbox_inches='tight')
plt.close()

# Coefficient of variation
summary_lines.append("\n\nANALYSIS 9: Policy Variance (ε=0.8)")
summary_lines.append("-" * 40)
summary_lines.append(f"  {'Method':20s} {'MO mean':>8s} {'MO std':>8s} {'MO CV':>8s} {'kWh mean':>9s} {'DH mean':>8s}")
sub = pol_df[pol_df['comfort_target'] == ct_default]
for m in policy_methods:
    ms = sub[sub['method'] == m]
    if len(ms) > 0:
        mo_mean = ms['mo_score'].mean()
        mo_std = ms['mo_score'].std()
        mo_cv = mo_std / mo_mean if mo_mean > 0 else 0
        summary_lines.append(f"  {m:20s} {mo_mean:8.3f} {mo_std:8.3f} {mo_cv:8.3f} "
                             f"{ms['kwh'].mean():9.3f} {ms['dh'].mean():8.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 10: Critical Sensor per Method (ablation)
# ═════════════════════════════════════════════════════════════════════════════
print("[10/10] Critical Sensor Analysis...")

fig, ax = plt.subplots(figsize=(14, 7))

abl_methods = abl_df['method'].unique()
sensor_order = ['co2', 'noisedb', 'lightlevel', 'airquality', 'humidity']

# For each method, compute the drop in combined accuracy when each sensor is removed
drop_data = {}
for method in abl_methods:
    mdf = abl_df[abl_df['method'] == method].sort_values('n_sensors', ascending=False)
    drops = {}
    prev_acc = mdf.iloc[0]['combined_accuracy']
    for _, row in mdf.iterrows():
        if row['dropped'] != 'none':
            drop = prev_acc - row['combined_accuracy']
            drops[row['dropped']] = drop
            prev_acc = row['combined_accuracy']
    drop_data[method] = drops

# Heatmap
drop_matrix = np.zeros((len(abl_methods), len(sensor_order)))
for i, method in enumerate(abl_methods):
    for j, sensor in enumerate(sensor_order):
        drop_matrix[i, j] = drop_data.get(method, {}).get(sensor, 0)

im = ax.imshow(drop_matrix, cmap='Reds', aspect='auto')
for i in range(len(abl_methods)):
    for j in range(len(sensor_order)):
        val = drop_matrix[i, j]
        color = 'white' if val > drop_matrix.max() * 0.5 else 'black'
        ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=9, color=color)

ax.set_xticks(range(len(sensor_order)))
ax.set_xticklabels([f'Drop {s}' for s in sensor_order], rotation=30, ha='right')
ax.set_yticks(range(len(abl_methods)))
ax.set_yticklabels(abl_methods, fontsize=9)
ax.set_title('Combined Accuracy Drop When Each Sensor Is Removed (pp)', fontsize=13)
plt.colorbar(im, ax=ax, label='Accuracy Drop (pp)')
plt.tight_layout()
fig.savefig(f'{OUT}/10_ablation_critical_sensor.png', dpi=150, bbox_inches='tight')
plt.close()

summary_lines.append("\n\nANALYSIS 10: Critical Sensor per Method")
summary_lines.append("-" * 40)
summary_lines.append(f"  {'Method':20s} " + " ".join(f'{s:>12s}' for s in sensor_order))
for method in abl_methods:
    drops = drop_data.get(method, {})
    vals = " ".join(f'{drops.get(s, 0):12.1f}' for s in sensor_order)
    summary_lines.append(f"  {method:20s} {vals}")


# ═════════════════════════════════════════════════════════════════════════════
# Save summary
# ═════════════════════════════════════════════════════════════════════════════
summary_text = "\n".join(summary_lines)
with open(f'{OUT}/posthoc_summary.txt', 'w') as f:
    f.write(summary_text)

print(f"\n{'='*70}")
print("ALL ANALYSES COMPLETE")
print(f"{'='*70}")
print(f"Output: {OUT}/")
print(summary_text)
