#!/usr/bin/env python3
"""
VARLiNGAM standalone ablation: test threshold/lag combos across all sims.
Reports edge count, TP, FP, precision, recall, F1 for each combo.
"""
import sys, os, json, warnings
import pandas as pd
import numpy as np
from itertools import product

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

from lingam import VARLiNGAM

# ── Ground truth ─────────────────────────────────────────────────────────────
with open('ground_truth_graphs.json') as f:
    ALL_GT = json.load(f)

# ── Datasets ─────────────────────────────────────────────────────────────────
SIMS = {
    'smart_building_rich': {
        'path': 'data_regen/smart_building_rich_processed.csv',
        'gt_key': 'smart_building_rich',
    },
    'open_window': {
        'path': 'data_regen/challenge1_data_10k_processed.csv',
        'gt_key': 'open_window',
    },
    'smart_room': {
        'path': 'data_regen/smart_room_processed.csv',
        'gt_key': 'smart_room',
    },
    'hidden_vars': {
        'path': 'data_regen/hidden_vars_processed.csv',
        'gt_key': 'hidden_vars',
    },
}

# ── Ablation grid ────────────────────────────────────────────────────────────
LAGS = [1, 2, 3, 5]
THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]


def run_varlingam(data, max_lags, threshold, criterion='bic'):
    """Run VARLiNGAM and return set of (source, target) edges."""
    model = VARLiNGAM(lags=max_lags, criterion=criterion, prune=True)
    model.fit(data.values)

    var_names = [c.lower() for c in data.columns]
    edges = set()
    for lag_idx in range(len(model.adjacency_matrices_)):
        adj = model.adjacency_matrices_[lag_idx]
        for i in range(len(var_names)):
            for j in range(len(var_names)):
                if i == j:
                    continue
                if abs(adj[i, j]) > threshold:
                    edges.add((var_names[i], var_names[j]))
    return edges


def evaluate(edges, gt_edges):
    """Compute SHD, precision, recall, F1."""
    gt_set = {(s.lower(), t.lower()) for s, t in gt_edges}
    tp = len(edges & gt_set)
    fp = len(edges - gt_set)
    fn = len(gt_set - edges)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    shd = fp + fn
    return {'edges': len(edges), 'tp': tp, 'fp': fp, 'fn': fn,
            'shd': shd, 'p': round(p, 3), 'r': round(r, 3), 'f1': round(f1, 3)}


def main():
    results = []

    for sim_name, sim_cfg in SIMS.items():
        print(f"\n{'='*70}")
        print(f"  {sim_name} (GT: {len(ALL_GT[sim_cfg['gt_key']]['edges'])} edges)")
        print(f"{'='*70}")

        data = pd.read_csv(sim_cfg['path'])
        gt_edges = ALL_GT[sim_cfg['gt_key']]['edges']

        print(f"{'Lags':>4}  {'Thresh':>6}  {'Edges':>5}  {'TP':>3}  {'FP':>3}  {'FN':>3}  "
              f"{'SHD':>4}  {'P':>5}  {'R':>5}  {'F1':>5}")
        print("-" * 60)

        for lags, thresh in product(LAGS, THRESHOLDS):
            try:
                edges = run_varlingam(data, lags, thresh)
                m = evaluate(edges, gt_edges)
                results.append({
                    'sim': sim_name, 'lags': lags, 'threshold': thresh, **m
                })
                # Highlight if F1 > 0
                marker = ' ***' if m['f1'] > 0.05 else ''
                print(f"{lags:>4}  {thresh:>6.2f}  {m['edges']:>5}  {m['tp']:>3}  "
                      f"{m['fp']:>3}  {m['fn']:>3}  {m['shd']:>4}  {m['p']:>5.3f}  "
                      f"{m['r']:>5.3f}  {m['f1']:>5.3f}{marker}")
            except Exception as e:
                print(f"{lags:>4}  {thresh:>6.2f}  ERROR: {e}")
                results.append({
                    'sim': sim_name, 'lags': lags, 'threshold': thresh,
                    'edges': -1, 'tp': 0, 'fp': 0, 'fn': len(gt_edges),
                    'shd': len(gt_edges), 'p': 0, 'r': 0, 'f1': 0, 'error': str(e)
                })

    # ── Summary: best combo per sim ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  BEST COMBOS PER SIM (by F1, then SHD)")
    print(f"{'='*70}")
    df = pd.DataFrame(results)
    for sim_name in SIMS:
        sim_df = df[df['sim'] == sim_name].copy()
        if sim_df.empty:
            continue
        sim_df = sim_df.sort_values(['f1', 'shd'], ascending=[False, True])
        top3 = sim_df.head(3)
        print(f"\n  {sim_name}:")
        for _, row in top3.iterrows():
            print(f"    lags={int(row['lags'])}, thresh={row['threshold']:.2f} → "
                  f"edges={int(row['edges'])}, TP={int(row['tp'])}, FP={int(row['fp'])}, "
                  f"SHD={int(row['shd'])}, F1={row['f1']:.3f}")

    # ── Cross-sim: which combo works reasonably everywhere? ──────────────────
    print(f"\n{'='*70}")
    print("  CROSS-SIM RANKING (mean F1 across all sims)")
    print(f"{'='*70}")
    combo_scores = df.groupby(['lags', 'threshold']).agg(
        mean_f1=('f1', 'mean'),
        mean_shd=('shd', 'mean'),
        mean_edges=('edges', 'mean'),
        min_f1=('f1', 'min'),
    ).sort_values(['mean_f1', 'mean_shd'], ascending=[False, True])

    for (lags, thresh), row in combo_scores.head(10).iterrows():
        print(f"  lags={lags}, thresh={thresh:.2f} → "
              f"mean_F1={row['mean_f1']:.3f}, mean_SHD={row['mean_shd']:.1f}, "
              f"mean_edges={row['mean_edges']:.1f}, worst_F1={row['min_f1']:.3f}")

    # Save full results
    df.to_csv('varlingam_ablation_results.csv', index=False)
    print(f"\nFull results saved to varlingam_ablation_results.csv")


if __name__ == '__main__':
    main()
