#!/usr/bin/env python3
"""Compare 4 monitor architectures for regime detection.

Architectures:
  A) Original factored: 2 binary monitors (occ, win), each trained on
     all data where that dimension is on/off.  Assumes independence.
  B) Decorrelated factored: same 2 binary monitors, but training data
     for each dimension is balanced w.r.t. the OTHER dimension.
  C) 4-way direct: one model per regime combination (base, occ, win, full).
     At inference, pick the regime whose model predicts best.
  D) Conditional factored: occ monitor is same as (A), but win monitor
     is split by detected occ state:  model_win_on|occ, model_win_off|occ,
     model_win_on|no_occ, model_win_off|no_occ.

All architectures use the same edge set, training data, test data, and
Bayesian update parameters (sigma=2.0, forget_factor=0.12).
"""

import sys, os, json, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format='%(message)s')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurips_final_experiments import (
    load_data, get_scaling_map, collect_sim_data,
    build_observable_graph, build_predictive_model,
    SENSOR_VARS, OBSERVABLE_VARS, LATENT_VARS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

SIGMA = 2.0
FORGET = 0.12


def run_factored(monitors, sim_records, sensor_vars=SENSOR_VARS):
    """Architecture A/B/D: factored binary monitors → 4-way via independence."""
    p_occ, p_win = 0.5, 0.5
    records = []

    for rec in sim_records:
        obs = rec['state_obs']

        for dim in ('occ', 'win'):
            p = p_occ if dim == 'occ' else p_win

            # Pick the right model pair — for conditional (D), depends on occ
            if dim == 'win' and 'win_cond' in monitors:
                occ_on = p_occ >= 0.5
                key = 'win_occ_on' if occ_on else 'win_occ_off'
                m_off = monitors[key]['off']
                m_on = monitors[key]['on']
            else:
                m_off = monitors[dim]['off']
                m_on = monitors[dim]['on']

            pred_off = m_off.predict(obs)
            pred_on = m_on.predict(obs)
            err_off = sum(abs(obs.get(v, 0.5) - pred_off.get(v, 0.5))
                          for v in sensor_vars)
            err_on = sum(abs(obs.get(v, 0.5) - pred_on.get(v, 0.5))
                         for v in sensor_vars)
            lik_off = np.exp(-SIGMA * err_off)
            lik_on = np.exp(-SIGMA * err_on)
            unnorm_off = lik_off * (1 - p)
            unnorm_on = lik_on * p
            total = unnorm_off + unnorm_on
            if total > 0:
                p_raw = unnorm_on / total
                p = (1 - FORGET) * p_raw + FORGET * 0.5

            if dim == 'occ':
                p_occ = p
            else:
                p_win = p

        p_base = (1 - p_occ) * (1 - p_win)
        p_occ_only = p_occ * (1 - p_win)
        p_win_only = (1 - p_occ) * p_win
        p_both = p_occ * p_win
        combined = {'base': p_base, 'occ': p_occ_only,
                    'win': p_win_only, 'full': p_both}
        winner = max(combined, key=combined.get)

        records.append({
            'step': rec['step'], 'hour': rec['hour'],
            'occ_active': rec['occ_active'], 'win_active': rec['win_active'],
            'regime_gt': rec['regime_gt'],
            'P_occ': p_occ, 'P_win': p_win, 'winner': winner,
        })
    return records


def run_4way(models_4way, sim_records, sensor_vars=SENSOR_VARS):
    """Architecture C: 4 regime-specific models, direct competition."""
    regime_names = ['base', 'occ', 'win', 'full']
    weights = {r: 0.25 for r in regime_names}
    records = []

    for rec in sim_records:
        obs = rec['state_obs']

        likelihoods = {}
        for rname in regime_names:
            model = models_4way[rname]
            pred = model.predict(obs)
            err = sum(abs(obs.get(v, 0.5) - pred.get(v, 0.5))
                      for v in sensor_vars)
            likelihoods[rname] = np.exp(-SIGMA * err)

        # Bayesian update
        unnorm = {r: likelihoods[r] * weights[r] for r in regime_names}
        total = sum(unnorm.values())
        if total > 0:
            raw = {r: unnorm[r] / total for r in regime_names}
            # Forget factor: leak toward uniform
            weights = {r: (1 - FORGET) * raw[r] + FORGET * 0.25
                       for r in regime_names}
        # Re-normalize weights
        w_total = sum(weights.values())
        if w_total > 0:
            weights = {r: weights[r] / w_total for r in regime_names}

        winner = max(weights, key=weights.get)

        # Reconstruct P_occ and P_win for metrics compatibility
        p_occ = weights['occ'] + weights['full']
        p_win = weights['win'] + weights['full']

        records.append({
            'step': rec['step'], 'hour': rec['hour'],
            'occ_active': rec['occ_active'], 'win_active': rec['win_active'],
            'regime_gt': rec['regime_gt'],
            'P_occ': p_occ, 'P_win': p_win, 'winner': winner,
        })
    return records


def compute_metrics(records):
    """Compute per-regime + overall accuracy."""
    if not records:
        return {}
    df = pd.DataFrame(records)
    regime_map = {0: 'base', 1: 'occ', 2: 'win', 3: 'full'}
    gt_labels = df['regime_gt'].map(regime_map)
    combined = (df['winner'] == gt_labels).mean() * 100

    occ_correct = ((df['P_occ'] >= 0.5) == df['occ_active'].astype(bool)).mean() * 100
    win_correct = ((df['P_win'] >= 0.5) == df['win_active'].astype(bool)).mean() * 100

    per_regime = {}
    for rid, rname in sorted(regime_map.items()):
        mask = df['regime_gt'] == rid
        if mask.sum() > 0:
            per_regime[rname] = round(
                (df.loc[mask, 'winner'] == rname).mean() * 100, 1)
        else:
            per_regime[rname] = None

    return {
        'occ_accuracy': round(occ_correct, 1),
        'win_accuracy': round(win_correct, 1),
        'combined': round(combined, 1),
        **{f'regime_{k}': v for k, v in per_regime.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD MONITORS
# ═══════════════════════════════════════════════════════════════════════════════

def build_arch_A(obs_edges, data, occ_vals, win_vals):
    """A) Original factored — train on all on/off rows per dimension."""
    return {
        'occ': {
            'off': build_predictive_model(obs_edges, data, occ_vals < 0.1),
            'on':  build_predictive_model(obs_edges, data, occ_vals >= 0.1),
        },
        'win': {
            'off': build_predictive_model(obs_edges, data, win_vals < 0.15),
            'on':  build_predictive_model(obs_edges, data, win_vals >= 0.15),
        },
    }


def _balanced_mask(primary_mask, other_vals, other_threshold, rng):
    """Balance primary_mask w.r.t. the other dimension.

    E.g., for occ_on mask, ensure equal representation of win_on and win_off
    rows.  Downsamples the majority sub-group to match the minority.
    """
    other_on = other_vals >= other_threshold
    other_off = ~other_on

    idx_both_on = np.where(primary_mask & other_on)[0]
    idx_primary_only = np.where(primary_mask & other_off)[0]

    n_min = min(len(idx_both_on), len(idx_primary_only))
    if n_min == 0:
        # Can't balance — fall back to unbalanced
        return primary_mask

    # Downsample majority to match minority
    if len(idx_both_on) > n_min:
        idx_both_on = rng.choice(idx_both_on, n_min, replace=False)
    else:
        idx_primary_only = rng.choice(idx_primary_only, n_min, replace=False)

    balanced = np.zeros(len(primary_mask), dtype=bool)
    balanced[idx_both_on] = True
    balanced[idx_primary_only] = True
    return balanced


def build_arch_B(obs_edges, data, occ_vals, win_vals, seed=42):
    """B) Decorrelated factored — balance each dimension w.r.t. the other."""
    rng = np.random.RandomState(seed)

    occ_on_bal = _balanced_mask(occ_vals >= 0.1, win_vals, 0.15, rng)
    occ_off_bal = _balanced_mask(occ_vals < 0.1, win_vals, 0.15, rng)
    win_on_bal = _balanced_mask(win_vals >= 0.15, occ_vals, 0.1, rng)
    win_off_bal = _balanced_mask(win_vals < 0.15, occ_vals, 0.1, rng)

    return {
        'occ': {
            'off': build_predictive_model(obs_edges, data, occ_off_bal),
            'on':  build_predictive_model(obs_edges, data, occ_on_bal),
        },
        'win': {
            'off': build_predictive_model(obs_edges, data, win_off_bal),
            'on':  build_predictive_model(obs_edges, data, win_on_bal),
        },
    }


def build_arch_C(obs_edges, data, occ_vals, win_vals):
    """C) 4-way direct — one model per regime combination."""
    occ_on = occ_vals >= 0.1
    win_on = win_vals >= 0.15

    return {
        'base': build_predictive_model(obs_edges, data, ~occ_on & ~win_on),
        'occ':  build_predictive_model(obs_edges, data,  occ_on & ~win_on),
        'win':  build_predictive_model(obs_edges, data, ~occ_on &  win_on),
        'full': build_predictive_model(obs_edges, data,  occ_on &  win_on),
    }


def build_arch_D(obs_edges, data, occ_vals, win_vals):
    """D) Conditional factored — occ monitor standard, win monitor conditioned
    on detected occ state."""
    occ_on = occ_vals >= 0.1
    win_on = win_vals >= 0.15

    return {
        'occ': {
            'off': build_predictive_model(obs_edges, data, ~occ_on),
            'on':  build_predictive_model(obs_edges, data,  occ_on),
        },
        # Standard win monitor (fallback)
        'win': {
            'off': build_predictive_model(obs_edges, data, ~win_on),
            'on':  build_predictive_model(obs_edges, data,  win_on),
        },
        # Conditional win monitors
        'win_cond': True,  # flag for run_factored
        'win_occ_on': {
            'off': build_predictive_model(obs_edges, data, occ_on & ~win_on),
            'on':  build_predictive_model(obs_edges, data, occ_on &  win_on),
        },
        'win_occ_off': {
            'off': build_predictive_model(obs_edges, data, ~occ_on & ~win_on),
            'on':  build_predictive_model(obs_edges, data, ~occ_on &  win_on),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    data, scaling, gt_set = load_data()
    smap = get_scaling_map(scaling)

    with open('results/neurips_final/discovered_edges.json') as f:
        d = json.load(f)
    grid_edges = [(e[0], e[1]) for e in d['consensus_validated']]
    obs_edges = build_observable_graph(set(grid_edges))
    print(f"Edges: {len(grid_edges)} total, {len(obs_edges)} observable")

    cols_lower = {c.lower(): c for c in data.columns}
    occ_vals = data[cols_lower['occupancy']].values
    win_vals = data[cols_lower['windowposition']].values

    # Training data regime distribution
    occ_on = occ_vals >= 0.1
    win_on = win_vals >= 0.15
    print(f"\nTraining data regime distribution ({len(data)} rows):")
    for name, mask in [('base', ~occ_on & ~win_on), ('occ', occ_on & ~win_on),
                       ('win', ~occ_on & win_on), ('full', occ_on & win_on)]:
        print(f"  {name}: {mask.sum()} ({mask.sum()/len(data)*100:.1f}%)")

    # Build all architectures
    print("\nBuilding monitors...")
    archs = {}
    for label, builder in [('A_original', lambda: build_arch_A(obs_edges, data, occ_vals, win_vals)),
                           ('B_decorrelated', lambda: build_arch_B(obs_edges, data, occ_vals, win_vals)),
                           ('C_4way', lambda: build_arch_C(obs_edges, data, occ_vals, win_vals)),
                           ('D_conditional', lambda: build_arch_D(obs_edges, data, occ_vals, win_vals))]:
        try:
            archs[label] = builder()
            print(f"  {label}: OK")
        except Exception as e:
            print(f"  {label}: FAILED ({e})")

    # Collect test data (3 seeds for robustness)
    N_SEEDS = 3
    all_results = {a: [] for a in archs}

    for seed_i in range(N_SEEDS):
        print(f"\n--- Test seed {seed_i + 1}/{N_SEEDS} ---")
        sim_records = collect_sim_data(smap, duration_steps=5760)

        from collections import Counter
        regime_counts = Counter(r['regime_gt'] for r in sim_records)
        regime_names = {0: 'base', 1: 'occ', 2: 'win', 3: 'full'}
        dist_str = ', '.join(f"{regime_names[k]}={regime_counts.get(k, 0)}"
                             for k in sorted(regime_names))
        print(f"  Test regimes: {dist_str}")

        for arch_name, monitors in archs.items():
            if arch_name == 'C_4way':
                recs = run_4way(monitors, sim_records)
            else:
                recs = run_factored(monitors, sim_records)

            metrics = compute_metrics(recs)
            all_results[arch_name].append(metrics)
            print(f"  {arch_name}: Combined={metrics['combined']}%  "
                  f"base={metrics.get('regime_base', '?')}  "
                  f"occ={metrics.get('regime_occ', '?')}  "
                  f"win={metrics.get('regime_win', '?')}  "
                  f"full={metrics.get('regime_full', '?')}")

    # Summary table
    print("\n" + "=" * 85)
    print(f"{'Architecture':<20} {'Combined':>9} {'Base':>7} {'Occ':>7} {'Win':>7} {'Full':>7}  {'P(occ)':>7} {'P(win)':>7}")
    print("-" * 85)

    for arch_name in archs:
        seeds = all_results[arch_name]
        comb = np.mean([s['combined'] for s in seeds])
        comb_std = np.std([s['combined'] for s in seeds])
        base = np.mean([s.get('regime_base', 0) or 0 for s in seeds])
        occ = np.mean([s.get('regime_occ', 0) or 0 for s in seeds])
        win = np.mean([s.get('regime_win', 0) or 0 for s in seeds])
        full = np.mean([s.get('regime_full', 0) or 0 for s in seeds])
        p_occ = np.mean([s['occ_accuracy'] for s in seeds])
        p_win = np.mean([s['win_accuracy'] for s in seeds])

        print(f"{arch_name:<20} {comb:>6.1f}±{comb_std:.1f} {base:>6.1f}  "
              f"{occ:>6.1f}  {win:>6.1f}  {full:>6.1f}  {p_occ:>6.1f}  {p_win:>6.1f}")

    print("=" * 85)


if __name__ == '__main__':
    main()
