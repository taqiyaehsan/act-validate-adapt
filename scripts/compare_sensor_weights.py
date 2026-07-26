#!/usr/bin/env python3
"""Compare sensor weighting strategies for 4-way regime monitoring.

All strategies derive weights from the discovered graph + training data,
NOT from domain knowledge or test data.  Fully domain-agnostic.

Strategies:
  A) Uniform (baseline) — all sensors weight=1.0
  B) Graph-derived — weight each sensor by the number of validated edges
     pointing TO it from regime-related parents (non-observable sources)
  C) Discriminability-derived — weight by how differently the 4 regime
     models predict each sensor (learned from training data coefficients)
  D) Combined (B × C) — product of graph and discriminability weights
"""

import sys, os, json, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurips_final_experiments import (
    load_data, get_scaling_map, collect_sim_data,
    build_observable_graph, construct_factored_monitors,
    compute_monitoring_metrics, SENSOR_VARS, OBSERVABLE_VARS, LATENT_VARS,
)


def compute_graph_weights(validated_edges, obs_edges):
    """Strategy B: Weight sensors by in-degree from regime-related parents.

    Sensors that receive validated causal edges from non-observable sources
    (occupancy, windowposition — the regime variables) carry more
    regime-relevant information.  This is derived purely from the
    discovered graph, not from domain knowledge.
    """
    # Regime-related parents: variables that are latent (not observable)
    # or that have high connectivity to latent variables
    regime_parents = LATENT_VARS  # occupancy, windowposition

    weights = {v: 1.0 for v in SENSOR_VARS}
    for source, target in validated_edges:
        if source in regime_parents and target in SENSOR_VARS:
            weights[target] += 2.0  # each edge from regime var adds weight

    # Also count edges from variables that are 1-hop from regime vars
    regime_children = set()
    for source, target in validated_edges:
        if source in regime_parents:
            regime_children.add(target)

    for source, target in obs_edges:
        if source in regime_children and target in SENSOR_VARS:
            weights[target] += 0.5  # indirect regime influence

    return weights


def compute_discriminability_weights(monitors):
    """Strategy C: Weight sensors by regime-model prediction variance.

    For each sensor, compute how differently the 4 regime models predict it.
    Sensors with high prediction variance across regimes are more
    discriminative — they're the sensors where the regime matters most.

    Computed from the trained SEM coefficients, not from test data.
    """
    regime_names = ['base', 'occ', 'win', 'full']
    weights = {}

    for v in SENSOR_VARS:
        # Collect the intercept + weighted parent contributions for each regime
        predictions_at_midpoint = []
        midpoint = {s: 0.5 for s in OBSERVABLE_VARS}  # normalized midpoint

        for rname in regime_names:
            pred = monitors[rname].predict(midpoint)
            predictions_at_midpoint.append(pred.get(v, 0.5))

        # Variance of predictions across regimes = discriminability
        var = np.var(predictions_at_midpoint)
        weights[v] = var

    # Normalize: min weight = 1.0, scale by relative discriminability
    if max(weights.values()) > 0:
        max_w = max(weights.values())
        weights = {v: 1.0 + 2.0 * (w / max_w) for v, w in weights.items()}
    else:
        weights = {v: 1.0 for v in SENSOR_VARS}

    return weights


def run_weighted_monitoring(monitors, sim_records, sensor_weights):
    """Run 4-way monitoring with custom sensor weights."""
    sigma = 1.5
    forget = 0.15
    regime_names = ['base', 'occ', 'win', 'full']
    weights = {r: 0.25 for r in regime_names}
    records = []

    for rec in sim_records:
        obs = rec['state_obs']
        likelihoods = {}
        for rname in regime_names:
            pred = monitors[rname].predict(obs)
            err = sum(sensor_weights.get(v, 1.0) * abs(obs.get(v, 0.5) - pred.get(v, 0.5))
                      for v in SENSOR_VARS)
            likelihoods[rname] = np.exp(-sigma * err)

        unnorm = {r: likelihoods[r] * weights[r] for r in regime_names}
        total = sum(unnorm.values())
        if total > 0:
            raw = {r: unnorm[r] / total for r in regime_names}
            weights = {r: (1 - forget) * raw[r] + forget * 0.25
                       for r in regime_names}
        w_total = sum(weights.values())
        if w_total > 0:
            weights = {r: weights[r] / w_total for r in regime_names}

        winner = max(weights, key=weights.get)
        p_occ = weights['occ'] + weights['full']
        p_win = weights['win'] + weights['full']
        records.append({
            'step': rec['step'], 'hour': rec['hour'],
            'occ_active': rec['occ_active'], 'win_active': rec['win_active'],
            'regime_gt': rec['regime_gt'],
            'P_occ': p_occ, 'P_win': p_win, 'winner': winner,
        })
    return records


def main():
    print("Loading data...")
    data, scaling, gt_set = load_data()
    smap = get_scaling_map(scaling)

    with open('results/neurips_final_good_result_3seeds/discovered_edges.json') as f:
        d = json.load(f)
    validated = set((e[0], e[1]) for e in d['consensus_validated'])
    obs_only = set((e[0], e[1]) for e in d['consensus_obs_only'])
    obs_edges = build_observable_graph(validated)

    # Build monitors (same for all weight strategies)
    monitors = construct_factored_monitors(validated, data)

    # Compute weights
    uniform = {v: 1.0 for v in SENSOR_VARS}
    graph_wt = compute_graph_weights(validated, obs_edges)
    disc_wt = compute_discriminability_weights(monitors)
    # Combined: product of graph and discriminability, re-normalized
    combo_raw = {v: graph_wt[v] * disc_wt[v] for v in SENSOR_VARS}
    combo_max = max(combo_raw.values())
    combo_wt = {v: 1.0 + 2.0 * (combo_raw[v] / combo_max) if combo_max > 0
                else 1.0 for v in SENSOR_VARS}

    print("\nSensor weights:")
    print(f"  {'Sensor':<15} {'Uniform':>8} {'Graph':>8} {'Discrim':>8} {'Combo':>8}")
    for v in SENSOR_VARS:
        print(f"  {v:<15} {uniform[v]:>8.2f} {graph_wt[v]:>8.2f} "
              f"{disc_wt[v]:>8.2f} {combo_wt[v]:>8.2f}")

    # Test across 3 seeds
    strategies = {
        'A_uniform': uniform,
        'B_graph': graph_wt,
        'C_discrim': disc_wt,
        'D_combo': combo_wt,
    }

    all_results = {s: [] for s in strategies}

    for seed_i in range(3):
        sim_records = collect_sim_data(smap, duration_steps=5760)
        print(f"\nSeed {seed_i}:")

        for sname, sweights in strategies.items():
            recs = run_weighted_monitoring(monitors, sim_records, sweights)
            m = compute_monitoring_metrics(recs)
            all_results[sname].append(m)

            df = pd.DataFrame(recs)
            per_r = {}
            for rid, rname in {0: 'base', 1: 'occ', 2: 'win', 3: 'full'}.items():
                mask = df['regime_gt'] == rid
                if mask.sum() > 0:
                    per_r[rname] = f"{(df.loc[mask, 'winner'] == rname).mean() * 100:.1f}"
            print(f"  {sname:<15} Comb={m['combined_accuracy']}%  "
                  f"base={per_r.get('base')} occ={per_r.get('occ')} "
                  f"win={per_r.get('win')} full={per_r.get('full')}")

    # Summary
    print("\n" + "=" * 75)
    print(f"{'Strategy':<15} {'Combined':>9} {'Base':>7} {'Occ':>7} {'Win':>7} {'Full':>7}")
    print("-" * 75)
    for sname in strategies:
        seeds = all_results[sname]
        comb = np.mean([s['combined_accuracy'] for s in seeds])
        std = np.std([s['combined_accuracy'] for s in seeds])
        print(f"{sname:<15} {comb:>6.1f}±{std:.1f}")
    print("=" * 75)


if __name__ == '__main__':
    main()
