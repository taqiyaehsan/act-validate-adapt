#!/usr/bin/env python3
"""
Experiment A: Sensor Ablation Study
====================================

Start from a full 15-variable smart building (S5) and systematically
ablate sensors to measure degradation in causal discovery quality and
policy performance.

Produces:
  - neurips_results/sensor_ablation_raw.csv
  - neurips_results/sensor_ablation_summary.csv
  - neurips_results/sensor_ablation_benchmarks.csv

Usage:
  python run_sensor_ablation.py --seeds 5 --episodes 5
  python run_sensor_ablation.py --tier S4 --seeds 3
  python run_sensor_ablation.py --skip-benchmarks --tier S5
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import traceback
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

# ── project imports ──────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

from src.pipeline_cwm import create_cwm_pipeline
from benchmarks_new.utils import (
    calculate_shd,
    calculate_precision_recall_f1,
    normalize_edge,
)

logger = logging.getLogger(__name__)

# ── API key (same as run_exp_suite_B_prelim.py) ─────────────────────
API_KEY = 'YOUR_OPENAI_API_KEY'

# ── Paths ────────────────────────────────────────────────────────────
SIM_PATH = os.path.join(_BASE_DIR, 'js', 'smart_building_rich.js')
OBS_DATA_PATH = os.path.join(_BASE_DIR, 'data_regen',
                             'smart_building_rich_processed.csv')
SCALING_PATH = os.path.join(_BASE_DIR, 'data_regen',
                            'smart_building_rich_processed_scaling.csv')
GT_PATH = os.path.join(_BASE_DIR, 'ground_truth_graphs.json')

# ── All 15 variables (ProperCase) ────────────────────────────────────
ALL_VARS = [
    'OutdoorTemp', 'SolarRadiation', 'Occupancy', 'WindowPosition',
    'Temperature', 'Humidity', 'CO2', 'LightLevel', 'NoiseDB',
    'AirQuality', 'PMV', 'HVACPower', 'LightingPower',
    'EnergyConsumption', 'OverallSatisfaction',
]

# ── Ground truth (28 edges, lowercase) ───────────────────────────────
def load_ground_truth():
    """Load full ground truth edges from JSON."""
    with open(GT_PATH) as f:
        gt = json.load(f)
    return {(s, t) for s, t in gt['smart_building_rich']['edges']}

FULL_GT = None  # lazily loaded


def get_full_gt():
    global FULL_GT
    if FULL_GT is None:
        FULL_GT = load_ground_truth()
    return FULL_GT


# ── Sensor tier definitions ──────────────────────────────────────────
# Each tier maps to a set of HIDDEN variables.
# The remaining visible variables are used for discovery.
SENSOR_TIERS = {
    'S5': {
        'name': 'Full smart building',
        'hidden': set(),
        'analogy': 'Cameras + mics + all sensors',
    },
    'S4': {
        'name': 'No video',
        'hidden': {'Occupancy'},
        'analogy': 'No camera/badge system',
    },
    'S3': {
        'name': 'No video + no audio',
        'hidden': {'Occupancy', 'NoiseDB'},
        'analogy': 'No camera, no microphone',
    },
    'S2': {
        'name': 'No rich sensors',
        'hidden': {'Occupancy', 'NoiseDB', 'CO2', 'LightLevel', 'LightingPower'},
        'analogy': 'Basic HVAC + weather',
    },
    'S1': {
        'name': 'Basic HVAC',
        'hidden': {'Occupancy', 'NoiseDB', 'CO2', 'LightLevel', 'LightingPower',
                   'OutdoorTemp', 'SolarRadiation', 'WindowPosition'},
        'analogy': 'Smart thermostat package',
    },
    'S0': {
        'name': 'Thermostat only',
        'hidden': {'Occupancy', 'NoiseDB', 'CO2', 'LightLevel', 'LightingPower',
                   'OutdoorTemp', 'SolarRadiation', 'WindowPosition',
                   'AirQuality', 'PMV', 'Humidity'},
        'analogy': 'Dumb thermostat',
    },
}


def get_visible_vars(tier):
    """Return list of visible (non-hidden) variables for a tier."""
    hidden = SENSOR_TIERS[tier]['hidden']
    return [v for v in ALL_VARS if v not in hidden]


def _norm_var(name):
    """Normalize variable name to match GT JSON convention.

    GT JSON uses 'satisfaction'; column schema uses 'OverallSatisfaction'.
    """
    n = name.lower()
    if n in ('overallsatisfaction', 'overall_satisfaction'):
        return 'satisfaction'
    return n


def project_ground_truth(tier):
    """Project ground truth to only edges between visible variables.

    Returns set of (source, target) tuples in lowercase.
    """
    visible = {_norm_var(v) for v in get_visible_vars(tier)}
    return {(s, t) for s, t in get_full_gt() if s in visible and t in visible}


# ── Extended graph fidelity metrics ──────────────────────────────────
def compute_graph_metrics(predicted_edges, true_edges, n_vars):
    """Compute full suite of graph fidelity metrics.

    Args:
        predicted_edges: set of (src, tgt) lowercase tuples
        true_edges: set of (src, tgt) lowercase tuples
        n_vars: number of visible variables

    Returns:
        dict of metric_name -> value
    """
    # Normalize variable names: overallsatisfaction → satisfaction to match GT
    def _norm(name):
        n = str(name).strip().lower()
        if n in ('overallsatisfaction', 'overall_satisfaction'):
            return 'satisfaction'
        return n
    pred = {(_norm(s), _norm(t)) for s, t in predicted_edges}
    true = {(_norm(s), _norm(t)) for s, t in true_edges}

    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    n_possible = n_vars * (n_vars - 1)  # directed edges
    tn = n_possible - len(true) - fp  # approximate

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    accuracy = (tp + max(tn, 0)) / max(n_possible, 1)

    shd = calculate_shd(pred, true)
    nshd = shd / max(n_possible, 1)

    # Reversed edges: (A→B) discovered when GT has (B→A)
    reversed_count = 0
    for s, t in pred:
        if (t, s) in true and (s, t) not in true:
            reversed_count += 1

    # Cost: α×(1−precision) + β×(1−recall) with equal weights
    alpha, beta = 0.5, 0.5
    cost = alpha * (1.0 - precision) + beta * (1.0 - recall)

    # Risk: confidence-weighted cost (uniform confidence for baselines)
    risk = cost  # baselines have no confidence scores

    return {
        'shd': shd,
        'nshd': round(nshd, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'accuracy': round(accuracy, 4),
        'cost': round(cost, 4),
        'risk': round(risk, 4),
        'reversed_edges': reversed_count,
        'n_edges_discovered': len(pred),
        'n_gt_edges': len(true),
        'n_vars': n_vars,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def compute_edge_stability(edge_sets):
    """Compute average Jaccard similarity across all pairs of edge sets.

    Args:
        edge_sets: list of sets of edges (one per seed)

    Returns:
        float: average pairwise Jaccard, or NaN if < 2 seeds
    """
    if len(edge_sets) < 2:
        return float('nan')
    jaccards = []
    for a, b in combinations(range(len(edge_sets)), 2):
        union = edge_sets[a] | edge_sets[b]
        if not union:
            jaccards.append(1.0)  # both empty → identical
        else:
            jaccards.append(len(edge_sets[a] & edge_sets[b]) / len(union))
    return round(np.mean(jaccards), 4)


# ── Benchmark baselines ─────────────────────────────────────────────
BENCHMARK_METHODS = {
    'PC': ('benchmarks_new.pc_baseline', 'run_pc_baseline', {}),
    # 'SAM': ('benchmarks_new.sam_baseline', 'run_sam_baseline', {}),  # DISABLED: too slow
    'GIES': ('benchmarks_new.gies_comparison', 'run_gies', {}),
    'JCI': ('benchmarks_new.jci_comparison', 'run_jci', {}),
    'ABCD': ('benchmarks_new.abcd_comparison', 'run_abcd', {}),
    'NOTEARS-I': ('benchmarks_new.notears_i', 'run_notears_i', {}),
    'IID': ('benchmarks_new.iid_comparison', 'run_iid', {}),
    'ICP': ('benchmarks_new.icp_comparison', 'run_icp', {}),
    'CausalBandits': ('benchmarks_new.causal_bandits', 'run_causal_bandits', {}),
}


def run_benchmark(method_name, data_df):
    """Run a single benchmark method, return edge list.

    Returns list of (source, target) tuples (lowercase).
    """
    mod_path, func_name, kwargs = BENCHMARK_METHODS[method_name]
    try:
        import importlib
        mod = importlib.import_module(mod_path)
        func = getattr(mod, func_name)
        edges = func(data_df, **kwargs)
        # Normalise to lowercase tuples
        return [(str(s).lower(), str(t).lower()) for s, t in edges]
    except Exception as e:
        logger.error(f"Benchmark {method_name} failed: {e}")
        traceback.print_exc()
        return []


# ── PolicyGRID discovery ─────────────────────────────────────────────
def run_policygrid_discovery(data_df, visible_cols, seed):
    """Run full PolicyGRID pipeline on tier-masked data.

    Returns dict with edges, timing, intervention count.
    """
    random.seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    # actuator_vars: user-specified domain knowledge — which variables
    # can be directly controlled via the simulator.  In a real building,
    # this is the HVAC and lighting control interface.
    actuator_vars = {'hvacpower', 'lightingpower'}
    non_intervenable_vars = {
        'outdoortemp', 'solarradiation', 'occupancy', 'windowposition',
        'pmv', 'energyconsumption', 'satisfaction', 'overallsatisfaction',
    }
    # Only include actuators/non-intervenable that are visible in this tier
    tier_actuators = actuator_vars & {c.lower() for c in visible_cols}
    tier_non_intervenable = non_intervenable_vars & {c.lower() for c in visible_cols}

    pipeline = create_cwm_pipeline(
        csv_data=data_df[visible_cols],
        api_key=API_KEY,
        smart_room_path=SIM_PATH,
        dataset_type='smart_building_rich',
        max_iterations=3,
        alpha=0.5,
        beta=0.5,
        effect_threshold=0.1,
        relevant_column=visible_cols,
        actuator_vars=tier_actuators,
        non_intervenable_vars=tier_non_intervenable,
    )

    final_dag, final_metrics = pipeline.run_with_cwm()
    disc_time = time.time() - t0

    # Extract validated edges
    validated = set(pipeline.pipeline.validated_edges)

    # Extract method DAG union (obs-only)
    method_dags = pipeline.pipeline.get_method_dags()
    obs_only = set()
    for G in method_dags.values():
        obs_only.update(G.edges())

    # Count interventions
    intervention_count = sum(
        len(r) for r in pipeline.pipeline.tester.intervention_results.values()
    )

    # Normalise to lowercase
    validated_lc = {(str(s).lower(), str(t).lower()) for s, t in validated}
    obs_only_lc = {(str(s).lower(), str(t).lower()) for s, t in obs_only}

    return {
        'validated_edges': validated_lc,
        'obs_only_edges': obs_only_lc,
        'intervention_count': intervention_count,
        'discovery_time_s': disc_time,
        'pipeline': pipeline,
    }


# ── Main ablation loop ───────────────────────────────────────────────
def run_ablation(tiers, n_seeds, n_episodes, skip_benchmarks, include_pets):
    """Run the full sensor ablation experiment.

    For each tier × seed:
      1. Mask data to visible columns
      2. Run PolicyGRID discovery
      3. Run benchmark baselines (unless --skip-benchmarks)
      4. Compute graph fidelity metrics
      5. (Optional) Run policy evaluation

    Returns (raw_df, summary_df, benchmarks_df)
    """
    # Load full data
    full_data = pd.read_csv(OBS_DATA_PATH)
    logger.info(f"Loaded {len(full_data)} rows × {len(full_data.columns)} cols")

    all_rows = []
    all_benchmark_rows = []

    # Track per-method edge sets for stability computation
    # key: (tier, method) -> list of edge sets
    edge_sets_by_tier_method = {}

    for tier in tiers:
        tier_info = SENSOR_TIERS[tier]
        visible = get_visible_vars(tier)
        gt_proj = project_ground_truth(tier)
        n_vars = len(visible)
        n_gt = len(gt_proj)

        print(f"\n{'='*60}")
        print(f"  Tier {tier}: {tier_info['name']}  "
              f"({n_vars} vars, {n_gt} GT edges)")
        print(f"  Hidden: {tier_info['hidden'] or 'none'}")
        print(f"  Analogy: {tier_info['analogy']}")
        print(f"{'='*60}")

        # Mask data to visible columns (match case-insensitively)
        col_map = {}
        for c in full_data.columns:
            for v in visible:
                if c.lower() == v.lower():
                    col_map[c] = v
                    break
        available_cols = [c for c in full_data.columns if c in col_map]
        masked_df = full_data[available_cols].copy()
        masked_df.columns = [col_map[c] for c in available_cols]

        if masked_df.empty or len(masked_df.columns) < 2:
            logger.warning(f"Tier {tier}: insufficient visible columns, skipping")
            continue

        for seed in range(n_seeds):
            print(f"\n  --- Tier {tier}, Seed {seed+1}/{n_seeds} ---")

            # ── PolicyGRID ───────────────────────────────────────
            print(f"    Running PolicyGRID discovery...")
            try:
                pgrid = run_policygrid_discovery(
                    masked_df, list(masked_df.columns), seed)

                # Use validated edges as the primary PolicyGRID output
                pgrid_edges = pgrid['validated_edges']
                metrics = compute_graph_metrics(pgrid_edges, gt_proj, n_vars)
                metrics.update({
                    'tier': tier,
                    'seed': seed,
                    'method': 'PolicyGRID',
                    'intervention_count': pgrid['intervention_count'],
                    'discovery_time_s': round(pgrid['discovery_time_s'], 1),
                })
                all_rows.append(metrics)

                # Also record obs-only union for comparison
                obs_metrics = compute_graph_metrics(
                    pgrid['obs_only_edges'], gt_proj, n_vars)
                obs_metrics.update({
                    'tier': tier,
                    'seed': seed,
                    'method': 'PolicyGRID_ObsOnly',
                    'intervention_count': 0,
                    'discovery_time_s': round(pgrid['discovery_time_s'], 1),
                })
                all_rows.append(obs_metrics)

                # Track edge sets for stability
                key_pg = (tier, 'PolicyGRID')
                edge_sets_by_tier_method.setdefault(key_pg, []).append(
                    pgrid_edges)
                key_obs = (tier, 'PolicyGRID_ObsOnly')
                edge_sets_by_tier_method.setdefault(key_obs, []).append(
                    pgrid['obs_only_edges'])

                print(f"    PolicyGRID: SHD={metrics['shd']}, "
                      f"F1={metrics['f1']}, P={metrics['precision']}, "
                      f"R={metrics['recall']}, "
                      f"interventions={pgrid['intervention_count']}")

            except Exception as e:
                logger.error(f"PolicyGRID failed tier={tier} seed={seed}: {e}")
                traceback.print_exc()

            # ── Benchmark baselines ──────────────────────────────
            if not skip_benchmarks:
                for bm_name in BENCHMARK_METHODS:
                    print(f"    Running {bm_name}...")
                    try:
                        t_bm = time.time()
                        bm_edges = run_benchmark(bm_name, masked_df)
                        bm_time = time.time() - t_bm
                        bm_edge_set = set(bm_edges)

                        bm_metrics = compute_graph_metrics(
                            bm_edge_set, gt_proj, n_vars)
                        bm_metrics.update({
                            'tier': tier,
                            'seed': seed,
                            'method': bm_name,
                            'intervention_count': 0,
                            'discovery_time_s': round(bm_time, 1),
                        })
                        all_benchmark_rows.append(bm_metrics)

                        # Track for stability
                        key_bm = (tier, bm_name)
                        edge_sets_by_tier_method.setdefault(
                            key_bm, []).append(bm_edge_set)

                        print(f"    {bm_name}: SHD={bm_metrics['shd']}, "
                              f"F1={bm_metrics['f1']}, "
                              f"edges={len(bm_edges)}")

                    except Exception as e:
                        logger.error(
                            f"Benchmark {bm_name} failed tier={tier} "
                            f"seed={seed}: {e}")
                        traceback.print_exc()

    # ── Compute edge stability per (tier, method) ────────────────
    def add_stability(rows):
        for row in rows:
            key = (row['tier'], row['method'])
            edge_sets = edge_sets_by_tier_method.get(key, [])
            row['edge_stability'] = compute_edge_stability(edge_sets)

    add_stability(all_rows)
    add_stability(all_benchmark_rows)

    # ── Build DataFrames ─────────────────────────────────────────
    raw_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    bm_df = pd.DataFrame(all_benchmark_rows) if all_benchmark_rows else pd.DataFrame()

    # Combine for summary
    combined = pd.concat([raw_df, bm_df], ignore_index=True) if not bm_df.empty else raw_df

    # Summary: mean ± std over seeds per (tier, method)
    summary_df = pd.DataFrame()
    if not combined.empty:
        metric_cols = ['shd', 'nshd', 'precision', 'recall', 'f1',
                       'accuracy', 'cost', 'risk', 'reversed_edges',
                       'n_edges_discovered', 'edge_stability',
                       'intervention_count', 'discovery_time_s']
        available_metric_cols = [c for c in metric_cols if c in combined.columns]

        grouped = combined.groupby(['tier', 'method'])
        means = grouped[available_metric_cols].mean().round(4)
        stds = grouped[available_metric_cols].std().round(4)

        # Merge mean and std
        summary_parts = []
        for col in available_metric_cols:
            means_col = means[col].rename(f'{col}_mean')
            stds_col = stds[col].rename(f'{col}_std')
            summary_parts.extend([means_col, stds_col])

        summary_df = pd.concat(summary_parts, axis=1).reset_index()

        # Add metadata
        for tier in summary_df['tier'].unique():
            mask = summary_df['tier'] == tier
            summary_df.loc[mask, 'n_vars'] = len(get_visible_vars(tier))
            summary_df.loc[mask, 'n_gt_edges'] = len(project_ground_truth(tier))

    return raw_df, summary_df, bm_df


# ── Ablation summary printing ────────────────────────────────────────
def print_ablation_summary(summary_df):
    """Print a readable ablation summary table."""
    if summary_df.empty:
        print("\n  No results to summarise.")
        return

    tier_order = ['S5', 'S4', 'S3', 'S2', 'S1', 'S0']

    print(f"\n{'='*80}")
    print("  SENSOR ABLATION SUMMARY")
    print(f"{'='*80}")

    for tier in tier_order:
        tier_rows = summary_df[summary_df['tier'] == tier]
        if tier_rows.empty:
            continue

        info = SENSOR_TIERS[tier]
        n_vars = int(tier_rows['n_vars'].iloc[0]) if 'n_vars' in tier_rows.columns else '?'
        n_gt = int(tier_rows['n_gt_edges'].iloc[0]) if 'n_gt_edges' in tier_rows.columns else '?'

        print(f"\n  Tier {tier}: {info['name']} "
              f"({n_vars} vars, {n_gt} GT edges)")
        print(f"  {'Method':<20} {'SHD':>6} {'NSHD':>7} {'Prec':>6} "
              f"{'Recall':>7} {'F1':>6} {'Stab':>6} {'Edges':>6}")
        print(f"  {'-'*66}")

        for _, row in tier_rows.iterrows():
            method = row['method']
            shd = f"{row.get('shd_mean', '?'):.1f}" if 'shd_mean' in row else '?'
            nshd = f"{row.get('nshd_mean', '?'):.3f}" if 'nshd_mean' in row else '?'
            prec = f"{row.get('precision_mean', '?'):.3f}" if 'precision_mean' in row else '?'
            rec = f"{row.get('recall_mean', '?'):.3f}" if 'recall_mean' in row else '?'
            f1 = f"{row.get('f1_mean', '?'):.3f}" if 'f1_mean' in row else '?'
            stab = f"{row.get('edge_stability_mean', '?'):.3f}" if 'edge_stability_mean' in row else '?'
            edges = f"{row.get('n_edges_discovered_mean', '?'):.1f}" if 'n_edges_discovered_mean' in row else '?'
            print(f"  {method:<20} {shd:>6} {nshd:>7} {prec:>6} "
                  f"{rec:>7} {f1:>6} {stab:>6} {edges:>6}")


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Experiment A: Sensor Ablation Study for NeurIPS')
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of discovery seeds (default 1)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Policy episodes per arm per seed (default 1)')
    parser.add_argument('--tier', type=str, default=None,
                        choices=list(SENSOR_TIERS.keys()),
                        help='Run only a specific tier (default: all)')
    parser.add_argument('--include-pets', action='store_true',
                        help='Include DDPC_PETS policy arm (slow)')
    parser.add_argument('--skip-benchmarks', action='store_true',
                        help='Skip benchmark baselines (PolicyGRID only)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    # Determine tiers to run
    if args.tier:
        tiers = [args.tier]
    else:
        tiers = ['S5', 'S4', 'S3', 'S2', 'S1', 'S0']

    print(f"\n  Sensor Ablation Study")
    print(f"  Tiers: {tiers}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Skip benchmarks: {args.skip_benchmarks}")

    os.makedirs('neurips_results', exist_ok=True)

    t0 = time.time()
    raw_df, summary_df, bm_df = run_ablation(
        tiers, args.seeds, args.episodes,
        args.skip_benchmarks, args.include_pets)
    elapsed = time.time() - t0

    # Save outputs
    if not raw_df.empty:
        raw_df.to_csv('neurips_results/sensor_ablation_raw.csv', index=False)
        print(f"\n  Saved neurips_results/sensor_ablation_raw.csv "
              f"({len(raw_df)} rows)")

    if not summary_df.empty:
        summary_df.to_csv('neurips_results/sensor_ablation_summary.csv',
                          index=False)
        print(f"  Saved neurips_results/sensor_ablation_summary.csv")

    if not bm_df.empty:
        bm_df.to_csv('neurips_results/sensor_ablation_benchmarks.csv',
                      index=False)
        print(f"  Saved neurips_results/sensor_ablation_benchmarks.csv "
              f"({len(bm_df)} rows)")

    # Print summary table
    print_ablation_summary(summary_df)

    print(f"\n  Total wall time: {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
