#!/usr/bin/env python3
"""
Explore fix combos for PolicyGRID discovery quality.

Tests 6 configurations on both smart_building_rich and open_window:
  A: Pre-fix baseline (no FDR, no weighted rescue, raw t-test)
  B: Current post-fix (FDR + weighted rescue)
  C: ANCOVA validation only (regime-adjusted effect sizes)
  D: FDR + weighted rescue + ANCOVA
  E: ANCOVA + 10 iterations
  F: All three + 10 iterations

Usage:
    python explore_fixes.py                    # all configs, both sims
    python explore_fixes.py --sim building     # just smart_building_rich
    python explore_fixes.py --configs A C D    # specific configs
"""

import sys, os, json, time, random, argparse, copy
import numpy as np
import pandas as pd
import logging
from unittest.mock import patch
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_BASE_DIR, 'ground_truth_graphs.json')) as f:
    ALL_GT = json.load(f)

API_KEY = 'YOUR_OPENAI_API_KEY'
SEED = 42

SIM_CONFIGS = {
    'smart_building_rich': {
        'sim_path': 'js/smart_building_rich.js',
        'obs_data_path': 'data_regen/smart_building_rich_processed.csv',
        'dataset_type': 'smart_building_rich',
        'gt_key': 'smart_building_rich',
        'actuator_vars': ['hvacpower', 'lightingpower'],
        'non_intervenable_vars': [
            'outdoortemp', 'solarradiation', 'occupancy', 'windowposition',
            'pmv',
        ],
        # Regime variables that resample each step (confounders)
        'regime_vars': ['occupancy', 'windowposition'],
    },
    'open_window': {
        'sim_path': 'open_window.js',
        'obs_data_path': 'data_regen/challenge1_data_10k_processed.csv',
        'dataset_type': 'open_window',
        'gt_key': 'open_window',
        'actuator_vars': [],
        'non_intervenable_vars': [
            'windowopen', 'pmv', 'outdoortemperature',
            'energyconsumption', 'satisfaction',
        ],
        'regime_vars': ['windowopen'],
    },
}

# Benchmark baselines for comparison
BENCHMARKS = {
    'smart_building_rich': {
        'PC':    {'shd': 27, 'precision': 0.522, 'recall': 0.429, 'f1': 0.471},
    },
    'open_window': {
        'PC':    {'shd': 6, 'precision': 0.50, 'recall': 0.50, 'f1': 0.50},
    },
}


# ─── Experiment Configurations ───────────────────────────────────────

CONFIGS = {
    # All configs keep rescue ON (needed for non-intervenable edges)
    'A': {
        'name': 'Baseline (rescue only)',
        'use_fdr': False,
        'use_weighted_rescue': True,
        'use_ancova': False,
        'regime_filter': False,
        'extra_runs': False,
        'max_iter': 5,
    },
    'B': {
        'name': 'FDR + rescue',
        'use_fdr': True,
        'use_weighted_rescue': True,
        'use_ancova': False,
        'regime_filter': False,
        'extra_runs': False,
        'max_iter': 5,
    },
    'C': {
        'name': 'ANCOVA + rescue',
        'use_fdr': False,
        'use_weighted_rescue': True,
        'use_ancova': True,
        'regime_filter': False,
        'extra_runs': False,
        'max_iter': 5,
    },
    'D': {
        'name': 'FDR + ANCOVA + rescue',
        'use_fdr': True,
        'use_weighted_rescue': True,
        'use_ancova': True,
        'regime_filter': False,
        'extra_runs': False,
        'max_iter': 5,
    },
    'E': {
        'name': 'Regime-filter + rescue',
        'use_fdr': False,
        'use_weighted_rescue': True,
        'use_ancova': False,
        'regime_filter': True,
        'extra_runs': False,
        'max_iter': 5,
    },
    'F': {
        'name': 'Filter + 8runs + rescue',
        'use_fdr': False,
        'use_weighted_rescue': True,
        'use_ancova': False,
        'regime_filter': True,
        'extra_runs': True,
        'max_iter': 5,
    },
    'G': {
        'name': 'Filter+FDR+rescue',
        'use_fdr': True,
        'use_weighted_rescue': True,
        'use_ancova': False,
        'regime_filter': True,
        'extra_runs': True,
        'max_iter': 5,
    },
    'H': {
        'name': '10 iter + rescue',
        'use_fdr': False,
        'use_weighted_rescue': True,
        'use_ancova': False,
        'regime_filter': False,
        'extra_runs': False,
        'max_iter': 10,
    },
    'I': {
        'name': 'Filter+FDR+10iter',
        'use_fdr': True,
        'use_weighted_rescue': True,
        'use_ancova': False,
        'regime_filter': True,
        'extra_runs': True,
        'max_iter': 10,
    },
    'J': {
        'name': 'FDR+obs-rescue',
        'use_fdr': True,
        'use_weighted_rescue': True,
        'use_ancova': False,
        'regime_filter': False,
        'extra_runs': False,
        'max_iter': 5,
        # FDR with obs-rescue is now built into pipeline_cwm.py:
        # FDR-rejected edges rescued if |partial_r|>0.3 AND p<0.15
    },
    'K': {
        'name': 'FDR+obs+filter+8runs',
        'use_fdr': True,
        'use_weighted_rescue': True,
        'use_ancova': False,
        'regime_filter': True,
        'extra_runs': True,
        'max_iter': 5,
    },
    'L': {
        'name': 'FDR+obs+10iter',
        'use_fdr': True,
        'use_weighted_rescue': True,
        'use_ancova': False,
        'regime_filter': False,
        'extra_runs': False,
        'max_iter': 10,
    },
}


# ─── ANCOVA Effect Size (regime-adjusted via partial regression) ──────

def ancova_effect_size(pre_state, post_state, source_var, target_var,
                       regime_vars, obs_data, variable_mapping, variable_ranges):
    """
    Compute regime-adjusted effect size using multivariate partial regression.

    Fits: target ~ source + regime_1 + ... + regime_k on observational data,
    then uses the source coefficient to compute the expected target change
    due to source alone, removing regime confounding.

    Unlike simple OLS per regime var, this correctly handles correlation
    between regime vars and the source variable itself.
    """
    from src.tester import EdgeValidator

    ev = EdgeValidator()
    source_key = ev._find_key(pre_state, source_var)
    target_key = ev._find_key(pre_state, target_var)

    if source_key is None or target_key is None:
        return 0.0

    try:
        source_pre = float(pre_state[source_key])
        source_post = float(post_state[source_key])
        target_pre = float(pre_state[target_key])
        target_post = float(post_state[target_key])

        source_change = source_post - source_pre
        if abs(source_change) < 0.001:
            return 0.0

        # Build multivariate regression: target ~ source + regime vars
        src_col = next((c for c in obs_data.columns if c.lower() == source_var.lower()), None)
        tgt_col = next((c for c in obs_data.columns if c.lower() == target_var.lower()), None)

        if not src_col or not tgt_col:
            # Fall back to raw effect
            return _raw_effect(pre_state, post_state, source_key, target_key,
                             source_var, target_var, variable_ranges)

        # Collect predictor columns: source + regime vars
        predictor_cols = [src_col]
        for rv in regime_vars:
            rv_col = next((c for c in obs_data.columns if c.lower() == rv.lower()), None)
            if rv_col and rv_col != src_col and rv_col != tgt_col:
                predictor_cols.append(rv_col)

        X = obs_data[predictor_cols].values
        y = obs_data[tgt_col].values

        # Add intercept
        X_aug = np.column_stack([np.ones(len(X)), X])

        # OLS: beta = (X'X)^-1 X'y
        try:
            beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return _raw_effect(pre_state, post_state, source_key, target_key,
                             source_var, target_var, variable_ranges)

        # beta[1] = partial effect of source on target, controlling for regime
        source_beta = beta[1]

        # Predicted target change due to source alone
        predicted_target_change = source_beta * source_change

        # Normalize
        source_range_info = variable_ranges.get(source_var, {})
        target_range_info = variable_ranges.get(target_var, {})
        source_range = source_range_info.get('max', 100) - source_range_info.get('min', 0) if source_range_info else 1.0
        target_range = target_range_info.get('max', 100) - target_range_info.get('min', 0) if target_range_info else 1.0

        source_norm = source_change / max(source_range, 0.01)
        target_norm = predicted_target_change / max(target_range, 0.01)

        signed_effect = target_norm / max(abs(source_norm), 0.01)
        return signed_effect

    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"ANCOVA effect error: {e}")
        return 0.0


def _raw_effect(pre_state, post_state, source_key, target_key,
                source_var, target_var, variable_ranges):
    """Fall back to raw effect size (same as EdgeValidator)."""
    source_change = float(post_state[source_key]) - float(pre_state[source_key])
    target_change = float(post_state[target_key]) - float(pre_state[target_key])
    if abs(source_change) < 0.001:
        return 0.0
    source_range_info = variable_ranges.get(source_var, {})
    target_range_info = variable_ranges.get(target_var, {})
    source_range = source_range_info.get('max', 100) - source_range_info.get('min', 0) if source_range_info else 1.0
    target_range = target_range_info.get('max', 100) - target_range_info.get('min', 0) if target_range_info else 1.0
    source_norm = source_change / max(source_range, 0.01)
    target_norm = target_change / max(target_range, 0.01)
    return target_norm / max(abs(source_norm), 0.01)


# ─── Regime-Filter Effect Size ───────────────────────────────────────

def make_regime_filter_validator(original_validate, regime_vars, variable_mapping,
                                 variable_ranges, regime_threshold=0.10):
    """
    Wraps the original validate_edge_changes to return 0.0 (no effect)
    when continuous regime variables changed more than `regime_threshold`
    fraction of their range between pre and post.

    Binary variables (range <=1) are excluded from filtering — they're
    either on or off, and filtering would discard too many runs.

    This filters out confounded runs rather than adjusting them.
    The t-test then only uses "clean" runs where the regime was stable.
    """
    from src.tester import EdgeValidator
    ev = EdgeValidator()

    def filtered_validate(pre_state, post_state, source_var, target_var):
        # Check if any CONTINUOUS regime var changed too much
        for rv in regime_vars:
            rv_mapped = variable_mapping.get(rv.lower(), rv)
            rv_key = ev._find_key(pre_state, rv_mapped)
            if rv_key is None:
                continue

            try:
                rv_pre = float(pre_state[rv_key])
                rv_post = float(post_state[rv_key])
                rv_range_info = variable_ranges.get(rv_mapped, {})
                rv_range = rv_range_info.get('max', 1) - rv_range_info.get('min', 0)

                # Skip binary/small-range variables — filtering them
                # would discard too many runs (e.g., windowOpen 0→1)
                if rv_range <= 1.0:
                    continue

                if rv_range > 0:
                    frac_change = abs(rv_post - rv_pre) / rv_range
                    if frac_change > regime_threshold:
                        # Regime shifted — return 0.0 to filter this run
                        return 0.0
            except (KeyError, ValueError, TypeError):
                continue

        # Regime stable — compute normal effect
        return original_validate(pre_state, post_state, source_var, target_var)

    return filtered_validate


def compute_metrics(discovered_edges, gt_key):
    gt_edges = {(s.lower(), t.lower()) for s, t in ALL_GT[gt_key]['edges']}
    disc = {(str(s).lower(), str(t).lower()) for s, t in discovered_edges}
    tp = len(disc & gt_edges)
    fp = len(disc - gt_edges)
    fn = len(gt_edges - disc)
    shd = fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'shd': shd, 'n_gt': len(gt_edges), 'n_disc': len(disc),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1}


def run_config(sim_name, config_key):
    """Run PolicyGRID with a specific configuration."""
    from src.pipeline_cwm import create_cwm_pipeline

    cfg = SIM_CONFIGS[sim_name]
    exp = CONFIGS[config_key]

    random.seed(SEED)
    np.random.seed(SEED)

    obs_path = os.path.join(_BASE_DIR, cfg['obs_data_path'])
    obs_df = pd.read_csv(obs_path)
    sim_path = os.path.join(_BASE_DIR, cfg['sim_path']) if cfg['sim_path'] else None

    pipeline = create_cwm_pipeline(
        csv_data=obs_df,
        api_key=API_KEY,
        smart_room_path=sim_path,
        dataset_type=cfg['dataset_type'],
        max_iterations=exp['max_iter'],
        alpha=0.5, beta=0.5,
        effect_threshold=0.1,
        actuator_vars=cfg.get('actuator_vars', []),
        non_intervenable_vars=cfg.get('non_intervenable_vars', []),
    )

    # ── Patch 1: ANCOVA effect sizes ──
    regime_vars = cfg.get('regime_vars', [])
    variable_mapping = pipeline.pipeline.tester.variable_mapping
    variable_ranges = pipeline.pipeline.tester.variable_ranges

    if exp['use_ancova']:
        def ancova_validate(pre_state, post_state, source_var, target_var):
            return ancova_effect_size(
                pre_state, post_state, source_var, target_var,
                regime_vars, obs_df, variable_mapping, variable_ranges
            )
        pipeline.pipeline.tester.edge_validator.validate_edge_changes = ancova_validate

    # ── Patch 2: Regime-filter effect sizes ──
    if exp.get('regime_filter', False) and regime_vars:
        original_validate = pipeline.pipeline.tester.edge_validator.validate_edge_changes
        filtered_validate = make_regime_filter_validator(
            original_validate, regime_vars, variable_mapping, variable_ranges,
            regime_threshold=0.10  # filter runs with >10% regime range change
        )
        pipeline.pipeline.tester.edge_validator.validate_edge_changes = filtered_validate

    # ── Patch 3: Extra runs per edge ──
    if exp.get('extra_runs', False):
        pipeline.pipeline.tester.initial_test_count = 4   # 4 instead of 2
        pipeline.pipeline.tester.extended_test_count = 4   # 4 instead of 3
        # Total: 8 runs per edge instead of 5

    print(f"\n{'='*60}")
    print(f"  Config {config_key}: {exp['name']}")
    print(f"  Sim: {sim_name} | Iter: {exp['max_iter']}")
    print(f"  FDR: {exp['use_fdr']} | Weighted: {exp['use_weighted_rescue']} "
          f"| ANCOVA: {exp['use_ancova']} | Filter: {exp.get('regime_filter', False)} "
          f"| ExtraRuns: {exp.get('extra_runs', False)}")
    print(f"{'='*60}")

    t0 = time.time()

    # Run with config-specific patches
    # We need to patch the FDR and rescue settings dynamically
    import src.pipeline_cwm as pcwm

    # Save originals
    orig_fdr_min = getattr(pcwm, '_FDR_MIN_EDGES_OVERRIDE', None)
    orig_rescue_override = getattr(pcwm, '_RESCUE_DISABLED', None)

    # Set overrides as module-level flags that the code will check
    pcwm._FDR_DISABLED = not exp['use_fdr']
    pcwm._RESCUE_DISABLED = not exp['use_weighted_rescue']

    try:
        final_dag, final_metrics = pipeline.run_with_cwm()
    finally:
        # Cleanup
        pcwm._FDR_DISABLED = False
        pcwm._RESCUE_DISABLED = False

    elapsed = time.time() - t0

    validated = set(pipeline.pipeline.validated_edges)
    m = compute_metrics(validated, cfg['gt_key'])

    print(f"  Time: {elapsed:.0f}s")
    print(f"  GT={m['n_gt']}  Disc={m['n_disc']}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(f"  SHD={m['shd']}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    # Show edges
    gt_edges = {(s.lower(), t.lower()) for s, t in ALL_GT[cfg['gt_key']]['edges']}
    for e in sorted(validated):
        mark = "TP" if (e[0].lower(), e[1].lower()) in gt_edges else "FP"
        print(f"    [{mark}] {e[0]}→{e[1]}")

    return {
        'config': config_key,
        'config_name': exp['name'],
        'sim': sim_name,
        'elapsed': elapsed,
        'metrics': m,
        'edges': list(validated),
    }


def print_summary(results):
    """Print comparison table across all configs."""
    # Group by sim
    by_sim = {}
    for r in results:
        by_sim.setdefault(r['sim'], []).append(r)

    for sim_name, sim_results in by_sim.items():
        benchmarks = BENCHMARKS.get(sim_name, {})

        print(f"\n{'='*80}")
        print(f"  COMPARISON: {sim_name}")
        print(f"{'='*80}")
        print(f"{'Config':<5} {'Name':<25} {'SHD':>5} {'P':>7} {'R':>7} {'F1':>7} "
              f"{'TP':>4} {'FP':>4} {'FN':>4} {'Time':>6}")
        print('-'*80)

        # Benchmarks
        for name, bm in benchmarks.items():
            print(f"{'---':<5} {name:<25} {bm['shd']:>5} {bm['precision']:>7.3f} "
                  f"{bm['recall']:>7.3f} {bm['f1']:>7.3f} {'?':>4} {'?':>4} {'?':>4} {'---':>6}")

        print('-'*80)

        # Our configs
        best_shd = min(r['metrics']['shd'] for r in sim_results)
        best_f1 = max(r['metrics']['f1'] for r in sim_results)

        for r in sim_results:
            m = r['metrics']
            shd_mark = ' *' if m['shd'] == best_shd else ''
            f1_mark = ' *' if m['f1'] == best_f1 else ''
            print(f"{r['config']:<5} {r['config_name']:<25} {m['shd']:>5}{shd_mark:<2} "
                  f"{m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f}{f1_mark:<2} "
                  f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {r['elapsed']:>5.0f}s")

        print('='*80)

        # Verdict vs benchmarks
        best_baseline_shd = min(bm['shd'] for bm in benchmarks.values()) if benchmarks else float('inf')
        best_baseline_f1 = max(bm['f1'] for bm in benchmarks.values()) if benchmarks else 0

        print(f"  Best SHD: Config {min(sim_results, key=lambda r: r['metrics']['shd'])['config']} "
              f"(SHD={best_shd}) vs PC baseline (SHD={best_baseline_shd})")
        print(f"  Best F1:  Config {max(sim_results, key=lambda r: r['metrics']['f1'])['config']} "
              f"(F1={best_f1:.3f}) vs PC baseline (F1={best_baseline_f1:.3f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', choices=['building', 'open_window', 'both'], default='both')
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS.keys()),
                        help='Which configs to run (e.g. A C D)')
    args = parser.parse_args()

    sims_to_run = []
    if args.sim in ('building', 'both'):
        sims_to_run.append('smart_building_rich')
    if args.sim in ('open_window', 'both'):
        sims_to_run.append('open_window')

    results = []
    for sim_name in sims_to_run:
        for config_key in args.configs:
            if config_key not in CONFIGS:
                print(f"Unknown config: {config_key}")
                continue
            try:
                r = run_config(sim_name, config_key)
                results.append(r)
            except Exception as e:
                print(f"\n  {config_key}/{sim_name} FAILED: {e}")
                import traceback; traceback.print_exc()

    if results:
        print_summary(results)

        # Save raw results
        out_path = os.path.join(_BASE_DIR, 'explore_fixes_results.json')
        # Convert sets to lists for JSON
        for r in results:
            r['edges'] = [list(e) for e in r['edges']]
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")
