#!/usr/bin/env python3
"""
Explore intersect-then-confirm strategy for PolicyGRID.

Key idea: Only test edges agreed on by 2+ discovery methods via intervention.
This narrows the candidate set from ~119 to ~20-30 edges, dramatically
reducing false positives from noisy N=5 intervention tests.

Configs:
  M: Intersect(2+) + FDR
  N: Intersect(2+) + FDR + obs-rescue
  O: Intersect(2+) + FDR + regime-filter + 8runs
  P: Intersect(3+) + FDR
  Q: Intersect(2+) only (no FDR) — see if intersect alone is enough
  R: Intersect(2+) + FDR + 10iter — more iterations to recover recall

Usage:
    python explore_intersect.py                     # all configs on building
    python explore_intersect.py --configs M N       # specific configs
    python explore_intersect.py --sim both          # both sims
"""

import sys, os, json, time, random, argparse
import numpy as np
import pandas as pd
import logging

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
            'energyconsumption', 'satisfaction',         ],
        'regime_vars': ['windowopen'],
    },
}

BENCHMARKS = {
    'smart_building_rich': {
        'PC': {'shd': 27, 'precision': 0.522, 'recall': 0.429, 'f1': 0.471},
    },
    'open_window': {
        'PC': {'shd': 6, 'precision': 0.50, 'recall': 0.50, 'f1': 0.50},
    },
}

CONFIGS = {
    'M': {
        'name': 'Intersect(2+)+FDR',
        'min_agree': 2,
        'use_fdr': True,
        'regime_filter': False,
        'extra_runs': False,
        'max_iter': 5,
    },
    'N': {
        'name': 'Intersect(2+)+FDR+obs',
        'min_agree': 2,
        'use_fdr': True,
        'regime_filter': False,
        'extra_runs': False,
        'max_iter': 5,
        # obs-rescue is built into pipeline_cwm.py already
    },
    'O': {
        'name': 'Intrsct(2+)+FDR+filt+8r',
        'min_agree': 2,
        'use_fdr': True,
        'regime_filter': True,
        'extra_runs': True,
        'max_iter': 5,
    },
    'P': {
        'name': 'Intersect(3+)+FDR',
        'min_agree': 3,
        'use_fdr': True,
        'regime_filter': False,
        'extra_runs': False,
        'max_iter': 5,
    },
    'Q': {
        'name': 'Intersect(2+) no FDR',
        'min_agree': 2,
        'use_fdr': False,
        'regime_filter': False,
        'extra_runs': False,
        'max_iter': 5,
    },
    'R': {
        'name': 'Intersect(2+)+FDR+10it',
        'min_agree': 2,
        'use_fdr': True,
        'regime_filter': False,
        'extra_runs': False,
        'max_iter': 10,
    },
    'S': {
        'name': 'Intrsct(2+)+FDR+obs+10it',
        'min_agree': 2,
        'use_fdr': True,
        'regime_filter': False,
        'extra_runs': False,
        'max_iter': 10,
        # obs-rescue + 10 iterations for max recall recovery
    },
    'T': {
        'name': 'Intrsct(2+)+all+10it',
        'min_agree': 2,
        'use_fdr': True,
        'regime_filter': True,
        'extra_runs': True,
        'max_iter': 10,
        # everything combined: intersect + FDR + obs-rescue + regime-filter + 8runs + 10iter
    },
    'U': {
        'name': 'T+relaxed_rescue',
        'min_agree': 2,
        'use_fdr': True,
        'regime_filter': True,
        'extra_runs': True,
        'max_iter': 10,
        # T config + lowered rescue partial_r threshold (0.3) for 2+ method edges
    },
}


def make_regime_filter_validator(original_validate, regime_vars, variable_mapping,
                                 variable_ranges, regime_threshold=0.10):
    """Filter runs where continuous regime variables changed too much."""
    from src.tester import EdgeValidator
    ev = EdgeValidator()

    def filtered_validate(pre_state, post_state, source_var, target_var):
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
                if rv_range <= 1.0:
                    continue
                if rv_range > 0:
                    frac_change = abs(rv_post - rv_pre) / rv_range
                    if frac_change > regime_threshold:
                        return 0.0
            except (KeyError, ValueError, TypeError):
                continue
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
    """Run PolicyGRID with intersect-then-confirm configuration."""
    from src.pipeline_cwm import create_cwm_pipeline
    import src.pipeline_cwm as pcwm

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

    # ── Patch: Regime-filter ──
    regime_vars = cfg.get('regime_vars', [])
    variable_mapping = pipeline.pipeline.tester.variable_mapping
    variable_ranges = pipeline.pipeline.tester.variable_ranges

    if exp.get('regime_filter', False) and regime_vars:
        original_validate = pipeline.pipeline.tester.edge_validator.validate_edge_changes
        filtered_validate = make_regime_filter_validator(
            original_validate, regime_vars, variable_mapping, variable_ranges,
            regime_threshold=0.10
        )
        pipeline.pipeline.tester.edge_validator.validate_edge_changes = filtered_validate

    # ── Patch: Extra runs ──
    if exp.get('extra_runs', False):
        pipeline.pipeline.tester.initial_test_count = 4
        pipeline.pipeline.tester.extended_test_count = 4

    print(f"\n{'='*60}")
    print(f"  Config {config_key}: {exp['name']}")
    print(f"  Sim: {sim_name} | Iter: {exp['max_iter']}")
    print(f"  MinAgree: {exp['min_agree']} | FDR: {exp['use_fdr']} "
          f"| Filter: {exp.get('regime_filter', False)} "
          f"| ExtraRuns: {exp.get('extra_runs', False)}")
    print(f"{'='*60}")

    t0 = time.time()

    # Set module-level flags
    pcwm._FDR_DISABLED = not exp['use_fdr']
    pcwm._RESCUE_DISABLED = False  # always keep rescue for non-intervenable
    pcwm._MIN_METHOD_AGREEMENT = exp['min_agree']

    try:
        final_dag, final_metrics = pipeline.run_with_cwm()
    finally:
        pcwm._FDR_DISABLED = False
        pcwm._RESCUE_DISABLED = False
        pcwm._MIN_METHOD_AGREEMENT = 0

    elapsed = time.time() - t0

    validated = set(pipeline.pipeline.validated_edges)
    m = compute_metrics(validated, cfg['gt_key'])

    print(f"\n  Time: {elapsed:.0f}s")
    print(f"  GT={m['n_gt']}  Disc={m['n_disc']}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(f"  SHD={m['shd']}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

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

        for name, bm in benchmarks.items():
            print(f"{'---':<5} {name:<25} {bm['shd']:>5} {bm['precision']:>7.3f} "
                  f"{bm['recall']:>7.3f} {bm['f1']:>7.3f} {'?':>4} {'?':>4} {'?':>4} {'---':>6}")
        print('-'*80)

        # Previous best results for reference
        prev_best = {'B': ('FDR only', 28, 0.500, 0.250, 0.333),
                     'J': ('FDR+obs-rescue', 37, 0.476, 0.357, 0.408)}
        for k, (name, shd, p, r, f1) in prev_best.items():
            print(f"{k:<5} {name:<25} {shd:>5} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {'?':>4} {'?':>4} {'?':>4} {'prev':>6}")
        print('-'*80)

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

        best_baseline_shd = min(bm['shd'] for bm in benchmarks.values()) if benchmarks else float('inf')
        best_baseline_f1 = max(bm['f1'] for bm in benchmarks.values()) if benchmarks else 0

        print(f"  Best SHD: {min(sim_results, key=lambda r: r['metrics']['shd'])['config']} "
              f"(SHD={best_shd}) vs PC (SHD={best_baseline_shd}) "
              f"{'>>> BEATS PC <<<' if best_shd < best_baseline_shd else ''}")
        print(f"  Best F1:  {max(sim_results, key=lambda r: r['metrics']['f1'])['config']} "
              f"(F1={best_f1:.3f}) vs PC (F1={best_baseline_f1:.3f}) "
              f"{'>>> BEATS PC <<<' if best_f1 > best_baseline_f1 else ''}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', choices=['building', 'open_window', 'both'],
                        default='building')
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS.keys()),
                        help='Which configs to run (e.g. M N P)')
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

        out_path = os.path.join(_BASE_DIR, 'explore_intersect_results.json')
        for r in results:
            r['edges'] = [list(e) for e in r['edges']]
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")
