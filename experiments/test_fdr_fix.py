#!/usr/bin/env python3
"""
Quick A/B test: pre-fix vs post-fix discovery quality.

Runs smart_building_rich and open_window with 1 seed, 5 iterations each.
The "fix" is already in pipeline_cwm.py (FDR correction + weighted rescue).
We compare against benchmark baselines (PC) to see if PolicyGRID now beats them.

Usage:
    python test_fdr_fix.py                    # both sims
    python test_fdr_fix.py --sim building     # just smart_building_rich
    python test_fdr_fix.py --sim open_window  # just open_window
"""

import sys, os, json, time, random, argparse
import numpy as np
import pandas as pd
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline_cwm import create_cwm_pipeline

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
MAX_ITER = 5

SIM_CONFIGS = {
    'smart_building_rich': {
        'sim_path': 'js/smart_building_rich.js',
        'obs_data_path': 'data_regen/smart_building_rich_processed.csv',
        'dataset_type': 'smart_building_rich',
        'gt_key': 'smart_building_rich',
        'actuator_vars': ['hvacpower', 'lightingpower'],
        'non_intervenable_vars': [
            'outdoortemp', 'solarradiation', 'occupancy', 'windowposition',
            'pmv', 'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
    },
    'open_window': {
        'sim_path': 'open_window.js',
        'obs_data_path': 'data_regen/challenge1_data_10k_processed.csv',
        'dataset_type': 'open_window',
        'gt_key': 'open_window',
        'actuator_vars': [],
        'non_intervenable_vars': [
            'windowopen', 'pmv', 'outdoortemperature',
            'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
    },
}

# Pre-fix benchmark results (from sensor_ablation_benchmarks.csv)
BENCHMARKS = {
    'smart_building_rich': {
        'PC':    {'shd': 27, 'precision': 0.522, 'recall': 0.429, 'f1': 0.471},
        'GIES':  {'shd': 27, 'precision': 0.522, 'recall': 0.429, 'f1': 0.471},
        'ICP':   {'shd': 27, 'precision': 0.524, 'recall': 0.393, 'f1': 0.449},
        'PolicyGRID_pre': {'shd': 39, 'precision': 0.343, 'recall': 0.429, 'f1': 0.381},
    },
    'open_window': {
        'PC':    {'shd': 6, 'precision': 0.50, 'recall': 0.50, 'f1': 0.50},
        'PolicyGRID_pre': {'shd': 6, 'precision': 0.60, 'recall': 0.50, 'f1': 0.55},
    },
}


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


def run_discovery(sim_name):
    """Run PolicyGRID discovery for one sim."""
    cfg = SIM_CONFIGS[sim_name]
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
        max_iterations=MAX_ITER,
        alpha=0.5, beta=0.5,
        effect_threshold=0.1,
        actuator_vars=cfg.get('actuator_vars', []),
        non_intervenable_vars=cfg.get('non_intervenable_vars', []),
    )

    print(f"\n{'='*60}")
    print(f"  Running PolicyGRID (post-fix) on {sim_name}")
    print(f"  GT edges: {len(ALL_GT[cfg['gt_key']]['edges'])}")
    print(f"{'='*60}")

    t0 = time.time()
    final_dag, final_metrics = pipeline.run_with_cwm()
    elapsed = time.time() - t0

    validated = set(pipeline.pipeline.validated_edges)
    m = compute_metrics(validated, cfg['gt_key'])

    print(f"\n  Time: {elapsed:.0f}s")
    print(f"  GT={m['n_gt']}  Disc={m['n_disc']}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(f"  SHD={m['shd']}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    # Show edges
    gt_edges = {(s.lower(), t.lower()) for s, t in ALL_GT[cfg['gt_key']]['edges']}
    for e in sorted(validated):
        mark = "TP" if (e[0].lower(), e[1].lower()) in gt_edges else "FP"
        print(f"    [{mark}] {e[0]}→{e[1]}")

    return {'sim': sim_name, 'elapsed': elapsed, 'metrics': m}


def print_comparison(sim_name, post_fix_metrics):
    """Print before/after comparison table."""
    benchmarks = BENCHMARKS.get(sim_name, {})
    m = post_fix_metrics

    print(f"\n{'='*70}")
    print(f"  COMPARISON: {sim_name}")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'SHD':>5} {'P':>7} {'R':>7} {'F1':>7} {'TP':>4} {'FP':>4} {'FN':>4}")
    print('-'*70)

    for name, bm in benchmarks.items():
        tp = bm.get('tp', '?')
        fp = bm.get('fp', '?')
        fn = bm.get('fn', '?')
        print(f"{name:<25} {bm['shd']:>5} {bm['precision']:>7.3f} {bm['recall']:>7.3f} "
              f"{bm['f1']:>7.3f} {str(tp):>4} {str(fp):>4} {str(fn):>4}")

    print(f"{'PolicyGRID (post-fix)':<25} {m['shd']:>5} {m['precision']:>7.3f} {m['recall']:>7.3f} "
          f"{m['f1']:>7.3f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}")
    print('='*70)

    # Verdict
    best_baseline_shd = min(bm['shd'] for bm in benchmarks.values())
    if m['shd'] < best_baseline_shd:
        print(f"  >>> PolicyGRID BEATS best baseline (SHD {m['shd']} < {best_baseline_shd})")
    elif m['shd'] == best_baseline_shd:
        print(f"  --- PolicyGRID TIES best baseline (SHD {m['shd']})")
    else:
        print(f"  <<< PolicyGRID LOSES to best baseline (SHD {m['shd']} > {best_baseline_shd})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', choices=['building', 'open_window', 'both'], default='both')
    args = parser.parse_args()

    sims_to_run = []
    if args.sim in ('building', 'both'):
        sims_to_run.append('smart_building_rich')
    if args.sim in ('open_window', 'both'):
        sims_to_run.append('open_window')

    results = []
    for sim_name in sims_to_run:
        try:
            r = run_discovery(sim_name)
            results.append(r)
            print_comparison(sim_name, r['metrics'])
        except Exception as e:
            print(f"\n  {sim_name} FAILED: {e}")
            import traceback; traceback.print_exc()

    # Final summary
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"  OVERALL SUMMARY")
        print(f"{'='*70}")
        for r in results:
            m = r['metrics']
            print(f"  {r['sim']}: SHD={m['shd']}, F1={m['f1']:.3f}, time={r['elapsed']:.0f}s")
