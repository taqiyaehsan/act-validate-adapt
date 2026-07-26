#!/usr/bin/env python3
"""
Quick test: verify back-door adjustment improves smart_building_rich
and doesn't regress ASHRAE or hidden_vars.

Runs 1 seed, 5 iterations (faster than full 10) for each sim.
"""

import sys, os, json, time, random
import numpy as np
import pandas as pd
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline_cwm import create_cwm_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('backdoor_test.log', mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load ground truth
with open(os.path.join(_BASE_DIR, 'ground_truth_graphs.json')) as f:
    ALL_GT = json.load(f)

SIMS = {
    'smart_building_rich': {
        'sim_path': 'js/smart_building_rich.js',
        'obs_data_path': 'data_regen/smart_building_rich_processed.csv',
        'dataset_type': 'smart_building_rich',
        'has_regime': True,
        'actuator_vars': ['hvacpower', 'lightingpower'],
        'non_intervenable_vars': [
            'outdoortemp', 'solarradiation', 'occupancy', 'windowposition',
            'pmv', 'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
        'gt_key': 'smart_building_rich',
    },
    'hidden_vars': {
        'sim_path': 'js/smart_room_hidden_vars.js',
        'obs_data_path': 'data_regen/hidden_vars_processed.csv',
        'dataset_type': 'hidden_vars',
        'has_regime': False,
        'actuator_vars': [],
        'non_intervenable_vars': [],
        'gt_key': 'hidden_vars',
    },
    'ashrae': {
        'sim_path': None,
        'obs_data_path': 'data/ashrae_data_processed.csv',
        'dataset_type': 'ashrae',
        'has_regime': False,
        'actuator_vars': [],
        'non_intervenable_vars': [],
        'gt_key': 'ashrae',
    },
}

API_KEY = 'YOUR_OPENAI_API_KEY'
SEED = 42
MAX_ITER = 5


def compute_metrics(discovered_edges, gt_key):
    """Compute SHD, precision, recall, F1 against ground truth."""
    gt_edges = {(s.lower(), t.lower()) for s, t in ALL_GT[gt_key]['edges']}
    disc = {(str(s).lower(), str(t).lower()) for s, t in discovered_edges}

    tp = len(disc & gt_edges)
    fp = len(disc - gt_edges)
    fn = len(gt_edges - disc)
    shd = fp + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'shd': shd,
        'n_gt': len(gt_edges),
        'n_disc': len(disc),
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def run_one(sim_name, cfg):
    random.seed(SEED)
    np.random.seed(SEED)

    obs_path = os.path.join(_BASE_DIR, cfg['obs_data_path'])
    obs_df = pd.read_csv(obs_path)
    sim_path = (os.path.join(_BASE_DIR, cfg['sim_path'])
                if cfg['sim_path'] else None)

    logger.info(f"\n{'='*60}")
    logger.info(f"  RUNNING: {sim_name} ({len(obs_df)} rows, {len(obs_df.columns)} cols)")
    logger.info(f"{'='*60}")

    t0 = time.time()
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

    final_dag, final_metrics = pipeline.run_with_cwm()
    elapsed = time.time() - t0

    # Collect discovered edges
    validated = set(pipeline.pipeline.validated_edges)
    method_dags = pipeline.pipeline.get_method_dags()
    obs_only = set()
    for name, dag in method_dags.items():
        for e in dag:
            obs_only.add((str(e[0]).lower(), str(e[1]).lower()))

    # Compute metrics
    metrics_val = compute_metrics(validated, cfg['gt_key'])
    metrics_obs = compute_metrics(obs_only, cfg['gt_key'])

    logger.info(f"\n  {sim_name} RESULTS ({elapsed:.0f}s):")
    logger.info(f"  GT edges: {metrics_val['n_gt']}")
    logger.info(f"  Validated edges: {metrics_val['n_disc']} "
                f"(TP={metrics_val['tp']}, FP={metrics_val['fp']}, FN={metrics_val['fn']})")
    logger.info(f"  SHD={metrics_val['shd']}, P={metrics_val['precision']:.2f}, "
                f"R={metrics_val['recall']:.2f}, F1={metrics_val['f1']:.2f}")
    logger.info(f"  Obs-only union: {metrics_obs['n_disc']} edges, "
                f"SHD={metrics_obs['shd']}, F1={metrics_obs['f1']:.2f}")

    # List validated edges
    logger.info(f"  Validated edge list:")
    for e in sorted(validated):
        gt_edges = {(s.lower(), t.lower()) for s, t in ALL_GT[cfg['gt_key']]['edges']}
        marker = "✓" if (e[0].lower(), e[1].lower()) in gt_edges else "✗"
        logger.info(f"    {marker} {e[0]}→{e[1]}")

    return {
        'sim': sim_name,
        'elapsed_s': elapsed,
        'validated': metrics_val,
        'obs_only': metrics_obs,
    }


if __name__ == '__main__':
    results = {}
    for sim_name, cfg in SIMS.items():
        try:
            results[sim_name] = run_one(sim_name, cfg)
        except Exception as e:
            logger.error(f"FAILED {sim_name}: {e}", exc_info=True)
            results[sim_name] = {'error': str(e)}

    # Summary table
    print("\n" + "="*80)
    print("  BACK-DOOR ADJUSTMENT TEST SUMMARY")
    print("="*80)
    print(f"{'Sim':<25} {'SHD':>5} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'GT':>4} {'Time':>6}")
    print("-"*80)
    for sim_name, r in results.items():
        if 'error' in r:
            print(f"{sim_name:<25} ERROR: {r['error'][:50]}")
        else:
            m = r['validated']
            print(f"{sim_name:<25} {m['shd']:>5} {m['precision']:>6.2f} {m['recall']:>6.2f} "
                  f"{m['f1']:>6.2f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['n_gt']:>4} "
                  f"{r['elapsed_s']:>5.0f}s")
    print("="*80)
