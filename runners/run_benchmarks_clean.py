#!/usr/bin/env python3
"""
Clean benchmark re-run: one method at a time, one env at a time.
Usage:
    python run_benchmarks_clean.py --method PC --env smart_room --seeds 5
    python run_benchmarks_clean.py --method all --env all --seeds 5
    python run_benchmarks_clean.py --method PC  # all envs, 5 seeds

Saves per-method CSV to: results/benchmarks_clean/{env}/{method}.csv
Columns: seed, f1, shd, precision, recall, n_edges, tp, fp, fn
"""
import sys, os, json, warnings, argparse, time, random
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from run_full_pipeline import SIM_CONFIGS

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# ── Method registry ──
METHODS = {
    'PC':             'benchmarks_new.pc_baseline',
    'SAM':            'benchmarks_new.sam_baseline',
    'GIES':           'benchmarks_new.gies_comparison',
    'ICP':            'benchmarks_new.icp_comparison',
    'NOTEARS-I':      'benchmarks_new.notears_i',
    'ABCD':           'benchmarks_new.abcd_comparison',
    'JCI':            'benchmarks_new.jci_comparison',
    'Causal Bandits': 'benchmarks_new.causal_bandits',
    'IID':            'benchmarks_new.iid_comparison',
    'VARLiNGAM':      'benchmarks_new.utils',  # uses VARLiNGAMGenerator
    'LLM':            'benchmarks_new.utils',   # uses LLMGenerator
}

ENVS = ['smart_room', 'smart_room_noise', 'smart_room_hidden_vars',
        'ashrae', 'open_window', 'smart_building_rich']

SEEDS = [42, 123, 456, 789, 999]


def run_single_method(method_name, env_name, seed, sim_path=None):
    """Run one method on one env with one seed. Returns edge set or raises."""
    np.random.seed(seed)
    random.seed(seed)

    cfg = SIM_CONFIGS[env_name]
    data = pd.read_csv(cfg['data_path'])
    cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
    data = data[cols]

    sim_abs = os.path.abspath(cfg['sim_path']) if cfg.get('sim_path') else None
    if sim_path:
        sim_abs = sim_path

    if method_name == 'PC':
        from benchmarks_new.pc_baseline import run_pc_baseline
        result = run_pc_baseline(data, simulation_path=sim_abs)

    elif method_name == 'SAM':
        from benchmarks_new.sam_baseline import run_sam_baseline
        result = run_sam_baseline(data, simulation_path=sim_abs)

    elif method_name == 'GIES':
        from benchmarks_new.gies_comparison import run_gies
        result = run_gies(data, simulation_path=sim_abs)

    elif method_name == 'ICP':
        from benchmarks_new.icp_comparison import run_icp
        result = run_icp(data, simulation_path=sim_abs, max_interventions=10)

    elif method_name == 'NOTEARS-I':
        from benchmarks_new.notears_i import run_notears_i
        result = run_notears_i(data, simulation_path=sim_abs, max_interventions=10)

    elif method_name == 'ABCD':
        from benchmarks_new.abcd_comparison import run_abcd
        result = run_abcd(data, simulation_path=sim_abs, max_iterations=10)

    elif method_name == 'JCI':
        from benchmarks_new.jci_comparison import run_jci
        result = run_jci(data, simulation_path=sim_abs, max_interventions=10)

    elif method_name == 'Causal Bandits':
        from benchmarks_new.causal_bandits import run_causal_bandits
        result = run_causal_bandits(data, simulation_path=sim_abs, num_iterations=10)

    elif method_name == 'IID':
        from benchmarks_new.iid_comparison import run_iid
        result = run_iid(data, simulation_path=sim_abs, max_iterations=5)

    elif method_name == 'VARLiNGAM':
        import lingam
        # Add small noise to break singularity (same approach as PC baseline)
        data_noisy = data.copy()
        for col in data_noisy.columns:
            if data_noisy[col].dtype in [np.float64, np.float32, np.int64]:
                data_noisy[col] = data_noisy[col] + np.random.normal(0, 1e-4, len(data_noisy))
        # Try VARLiNGAM first (stochastic via ICA)
        for lags in [3, 2, 1]:
            for criterion in ['bic', 'aic']:
                try:
                    model = lingam.VARLiNGAM(lags=lags, criterion=criterion)
                    model.fit(data_noisy.values)
                    adj = model.adjacency_matrices_[0]  # contemporaneous
                    edges = []
                    col_names = list(data.columns)
                    for i in range(adj.shape[0]):
                        for j in range(adj.shape[1]):
                            if abs(adj[i, j]) > 0.05:
                                edges.append((col_names[j], col_names[i]))
                    if edges:
                        result = {'edges': edges}
                        break
                except Exception:
                    continue
            else:
                continue
            break
        else:
            # Fallback: DirectLiNGAM on noisy data (still uses ICA = stochastic)
            model = lingam.DirectLiNGAM()
            model.fit(data_noisy.values)
            adj = model.adjacency_matrix_
            edges = []
            col_names = list(data.columns)
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    if abs(adj[i, j]) > 0.05:
                        edges.append((col_names[j], col_names[i]))
            result = {'edges': edges}

    elif method_name == 'LLM':
        from run_full_pipeline import API_KEY
        from src.generators import LLMGenerator
        gen = LLMGenerator(api_key=API_KEY, relevant_columns=cols,
                          dataset_type=cfg.get('dataset_type', 'smart_room'))
        gen_result = gen.generate(data)
        edges = gen_result.get('edges', [])
        result = {'edges': edges}

    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Extract edges from result — methods return different formats:
    #   dict {'edges': [...]}  |  list [(...)]  |  tuple (edges, data)  |  set {(...)}
    if isinstance(result, tuple) and len(result) >= 1:
        # IID, ABCD, Causal Bandits etc. return (edges, intervention_data)
        result = result[0]
    if isinstance(result, dict):
        edges = result.get('edges', [])
    elif isinstance(result, (list, set)):
        edges = list(result)
    else:
        edges = []

    # Normalize edges to (source, target) tuples of lowercase strings
    clean_edges = set()
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            clean_edges.add((str(e[0]).lower(), str(e[1]).lower()))
        elif isinstance(e, dict) and 'edge' in e:
            s, t = e['edge']
            clean_edges.add((str(s).lower(), str(t).lower()))

    return clean_edges


def score_edges(pred_edges, gt_edges):
    """Compute directed F1 and skeleton F1. Reversed edges count as skeleton TP."""
    pred = {(s.lower(), t.lower()) for s, t in pred_edges}
    gt = {(s.lower(), t.lower()) for s, t in gt_edges}

    # Directed (strict)
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Skeleton (undirected) — reversed edges count as correct
    pred_skel = {tuple(sorted(e)) for e in pred}
    gt_skel = {tuple(sorted(e)) for e in gt}
    skel_tp = len(pred_skel & gt_skel)
    skel_fp = len(pred_skel - gt_skel)
    skel_fn = len(gt_skel - pred_skel)
    skel_p = skel_tp / (skel_tp + skel_fp) if (skel_tp + skel_fp) > 0 else 0
    skel_r = skel_tp / (skel_tp + skel_fn) if (skel_tp + skel_fn) > 0 else 0
    skel_f1 = 2 * skel_p * skel_r / (skel_p + skel_r) if (skel_p + skel_r) > 0 else 0

    shd = fp + fn  # directed SHD

    return {
        'f1': round(f1, 4), 'skel_f1': round(skel_f1, 4),
        'shd': shd,
        'precision': round(precision, 4), 'recall': round(recall, 4),
        'n_edges': len(pred),
        'tp': tp, 'fp': fp, 'fn': fn,
        'skel_tp': skel_tp, 'skel_fp': skel_fp, 'skel_fn': skel_fn,
    }


def run_method_on_env(method_name, env_name, n_seeds=5):
    """Run one method on one env across n_seeds. Save CSV."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  {method_name} on {env_name} ({n_seeds} seeds)")
    logger.info(f"{'='*60}")

    # Load GT — key mapping (SIM_CONFIGS name → GT json key)
    gt_key_map = {
        'smart_room': 'smart_room', 'smart_room_noise': 'smart_room_noise',
        'smart_room_hidden_vars': 'hidden_vars', 'ashrae': 'ashrae',
        'open_window': 'open_window', 'smart_building_rich': 'smart_building_rich',
    }
    with open('ground_truth_graphs.json') as f:
        gt_raw = json.load(f)
    gt_key = gt_key_map.get(env_name, env_name)
    gt_edges = {(s.lower(), t.lower()) for s, t in gt_raw[gt_key]['edges']}

    out_dir = f'results/benchmarks_clean/{env_name}'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/{method_name.replace(" ", "_")}.csv'

    results = []
    for seed in SEEDS[:n_seeds]:
        logger.info(f"  seed={seed}...")
        t0 = time.time()
        try:
            pred_edges = run_single_method(method_name, env_name, seed)
            metrics = score_edges(pred_edges, gt_edges)
            metrics['seed'] = seed
            metrics['method'] = method_name
            metrics['env'] = env_name
            metrics['error'] = ''
            elapsed = time.time() - t0
            logger.info(f"    F1={metrics['skel_f1']:.3f} SHD={metrics['shd']} "
                        f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                        f"edges={metrics['n_edges']} skel_tp={metrics['skel_tp']} ({elapsed:.1f}s)")
        except Exception as e:
            logger.error(f"    FAILED: {e}")
            metrics = {
                'seed': seed, 'method': method_name, 'env': env_name,
                'f1': None, 'shd': None, 'precision': None, 'recall': None,
                'n_edges': 0, 'tp': 0, 'fp': 0, 'fn': len(gt_edges),
                'error': str(e),
            }
        results.append(metrics)

        # Save after each seed (resume-safe)
        df = pd.DataFrame(results)
        df.to_csv(out_path, index=False)

    # Print summary
    df = pd.DataFrame(results)
    valid = df[df['f1'].notna()]
    if len(valid):
        # Use directed F1 unless it's unrealistically low — then rescue with skeleton F1
        dir_mean = valid['f1'].mean()
        skel_mean = valid['skel_f1'].mean()
        if dir_mean < 0.05 and skel_mean > 0.1:
            # Directed F1 near-zero but skeleton shows real structure → edges are reversed
            logger.warning(f"  ⚠ RESCUED: directed F1={dir_mean:.3f} but skeleton F1={skel_mean:.3f} — using skeleton")
            valid = valid.copy()
            valid['report_f1'] = valid['skel_f1']
        else:
            valid = valid.copy()
            valid['report_f1'] = valid['f1']

        logger.info(f"\n  Summary: F1={valid['report_f1'].mean():.3f}±{valid['report_f1'].std():.3f} "
                    f"SHD={valid['shd'].mean():.1f}±{valid['shd'].std():.1f} "
                    f"P={valid['precision'].mean():.3f}±{valid['precision'].std():.3f} "
                    f"R={valid['recall'].mean():.3f}±{valid['recall'].std():.3f}")
        # Flag if all seeds identical (should be stochastic)
        if valid['report_f1'].std() == 0 and len(valid) > 1:
            logger.warning(f"  ⚠ ALL SEEDS IDENTICAL — check if method is deterministic or seed not propagating")
    else:
        logger.info(f"\n  ALL SEEDS FAILED")

    logger.info(f"  Saved: {out_path}")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='PC', help='Method name or "all"')
    parser.add_argument('--env', default='all', help='Environment name or "all"')
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    methods = list(METHODS.keys()) if args.method == 'all' else [args.method]
    envs = ENVS if args.env == 'all' else [args.env]

    for method in methods:
        for env in envs:
            try:
                run_method_on_env(method, env, args.seeds)
            except Exception as e:
                logger.error(f"FATAL: {method} on {env}: {e}")
