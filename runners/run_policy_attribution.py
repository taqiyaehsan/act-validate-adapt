#!/usr/bin/env python3
"""
Policy attribution: run exceeding methods' edge sets through PolicyGRID's policy engine.
Tests whether higher F1 → better policy. Saves per-method CSVs.

Usage:
    python run_policy_attribution.py --method GIES --env open_window
    python run_policy_attribution.py --method all --env all
"""
import sys, os, json, warnings, random, time, argparse
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from run_full_pipeline import (SIM_CONFIGS, load_scaling, run_policy_for_seed,
                                COMFORT_TARGETS, API_KEY)

# Methods that exceed PolicyGRID and their envs
TARGETS = {
    'open_window': ['GIES', 'SAM', 'LLM'],
    'smart_building_rich': ['IID', 'LLM'],
}

def get_edges(method, env, data, cfg):
    """Generate edges for a method on an env."""
    cols = list(data.columns)
    sim_abs = os.path.abspath(cfg['sim_path']) if cfg.get('sim_path') else None
    np.random.seed(42); random.seed(42)

    if method == 'GIES':
        from benchmarks_new.gies_comparison import run_gies
        result = run_gies(data, simulation_path=sim_abs)
    elif method == 'SAM':
        from benchmarks_new.sam_baseline import run_sam_baseline
        result = run_sam_baseline(data, simulation_path=sim_abs)
    elif method == 'LLM':
        from src.generators import LLMGenerator
        gen = LLMGenerator(api_key=API_KEY, relevant_columns=cols,
                          dataset_type=cfg.get('dataset_type', 'smart_room'))
        result = gen.generate(data)
    elif method == 'IID':
        from benchmarks_new.iid_comparison import run_iid
        result = run_iid(data, simulation_path=sim_abs, max_iterations=5)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract edges
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        edges = result.get('edges', [])
    elif isinstance(result, (list, set)):
        edges = list(result)
    else:
        edges = []

    return {(str(s), str(t)) for s, t in edges}


def run_attribution(method, env):
    """Run one method's edges through the policy engine on one env."""
    cfg = SIM_CONFIGS[env]
    data = pd.read_csv(cfg['data_path'])
    cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
    data = data[cols]
    smap = load_scaling(cfg['scaling_path'])

    print(f"\n{'='*60}")
    print(f"  {method} on {env}")
    print(f"{'='*60}")

    # Get edges
    print(f"  Generating {method} edges...")
    t0 = time.time()
    edges = get_edges(method, env, data, cfg)
    print(f"  {len(edges)} edges ({time.time()-t0:.0f}s)")

    # Run policy
    print(f"  Running policy engine...")
    t0 = time.time()
    df = run_policy_for_seed(cfg, data, edges, smap)
    elapsed = time.time() - t0

    if df is not None:
        pg = df[df['method'] == 'PolicyGRID']
        if len(pg):
            for _, row in pg.iterrows():
                print(f"    ε={row['comfort_target']}: MO={row['MO']:.3f} kWh={row['kWh']:.2f} sat={row['satisfaction']:.1f}")

        out_dir = 'results/policy_attribution'
        os.makedirs(out_dir, exist_ok=True)
        out_path = f'{out_dir}/{method}_{env}_policy.csv'
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({elapsed:.0f}s)")
    else:
        print(f"  FAILED ({elapsed:.0f}s)")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='all')
    parser.add_argument('--env', default='all')
    args = parser.parse_args()

    if args.env == 'all':
        targets = TARGETS
    else:
        targets = {args.env: TARGETS.get(args.env, [])}

    for env, methods in targets.items():
        if args.method != 'all':
            methods = [args.method] if args.method in methods else []
        for method in methods:
            try:
                run_attribution(method, env)
            except Exception as e:
                print(f"  FATAL: {method} on {env}: {e}")
