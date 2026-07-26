#!/usr/bin/env python3
"""
Policy attribution v2: generate edges ONCE, save to JSON, then run policy.
Usage:
    python run_policy_attribution_v2.py --method SAM --env open_window
    python run_policy_attribution_v2.py --method all --env open_window
"""
import sys, os, json, warnings, random, time, argparse
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from run_full_pipeline import (SIM_CONFIGS, load_scaling, run_policy_for_seed,
                                COMFORT_TARGETS, API_KEY)

EDGE_CACHE_DIR = 'results/policy_attribution/edge_cache'
OUT_DIR = 'results/policy_attribution'
os.makedirs(EDGE_CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def generate_edges(method, env, seed=42):
    """Generate edges and cache to JSON. Returns edge set."""
    cache_path = f'{EDGE_CACHE_DIR}/{method}_{env}_seed{seed}.json'
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            edges = json.load(f)
        print(f"  Loaded cached edges: {cache_path} ({len(edges)} edges)")
        return {(s, t) for s, t in edges}

    cfg = SIM_CONFIGS[env]
    data = pd.read_csv(cfg['data_path'])
    cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
    data = data[cols]
    sim_abs = os.path.abspath(cfg['sim_path']) if cfg.get('sim_path') else None
    np.random.seed(seed); random.seed(seed)

    print(f"  Generating {method} edges on {env} (seed={seed})...")
    t0 = time.time()

    if method == 'SAM':
        from benchmarks_new.sam_baseline import run_sam_baseline
        result = run_sam_baseline(data, simulation_path=sim_abs)
    elif method == 'GIES':
        from benchmarks_new.gies_comparison import run_gies
        result = run_gies(data, simulation_path=sim_abs)
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

    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        raw_edges = result.get('edges', [])
    elif isinstance(result, (list, set)):
        raw_edges = list(result)
    else:
        raw_edges = []

    edges = [(str(s), str(t)) for s, t in raw_edges]
    print(f"  Generated {len(edges)} edges ({time.time()-t0:.0f}s)")

    with open(cache_path, 'w') as f:
        json.dump(edges, f)
    print(f"  Cached: {cache_path}")

    return {(s, t) for s, t in edges}


def run_policy(method, env, edges):
    """Run policy engine with pre-generated edges."""
    cfg = SIM_CONFIGS[env]
    data = pd.read_csv(cfg['data_path'])
    cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
    data = data[cols]
    smap = load_scaling(cfg['scaling_path'])

    print(f"  Running policy engine ({len(edges)} edges)...")
    t0 = time.time()
    df = run_policy_for_seed(cfg, data, edges, smap)
    elapsed = time.time() - t0

    if df is not None:
        pg = df[df['method'] == 'PolicyGRID']
        if len(pg):
            for _, row in pg.iterrows():
                print(f"    ε={row['comfort_target']}: MO={row['MO']:.3f} kWh={row['kWh']:.2f} sat={row['satisfaction']:.1f}")
        out_path = f'{OUT_DIR}/{method}_{env}_policy.csv'
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({elapsed:.0f}s)")
    else:
        print(f"  FAILED ({elapsed:.0f}s)")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--env', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--edges-only', action='store_true', help='Only generate + cache edges, skip policy')
    args = parser.parse_args()

    methods = ['SAM', 'GIES', 'LLM'] if args.method == 'all' else [args.method]

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  {method} on {args.env}")
        print(f"{'='*60}")
        edges = generate_edges(method, args.env, args.seed)
        if not args.edges_only:
            run_policy(method, args.env, edges)
