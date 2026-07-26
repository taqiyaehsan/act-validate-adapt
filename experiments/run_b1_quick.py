#!/usr/bin/env python3
"""
Quick B1 experiment — 8 arms × 5 seeds on open_window.js
=========================================================
Produces a comparison table and saves CSV.

Usage:
    python run_b1_quick.py                  # JS simulator
    python run_b1_quick.py --analytical     # fast analytical fallback (no node)
    python run_b1_quick.py --seeds 10       # more seeds for tighter CIs
"""
import argparse, time, warnings, logging, os
import numpy as np
import pandas as pd

# Suppress macOS Accelerate BLAS warnings
for _m in ('divide by zero encountered in matmul',
           'overflow encountered in matmul',
           'invalid value encountered in matmul'):
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=f'.*{_m}.*')

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('run_b1_quick')
logger.setLevel(logging.INFO)

from experiments_neurips import NeurIPSExperiments
import networkx as nx


# ── Ground-truth edges ───────────────────────────────────────────────────────
GT_EDGES = {
    ('Temperature', 'EnergyConsumption'),
    ('Temperature', 'Satisfaction'),
    ('Humidity', 'EnergyConsumption'),
    ('Humidity', 'Satisfaction'),
    ('AirQuality', 'EnergyConsumption'),
    ('AirQuality', 'Satisfaction'),
}

H1_EDGES = GT_EDGES | {('OutdoorTemperature', 'Temperature')}


class MockPipeline:
    """Supplies the validated DAG so the experiment can build policy engines."""
    def __init__(self, validated_edges, obs_edges):
        self._validated = validated_edges
        self._obs = obs_edges

    def export_for_policy_engine(self):
        return {
            'validated_edges': self._validated,
            'final_dag': {}, 'intervention_results': {},
            'edge_confidence': {}, 'metadata': {},
        }

    def get_method_dags(self):
        G = nx.DiGraph()
        G.add_edges_from(self._obs)
        return {'union': G}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analytical', action='store_true',
                        help='Use analytical fallback instead of JS simulator')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds per arm (default 5)')
    parser.add_argument('--skip-ddpc', action='store_true',
                        help='Skip DDPC baselines (much faster)')
    args = parser.parse_args()

    n_seeds = args.seeds
    sim_path = None if args.analytical else 'open_window.js'
    sim_label = 'analytical' if args.analytical else 'open_window.js'

    logger.info(f"B1 quick experiment: {n_seeds} seeds, simulator={sim_label}")

    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv('data/challenge1_data_10k_processed.csv')
    logger.info(f"Dataset: {len(df)} rows, columns={list(df.columns)}")

    # ObsOnly edges: GT + spurious PMV edges (simulating unvalidated discovery)
    obs_edges = GT_EDGES | {
        ('PMV', 'EnergyConsumption'),
        ('PMV', 'Satisfaction'),
    }

    pipeline = MockPipeline(GT_EDGES, obs_edges)

    # ── Build experiment ─────────────────────────────────────────────────────
    exp = NeurIPSExperiments(
        pipeline=pipeline,
        dataset=df,
        smart_room_path=sim_path,
        comfort_target=0.8,
        h0_edges=GT_EDGES,
        h1_edges=H1_EDGES,
    )

    t0 = time.time()
    exp._build_policy_arms()

    if args.skip_ddpc:
        for k in list(exp.policy_arms.keys()):
            if 'DDPC' in k:
                del exp.policy_arms[k]

    # Always skip PETS (very slow: ensemble + CEM per step)
    if 'DDPC_PETS' in exp.policy_arms:
        del exp.policy_arms['DDPC_PETS']

    arms = list(exp.policy_arms.keys())
    logger.info(f"Arms: {arms}")

    # ── Run episodes ─────────────────────────────────────────────────────────
    rows = []
    for arm_name in arms:
        fn = exp.policy_arms[arm_name]
        logger.info(f"  Running {arm_name} ({n_seeds} seeds) ...")
        for seed in range(n_seeds):
            sat, eng = exp._evaluate_policy_episode(fn, arm_name, seed)
            rows.append({
                'arm': arm_name, 'seed': seed,
                'satisfaction': sat, 'energy': eng,
            })
            if (seed + 1) % 5 == 0:
                logger.info(f"    {arm_name}: {seed+1}/{n_seeds}")

    elapsed = time.time() - t0

    # ── Aggregate ────────────────────────────────────────────────────────────
    raw = pd.DataFrame(rows)

    summary = raw.groupby('arm').agg(
        sat_mean=('satisfaction', 'mean'),
        sat_std=('satisfaction', 'std'),
        eng_mean=('energy', 'mean'),
        eng_std=('energy', 'std'),
        n=('seed', 'count'),
    ).reset_index()

    # Multi-objective score: 0.6 * (sat/100) + 0.4 * (1 - eng/100)
    summary['mo_score'] = (0.6 * summary['sat_mean'] / 100
                           + 0.4 * (1 - summary['eng_mean'] / 100))

    summary = summary.sort_values('mo_score', ascending=False)

    # ── Print table ──────────────────────────────────────────────────────────
    print()
    print(f"{'='*72}")
    print(f"  B1 Quick Experiment — {n_seeds} seeds × {len(arms)} arms "
          f"({sim_label})  [{elapsed:.1f}s]")
    print(f"{'='*72}")
    print(f"{'Arm':25s} {'Sat%':>8s} {'Eng%':>8s} {'MO-Score':>9s}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*9}")
    for _, r in summary.iterrows():
        print(f"{r['arm']:25s} "
              f"{r['sat_mean']:5.1f}±{r['sat_std']:4.1f} "
              f"{r['eng_mean']:5.1f}±{r['eng_std']:4.1f} "
              f"{r['mo_score']:9.4f}")
    print(f"{'='*72}")
    print()

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs('neurips_results', exist_ok=True)
    raw.to_csv('neurips_results/b1_quick_raw.csv', index=False)
    summary.to_csv('neurips_results/b1_quick_summary.csv', index=False)
    logger.info("Saved to neurips_results/b1_quick_*.csv")


if __name__ == '__main__':
    main()
