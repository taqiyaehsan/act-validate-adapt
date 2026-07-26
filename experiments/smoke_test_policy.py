#!/usr/bin/env python
"""
Policy-only smoke test — skip discovery, use fixed edges from previous run.
Tests that GRID (with corrected normalization) outperforms baselines on all metrics.
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

from experiments_neurips import NeurIPSExperiments

# ── Fixed edges from neurips_results_old/app.log (seed 0, 10 validated) ──────

VALIDATED_EDGES = [
    ('airquality', 'satisfaction'),
    ('temperature', 'satisfaction'),
    ('temperature', 'airquality'),
    ('temperature', 'pmv'),
    ('airquality', 'energyconsumption'),
    ('temperature', 'energyconsumption'),
    ('humidity', 'airquality'),
    ('humidity', 'energyconsumption'),
    ('humidity', 'satisfaction'),
    ('temperature', 'humidity'),
]

# Obs-only edges = superset (union of PC + LLM + VARLiNGAM before validation)
# Use validated as a conservative proxy
OBS_ONLY_EDGES = VALIDATED_EDGES

# ── Simulator configs (matching run_exp_suite_B_prelim.py) ───────────────────

SIM_CONFIGS = {
    'open_window': {
        'sim_path': 'js/open_window.js',
        'obs_data_path': 'data_regen/challenge1_data_10k_processed.csv',
        'scaling_path': 'data_regen/challenge1_data_10k_processed_scaling.csv',
        'dataset_type': 'open_window',
        'data_in_physical_units': True,
    },
    'smart_room': {
        'sim_path': 'js/smart_room.js',
        'obs_data_path': 'data_regen/smart_room_processed.csv',
        'scaling_path': 'data_regen/smart_room_processed_scaling.csv',
        'dataset_type': 'open_window',  # same variable schema
        'data_in_physical_units': False,
    },
}

COMFORT_TARGETS = [0.5, 1.0, 2.0]
N_EPISODES = 5


def load_scaling(path):
    if not path or not os.path.exists(path):
        return None
    sc = pd.read_csv(path).set_index('feature')
    return {row: {'min': sc.loc[row, 'data_min'], 'max': sc.loc[row, 'data_max']}
            for row in sc.index}


class FakePipeline:
    """Minimal mock so NeurIPSExperiments can call export_for_policy_engine()."""
    def __init__(self, edges):
        self.validated_edges = set(edges)

    def export_for_policy_engine(self):
        return {'validated_edges': list(self.validated_edges)}

    def get_method_dags(self):
        import networkx as nx
        G = nx.DiGraph()
        G.add_edges_from(self.validated_edges)
        return {'fixed': G}


def run_sim(sim_label, cfg):
    print(f"\n{'=' * 70}")
    print(f"  {sim_label}  (policy-only, fixed edges)")
    print(f"{'=' * 70}")

    obs_df = pd.read_csv(cfg['obs_data_path'])
    scaling = load_scaling(cfg.get('scaling_path'))
    sim_path = cfg['sim_path'] if cfg['sim_path'] and os.path.exists(cfg['sim_path']) else None

    fake_pipe = FakePipeline(VALIDATED_EDGES)

    exp = NeurIPSExperiments(
        pipeline=fake_pipe,
        dataset=obs_df,
        smart_room_path=sim_path,
        comfort_target=COMFORT_TARGETS[0],
        obs_only_edges=OBS_ONLY_EDGES,
        h0_edges=None,
        h1_edges=None,
        dataset_type=cfg['dataset_type'],
        scaling_ranges=scaling,
        data_in_physical_units=cfg['data_in_physical_units'],
    )
    exp._build_policy_arms()

    # Remove PETS (slow)
    if 'DDPC_PETS' in exp.policy_arms:
        del exp.policy_arms['DDPC_PETS']

    print(f"  Arms: {list(exp.policy_arms.keys())}")
    print(f"  Comfort targets: {COMFORT_TARGETS}")
    print(f"  Episodes per arm per ε: {N_EPISODES}")
    print()

    rows = []
    static_arms = {'ASHRAE', 'Static_Baseline'}

    for ct_idx, ct in enumerate(COMFORT_TARGETS):
        if ct_idx > 0:
            exp.set_comfort_target(ct)
            if 'DDPC_PETS' in exp.policy_arms:
                del exp.policy_arms['DDPC_PETS']

        for arm_name, fn in exp.policy_arms.items():
            if arm_name in static_arms and ct_idx > 0:
                for ep in range(N_EPISODES):
                    base = next(r for r in rows
                                if r['arm'] == arm_name and r['ep'] == ep
                                and r['ct'] == COMFORT_TARGETS[0])
                    rows.append({**base, 'ct': ct})
                continue

            t0 = time.time()
            for ep in range(N_EPISODES):
                sat, eng, cv, cvr, kwh, dh = \
                    exp._evaluate_policy_episode_with_cv(fn, arm_name, ep)
                mo = 0.6 * (sat / 100) + 0.4 * (1 - eng / 100)
                rows.append({
                    'arm': arm_name, 'ct': ct, 'ep': ep,
                    'sat': sat, 'eng': eng, 'mo': mo,
                    'cv': cv, 'cvr': cvr, 'kwh': kwh, 'dh': dh,
                })
            wall = time.time() - t0
            logger.info(f"  {arm_name} ε={ct}: {N_EPISODES} ep in {wall:.1f}s")

    df = pd.DataFrame(rows)
    return df


def print_results(df, sim_label):
    print(f"\n{'=' * 70}")
    print(f"  RESULTS — {sim_label}")
    print(f"{'=' * 70}")

    # Per-arm summary (across all ε)
    summary = (df.groupby('arm')
               .agg(sat_mean=('sat', 'mean'), sat_std=('sat', 'std'),
                    eng_mean=('eng', 'mean'), eng_std=('eng', 'std'),
                    mo_mean=('mo', 'mean'), mo_std=('mo', 'std'),
                    kwh_mean=('kwh', 'mean'), dh_mean=('dh', 'mean'))
               .sort_values('mo_mean', ascending=False))

    print(f"\n{'Arm':25s} {'Sat%':>10s} {'Eng%':>10s} "
          f"{'MO':>8s} {'kWh':>8s} {'DH':>8s}")
    print(f"{'-' * 25} {'-' * 10} {'-' * 10} "
          f"{'-' * 8} {'-' * 8} {'-' * 8}")
    for arm, r in summary.iterrows():
        print(f"{arm:25s} "
              f"{r['sat_mean']:5.1f}±{r['sat_std']:4.1f} "
              f"{r['eng_mean']:5.1f}±{r['eng_std']:4.1f} "
              f"{r['mo_mean']:7.4f} "
              f"{r['kwh_mean']:7.1f} "
              f"{r['dh_mean']:7.1f}")

    # Per comfort-target breakdown
    print(f"\n  Per comfort target (ε):")
    for ct in sorted(df['ct'].unique()):
        print(f"\n  ε = {ct}:")
        sub = df[df['ct'] == ct]
        ct_sum = (sub.groupby('arm')
                  .agg(sat=('sat', 'mean'), eng=('eng', 'mean'),
                       mo=('mo', 'mean'), kwh=('kwh', 'mean'),
                       dh=('dh', 'mean'))
                  .sort_values('mo', ascending=False))
        for arm, r in ct_sum.iterrows():
            print(f"    {arm:25s} sat={r['sat']:5.1f}  eng={r['eng']:5.1f}  "
                  f"MO={r['mo']:.4f}  kWh={r['kwh']:.1f}  DH={r['dh']:.1f}")

    # Pareto: hypervolume per arm (using kwh, dh — lower is better for both)
    print(f"\n  Pareto (kWh vs DH) — hypervolume (higher = better front):")
    # Reference point = worst across all arms + 10%
    kwh_ref = df['kwh'].max() * 1.1
    dh_ref = df['dh'].max() * 1.1
    if kwh_ref < 1e-6:
        kwh_ref = 1.0
    if dh_ref < 1e-6:
        dh_ref = 1.0

    for arm in summary.index:
        arm_df = df[df['arm'] == arm]
        # Mean per comfort target
        means = (arm_df.groupby('ct')
                 .agg(kwh=('kwh', 'mean'), dh=('dh', 'mean'))
                 .values)
        # 2D hypervolume: area dominated by points below reference
        # Sort by kwh ascending
        pts = sorted(means.tolist(), key=lambda p: p[0])
        hv = 0.0
        prev_kwh = 0.0
        for kwh_p, dh_p in pts:
            if kwh_p < kwh_ref and dh_p < dh_ref:
                hv += (kwh_ref - kwh_p) * (dh_ref - dh_p)
        print(f"    {arm:25s} HV = {hv:.1f}")

    # Key comparisons
    print(f"\n  Key comparisons (GRID vs baselines):")
    print(f"  {'-' * 60}")
    arms_present = set(df['arm'].unique())
    baseline = 'ASHRAE' if 'ASHRAE' in arms_present else 'Static_Baseline'

    comparisons = []
    if baseline in arms_present:
        comparisons.append(('GRID', baseline, 'GRID vs static baseline'))
    if 'GRID_ObsOnly' in arms_present:
        comparisons.append(('GRID', 'GRID_ObsOnly', 'Validation helps'))
    for ddpc in ['DDPC_Neural', 'DDPC_Behavioral', 'DDPC_Subspace']:
        if ddpc in arms_present:
            comparisons.append(('GRID', ddpc, f'GRID vs {ddpc}'))

    from scipy import stats
    for arm_a, arm_b, label in comparisons:
        a_mo = df[df['arm'] == arm_a]['mo'].values
        b_mo = df[df['arm'] == arm_b]['mo'].values
        diff = a_mo.mean() - b_mo.mean()
        if len(a_mo) > 1 and len(b_mo) > 1:
            _, p = stats.mannwhitneyu(a_mo, b_mo, alternative='two-sided')
        else:
            p = 1.0
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        sign = '+' if diff > 0 else ''
        print(f"  {label:42s} {sign}{diff:+.4f} MO  p={p:.3f} {sig}")

    print(f"  {'-' * 60}")


def main():
    all_dfs = []
    for sim_label, cfg in SIM_CONFIGS.items():
        if not os.path.exists(cfg['obs_data_path']):
            print(f"  Skipping {sim_label}: {cfg['obs_data_path']} not found")
            continue
        df = run_sim(sim_label, cfg)
        df['simulator'] = sim_label
        print_results(df, sim_label)
        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = 'neurips_results/smoke_test_policy_results.csv'
        os.makedirs('neurips_results', exist_ok=True)
        combined.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
