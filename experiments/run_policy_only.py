#!/usr/bin/env python
"""Policy-only evaluation using pre-made ground-truth causal graphs.

Skips discovery entirely — loads edges from ground_truth_graphs.json,
builds NeurIPSExperiments with FakePipeline, and runs policy episodes
+ Pareto plots for each simulator.

Usage:
    python run_policy_only.py                     # all 4 sims
    python run_policy_only.py --sim smart_room    # single sim
    python run_policy_only.py --seeds 3 --episodes 5
"""
import os, sys, json, csv, time, argparse, logging, random
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

from experiments_neurips import NeurIPSExperiments
from run_exp_suite_B_prelim import (
    build_summary, generate_b1_plots, compute_hypervolume_from_rows,
    DEFAULT_COMFORT_TARGETS, _compute_mo, _load_scaling_ranges,
    _BASE_DIR,
)

# ── Configs (same as run_exp_suite_B_prelim.py) ─────────────────────────────
SIM_CONFIGS = {
    'smart_room': {
        'sim_path': 'js/smart_room.js',
        'obs_data_path': 'data_regen/smart_room_processed.csv',
        'scaling_path': 'data_regen/smart_room_processed_scaling.csv',
        'dataset_type': 'open_window',
        'has_regime': False,
        'data_in_physical_units': False,
    },
    'smart_room_noise': {
        'sim_path': 'js/smart_room_noise.js',
        'obs_data_path': 'data_regen/smart_room_noise_processed.csv',
        'scaling_path': 'data_regen/smart_room_noise_processed_scaling.csv',
        'dataset_type': 'open_window',
        'has_regime': False,
        'data_in_physical_units': False,
    },
    'hidden_vars': {
        'sim_path': 'js/smart_room_hidden_vars.js',
        'obs_data_path': 'data_regen/hidden_vars_processed.csv',
        'scaling_path': 'data_regen/hidden_vars_processed_scaling.csv',
        'dataset_type': 'open_window',
        'has_regime': False,
        'data_in_physical_units': False,
    },
    'ashrae': {
        'sim_path': None,
        'obs_data_path': 'data/ashrae_data_processed.csv',
        'scaling_path': None,
        'dataset_type': 'ashrae',
        'has_regime': False,
        'data_in_physical_units': True,
    },
}


class FakePipeline:
    """Mimics the CWM pipeline interface using pre-defined edges."""

    def __init__(self, edges):
        self.validated_edges = set(edges)

    def export_for_policy_engine(self):
        return {'validated_edges': list(self.validated_edges)}

    def get_method_dags(self):
        import networkx as nx
        G = nx.DiGraph()
        G.add_edges_from(self.validated_edges)
        return {'fixed': G}


def load_ground_truth_graphs(path='ground_truth_graphs.json'):
    with open(path) as f:
        data = json.load(f)
    return {k: [tuple(e) for e in v['edges']] for k, v in data.items()}


def collect_intervention_data(exp, n_per_var=20):
    """Sweep the JS simulator to collect intervention samples for SEM fitting.

    Mimics what the CWM pipeline collects during discovery: for each action
    variable, sweep across its range at multiple base contexts.  Returns a
    DataFrame in the same scale as the observational data.
    """
    rows = []
    base_seeds = [0, 7, 21, 42, 99]
    action_vars = exp._action_vars            # e.g. ['Temperature', ...]
    # Outcome columns in obs data (0-1 scale for smart_room, % for open_window)
    sat_col = exp._cfg.get('sat_key', 'OverallSatisfaction')
    eng_col = exp._cfg.get('eng_key', 'EnergyConsumption')

    for s in base_seeds:
        base_state = exp._get_initial_state({'seed': s})
        base_action = dict(exp.ashrae_setpoints)

        for var in action_vars:
            for level in np.linspace(0.05, 0.95, n_per_var):
                action = dict(base_action)
                action[var] = level
                ns = exp._step_simulator(base_state, action, {'seed': s}, 0)

                row = {}
                for v in action_vars:
                    row[v] = ns.get(v, 0.5)
                # Sim returns 0-100 for sat/eng; obs data is 0-1 (normalised)
                if sat_col and sat_col in ns:
                    col_out = 'Satisfaction' if 'Satisfaction' in exp.dataset.columns else sat_col
                    row[col_out] = ns[sat_col] / 100.0
                if eng_col and eng_col in ns:
                    col_out = 'EnergyConsumption' if 'EnergyConsumption' in exp.dataset.columns else eng_col
                    row[col_out] = ns[eng_col] / 100.0
                rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"  Collected {len(df)} intervention samples")
    return df


def run_one_sim(sim_label, cfg, edges, obs_only_edges, n_seeds, n_episodes, comfort_targets):
    """Run policy episodes for one simulator with pre-defined edges."""
    print()
    print("=" * 72)
    print(f"  POLICY-ONLY: {sim_label}")
    print(f"  GRID edges: {len(edges)}  |  ObsOnly edges: {len(obs_only_edges)}")
    print(f"  Seeds: {n_seeds}  "
          f"|  Episodes: {n_episodes}  |  Comfort targets: {len(comfort_targets)}")
    print("=" * 72)

    obs_df = pd.read_csv(os.path.join(_BASE_DIR, cfg['obs_data_path']))
    scaling_ranges = _load_scaling_ranges(cfg)
    sim_path = (os.path.join(_BASE_DIR, cfg['sim_path'])
                if cfg['sim_path'] else None)

    pipeline = FakePipeline(edges)
    static_arms = {'Static_Baseline'}

    # Collect intervention data from the real simulator to enrich SEM fitting
    # (mirrors what the CWM pipeline does during discovery)
    combined_df = obs_df
    if sim_path:
        tmp_exp = NeurIPSExperiments(
            pipeline=pipeline, dataset=obs_df,
            smart_room_path=sim_path, comfort_target=1.0,
            obs_only_edges=obs_only_edges, h0_edges=None, h1_edges=None,
            dataset_type=cfg['dataset_type'],
            scaling_ranges=scaling_ranges,
            data_in_physical_units=cfg.get('data_in_physical_units', True),
        )
        int_data = collect_intervention_data(tmp_exp)
        combined_df = pd.concat([
            obs_df.assign(weight=1.0),
            int_data.assign(weight=2.0),
        ], ignore_index=True)
        print(f"  Combined dataset: {len(obs_df)} obs + {len(int_data)} intervention rows")

    all_rows = []
    t0 = time.time()

    for seed in range(n_seeds):
        random.seed(seed)
        np.random.seed(seed)
        print(f"\n  --- Seed {seed + 1}/{n_seeds} ---")

        exp = NeurIPSExperiments(
            pipeline=pipeline,
            dataset=combined_df,
            smart_room_path=sim_path,
            comfort_target=comfort_targets[0],
            obs_only_edges=obs_only_edges,
            h0_edges=None, h1_edges=None,
            dataset_type=cfg['dataset_type'],
            scaling_ranges=scaling_ranges,
            data_in_physical_units=cfg.get('data_in_physical_units', True),
            obs_only_dataset=obs_df,
        )
        exp._build_policy_arms()

        # Rename ASHRAE → ASHRAE-PID and add small comfort-target jitter
        # so it doesn't collapse to a single Pareto point
        if 'ASHRAE' in exp.policy_arms:
            del exp.policy_arms['ASHRAE']
        if 'DDPC_Neural' in exp.policy_arms:
            del exp.policy_arms['DDPC_Neural']
        base_sp = dict(exp.ashrae_setpoints)

        print(f"  Arms: {list(exp.policy_arms.keys())}")

        for ct_idx, ct in enumerate(comfort_targets):
            if ct_idx > 0:
                exp.set_comfort_target(ct)

            # ASHRAE-PID: static setpoints + tiny comfort-target shift
            jitter = (ct - 1.0) * 0.02  # ±0.02 around baseline
            pid_sp = {k: np.clip(v + jitter, 0.0, 1.0)
                      for k, v in base_sp.items()}
            exp.policy_arms['ASHRAE-PID'] = lambda state, t, _sp=pid_sp: _sp

            for arm_name in list(exp.policy_arms.keys()):
                # Static arms: run once, copy for other comfort targets
                if arm_name in static_arms and ct_idx > 0:
                    for ep_seed in range(n_episodes):
                        base = next(
                            r for r in all_rows
                            if r['arm'] == arm_name
                            and r['episode_seed'] == ep_seed
                            and r['discovery_seed'] == seed
                            and r['comfort_target'] == comfort_targets[0]
                        )
                        all_rows.append({**base, 'comfort_target': ct})
                    continue

                fn = exp.policy_arms[arm_name]
                t_arm = time.time()
                for ep_seed in range(n_episodes):
                    sat, eng, cv_mean_val, cvr, kwh, dh = \
                        exp._evaluate_policy_episode_with_cv(
                            fn, arm_name, ep_seed)
                    mo = _compute_mo(sat, eng, cfg['dataset_type'])

                    all_rows.append({
                        'arm': arm_name,
                        'discovery_seed': seed,
                        'episode_seed': ep_seed,
                        'comfort_target': ct,
                        'satisfaction': sat,
                        'energy': eng,
                        'mo_score': mo,
                        'cv_mean': cv_mean_val,
                        'cvr': cvr,
                        'kwh': kwh,
                        'dh': dh,
                        'intervention_count': exp._intervention_count
                            if hasattr(exp, '_intervention_count') else 0,
                    })

                wall = time.time() - t_arm
                logger.info(f"    {arm_name} (ε={ct}): "
                            f"{n_episodes} ep in {wall:.1f}s")

    elapsed = time.time() - t0
    print(f"\n  {sim_label} done in {elapsed:.1f}s "
          f"({len(all_rows)} rows)")

    return all_rows, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='Policy-only evaluation with ground-truth causal graphs')
    parser.add_argument('--sim', type=str, default=None,
                        choices=list(SIM_CONFIGS.keys()),
                        help='Single simulator (default: run all)')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--comfort-targets', type=str, default=None,
                        help=f'Comma-separated (default: {DEFAULT_COMFORT_TARGETS})')
    parser.add_argument('--graphs', type=str, default='ground_truth_graphs.json',
                        help='Path to ground-truth graph JSON')
    parser.add_argument('--obs-graphs', type=str, default='obs_only_graphs.json',
                        help='Path to obs-only graph JSON (for GRID_ObsOnly)')
    args = parser.parse_args()

    comfort_targets = (
        [float(x) for x in args.comfort_targets.split(',')]
        if args.comfort_targets
        else DEFAULT_COMFORT_TARGETS
    )

    graphs = load_ground_truth_graphs(args.graphs)
    obs_graphs = load_ground_truth_graphs(args.obs_graphs)
    out_dir = 'neurips_results'
    os.makedirs(out_dir, exist_ok=True)

    sims_to_run = [args.sim] if args.sim else list(SIM_CONFIGS.keys())

    grand_t0 = time.time()
    all_raw = []

    for sim_label in sims_to_run:
        if sim_label not in graphs:
            print(f"  Skipping {sim_label}: no ground-truth graph")
            continue
        cfg = SIM_CONFIGS[sim_label]
        if not os.path.exists(os.path.join(_BASE_DIR, cfg['obs_data_path'])):
            print(f"  Skipping {sim_label}: data not found")
            continue

        edges = graphs[sim_label]
        oo_edges = obs_graphs.get(sim_label, edges)  # fallback to GT if missing
        rows, elapsed = run_one_sim(
            sim_label, cfg, edges, oo_edges,
            args.seeds, args.episodes, comfort_targets)

        if not rows:
            continue

        raw_df = pd.DataFrame(rows)
        raw_df['simulator'] = sim_label
        all_raw.append(raw_df)

        # Save CSV
        raw_df.to_csv(f'{out_dir}/b1_{sim_label}_raw.csv', index=False)

        summary_df = build_summary(raw_df, cfg['dataset_type'])
        summary_df['simulator'] = sim_label
        summary_df.to_csv(f'{out_dir}/b1_{sim_label}_summary.csv', index=False)

        # Print summary
        print(f"\n{'=' * 72}")
        print(f"  RESULTS -- {sim_label} [{elapsed:.1f}s]")
        print(f"{'=' * 72}")
        print(f"{'Arm':25s} {'kWh':>8s} {'DH':>8s} {'Sat%':>8s} "
              f"{'Eng%':>8s} {'MO':>8s} {'HV':>8s}")
        print("-" * 72)
        for _, r in summary_df.iterrows():
            print(f"{r['arm']:25s} {r['kwh_mean']:8.1f} {r['dh_mean']:8.1f} "
                  f"{r['sat_mean']:8.1f} {r['eng_mean']:8.1f} "
                  f"{r['mo_mean']:8.4f} {r['hypervolume']:8.3f}")

        # HV details
        hv = compute_hypervolume_from_rows(rows)
        print(f"\n  Combined HV = {hv['combined']:.4f}")
        print(f"  Frontier points: {len(hv['frontier'])}")
        for arm, val in sorted(hv['per_arm'].items(), key=lambda x: -x[1]):
            mk, md = hv['arm_means'][arm]
            print(f"    {arm:25s}  HV={val:.4f}  ({mk:.1f} kWh, {md:.1f} DH)")

        # Generate plots
        generate_b1_plots(raw_df, summary_df, sim_label, output_dir=out_dir)
        print(f"\n  Plots saved to {out_dir}/b1_prelim_{sim_label}_*.png")

    # Combined CSV if multiple sims
    if len(all_raw) > 1:
        combined = pd.concat(all_raw, ignore_index=True)
        combined.to_csv(f'{out_dir}/b1_policy_only_all_raw.csv', index=False)
        print(f"\n  Combined: {out_dir}/b1_policy_only_all_raw.csv")

    grand_wall = time.time() - grand_t0
    print(f"\n{'=' * 72}")
    print(f"  TOTAL: {grand_wall:.1f}s ({grand_wall/3600:.2f}h)")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
