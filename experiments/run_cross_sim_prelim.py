#!/usr/bin/env python3
"""
Cross-simulator preliminary results: discovery on ALL sims.
Lightweight — no policy/monitoring for small sims (no regimes).

Tests PolicyGRID discovery pipeline + benchmarks on:
  1. smart_building_rich (15 vars, 29 GT edges)
  2. open_window (8 vars, 13 GT edges)
  3. smart_room (5 vars, 6 GT edges)
  4. smart_room_hidden_vars (5 vars, 6 GT edges)
  5. smart_room_noise (5 vars, 6 GT edges)
"""

import sys, os, json, time, warnings, random
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
sys.path.insert(0, 'benchmarks_new')
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Sim configurations
SIMS = [
    {
        'name': 'smart_building_rich',
        'data': 'data_regen/smart_building_rich_processed.csv',
        'scaling': 'data_regen/smart_building_rich_processed_scaling.csv',
        'sim': 'js/smart_building_rich.js',
        'gt_key': 'smart_building_rich',
        'dataset_type': 'smart_building_rich',
        'actuator_vars': ['hvacpower', 'lightingpower'],
        'non_intervenable': [
            'outdoortemp', 'solarradiation', 'occupancy', 'windowposition',
            'pmv', 'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
    },
    {
        'name': 'open_window',
        'data': 'data_regen/challenge1_data_10k_processed.csv',
        'scaling': 'data_regen/challenge1_data_10k_processed_scaling.csv',
        'sim': 'js/open_window.js',
        'gt_key': 'open_window',
        'dataset_type': 'open_window',
        'actuator_vars': ['temperature', 'humidity'],
        'non_intervenable': [
            'windowopen', 'outdoortemperature', 'pmv',
            'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
    },
    {
        'name': 'smart_room',
        'data': 'data_regen/smart_room_processed.csv',
        'scaling': 'data_regen/smart_room_processed_scaling.csv',
        'sim': 'js/smart_room.js',
        'gt_key': 'smart_room',
        'dataset_type': 'smart_room',
        'actuator_vars': ['temperature', 'humidity', 'airquality'],
        'non_intervenable': [
            'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
    },
    {
        'name': 'smart_room_hidden_vars',
        'data': 'data_regen/smart_room_hidden_vars_processed.csv',
        'scaling': 'data_regen/smart_room_hidden_vars_processed_scaling.csv',
        'sim': 'js/smart_room_hidden_vars.js',
        'gt_key': 'hidden_vars',
        'dataset_type': 'smart_room',
        'actuator_vars': ['temperature', 'humidity', 'airquality'],
        'non_intervenable': [
            'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
    },
    {
        'name': 'smart_room_noise',
        'data': 'data_regen/smart_room_noise_processed.csv',
        'scaling': 'data_regen/smart_room_noise_processed_scaling.csv',
        'sim': 'js/smart_room_noise.js',
        'gt_key': 'smart_room_noise',
        'dataset_type': 'smart_room',
        'actuator_vars': ['temperature', 'humidity', 'airquality'],
        'non_intervenable': [
            'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
    },
]

API_KEY = 'YOUR_OPENAI_API_KEY'
SEED = 42


def load_gt(gt_key):
    with open('ground_truth_graphs.json') as f:
        gt = json.load(f)
    edges = gt[gt_key]['edges']
    return {(s.lower(), t.lower()) for s, t in edges}


def graph_metrics(discovered, gt_set):
    gt_lower = {(s.lower(), t.lower()) for s, t in gt_set}
    disc_lower = {(s.lower(), t.lower()) for s, t in discovered}
    tp = len(disc_lower & gt_lower)
    fp = len(disc_lower - gt_lower)
    fn = len(gt_lower - disc_lower)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    shd = fp + fn
    return {'f1': round(f1, 3), 'precision': round(precision, 3),
            'recall': round(recall, 3), 'shd': shd,
            'tp': tp, 'fp': fp, 'fn': fn, 'n_edges': len(disc_lower)}


def run_policygrid_discovery(data, sim_config):
    """Run full PolicyGRID discovery pipeline on a given sim."""
    from src.pipeline_cwm import create_cwm_pipeline

    sim_path = os.path.abspath(sim_config['sim'])
    random.seed(SEED)
    np.random.seed(SEED)

    pipeline = create_cwm_pipeline(
        csv_data=data,
        api_key=API_KEY,
        smart_room_path=sim_path,
        dataset_type=sim_config['dataset_type'],
        max_iterations=10,
        alpha=0.5, beta=0.5,
        effect_threshold=0.1,
        actuator_vars=sim_config['actuator_vars'],
        non_intervenable_vars=sim_config['non_intervenable'],
    )

    t0 = time.time()
    final_dag, metrics = pipeline.run_with_cwm()
    elapsed = time.time() - t0

    validated = set(pipeline.pipeline.validated_edges)
    method_dags = pipeline.pipeline.get_method_dags()
    obs_only = set()
    for G in method_dags.values():
        obs_only.update(G.edges())

    n_interventions = sum(
        len(r) for r in pipeline.pipeline.tester.intervention_results.values()
    )

    return {
        'validated': validated,
        'obs_only': obs_only,
        'n_interventions': n_interventions,
        'time_s': elapsed,
    }


def run_benchmarks(data):
    """Run benchmark discovery methods."""
    from neurips_final_experiments import run_benchmark_discovery
    return run_benchmark_discovery(data)


def main():
    t_total = time.time()

    out_dir = 'results/cross_sim_prelim'
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []

    for sim_config in SIMS:
        name = sim_config['name']
        logger.info("\n" + "=" * 70)
        logger.info(f"  SIMULATOR: {name}")
        logger.info("=" * 70)

        # Load data
        if not os.path.exists(sim_config['data']):
            logger.warning(f"  Data file not found: {sim_config['data']}")
            continue

        data = pd.read_csv(sim_config['data'])
        gt_set = load_gt(sim_config['gt_key'])

        logger.info(f"  Data: {len(data)} rows, {len(data.columns)} cols")
        logger.info(f"  GT: {len(gt_set)} edges")
        logger.info(f"  Variables: {list(data.columns)}")

        # Run PolicyGRID discovery
        logger.info(f"\n  Running PolicyGRID discovery...")
        try:
            disc = run_policygrid_discovery(data, sim_config)
            val_m = graph_metrics(disc['validated'], gt_set)
            obs_m = graph_metrics(disc['obs_only'], gt_set)

            logger.info(f"  PolicyGRID (validated): {len(disc['validated'])} edges, "
                        f"F1={val_m['f1']:.3f}, SHD={val_m['shd']}, "
                        f"P={val_m['precision']:.3f}, R={val_m['recall']:.3f}")
            logger.info(f"  PolicyGRID-O (obs-only): {len(disc['obs_only'])} edges, "
                        f"F1={obs_m['f1']:.3f}, SHD={obs_m['shd']}")
            logger.info(f"  Interventions: {disc['n_interventions']}, "
                        f"Time: {disc['time_s']:.0f}s")

            all_rows.append({
                'sim': name, 'method': 'PolicyGRID',
                **val_m, 'time_s': round(disc['time_s'], 1),
                'n_interventions': disc['n_interventions'],
            })
            all_rows.append({
                'sim': name, 'method': 'PolicyGRID-O',
                **obs_m, 'time_s': round(disc['time_s'], 1),
                'n_interventions': 0,
            })
        except Exception as e:
            logger.error(f"  PolicyGRID FAILED: {e}")
            import traceback; traceback.print_exc()

        # Run benchmarks
        logger.info(f"\n  Running benchmarks...")
        try:
            benchmark_edges = run_benchmarks(data)
            for bm_name, edges in benchmark_edges.items():
                m = graph_metrics(edges, gt_set)
                all_rows.append({
                    'sim': name, 'method': bm_name,
                    **m, 'time_s': 0, 'n_interventions': 0,
                })
                logger.info(f"  {bm_name:<20} {len(edges):>3} edges  "
                            f"F1={m['f1']:.3f}  SHD={m['shd']}")
        except Exception as e:
            logger.error(f"  Benchmarks FAILED: {e}")

    # ══════════════════════════════════════════════════════════════════
    # POLICY COMPARISON (sims with actuators + simulator)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n\n" + "=" * 70)
    logger.info("  CROSS-SIM POLICY COMPARISON")
    logger.info("=" * 70)

    from neurips_final_experiments import (
        make_grid_policy, make_pid_policy, run_policy_episode,
        get_scaling_map, phys_to_norm, norm_to_phys, step_simulator,
        MPCBaseline, CausalMBPO, DDPCBehavioral, DDPCNeural,
        HVAC_MAX_KW, LIGHT_MAX_KW, STEP_HOURS,
    )

    policy_rows = []
    comfort_targets = [0.5, 0.8, 1.0, 1.5, 2.0]
    n_runs = 3
    n_steps = 120

    for sim_config in SIMS:
        name = sim_config['name']

        if not os.path.exists(sim_config['data']):
            continue

        data = pd.read_csv(sim_config['data'])
        scaling = pd.read_csv(sim_config['scaling'])
        smap = {}
        feat_col = 'feature' if 'feature' in scaling.columns else 'column'
        min_col = 'data_min' if 'data_min' in scaling.columns else 'min'
        max_col = 'data_max' if 'data_max' in scaling.columns else 'max'
        for _, row in scaling.iterrows():
            smap[str(row[feat_col]).lower()] = (float(row[min_col]), float(row[max_col]))

        # Only run policy on sims with hvac/lighting actuators
        has_hvac = 'hvacpower' in smap or any('hvac' in a for a in sim_config['actuator_vars'])
        if name not in ('smart_building_rich',):
            # For small sims, just run PolicyGRID + PID + DDPC
            # They use temperature/humidity as actuators which is different
            logger.info(f"\n  {name}: skipping policy (no HVAC/lighting actuators)")
            continue

        logger.info(f"\n  {name}: running policy comparison...")

        # Get edges for this sim from the discovery results
        sim_disc_rows = [r for r in all_rows if r['sim'] == name and r['method'] == 'PolicyGRID']
        if not sim_disc_rows:
            logger.warning(f"  No discovery results for {name}, skipping policy")
            continue

        # Re-run discovery to get actual edge sets (we only stored metrics)
        logger.info(f"    Re-running discovery for edge sets...")
        try:
            disc = run_policygrid_discovery(data, sim_config)
            grid_val = disc['validated']
            grid_obs = disc['obs_only']
        except Exception as e:
            logger.error(f"    Discovery failed: {e}")
            continue

        # Fit baselines
        logger.info(f"    Fitting baselines...")
        mpc_lin = MPCBaseline(horizon=8, dynamics='linear', cem_pop=500, cem_iters=5)
        mpc_lin.fit(data)
        ddpc_beh = DDPCBehavioral(lag=5)
        ddpc_beh.fit(data)
        ddpc_nn = DDPCNeural()
        ddpc_nn.fit(data)
        pid_fn = make_pid_policy()

        for ct in comfort_targets:
            logger.info(f"    ε={ct}...")
            grid_fn = make_grid_policy(grid_val, data, comfort_target=ct)
            obs_fn = make_grid_policy(grid_obs, data, comfort_target=ct)

            # C-MBPO per ε
            cmbpo = CausalMBPO(edges=grid_val, n_real_episodes=30,
                                n_imagined_per_real=5, episode_len=60)
            cmbpo.set_comfort_target(ct)
            cmbpo.fit(data, smap=smap)

            mpc_lin.set_comfort_target(ct)
            ddpc_beh.set_comfort_target(ct)
            ddpc_nn.set_comfort_target(ct)

            arms = {
                'PolicyGRID': grid_fn,
                'PolicyGRID-O': obs_fn,
                'MPC_Linear': mpc_lin.get_action,
                'C-MBPO': cmbpo.get_action,
                'DDPC_Behavioral': ddpc_beh.get_action,
                'DDPC_Neural': ddpc_nn.get_action,
                'PID': pid_fn,
            }

            for arm_name, pol_fn in arms.items():
                if pol_fn is None:
                    continue
                for run_i in range(n_runs):
                    start_hour = 6.0 + run_i * 2
                    offset = [-2.0, 4.0, 10.0][run_i]
                    recs = run_policy_episode(pol_fn, smap, n_steps=n_steps,
                                              start_hour=start_hour,
                                              outdoor_offset=offset)
                    if recs:
                        avg_sat = np.mean([r['satisfaction'] for r in recs])
                        avg_eng = np.mean([r['energy'] for r in recs])
                        total_kwh = sum(r['kwh'] for r in recs)
                        total_dh = sum(r['dh'] for r in recs)
                        mo = 0.6 * (avg_sat / 100) + 0.4 * (1 - avg_eng / 100)
                        n_viol = sum(1 for r in recs if r['dh'] > 0)
                        policy_rows.append({
                            'sim': name, 'method': arm_name, 'run': run_i,
                            'comfort_target': ct,
                            'satisfaction': avg_sat, 'energy': avg_eng,
                            'kwh': round(total_kwh, 4),
                            'dh': round(total_dh, 4),
                            'mo_score': round(mo, 3),
                            'violation_rate': round(n_viol / len(recs), 3),
                        })

            # Log summary for this ε
            ct_df = pd.DataFrame([r for r in policy_rows
                                  if r['sim'] == name and r['comfort_target'] == ct])
            for arm in arms:
                sub = ct_df[ct_df['method'] == arm]
                if len(sub) > 0:
                    logger.info(f"      {arm:<20} MO={sub['mo_score'].mean():.3f} "
                                f"kWh={sub['kwh'].mean():.2f} "
                                f"DH={sub['dh'].mean():.3f}")

    if policy_rows:
        policy_df = pd.DataFrame(policy_rows)
        policy_df.to_csv(f'{out_dir}/cross_sim_policy.csv', index=False)

    # Save results
    df = pd.DataFrame(all_rows)
    df.to_csv(f'{out_dir}/cross_sim_discovery.csv', index=False)

    # Print summary table
    elapsed = time.time() - t_total
    logger.info("\n\n" + "=" * 70)
    logger.info("  CROSS-SIM DISCOVERY SUMMARY")
    logger.info("=" * 70)

    sims = df['sim'].unique()
    methods = ['PolicyGRID', 'PolicyGRID-O', 'PC', 'SAM', 'ICP',
               'NOTEARS-I', 'ABCD', 'IID']

    # Header
    header = f"{'Method':<20}"
    for s in sims:
        header += f" {s[:12]:>12}"
    logger.info(header)
    logger.info("-" * (20 + 13 * len(sims)))

    for m in methods:
        row = f"{m:<20}"
        for s in sims:
            sub = df[(df['sim'] == s) & (df['method'] == m)]
            if len(sub) > 0:
                row += f" F1={sub.iloc[0]['f1']:.3f}  "
            else:
                row += f" {'N/A':>12}"
        logger.info(row)

    logger.info(f"\n  Total time: {elapsed/60:.1f} minutes")
    logger.info(f"  Results saved to {out_dir}/cross_sim_discovery.csv")


if __name__ == '__main__':
    main()
