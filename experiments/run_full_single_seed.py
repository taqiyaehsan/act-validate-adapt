#!/usr/bin/env python3
"""
Full single-seed NeurIPS run: discovery → monitoring → policy → ablation
→ M3(A) → A2 → N2 → M5 → post-hoc analyses.

Sim: smart_building_rich (15 vars, 28 GT edges, 2 hidden regimes)
Seed: 42
Output: results/neurips_final/

Usage:
    python run_full_single_seed.py
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

from neurips_final_experiments import (
    # Data
    load_data, get_scaling_map, graph_metrics,
    DATA_PATH, SCALING_PATH, SIM_PATH, GT_PATH,
    ALL_VARS, LATENT_VARS, SENSOR_VARS, OBSERVABLE_VARS,
    DEFAULT_COMFORT_TARGETS,
    HVAC_MAX_KW, LIGHT_MAX_KW, STEP_HOURS,
    # Discovery
    run_full_discovery, run_benchmark_discovery,
    # Monitoring
    construct_factored_monitors, run_monitoring, compute_monitoring_metrics,
    compute_sensor_weights_from_graph,
    run_monitoring_active, collect_sim_data,
    # Policy
    make_grid_policy, make_regime_aware_policy, make_pid_policy,
    run_policy_episode,
    MPCBaseline, CausalMBPO,
    DDPCBehavioral, DDPCSubspace, DDPCNeural,
    # Ablation
    run_sensor_ablation,
    # New experiments
    run_exp_m3a_attribution,
    run_exp_a2_regime_recovery,
    run_exp_n2_wrong_prior,
    enhance_compute_tracker,
    # Viz
    compute_pareto_hypervolume,
)

OUT_DIR = 'results/neurips_final'
API_KEY = 'YOUR_OPENAI_API_KEY'

SEED = 42
N_RUNS = 3
N_STEPS = 120


def main():
    t_total = time.time()

    os.makedirs(f'{OUT_DIR}/exp1_benchmarks', exist_ok=True)
    os.makedirs(f'{OUT_DIR}/exp2_ablation', exist_ok=True)
    os.makedirs(f'{OUT_DIR}/exp3_ablation_m3', exist_ok=True)
    os.makedirs(f'{OUT_DIR}/exp4_appendix', exist_ok=True)
    os.makedirs(f'{OUT_DIR}/exp5_stress', exist_ok=True)
    os.makedirs(f'{OUT_DIR}/posthoc', exist_ok=True)

    from compute_tracker import ComputeTracker
    tracker = ComputeTracker()

    data, scaling, gt_set = load_data()
    smap = get_scaling_map(scaling)
    logger.info(f"Data: {len(data)} rows, {len(data.columns)} cols, GT: {len(gt_set)} edges")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 0: DISCOVERY (1 seed)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 0: POLICYGRID DISCOVERY (seed=%d)", SEED)
    logger.info("=" * 70)

    tracker.start_phase('discovery')
    random.seed(SEED)
    np.random.seed(SEED)
    disc_result = run_full_discovery(data, seed=SEED)
    tracker.end_phase()

    grid_val = disc_result['validated_edges']
    grid_obs = disc_result['obs_only_edges']
    n_interventions = disc_result['intervention_count']

    logger.info(f"  Validated: {len(grid_val)} edges")
    logger.info(f"  Obs-only: {len(grid_obs)} edges")
    logger.info(f"  Interventions: {n_interventions}")

    # Benchmark discovery
    tracker.start_phase('benchmark_discovery')
    benchmark_edges = run_benchmark_discovery(data)
    tracker.end_phase()

    # Compute metrics
    discovery_rows = []
    for name, edges in [('PolicyGRID', grid_val), ('PolicyGRID-O', grid_obs)]:
        m = graph_metrics(edges, gt_set)
        m['method'] = name
        m['f1_std'] = m['shd_std'] = m['precision_std'] = m['recall_std'] = 0.0
        discovery_rows.append(m)
        logger.info(f"  {name}: F1={m['f1']:.3f}, SHD={m['shd']}")

    for name, edges in benchmark_edges.items():
        m = graph_metrics(edges, gt_set)
        m['method'] = name
        m['f1_std'] = m['shd_std'] = m['precision_std'] = m['recall_std'] = 0.0
        discovery_rows.append(m)
        logger.info(f"  {name}: F1={m['f1']:.3f}, SHD={m['shd']}")

    discovery_df = pd.DataFrame(discovery_rows)
    discovery_df.to_csv(f'{OUT_DIR}/exp1_benchmarks/discovery_metrics.csv', index=False)

    # Save edges
    edge_cache = {
        'seeds': [SEED],
        'consensus_validated': [list(e) for e in grid_val],
        'consensus_obs_only': [list(e) for e in grid_obs],
        'per_seed_validated': [[list(e) for e in grid_val]],
        'per_seed_obs_only': [[list(e) for e in grid_obs]],
    }
    with open(f'{OUT_DIR}/discovered_edges.json', 'w') as f:
        json.dump(edge_cache, f, indent=2)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1B: MONITORING
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 1B: MONITORING BENCHMARKS")
    logger.info("=" * 70)

    tracker.start_phase('monitoring')

    # Build monitors for each method
    method_monitors = {}
    for name, edges in [('PolicyGRID', grid_val), ('PolicyGRID-O', grid_obs)]:
        method_monitors[name] = construct_factored_monitors(edges, data)
    for name, edges in benchmark_edges.items():
        method_monitors[name] = construct_factored_monitors(edges, data)

    # Collect test data
    logger.info("  Collecting 4-day test episode (5760 steps)...")
    sim_records = collect_sim_data(smap, duration_steps=5760)
    all_sim_records = [sim_records]

    # Run monitoring for each method
    monitoring_rows = []
    all_monitoring_records = {}
    for name, monitors in method_monitors.items():
        if monitors is None:
            continue
        recs = run_monitoring(monitors, sim_records, sensor_vars=SENSOR_VARS)
        m = compute_monitoring_metrics(recs)
        monitoring_rows.append({'method': name, **m,
                                'occ_std': 0.0, 'win_std': 0.0, 'combined_std': 0.0})
        all_monitoring_records[name] = recs
        logger.info(f"  {name}: Occ={m['occ_accuracy']:.1f}% "
                    f"Win={m['win_accuracy']:.1f}% Comb={m['combined_accuracy']:.1f}%")

    monitoring_df = pd.DataFrame(monitoring_rows)
    monitoring_df.to_csv(f'{OUT_DIR}/exp1_benchmarks/monitoring_metrics.csv', index=False)

    # Active probing (PolicyGRID only)
    logger.info("\n  Active probing (PolicyGRID)...")
    try:
        from src.gpt_client import GPTClient
        gpt = GPTClient(api_key=API_KEY)
    except Exception:
        gpt = None

    grid_monitors = method_monitors.get('PolicyGRID')
    if grid_monitors:
        active_recs, probe_log = run_monitoring_active(
            grid_monitors, sim_records, smap, sensor_vars=SENSOR_VARS,
            gpt_client=gpt, probe_cooldown=60, max_probes=100)
        a_met = compute_monitoring_metrics(active_recs)
        p_met = compute_monitoring_metrics(
            all_monitoring_records.get('PolicyGRID', []))
        logger.info(f"  Passive: {p_met['combined_accuracy']:.1f}%")
        logger.info(f"  Active:  {a_met['combined_accuracy']:.1f}% "
                    f"(+{a_met['combined_accuracy'] - p_met['combined_accuracy']:.1f}pp, "
                    f"{len(probe_log)} probes)")

    tracker.end_phase()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1C: POLICY COMPARISON (full Pareto sweep)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 1C: POLICY COMPARISON (10 arms × 5ε × 3 runs)")
    logger.info("=" * 70)

    tracker.start_phase('policy')

    comfort_targets = DEFAULT_COMFORT_TARGETS  # [0.5, 0.8, 1.0, 1.5, 2.0]

    # Pre-fit methods that don't depend on ε
    logger.info("  Fitting baselines...")
    mpc_lin = MPCBaseline(horizon=8, dynamics='linear', cem_pop=500, cem_iters=5)
    mpc_lin.fit(data)
    mpc_nn = MPCBaseline(horizon=8, dynamics='neural', cem_pop=500, cem_iters=5,
                          n_ensemble=5)
    mpc_nn.fit(data)
    ddpc_beh = DDPCBehavioral(lag=5); ddpc_beh.fit(data)
    ddpc_sub = DDPCSubspace(lag=3); ddpc_sub.fit(data)
    ddpc_nn = DDPCNeural(); ddpc_nn.fit(data)
    pid_fn = make_pid_policy()

    policy_rows = []

    for ct_idx, ct in enumerate(comfort_targets):
        logger.info(f"\n  ε={ct}:")

        # Build per-ε policies
        grid_fn = make_grid_policy(grid_val, data, comfort_target=ct)
        obs_fn = make_grid_policy(grid_obs, data, comfort_target=ct)
        regime_fn = make_regime_aware_policy(
            grid_val, data, grid_monitors, comfort_target=ct)

        # C-MBPO: train fresh per ε
        cmbpo = CausalMBPO(edges=grid_val, n_real_episodes=30,
                            n_imagined_per_real=5, episode_len=60)
        cmbpo.set_comfort_target(ct)
        cmbpo.fit(data, smap=smap)

        # Set ε on DDPC/MPC
        for ctrl in [mpc_lin, mpc_nn, ddpc_beh, ddpc_sub, ddpc_nn]:
            ctrl.set_comfort_target(ct)

        arms = {
            'PolicyGRID': grid_fn,
            'PolicyGRID-O': obs_fn,
            'PolicyGRID-R': regime_fn,
            'MPC_Linear': mpc_lin.get_action,
            'MPC_Neural': mpc_nn.get_action,
            'C-MBPO': cmbpo.get_action,
            'DDPC_Behavioral': ddpc_beh.get_action,
            'DDPC_Subspace': ddpc_sub.get_action,
            'DDPC_Neural': ddpc_nn.get_action,
            'PID': pid_fn,
        }

        for arm_name, pol_fn in arms.items():
            if pol_fn is None:
                continue
            for run_i in range(N_RUNS):
                start_hour = 6.0 + run_i * 2
                offset = [-2.0, 4.0, 10.0][run_i]
                recs = run_policy_episode(pol_fn, smap, n_steps=N_STEPS,
                                          start_hour=start_hour,
                                          outdoor_offset=offset)
                if recs:
                    avg_sat = np.mean([r['satisfaction'] for r in recs])
                    avg_eng = np.mean([r['energy'] for r in recs])
                    total_kwh = sum(r['kwh'] for r in recs)
                    total_dh = sum(r['dh'] for r in recs)
                    mo = 0.6 * (avg_sat / 100) + 0.4 * (1 - avg_eng / 100)
                    n_violations = sum(1 for r in recs if r['dh'] > 0)
                    violation_rate = n_violations / len(recs)
                    policy_rows.append({
                        'method': arm_name, 'run': run_i,
                        'comfort_target': ct,
                        'satisfaction': avg_sat, 'energy': avg_eng,
                        'kwh': round(total_kwh, 4),
                        'dh': round(total_dh, 4),
                        'mo_score': round(mo, 3),
                        'violation_rate': round(violation_rate, 3),
                        'n_steps': len(recs),
                    })

        # Log summary
        ct_df = pd.DataFrame([r for r in policy_rows if r['comfort_target'] == ct])
        for name in arms:
            sub = ct_df[ct_df['method'] == name]
            if len(sub) > 0:
                logger.info(f"    {name:<20} MO={sub['mo_score'].mean():.3f} "
                            f"kWh={sub['kwh'].mean():.2f} "
                            f"DH={sub['dh'].mean():.3f} "
                            f"Viol={sub['violation_rate'].mean()*100:.1f}%")

    policy_df = pd.DataFrame(policy_rows)
    policy_df.to_csv(f'{OUT_DIR}/exp1_benchmarks/policy_metrics.csv', index=False)

    tracker.end_phase()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: SENSOR ABLATION
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 2: SENSOR ABLATION")
    logger.info("=" * 70)

    tracker.start_phase('ablation')
    ablation_rows = []
    for name, monitors in method_monitors.items():
        if monitors is None:
            continue
        abl = run_sensor_ablation(monitors, sim_records, name)
        ablation_rows.extend(abl)

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(f'{OUT_DIR}/exp2_ablation/ablation_metrics.csv', index=False)
    # Also save as raw (same format for single seed)
    ablation_df.to_csv(f'{OUT_DIR}/exp2_ablation/ablation_metrics_raw.csv', index=False)
    tracker.end_phase()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: M3(A) ATTRIBUTION ABLATION
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 3: M3(A) ATTRIBUTION ABLATION")
    logger.info("=" * 70)

    tracker.start_phase('M3a_attribution')
    m3a_df = run_exp_m3a_attribution(
        grid_val, grid_obs, data, smap, gt_set,
        comfort_targets=comfort_targets, n_runs=N_RUNS, n_steps=N_STEPS)
    m3a_df.to_csv(f'{OUT_DIR}/exp3_ablation_m3/m3a_graph_attribution.csv', index=False)
    tracker.end_phase()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: A2 REGIME-SHIFT RECOVERY
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 4: A2 REGIME-SHIFT RECOVERY")
    logger.info("=" * 70)

    tracker.start_phase('A2_regime_recovery')
    a2_df = run_exp_a2_regime_recovery(grid_val, data, smap, n_episodes=3)
    a2_df.to_csv(f'{OUT_DIR}/exp4_appendix/a2_regime_recovery.csv', index=False)
    tracker.end_phase()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: N2 WRONG-PRIOR STRESS TEST
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 5: N2 WRONG-PRIOR STRESS TEST")
    logger.info("=" * 70)

    tracker.start_phase('N2_wrong_prior')
    n2_df = run_exp_n2_wrong_prior(data, gt_set, n_bad_edges=10, n_seeds=1)
    n2_df.to_csv(f'{OUT_DIR}/exp5_stress/n2_wrong_prior.csv', index=False)
    tracker.end_phase()

    # ══════════════════════════════════════════════════════════════════
    # M5: COMPUTE TRACKING
    # ══════════════════════════════════════════════════════════════════
    n_sim = sum(len(sr) for sr in all_sim_records)
    enhance_compute_tracker(
        tracker, discovery_results=[disc_result],
        n_sim_steps=n_sim, n_interventions=n_interventions)

    if gpt is not None:
        tracker.record_llm_usage(gpt)
    tracker.save(f'{OUT_DIR}/compute_resources.json')

    # ══════════════════════════════════════════════════════════════════
    # POST-HOC ANALYSES (10 figures + summary)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("  POST-HOC ANALYSES")
    logger.info("=" * 70)

    tracker.start_phase('posthoc')
    try:
        os.system(f'{sys.executable} scripts/posthoc_analysis.py')
        logger.info("  Post-hoc analyses complete")
    except Exception as e:
        logger.warning(f"  Post-hoc failed: {e}")
    tracker.end_phase()

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - t_total
    logger.info("\n" + "=" * 70)
    logger.info("  FULL SINGLE-SEED RUN COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time: {total_elapsed/60:.1f} minutes")

    # Write summary
    summary = []
    summary.append("NeurIPS Full Single-Seed Run — Summary")
    summary.append("=" * 55)
    summary.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    summary.append(f"Seed: {SEED}")
    summary.append(f"Runtime: {total_elapsed/60:.1f} minutes")
    summary.append(f"Interventions: {n_interventions}")
    summary.append("")

    summary.append("DISCOVERY")
    summary.append("-" * 40)
    for _, row in discovery_df.iterrows():
        summary.append(f"  {row['method']:<22} F1={row['f1']:.3f}  "
                       f"P={row['precision']:.3f}  R={row['recall']:.3f}  "
                       f"SHD={row['shd']}")

    summary.append("")
    summary.append("MONITORING")
    summary.append("-" * 40)
    for _, row in monitoring_df.iterrows():
        summary.append(f"  {row['method']:<22} Occ={row['occ_accuracy']:>5.1f}%  "
                       f"Win={row['win_accuracy']:>5.1f}%  "
                       f"Comb={row['combined_accuracy']:>5.1f}%")

    summary.append("")
    summary.append("POLICY (Pareto sweep)")
    summary.append("-" * 40)
    for ct in comfort_targets:
        summary.append(f"\n  ε={ct}:")
        ct_sub = policy_df[policy_df['comfort_target'] == ct]
        for name in ['PolicyGRID', 'PolicyGRID-O', 'PolicyGRID-R',
                     'MPC_Linear', 'MPC_Neural', 'C-MBPO',
                     'DDPC_Behavioral', 'DDPC_Subspace', 'DDPC_Neural', 'PID']:
            sub = ct_sub[ct_sub['method'] == name]
            if len(sub) > 0:
                summary.append(f"    {name:<20} MO={sub['mo_score'].mean():.3f}  "
                               f"kWh={sub['kwh'].mean():.2f}  "
                               f"DH={sub['dh'].mean():.3f}  "
                               f"Viol={sub['violation_rate'].mean()*100:.1f}%")

    summary.append("")
    summary.append("HYPERVOLUME")
    summary.append("-" * 40)
    try:
        hv_info = compute_pareto_hypervolume(policy_df)
        for arm, hv_val in hv_info.get('per_arm', {}).items():
            summary.append(f"  {arm:<22} HV={hv_val:.3f}")
    except Exception:
        summary.append("  (HV computation failed)")

    summary.append("")
    summary.append(tracker.format_for_summary(n_seeds=1))

    summary_text = '\n'.join(summary)
    with open(f'{OUT_DIR}/summary.txt', 'w') as f:
        f.write(summary_text)

    print("\n" + summary_text)


if __name__ == '__main__':
    main()
