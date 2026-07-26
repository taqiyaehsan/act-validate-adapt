#!/usr/bin/env python3
"""
Unified Discovery + Monitoring for ALL simulators.
===================================================
Runs PolicyGRID discovery pipeline + 9 discovery benchmarks + monitoring
(where applicable) across multiple seeds.

Usage:
    python run_discovery_monitoring.py                          # all sims, 3 seeds
    python run_discovery_monitoring.py --sim smart_building_rich # one sim
    python run_discovery_monitoring.py --sim open_window --seeds 5
    python run_discovery_monitoring.py --sim all --seeds 3

Output: results/discovery_monitoring/<sim_name>/
    - discovered_edges.json      (per-seed + consensus)
    - benchmark_results.csv      (F1/SHD for all methods)
    - monitoring_metrics.csv     (if sim has regimes)
    - figures/                   (comparison plots)
"""

import sys, os, json, time, warnings, argparse, random
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
sys.path.insert(0, 'benchmarks_new')
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# SIM CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════

SIM_CONFIGS = {
    'smart_room': {
        'sim_path': 'js/smart_room.js',
        'data_path': 'data_regen/smart_room_varied_processed.csv',
        'scaling_path': 'data_regen/smart_room_varied_scaling.csv',
        'gt_key': 'smart_room',
        'has_regimes': False,
        'actuators': ['temperature', 'humidity', 'airquality'],
        'non_intervenable': ['energyconsumption', 'satisfaction'],
        'dataset_type': 'smart_room',
    },
    'smart_room_noise': {
        'sim_path': 'js/smart_room_noise.js',
        'data_path': 'data_regen/smart_room_noise_varied_processed.csv',
        'scaling_path': 'data_regen/smart_room_noise_varied_scaling.csv',
        'gt_key': 'smart_room_noise',
        'has_regimes': False,
        'actuators': ['temperature', 'humidity', 'airquality'],
        'non_intervenable': ['energyconsumption', 'satisfaction'],
        'dataset_type': 'smart_room_noise',
    },
    'smart_room_hidden_vars': {
        'sim_path': 'js/smart_room_hidden_vars.js',
        'data_path': 'data_regen/smart_room_hidden_vars_varied_processed.csv',
        'scaling_path': 'data_regen/smart_room_hidden_vars_varied_scaling.csv',
        'gt_key': 'hidden_vars',
        'has_regimes': False,
        'actuators': ['temperature', 'humidity', 'airquality'],
        'non_intervenable': ['energyconsumption', 'satisfaction'],
        'dataset_type': 'hidden_vars',
    },
    'open_window': {
        'sim_path': 'js/open_window.js',
        'data_path': 'data_regen/open_window_varied_processed.csv',
        'scaling_path': 'data_regen/open_window_varied_scaling.csv',
        'gt_key': 'open_window',
        'has_regimes': True,
        'regime_vars': ['windowopen'],
        'observable_vars': ['temperature', 'humidity', 'airquality', 'pmv',
                           'energyconsumption', 'satisfaction'],
        'latent_vars': ['windowopen', 'outdoortemperature'],
        'actuators': ['temperature', 'humidity', 'airquality'],
        'non_intervenable': ['windowopen', 'outdoortemperature', 'pmv',
                            'energyconsumption', 'satisfaction', 'elapsedtime'],
        'dataset_type': 'open_window',
    },
    'smart_building_rich': {
        'sim_path': 'js/smart_building_rich.js',
        'data_path': 'data_regen/smart_building_rich_processed.csv',
        'scaling_path': 'data_regen/smart_building_rich_processed_scaling.csv',
        'gt_key': 'smart_building_rich',
        'has_regimes': True,
        'regime_vars': ['occupancy', 'windowposition'],
        'observable_vars': ['outdoortemp', 'solarradiation', 'temperature',
                           'humidity', 'co2', 'lightlevel', 'noisedb',
                           'airquality', 'pmv', 'hvacpower', 'lightingpower',
                           'energyconsumption', 'satisfaction'],
        'latent_vars': ['occupancy', 'windowposition'],
        'actuators': ['hvacpower', 'lightingpower'],
        'non_intervenable': ['outdoortemp', 'solarradiation', 'occupancy',
                            'windowposition', 'pmv', 'energyconsumption',
                            'satisfaction'],
        'dataset_type': 'smart_building_rich',
    },
    'ashrae': {
        'sim_path': None,
        'data_path': 'data/ashrae_data_processed.csv',
        'scaling_path': 'data/ashrae_data_processed_scaling_params.csv',
        'gt_key': 'ashrae',
        'has_regimes': False,
        'actuators': [],
        'non_intervenable': ['meter_reading'],
        'dataset_type': 'ashrae',
    },
}

DEFAULT_SEEDS = [42, 123, 456]

API_KEY = 'YOUR_OPENAI_API_KEY'


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def graph_metrics(predicted, gt):
    pred = {(s.lower(), t.lower()) for s, t in predicted}
    tp = len(pred & gt); fp = len(pred - gt); fn = len(gt - pred)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2*p*r / max(p+r, 1e-10)
    return {'f1': f1, 'precision': p, 'recall': r, 'shd': fp + fn, 'n_edges': len(pred)}


# ═══════════════════════════════════════════════════════════════════════════════
# DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def run_policygrid_discovery(cfg, data, seed):
    """Run PolicyGRID discovery pipeline for one seed."""
    from src.pipeline_cwm import create_cwm_pipeline

    random.seed(seed)
    np.random.seed(seed)

    sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None

    pipeline = create_cwm_pipeline(
        csv_data=data, api_key=API_KEY,
        smart_room_path=sim_path,
        dataset_type=cfg['dataset_type'],
        max_iterations=3,
        actuator_vars=cfg['actuators'],
        non_intervenable_vars=cfg['non_intervenable'],
    )
    final_dag, final_metrics = pipeline.run_with_cwm()
    validated = set(pipeline.pipeline.validated_edges)
    return validated


def run_benchmarks(cfg, data):
    """Run 9 discovery benchmarks. Returns dict of {name: edge_set}."""
    sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None

    benchmarks = [
        ('PC',              'pc_baseline',      'run_pc_baseline'),
        ('SAM',             'sam_baseline',      'run_sam_baseline'),
        ('GIES',            'gies_comparison',   'run_gies'),
        ('ICP',             'icp_comparison',    'run_icp'),
        ('NOTEARS-I',       'notears_i',         'run_notears_i'),
        ('ABCD',            'abcd_comparison',   'run_abcd'),
        ('JCI',             'jci_comparison',    'run_jci'),
        ('Causal Bandits',  'causal_bandits',    'run_causal_bandits'),
        ('IID',             'iid_comparison',    'run_iid'),
    ]

    results = {}
    for name, module_name, func_name in benchmarks:
        try:
            logger.info(f"    Running {name}...")
            t0 = time.time()
            import importlib
            mod = importlib.import_module(module_name)
            func = getattr(mod, func_name)
            result = func(data, simulation_path=sim_path)
            elapsed = time.time() - t0

            if isinstance(result, tuple) and len(result) == 2:
                edges = result[0]
            elif hasattr(result, 'edges'):
                edges = list(result.edges())
            elif isinstance(result, dict):
                edges = result.get('edges', [])
            else:
                edges = result

            edge_set = {(s.lower(), t.lower()) for s, t in edges}
            results[name] = edge_set
            logger.info(f"      {name}: {len(edge_set)} edges ({elapsed:.1f}s)")
        except Exception as e:
            logger.warning(f"      {name} FAILED: {e}")
            results[name] = set()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

def run_monitoring_eval(cfg, data, edges, sim_name, out_dir):
    """Run monitoring evaluation if sim has regimes."""
    if not cfg['has_regimes']:
        logger.info("    No regimes — skipping monitoring")
        return None

    # Import monitoring functions from neurips_final_experiments
    # This is the same code that achieved 90% accuracy
    try:
        from neurips_final_experiments import (
            construct_factored_monitors, collect_sim_data,
            run_monitoring, compute_monitoring_metrics,
            build_observable_graph, get_scaling_map,
            OBSERVABLE_VARS, SENSOR_VARS, LATENT_VARS
        )
    except ImportError:
        logger.warning("    Cannot import monitoring functions — skipping")
        return None

    # Only run monitoring for smart_building_rich (has the full monitoring pipeline)
    if sim_name != 'smart_building_rich':
        logger.info("    Monitoring only implemented for smart_building_rich — skipping")
        return None

    logger.info("    Building monitors...")
    monitors = construct_factored_monitors(edges, data)
    if monitors is None:
        logger.warning("    No observable edges — cannot build monitors")
        return None

    logger.info("    Collecting sim data for monitoring evaluation...")
    scaling = pd.read_csv(cfg['scaling_path'])
    smap = get_scaling_map(scaling)
    sim_records = collect_sim_data(smap, duration_steps=1440)  # 1 day

    logger.info("    Running monitoring...")
    records = run_monitoring(monitors, sim_records)
    metrics = compute_monitoring_metrics(records)

    logger.info(f"    Monitoring: occ={metrics['occ_accuracy']:.1f}%, "
                f"win={metrics['win_accuracy']:.1f}%, "
                f"combined={metrics['combined_accuracy']:.1f}%")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_sim(sim_name, seeds):
    cfg = SIM_CONFIGS[sim_name]
    out_dir = f'results/discovery_monitoring/{sim_name}'
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"  {sim_name}")
    logger.info(f"{'='*70}")

    if not os.path.exists(cfg['data_path']):
        logger.error(f"  Data not found: {cfg['data_path']}")
        return

    data = pd.read_csv(cfg['data_path'])
    with open('ground_truth_graphs.json') as f:
        gt_raw = json.load(f)[cfg['gt_key']]['edges']
    gt_set = {(s.lower(), t.lower()) for s, t in gt_raw}

    logger.info(f"  Data: {data.shape}, GT: {len(gt_set)} edges, Seeds: {seeds}")

    # ── PolicyGRID Discovery (multi-seed) ─────────────────────────────────
    logger.info(f"\n  PolicyGRID Discovery ({len(seeds)} seeds)...")

    per_seed_edges = []
    per_seed_metrics = []

    for seed in seeds:
        logger.info(f"\n    Seed {seed}...")
        t0 = time.time()

        if cfg['sim_path'] is None:
            # ASHRAE: no simulator, run observational-only
            logger.info("      No simulator — running observational pipeline only")
            from src.pipeline import CausalPipeline
            pipeline = CausalPipeline(data, api_key=API_KEY,
                                      dataset_type=cfg['dataset_type'])
            random.seed(seed); np.random.seed(seed)
            result = pipeline.run()
            # Extract edges from pipeline result
            validated = set()
            if hasattr(pipeline, 'validated_edges'):
                validated = set(pipeline.validated_edges)
            elif isinstance(result, dict):
                for method_name, method_result in result.items():
                    if isinstance(method_result, dict) and 'edges' in method_result:
                        validated.update(method_result['edges'])
        else:
            validated = run_policygrid_discovery(cfg, data, seed)

        elapsed = time.time() - t0
        val_set = {(s.lower(), t.lower()) for s, t in validated}
        m = graph_metrics(val_set, gt_set)
        m['time_s'] = elapsed
        per_seed_edges.append(validated)
        per_seed_metrics.append(m)

        logger.info(f"      {len(validated)} edges, F1={m['f1']:.3f}, "
                    f"P={m['precision']:.3f}, R={m['recall']:.3f}, "
                    f"SHD={m['shd']}, Time={elapsed:.0f}s")

    # Consensus (majority vote)
    from collections import Counter
    edge_counts = Counter()
    for edges in per_seed_edges:
        for e in edges:
            edge_counts[(e[0].lower() if isinstance(e[0], str) else e[0],
                         e[1].lower() if isinstance(e[1], str) else e[1])] += 1

    majority = len(seeds) // 2 + 1
    consensus = {e for e, count in edge_counts.items() if count >= majority}
    consensus_metrics = graph_metrics(consensus, gt_set)

    logger.info(f"\n  Consensus (≥{majority}/{len(seeds)}): "
                f"{len(consensus)} edges, F1={consensus_metrics['f1']:.3f}, "
                f"SHD={consensus_metrics['shd']}")

    # ── Benchmarks ────────────────────────────────────────────────────────
    logger.info(f"\n  Discovery Benchmarks...")
    benchmark_results = run_benchmarks(cfg, data)

    # Compute metrics for all methods
    all_results = []
    all_results.append({
        'method': 'PolicyGRID', 'type': 'consensus',
        **consensus_metrics, 'time_s': sum(m['time_s'] for m in per_seed_metrics)
    })
    for name, edges in benchmark_results.items():
        m = graph_metrics(edges, gt_set)
        all_results.append({'method': name, 'type': 'benchmark', **m})

    results_df = pd.DataFrame(all_results).sort_values('f1', ascending=False)

    logger.info(f"\n  Discovery Summary:")
    logger.info(f"  {'Method':<20} {'Edges':>5} {'F1':>6} {'P':>6} {'R':>6} {'SHD':>5}")
    logger.info(f"  {'-'*50}")
    for _, r in results_df.iterrows():
        logger.info(f"  {r['method']:<20} {r['n_edges']:>5} {r['f1']:>6.3f} "
                    f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['shd']:>5}")

    # ── Monitoring (if applicable) ────────────────────────────────────────
    monitoring_metrics = None
    if cfg['has_regimes']:
        logger.info(f"\n  Monitoring...")
        monitoring_metrics = run_monitoring_eval(cfg, data, consensus, sim_name, out_dir)

    # ── Save Results ──────────────────────────────────────────────────────
    edges_out = {
        'sim': sim_name,
        'seeds': seeds,
        'per_seed_validated': [[list(e) for e in v] for v in per_seed_edges],
        'per_seed_metrics': per_seed_metrics,
        'consensus_validated': [list(e) for e in consensus],
        'consensus_metrics': consensus_metrics,
    }
    with open(f'{out_dir}/discovered_edges.json', 'w') as f:
        json.dump(edges_out, f, indent=2, default=str)

    results_df.to_csv(f'{out_dir}/benchmark_results.csv', index=False)

    if monitoring_metrics:
        with open(f'{out_dir}/monitoring_metrics.json', 'w') as f:
            json.dump(monitoring_metrics, f, indent=2, default=str)

    logger.info(f"\n  Results saved to {out_dir}/")
    return edges_out, results_df, monitoring_metrics


def main():
    parser = argparse.ArgumentParser(description='Run discovery + monitoring for all sims')
    parser.add_argument('--sim', default='all',
                        choices=list(SIM_CONFIGS.keys()) + ['all'])
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of seeds (default: 3)')
    parser.add_argument('--seed-list', type=str, default=None,
                        help='Comma-separated seed list (overrides --seeds)')
    args = parser.parse_args()

    if args.seed_list:
        seeds = [int(s) for s in args.seed_list.split(',')]
    else:
        seeds = DEFAULT_SEEDS[:args.seeds]

    t_start = time.time()

    if args.sim == 'all':
        for sim_name in SIM_CONFIGS:
            run_sim(sim_name, seeds)
    else:
        run_sim(args.sim, seeds)

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*70}")
    logger.info(f"  TOTAL TIME: {elapsed/60:.1f} minutes")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
