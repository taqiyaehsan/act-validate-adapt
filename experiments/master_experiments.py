#!/usr/bin/env python3
"""
Master Experiment Suite — Multi-Simulator 3-Seed Experiments
============================================================
Runs discovery (+ monitoring/policy where applicable) across all 6 simulators
for NeurIPS 2026. Saves results in results/master-3seed-results/.

Usage:
    python master_experiments.py                    # Run all sims
    python master_experiments.py --sims smart_room  # Run specific sim
    python master_experiments.py --sims smart_building_rich --skip-discovery
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import resource
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sim_configs import get_config, SIM_CONFIGS
from src.pipeline_cwm import create_cwm_pipeline
from src.gpt_client import GPTClient

# ─── Config ───────────────────────────────────────────────────────────────────
API_KEY = os.environ.get('OPENAI_API_KEY',
    'YOUR_OPENAI_API_KEY')

SEEDS = [42, 123, 456]
OUT_ROOT = ROOT / 'results' / 'master-3seed-results'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_sim_data(config):
    """Load data + GT + scaling for any simulator config."""
    data_path = ROOT / config['data_path']
    data = pd.read_csv(data_path)

    # Normalize satisfaction column name
    if 'Satisfaction' in data.columns and 'OverallSatisfaction' not in data.columns:
        data = data.rename(columns={'Satisfaction': 'OverallSatisfaction'})

    # Drop extra columns (e.g., ASHRAE 'weight')
    drop_cols = config.get('drop_columns', [])
    for col in drop_cols:
        if col in data.columns:
            data = data.drop(columns=[col])

    # Load GT
    gt_path = ROOT / config['gt_path']
    with open(gt_path) as f:
        gt_raw = json.load(f)[config['gt_key']]['edges']
    gt_set = {(s.lower(), t.lower()) for s, t in gt_raw}

    # Load scaling
    scaling_path = ROOT / config['scaling_path']
    scaling = pd.read_csv(scaling_path)
    smap = build_scaling_map(scaling, config.get('scaling_format', 'minmax'))

    return data, gt_set, smap


def build_scaling_map(scaling, fmt='minmax'):
    """Build {var_lower: (min, max)} from scaling CSV.
    Supports 'minmax' and 'zscore' formats.
    """
    feat_col = 'feature' if 'feature' in scaling.columns else 'column'
    smap = {}
    if fmt == 'zscore':
        # zscore: center, scale_factor → approximate min/max as center ± 3*scale
        for _, row in scaling.iterrows():
            center = row['center']
            scale = row['scale_factor']
            smap[row[feat_col].lower()] = (center - 3 * scale, center + 3 * scale)
    else:
        min_col = 'data_min' if 'data_min' in scaling.columns else 'min'
        max_col = 'data_max' if 'data_max' in scaling.columns else 'max'
        for _, row in scaling.iterrows():
            smap[row[feat_col].lower()] = (row[min_col], row[max_col])
    return smap


# ═════════════════════════════════════════════════════════════════════════════
# DISCOVERY
# ═════════════════════════════════════════════════════════════════════════════

def _norm_var(v):
    """Normalize variable name for comparison."""
    v = str(v).lower()
    if v == 'overallsatisfaction':
        v = 'satisfaction'
    return v


def graph_metrics(predicted_edges, gt_set):
    """Compute F1, SHD, Precision, Recall against GT."""
    pred = {(_norm_var(s), _norm_var(t)) for s, t in predicted_edges}
    gt = {(_norm_var(s), _norm_var(t)) for s, t in gt_set}
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    shd = fp + fn
    return {'F1': f1, 'SHD': shd, 'Precision': prec, 'Recall': rec,
            'TP': tp, 'FP': fp, 'FN': fn, 'n_edges': len(pred)}


def run_discovery_for_config(config, data, seeds=SEEDS):
    """Run PolicyGRID discovery + 9 benchmarks for a single simulator.
    Returns (discovery_df, all_edges_dict).
    """
    sim_name = config['name']
    dataset_type = config.get('gt_key', sim_name)

    # Load GT for metrics
    gt_path = ROOT / config['gt_path']
    with open(gt_path) as f:
        gt_raw = json.load(f)[config['gt_key']]['edges']
    gt_set = {(s.lower(), t.lower()) for s, t in gt_raw}

    # Map gt_key to pipeline dataset_type
    dt_map = {
        'smart_building_rich': 'smart_building_rich',
        'open_window': 'open_window',
        'smart_room': 'smart_room',
        'smart_room_noise': 'smart_room_noise',
        'hidden_vars': 'hidden_vars',
        'ashrae': 'ashrae',
    }
    pipeline_dt = dt_map.get(dataset_type, dataset_type)

    # Determine sim path (None only for ASHRAE which has no JS simulator)
    sim_path = None
    if config.get('sim_path'):
        sp = ROOT / config['sim_path']
        if sp.exists():
            sim_path = str(sp)

    rows = []
    all_edges = {}

    # ── PolicyGRID discovery (per seed) ──
    for seed in seeds:
        logger.info(f"[{sim_name}] PolicyGRID discovery seed={seed}")
        random.seed(seed)
        np.random.seed(seed)

        try:
            pipeline = create_cwm_pipeline(
                csv_data=data,
                api_key=API_KEY,
                smart_room_path=sim_path,
                dataset_type=pipeline_dt,
                max_iterations=10,
                alpha=0.5, beta=0.5,
                # Fixed practical significance threshold (Cohen 1988):
                # 0.03 standardized effect — consistent across all domains.
                # Combined with p<0.05 t-test + sign consistency check in tester.
                effect_threshold=0.03,
                actuator_vars=config.get('actuator_vars', []),
                non_intervenable_vars=config.get('non_intervenable_vars', []),
            )

            t0 = time.time()
            final_dag, final_metrics = pipeline.run_with_cwm()
            disc_time = time.time() - t0

            validated = set(pipeline.pipeline.validated_edges)
            method_dags = pipeline.pipeline.get_method_dags()
            obs_only = set()
            for G in method_dags.values():
                obs_only.update(G.edges())

            all_edges[f'grid_val_seed{seed}'] = validated
            all_edges[f'grid_obs_seed{seed}'] = obs_only

            intervention_count = sum(
                len(r) for r in pipeline.pipeline.tester.intervention_results.values()
            )

            logger.info(f"  Validated: {len(validated)}, Obs: {len(obs_only)}, "
                        f"Interventions: {intervention_count}, Time: {disc_time:.1f}s")

        except Exception as e:
            logger.error(f"  PolicyGRID discovery failed: {e}")
            validated = set()
            obs_only = set()
            intervention_count = 0
            disc_time = 0

        # Record both validated and obs-only
        for variant, edges, label in [
            ('PolicyGRID', validated, 'val'),
            ('PolicyGRID-O', obs_only, 'obs'),
        ]:
            rows.append({
                'method': variant, 'seed': seed, 'sim': sim_name,
                **graph_metrics(edges, gt_set),
                'interventions': intervention_count if label == 'val' else 0,
                'time_s': disc_time,
            })

    # ── Benchmark discovery (per seed, same as PolicyGRID) ──
    logger.info(f"[{sim_name}] Running 9 benchmark methods × {len(seeds)} seeds...")
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

    # Normalize column names for benchmarks that expect OverallSatisfaction
    bench_data = data.copy()
    if 'Satisfaction' in bench_data.columns and 'OverallSatisfaction' not in bench_data.columns:
        bench_data = bench_data.rename(columns={'Satisfaction': 'OverallSatisfaction'})

    # Benchmarks that accept simulation_path for interventional data
    intervention_benchmarks = {'GIES', 'ICP', 'NOTEARS-I', 'ABCD', 'JCI',
                               'Causal Bandits', 'IID'}

    for name, module, func_name in benchmarks:
        for seed in seeds:
            try:
                random.seed(seed)
                np.random.seed(seed)
                mod = __import__(f'benchmarks_new.{module}', fromlist=[func_name])
                fn = getattr(mod, func_name)
                t0 = time.time()
                if name in intervention_benchmarks and sim_path:
                    result = fn(bench_data, simulation_path=sim_path)
                    # Some benchmarks return (edges, intervention_results)
                    if isinstance(result, tuple):
                        edges = result[0]
                    else:
                        edges = result
                else:
                    edges = fn(bench_data)
                elapsed = time.time() - t0
                if edges is None:
                    edges = set()
                all_edges[f'{name}_seed{seed}'] = edges
                metrics = graph_metrics(edges, gt_set)
                logger.info(f"  {name} (seed={seed}): F1={metrics['F1']:.3f}, "
                            f"SHD={metrics['SHD']}, edges={metrics['n_edges']}, "
                            f"time={elapsed:.1f}s")
                rows.append({
                    'method': name, 'seed': seed, 'sim': sim_name,
                    **metrics, 'interventions': 0, 'time_s': elapsed,
                })
            except Exception as e:
                logger.error(f"  {name} (seed={seed}) failed: {e}")
                rows.append({
                    'method': name, 'seed': seed, 'sim': sim_name,
                    'F1': 0, 'SHD': 999, 'Precision': 0, 'Recall': 0,
                    'TP': 0, 'FP': 0, 'FN': len(gt_set),
                    'n_edges': 0, 'interventions': 0, 'time_s': 0,
                })

    discovery_df = pd.DataFrame(rows)
    return discovery_df, all_edges


# ═════════════════════════════════════════════════════════════════════════════
# MONITORING (generalized for 2-way and 4-way)
# ═════════════════════════════════════════════════════════════════════════════

def run_monitoring_for_config(config, edges_dict, data, smap, n_sim_seeds=3):
    """Run regime monitoring for sims with regime variables.

    Supports:
    - 4-way (smart_building_rich): base, occ, win, full
    - 2-way (open_window): closed, open
    """
    from sklearn.linear_model import Ridge

    sim_name = config['name']
    regime_vars = config['regime_vars']
    regime_names = config['regime_names']
    sensor_vars = config['sensor_vars']
    observable_vars = [v for v in config['all_vars'] if v not in config['latent_vars']]

    if not regime_vars:
        logger.info(f"[{sim_name}] No regime variables — skipping monitoring")
        return None

    # Use consensus validated edges (or obs-only if no validated)
    val_keys = [k for k in edges_dict if k.startswith('grid_val')]
    if val_keys:
        # Consensus: edges appearing in ≥2 of 3 seeds
        edge_counts = defaultdict(int)
        for k in val_keys:
            for e in edges_dict[k]:
                edge_counts[e] += 1
        edges = {e for e, c in edge_counts.items() if c >= 2}
    else:
        obs_keys = [k for k in edges_dict if k.startswith('grid_obs')]
        edge_counts = defaultdict(int)
        for k in obs_keys:
            for e in edges_dict[k]:
                edge_counts[e] += 1
        edges = {e for e, c in edge_counts.items() if c >= 2}

    # Filter to observable edges only
    obs_edges = {(s, t) for s, t in edges
                 if s in observable_vars and t in observable_vars}

    logger.info(f"[{sim_name}] Building {len(regime_names)}-way monitors "
                f"with {len(obs_edges)} observable edges")

    # Build regime masks from training data
    regime_masks = {}
    for rname in regime_names:
        mask = pd.Series([True] * len(data), index=data.index)
        for rvar, rinfo in regime_vars.items():
            col = rvar
            # Find matching column (case-insensitive)
            for c in data.columns:
                if c.lower() == rvar:
                    col = c
                    break
            if col in data.columns:
                if rname == 'base' or rname == 'closed':
                    # All regime vars OFF
                    mask &= (data[col] < rinfo['threshold'])
                elif rname == 'full' or rname == 'open':
                    # All regime vars ON (for 2-way, 'open' = var ON)
                    mask &= (data[col] >= rinfo['threshold'])
                elif rinfo['label'] in rname:
                    # This regime var ON
                    mask &= (data[col] >= rinfo['threshold'])
                else:
                    # This regime var OFF
                    mask &= (data[col] < rinfo['threshold'])
        regime_masks[rname] = mask

    # For 2-way sims, regime logic is simpler
    if len(regime_names) == 2 and len(regime_vars) == 1:
        rvar = list(regime_vars.keys())[0]
        rinfo = regime_vars[rvar]
        col = rvar
        for c in data.columns:
            if c.lower() == rvar:
                col = c
                break
        if col in data.columns:
            regime_masks[regime_names[0]] = data[col] < rinfo['threshold']  # closed/base
            regime_masks[regime_names[1]] = data[col] >= rinfo['threshold']  # open

    # Log regime distribution
    for rname, mask in regime_masks.items():
        pct = mask.sum() / len(data) * 100
        logger.info(f"  Training regime '{rname}': {mask.sum()} rows ({pct:.1f}%)")

    # Build predictive models per regime
    models = {}
    for rname in regime_names:
        mask = regime_masks[rname]
        if mask.sum() < 20:
            logger.warning(f"  Regime '{rname}' has only {mask.sum()} rows — skipping")
            continue

        regime_data = data[mask]
        model_dict = {}
        for target in sensor_vars:
            tcol = None
            for c in data.columns:
                if c.lower() == target:
                    tcol = c
                    break
            if tcol is None:
                continue

            parents = [s for s, t in obs_edges if t.lower() == target]
            if not parents:
                continue

            parent_cols = []
            for p in parents:
                for c in data.columns:
                    if c.lower() == p:
                        parent_cols.append(c)
                        break

            if not parent_cols:
                continue

            X = regime_data[parent_cols].values
            y = regime_data[tcol].values

            if len(X) < 5:
                continue

            model = Ridge(alpha=0.5)
            model.fit(X, y)
            model_dict[target] = {
                'model': model,
                'parent_cols': parent_cols,
                'target_col': tcol,
            }

        models[rname] = model_dict
        logger.info(f"  Model '{rname}': {len(model_dict)} predictive models")

    if not models:
        logger.warning(f"[{sim_name}] No models built — skipping monitoring")
        return None

    # Now we need test data to evaluate monitoring
    # For open_window: generate from JS sim
    # For smart_building_rich: generate from JS sim
    sim_path = config.get('sim_path')
    if not sim_path:
        logger.warning(f"[{sim_name}] No simulator path — skipping monitoring eval")
        return None

    sim_abs = str(ROOT / sim_path)
    if not os.path.exists(sim_abs):
        logger.warning(f"[{sim_name}] Simulator not found: {sim_abs}")
        return None

    # Collect test data
    import subprocess
    test_steps = config.get('test_duration_steps', 4320)
    logger.info(f"[{sim_name}] Collecting {test_steps} test steps...")

    sim_records = []
    state = None
    for step in range(test_steps):
        elapsed_ms = step * 60000  # 1 minute per step
        if state is None:
            cmd = ['node', sim_abs, '--single-step',
                   '--state', '{}',
                   '--intervention', '{}',
                   '--elapsedMs', str(elapsed_ms)]
        else:
            cmd = ['node', sim_abs, '--single-step',
                   '--state', json.dumps(state),
                   '--intervention', '{}',
                   '--elapsedMs', str(elapsed_ms)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                continue
            state = json.loads(result.stdout.strip())

            # Convert to normalized record
            record = {}
            for k, v in state.items():
                var_lower = k.lower()
                if var_lower in smap:
                    mn, mx = smap[var_lower]
                    record[var_lower] = (v - mn) / (mx - mn) if mx > mn else 0.5
                else:
                    record[var_lower] = v

            sim_records.append(record)
        except Exception:
            continue

    if len(sim_records) < 100:
        logger.warning(f"[{sim_name}] Only {len(sim_records)} test steps — insufficient")
        return None

    logger.info(f"[{sim_name}] Collected {len(sim_records)} test steps")

    # Run Bayesian monitoring
    sigma = 1.5
    forget = 0.15
    n_regimes = len(regime_names)
    weights = {r: 1.0 / n_regimes for r in regime_names}

    monitoring_records = []
    for step, record in enumerate(sim_records):
        # Compute prediction error per regime model
        errors = {}
        for rname, model_dict in models.items():
            total_err = 0.0
            n_pred = 0
            for target, minfo in model_dict.items():
                parent_vals = []
                for pc in minfo['parent_cols']:
                    parent_vals.append(record.get(pc.lower(), 0.5))
                if parent_vals:
                    pred = minfo['model'].predict([parent_vals])[0]
                    actual = record.get(target, 0.5)
                    total_err += abs(actual - pred)
                    n_pred += 1
            errors[rname] = total_err / max(n_pred, 1)

        # Bayesian update
        likelihoods = {}
        for rname in regime_names:
            if rname in errors:
                likelihoods[rname] = np.exp(-sigma * errors[rname])
            else:
                likelihoods[rname] = 0.001

        Z = sum(likelihoods[r] * weights[r] for r in regime_names)
        if Z > 0:
            for r in regime_names:
                weights[r] = (1 - forget) * (likelihoods[r] * weights[r] / Z) + \
                             forget * (1.0 / n_regimes)

        predicted_regime = max(weights, key=weights.get)

        # Determine GT regime
        gt_regime = _determine_gt_regime(record, config)

        monitoring_records.append({
            'step': step,
            'gt_regime': gt_regime,
            'predicted_regime': predicted_regime,
            **{f'w_{r}': weights[r] for r in regime_names},
        })

    mon_df = pd.DataFrame(monitoring_records)
    correct = (mon_df['gt_regime'] == mon_df['predicted_regime']).sum()
    total = len(mon_df)
    accuracy = correct / total * 100 if total > 0 else 0

    logger.info(f"[{sim_name}] Monitoring accuracy: {accuracy:.1f}% ({correct}/{total})")

    # Per-regime accuracy
    for rname in regime_names:
        mask = mon_df['gt_regime'] == rname
        if mask.sum() > 0:
            acc = (mon_df.loc[mask, 'predicted_regime'] == rname).sum() / mask.sum() * 100
            logger.info(f"  {rname}: {acc:.1f}% ({mask.sum()} samples)")

    return mon_df


def _determine_gt_regime(record, config):
    """Determine ground-truth regime from a record using config thresholds."""
    regime_vars = config['regime_vars']
    regime_names = config['regime_names']

    if len(regime_vars) == 1:
        # 2-way
        rvar = list(regime_vars.keys())[0]
        rinfo = regime_vars[rvar]
        val = record.get(rvar, 0)
        # For normalized data, threshold applies to raw
        return regime_names[1] if val >= rinfo['threshold'] else regime_names[0]
    elif len(regime_vars) == 2:
        # 4-way (occupancy × window)
        states = {}
        for rvar, rinfo in regime_vars.items():
            val = record.get(rvar, 0)
            states[rinfo['label']] = val >= rinfo['threshold']

        occ = states.get('occ', False)
        win = states.get('win', False)
        if occ and win:
            return 'full'
        elif occ:
            return 'occ'
        elif win:
            return 'win'
        else:
            return 'base'

    return regime_names[0] if regime_names else 'unknown'


# ═════════════════════════════════════════════════════════════════════════════
# POLICY (small sims via ParetoAnalyzer)
# ═════════════════════════════════════════════════════════════════════════════

def run_policy_pareto(config, edges_dict, data, out_dir, n_scenarios=10):
    """Run Pareto policy analysis using ParetoAnalyzer + DDPC baselines.

    Uses sim-reported EnergyConsumption for kWh and action-derived setpoint for DH.
    Runs: GRID_Causal, ASHRAE_PID, WorldModel_Only, Correlation,
          DDPC_Behavioral, DDPC_Subspace, DDPC_Neural.
    """
    from src.policy_engine import CausalPolicyEngine
    from src.ddpc_baseline import create_ddpc_suite
    from experiments.pareto_frontier_system import (
        ParetoAnalyzer, ParetoOptimizer, ParetoPoint, PolicyResult
    )

    sim_name = config['name']
    sim_path = str(ROOT / config['sim_path']) if config.get('sim_path') else None

    # Build consensus validated edges for the policy engine
    val_keys = [k for k in edges_dict if k.startswith('grid_val')]
    if val_keys:
        from collections import Counter
        edge_counts = Counter()
        for k in val_keys:
            for e in edges_dict[k]:
                edge_counts[e] += 1
        consensus_edges = {e for e, c in edge_counts.items() if c >= 2}
    else:
        consensus_edges = set()

    # Build pipeline results dict for CausalPolicyEngine
    pipeline_results = {
        'final_dag': {'edges': list(consensus_edges)},
        'validated_edges': consensus_edges,
        'intervention_results': {},
    }

    logger.info(f"[{sim_name}] Initializing policy engine with {len(consensus_edges)} edges")

    try:
        policy_engine = CausalPolicyEngine(pipeline_results, data, use_llm=False)

        # Configure ParetoOptimizer with correct sim path
        # Comfort targets in degree-hours (Δt=1/60 hr per step × 12 steps)
        analyzer = ParetoAnalyzer(
            policy_engine,
            comfort_targets=[0.5, 1.0, 2.0, 4.0, 8.0],
            n_bootstrap=1000,
        )
        analyzer.optimizer.sim_path = sim_path

        # Run ParetoAnalyzer policies (GRID_Causal, ASHRAE_PID, WorldModel_Only, Correlation)
        logger.info(f"[{sim_name}] Running Pareto analysis ({n_scenarios} scenarios)")
        results = analyzer.run_pareto_analysis(n_scenarios=n_scenarios)

        # ── DDPC baselines ──
        # Normalize column names for DDPC (expects ProperCase)
        ddpc_data = data.copy()
        col_map = {}
        for c in ddpc_data.columns:
            if c.lower() == 'satisfaction':
                col_map[c] = 'OverallSatisfaction'
            elif c.lower() == 'overallsatisfaction':
                col_map[c] = 'OverallSatisfaction'
            else:
                col_map[c] = c[0].upper() + c[1:] if c else c
        ddpc_data = ddpc_data.rename(columns=col_map)

        # DDPC requires Temperature/Humidity/AirQuality columns — skip for ASHRAE
        has_ddpc_cols = all(c in ddpc_data.columns
                           for c in ['Temperature', 'Humidity', 'AirQuality'])
        if not has_ddpc_cols:
            logger.info(f"[{sim_name}] Skipping DDPC (columns don't match STATE_VARS)")
        else:
            logger.info(f"[{sim_name}] Fitting DDPC baselines...")
        if has_ddpc_cols:
            try:
                ddpc_controllers = create_ddpc_suite(
                    ddpc_data,
                    cem_population=50,   # 50 candidates (default 300) — 6x faster
                    cem_n_iter=5,        # 5 iterations (default 20) — 4x faster
                    pets_ensemble=3,     # 3 MLPs (default 5)
                    pets_particles=5,    # 5 particles (default 10)
                    pets_cem_pop=30,     # 30 candidates (default 50)
                    pets_cem_iter=3,     # 3 iterations (default 5)
                )

                scenarios = analyzer.optimizer._generate_scenarios(n_scenarios) \
                    if hasattr(analyzer.optimizer, '_generate_scenarios') \
                    else analyzer.load_disturbance_scenarios(n_scenarios)

                for ddpc_name, controller in ddpc_controllers.items():

                    logger.info(f"  Running {ddpc_name}...")
                    ddpc_points = []
                    ddpc_hvs = []

                    for scenario in scenarios:
                        episode_points = []
                        for comfort_target in [0.5, 1.0, 2.0, 4.0, 8.0]:
                            try:
                                initial_state = {
                                    'Temperature': 0.5 + np.random.normal(0, 0.05),
                                    'Humidity': 0.5 + np.random.normal(0, 0.05),
                                    'AirQuality': 0.7 + np.random.normal(0, 0.05),
                                }
                                action = controller.get_action(initial_state, comfort_target)
                                raw_kwh, raw_dh = analyzer.optimizer._simulate_episode(
                                    action, scenario, comfort_target)
                                dh = analyzer.optimizer._normalize_comfort_priority(raw_dh)

                                if dh <= comfort_target:
                                    episode_points.append(ParetoPoint(
                                        kwh=raw_kwh, dh=dh,
                                        policy=ddpc_name,
                                        episode=scenario.get('episode_id', 0),
                                        constraint_target=comfort_target,
                                        feasible=True,
                                    ))
                            except Exception:
                                continue

                        if episode_points:
                            frontier = analyzer._pareto_filter(episode_points)
                            ddpc_points.extend(frontier)
                            hv = analyzer._calculate_hypervolume(frontier)
                            ddpc_hvs.append(hv)

                    results['pareto_points'][ddpc_name] = ddpc_points
                    results['hypervolumes'][ddpc_name] = ddpc_hvs

                    if ddpc_hvs:
                        mean_hv = np.mean(ddpc_hvs)
                        bootstrap = [np.mean(np.random.choice(ddpc_hvs, len(ddpc_hvs), replace=True))
                                     for _ in range(1000)]
                        ci = np.percentile(bootstrap, [2.5, 97.5])
                        results['statistics'][ddpc_name] = {
                            'mean_hypervolume': mean_hv,
                            'std_hypervolume': np.std(ddpc_hvs),
                            'ci_lower': ci[0],
                            'ci_upper': ci[1],
                            'n_episodes': len(ddpc_hvs),
                        }
                        logger.info(f"  {ddpc_name}: HV={mean_hv:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

            except Exception as e:
                logger.error(f"  DDPC baselines failed: {e}")
                import traceback
                traceback.print_exc()

        # Rename ParetoAnalyzer policy names to match our convention BEFORE plotting
        name_map = {
            'GRID_Causal': 'PolicyGRID',
            'WorldModel_Only': 'PolicyGRID-O',
            'ASHRAE_PID': 'PID',
        }
        for old_name, new_name in name_map.items():
            if old_name in results['pareto_points']:
                results['pareto_points'][new_name] = results['pareto_points'].pop(old_name)
            if old_name in results['hypervolumes']:
                results['hypervolumes'][new_name] = results['hypervolumes'].pop(old_name)
            if old_name in results.get('statistics', {}):
                results['statistics'][new_name] = results['statistics'].pop(old_name)
        # Update analyzer's internal dicts so plots use new names
        analyzer.pareto_points = results['pareto_points']
        analyzer.hypervolumes = results['hypervolumes']
        analyzer.statistics = results.get('statistics', {})

        # Generate plots (now includes DDPC)
        policy_out = str(out_dir / 'policy')
        analyzer.generate_plots(output_dir=policy_out)

        # Compute operational metrics per policy (MO, satisfaction %, energy %)
        # Collect global max for normalization across all policies
        all_kwh = [p.kwh for pts in results['pareto_points'].values() for p in pts]
        all_dh = [p.dh for pts in results['pareto_points'].values() for p in pts]
        global_max_kwh = max(all_kwh) if all_kwh else 1.0
        global_max_dh = max(all_dh) if all_dh else 1.0

        policy_metrics_rows = []
        for policy, points in results['pareto_points'].items():
            if not points:
                continue
            kwh_vals = [p.kwh for p in points]
            dh_vals = [p.dh for p in points]
            mean_kwh = np.mean(kwh_vals)
            mean_dh = np.mean(dh_vals)
            # Normalize against global worst for comparable MO
            energy_norm = mean_kwh / global_max_kwh if global_max_kwh > 0 else 0
            comfort_norm = mean_dh / global_max_dh if global_max_dh > 0 else 0
            energy_pct = energy_norm * 100
            sat_pct = max(0, (1 - comfort_norm) * 100)
            mo_score = 0.6 * (sat_pct / 100) + 0.4 * (1 - energy_norm)
            policy_metrics_rows.append({
                'method': policy,
                'MO': round(mo_score, 4),
                'satisfaction_pct': round(sat_pct, 1),
                'energy_pct': round(energy_pct, 1),
                'mean_kwh': round(mean_kwh, 2),
                'mean_dh': round(mean_dh, 4),
                'HV': results.get('statistics', {}).get(policy, {}).get(
                    'mean_hypervolume', 0),
                'n_pareto_points': len(points),
            })

        policy_df = pd.DataFrame(policy_metrics_rows)
        policy_df.to_csv(out_dir / 'policy_metrics.csv', index=False)
        logger.info(f"\n  Policy metrics:")
        for _, row in policy_df.sort_values('MO', ascending=False).iterrows():
            logger.info(f"    {row['method']:20s} MO={row['MO']:.3f}  "
                        f"Sat={row['satisfaction_pct']:.1f}%  "
                        f"Eng={row['energy_pct']:.1f}%  "
                        f"HV={row['HV']:.3f}")

        # Generate policy metric plots
        try:
            plot_policy_metrics(policy_df, sim_name, out_dir)
        except Exception as e:
            logger.error(f"  Policy plots failed: {e}")

        # Save results
        json_results = {}
        for policy, points in results['pareto_points'].items():
            json_results[policy] = [{
                'kwh': p.kwh, 'dh': p.dh,
                'episode': p.episode,
                'constraint_target': p.constraint_target,
                'feasible': p.feasible,
            } for p in points]

        with open(out_dir / 'policy_results.json', 'w') as f:
            json.dump({
                'pareto_points': json_results,
                'hypervolumes': results['hypervolumes'],
                'statistics': results['statistics'],
            }, f, indent=2)

        # Log summary
        for policy, stats in results.get('statistics', {}).items():
            logger.info(f"  {policy}: HV={stats['mean_hypervolume']:.3f} "
                        f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")

    except Exception as e:
        logger.error(f"[{sim_name}] Policy analysis failed: {e}")
        import traceback
        traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# PER-SIM PLOTS
# ═════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Professional NeurIPS style — muted palette, no heavy hatching
COLORS = {
    'PolicyGRID': '#c0392b', 'PolicyGRID-O': '#e67e22',
    'PC': '#2980b9', 'SAM': '#27ae60', 'GIES': '#8e44ad',
    'ICP': '#2c3e50', 'NOTEARS-I': '#16a085', 'ABCD': '#7f8c8d',
    'JCI': '#34495e', 'Causal Bandits': '#95a5a6', 'IID': '#d35400',
    'PID': '#bdc3c7', 'Correlation': '#95a5a6',
    'DDPC_Behavioral': '#1abc9c', 'DDPC_Subspace': '#3498db',
    'DDPC_Neural': '#9b59b6',
}
# Light fill patterns for B&W distinguishability (subtle, not heavy)
HATCHES = ['', '///', '...', 'xxx', '\\\\\\', '|||', '---', '+++',
           'ooo', 'OOO', '///']

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '-',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_discovery_bars(df, sim_name, out_dir):
    """Per-sim: side-by-side F1 and SHD bar charts."""
    n_seeds = df.groupby('method')['seed'].nunique().max()
    ci_scale = 1.96 / np.sqrt(max(n_seeds, 1))  # 95% CI = ±1.96*std/√n

    summary = df.groupby('method').agg(
        F1_mean=('F1', 'mean'), F1_std=('F1', 'std'),
        SHD_mean=('SHD', 'mean'), SHD_std=('SHD', 'std'),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(3.5, len(summary) * 0.38)))

    # F1 bars (sorted by F1) with 95% CI
    s1 = summary.sort_values('F1_mean', ascending=True)
    colors = [COLORS.get(m, '#bdc3c7') for m in s1['method']]
    bars1 = ax1.barh(range(len(s1)), s1['F1_mean'],
                     xerr=s1['F1_std'].fillna(0) * ci_scale,
                     color=colors, edgecolor='#2c3e50', linewidth=0.4,
                     capsize=2, error_kw={'linewidth': 0.8})
    for i, bar in enumerate(bars1):
        bar.set_hatch(HATCHES[i % len(HATCHES)])
        val = s1.iloc[i]['F1_mean']
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{val:.2f}', va='center', fontsize=8, color='#2c3e50')
    ax1.set_yticks(range(len(s1)))
    ax1.set_yticklabels(s1['method'], fontsize=9)
    ax1.set_xlabel('F1 Score', fontsize=10)
    ax1.set_title('F1 Score (higher is better)', fontsize=11)
    ax1.set_xlim(0, 1.05)

    # SHD bars (sorted by SHD, ascending = best)
    s2 = summary.sort_values('SHD_mean', ascending=False)
    colors2 = [COLORS.get(m, '#bdc3c7') for m in s2['method']]
    bars2 = ax2.barh(range(len(s2)), s2['SHD_mean'],
                     xerr=s2['SHD_std'].fillna(0) * ci_scale,
                     color=colors2, edgecolor='#2c3e50', linewidth=0.4,
                     capsize=2, error_kw={'linewidth': 0.8})
    for i, bar in enumerate(bars2):
        bar.set_hatch(HATCHES[i % len(HATCHES)])
        val = s2.iloc[i]['SHD_mean']
        ax2.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}', va='center', fontsize=8, color='#2c3e50')
    ax2.set_yticks(range(len(s2)))
    ax2.set_yticklabels(s2['method'], fontsize=9)
    ax2.set_xlabel('SHD', fontsize=10)
    ax2.set_title('Structural Hamming Distance (lower is better)', fontsize=11)

    fig.suptitle(f'{sim_name}', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'fig_discovery.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved fig_discovery.png")


def plot_policy_metrics(df, sim_name, out_dir):
    """Generate 3 policy metric plots: MO, satisfaction vs energy, HV."""
    df = df.sort_values('MO', ascending=True)

    # 1. MO score bar chart
    fig, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.4)))
    colors = [COLORS.get(m, '#bdc3c7') for m in df['method']]
    bars = ax.barh(range(len(df)), df['MO'],
                   color=colors, edgecolor='#2c3e50', linewidth=0.4)
    for i, bar in enumerate(bars):
        bar.set_hatch(HATCHES[i % len(HATCHES)])
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{df.iloc[i]['MO']:.3f}", va='center', fontsize=8, color='#2c3e50')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['method'], fontsize=9)
    ax.set_xlabel('Multi-Objective Score', fontsize=10)
    ax.set_title(f'Policy MO Score — {sim_name}', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.05)
    fig.savefig(out_dir / 'fig_policy_mo.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2. Satisfaction vs Energy scatter
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*', 'X']
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (_, row) in enumerate(df.iterrows()):
        m = row['method']
        ax.scatter(row['energy_pct'], row['satisfaction_pct'],
                   color=COLORS.get(m, '#bdc3c7'), s=80, zorder=5,
                   edgecolors='#2c3e50', linewidth=0.6,
                   marker=markers[i % len(markers)], label=m)
    ax.set_xlabel('Energy (%)', fontsize=10)
    ax.set_ylabel('Satisfaction (%)', fontsize=10)
    ax.set_title(f'Satisfaction vs Energy — {sim_name}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.9, loc='best')
    fig.savefig(out_dir / 'fig_policy_sat_vs_energy.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3. HV bar chart
    hv_df = df[df['HV'] > 0].sort_values('HV', ascending=True)
    if len(hv_df) > 0:
        fig, ax = plt.subplots(figsize=(7, max(3, len(hv_df) * 0.4)))
        colors = [COLORS.get(m, '#bdc3c7') for m in hv_df['method']]
        bars = ax.barh(range(len(hv_df)), hv_df['HV'],
                       color=colors, edgecolor='#2c3e50', linewidth=0.4)
        for i, bar in enumerate(bars):
            bar.set_hatch(HATCHES[i % len(HATCHES)])
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{hv_df.iloc[i]['HV']:.1f}", va='center', fontsize=8, color='#2c3e50')
        ax.set_yticks(range(len(hv_df)))
        ax.set_yticklabels(hv_df['method'], fontsize=9)
        ax.set_xlabel('Hypervolume', fontsize=10)
        ax.set_title(f'Policy Hypervolume — {sim_name}', fontsize=11, fontweight='bold')
        fig.savefig(out_dir / 'fig_policy_hv.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    logger.info(f"  Saved policy plots")


# ═════════════════════════════════════════════════════════════════════════════
# M5: COMPUTE TRACKING
# ═════════════════════════════════════════════════════════════════════════════

class ComputeTracker:
    def __init__(self):
        self.timings = {}
        self.start_time = time.time()

    def start(self, phase):
        self.timings[phase] = {'start': time.time()}

    def stop(self, phase):
        if phase in self.timings:
            self.timings[phase]['end'] = time.time()
            self.timings[phase]['elapsed'] = (
                self.timings[phase]['end'] - self.timings[phase]['start']
            )

    def report(self):
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        total = time.time() - self.start_time
        return {
            'total_wall_clock_s': total,
            'peak_memory_mb': peak_mem,
            'phases': {k: v.get('elapsed', 0) for k, v in self.timings.items()},
            'system': {
                'platform': sys.platform,
                'python': sys.version.split()[0],
                'cpu_count': os.cpu_count(),
            },
            'llm': {
                'model': 'gpt-3.5-turbo',
                'temperature': 1.0,
            },
        }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Master experiment suite')
    parser.add_argument('--sims', nargs='+', default=list(SIM_CONFIGS.keys()),
                        help='Simulators to run')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--skip-discovery', action='store_true',
                        help='Skip discovery, load from saved edges')
    parser.add_argument('--discovery-only', action='store_true',
                        help='Only run discovery (no monitoring/policy)')
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tracker = ComputeTracker()

    all_discovery_rows = []

    for sim_name in args.sims:
        logger.info(f"\n{'='*60}")
        logger.info(f"  SIMULATOR: {sim_name}")
        logger.info(f"{'='*60}")

        config = get_config(sim_name)
        sim_out = OUT_ROOT / sim_name
        sim_out.mkdir(parents=True, exist_ok=True)

        # ── Load data ──
        tracker.start(f'{sim_name}_load')
        data, gt_set, smap = load_sim_data(config)
        tracker.stop(f'{sim_name}_load')
        logger.info(f"  Data: {len(data)} rows, {len(data.columns)} cols, "
                     f"GT: {len(gt_set)} edges")

        # ── Discovery ──
        tracker.start(f'{sim_name}_discovery')

        if args.skip_discovery:
            edges_path = sim_out / 'discovered_edges.json'
            if edges_path.exists():
                with open(edges_path) as f:
                    raw = json.load(f)
                all_edges = {k: {tuple(e) for e in v} for k, v in raw.items()}
                logger.info(f"  Loaded saved edges from {edges_path}")
                discovery_df = pd.read_csv(sim_out / 'discovery_metrics.csv')
            else:
                logger.warning(f"  No saved edges — running discovery")
                discovery_df, all_edges = run_discovery_for_config(
                    config, data, seeds=args.seeds)
        else:
            discovery_df, all_edges = run_discovery_for_config(
                config, data, seeds=args.seeds)

        tracker.stop(f'{sim_name}_discovery')

        # Save discovery results
        discovery_df.to_csv(sim_out / 'discovery_metrics.csv', index=False)
        edges_serializable = {k: [list(e) for e in v]
                              for k, v in all_edges.items()}
        with open(sim_out / 'discovered_edges.json', 'w') as f:
            json.dump(edges_serializable, f, indent=2)

        all_discovery_rows.append(discovery_df)

        # Generate per-sim discovery bar chart
        try:
            plot_discovery_bars(discovery_df, sim_name, sim_out)
        except Exception as e:
            logger.error(f"  Discovery plot failed: {e}")
        logger.info(f"\n  Discovery summary for {sim_name}:")
        for _, row in discovery_df.groupby('method').agg(
                {'F1': 'mean', 'SHD': 'mean'}).iterrows():
            pass  # logged inline above

        # ── Monitoring (if applicable) ──
        if not args.discovery_only and config['regime_vars']:
            tracker.start(f'{sim_name}_monitoring')
            mon_df = run_monitoring_for_config(config, all_edges, data, smap)
            tracker.stop(f'{sim_name}_monitoring')
            if mon_df is not None:
                mon_df.to_csv(sim_out / 'monitoring_metrics.csv', index=False)

        # ── Policy ──
        if not args.discovery_only and config.get('actuator_vars'):
            tracker.start(f'{sim_name}_policy')

            if sim_name == 'smart_building_rich':
                # smart_building_rich: use neurips_final_experiments.py
                # (HVAC/lighting duty cycle → kWh, fixed 22°C setpoint → DH)
                logger.info(f"[{sim_name}] Policy: delegating to neurips_final_experiments")
                try:
                    edges_file = sim_out / 'discovered_edges.json'
                    if edges_file.exists():
                        import subprocess as sp
                        sp.run([
                            sys.executable, 'neurips_final_experiments.py',
                            '--skip-discovery', '--edges-path', str(edges_file),
                        ], check=False, timeout=7200)
                        # Copy results
                        nfe_out = ROOT / 'results' / 'neurips_final'
                        for subdir in ['exp1_benchmarks', 'exp2_ablation', 'posthoc']:
                            src = nfe_out / subdir
                            dst = sim_out / subdir
                            if src.exists():
                                import shutil
                                if dst.exists():
                                    shutil.rmtree(dst)
                                shutil.copytree(src, dst)
                        # Copy summary
                        for f in ['summary.txt', 'compute_resources.json']:
                            src = nfe_out / f
                            if src.exists():
                                shutil.copy2(src, sim_out / f)
                    else:
                        logger.warning(f"  No edges file for policy — skipping")
                except Exception as e:
                    logger.error(f"  smart_building_rich policy failed: {e}")

            else:
                # Small sims + open_window: use ParetoAnalyzer
                # (sim-reported EnergyConsumption → kWh, action-derived setpoint → DH)
                logger.info(f"[{sim_name}] Policy: running ParetoAnalyzer")
                try:
                    run_policy_pareto(config, all_edges, data, sim_out)
                except Exception as e:
                    logger.error(f"  Policy failed for {sim_name}: {e}")

            tracker.stop(f'{sim_name}_policy')

    # ── Cross-sim summary ──
    if all_discovery_rows:
        cross_df = pd.concat(all_discovery_rows, ignore_index=True)
        cross_out = OUT_ROOT / 'cross_sim'
        cross_out.mkdir(parents=True, exist_ok=True)
        cross_df.to_csv(cross_out / 'cross_sim_discovery.csv', index=False)

        # Print summary table
        logger.info(f"\n{'='*60}")
        logger.info("  CROSS-SIMULATOR DISCOVERY SUMMARY")
        logger.info(f"{'='*60}")

        summary = cross_df.groupby(['sim', 'method']).agg(
            F1_mean=('F1', 'mean'),
            F1_std=('F1', 'std'),
            SHD_mean=('SHD', 'mean'),
        ).reset_index()

        # Best method per sim
        for sim in summary['sim'].unique():
            sim_data = summary[summary['sim'] == sim]
            best = sim_data.loc[sim_data['F1_mean'].idxmax()]
            n_vars = len(get_config(sim)['all_vars'])
            logger.info(f"  {sim} ({n_vars} vars): best={best['method']} "
                        f"F1={best['F1_mean']:.3f}")

    # ── Save compute resources ──
    compute = tracker.report()
    with open(OUT_ROOT / 'compute_resources.json', 'w') as f:
        json.dump(compute, f, indent=2)

    logger.info(f"\nTotal wall-clock: {compute['total_wall_clock_s']:.1f}s")
    logger.info(f"Peak memory: {compute['peak_memory_mb']:.1f} MB")
    logger.info(f"\nResults saved to {OUT_ROOT}")


if __name__ == '__main__':
    main()
