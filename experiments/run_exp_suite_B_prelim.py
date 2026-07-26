#!/usr/bin/env python3
"""
Experiment Suite B: Policy Attribution + Counterfactual Checks
===============================================================

Maps to the advisor's checklist items B1 (Must-have) and B2 (Nice-to-have).

B1 — Policy Improvement Attribution:
  "Does causal structure help control, and does validation make it better?"

B2 — Counterfactual Sanity Checks:
  "For each validated edge, does the SEM-predicted direction of effect
   match the observed direction from simulator interventions?"

Runs the REAL CWMIntegratedPipeline (discovery -> validation -> monitoring)
for each simulator and seed, then evaluates all policy arms using the
DISCOVERED causal structure (no ground-truth injection).

Arms (the advisor's 4-way comparison):
  +---------------------+-----------------------------------------------+
  | Checklist Role       | Arm Name                                     |
  +---------------------+-----------------------------------------------+
  | Static baseline      | ASHRAE-PID       (PID-style thermostat)      |
  | Strong learned (M2)  | DDPC_Behavioral  (DeePC, Willems' lemma)    |
  |                      | DDPC_Subspace    (N4SID + linear MPC)       |
  |                      | DDPC_Neural      (MLP dynamics + CEM)       |
  |                      | DDPC_PETS        (Ensemble MLP + TS-inf)    |
  | Obs-only DAG         | GRID_ObsOnly     (method DAG union, no val) |
  | Validated DAG        | GRID             (intervention-validated)    |
  | Full PolicyGRID loop | GRID_RegimeAware (validated + Bayesian       |
  |                      |   regime detection + BMA switching)          |
  +---------------------+-----------------------------------------------+

Expected staircase (the advisor's Figure 3):
  ASHRAE-PID < DDPC_* < GRID_ObsOnly < GRID <= GRID_RegimeAware

Simulators:
  1. open_window      -- JS sim, window regime
  2. smart_room       -- JS sim, no regime
  3. smart_room_noise -- JS sim + measurement noise
  4. hidden_vars      -- JS sim + hidden confounders
  5. ashrae           -- ASHRAE building energy (RF-based simulation)

Metrics per (seed x arm):
  - Satisfaction %   (higher is better)
  - Energy %         (lower is better)
  - MO-Score:        0.6 * (sat/100) + 0.4 * (1 - eng/100)
  - CV (mean):       avg constraint violation degree (Deb, IEEE TEC 2002)
  - CVR:             fraction of timesteps with any violation
  - Hypervolume:     Pareto 2D area (sat vs energy)
  - Intervention count from discovery phase

Reproducibility (M5):
  - >=5 seeds per arm, report mean +/- std
  - Results saved to neurips_results/b1_prelim_*.csv + *.png

Usage:
  python run_exp_suite_B_prelim.py                              # open_window, 5 seeds
  python run_exp_suite_B_prelim.py --sim smart_room             # smart_room sim
  python run_exp_suite_B_prelim.py --sim smart_room_noise       # with measurement noise
  python run_exp_suite_B_prelim.py --sim hidden_vars            # hidden confounders
  python run_exp_suite_B_prelim.py --dataset ashrae             # ASHRAE building energy
  python run_exp_suite_B_prelim.py --seeds 10                   # more seeds
  python run_exp_suite_B_prelim.py --episodes 10                # more policy episodes
  python run_exp_suite_B_prelim.py --include-pets               # include DDPC_PETS (slow)
  python run_exp_suite_B_prelim.py --all-sims                   # ALL sims in sequence
"""
import argparse
import json
import logging
import os
import platform
import random
import time
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Suppress macOS Accelerate BLAS warnings
for _m in ('divide by zero encountered in matmul',
           'overflow encountered in matmul',
           'invalid value encountered in matmul'):
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=f'.*{_m}.*')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("app.log", mode='w'),
        logging.StreamHandler()
    ]
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logger = logging.getLogger('run_exp_suite_B_prelim')
logger.setLevel(logging.INFO)

from src.pipeline_cwm import create_cwm_pipeline
from experiments_neurips import NeurIPSExperiments, ARM_COLORS


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── M5: Compute-resource tracking (the advisor's checklist) ───────────────────────

def _get_system_info():
    """Collect machine specs for NeurIPS reproducibility (M5)."""
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count_logical': os.cpu_count(),
        'python_version': platform.python_version(),
        'gpu': 'None (CPU only)',
    }
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['ram_total_gb'] = round(mem.total / (1024 ** 3), 1)
    except ImportError:
        info['ram_total_gb'] = 'unknown (install psutil)'
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['gpu'] = 'Apple MPS (Metal)'
    except ImportError:
        pass
    return info


def _get_llm_stats(pipeline_obj):
    """Extract LLM call counts and token usage from the pipeline."""
    stats = {
        'model': 'gpt-3.5-turbo',
        'temperature': 1,
        'top_p': 0.9,
        'max_tokens': 1000,
        'api_calls': 0,
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0,
    }
    # Walk the pipeline to find the GPT client
    try:
        base = pipeline_obj.pipeline if hasattr(pipeline_obj, 'pipeline') else pipeline_obj
        for gen in base.generators.values():
            if hasattr(gen, 'client'):
                c = gen.client
                stats['api_calls'] += getattr(c, 'call_count', 0)
                stats['prompt_tokens'] += getattr(c, 'total_prompt_tokens', 0)
                stats['completion_tokens'] += getattr(c, 'total_completion_tokens', 0)
                stats['total_tokens'] += getattr(c, 'total_tokens', 0)
    except Exception:
        pass
    return stats


# ── Per-simulator configuration ──────────────────────────────────────────────
# No GT edges — edges are DISCOVERED by the full pipeline.

SIM_CONFIGS = {
    'open_window': {
        'sim_path': 'open_window.js',
        'obs_data_path': 'data_regen/challenge1_data_10k_processed.csv',
        'scaling_path': 'data_regen/challenge1_data_10k_processed_scaling.csv',
        'dataset_type': 'open_window',
        'has_regime': True,
        'data_in_physical_units': True,   # Temperature 9.6-22.3 °C
        'actuator_vars': [],
        'non_intervenable_vars': [
            'windowopen', 'pmv', 'outdoortemperature',
            'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
    },
    'smart_room': {
        'sim_path': 'js/smart_room.js',
        'obs_data_path': 'data_regen/smart_room_processed.csv',
        'scaling_path': 'data_regen/smart_room_processed_scaling.csv',
        'dataset_type': 'open_window',   # same schema (Temp/Hum/AQ/Energy/Sat)
        'has_regime': False,
        'data_in_physical_units': False,  # Temperature 0.05-0.95 (internal [0,1])
        'actuator_vars': [],
        'non_intervenable_vars': [
            'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
    },
    'smart_room_noise': {
        'sim_path': 'js/smart_room_noise.js',
        'obs_data_path': 'data_regen/smart_room_noise_processed.csv',
        'scaling_path': 'data_regen/smart_room_noise_processed_scaling.csv',
        'dataset_type': 'open_window',
        'has_regime': False,
        'data_in_physical_units': False,  # Temperature 0.05-0.95 (internal [0,1])
        'actuator_vars': [],
        'non_intervenable_vars': [
            'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
    },
    'hidden_vars': {
        'sim_path': 'js/smart_room_hidden_vars.js',
        'obs_data_path': 'data_regen/hidden_vars_processed.csv',
        'scaling_path': 'data_regen/hidden_vars_processed_scaling.csv',
        'dataset_type': 'open_window',
        'has_regime': False,
        'data_in_physical_units': False,  # Temperature 0.05-0.95 (internal [0,1])
        'actuator_vars': [],
        'non_intervenable_vars': [
            'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
    },
    'ashrae': {
        'sim_path': None,
        'obs_data_path': 'data/ashrae_data_processed.csv',
        'scaling_path': None,
        'dataset_type': 'ashrae',
        'has_regime': False,
        'data_in_physical_units': True,   # physical weather measurements
        'actuator_vars': [],
        'non_intervenable_vars': [],
    },
    'smart_building_rich': {
        'sim_path': 'js/smart_building_rich.js',
        'obs_data_path': 'data_regen/smart_building_rich_processed.csv',
        'scaling_path': 'data_regen/smart_building_rich_processed_scaling.csv',
        'dataset_type': 'smart_building_rich',
        'has_regime': True,
        'data_in_physical_units': True,   # Temperature °C, CO2 ppm, etc.
        'actuator_vars': ['hvacpower', 'lightingpower'],
        'non_intervenable_vars': [
            'outdoortemp', 'solarradiation', 'occupancy', 'windowposition',
            'pmv', 'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
    },
}

# Default epsilon-constraint comfort targets (DH budget) for Pareto sweep
DEFAULT_COMFORT_TARGETS = [0.5, 0.8, 1.0, 1.5, 2.0]

# ── Discovery ────────────────────────────────────────────────────────────────

def run_discovery_pipeline(cfg, seed):
    """Run full CWM pipeline (discovery + monitoring) for one seed.

    Returns dict with:
        pipeline, validated_edges, obs_only_edges, h0_edges, h1_edges,
        intervention_count, discovery_time_s
    """
    random.seed(seed)
    np.random.seed(seed)

    api_key = 'YOUR_OPENAI_API_KEY'
    obs_data_path = os.path.join(_BASE_DIR, cfg['obs_data_path'])
    obs_df = pd.read_csv(obs_data_path)
    logger.info(f"Discovery: {len(obs_df)} rows, cols={list(obs_df.columns)}")

    t0 = time.time()

    sim_path = (os.path.join(_BASE_DIR, cfg['sim_path'])
                if cfg['sim_path'] else None)

    pipeline = create_cwm_pipeline(
        csv_data=obs_df,
        api_key=api_key,
        smart_room_path=sim_path,
        dataset_type=cfg['dataset_type'],
        max_iterations=10,
        alpha=0.5,
        beta=0.5,
        effect_threshold=0.1,
        actuator_vars=cfg.get('actuator_vars', []),
        non_intervenable_vars=cfg.get('non_intervenable_vars', []),
    )

    # Phase 1+2: Discovery + Validation
    t_disc = time.time()
    final_dag, final_metrics = pipeline.run_with_cwm()
    disc_wall_s = time.time() - t_disc

    # Phase 3: Monitoring (only for regime datasets)
    t_mon = time.time()
    if cfg['has_regime']:
        try:
            pipeline.monitor_regime_changes(duration=300, sample_interval=5)
        except Exception as e:
            logger.warning(f"Monitoring failed (continuing): {e}")
    mon_wall_s = time.time() - t_mon

    discovery_time = time.time() - t0

    # Extract validated edges
    validated_edges = set(pipeline.pipeline.validated_edges)

    # Extract method DAG union (obs-only edges for GRID_ObsOnly)
    method_dags = pipeline.pipeline.get_method_dags()
    obs_only_edges = set()
    for G in method_dags.values():
        obs_only_edges.update(G.edges())

    # Extract H0/H1 edges (if regime-aware)
    h0_edges = None
    h1_edges = None
    if hasattr(pipeline, 'H0_graph') and pipeline.H0_graph is not None:
        h0_edges = set(pipeline.H0_graph.edges())
    if hasattr(pipeline, 'H1_graph') and pipeline.H1_graph is not None:
        h1_edges = set(pipeline.H1_graph.edges())

    # Count total interventions
    intervention_count = sum(
        len(r) for r in pipeline.pipeline.tester.intervention_results.values()
    )

    logger.info(f"Discovery done in {discovery_time:.1f}s: "
                f"validated={len(validated_edges)}, obs_only={len(obs_only_edges)}, "
                f"interventions={intervention_count}")

    return {
        'pipeline': pipeline,
        'validated_edges': validated_edges,
        'obs_only_edges': obs_only_edges,
        'h0_edges': h0_edges,
        'h1_edges': h1_edges,
        'intervention_count': intervention_count,
        'discovery_time_s': discovery_time,
        'timings': {
            'discovery_validation_s': disc_wall_s,
            'monitoring_s': mon_wall_s,
            'total_s': discovery_time,
        },
        'llm_stats': _get_llm_stats(pipeline),
    }


# ── Policy evaluation ────────────────────────────────────────────────────────

def _load_scaling_ranges(cfg):
    """Load training-data scaling from the CSV produced by data_regen.

    Returns a dict: {feature_name: {'min': float, 'max': float}} or None.
    """
    scaling_path = cfg.get('scaling_path')
    if not scaling_path:
        return None
    full = os.path.join(_BASE_DIR, scaling_path)
    if not os.path.exists(full):
        logger.warning(f"Scaling file not found: {full}")
        return None
    df = pd.read_csv(full)
    out = {}
    for _, row in df.iterrows():
        out[row['feature']] = {'min': row['data_min'], 'max': row['data_max']}
    logger.info(f"Loaded scaling ranges from {scaling_path}: {list(out.keys())}")
    return out


def _compute_mo(sat, eng, dataset_type):
    """MO-Score helper."""
    if dataset_type == 'ashrae':
        return -eng
    return 0.6 * (sat / 100) + 0.4 * (1 - eng / 100)


def _regime_rows_from_steps(steps):
    """Partition per-step data into H0 and H1 sub-metrics.

    Returns dict: {0: {sat, eng, cv, kwh, dh, n_steps},
                   1: {sat, eng, cv, kwh, dh, n_steps}}
    """
    regime_agg = {}
    for r_val in (0, 1):
        r_steps = [s for s in steps if s['regime'] == r_val]
        n = len(r_steps)
        if n == 0:
            continue
        regime_agg[r_val] = {
            'sat': sum(s['sat'] for s in r_steps) / n,
            'eng': sum(s['eng'] for s in r_steps) / n,
            'cv': sum(s['cv'] for s in r_steps) / n,
            'kwh': sum(s['kwh'] for s in r_steps),
            'dh': sum(s['dh'] for s in r_steps),
            'n_steps': n,
        }
    return regime_agg


def run_policy_episodes(discovery_result, cfg, n_episodes, seed, include_pets,
                        comfort_targets=None):
    """Run all policy arms for one discovered pipeline, return list of row dicts.

    Epsilon-constraint Pareto sweep
    --------------------------------
    When comfort_targets is provided (e.g. [0.5, 0.8, 1.0, 1.5, 2.0]),
    each arm is evaluated at every comfort target ε.  For GRID, ε maps to
    objective weights (tighter comfort → higher satisfaction weight).
    For DDPC, ε is passed as the comfort_target parameter.
    ASHRAE is static, so it runs once (duplicated across ε for consistency).

    Returns
    -------
    rows : list[dict]
        Per-episode aggregate rows (arm, seed, comfort_target, sat, eng, ...).
    arm_timings : dict
    regime_rows : list[dict]
        Per-episode per-regime rows (only when cfg['has_regime']).
        Keys: arm, discovery_seed, episode_seed, comfort_target, regime,
              satisfaction, energy, mo_score, kwh, dh, n_steps.
    step_rows : list[dict]
        Per-timestep rows for episode timeline plots (only when cfg['has_regime']).
        Keys: arm, discovery_seed, episode_seed, comfort_target, t,
              sat, eng, regime.
    """
    if comfort_targets is None:
        comfort_targets = [0.8]

    has_regime = cfg.get('has_regime', False)

    pipeline_obj = discovery_result['pipeline']
    obs_only_edges = discovery_result['obs_only_edges']
    h0_edges = discovery_result['h0_edges']
    h1_edges = discovery_result['h1_edges']
    intervention_count = discovery_result['intervention_count']

    obs_df = pd.read_csv(os.path.join(_BASE_DIR, cfg['obs_data_path']))

    # Load training-data scaling so SEM & episode evaluator share [0,1] space
    scaling_ranges = _load_scaling_ranges(cfg)

    # Create NeurIPSExperiments once (DDPC suite is cached internally)
    exp = NeurIPSExperiments(
        pipeline=pipeline_obj.pipeline,
        dataset=obs_df,
        smart_room_path=(os.path.join(_BASE_DIR, cfg['sim_path'])
                         if cfg['sim_path'] else None),
        comfort_target=comfort_targets[0],
        obs_only_edges=obs_only_edges,
        h0_edges=h0_edges,
        h1_edges=h1_edges,
        dataset_type=cfg['dataset_type'],
        scaling_ranges=scaling_ranges,
        data_in_physical_units=cfg.get('data_in_physical_units', True),
    )
    exp._build_policy_arms()

    # Remove PETS if not requested
    if not include_pets and 'DDPC_PETS' in exp.policy_arms:
        del exp.policy_arms['DDPC_PETS']

    rows = []
    regime_rows = []
    step_rows = []
    arm_timings = {}

    # Track which arms are static (ASHRAE) — only need to run once
    static_arms = {'ASHRAE-PID'}

    for ct_idx, ct in enumerate(comfort_targets):
        logger.info(f"  Comfort target ε={ct} ({ct_idx+1}/{len(comfort_targets)})")
        if ct_idx > 0:
            exp.set_comfort_target(ct)
            # Re-remove PETS after rebuild
            if not include_pets and 'DDPC_PETS' in exp.policy_arms:
                del exp.policy_arms['DDPC_PETS']

        arms = list(exp.policy_arms.keys())

        for arm_name in arms:
            # Static arms produce the same result regardless of ε;
            # run once and copy results for each comfort target.
            if arm_name in static_arms and ct_idx > 0:
                for ep_seed in range(n_episodes):
                    # Copy from ct_idx=0 results
                    base = next(
                        r for r in rows
                        if r['arm'] == arm_name
                        and r['episode_seed'] == ep_seed
                        and r['discovery_seed'] == seed
                        and r['comfort_target'] == comfort_targets[0]
                    )
                    rows.append({**base, 'comfort_target': ct})
                    # Copy regime rows too
                    if has_regime:
                        for rr in regime_rows:
                            if (rr['arm'] == arm_name
                                    and rr['episode_seed'] == ep_seed
                                    and rr['discovery_seed'] == seed
                                    and rr['comfort_target'] == comfort_targets[0]):
                                regime_rows.append(
                                    {**rr, 'comfort_target': ct})
                continue

            fn = exp.policy_arms[arm_name]
            t_arm = time.time()
            for ep_seed in range(n_episodes):
                if has_regime:
                    sat, eng, cv_mean_val, cvr, kwh, dh, steps = \
                        exp._evaluate_policy_episode_regime(
                            fn, arm_name, ep_seed)
                else:
                    sat, eng, cv_mean_val, cvr, kwh, dh = \
                        exp._evaluate_policy_episode_with_cv(
                            fn, arm_name, ep_seed)
                    steps = None

                mo = _compute_mo(sat, eng, cfg['dataset_type'])

                rows.append({
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
                    'intervention_count': intervention_count,
                })

                # Regime-partitioned rows
                if steps is not None:
                    regime_agg = _regime_rows_from_steps(steps)
                    for r_val, agg in regime_agg.items():
                        r_mo = _compute_mo(agg['sat'], agg['eng'],
                                           cfg['dataset_type'])
                        regime_rows.append({
                            'arm': arm_name,
                            'discovery_seed': seed,
                            'episode_seed': ep_seed,
                            'comfort_target': ct,
                            'regime': r_val,
                            'satisfaction': agg['sat'],
                            'energy': agg['eng'],
                            'mo_score': r_mo,
                            'cv_mean': agg['cv'],
                            'kwh': agg['kwh'],
                            'dh': agg['dh'],
                            'n_steps': agg['n_steps'],
                            'intervention_count': intervention_count,
                        })

                    # Per-step rows for timeline plots
                    for s in steps:
                        step_rows.append({
                            'arm': arm_name,
                            'discovery_seed': seed,
                            'episode_seed': ep_seed,
                            'comfort_target': ct,
                            't': s['t'],
                            'sat': s['sat'],
                            'eng': s['eng'],
                            'regime': s['regime'],
                        })

            arm_wall_s = time.time() - t_arm
            key = f"{arm_name}@ε={ct}"
            arm_timings[key] = round(arm_wall_s, 2)
            logger.info(f"    {arm_name} (ε={ct}): {n_episodes} ep in "
                        f"{arm_wall_s:.1f}s")

    return rows, arm_timings, regime_rows, step_rows


# ── B2: Counterfactual sanity checks ─────────────────────────────────────────

def run_b2_counterfactual_checks(discovery_result, cfg, seed):
    """B2: Counterfactual sanity checks (the advisor checklist, Nice-to-have).

    For each validated edge, compare:
      - SEM-predicted direction of effect (sign of linear coefficient beta)
      - Observed direction from simulator interventions

    Returns a DataFrame with one row per validated edge.
    """
    from src.policy_engine import CausalPolicyEngine

    pipeline_obj = discovery_result['pipeline']
    validated_edges = discovery_result['validated_edges']

    if not validated_edges:
        logger.warning(f"B2: No validated edges for seed={seed}, skipping")
        return pd.DataFrame()

    # Get intervention results from the discovery phase
    intervention_results = pipeline_obj.pipeline.tester.intervention_results

    # Fit a policy engine on validated edges to get SEM coefficients
    obs_df = pd.read_csv(os.path.join(_BASE_DIR, cfg['obs_data_path']))
    try:
        engine = CausalPolicyEngine(
            {'validated_edges': validated_edges},
            obs_df,
            use_llm=False,
            action_vars=(['Temperature', 'Humidity', 'AirQuality']
                         if cfg['dataset_type'] != 'ashrae'
                         else ['air_temperature', 'dew_temperature',
                               'sea_level_pressure']),
        )
    except Exception as e:
        logger.warning(f"B2: Policy engine init failed: {e}")
        return pd.DataFrame()

    # Map lowercase edge names to dataset column names (policy engine normalizes)
    col_lookup = {c.lower(): c for c in obs_df.columns}

    rows = []
    for edge in sorted(validated_edges):
        source_raw, target_raw = edge
        source = col_lookup.get(source_raw, source_raw)
        target = col_lookup.get(target_raw, target_raw)

        # ── SEM predicted direction ───────────────────────────────────
        sem_info = engine._sem.get(target, None)
        if sem_info is None:
            sem_info = engine._sem.get(target_raw, None)
        if sem_info is None:
            continue

        _intercept, lin_coeffs, _quad_coeffs, _inter_coeffs, _r2 = sem_info
        sem_coef = lin_coeffs.get(source,
                                  lin_coeffs.get(source_raw, 0.0))
        sem_direction = '+' if sem_coef > 0 else ('-' if sem_coef < 0 else '0')

        # ── Observed direction from interventions ─────────────────────
        # Try both original (lowercase) and proper case edge keys
        edge_results = intervention_results.get((source_raw, target_raw), [])
        if not edge_results:
            edge_results = intervention_results.get((source, target), [])

        observed_effects = []
        for r in edge_results:
            pre = r.get('pre_state', r.get('preInterventionData', {}))
            post = r.get('post_state', r.get('postInterventionData', {}))
            if not pre or not post:
                continue

            # Try proper case first, then lowercase
            s_pre = pre.get(source, pre.get(source_raw, None))
            s_post = post.get(source, post.get(source_raw, None))
            t_pre = pre.get(target, pre.get(target_raw, None))
            t_post = post.get(target, post.get(target_raw, None))

            if any(v is None for v in (s_pre, s_post, t_pre, t_post)):
                continue

            source_change = float(s_post) - float(s_pre)
            target_change = float(t_post) - float(t_pre)

            if abs(source_change) > 1e-6:
                observed_effects.append(target_change / source_change)

        if observed_effects:
            avg_effect = float(np.mean(observed_effects))
            obs_direction = ('+' if avg_effect > 0
                             else '-' if avg_effect < 0 else '0')
            match = (sem_direction == obs_direction)
        else:
            avg_effect = float('nan')
            obs_direction = '?'
            match = None

        rows.append({
            'discovery_seed': seed,
            'edge': f'{source} -> {target}',
            'sem_coefficient': sem_coef,
            'sem_direction': sem_direction,
            'observed_effect': avg_effect,
            'observed_direction': obs_direction,
            'direction_match': match,
            'n_interventions': len(edge_results),
        })

    return pd.DataFrame(rows)


# ── Pareto dominance filter ──────────────────────────────────────────────────

def pareto_filter_2d(points):
    """Return non-dominated (kwh, dh) points (both minimised).

    Parameters
    ----------
    points : list of (kwh, dh) tuples

    Returns
    -------
    list of (kwh, dh) tuples, sorted by kwh ascending.
    """
    if not points:
        return []
    non_dominated = []
    for i, (k1, d1) in enumerate(points):
        dominated = False
        for j, (k2, d2) in enumerate(points):
            if i != j and k2 <= k1 and d2 <= d1 and (k2 < k1 or d2 < d1):
                dominated = True
                break
        if not dominated:
            non_dominated.append((k1, d1))
    non_dominated.sort(key=lambda p: p[0])
    return non_dominated


# ── Hypervolume ──────────────────────────────────────────────────────────────

def _compute_hypervolume_2d(points, ref_point):
    """2D sweepline hypervolume for two minimized objectives.

    Parameters
    ----------
    points : list of (obj1, obj2) tuples
        Each point has two cost values to be minimized.
    ref_point : (r1, r2)
        Reference (nadir) point — must dominate no feasible solution.

    Returns
    -------
    float : dominated hypervolume (higher = better Pareto front).

    Algorithm: sort non-dominated points by obj1 ascending (energy).
    For non-dominated points, obj2 is necessarily descending.
    Each point i contributes a rectangle:
        width  = obj1_{i+1} - obj1_i   (or r1 - obj1_k for last point)
        height = r2 - obj2_i
    """
    if not points:
        return 0.0
    # Keep only points inside the reference box (inclusive boundary)
    points = [(e, c) for e, c in points
              if e <= ref_point[0] and c <= ref_point[1]]
    if not points:
        return 0.0

    # Non-domination filter (O(n²), fine for small sets)
    non_dominated = []
    for p in points:
        if not any(o[0] <= p[0] and o[1] <= p[1]
                   and (o[0] < p[0] or o[1] < p[1])
                   for o in non_dominated):
            non_dominated = [o for o in non_dominated
                             if not (p[0] <= o[0] and p[1] <= o[1]
                                     and (p[0] < o[0] or p[1] < o[1]))]
            non_dominated.append(p)
    if not non_dominated:
        return 0.0

    # Sort by obj1 (energy cost) ascending; obj2 (comfort cost) is then
    # monotonically decreasing on the non-dominated front.
    sorted_pts = sorted(non_dominated, key=lambda p: p[0])

    hypervolume = 0.0
    for i, (e_i, c_i) in enumerate(sorted_pts):
        # Width: gap in obj1 from this point to the next (or ref)
        next_e = sorted_pts[i + 1][0] if i + 1 < len(sorted_pts) else ref_point[0]
        width = next_e - e_i
        # Height: gap from this point's obj2 to reference
        height = ref_point[1] - c_i
        hypervolume += width * height

    return hypervolume


def compute_hypervolume_from_rows(rows):
    """Compute hypervolume per paper A.3.5 — cross-policy Pareto set.

    Each (arm, comfort_target) → one mean point in (kWh, DH) space.
    All points are pooled; the non-dominated set defines the Pareto
    frontier.  HV is the area dominated by this frontier relative to
    reference point r = (1.2, 1.0) after normalisation.

    Returns
    -------
    dict with keys:
        'combined'  : float — HV of the cross-policy Pareto front
        'per_arm'   : dict arm → float — HV of the single arm-mean point
        'frontier'  : list of (kwh, dh) — non-dominated raw points
        'arm_means' : dict arm → (kwh_mean, dh_mean) — per-arm means
    """
    ref_point = (1.2, 1.0)

    # ── 1. Group by (arm, comfort_target) → mean per operating point ──
    groups = defaultdict(lambda: {'kwh': [], 'dh': []})
    for r in rows:
        key = (r['arm'], r.get('comfort_target', 0))
        groups[key]['kwh'].append(r.get('kwh', 0.0))
        groups[key]['dh'].append(r.get('dh', 0.0))

    labeled_pts = []          # [(arm, kwh_mean, dh_mean), ...]
    for (arm, _ct), data in groups.items():
        labeled_pts.append((arm,
                            float(np.mean(data['kwh'])),
                            float(np.mean(data['dh']))))

    # ── 2. Per-arm overall mean (average across comfort_targets) ──
    arm_accum = defaultdict(lambda: {'kwh': [], 'dh': []})
    for arm, k, d in labeled_pts:
        arm_accum[arm]['kwh'].append(k)
        arm_accum[arm]['dh'].append(d)
    arm_means = {arm: (float(np.mean(v['kwh'])), float(np.mean(v['dh'])))
                 for arm, v in arm_accum.items()}

    # ── 3. Normalise using observed maxima ──
    all_kwh = [k for _, k, _ in labeled_pts]
    all_dh  = [d for _, _, d in labeled_pts]
    max_kwh = max(max(all_kwh, default=100.0), 1e-6)
    max_dh  = max(max(all_dh,  default=10.0),  1e-6)

    def _norm(kwh, dh):
        return ((kwh / max_kwh) * 1.2, (dh / max_dh) * 1.0)

    # ── 4. Combined HV across all operating points (paper A.3.5) ──
    all_norm = [_norm(k, d) for _, k, d in labeled_pts]
    combined_hv = _compute_hypervolume_2d(all_norm, ref_point)

    # ── 5. Per-arm HV (single mean point → rectangle to ref) ──
    per_arm_hv = {}
    for arm, (k, d) in arm_means.items():
        ne, nc = _norm(k, d)
        if ne <= ref_point[0] and nc <= ref_point[1]:
            per_arm_hv[arm] = (ref_point[0] - ne) * (ref_point[1] - nc)
        else:
            per_arm_hv[arm] = 0.0

    # ── 6. Raw-space Pareto frontier for plotting ──
    raw_pts = [(k, d) for _, k, d in labeled_pts]
    frontier = pareto_filter_2d(raw_pts)

    return {
        'combined':  combined_hv,
        'per_arm':   per_arm_hv,
        'frontier':  frontier,
        'arm_means': arm_means,
    }


# ── Statistics ───────────────────────────────────────────────────────────────

def significance_test(raw_df, arm_a, arm_b, metric='satisfaction'):
    """Two-sided t-test between two arms."""
    a = raw_df[raw_df['arm'] == arm_a][metric].values
    b = raw_df[raw_df['arm'] == arm_b][metric].values
    if len(a) < 2 or len(b) < 2:
        return None, None
    t_stat, p_val = stats.ttest_ind(a, b)
    return float(np.mean(a) - np.mean(b)), float(p_val)


def build_summary(raw_df, dataset_type):
    """Aggregate raw rows into summary statistics per arm.

    When the Pareto sweep is active (comfort_target column present), the
    hypervolume uses ALL (kwh, dh) points across comfort targets — this is
    the true Pareto hypervolume for each arm's frontier.
    """
    agg_dict = {
        'sat_mean': ('satisfaction', 'mean'),
        'sat_std': ('satisfaction', 'std'),
        'eng_mean': ('energy', 'mean'),
        'eng_std': ('energy', 'std'),
        'mo_mean': ('mo_score', 'mean'),
        'mo_std': ('mo_score', 'std'),
        'cv_mean': ('cv_mean', 'mean'),
        'cv_std': ('cv_mean', 'std'),
        'cvr_mean': ('cvr', 'mean'),
        'cvr_std': ('cvr', 'std'),
        'intervention_count_mean': ('intervention_count', 'mean'),
        'n': ('episode_seed', 'count'),
    }
    if 'kwh' in raw_df.columns:
        agg_dict['kwh_mean'] = ('kwh', 'mean')
        agg_dict['kwh_std'] = ('kwh', 'std')
    if 'dh' in raw_df.columns:
        agg_dict['dh_mean'] = ('dh', 'mean')
        agg_dict['dh_std'] = ('dh', 'std')

    summary = raw_df.groupby('arm').agg(**agg_dict).reset_index()

    # Hypervolume — per-arm rectangle to nadir (paper A.3.5)
    hv_info = compute_hypervolume_from_rows(raw_df.to_dict('records'))
    summary['hypervolume'] = summary['arm'].map(hv_info['per_arm'])

    summary['simulator'] = dataset_type
    summary = summary.sort_values('mo_mean', ascending=False)
    return summary


# ── Visualization ────────────────────────────────────────────────────────────

def _arm_colors(arms):
    """Assign colors to arms, falling back to a palette for unknowns."""
    fallback = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA',
                '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
    colors = {}
    fi = 0
    for a in arms:
        if a in ARM_COLORS:
            colors[a] = ARM_COLORS[a]
        else:
            colors[a] = fallback[fi % len(fallback)]
            fi += 1
    return colors


def generate_b1_plots(raw_df, summary_df, sim_label,
                      output_dir='neurips_results'):
    """Generate 4 publication-quality PNGs for one simulator."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
    })
    arms = list(summary_df['arm'])
    colors = _arm_colors(arms)

    # 1. Staircase MO-Score
    fig, ax = plt.subplots(figsize=(max(7, len(arms) * 1.2), 5))
    mo_means = summary_df['mo_mean'].values
    mo_stds = summary_df['mo_std'].values
    bars = ax.bar(arms, mo_means, yerr=mo_stds, capsize=5,
                  color=[colors[a] for a in arms], alpha=0.9,
                  edgecolor='white', linewidth=0.8,
                  error_kw={'linewidth': 1.2, 'capthick': 1.2})
    ax.set_ylabel('Multi-Objective Score', fontsize=12)
    ax.set_title(f'Multi-Objective Performance -- {sim_label}',
                 fontsize=13, fontweight='bold')
    ax.set_xticklabels(arms, rotation=30, ha='right', fontsize=9)
    for bar, val, std_val in zip(bars, mo_means, mo_stds):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + std_val + 0.005,
                f'{val:.3f}', ha='center', va='bottom',
                fontweight='bold', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f'b1_prelim_{sim_label}_staircase.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2. Pareto Frontier: kWh vs Degree-Hours  (paper A.3.4 / A.3.5)
    #    Cross-policy frontier: each arm is ONE mean point.
    #    Non-dominated set across ALL arms defines the Pareto front.
    #    HV = area dominated by the frontier relative to r=(1.2,1.0).
    fig, ax = plt.subplots(figsize=(9, 7))

    hv_info = compute_hypervolume_from_rows(raw_df.to_dict('records'))
    arm_means = hv_info['arm_means']      # arm → (kwh, dh)
    frontier  = hv_info['frontier']       # [(kwh, dh), ...] non-dominated

    all_kwh = raw_df['kwh'].values
    all_dh = raw_df['dh'].values

    # Axis limits with padding
    kwh_pad = max((all_kwh.max() - all_kwh.min()) * 0.08, 1.0)
    dh_pad = max((all_dh.max() - all_dh.min()) * 0.08, 0.5)
    kwh_lo = max(0, all_kwh.min() - kwh_pad)
    kwh_hi = all_kwh.max() + kwh_pad
    dh_lo = max(0, all_dh.min() - dh_pad)
    dh_hi = all_dh.max() + dh_pad

    kwh_mid = (kwh_lo + kwh_hi) / 2
    dh_mid = (dh_lo + dh_hi) / 2

    # Quadrant shading — optimal = bottom-left, worst = top-right
    xfrac = (kwh_mid - kwh_lo) / (kwh_hi - kwh_lo)
    ax.axhspan(dh_lo, dh_mid, xmin=0.0, xmax=xfrac,
               alpha=0.12, color='#27AE60', zorder=0)      # optimal (green)
    ax.axhspan(dh_mid, dh_hi, xmin=xfrac, xmax=1.0,
               alpha=0.15, color='#F4D03F', zorder=0)      # worst (yellow)
    ax.axhspan(dh_lo, dh_mid, xmin=xfrac, xmax=1.0,
               alpha=0.08, color='#F39C12', zorder=0)      # mixed
    ax.axhspan(dh_mid, dh_hi, xmin=0.0, xmax=xfrac,
               alpha=0.08, color='#F39C12', zorder=0)      # mixed

    ax.axvline(kwh_mid, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(dh_mid, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.text(kwh_lo + kwh_pad * 0.3, dh_lo + dh_pad * 0.3,
            'OPTIMAL\nLow Energy\nLow Discomfort',
            ha='left', va='bottom', fontsize=9, fontweight='bold',
            color='#1E8449', alpha=0.7)
    ax.text(kwh_hi - kwh_pad * 0.3, dh_hi - dh_pad * 0.3,
            'WORST\nHigh Energy\nHigh Discomfort',
            ha='right', va='top', fontsize=9, fontweight='bold',
            color='#B7950B', alpha=0.7)

    # Marker styles per arm category
    markers = {
        'ASHRAE-PID': 's', 'GRID': 'o', 'GRID_ObsOnly': 'o',
        'GRID_RegimeAware': 'o', 'DDPC_Behavioral': 'D',
        'DDPC_Subspace': 'D', 'DDPC_Neural': 'D', 'DDPC_PETS': 'D',
    }

    # ── Cross-policy Pareto frontier line (paper A.3.4) ──
    if len(frontier) >= 2:
        f_kwh, f_dh = zip(*frontier)
        ax.plot(f_kwh, f_dh, color='black', linewidth=2.5,
                linestyle='-', zorder=8, alpha=0.7, label='Pareto Front')
        # Shade the dominated region above the frontier
        ax.fill_between(f_kwh, f_dh, dh_hi,
                        alpha=0.04, color='red', zorder=0)

    # ── Per-arm: faint episode scatter + bold mean point ──
    for arm in arms:
        arm_rows = raw_df[raw_df['arm'] == arm]
        mk = markers.get(arm, 'o')
        clr = colors[arm]

        # Faint scatter of all raw episode points
        ax.scatter(arm_rows['kwh'], arm_rows['dh'],
                   alpha=0.20, s=25, color=clr,
                   edgecolors='none', marker=mk, zorder=4)

        # Bold arm-mean point (paper: one point per policy π)
        if arm in arm_means:
            mk_kwh, mk_dh = arm_means[arm]
            # Thicker edge for arms on the Pareto frontier
            on_front = any(abs(mk_kwh - fk) < 0.5 and abs(mk_dh - fd) < 0.5
                           for fk, fd in frontier)
            ew = 2.0 if on_front else 0.8
            sz = 130 if on_front else 80
            ax.scatter(mk_kwh, mk_dh, label=arm, s=sz, color=clr,
                       marker=mk, edgecolors='black', linewidth=ew,
                       zorder=9)

    # ── HV annotations ──
    combined_hv = hv_info['combined']
    per_arm_hv = hv_info['per_arm']
    ax.text(kwh_lo + kwh_pad * 0.3, dh_hi - dh_pad * 0.3,
            f'Combined HV = {combined_hv:.3f}',
            fontsize=10, fontweight='bold', color='black', ha='left')
    y_ann = dh_hi - dh_pad * 0.3 - (dh_hi - dh_lo) * 0.055
    for arm in arms:
        hv_val = per_arm_hv.get(arm, 0.0)
        ax.text(kwh_lo + kwh_pad * 0.3, y_ann,
                f'{arm}: {hv_val:.3f}', fontsize=7.5,
                color=colors[arm], fontweight='bold', ha='left')
        y_ann -= (dh_hi - dh_lo) * 0.042

    ax.set_xlabel('Energy Consumption (kWh)', fontsize=12)
    ax.set_ylabel('Comfort Violation (Degree-Hours)', fontsize=12)
    ax.set_title(f'Pareto Frontier: Energy vs Comfort -- {sim_label}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9,
              framealpha=0.9)
    ax.set_xlim(kwh_lo, kwh_hi)
    ax.set_ylim(dh_lo, dh_hi)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f'b1_prelim_{sim_label}_pareto.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 3. Satisfaction Bars
    fig, ax = plt.subplots(figsize=(max(7, len(arms) * 1.2), 5))
    sat_means = summary_df['sat_mean'].values
    sat_stds = summary_df['sat_std'].values
    bars = ax.bar(arms, sat_means, yerr=sat_stds, capsize=5,
                  color=[colors[a] for a in arms], alpha=0.9,
                  edgecolor='white', linewidth=0.8,
                  error_kw={'linewidth': 1.2, 'capthick': 1.2})
    ax.set_ylabel('Satisfaction (%)', fontsize=12)
    ax.set_title(f'Occupant Comfort -- {sim_label}',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_xticklabels(arms, rotation=30, ha='right', fontsize=9)
    for bar, val in zip(bars, sat_means):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f'b1_prelim_{sim_label}_satisfaction.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 4. Energy Bars
    fig, ax = plt.subplots(figsize=(max(7, len(arms) * 1.2), 5))
    eng_means = summary_df['eng_mean'].values
    eng_stds = summary_df['eng_std'].values
    bars = ax.bar(arms, eng_means, yerr=eng_stds, capsize=5,
                  color=[colors[a] for a in arms], alpha=0.9,
                  edgecolor='white', linewidth=0.8,
                  error_kw={'linewidth': 1.2, 'capthick': 1.2})
    ax.set_ylabel('Energy (%)', fontsize=12)
    ax.set_title(f'Energy Consumption -- {sim_label}',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_xticklabels(arms, rotation=30, ha='right', fontsize=9)
    for bar, val in zip(bars, eng_means):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f'b1_prelim_{sim_label}_energy.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


# ── Regime-specific visualization (open_window) ─────────────────────────────

def _pareto_panel(ax, arm_df, arms, colors, markers, title, hv_map=None):
    """Draw one Pareto panel on the given axes (reusable helper)."""
    all_kwh = arm_df['kwh'].values
    all_dh = arm_df['dh'].values
    if len(all_kwh) == 0:
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, alpha=0.5)
        return

    kwh_pad = max((all_kwh.max() - all_kwh.min()) * 0.08, 1.0)
    dh_pad = max((all_dh.max() - all_dh.min()) * 0.08, 0.5)
    kwh_lo = max(0, all_kwh.min() - kwh_pad)
    kwh_hi = all_kwh.max() + kwh_pad
    dh_lo = max(0, all_dh.min() - dh_pad)
    dh_hi = all_dh.max() + dh_pad
    kwh_mid = (kwh_lo + kwh_hi) / 2
    dh_mid = (dh_lo + dh_hi) / 2

    # Quadrant shading
    xfrac = (kwh_mid - kwh_lo) / (kwh_hi - kwh_lo)
    ax.axhspan(dh_lo, dh_mid, xmin=0.0, xmax=xfrac,
               alpha=0.12, color='#27AE60', zorder=0)
    ax.axhspan(dh_mid, dh_hi, xmin=xfrac, xmax=1.0,
               alpha=0.15, color='#F4D03F', zorder=0)
    ax.axhspan(dh_lo, dh_mid, xmin=xfrac, xmax=1.0,
               alpha=0.08, color='#F39C12', zorder=0)
    ax.axhspan(dh_mid, dh_hi, xmin=0.0, xmax=xfrac,
               alpha=0.08, color='#F39C12', zorder=0)
    ax.axvline(kwh_mid, color='grey', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.axhline(dh_mid, color='grey', linewidth=0.8, linestyle='--', alpha=0.4)

    ax.text(kwh_lo + kwh_pad * 0.2, dh_lo + dh_pad * 0.2,
            'OPTIMAL', fontsize=8, color='#1E8449', alpha=0.6,
            fontweight='bold')
    ax.text(kwh_hi - kwh_pad * 0.2, dh_hi - dh_pad * 0.2,
            'WORST', fontsize=8, color='#B7950B', alpha=0.6,
            fontweight='bold', ha='right', va='top')

    has_sweep = ('comfort_target' in arm_df.columns
                 and arm_df['comfort_target'].nunique() > 1)

    for arm in arms:
        rows = arm_df[arm_df['arm'] == arm]
        if rows.empty:
            continue
        mk = markers.get(arm, 'o')
        clr = colors[arm]

        ax.scatter(rows['kwh'], rows['dh'], alpha=0.2, s=25,
                   color=clr, edgecolors='none', marker=mk, zorder=4)

        if has_sweep:
            means = (rows.groupby('comfort_target')
                     .agg(kwh_mean=('kwh', 'mean'),
                          dh_mean=('dh', 'mean'))
                     .reset_index())
            pts = list(zip(means['kwh_mean'], means['dh_mean']))
            frontier = pareto_filter_2d(pts)
            if len(frontier) >= 2:
                f_kwh, f_dh = zip(*frontier)
                ax.plot(f_kwh, f_dh, color=clr, linewidth=2.0,
                        zorder=6, alpha=0.9)
            ax.scatter(means['kwh_mean'], means['dh_mean'],
                       label=arm, s=80, color=clr, marker=mk,
                       edgecolors='black', linewidth=0.8, zorder=7)
        else:
            ax.scatter(rows['kwh'], rows['dh'],
                       label=arm, alpha=0.8, s=60, color=clr,
                       edgecolors='black', linewidth=0.5, marker=mk,
                       zorder=5)

    # HV annotations
    if hv_map:
        y_ann = dh_hi - dh_pad * 0.3
        for arm in arms:
            hv_val = hv_map.get(arm, 0.0)
            if hv_val > 0:
                ax.text(kwh_lo + kwh_pad * 0.2, y_ann,
                        f'{arm}: HV={hv_val:.1f}', fontsize=7,
                        color=colors[arm], fontweight='bold')
                y_ann -= (dh_hi - dh_lo) * 0.05

    ax.set_xlim(kwh_lo, kwh_hi)
    ax.set_ylim(dh_lo, dh_hi)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.2, linestyle='--')


def generate_regime_pareto_plot(raw_df, regime_df, sim_label,
                                output_dir='neurips_results'):
    """3-panel Pareto: H0 (window closed) | H1 (window open) | Combined."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    arms = list(raw_df.groupby('arm')['mo_score'].mean()
                .sort_values(ascending=False).index)
    colors = _arm_colors(arms)
    markers = {
        'ASHRAE-PID': 's', 'GRID': 'o', 'GRID_ObsOnly': 'o',
        'GRID_RegimeAware': 'o', 'DDPC_Behavioral': 'D',
        'DDPC_Subspace': 'D', 'DDPC_Neural': 'D', 'DDPC_PETS': 'D',
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f'Pareto Frontiers by Regime -- {sim_label}',
                 fontsize=14, fontweight='bold', y=1.02)

    # H0 and H1 sub-DataFrames from regime_df
    h0_df = regime_df[regime_df['regime'] == 0].copy()
    h1_df = regime_df[regime_df['regime'] == 1].copy()

    # Compute HV per regime — extract per_arm dict for panel annotation
    hv_h0 = (compute_hypervolume_from_rows(h0_df.to_dict('records'))['per_arm']
             if not h0_df.empty else {})
    hv_h1 = (compute_hypervolume_from_rows(h1_df.to_dict('records'))['per_arm']
             if not h1_df.empty else {})
    hv_all = compute_hypervolume_from_rows(raw_df.to_dict('records'))['per_arm']

    panels = [
        (axes[0], h0_df, 'H0: Window Closed', hv_h0),
        (axes[1], h1_df, 'H1: Window Open', hv_h1),
        (axes[2], raw_df, 'Combined', hv_all),
    ]
    for ax, df, title, hv_map in panels:
        _pareto_panel(ax, df, arms, colors, markers, title, hv_map)
        ax.set_xlabel('Energy (kWh)', fontsize=10)

    axes[0].set_ylabel('Comfort Violation (Degree-Hours)', fontsize=10)
    axes[0].legend(loc='upper left', fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f'b1_prelim_{sim_label}_regime_pareto.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_regime_episode_plot(raw_df, regime_df, step_df, sim_label,
                                 output_dir='neurips_results'):
    """3-row plot: regime signal | satisfaction timeline | MO by regime bars."""
    from scipy.ndimage import uniform_filter1d

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    arms = list(raw_df.groupby('arm')['mo_score'].mean()
                .sort_values(ascending=False).index)
    colors = _arm_colors(arms)

    fig, axes_arr = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(f'Regime-Aware Episode Performance -- {sim_label}',
                 fontsize=14, fontweight='bold')

    # Use the first seed's default comfort target for the timeline
    if not step_df.empty:
        first_seed = step_df['discovery_seed'].min()
        first_ep = step_df['episode_seed'].min()
        first_ct = step_df['comfort_target'].min()
        ep_steps = step_df[
            (step_df['discovery_seed'] == first_seed) &
            (step_df['episode_seed'] == first_ep) &
            (step_df['comfort_target'] == first_ct)
        ]
    else:
        ep_steps = pd.DataFrame()

    # ── Panel 1: Regime signal ────────────────────────────────────────────
    ax = axes_arr[0]
    if not ep_steps.empty:
        # Pick one arm's regime trace (all arms see same window state)
        one_arm = ep_steps[ep_steps['arm'] == ep_steps['arm'].iloc[0]]
        ts = one_arm['t'].values
        reg = one_arm['regime'].values
        ax.fill_between(ts, 0, reg, alpha=0.3, color='#3498db', step='mid')
        ax.step(ts, reg, color='#2c3e50', linewidth=1.5, where='mid')
        # Shade regime zones
        for i in range(len(ts)):
            if reg[i] == 0:
                ax.axvspan(ts[i] - 0.5, ts[i] + 0.5,
                           alpha=0.04, color='orange', zorder=0)
            else:
                ax.axvspan(ts[i] - 0.5, ts[i] + 0.5,
                           alpha=0.04, color='blue', zorder=0)
    ax.set_ylabel('Window\nState', fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Closed (H0)', 'Open (H1)'], fontsize=9)
    ax.set_title('Regime Signal (WindowOpen sensor)', fontsize=11)

    # ── Panel 2: Satisfaction over time ───────────────────────────────────
    ax = axes_arr[1]
    if not ep_steps.empty:
        ts = sorted(ep_steps['t'].unique())
        for arm in arms:
            arm_steps = ep_steps[ep_steps['arm'] == arm].sort_values('t')
            if arm_steps.empty:
                continue
            sat_vals = arm_steps['sat'].values
            if len(sat_vals) >= 3:
                smooth = uniform_filter1d(sat_vals.astype(float), size=3)
            else:
                smooth = sat_vals.astype(float)
            ax.plot(arm_steps['t'].values, smooth, color=colors[arm],
                    linewidth=2, label=arm, alpha=0.9)
            ax.fill_between(arm_steps['t'].values, smooth - 2, smooth + 2,
                            color=colors[arm], alpha=0.1)

        # Mark regime transitions
        if len(ts) > 1:
            one_arm = ep_steps[ep_steps['arm'] == ep_steps['arm'].iloc[0]]
            reg = one_arm.sort_values('t')['regime'].values
            transitions = np.diff(reg, prepend=reg[0])
            for i, tr in enumerate(transitions):
                if tr != 0:
                    ax.axvline(ts[i], color='red', linewidth=1,
                               linestyle='--', alpha=0.5)
                    ax.annotate('regime\nswitch', xy=(ts[i], 85),
                                fontsize=7, color='red', ha='center',
                                alpha=0.7)

    ax.set_ylabel('Satisfaction (%)', fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower left', fontsize=8, ncol=2, framealpha=0.9)
    ax.set_title('Satisfaction Trajectory', fontsize=11)
    ax.grid(alpha=0.2, linestyle='--')

    # ── Panel 3: MO-Score bar chart by regime ─────────────────────────────
    ax = axes_arr[2]
    regime_labels = ['H0\n(Closed)', 'H1\n(Open)', 'Combined']
    x = np.arange(len(regime_labels))
    bar_width = 0.8 / max(len(arms), 1)

    for i, arm in enumerate(arms):
        h0_mo = regime_df[(regime_df['arm'] == arm) &
                          (regime_df['regime'] == 0)]['mo_score']
        h1_mo = regime_df[(regime_df['arm'] == arm) &
                          (regime_df['regime'] == 1)]['mo_score']
        comb_mo = raw_df[raw_df['arm'] == arm]['mo_score']

        vals = [h0_mo.mean() if len(h0_mo) > 0 else 0,
                h1_mo.mean() if len(h1_mo) > 0 else 0,
                comb_mo.mean() if len(comb_mo) > 0 else 0]

        offset = (i - (len(arms) - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, bar_width,
                      color=colors[arm], label=arm, alpha=0.9,
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=6.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels, fontsize=10)
    ax.set_ylabel('MO-Score', fontsize=10)
    ax.set_title('Multi-Objective Score by Regime', fontsize=11)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9, ncol=2)
    ax.grid(axis='y', alpha=0.2, linestyle='--')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,
                             f'b1_prelim_{sim_label}_regime_episode.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


# ── Per-simulator runner ─────────────────────────────────────────────────────

def run_one_simulator(sim_label, cfg, n_seeds, n_episodes, include_pets,
                      comfort_targets=None):
    """Run B1 for one simulator across multiple discovery seeds.

    For each seed:
      1. Run full CWM pipeline (discovery + validation + monitoring)
      2. Extract discovered edges + method DAG union + intervention count
      3. Build NeurIPSExperiments with REAL pipeline
      4. Evaluate all policy arms at each comfort target (Pareto sweep)
      5. Collect expanded metrics

    Returns (raw_df, summary_df, b2_df, regime_df, step_df,
             elapsed_seconds, compute_log).
    """
    print()
    print("=" * 72)
    print("  B1 -- Policy Improvement Attribution (Full Pipeline)")
    ct_str = (f"  Comfort targets: {comfort_targets}"
              if comfort_targets and len(comfort_targets) > 1
              else "  (single comfort target)")
    print(f"  Simulator: {sim_label}  |  Seeds: {n_seeds}  "
          f"|  Episodes/arm: {n_episodes}")
    print(ct_str)
    print("=" * 72)
    print()
    print("  Question: Does causal structure help control?")
    print("            Does validation make it better?")
    if cfg['has_regime']:
        print("            Does the full loop (regime detection) add more?")
    print()

    t0 = time.time()
    all_rows = []
    all_regime_rows = []
    all_step_rows = []
    all_b2_rows = []
    compute_log = {
        'simulator': sim_label,
        'n_seeds': n_seeds,
        'n_episodes': n_episodes,
        'seeds': [],
    }

    for seed in range(n_seeds):
        print(f"\n  --- Seed {seed + 1}/{n_seeds} ---")

        # 1. Run full discovery pipeline
        logger.info(f"[{sim_label}] seed={seed}: Running discovery pipeline...")
        try:
            discovery = run_discovery_pipeline(cfg, seed)
        except Exception as e:
            logger.error(f"[{sim_label}] seed={seed}: Discovery failed: {e}",
                         exc_info=True)
            continue

        logger.info(
            f"[{sim_label}] seed={seed}: Discovery complete in "
            f"{discovery['discovery_time_s']:.1f}s -- "
            f"validated={len(discovery['validated_edges'])} edges, "
            f"obs_only={len(discovery['obs_only_edges'])} edges, "
            f"interventions={discovery['intervention_count']}"
        )

        # 2. Run policy episodes (B1)
        logger.info(f"[{sim_label}] seed={seed}: Running policy episodes...")
        t_pol = time.time()
        arm_timings = {}
        try:
            rows, arm_timings, regime_rows, step_rows = run_policy_episodes(
                discovery, cfg, n_episodes, seed, include_pets,
                comfort_targets=comfort_targets)
            all_rows.extend(rows)
            all_regime_rows.extend(regime_rows)
            all_step_rows.extend(step_rows)
        except Exception as e:
            logger.error(f"[{sim_label}] seed={seed}: Policy eval failed: {e}",
                         exc_info=True)
        policy_wall_s = time.time() - t_pol

        # 3. Counterfactual sanity checks (B2)
        logger.info(f"[{sim_label}] seed={seed}: Running B2 counterfactual checks...")
        t_b2 = time.time()
        try:
            b2_df = run_b2_counterfactual_checks(discovery, cfg, seed)
            if not b2_df.empty:
                all_b2_rows.append(b2_df)
        except Exception as e:
            logger.warning(f"[{sim_label}] seed={seed}: B2 checks failed: {e}")
        b2_wall_s = time.time() - t_b2

        # Collect per-seed compute info
        compute_log['seeds'].append({
            'seed': seed,
            'discovery': discovery.get('timings', {}),
            'policy_eval_s': round(policy_wall_s, 2),
            'arm_timings_s': arm_timings,
            'b2_checks_s': round(b2_wall_s, 2),
            'llm': discovery.get('llm_stats', {}),
            'n_validated_edges': len(discovery['validated_edges']),
            'n_obs_only_edges': len(discovery['obs_only_edges']),
            'n_interventions': discovery['intervention_count'],
        })

    elapsed = time.time() - t0

    # Combine B2 results
    b2_combined = (pd.concat(all_b2_rows, ignore_index=True)
                   if all_b2_rows else pd.DataFrame())
    if not b2_combined.empty:
        b2_combined['simulator'] = sim_label

    if not all_rows:
        logger.error(f"[{sim_label}] No results collected!")
        return (pd.DataFrame(), pd.DataFrame(), b2_combined,
                pd.DataFrame(), pd.DataFrame(), elapsed, compute_log)

    raw_df = pd.DataFrame(all_rows)
    raw_df['simulator'] = sim_label

    summary_df = build_summary(raw_df, cfg['dataset_type'])
    summary_df['simulator'] = sim_label

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  RESULTS -- {n_seeds} seeds x {n_episodes} episodes "
          f"({sim_label})  [{elapsed:.1f}s]")
    print(f"{'=' * 72}")
    print(f"{'Arm':25s} {'Sat%':>10s} {'Eng%':>10s} "
          f"{'MO':>8s} {'HV':>8s} {'CV':>7s} {'CVR':>6s} {'#Intv':>6s}")
    print(f"{'-' * 25} {'-' * 10} {'-' * 10} "
          f"{'-' * 8} {'-' * 8} {'-' * 7} {'-' * 6} {'-' * 6}")
    for _, r in summary_df.iterrows():
        print(f"{r['arm']:25s} "
              f"{r['sat_mean']:5.1f}+/-{r['sat_std']:4.1f} "
              f"{r['eng_mean']:5.1f}+/-{r['eng_std']:4.1f} "
              f"{r['mo_mean']:7.4f} "
              f"{r['hypervolume']:7.1f} "
              f"{r['cv_mean']:6.2f} "
              f"{r['cvr_mean']:5.3f} "
              f"{r['intervention_count_mean']:5.0f}")
    print(f"{'=' * 72}")

    # ── Key comparisons (the advisor's staircase) ───────────────────────────────
    print("\n  Key Comparisons (the advisor's B1 staircase):")
    print(f"  {'-' * 60}")

    baseline_arm = 'ASHRAE-PID'

    comparisons = [
        ('GRID', baseline_arm, 'Causal structure vs static baseline'),
        ('GRID', 'GRID_ObsOnly', 'Validation helps (prunes spurious)'),
    ]
    if 'GRID_RegimeAware' in raw_df['arm'].unique():
        comparisons.append(
            ('GRID_RegimeAware', 'GRID', 'Regime detection adds value'))

    for ddpc in ['DDPC_Neural', 'DDPC_Behavioral', 'DDPC_Subspace']:
        if ddpc in raw_df['arm'].unique():
            comparisons.append(('GRID', ddpc, f'GRID vs {ddpc}'))

    test_metric = ('energy' if cfg['dataset_type'] == 'ashrae'
                   else 'satisfaction')

    for arm_a, arm_b, label in comparisons:
        if (arm_a not in raw_df['arm'].unique() or
                arm_b not in raw_df['arm'].unique()):
            continue
        diff, p_val = significance_test(raw_df, arm_a, arm_b,
                                        metric=test_metric)
        if diff is None:
            continue
        sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01
               else '*' if p_val < 0.05 else 'ns')
        sign = '+' if diff > 0 else ''
        unit = '' if cfg['dataset_type'] == 'ashrae' else 'pp'
        print(f"  {label:42s} {sign}{diff:5.2f}{unit}  "
              f"p={p_val:.3f} {sig}")

    print(f"  {'-' * 60}")

    # ── B2: Counterfactual summary ────────────────────────────────────────
    if not b2_combined.empty:
        n_checks = len(b2_combined)
        n_matched = b2_combined['direction_match'].sum()
        n_with_data = b2_combined['direction_match'].notna().sum()
        match_rate = (n_matched / n_with_data * 100) if n_with_data > 0 else 0
        print(f"\n  B2 Counterfactual Sanity Checks:")
        print(f"  {'-' * 60}")
        print(f"  Edges checked: {n_checks}  |  "
              f"With intervention data: {n_with_data}  |  "
              f"Direction match: {n_matched}/{n_with_data} "
              f"({match_rate:.0f}%)")
        # Show per-edge details for the first seed
        first_seed = b2_combined[
            b2_combined['discovery_seed'] == b2_combined['discovery_seed'].min()
        ]
        for _, row in first_seed.iterrows():
            m = ('Y' if row['direction_match'] is True
                 else 'N' if row['direction_match'] is False else '?')
            print(f"    {row['edge']:30s}  SEM={row['sem_direction']}  "
                  f"Obs={row['observed_direction']}  Match={m}  "
                  f"(beta={row['sem_coefficient']:+.4f}, "
                  f"n_intv={row['n_interventions']})")
        print(f"  {'-' * 60}")

    # ── Regime-specific results ─────────────────────────────────────────
    regime_combined = (pd.concat([pd.DataFrame(all_regime_rows)],
                                 ignore_index=True)
                       if all_regime_rows else pd.DataFrame())
    step_combined = (pd.concat([pd.DataFrame(all_step_rows)],
                                ignore_index=True)
                     if all_step_rows else pd.DataFrame())

    if not regime_combined.empty:
        regime_combined['simulator'] = sim_label
        print(f"\n  Regime-Specific MO-Scores ({sim_label}):")
        print(f"  {'-' * 60}")
        print(f"  {'Arm':25s} {'MO(H0)':>8s} {'MO(H1)':>8s} "
              f"{'MO(All)':>8s} {'Delta':>8s}")
        print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
        for arm in summary_df['arm']:
            h0 = regime_combined[(regime_combined['arm'] == arm) &
                                 (regime_combined['regime'] == 0)]
            h1 = regime_combined[(regime_combined['arm'] == arm) &
                                 (regime_combined['regime'] == 1)]
            mo_h0 = h0['mo_score'].mean() if not h0.empty else float('nan')
            mo_h1 = h1['mo_score'].mean() if not h1.empty else float('nan')
            mo_all = raw_df[raw_df['arm'] == arm]['mo_score'].mean()
            delta = mo_h1 - mo_h0 if not (
                np.isnan(mo_h0) or np.isnan(mo_h1)) else float('nan')
            print(f"  {arm:25s} {mo_h0:8.4f} {mo_h1:8.4f} "
                  f"{mo_all:8.4f} {delta:+8.4f}")
        print(f"  {'-' * 60}")

    if not step_combined.empty:
        step_combined['simulator'] = sim_label

    # ── M5: Compute resource summary ─────────────────────────────────────
    compute_log['total_wall_s'] = round(elapsed, 2)

    # Aggregate LLM stats across seeds
    total_llm_calls = sum(s.get('llm', {}).get('api_calls', 0)
                          for s in compute_log['seeds'])
    total_llm_tokens = sum(s.get('llm', {}).get('total_tokens', 0)
                           for s in compute_log['seeds'])
    print(f"\n  M5 Compute Resources ({sim_label}):")
    print(f"  {'-' * 60}")
    print(f"    Total wall-clock: {elapsed:.1f}s "
          f"({elapsed / 60:.1f} min)")
    for sd in compute_log['seeds']:
        disc_t = sd.get('discovery', {}).get('total_s', 0)
        mon_t = sd.get('discovery', {}).get('monitoring_s', 0)
        pol_t = sd.get('policy_eval_s', 0)
        print(f"    Seed {sd['seed']}: discovery={disc_t:.1f}s "
              f"monitoring={mon_t:.1f}s "
              f"policy_eval={pol_t:.1f}s")
        for arm, t in sd.get('arm_timings_s', {}).items():
            print(f"      {arm}: {t:.1f}s")
    print(f"    LLM: {total_llm_calls} API calls, "
          f"{total_llm_tokens} tokens (gpt-3.5-turbo)")
    print(f"  {'-' * 60}")
    print()

    return (raw_df, summary_df, b2_combined, regime_combined,
            step_combined, elapsed, compute_log)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    sim_names = [k for k in SIM_CONFIGS.keys() if k != 'ashrae']

    parser = argparse.ArgumentParser(
        description='Experiment Suite B: B1 Policy Attribution + '
                    'B2 Counterfactual Checks (Full Pipeline)')
    parser.add_argument('--sim', type=str, default='open_window',
                        choices=sim_names,
                        help='Which JS simulator to use (default: open_window)')
    parser.add_argument('--all-sims', action='store_true',
                        help='Run ALL simulators in sequence')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['ashrae'],
                        help='Use ASHRAE dataset')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of discovery seeds (default 5)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Policy episode seeds per arm per discovery '
                             'seed (default 5)')
    parser.add_argument('--include-pets', action='store_true',
                        help='Include DDPC_PETS (very slow)')
    parser.add_argument('--comfort-targets', type=str, default=None,
                        help='Comma-separated DH budget targets for '
                             'epsilon-constraint Pareto sweep '
                             f'(default: {",".join(str(c) for c in DEFAULT_COMFORT_TARGETS)})')
    args = parser.parse_args()

    # Parse comfort targets
    if args.comfort_targets:
        comfort_targets = [float(x) for x in args.comfort_targets.split(',')]
    else:
        comfort_targets = DEFAULT_COMFORT_TARGETS

    os.makedirs('neurips_results', exist_ok=True)

    # Build list of sims to run
    if args.dataset == 'ashrae':
        run_labels = ['ashrae']
    elif args.all_sims:
        run_labels = list(SIM_CONFIGS.keys())
    else:
        run_labels = [args.sim]

    all_raw = []
    all_summary = []
    all_b2 = []
    all_compute = []
    grand_t0 = time.time()

    for sim_label in run_labels:
        cfg = SIM_CONFIGS[sim_label]
        (raw_df, summary_df, b2_df, regime_df, step_df,
         elapsed, compute_log) = run_one_simulator(
            sim_label, cfg, args.seeds, args.episodes, args.include_pets,
            comfort_targets=comfort_targets)

        all_compute.append(compute_log)

        if raw_df.empty:
            continue

        all_raw.append(raw_df)
        all_summary.append(summary_df)
        if not b2_df.empty:
            all_b2.append(b2_df)

        # Save per-simulator outputs
        tag = sim_label
        raw_df.to_csv(f'neurips_results/b1_{tag}_raw.csv',
                       index=False)
        summary_df.to_csv(f'neurips_results/b1_{tag}_summary.csv',
                           index=False)
        if not b2_df.empty:
            b2_df.to_csv(f'neurips_results/b2_{tag}_counterfactual.csv',
                          index=False)

        # Save regime-specific outputs (open_window)
        if not regime_df.empty:
            regime_df.to_csv(f'neurips_results/b1_{tag}_regime.csv',
                              index=False)
        if not step_df.empty:
            step_df.to_csv(f'neurips_results/b1_{tag}_steps.csv',
                            index=False)

        # Generate PNGs
        generate_b1_plots(raw_df, summary_df, sim_label)

        # Regime-specific plots (open_window)
        if not regime_df.empty:
            generate_regime_pareto_plot(raw_df, regime_df, sim_label)
            generate_regime_episode_plot(raw_df, regime_df, step_df,
                                         sim_label)

        print(f"  Saved: neurips_results/b1_{tag}_*.csv + "
              f"b2_{tag}_counterfactual.csv + PNGs")
        print()

    # Combined results
    if len(all_raw) > 1:
        combined_raw = pd.concat(all_raw, ignore_index=True)
        combined_summary = pd.concat(all_summary, ignore_index=True)
        combined_raw.to_csv('neurips_results/b1_all_raw.csv',
                             index=False)
        combined_summary.to_csv('neurips_results/b1_all_summary.csv',
                                 index=False)
        print(f"  Combined B1: neurips_results/b1_all_*.csv")

    if len(all_b2) > 1:
        combined_b2 = pd.concat(all_b2, ignore_index=True)
        combined_b2.to_csv('neurips_results/b2_all_counterfactual.csv',
                            index=False)
        print(f"  Combined B2: neurips_results/b2_all_counterfactual.csv")

    # ── M5: Save compute_resources.json ──────────────────────────────────
    grand_wall = time.time() - grand_t0
    m5_report = {
        'system': _get_system_info(),
        'experiment': {
            'script': 'run_exp_suite_B_prelim.py',
            'args': vars(args),
            'total_wall_clock_s': round(grand_wall, 2),
            'total_wall_clock_h': round(grand_wall / 3600, 2),
        },
        'per_simulator': all_compute,
    }
    m5_path = 'neurips_results/compute_resources.json'
    with open(m5_path, 'w') as f:
        json.dump(m5_report, f, indent=2, default=str)

    print(f"\n{'=' * 72}")
    print(f"  M5 COMPUTE SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Machine: {m5_report['system']['platform']}")
    print(f"  CPU: {m5_report['system']['processor']} "
          f"({m5_report['system']['cpu_count_logical']} logical cores)")
    print(f"  RAM: {m5_report['system']['ram_total_gb']} GB")
    print(f"  GPU: {m5_report['system']['gpu']}")
    print(f"  Total wall-clock: {grand_wall:.1f}s ({grand_wall / 3600:.2f}h)")
    total_llm_calls = sum(
        s.get('llm', {}).get('api_calls', 0)
        for c in all_compute for s in c.get('seeds', []))
    total_llm_tokens = sum(
        s.get('llm', {}).get('total_tokens', 0)
        for c in all_compute for s in c.get('seeds', []))
    print(f"  LLM: gpt-3.5-turbo, temp=1, top_p=0.9")
    print(f"  LLM calls: {total_llm_calls}  |  "
          f"tokens: {total_llm_tokens}")
    print(f"  Saved: {m5_path}")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
