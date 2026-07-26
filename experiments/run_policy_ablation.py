#!/usr/bin/env python3
"""
Policy Engine Ablation Study
=============================
Tests how different policy engine configurations affect GRID performance
vs all DDPC baselines.

Studies:
  A. Gradient iterations  — steps per timestep: 1, 5, 10, 20
  B. Lambda (step size)   — regularisation: 0.1, 0.5, 1.0, 2.0
  C. Regime robustness    — 50-step episodes with mid-episode regime changes
     + GRID_ObsOnly + GRID_RegimeAware arms
  D. Sample efficiency    — DDPC trained on 500/1K/2K/5K/10K rows
  E. Pareto frontier      — sweep trade-off params, compute hypervolume

Usage:
  python run_policy_ablation.py --discover              # discovery + all studies
  python run_policy_ablation.py --discover --max-iter 30
  python run_policy_ablation.py --edges-file path.json  # load saved edges
  python run_policy_ablation.py                         # use FIXED_EDGES fallback
  python run_policy_ablation.py --study a,b             # run specific studies
  python run_policy_ablation.py --episodes 10           # more episodes
"""

import argparse
import os
import sys
import json
import time
import logging
import subprocess
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Suppress macOS Accelerate BLAS warnings
for _m in ('divide by zero encountered in matmul',
           'overflow encountered in matmul',
           'invalid value encountered in matmul'):
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=f'.*{_m}.*')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ablation.log", mode='w'),
        logging.StreamHandler()
    ]
)
for _quiet in ['matplotlib', 'PIL', 'urllib3']:
    logging.getLogger(_quiet).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.policy_engine import CausalPolicyEngine
from src.ddpc_baseline import create_ddpc_suite

# ── Fallback edge set from CWM pipeline run 7 ────────────────────────────────
# Used only when --discover and --edges-file are both absent.

FALLBACK_EDGES = {
    ('Temperature', 'AirQuality'),
    ('Temperature', 'EnergyConsumption'),
    ('Temperature', 'Satisfaction'),
    ('AirQuality', 'Satisfaction'),
    ('Humidity', 'AirQuality'),
}

FALLBACK_H0 = set(FALLBACK_EDGES)

FALLBACK_H1 = FALLBACK_EDGES | {
    ('OutdoorTemperature', 'WindowOpen'),
    ('WindowOpen', 'Temperature'),
}

# ── Runtime edge sets (populated by main) ─────────────────────────────────────
EDGES = None       # validated edges
OBS_EDGES = None   # observational-only (method DAG union)
REGIME_H0 = None   # H0 edges (window closed)
REGIME_H1 = None   # H1 edges (window open)

# ── Simulator interface ───────────────────────────────────────────────────────

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_PATH = os.path.join(_BASE_DIR, 'open_window.js')
DATA_PATH = os.path.join(_BASE_DIR, 'data_regen', 'challenge1_data_10k_processed.csv')
OUTPUT_DIR = os.path.join('neurips_results', 'ablation')

ACTION_VARS = ['Temperature', 'Humidity', 'AirQuality']
JS_RANGES = {
    'Temperature': (18.0, 30.0),
    'Humidity':    (30.0, 70.0),
    'AirQuality':  (200.0, 1000.0),
}
STEP_DURATION_MS = 30_000


def _norm_to_phys(var, val):
    lo, hi = JS_RANGES[var]
    return float(np.clip(val, 0, 1)) * (hi - lo) + lo


def _phys_to_norm(var, val):
    lo, hi = JS_RANGES[var]
    return float(np.clip((val - lo) / (hi - lo), 0, 1))


def _dual_sat(state):
    if 'OverallSatisfaction' in state and 'Satisfaction' not in state:
        state['Satisfaction'] = state['OverallSatisfaction']
    elif 'Satisfaction' in state and 'OverallSatisfaction' not in state:
        state['OverallSatisfaction'] = state['Satisfaction']
    return state


def step_simulator(state, action, timestep):
    """Run one timestep via open_window.js."""
    state_payload = json.dumps({
        'temperature': _norm_to_phys('Temperature', state.get('Temperature', 0.5)),
        'humidity': _norm_to_phys('Humidity', state.get('Humidity', 0.5)),
        'airQuality': _norm_to_phys('AirQuality', state.get('AirQuality', 0.5)),
    })
    intervention_phys = json.dumps({
        'temperature': _norm_to_phys('Temperature', action.get('Temperature', 0.5)),
        'humidity': _norm_to_phys('Humidity', action.get('Humidity', 0.5)),
        'airQuality': _norm_to_phys('AirQuality', action.get('AirQuality', 0.5)),
    })
    elapsed_ms = timestep * STEP_DURATION_MS
    cmd = ['node', SIM_PATH, '--single-step',
           '--state', state_payload,
           '--intervention', intervention_phys,
           '--elapsed-ms', str(elapsed_ms)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    for line in result.stdout.splitlines():
        if line.startswith('RESULT:'):
            sim = json.loads(line[len('RESULT:'):])
            out = {
                'Temperature': _phys_to_norm('Temperature', sim.get('temperature', 22)),
                'Humidity': _phys_to_norm('Humidity', sim.get('humidity', 50)),
                'AirQuality': _phys_to_norm('AirQuality', sim.get('airQuality', 300)),
                'OverallSatisfaction': sim.get('overallSatisfaction', 50.0),
                'EnergyConsumption': sim.get('energyConsumption', 50.0),
            }
            if 'windowOpen' in sim:
                out['WindowOpen'] = sim['windowOpen']
            return _dual_sat(out)
    raise RuntimeError(f"Simulator failed: {result.stderr[:200]}")


def get_initial_state(seed):
    np.random.seed(seed)
    return _dual_sat({
        'Temperature': 0.5 + np.random.normal(0, 0.05),
        'Humidity': 0.5 + np.random.normal(0, 0.05),
        'AirQuality': 0.7 + np.random.normal(0, 0.05),
        'EnergyConsumption': 30.0,
        'OverallSatisfaction': 60.0,
    })


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(policy_fn, seed, timesteps=12):
    """Run one episode, return per-timestep records."""
    np.random.seed(seed)
    state = get_initial_state(seed)
    records = []
    for t in range(timesteps):
        action = policy_fn(state, t)
        next_state = step_simulator(state, action, t)
        records.append({
            'timestep': t,
            'satisfaction': next_state.get('OverallSatisfaction', 50.0),
            'energy': next_state.get('EnergyConsumption', 50.0),
            'window_open': next_state.get('WindowOpen', 0),
            'temperature': next_state.get('Temperature', 0.5),
        })
        state = next_state
    return records


def summarize_episode(records):
    """Aggregate episode records -> (sat_avg, eng_avg)."""
    sat = np.mean([r['satisfaction'] for r in records])
    eng = np.mean([r['energy'] for r in records])
    return sat, eng


def mo_score(sat, eng):
    """Multi-objective score: 0.6*(sat/100) + 0.4*(1-eng/100)."""
    return 0.6 * (sat / 100.0) + 0.4 * (1.0 - eng / 100.0)


# ── Discovery phase ──────────────────────────────────────────────────────────

def run_discovery(max_iterations=25):
    """Run full CWM pipeline and extract all edge sets.

    Returns dict with validated_edges, obs_only_edges, h0_edges, h1_edges,
    plus metadata (intervention_count, discovery_time_s, etc.).
    """
    from src.pipeline_cwm import create_cwm_pipeline

    obs_df = pd.read_csv(DATA_PATH)
    logger.info(f"Discovery: {len(obs_df)} rows, max_iterations={max_iterations}")

    api_key = os.environ.get('OPENAI_API_KEY',
        'YOUR_OPENAI_API_KEY')

    t0 = time.time()
    pipeline = create_cwm_pipeline(
        csv_data=obs_df,
        api_key=api_key,
        smart_room_path=SIM_PATH,
        dataset_type='open_window',
        max_iterations=max_iterations,
        alpha=0.5,
        beta=0.5,
        effect_threshold=0.1,
    )

    # Phase 1+2: Discovery + Validation
    final_dag, final_metrics = pipeline.run_with_cwm()

    # Phase 3: Monitoring (regime detection, builds H0/H1 graphs)
    try:
        pipeline.monitor_regime_changes(duration=300, sample_interval=5)
    except Exception as e:
        logger.warning(f"Monitoring failed (continuing): {e}")

    discovery_time = time.time() - t0

    # Extract validated edges
    validated_edges = set()
    for e in pipeline.pipeline.validated_edges:
        validated_edges.add(tuple(e) if isinstance(e, list) else e)

    # Extract obs-only edges (method DAG union)
    obs_only_edges = set()
    try:
        method_dags = pipeline.pipeline.get_method_dags()
        for G in method_dags.values():
            obs_only_edges.update(G.edges())
    except Exception as e:
        logger.warning(f"Could not extract method DAGs: {e}")

    # Extract H0/H1 edges
    h0_edges = None
    h1_edges = None
    if hasattr(pipeline, 'H0_graph') and pipeline.H0_graph is not None:
        h0_edges = set(pipeline.H0_graph.edges())
    if hasattr(pipeline, 'H1_graph') and pipeline.H1_graph is not None:
        h1_edges = set(pipeline.H1_graph.edges())

    # Count interventions
    intervention_count = sum(
        len(r) for r in pipeline.pipeline.tester.intervention_results.values()
    )

    logger.info(f"Discovery complete in {discovery_time:.1f}s: "
                f"validated={len(validated_edges)}, obs_only={len(obs_only_edges)}, "
                f"h0={len(h0_edges) if h0_edges else 0}, "
                f"h1={len(h1_edges) if h1_edges else 0}, "
                f"interventions={intervention_count}")

    # Get combined dataset (obs + upweighted intervention rows) from pipeline
    combined_dataset = None
    if hasattr(pipeline, 'pipeline') and hasattr(pipeline.pipeline, 'data'):
        combined_dataset = pipeline.pipeline.data.copy()
        logger.info(f"Combined dataset: {len(combined_dataset)} rows, "
                    f"weight col={'weight' in combined_dataset.columns}")

    return {
        'validated_edges': validated_edges,
        'obs_only_edges': obs_only_edges,
        'h0_edges': h0_edges,
        'h1_edges': h1_edges,
        'intervention_count': intervention_count,
        'discovery_time_s': round(discovery_time, 1),
        'max_iterations': max_iterations,
        'timestamp': datetime.now().isoformat(),
        'combined_dataset': combined_dataset,
    }


def save_edges(edge_data, path):
    """Save edge sets to JSON (sets -> lists for serialization)."""
    serializable = {}
    for k, v in edge_data.items():
        if isinstance(v, set):
            serializable[k] = sorted([list(e) for e in v])
        else:
            serializable[k] = v
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved edges to {path}")


def load_edges(path):
    """Load edge sets from JSON."""
    with open(path) as f:
        data = json.load(f)
    result = {}
    for k, v in data.items():
        if k.endswith('_edges') and isinstance(v, list) and v and isinstance(v[0], list):
            result[k] = {tuple(e) for e in v}
        else:
            result[k] = v
    logger.info(f"Loaded edges from {path}: "
                f"validated={len(result.get('validated_edges', []))}, "
                f"obs_only={len(result.get('obs_only_edges', []))}")
    return result


# ── Policy builders ───────────────────────────────────────────────────────────

def make_ashrae_fn():
    """Static ASHRAE baseline setpoints."""
    sp = {'Temperature': 0.5, 'Humidity': 0.5, 'AirQuality': 0.75}
    return lambda state, t: sp


def make_grid_fn(dataset, edges, reg_lambda=0.5, n_gradient_steps=1,
                 w_sat=0.6):
    """Build GRID policy with configurable lambda, gradient iterations, and
    satisfaction weight."""
    engine = CausalPolicyEngine(
        {'validated_edges': edges}, dataset,
        use_llm=False, api_key='',
        action_vars=ACTION_VARS,
        reg_lambda=reg_lambda,
    )
    objectives = {
        'satisfaction': {'weight': w_sat, 'direction': 'maximize'},
        'energy': {'weight': 1.0 - w_sat, 'direction': 'minimize'},
    }
    constraints = {v: (0.0, 1.0) for v in ACTION_VARS}
    sp = {'Temperature': 0.5, 'Humidity': 0.5, 'AirQuality': 0.75}

    def fn(state, t):
        if n_gradient_steps == 1:
            result = engine.optimize_policy(objectives, constraints, state)
            return result.action_plan if result else sp
        else:
            current = dict(state)
            for _ in range(n_gradient_steps):
                result = engine.optimize_policy(objectives, constraints, current)
                if result is None:
                    return sp
                for v in ACTION_VARS:
                    current[v] = result.action_plan.get(v, current.get(v, 0.5))
            return result.action_plan
    return fn


def make_regime_aware_fn(dataset, h0_edges, h1_edges, reg_lambda=0.5,
                         n_gradient_steps=1, w_sat=0.6):
    """Build regime-aware GRID: dispatches between H0/H1 engines based on
    observed window state."""
    engine_h0 = CausalPolicyEngine(
        {'validated_edges': h0_edges}, dataset,
        use_llm=False, api_key='',
        action_vars=ACTION_VARS,
        reg_lambda=reg_lambda,
    )
    engine_h1 = CausalPolicyEngine(
        {'validated_edges': h1_edges}, dataset,
        use_llm=False, api_key='',
        action_vars=ACTION_VARS,
        reg_lambda=reg_lambda,
    )
    objectives = {
        'satisfaction': {'weight': w_sat, 'direction': 'maximize'},
        'energy': {'weight': 1.0 - w_sat, 'direction': 'minimize'},
    }
    constraints = {v: (0.0, 1.0) for v in ACTION_VARS}
    sp = {'Temperature': 0.5, 'Humidity': 0.5, 'AirQuality': 0.75}

    def fn(state, t):
        window_open = state.get('WindowOpen', 0)
        engine = engine_h1 if window_open else engine_h0
        if n_gradient_steps == 1:
            result = engine.optimize_policy(objectives, constraints, state)
            return result.action_plan if result else sp
        else:
            current = dict(state)
            for _ in range(n_gradient_steps):
                result = engine.optimize_policy(objectives, constraints, current)
                if result is None:
                    return sp
                for v in ACTION_VARS:
                    current[v] = result.action_plan.get(v, current.get(v, 0.5))
            return result.action_plan
    return fn


def make_ddpc_fns(dataset, comfort_target=80.0, suite=None):
    """Build all 4 DDPC policy callables from dataset."""
    if suite is None:
        suite = create_ddpc_suite(dataset)
    fns = {}
    for name, ctrl in suite.items():
        def _make(c=ctrl, ct=comfort_target):
            def fn(state, t):
                return c.get_action(state, comfort_target=ct)
            return fn
        fns[name] = _make()
    return fns, suite


# ── Study runners ─────────────────────────────────────────────────────────────

def study_a_gradient_iterations(dataset, n_episodes=5):
    """Study A: How many gradient steps per timestep?"""
    print(f"\n{'='*70}")
    print(f"  STUDY A: Gradient Iterations (1, 5, 10, 20 steps)")
    print(f"{'='*70}\n")

    steps_list = [1, 5, 10, 20]
    rows = []

    # DDPC baselines (constant across configs)
    logger.info("Fitting DDPC baselines...")
    ddpc_fns, _ = make_ddpc_fns(dataset)
    ashrae_fn = make_ashrae_fn()

    for name, fn in [('ASHRAE', ashrae_fn)] + list(ddpc_fns.items()):
        logger.info(f"  {name}...")
        for ep in range(n_episodes):
            recs = run_episode(fn, seed=ep)
            sat, eng = summarize_episode(recs)
            rows.append({'study': 'A', 'method': name, 'config': 'default',
                         'episode': ep, 'satisfaction': sat, 'energy': eng,
                         'mo_score': mo_score(sat, eng)})

    # GRID with varying gradient steps
    for n_steps in steps_list:
        label = f'GRID_grad{n_steps}'
        logger.info(f"  {label}...")
        fn = make_grid_fn(dataset, EDGES, n_gradient_steps=n_steps)
        for ep in range(n_episodes):
            recs = run_episode(fn, seed=ep)
            sat, eng = summarize_episode(recs)
            rows.append({'study': 'A', 'method': label,
                         'config': f'steps={n_steps}',
                         'episode': ep, 'satisfaction': sat, 'energy': eng,
                         'mo_score': mo_score(sat, eng)})

    df = pd.DataFrame(rows)
    _print_study_summary(df, 'A', 'Gradient Iterations')
    _plot_study_bars(df, 'A', 'Gradient Iterations')
    return df


def study_b_lambda(dataset, n_episodes=5):
    """Study B: Step size regularisation."""
    print(f"\n{'='*70}")
    print(f"  STUDY B: Lambda Regularisation (0.1, 0.5, 1.0, 2.0)")
    print(f"{'='*70}\n")

    lambdas = [0.1, 0.5, 1.0, 2.0]
    rows = []

    # DDPC baselines
    logger.info("Fitting DDPC baselines...")
    ddpc_fns, _ = make_ddpc_fns(dataset)
    ashrae_fn = make_ashrae_fn()

    for name, fn in [('ASHRAE', ashrae_fn)] + list(ddpc_fns.items()):
        logger.info(f"  {name}...")
        for ep in range(n_episodes):
            recs = run_episode(fn, seed=ep)
            sat, eng = summarize_episode(recs)
            rows.append({'study': 'B', 'method': name, 'config': 'default',
                         'episode': ep, 'satisfaction': sat, 'energy': eng,
                         'mo_score': mo_score(sat, eng)})

    for lam in lambdas:
        label = f'GRID_lam{lam}'
        logger.info(f"  {label}...")
        fn = make_grid_fn(dataset, EDGES, reg_lambda=lam)
        for ep in range(n_episodes):
            recs = run_episode(fn, seed=ep)
            sat, eng = summarize_episode(recs)
            rows.append({'study': 'B', 'method': label,
                         'config': f'lambda={lam}',
                         'episode': ep, 'satisfaction': sat, 'energy': eng,
                         'mo_score': mo_score(sat, eng)})

    df = pd.DataFrame(rows)
    _print_study_summary(df, 'B', 'Lambda Regularisation')
    _plot_study_bars(df, 'B', 'Lambda Regularisation')
    return df


def study_c_regime_robustness(dataset, n_episodes=3):
    """Study C: Longer episodes with regime changes.

    50 timesteps x 30s = 1500s = 25 min.
    Tests GRID, GRID_ObsOnly, GRID_RegimeAware, DDPC, and ASHRAE.
    """
    print(f"\n{'='*70}")
    print(f"  STUDY C: Regime Robustness (50-step episodes)")
    print(f"{'='*70}\n")

    timesteps = 50
    rows = []

    # Build all arms
    logger.info("Fitting DDPC baselines...")
    ddpc_fns, _ = make_ddpc_fns(dataset)
    ashrae_fn = make_ashrae_fn()
    grid_fn = make_grid_fn(dataset, EDGES, n_gradient_steps=1)

    all_arms = [('ASHRAE', ashrae_fn), ('GRID', grid_fn)]

    # Add GRID_ObsOnly if we have obs edges
    if OBS_EDGES and OBS_EDGES != EDGES:
        obs_fn = make_grid_fn(dataset, OBS_EDGES, n_gradient_steps=1)
        all_arms.append(('GRID_ObsOnly', obs_fn))

    # Add GRID_RegimeAware if we have H0/H1 edges
    if REGIME_H0 and REGIME_H1:
        regime_fn = make_regime_aware_fn(dataset, REGIME_H0, REGIME_H1)
        all_arms.append(('GRID_RegimeAware', regime_fn))

    all_arms += list(ddpc_fns.items())

    for name, fn in all_arms:
        logger.info(f"  {name} ({timesteps} steps)...")
        for ep in range(n_episodes):
            recs = run_episode(fn, seed=ep, timesteps=timesteps)
            sat, eng = summarize_episode(recs)

            # Compute adaptation metrics
            window_changes = []
            for i in range(1, len(recs)):
                if recs[i]['window_open'] != recs[i-1]['window_open']:
                    window_changes.append(i)

            latencies = []
            for change_t in window_changes:
                if change_t == 0:
                    continue
                pre_sat = recs[change_t - 1]['satisfaction']
                threshold = pre_sat * 0.95
                recovered = False
                for k in range(change_t, min(change_t + 10, len(recs))):
                    if recs[k]['satisfaction'] >= threshold:
                        latencies.append(k - change_t)
                        recovered = True
                        break
                if not recovered:
                    latencies.append(10)

            avg_latency = np.mean(latencies) if latencies else 0.0

            sat_open = [r['satisfaction'] for r in recs if r['window_open']]
            sat_closed = [r['satisfaction'] for r in recs if not r['window_open']]

            rows.append({
                'study': 'C', 'method': name, 'config': f'{timesteps}steps',
                'episode': ep,
                'satisfaction': sat, 'energy': eng,
                'mo_score': mo_score(sat, eng),
                'n_regime_changes': len(window_changes),
                'adaptation_latency': avg_latency,
                'sat_window_open': np.mean(sat_open) if sat_open else 0.0,
                'sat_window_closed': np.mean(sat_closed) if sat_closed else 0.0,
            })

    df = pd.DataFrame(rows)
    _print_study_summary(df, 'C', 'Regime Robustness',
                         extra_cols=['adaptation_latency', 'sat_window_open',
                                     'sat_window_closed'])
    _plot_study_c(df)
    return df


def study_d_sample_efficiency(full_dataset, n_episodes=3):
    """Study D: How much data does each method need?"""
    print(f"\n{'='*70}")
    print(f"  STUDY D: Sample Efficiency (500, 1K, 2K, 5K, 10K rows)")
    print(f"{'='*70}\n")

    sample_sizes = [500, 1000, 2000, 5000, len(full_dataset)]
    rows = []

    for n in sample_sizes:
        logger.info(f"  --- Data size: {n} rows ---")
        if n >= len(full_dataset):
            subset = full_dataset
            n_label = len(full_dataset)
        else:
            subset = full_dataset.sample(n, random_state=42)
            n_label = n

        # GRID with discovered edges, SEM fitted on subset
        logger.info(f"    GRID (SEM on {n_label} rows)...")
        grid_fn = make_grid_fn(subset, EDGES)
        for ep in range(n_episodes):
            recs = run_episode(grid_fn, seed=ep)
            sat, eng = summarize_episode(recs)
            rows.append({'study': 'D', 'method': 'GRID',
                         'config': f'n={n_label}',
                         'episode': ep, 'satisfaction': sat, 'energy': eng,
                         'mo_score': mo_score(sat, eng),
                         'data_size': n_label})

        # All DDPC baselines trained on subset
        logger.info(f"    DDPC suite (trained on {n_label} rows)...")
        try:
            ddpc_fns, _ = make_ddpc_fns(subset)
        except Exception as e:
            logger.warning(f"    DDPC suite failed on {n_label} rows: {e}")
            continue

        for name, fn in ddpc_fns.items():
            logger.info(f"      {name}...")
            for ep in range(n_episodes):
                recs = run_episode(fn, seed=ep)
                sat, eng = summarize_episode(recs)
                rows.append({'study': 'D', 'method': name,
                             'config': f'n={n_label}',
                             'episode': ep, 'satisfaction': sat, 'energy': eng,
                             'mo_score': mo_score(sat, eng),
                             'data_size': n_label})

    df = pd.DataFrame(rows)
    _print_study_d(df)
    _plot_study_d(df)
    return df


def study_e_pareto_frontier(dataset, n_episodes=5):
    """Study E: Pareto frontier sweep + hypervolume."""
    print(f"\n{'='*70}")
    print(f"  STUDY E: Pareto Frontier Sweep")
    print(f"{'='*70}\n")

    rows = []

    # ASHRAE
    ashrae_fn = make_ashrae_fn()
    logger.info("  ASHRAE (static)...")
    for ep in range(n_episodes):
        recs = run_episode(ashrae_fn, seed=ep)
        sat, eng = summarize_episode(recs)
        rows.append({'study': 'E', 'method': 'ASHRAE',
                     'config': 'static', 'tradeoff_param': 0.5,
                     'episode': ep, 'satisfaction': sat, 'energy': eng,
                     'mo_score': mo_score(sat, eng)})

    # GRID — sweep w_sat, 10 gradient steps (best from Study A default)
    w_sat_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for w in w_sat_values:
        logger.info(f"  GRID (w_sat={w})...")
        fn = make_grid_fn(dataset, EDGES, n_gradient_steps=10, w_sat=w)
        for ep in range(n_episodes):
            recs = run_episode(fn, seed=ep)
            sat, eng = summarize_episode(recs)
            rows.append({'study': 'E', 'method': 'GRID',
                         'config': f'w_sat={w}', 'tradeoff_param': w,
                         'episode': ep, 'satisfaction': sat, 'energy': eng,
                         'mo_score': mo_score(sat, eng)})

    # DDPC — fit once, sweep comfort_target
    logger.info("  Fitting DDPC suite (once)...")
    _, suite = make_ddpc_fns(dataset)
    ct_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for ct in ct_values:
        for name, ctrl in suite.items():
            logger.info(f"  {name} (ct={ct})...")
            def _make_fn(c=ctrl, t=ct):
                def fn(state, ts):
                    return c.get_action(state, comfort_target=t)
                return fn
            fn = _make_fn()
            for ep in range(n_episodes):
                recs = run_episode(fn, seed=ep)
                sat, eng = summarize_episode(recs)
                rows.append({'study': 'E', 'method': name,
                             'config': f'ct={ct}', 'tradeoff_param': ct,
                             'episode': ep, 'satisfaction': sat, 'energy': eng,
                             'mo_score': mo_score(sat, eng)})

    df = pd.DataFrame(rows)
    _print_study_e(df)
    _plot_study_e(df)
    return df


# ── Output: printing ─────────────────────────────────────────────────────────

def _print_study_summary(df, study_id, title, extra_cols=None):
    """Print mean +/- std summary table for a study."""
    cols = ['satisfaction', 'energy', 'mo_score']
    if extra_cols:
        cols += extra_cols

    agg = df.groupby('method')[cols].agg(['mean', 'std']).fillna(0)
    agg.columns = [f'{c}_{s}' for c, s in agg.columns]
    agg = agg.sort_values('mo_score_mean', ascending=False)

    print(f"\n  {'Method':<25s}  {'Sat%':>10s}  {'Energy%':>10s}  {'MO-Score':>10s}", end='')
    if extra_cols:
        for c in extra_cols:
            print(f"  {c:>14s}", end='')
    print()
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}", end='')
    if extra_cols:
        for _ in extra_cols:
            print(f"  {'-'*14}", end='')
    print()

    for method, row in agg.iterrows():
        sat_m, sat_s = row['satisfaction_mean'], row['satisfaction_std']
        eng_m, eng_s = row['energy_mean'], row['energy_std']
        mo_m, mo_s = row['mo_score_mean'], row['mo_score_std']
        line = f"  {method:<25s}  {sat_m:5.1f}+/-{sat_s:3.1f}  {eng_m:5.1f}+/-{eng_s:3.1f}  {mo_m:6.4f}+/-{mo_s:4.4f}"
        if extra_cols:
            for c in extra_cols:
                m = row[f'{c}_mean']
                s = row[f'{c}_std']
                line += f"  {m:>6.1f}+/-{s:<5.1f}"
        print(line)

    # Highlight best GRID and best DDPC
    grid_rows = agg[agg.index.str.startswith('GRID')]
    ddpc_rows = agg[agg.index.str.startswith('DDPC')]
    if not grid_rows.empty:
        best_grid = grid_rows['mo_score_mean'].idxmax()
        print(f"\n  Best GRID: {best_grid} (MO={grid_rows.loc[best_grid, 'mo_score_mean']:.4f})")
    if not ddpc_rows.empty:
        best_ddpc = ddpc_rows['mo_score_mean'].idxmax()
        print(f"  Best DDPC: {best_ddpc} (MO={ddpc_rows.loc[best_ddpc, 'mo_score_mean']:.4f})")
    if not grid_rows.empty and not ddpc_rows.empty:
        gap = grid_rows['mo_score_mean'].max() - ddpc_rows['mo_score_mean'].max()
        winner = 'GRID' if gap > 0 else 'DDPC'
        print(f"  Gap: {winner} leads by {abs(gap):.4f} MO-score")


def _print_study_d(df):
    """Print sample efficiency summary by data size."""
    print(f"\n  {'Method':<20s}", end='')
    for n in sorted(df['data_size'].unique()):
        print(f"  {int(n):>8d}", end='')
    print("  (MO-score)")
    print(f"  {'-'*20}", end='')
    for _ in df['data_size'].unique():
        print(f"  {'-'*8}", end='')
    print()

    for method in sorted(df['method'].unique()):
        sub = df[df['method'] == method]
        line = f"  {method:<20s}"
        for n in sorted(df['data_size'].unique()):
            nsub = sub[sub['data_size'] == n]
            if nsub.empty:
                line += f"  {'--':>8s}"
            else:
                line += f"  {nsub['mo_score'].mean():>8.4f}"
        print(line)


def _print_study_e(df):
    """Print Pareto frontier summary with hypervolumes."""
    ref = np.array([0.0, -100.0])

    print(f"\n  {'Method':<25s}  {'HV':>8s}  {'Pareto pts':>10s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*10}")

    for method in sorted(df['method'].unique()):
        sub = df[df['method'] == method]
        points = sub[['satisfaction', 'energy']].values
        points_max = np.column_stack([points[:, 0], -points[:, 1]])
        sorted_pts = points_max[points_max[:, 0].argsort()[::-1]]

        pareto = []
        best_y = ref[1]
        for p in sorted_pts:
            if p[1] > best_y:
                pareto.append(p)
                best_y = p[1]

        if not pareto:
            print(f"  {method:<25s}  {'0.0':>8s}  {'0':>10s}")
            continue

        pareto = np.array(pareto)
        hv = 0.0
        for i, p in enumerate(pareto):
            x_w = p[0] - (pareto[i+1][0] if i+1 < len(pareto) else ref[0])
            y_h = p[1] - ref[1]
            hv += x_w * y_h

        print(f"  {method:<25s}  {hv:>8.1f}  {len(pareto):>10d}")


# ── Output: plots ─────────────────────────────────────────────────────────────

_COLORS = {
    'ASHRAE': '#95A5A6', 'GRID': '#2ECC71', 'GRID_ObsOnly': '#82E0AA',
    'GRID_RegimeAware': '#1ABC9C',
    'DDPC_Behavioral': '#3498DB', 'DDPC_Subspace': '#5DADE2',
    'DDPC_Neural': '#E74C3C', 'DDPC_PETS': '#F39C12',
}


def _get_color(name):
    for key, color in _COLORS.items():
        if key in name:
            return color
    # GRID variants
    if name.startswith('GRID'):
        return '#27AE60'
    return '#7F8C8D'


def _plot_study_bars(df, study_id, title):
    """Bar chart: MO-score by method + scatter: Sat vs Energy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Study {study_id}: {title}', fontsize=14, fontweight='bold')

    agg = df.groupby('method').agg(
        mo_mean=('mo_score', 'mean'), mo_std=('mo_score', 'std'),
        sat_mean=('satisfaction', 'mean'), eng_mean=('energy', 'mean'),
    ).sort_values('mo_mean', ascending=False).fillna(0)

    methods = list(agg.index)
    colors = [_get_color(m) for m in methods]

    # Left: MO-score bars
    bars = ax1.bar(range(len(methods)), agg['mo_mean'], yerr=agg['mo_std'],
                   capsize=4, color=colors, alpha=0.9, edgecolor='white', linewidth=0.8)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=35, ha='right', fontsize=8)
    ax1.set_ylabel('Multi-Objective Score')
    ax1.set_title('MO-Score (higher = better)')
    for i, (bar, val) in enumerate(zip(bars, agg['mo_mean'])):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Right: Sat vs Energy scatter
    for method in methods:
        sub = df[df['method'] == method]
        ax2.scatter(sub['satisfaction'], sub['energy'],
                    label=method, color=_get_color(method), alpha=0.7,
                    s=50, edgecolors='black', linewidth=0.3)
    ax2.set_xlabel('Satisfaction (%)')
    ax2.set_ylabel('Energy (%)')
    ax2.set_title('Satisfaction vs Energy (bottom-right = optimal)')
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'study_{study_id.lower()}_{title.lower().replace(" ", "_")}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_study_c(df):
    """Study C: MO-score bars + adaptation latency bars + regime-split satisfaction."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Study C: Regime Robustness (50-step episodes)', fontsize=14, fontweight='bold')

    agg = df.groupby('method').agg(
        mo_mean=('mo_score', 'mean'), mo_std=('mo_score', 'std'),
        lat_mean=('adaptation_latency', 'mean'), lat_std=('adaptation_latency', 'std'),
        sat_open_mean=('sat_window_open', 'mean'),
        sat_closed_mean=('sat_window_closed', 'mean'),
    ).sort_values('mo_mean', ascending=False).fillna(0)

    methods = list(agg.index)
    colors = [_get_color(m) for m in methods]
    x = range(len(methods))

    # MO-score
    ax1.bar(x, agg['mo_mean'], yerr=agg['mo_std'], capsize=4,
            color=colors, alpha=0.9, edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=35, ha='right', fontsize=8)
    ax1.set_ylabel('MO-Score')
    ax1.set_title('Overall Performance')
    ax1.grid(axis='y', alpha=0.3)

    # Adaptation latency
    ax2.bar(x, agg['lat_mean'], yerr=agg['lat_std'], capsize=4,
            color=colors, alpha=0.9, edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=35, ha='right', fontsize=8)
    ax2.set_ylabel('Adaptation Latency (steps)')
    ax2.set_title('Regime Change Recovery (lower = better)')
    ax2.grid(axis='y', alpha=0.3)

    # Regime-split satisfaction
    width = 0.35
    x_arr = np.arange(len(methods))
    ax3.bar(x_arr - width/2, agg['sat_closed_mean'], width, label='Window Closed',
            color='#3498DB', alpha=0.8)
    ax3.bar(x_arr + width/2, agg['sat_open_mean'], width, label='Window Open',
            color='#E74C3C', alpha=0.8)
    ax3.set_xticks(x_arr)
    ax3.set_xticklabels(methods, rotation=35, ha='right', fontsize=8)
    ax3.set_ylabel('Satisfaction (%)')
    ax3.set_title('Satisfaction by Regime')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'study_c_regime_robustness.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_study_d(df):
    """Study D: Line plot of MO-score vs data size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Study D: Sample Efficiency', fontsize=14, fontweight='bold')

    for method in sorted(df['method'].unique()):
        sub = df[df['method'] == method]
        agg = sub.groupby('data_size')['mo_score'].agg(['mean', 'std']).reset_index()
        ax.errorbar(agg['data_size'], agg['mean'], yerr=agg['std'].fillna(0),
                    marker='o', capsize=4, label=method, color=_get_color(method),
                    linewidth=2)

    ax.set_xlabel('Training Data Size (rows)')
    ax.set_ylabel('MO-Score')
    ax.set_title('Performance vs Training Data Size')
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'study_d_sample_efficiency.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_study_e(df):
    """Study E: Pareto frontier scatter."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('Study E: Pareto Frontier Sweep', fontsize=14, fontweight='bold')

    markers = {'ASHRAE': 's', 'GRID': 'o',
               'DDPC_Behavioral': 'D', 'DDPC_Subspace': '^',
               'DDPC_Neural': 'v', 'DDPC_PETS': 'P'}

    for method in sorted(df['method'].unique()):
        sub = df[df['method'] == method]
        mk = markers.get(method, 'o')
        ax.scatter(sub['satisfaction'], sub['energy'],
                   label=method, marker=mk, color=_get_color(method),
                   alpha=0.6, s=40, edgecolors='black', linewidth=0.3)

        # Connect means per config with a line (Pareto front)
        means = sub.groupby('config')[['satisfaction', 'energy']].mean()
        means = means.sort_values('satisfaction')
        ax.plot(means['satisfaction'], means['energy'], color=_get_color(method),
                linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Satisfaction (%)', fontsize=12)
    ax.set_ylabel('Energy (%)', fontsize=12)
    ax.set_title('Pareto Frontiers (bottom-right = optimal zone)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'study_e_pareto_frontier.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Master summary ────────────────────────────────────────────────────────────

def write_summary(study_dfs, edge_info, wall_time):
    """Write human-readable summary to summary.txt and print to console."""
    lines = []
    lines.append("=" * 70)
    lines.append("  POLICY ABLATION SUMMARY")
    lines.append("=" * 70)

    # Edge info
    n_val = len(edge_info.get('validated_edges', []))
    n_obs = len(edge_info.get('obs_only_edges', []))
    n_intv = edge_info.get('intervention_count', 'N/A')
    disc_t = edge_info.get('discovery_time_s', 'N/A')
    max_it = edge_info.get('max_iterations', 'N/A')
    lines.append(f"  Discovery: {max_it} iter, {n_val} validated edges, "
                 f"{n_obs} obs-only, {n_intv} interventions")
    lines.append(f"  Discovery time: {disc_t}s")
    lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Total wall time: {wall_time:.0f}s ({wall_time/60:.1f} min)")
    lines.append("")

    # Validated edges
    lines.append("  Validated edges:")
    for e in sorted(edge_info.get('validated_edges', [])):
        lines.append(f"    {e[0]} -> {e[1]}")
    lines.append("")

    # Per-study best results
    for study_id, label in [('A', 'Gradient Iterations'),
                            ('B', 'Lambda Regularisation'),
                            ('C', 'Regime Robustness'),
                            ('D', 'Sample Efficiency'),
                            ('E', 'Pareto Frontier')]:
        df = study_dfs.get(study_id)
        if df is None or df.empty:
            continue

        lines.append(f"  STUDY {study_id}: {label}")
        lines.append(f"  {'-'*60}")

        agg = df.groupby('method')['mo_score'].agg(['mean', 'std']).fillna(0)
        agg = agg.sort_values('mean', ascending=False)

        for method, row in agg.iterrows():
            lines.append(f"    {method:<25s}  MO={row['mean']:.4f} +/- {row['std']:.4f}")

        # Best GRID vs best DDPC
        grid = agg[agg.index.str.startswith('GRID')]
        ddpc = agg[agg.index.str.startswith('DDPC')]
        if not grid.empty and not ddpc.empty:
            best_g = grid['mean'].max()
            best_d = ddpc['mean'].max()
            gap = best_g - best_d
            winner = 'GRID' if gap > 0 else 'DDPC'
            lines.append(f"    >>> {winner} leads by {abs(gap):.4f} MO-score")
        lines.append("")

    text = '\n'.join(lines)
    print(text)

    path = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(path, 'w') as f:
        f.write(text)
    print(f"\n  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global EDGES, OBS_EDGES, REGIME_H0, REGIME_H1

    parser = argparse.ArgumentParser(description='Policy Engine Ablation Study')
    parser.add_argument('--discover', action='store_true',
                        help='Run CWM discovery pipeline first')
    parser.add_argument('--max-iter', type=int, default=25,
                        help='Max discovery iterations (default: 25)')
    parser.add_argument('--edges-file', type=str, default=None,
                        help='Load edges from JSON file (skip discovery)')
    parser.add_argument('--study', type=str, default=None,
                        help='Run specific studies (comma-separated: a,b,c,d,e)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Episodes per config (default: 5)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Resolve edge sets ─────────────────────────────────────────────────
    edge_info = {}

    if args.discover:
        print("\n" + "=" * 70)
        print("  PHASE 0: CWM Discovery Pipeline")
        print("=" * 70 + "\n")

        edge_info = run_discovery(max_iterations=args.max_iter)
        EDGES = edge_info['validated_edges']
        OBS_EDGES = edge_info['obs_only_edges']
        REGIME_H0 = edge_info.get('h0_edges')
        REGIME_H1 = edge_info.get('h1_edges')

        # Save edges for reproducibility
        save_edges(edge_info, os.path.join(OUTPUT_DIR, 'discovered_edges.json'))

    elif args.edges_file:
        edge_info = load_edges(args.edges_file)
        EDGES = edge_info['validated_edges']
        OBS_EDGES = edge_info.get('obs_only_edges', set())
        REGIME_H0 = edge_info.get('h0_edges')
        REGIME_H1 = edge_info.get('h1_edges')

    else:
        logger.info("Using fallback edges (no discovery)")
        EDGES = FALLBACK_EDGES
        OBS_EDGES = FALLBACK_EDGES  # same as validated for fallback
        REGIME_H0 = FALLBACK_H0
        REGIME_H1 = FALLBACK_H1
        edge_info = {
            'validated_edges': EDGES,
            'obs_only_edges': OBS_EDGES,
            'h0_edges': REGIME_H0,
            'h1_edges': REGIME_H1,
            'intervention_count': 'N/A (fallback)',
            'discovery_time_s': 0,
            'max_iterations': 'N/A (fallback)',
        }

    print(f"\n  Edge sets loaded:")
    print(f"    Validated:  {len(EDGES)} edges")
    print(f"    Obs-only:   {len(OBS_EDGES)} edges")
    print(f"    H0 (closed): {len(REGIME_H0) if REGIME_H0 else 0} edges")
    print(f"    H1 (open):   {len(REGIME_H1) if REGIME_H1 else 0} edges")
    print(f"    Validated edges:")
    for e in sorted(EDGES):
        print(f"      {e[0]} -> {e[1]}")
    print()

    # ── Load dataset ──────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    obs_dataset = pd.read_csv(DATA_PATH)
    logger.info(f"Obs dataset: {len(obs_dataset)} rows, {list(obs_dataset.columns)}")

    # Use combined dataset (obs + intervention with weight column) if
    # available from discovery; otherwise fall back to obs-only.
    combined_ds = edge_info.get('combined_dataset')
    if combined_ds is not None and 'weight' in combined_ds.columns:
        dataset = combined_ds
        logger.info(f"Using combined dataset for GRID: {len(dataset)} rows "
                    f"({(dataset['weight'] > 1).sum()} intervention rows)")
    else:
        dataset = obs_dataset
        logger.info("No combined dataset available; using obs-only for GRID")

    # Verify simulator
    logger.info("Verifying simulator...")
    state = get_initial_state(0)
    action = {'Temperature': 0.5, 'Humidity': 0.5, 'AirQuality': 0.75}
    out = step_simulator(state, action, 0)
    logger.info(f"Simulator check: sat={out.get('OverallSatisfaction', '?'):.1f}, "
                f"eng={out.get('EnergyConsumption', '?'):.1f}")

    # ── Determine which studies to run ────────────────────────────────────
    if args.study:
        studies = [s.strip().upper() for s in args.study.split(',')]
    else:
        studies = ['A', 'B', 'C', 'D', 'E']

    n_ep = args.episodes
    t0 = time.time()
    study_dfs = {}

    # ── Run studies ───────────────────────────────────────────────────────
    if 'A' in studies:
        study_dfs['A'] = study_a_gradient_iterations(dataset, n_episodes=n_ep)
        study_dfs['A'].to_csv(os.path.join(OUTPUT_DIR, 'study_a_gradient_steps.csv'),
                              index=False)

    if 'B' in studies:
        study_dfs['B'] = study_b_lambda(dataset, n_episodes=n_ep)
        study_dfs['B'].to_csv(os.path.join(OUTPUT_DIR, 'study_b_lambda.csv'),
                              index=False)

    if 'C' in studies:
        study_dfs['C'] = study_c_regime_robustness(dataset,
                                                    n_episodes=max(3, n_ep))
        study_dfs['C'].to_csv(os.path.join(OUTPUT_DIR, 'study_c_regime_robustness.csv'),
                              index=False)

    if 'D' in studies:
        study_dfs['D'] = study_d_sample_efficiency(dataset,
                                                    n_episodes=max(3, n_ep))
        study_dfs['D'].to_csv(os.path.join(OUTPUT_DIR, 'study_d_sample_efficiency.csv'),
                              index=False)

    if 'E' in studies:
        study_dfs['E'] = study_e_pareto_frontier(dataset, n_episodes=n_ep)
        study_dfs['E'].to_csv(os.path.join(OUTPUT_DIR, 'study_e_pareto_frontier.csv'),
                              index=False)

    # ── Combined CSV ──────────────────────────────────────────────────────
    all_dfs = [df for df in study_dfs.values() if not df.empty]
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(OUTPUT_DIR, 'all_studies_combined.csv'),
                        index=False)

    wall_time = time.time() - t0

    # ── Master summary ────────────────────────────────────────────────────
    write_summary(study_dfs, edge_info, wall_time)

    print(f"\n  All results in: {OUTPUT_DIR}/")
    print(f"  Total ablation time: {wall_time:.0f}s ({wall_time/60:.1f} min)")


if __name__ == '__main__':
    main()
