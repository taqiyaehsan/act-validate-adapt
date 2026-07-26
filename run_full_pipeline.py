#!/usr/bin/env python3
"""
Full Pipeline: Discovery → Monitoring → Policy, per seed.
==========================================================
Runs the entire PolicyGRID pipeline independently for each seed,
then aggregates results with mean ± std across seeds.

Usage:
    python run_full_pipeline.py --sim smart_building_rich --seeds 3
    python run_full_pipeline.py --sim all --seeds 5
    python run_full_pipeline.py --sim open_window --seed-list 42,123,456

Output: results/full_pipeline/<sim_name>/
    ├── seed_42/
    │   ├── edges.json           # discovered edges for this seed
    │   ├── discovery_metrics.json
    │   ├── monitoring_metrics.json  (if applicable)
    │   └── policy_metrics.csv
    ├── seed_123/
    │   └── ...
    ├── benchmarks.csv           # discovery benchmarks (run once)
    ├── consensus_analysis.json  # edge stability, Jaccard, consensus graph
    └── summary.json             # mean ± std across seeds
"""

import sys, os, json, time, warnings, argparse, random
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
sys.path.insert(0, 'benchmarks_new')
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

import subprocess
from src.policy_engine import CausalPolicyEngine

# ═══════════════════════════════════════════════════════════════════════════════
# SIM CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════

SIM_CONFIGS = {
    'smart_room': {
        'sim_path': 'js/smart_room.js',
        'data_path': 'data_regen/smart_room_varied_processed.csv',
        'scaling_path': 'data_regen/smart_room_varied_scaling.csv',
        'gt_key': 'smart_room',
        'has_monitoring': False,
        'action_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'output_vars': ['EnergyConsumption', 'Satisfaction'],
        'state_vars': ['Temperature', 'Humidity', 'AirQuality',
                       'EnergyConsumption', 'Satisfaction'],
        'actuators': ['temperature', 'humidity', 'airquality'],
        'non_intervenable': ['energyconsumption', 'satisfaction'],
        'dataset_type': 'smart_room',
        'js_var_map': {'temperature': 'temperature', 'humidity': 'humidity',
                       'airquality': 'airQuality', 'energyconsumption': 'energyConsumption',
                       'satisfaction': 'overallSatisfaction'},
        'sat_key': 'overallSatisfaction', 'energy_key': 'energyConsumption',
        'latent_vars': set(),
        'policy_config': {'linear_only': False, 'lambda': 5.0,
                         'intervention_directions': None},
    },
    'smart_room_noise': {
        'sim_path': 'js/smart_room_noise.js',
        'data_path': 'data_regen/smart_room_noise_varied_processed.csv',
        'scaling_path': 'data_regen/smart_room_noise_varied_scaling.csv',
        'gt_key': 'smart_room_noise',
        'has_monitoring': False,
        'action_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'output_vars': ['EnergyConsumption', 'Satisfaction'],
        'state_vars': ['Temperature', 'Humidity', 'AirQuality',
                       'EnergyConsumption', 'Satisfaction'],
        'actuators': ['temperature', 'humidity', 'airquality'],
        'non_intervenable': ['energyconsumption', 'satisfaction'],
        'dataset_type': 'smart_room_noise',
        'js_var_map': {'temperature': 'temperature', 'humidity': 'humidity',
                       'airquality': 'airQuality', 'energyconsumption': 'energyConsumption',
                       'satisfaction': 'overallSatisfaction'},
        'sat_key': 'overallSatisfaction', 'energy_key': 'energyConsumption',
        'latent_vars': set(),
        'policy_config': {'linear_only': False, 'lambda': 5.0,
                         'intervention_directions': None},
    },
    'smart_room_hidden_vars': {
        'sim_path': 'js/smart_room_hidden_vars.js',
        'data_path': 'data_regen/smart_room_hidden_vars_varied_processed.csv',
        'scaling_path': 'data_regen/smart_room_hidden_vars_varied_scaling.csv',
        'gt_key': 'hidden_vars',
        'has_monitoring': False,
        'action_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'output_vars': ['EnergyConsumption', 'Satisfaction'],
        'state_vars': ['Temperature', 'Humidity', 'AirQuality',
                       'EnergyConsumption', 'Satisfaction'],
        'actuators': ['temperature', 'humidity', 'airquality'],
        'non_intervenable': ['energyconsumption', 'satisfaction'],
        'dataset_type': 'hidden_vars',
        'js_var_map': {'temperature': 'temperature', 'humidity': 'humidity',
                       'airquality': 'airQuality', 'energyconsumption': 'energyConsumption',
                       'satisfaction': 'overallSatisfaction'},
        'sat_key': 'overallSatisfaction', 'energy_key': 'energyConsumption',
        'latent_vars': set(),
        'policy_config': {'linear_only': False, 'lambda': 5.0,
                         'intervention_directions': None},
    },
    'open_window': {
        'sim_path': 'js/open_window.js',
        'data_path': 'data_regen/open_window_varied_processed.csv',
        'scaling_path': 'data_regen/open_window_varied_scaling.csv',
        'gt_key': 'open_window',
        'has_monitoring': True,
        'regime_vars': {'windowopen'},
        'observable_vars': ['temperature', 'humidity', 'airquality', 'pmv',
                           'energyconsumption', 'satisfaction'],
        'action_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'output_vars': ['EnergyConsumption', 'Satisfaction'],
        'state_vars': ['Temperature', 'Humidity', 'AirQuality', 'PMV',
                       'EnergyConsumption', 'Satisfaction', 'WindowOpen',
                       'OutdoorTemperature'],
        'actuators': ['temperature', 'humidity', 'airquality'],
        'non_intervenable': ['windowopen', 'outdoortemperature', 'pmv',
                            'energyconsumption', 'satisfaction', 'elapsedtime'],
        'dataset_type': 'open_window',
        'js_var_map': {'temperature': 'temperature', 'humidity': 'humidity',
                       'airquality': 'airQuality', 'energyconsumption': 'energyConsumption',
                       'satisfaction': 'overallSatisfaction', 'pmv': 'pmv',
                       'windowopen': 'windowOpen', 'outdoortemperature': 'outdoorTemperature'},
        'sat_key': 'overallSatisfaction', 'energy_key': 'energyConsumption',
        'latent_vars': {'windowopen', 'outdoortemperature'},
        'policy_config': {'linear_only': False, 'lambda': 5.0,
                         'intervention_directions': None},
    },
    'smart_building_rich': {
        'sim_path': 'js/smart_building_rich.js',
        'data_path': 'data_regen/smart_building_rich_processed.csv',
        'scaling_path': 'data_regen/smart_building_rich_processed_scaling.csv',
        'gt_key': 'smart_building_rich',
        'has_monitoring': True,
        'regime_vars': {'occupancy', 'windowposition'},
        'observable_vars': ['outdoortemp', 'solarradiation', 'temperature',
                           'humidity', 'co2', 'lightlevel', 'noisedb',
                           'airquality', 'pmv', 'hvacpower', 'lightingpower',
                           'energyconsumption', 'satisfaction'],
        'action_vars': ['HVACPower', 'LightingPower'],
        'output_vars': ['EnergyConsumption', 'Satisfaction'],
        'state_vars': ['OutdoorTemp', 'SolarRadiation', 'Temperature', 'Humidity',
                       'CO2', 'LightLevel', 'NoiseDB', 'AirQuality', 'PMV',
                       'HVACPower', 'LightingPower', 'EnergyConsumption', 'Satisfaction'],
        'actuators': ['hvacpower', 'lightingpower'],
        'non_intervenable': ['outdoortemp', 'solarradiation', 'occupancy',
                            'windowposition', 'pmv', 'energyconsumption', 'satisfaction'],
        'dataset_type': 'smart_building_rich',
        'js_var_map': {'hvacpower': 'hvacPower', 'lightingpower': 'lightingPower'},
        'sat_key': 'satisfaction', 'energy_key': 'energyConsumption',
        'latent_vars': {'occupancy', 'windowposition'},
        'policy_config': {
            'linear_only': True, 'lambda': 0.5,
            'intervention_directions': {
                ('HVACPower', 'Temperature'): +1,
                ('HVACPower', 'EnergyConsumption'): +1,
                ('LightingPower', 'LightLevel'): +1,
                ('LightingPower', 'EnergyConsumption'): +1,
            },
        },
    },
    'ashrae': {
        'sim_path': None,
        'data_path': 'data/ashrae_data_processed.csv',
        'scaling_path': 'data/ashrae_data_processed_scaling_params.csv',
        'gt_key': 'ashrae',
        'has_monitoring': False,
        'action_vars': [],
        'output_vars': [],
        'state_vars': [],
        'actuators': [],
        'non_intervenable': ['meter_reading'],
        'dataset_type': 'ashrae',
        'latent_vars': set(),
        'policy_config': None,
    },
}

DEFAULT_SEEDS = [42, 123, 456, 789, 999]
API_KEY = 'YOUR_OPENAI_API_KEY'
COMFORT_TARGETS = [0.5, 0.8, 1.0, 1.5, 2.0]
N_POLICY_RUNS = 3
N_POLICY_STEPS = 120
N_CONSENSUS_POLICY_SEEDS = 3  # seeds for consensus-graph policy evaluation

# Physical constants for kWh computation per sim
SIM_POWER_SPECS = {
    'smart_building_rich': {'hvacpower': 5.0, 'lightingpower': 0.5},  # kW
    'open_window': {'temperature': 2.0, 'humidity': 0.5, 'airquality': 0.3},  # kW
    'smart_room': {'temperature': 2.0, 'humidity': 0.5, 'airquality': 0.3},
    'smart_room_noise': {'temperature': 2.0, 'humidity': 0.5, 'airquality': 0.3},
    'smart_room_hidden_vars': {'temperature': 2.0, 'humidity': 0.5, 'airquality': 0.3},
}
STEP_HOURS = 1.0 / 60  # each step = 1 minute


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def graph_metrics(predicted, gt):
    pred = {(s.lower(), t.lower()) for s, t in predicted}
    tp = len(pred & gt); fp = len(pred - gt); fn = len(gt - pred)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-10)
    return {'f1': round(f1, 4), 'precision': round(p, 4), 'recall': round(r, 4),
            'shd': fp + fn, 'n_edges': len(pred), 'tp': tp, 'fp': fp, 'fn': fn}


def load_scaling(path):
    sp = pd.read_csv(path)
    feat_col = 'feature' if 'feature' in sp.columns else 'column'
    smap = {}
    for _, row in sp.iterrows():
        name = row[feat_col].lower()
        if 'data_min' in sp.columns:
            smap[name] = (row['data_min'], row['data_max'])
        elif 'min' in sp.columns:
            smap[name] = (row['min'], row['max'])
        elif 'center' in sp.columns:
            # StandardScaler format: approximate min/max from center ± 3*scale
            c, s = row['center'], row['scale_factor']
            smap[name] = (c - 3 * s, c + 3 * s)
        else:
            smap[name] = (0, 1)
    return smap


def norm_to_phys(val, var, smap):
    if var in smap:
        mn, mx = smap[var]; return mn + val * (mx - mn)
    return val


def phys_to_norm(val, var, smap):
    if var in smap:
        mn, mx = smap[var]; return (val - mn) / (mx - mn + 1e-12)
    return val


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def run_discovery(cfg, data, seed, generators=None):
    """Run PolicyGRID discovery for one seed.

    generators: optional subset of {'pc','sam','llm','varlingam'} for the
    leave-one-generator-out ablation (rebuttal C2). None = all four.

    Returns: (validated_edges, obs_only_union_edges, method_edges)
    where method_edges maps generator name -> list of Phase-1 candidate edges.
    """
    from src.pipeline_cwm import create_cwm_pipeline

    random.seed(seed); np.random.seed(seed)
    sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None

    pipeline = create_cwm_pipeline(
        csv_data=data, api_key=API_KEY, smart_room_path=sim_path,
        dataset_type=cfg['dataset_type'], max_iterations=10,
        actuator_vars=cfg['actuators'],
        non_intervenable_vars=cfg['non_intervenable'],
        enabled_generators=generators,
    )
    pipeline.run_with_cwm()
    validated = set(pipeline.pipeline.validated_edges)
    # edge_ranker is attached to the INNER pipeline (pipeline_cwm.py:432).
    inner = getattr(pipeline, 'pipeline', None)
    obs_only_union = set()
    method_edges = {}
    if inner is not None:
        if getattr(inner, 'edge_ranker', None) is not None:
            obs_only_union = set(inner.edge_ranker.get_union_edges())
        # Per-generator Phase-1 candidates (provenance for the C2
        # complementarity analysis; previously not persisted).
        try:
            method_edges = {m: sorted(list(g.edges()))
                            for m, g in inner.get_method_dags().items()}
        except Exception as e:
            logger.warning(f"    Could not extract per-method DAGs: {e}")
    return validated, obs_only_union, method_edges


def run_benchmarks(cfg, data):
    """Run 10 discovery benchmarks. Returns dict {name: edge_set}."""
    sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None
    benchmarks = [
        ('PC', 'pc_baseline', 'run_pc_baseline'),
        ('SAM', 'sam_baseline', 'run_sam_baseline'),
        ('GIES', 'gies_comparison', 'run_gies'),
        ('ICP', 'icp_comparison', 'run_icp'),
        ('NOTEARS-I', 'notears_i', 'run_notears_i'),
        ('ABCD', 'abcd_comparison', 'run_abcd'),
        ('JCI', 'jci_comparison', 'run_jci'),
        ('Causal Bandits', 'causal_bandits', 'run_causal_bandits'),
        ('IID', 'iid_comparison', 'run_iid'),
    ]
    results = {}
    for name, mod_name, func_name in benchmarks:
        try:
            logger.info(f"      {name}...")
            import importlib
            mod = importlib.import_module(mod_name)
            func = getattr(mod, func_name)
            result = func(data, simulation_path=sim_path)
            if isinstance(result, tuple) and len(result) == 2: edges = result[0]
            elif hasattr(result, 'edges'): edges = list(result.edges())
            elif isinstance(result, dict): edges = result.get('edges', [])
            else: edges = result
            results[name] = {(s.lower(), t.lower()) for s, t in edges}
        except Exception as e:
            logger.warning(f"      {name} FAILED: {e}")
            results[name] = set()

    # LLM-only baseline: run LLM generator alone (no validation, no other methods)
    try:
        logger.info(f"      LLM-only...")
        from src.generators import LLMGenerator
        from src.gpt_client import GPTClient
        api_key = os.environ.get('OPENAI_API_KEY', '')
        cols = [c for c in data.columns if c.lower() not in cfg.get('latent_vars', set())]
        llm_gen = LLMGenerator(api_key=api_key, relevant_columns=cols,
                               dataset_type=cfg.get('dataset_type', 'smart_room'))
        llm_result = llm_gen.generate(data)
        if isinstance(llm_result, dict) and 'edges' in llm_result:
            llm_edges = llm_result['edges']
        elif hasattr(llm_result, 'edges'):
            llm_edges = list(llm_result.edges())
        else:
            llm_edges = llm_result if llm_result else []
        results['LLM-only'] = {(s.lower(), t.lower()) for s, t in llm_edges}
    except Exception as e:
        logger.warning(f"      LLM-only FAILED: {e}")
        results['LLM-only'] = set()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

def run_monitoring_for_seed(cfg, data, edges, smap):
    """Run monitoring evaluation using this seed's discovered edges.

    Supports:
      - smart_building_rich: 4-way (base/occ/win/full) via neurips_final_experiments
      - open_window: 2-way (window-closed/window-open) built inline
    """
    if not cfg['has_monitoring']:
        return None

    if cfg['dataset_type'] == 'smart_building_rich':
        return _monitoring_rich_building(cfg, data, edges, smap)
    elif cfg['dataset_type'] == 'open_window':
        return _monitoring_open_window(cfg, data, edges, smap)
    else:
        logger.info("      Monitoring not applicable for this sim")
        return None


def _monitoring_rich_building(cfg, data, edges, smap):
    """4-way monitoring for smart_building_rich."""
    try:
        from neurips_final_experiments import (
            construct_factored_monitors, collect_sim_data,
            run_monitoring, compute_monitoring_metrics, get_scaling_map
        )
    except ImportError:
        logger.warning("      Cannot import monitoring functions")
        return None

    monitors = construct_factored_monitors(edges, data)
    if monitors is None:
        return None

    scaling = pd.read_csv(cfg['scaling_path'])
    smap_full = get_scaling_map(scaling)
    sim_records = collect_sim_data(smap_full, duration_steps=1440)
    records = run_monitoring(monitors, sim_records)
    return compute_monitoring_metrics(records)


def _monitoring_open_window(cfg, data, edges, smap):
    """2-way monitoring for open_window — mirrors rich_building architecture.

    Uses the same PredictiveModel + Bayesian update as the working 4-way
    monitoring, just with 2 regime models (window-closed vs window-open).
    """
    import networkx as nx
    from src.causal_world_model import PredictiveModel

    obs_vars = cfg.get('observable_vars',
                       [c.lower() for c in data.columns
                        if c.lower() not in cfg.get('latent_vars', set())])
    obs_set = set(obs_vars)

    # Observable-only edges (same filter as rich_building's build_observable_graph)
    obs_edges = {(s.lower(), t.lower()) for s, t in edges
                 if s.lower() in obs_set and t.lower() in obs_set}

    if len(obs_edges) == 0:
        logger.warning("      No observable edges for monitoring")
        return None

    # Split data by window regime
    col_map = {c.lower(): c for c in data.columns}
    win_col = col_map.get('windowopen')
    if win_col is None:
        logger.warning("      No WindowOpen column in data")
        return None

    win_vals = data[win_col].values
    win_on = win_vals >= 0.5

    # Build models exactly like rich_building: graph over obs_vars, fit on masked data
    def _build_model(mask):
        G = nx.DiGraph()
        G.add_nodes_from(obs_vars)
        G.add_edges_from(obs_edges)
        model = PredictiveModel(G, obs_vars)
        obs_cols = [c for c in data.columns if c.lower() in obs_set]
        df = data[obs_cols].copy()
        df.columns = [c.lower() for c in df.columns]
        temporal = {}
        for var in obs_vars:
            if var in df.columns:
                temporal[var] = df[var].values[:-1]
                temporal[f'{var}_next'] = df[var].values[1:]
        tdf = pd.DataFrame(temporal)
        tdf = tdf[mask[:-1]].reset_index(drop=True)
        if len(tdf) > 10:
            model.fit_temporal(tdf)
        return model

    model_closed = _build_model(~win_on)
    model_open = _build_model(win_on)

    logger.info(f"      Built 2-way monitors: closed={int((~win_on).sum())} rows, "
                f"open={int(win_on.sum())} rows, {len(obs_edges)} obs edges")

    # Compute sensor weights from graph (same as rich_building)
    latent = cfg.get('latent_vars', set())
    sensor_weights = {v: 1.0 for v in obs_vars}
    for s, t in edges:
        sl, tl = s.lower(), t.lower()
        if sl in latent and tl in obs_vars:
            sensor_weights[tl] += 1.0
    logger.info(f"      Sensor weights: {sensor_weights}")

    # Collect test data via simulator (with persistent state)
    sim_path = os.path.abspath(cfg['sim_path'])
    sigma = 2.0
    forget_factor = 0.12
    weights = {'closed': 0.5, 'open': 0.5}
    records = []
    state_phys = None

    # Run 240 steps (covers window open/close schedule)
    for step_i in range(240):
        elapsed_ms = step_i * 60 * 1000
        cmd = ['node', sim_path, '--single-step',
               '--elapsed-ms', str(elapsed_ms)]
        if state_phys:
            cmd += ['--state', json.dumps(state_phys)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except:
            continue

        for line in result.stdout.split('\n'):
            if 'RESULT:' in line:
                state_phys = json.loads(line.split('RESULT:', 1)[1].strip())
                break
        else:
            continue

        # Normalize to [0,1]
        state_norm = {}
        for k, v in state_phys.items():
            if isinstance(v, (int, float)):
                state_norm[k.lower()] = phys_to_norm(v, k.lower(), smap)

        # GT regime
        win_gt = state_phys.get('windowOpen', False)
        win_active = 1 if (isinstance(win_gt, bool) and win_gt) or (isinstance(win_gt, (int, float)) and win_gt > 0.3) else 0

        # Observable state only
        state_obs = {k: v for k, v in state_norm.items()
                     if k not in latent}

        # Predict from each model — same as run_monitoring in neurips_final
        pred_closed = model_closed.predict(state_obs)
        pred_open = model_open.predict(state_obs)

        err_closed = sum(sensor_weights.get(v, 1.0) *
                         abs(state_obs.get(v, 0.5) - pred_closed.get(v, 0.5))
                         for v in obs_vars if v in state_obs)
        err_open = sum(sensor_weights.get(v, 1.0) *
                       abs(state_obs.get(v, 0.5) - pred_open.get(v, 0.5))
                       for v in obs_vars if v in state_obs)

        lik_closed = np.exp(-sigma * err_closed)
        lik_open = np.exp(-sigma * err_open)

        unnorm = {'closed': lik_closed * weights['closed'],
                  'open': lik_open * weights['open']}
        total = sum(unnorm.values())
        if total > 0:
            raw = {r: unnorm[r] / total for r in unnorm}
            weights = {r: (1 - forget_factor) * raw[r] + forget_factor * 0.5
                       for r in raw}
        w_total = sum(weights.values())
        if w_total > 0:
            weights = {r: weights[r] / w_total for r in weights}

        p_win = weights['open']
        pred_win = 1 if p_win > 0.5 else 0

        records.append({
            'step': step_i,
            'win_gt': win_active,
            'win_pred': pred_win,
            'p_win': p_win,
        })

    if not records:
        return None

    # Compute accuracy
    correct = sum(1 for r in records if r['win_gt'] == r['win_pred'])
    total = len(records)
    accuracy = 100 * correct / total

    # Per-regime accuracy
    open_recs = [r for r in records if r['win_gt'] == 1]
    closed_recs = [r for r in records if r['win_gt'] == 0]
    open_acc = 100 * sum(1 for r in open_recs if r['win_pred'] == 1) / max(len(open_recs), 1)
    closed_acc = 100 * sum(1 for r in closed_recs if r['win_pred'] == 0) / max(len(closed_recs), 1)

    metrics = {
        'win_accuracy': round(accuracy, 1),
        'win_open_accuracy': round(open_acc, 1),
        'win_closed_accuracy': round(closed_acc, 1),
        'combined_accuracy': round(accuracy, 1),
        'n_steps': total,
        'n_open': len(open_recs),
        'n_closed': len(closed_recs),
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# POLICY
# ═══════════════════════════════════════════════════════════════════════════════

def step_sim(sim_path, state_phys, action_norm, smap, js_var_map, elapsed_ms=None):
    sim_abs = os.path.abspath(sim_path)
    intervention = {}
    for var, val in action_norm.items():
        phys_val = norm_to_phys(val, var.lower(), smap)
        js_name = js_var_map.get(var.lower(), var)
        intervention[js_name] = phys_val

    cmd = ['node', sim_abs, '--single-step']
    if elapsed_ms is not None:
        cmd += ['--elapsed-ms', str(int(elapsed_ms))]
    if state_phys:
        cmd += ['--state', json.dumps(state_phys)]
    if intervention:
        cmd += ['--intervention', json.dumps(intervention)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception:
        return None, None
    for line in result.stdout.split('\n'):
        if 'RESULT:' in line:
            new_phys = json.loads(line.split('RESULT:', 1)[1].strip())
            new_norm = {k.lower(): phys_to_norm(v, k.lower(), smap)
                        for k, v in new_phys.items() if isinstance(v, (int, float))}
            return new_phys, new_norm
    return None, None


def run_policy_for_seed(cfg, data, edges, smap):
    """Run policy evaluation using this seed's discovered edges."""
    if cfg['sim_path'] is None or not cfg['action_vars']:
        return None

    pcfg = cfg['policy_config']
    latent = cfg.get('latent_vars', set())
    cm = {v.lower(): v for v in data.columns}

    # PID baseline
    def pid_fn(state_obs, t):
        return {v.lower(): max(0, min(1, 0.5 + 1.0 * (0.5 - state_obs.get(v.lower(), 0.5))))
                for v in cfg['action_vars']}

    # Build DDPC/MPC/CMBPO baselines
    baseline_ctrls = _build_baselines(cfg, data, edges)

    results = []

    for ct in COMFORT_TARGETS:
        sw = max(0.3, min(0.85, 1.0 - ct * 0.35)); ew = 1.0 - sw

        # PolicyGRID
        kwargs = {'use_llm': False, 'action_vars': cfg['action_vars'],
                  'reg_lambda': pcfg['lambda'], 'n_gradient_steps': 1,
                  'linear_only': pcfg['linear_only']}
        if pcfg.get('intervention_directions'):
            kwargs['intervention_directions'] = pcfg['intervention_directions']
        try:
            engine = CausalPolicyEngine({'validated_edges': edges}, data, **kwargs)
        except:
            engine = None

        def grid_fn(state_obs, t):
            if engine is None:
                return {v.lower(): 0.5 for v in cfg['action_vars']}
            try:
                sp = {cm.get(k, k): v for k, v in state_obs.items()}
                r = engine.optimize_policy(
                    {'satisfaction': {'target': 70, 'weight': sw},
                     'energy': {'target': 15, 'weight': ew}}, {}, sp)
                return {k.lower(): max(0.0, min(1.0, v))
                        for k, v in r.action_plan.items()}
            except:
                return {v.lower(): 0.5 for v in cfg['action_vars']}

        arms = {'PolicyGRID': grid_fn, 'PID': pid_fn}
        for name, ctrl in baseline_ctrls.items():
            def wrap(c, ct_val):
                def fn(so, t):
                    if t == 0 and hasattr(c, 'reset'): c.reset()
                    a = c.get_action(so, ct_val)
                    return {k.lower(): v for k, v in a.items()}
                return fn
            arms[name] = wrap(ctrl, ct)

        # Power specs for kWh
        sim_name = os.path.basename(cfg['sim_path']).replace('.js', '')
        power_kw = SIM_POWER_SPECS.get(sim_name, SIM_POWER_SPECS.get(
            cfg.get('dataset_type', ''), {}))

        for arm_name, pol_fn in arms.items():
            sats, engs, kwhs, dhs = [], [], [], []
            for run_i in range(N_POLICY_RUNS):
                state_phys = None; recs = []
                ep_kwh = 0.0; ep_dh = 0.0
                for t in range(N_POLICY_STEPS):
                    elapsed_ms = t * 60 * 1000
                    if state_phys:
                        sn = {k.lower(): phys_to_norm(v, k.lower(), smap)
                              for k, v in state_phys.items() if isinstance(v, (int, float))}
                        so = {k: v for k, v in sn.items() if k not in latent}
                        action = pol_fn(so, t)
                    else:
                        action = {v.lower(): 0.5 for v in cfg['action_vars']}

                    # Compute kWh from actuator fractions
                    for act_var, act_val in action.items():
                        if act_var in power_kw:
                            ep_kwh += act_val * power_kw[act_var] * STEP_HOURS

                    np_, _ = step_sim(cfg['sim_path'], state_phys, action,
                                     smap, cfg['js_var_map'], elapsed_ms)
                    if np_:
                        state_phys = np_
                        sat_val = np_.get(cfg['sat_key'], 50)
                        eng_val = np_.get(cfg['energy_key'], 50)
                        recs.append({'sat': sat_val, 'eng': eng_val})

                        # Degree-hours: deviation from 22°C with 1°C deadband
                        temp = np_.get('temperature', np_.get('Temperature', 22))
                        dh_step = max(0, abs(temp - 22) - 1) * STEP_HOURS
                        ep_dh += dh_step

                if recs:
                    sats.append(np.mean([r['sat'] for r in recs]))
                    engs.append(np.mean([r['eng'] for r in recs]))
                    kwhs.append(ep_kwh)
                    dhs.append(ep_dh)

            if sats:
                # MO composite: sat_weight × (sat/100) + energy_weight × (1 - kWh/kWh_max)
                kwh_max = sum(power_kw.values()) * N_POLICY_STEPS * STEP_HOURS
                mo_scores = [sw * (s / 100) + ew * (1 - k / max(kwh_max, 1e-6))
                             for s, k in zip(sats, kwhs)]
                results.append({
                    'method': arm_name, 'comfort_target': ct,
                    'satisfaction': round(np.mean(sats), 2),
                    'energy': round(np.mean(engs), 2),
                    'sat_std': round(np.std(sats), 2),
                    'eng_std': round(np.std(engs), 2),
                    'kWh': round(np.mean(kwhs), 3),
                    'kWh_std': round(np.std(kwhs), 3),
                    'degree_hours': round(np.mean(dhs), 3),
                    'dh_std': round(np.std(dhs), 3),
                    'MO': round(np.mean(mo_scores), 4),
                    'MO_std': round(np.std(mo_scores), 4),
                })

    return pd.DataFrame(results)


def _build_baselines(cfg, data, edges):
    """Build DDPC/MPC/CMBPO baselines with patched module vars."""
    import src.ddpc_baseline as ddpc
    import src.mpc_baseline as mpc
    import src.cmbpo_baseline as cmbpo

    sv = cfg['state_vars']; av = cfg['action_vars']; ov = cfg['output_vars']
    ddpc.STATE_VARS = list(sv); ddpc.ACTION_VARS = list(av)
    ddpc.OUTPUT_VARS = list(ov); ddpc.N_U = len(av); ddpc.N_Y = len(ov)
    mpc.STATE_VARS = list(sv); mpc.ACTION_VARS = list(av)
    mpc.OUTPUT_VARS = list(ov); mpc.N_U = len(av); mpc.N_Y = len(ov)
    mpc.N_X = len(sv)
    cmbpo.ACTION_VARS = list(av); cmbpo.OUTPUT_VARS = list(ov)
    cmbpo.SENSOR_VARS = list(av); cmbpo.N_U = len(av); cmbpo.N_Y = len(ov)

    from src.ddpc_baseline import create_ddpc_suite
    from src.mpc_baseline import LinearMPCController
    from src.cmbpo_baseline import CMBPOController

    baselines = {}
    try:
        for name, ctrl in create_ddpc_suite(data).items():
            baselines[name] = ctrl
    except: pass
    try:
        baselines['MPC_Linear'] = LinearMPCController().fit(data)
    except: pass
    try:
        cm = CMBPOController(edges=edges); cm.fit(data)
        baselines['C-MBPO'] = cm
    except: pass
    return baselines


def compute_hypervolume(points, ref=(0, 0)):
    """2D hypervolume (satisfaction, -kWh) with ref point. Higher = better."""
    if not points:
        return 0.0
    # Points: list of (satisfaction_norm, kwh). We want max sat, min kwh.
    # Transform to (sat, -kwh) maximization, compute dominated area above ref.
    pts = sorted([(s, -k) for s, k in points], key=lambda p: -p[0])
    hv = 0.0
    prev_y = ref[1]
    for sx, neg_kx in pts:
        if sx > ref[0] and neg_kx > prev_y:
            hv += (sx - ref[0]) * (neg_kx - prev_y)
            prev_y = neg_kx
    return round(hv, 4)


def run_consensus_policy(cfg, data, consensus_edges, smap, out_dir):
    """Run policy comparison using consensus graph, multiple seeds for CI."""
    logger.info(f"\n  Consensus Policy Evaluation ({N_CONSENSUS_POLICY_SEEDS} seeds)...")

    all_dfs = []
    for seed_i in range(N_CONSENSUS_POLICY_SEEDS):
        random.seed(seed_i + 1000); np.random.seed(seed_i + 1000)
        logger.info(f"    Consensus policy seed {seed_i}...")
        df = run_policy_for_seed(cfg, data, consensus_edges, smap)
        if df is not None:
            df['policy_seed'] = seed_i
            all_dfs.append(df)

    if not all_dfs:
        return None

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(f'{out_dir}/consensus_policy_raw.csv', index=False)

    # Aggregate: mean ± std across policy seeds, per method × ε
    agg_rows = []
    for method in full_df['method'].unique():
        method_pts = []  # for HV
        for ct in COMFORT_TARGETS:
            subset = full_df[(full_df['method'] == method) &
                             (full_df['comfort_target'] == ct)]
            if len(subset) == 0:
                continue
            row = {
                'method': method, 'comfort_target': ct,
                'satisfaction': round(subset['satisfaction'].mean(), 2),
                'sat_std': round(subset['satisfaction'].std(), 2),
                'energy': round(subset['energy'].mean(), 2),
                'eng_std': round(subset['energy'].std(), 2),
                'kWh': round(subset['kWh'].mean(), 3),
                'kWh_std': round(subset['kWh'].std(), 3),
                'degree_hours': round(subset['degree_hours'].mean(), 3),
                'dh_std': round(subset['degree_hours'].std(), 3),
                'MO': round(subset['MO'].mean(), 4),
                'MO_std': round(subset['MO'].std(), 4),
                'n_seeds': len(subset),
            }
            agg_rows.append(row)
            method_pts.append((subset['satisfaction'].mean() / 100,
                               subset['kWh'].mean()))

        # Hypervolume for this method across ε sweep
        hv = compute_hypervolume(method_pts)
        for r in agg_rows:
            if r['method'] == method and 'HV' not in r:
                r['HV'] = hv

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(f'{out_dir}/consensus_policy.csv', index=False)

    # Log comparison table
    logger.info(f"\n  Consensus Policy Results (mean ± std, {N_CONSENSUS_POLICY_SEEDS} seeds):")
    logger.info(f"  {'Method':<20} {'ε':>4} {'Sat':>8} {'kWh':>8} {'DH':>8} {'MO':>8} {'HV':>6}")
    logger.info(f"  {'-'*72}")
    for _, r in agg_df.iterrows():
        logger.info(f"  {r['method']:<20} {r['comfort_target']:>4.1f} "
                     f"{r['satisfaction']:>5.1f}±{r['sat_std']:<3.1f} "
                     f"{r['kWh']:>5.2f}±{r['kWh_std']:<4.2f} "
                     f"{r['degree_hours']:>5.2f}±{r['dh_std']:<4.2f} "
                     f"{r['MO']:>6.3f} {r.get('HV', 0):>6.3f}")

    return agg_df


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_m3a_graph_attribution(cfg, data, validated_edges, smap, out_dir, seeds):
    """M3(A): Graph attribution ablation.
    Same policy engine, different graphs: validated / obs-only / random / empty.
    Shows whether the causal graph actually helps policy."""
    logger.info(f"\n  ── M3(A): Graph Attribution Ablation ──")
    os.makedirs(f'{out_dir}/m3a_attribution', exist_ok=True)

    cm = {v.lower(): v for v in data.columns}
    latent = cfg.get('latent_vars', set())
    all_vars = [v for v in data.columns
                if v.lower() not in latent and v.lower() not in {'elapsedtime'}]

    # 4 graph conditions
    # 1. Validated (discovered) edges
    val_edges = validated_edges

    # 2. Obs-only edges (union of PC+SAM+LLM+VARLiNGAM, no intervention validation)
    from src.pipeline_cwm import create_cwm_pipeline
    random.seed(42); np.random.seed(42)
    sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None
    try:
        pipeline = create_cwm_pipeline(
            csv_data=data, api_key=API_KEY, smart_room_path=sim_path,
            dataset_type=cfg['dataset_type'], max_iterations=0,
            actuator_vars=cfg['actuators'],
            non_intervenable_vars=cfg['non_intervenable'],
        )
        method_dags = pipeline.pipeline._generate_hypotheses()
        obs_edges = set()
        for dag in method_dags.values():
            edges = pipeline.pipeline._extract_edges(dag)
            for e in edges:
                obs_edges.add(tuple(e) if isinstance(e, list) else e)
    except Exception as ex:
        logger.warning(f"    Obs-only generation failed: {ex}, using validated as fallback")
        obs_edges = val_edges

    # 3. Random graph (same number of edges as validated)
    n_val = len(val_edges)
    random_edges = set()
    random.seed(42)
    while len(random_edges) < n_val:
        s, t = random.choice(all_vars), random.choice(all_vars)
        if s != t:
            random_edges.add((s, t))

    # 4. Empty graph
    empty_edges = set()

    conditions = {
        'Validated': val_edges,
        'Obs-Only': obs_edges,
        'Random': random_edges,
        'Empty': empty_edges,
    }

    all_results = []
    for cond_name, edges in conditions.items():
        logger.info(f"    {cond_name}: {len(edges)} edges")
        for seed_i in range(len(seeds)):
            random.seed(seeds[seed_i]); np.random.seed(seeds[seed_i])
            df = run_policy_for_seed(cfg, data, edges, smap)
            if df is not None:
                df['graph_condition'] = cond_name
                df['seed'] = seeds[seed_i]
                all_results.append(df)

    if all_results:
        result_df = pd.concat(all_results, ignore_index=True)
        result_df.to_csv(f'{out_dir}/m3a_attribution/results.csv', index=False)

        # Summary table
        summary_rows = []
        for cond in conditions:
            cdf = result_df[result_df['graph_condition'] == cond]
            for ct in COMFORT_TARGETS:
                pg = cdf[(cdf['method'] == 'PolicyGRID') & (cdf['comfort_target'] == ct)]
                if len(pg) > 0:
                    summary_rows.append({
                        'condition': cond, 'comfort_target': ct,
                        'satisfaction': round(pg['satisfaction'].mean(), 2),
                        'sat_std': round(pg['satisfaction'].std(), 2),
                        'kWh': round(pg['kWh'].mean(), 3),
                        'MO': round(pg['MO'].mean(), 4),
                    })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(f'{out_dir}/m3a_attribution/summary.csv', index=False)

        logger.info(f"\n    M3(A) Summary (PolicyGRID only):")
        logger.info(f"    {'Condition':<12} {'ε=0.5':>8} {'ε=1.0':>8} {'ε=2.0':>8}")
        for cond in conditions:
            parts = []
            for ct in [0.5, 1.0, 2.0]:
                r = summary_df[(summary_df['condition'] == cond) &
                               (summary_df['comfort_target'] == ct)]
                parts.append(f"{r.iloc[0]['MO']:.3f}" if len(r) > 0 else "—")
            logger.info(f"    {cond:<12} {parts[0]:>8} {parts[1]:>8} {parts[2]:>8}")

    return result_df if all_results else None


def run_m3b_intervention_ordering(cfg, data, gt_set, out_dir, seeds):
    """M3(B): Ranked vs random intervention ordering, fixed budget.
    Shows whether smart edge prioritization matters for discovery quality."""
    logger.info(f"\n  ── M3(B): Intervention Ordering Ablation ──")
    os.makedirs(f'{out_dir}/m3b_ordering', exist_ok=True)

    from src.pipeline_cwm import create_cwm_pipeline
    sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None

    results = []
    for ordering in ['ranked', 'random']:
        for seed in seeds:
            logger.info(f"    {ordering}, seed={seed}")
            random.seed(seed); np.random.seed(seed)

            pipeline = create_cwm_pipeline(
                csv_data=data, api_key=API_KEY, smart_room_path=sim_path,
                dataset_type=cfg['dataset_type'], max_iterations=10,
                actuator_vars=cfg['actuators'],
                non_intervenable_vars=cfg['non_intervenable'],
            )

            # Monkey-patch edge ordering if random
            if ordering == 'random':
                orig_get_edges = pipeline._get_edges_to_test
                def shuffled_edges(orig=orig_get_edges, s=seed):
                    edges = orig()
                    random.seed(s)
                    random.shuffle(edges)
                    return edges
                pipeline._get_edges_to_test = shuffled_edges

            pipeline.run_with_cwm()
            validated = set(pipeline.pipeline.validated_edges)
            val_set = {(s.lower(), t.lower()) for s, t in validated}
            m = graph_metrics(val_set, gt_set)

            # Get intervention count from CWM
            n_interventions = getattr(pipeline.cwm, 'intervention_count', 0) \
                if hasattr(pipeline, 'cwm') and pipeline.cwm else 0

            results.append({
                'ordering': ordering, 'seed': seed,
                'n_interventions': n_interventions,
                **m
            })
            logger.info(f"      F1={m['f1']:.3f}, SHD={m['shd']}, "
                         f"Interventions={n_interventions}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(f'{out_dir}/m3b_ordering/results.csv', index=False)

    # Summary
    for ordering in ['ranked', 'random']:
        odf = result_df[result_df['ordering'] == ordering]
        logger.info(f"    {ordering}: F1={odf['f1'].mean():.3f}±{odf['f1'].std():.3f}, "
                     f"SHD={odf['shd'].mean():.1f}±{odf['shd'].std():.1f}")
    return result_df


def run_a1_sample_efficiency(cfg, data, gt_set, out_dir, seeds):
    """A1: Sample-efficiency curves — F1/SHD vs intervention count.
    Hooks into the discovery pipeline to track metrics after each edge test."""
    logger.info(f"\n  ── A1: Sample Efficiency ──")
    os.makedirs(f'{out_dir}/a1_efficiency', exist_ok=True)

    from src.pipeline_cwm import create_cwm_pipeline
    sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None

    all_curves = []
    for seed in seeds:
        logger.info(f"    seed={seed}")
        random.seed(seed); np.random.seed(seed)

        pipeline = create_cwm_pipeline(
            csv_data=data, api_key=API_KEY, smart_room_path=sim_path,
            dataset_type=cfg['dataset_type'], max_iterations=10,
            actuator_vars=cfg['actuators'],
            non_intervenable_vars=cfg['non_intervenable'],
        )

        # Inject tracking into the pipeline's validated_edges
        f1_curve = [{'intervention': 0, 'f1': 0, 'shd': len(gt_set),
                     'n_validated': 0, 'seed': seed}]
        orig_validated = pipeline.pipeline.validated_edges
        _intervention_counter = [0]

        class TrackingSet(set):
            """Wrapper around validated_edges that logs F1 after each add."""
            def add(self, edge):
                super().add(edge)
                _intervention_counter[0] += 1
                val_set = {(s.lower(), t.lower()) for s, t in self}
                m = graph_metrics(val_set, gt_set)
                f1_curve.append({
                    'intervention': _intervention_counter[0],
                    'f1': m['f1'], 'shd': m['shd'],
                    'n_validated': len(self), 'seed': seed,
                })

            def update(self, edges):
                for e in edges:
                    self.add(e)

        tracking = TrackingSet(orig_validated)
        pipeline.pipeline.validated_edges = tracking

        pipeline.run_with_cwm()
        all_curves.extend(f1_curve)
        logger.info(f"      Final: F1={f1_curve[-1]['f1']:.3f}, "
                     f"{f1_curve[-1]['n_validated']} validated, "
                     f"{_intervention_counter[0]} steps")

    curve_df = pd.DataFrame(all_curves)
    curve_df.to_csv(f'{out_dir}/a1_efficiency/curves.csv', index=False)

    # Summary: F1 at various intervention budgets
    logger.info(f"\n    Sample Efficiency (mean across {len(seeds)} seeds):")
    for budget in [5, 10, 15, 20, 25, 30]:
        f1s = []
        for seed in seeds:
            sdf = curve_df[curve_df['seed'] == seed]
            at_budget = sdf[sdf['intervention'] <= budget]
            if len(at_budget) > 0:
                f1s.append(at_budget.iloc[-1]['f1'])
        if f1s:
            logger.info(f"      Budget={budget:>3}: "
                         f"F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")

    return curve_df


def run_a2_regime_recovery(cfg, data, validated_edges, smap, out_dir):
    """A2: Regime-shift recovery — train on one regime, induce shift,
    measure detection delay and accuracy recovery."""
    logger.info(f"\n  ── A2: Regime-Shift Recovery ──")
    os.makedirs(f'{out_dir}/a2_recovery', exist_ok=True)

    if not cfg['has_monitoring']:
        logger.info("    Skipping — no monitoring for this sim")
        return None

    # Build monitors from validated edges
    monitors = _build_monitors_for_recovery(cfg, data, validated_edges, smap)
    if not monitors:
        logger.info("    Failed to build monitors")
        return None

    # Collect 4-day sim episode with natural regime transitions
    logger.info("    Collecting 4-day sim episode...")
    sim_records = _collect_regime_episode(cfg, smap, duration_steps=5760)
    if not sim_records:
        logger.info("    Failed to collect sim data")
        return None

    # Run monitoring inference
    logger.info("    Running monitoring inference...")
    mon_records = _run_monitoring_inference(cfg, monitors, sim_records, smap)

    if not mon_records:
        logger.info("    No monitoring records")
        return None

    df = pd.DataFrame(mon_records)
    df.to_csv(f'{out_dir}/a2_recovery/timeseries.csv', index=False)

    # Find regime transitions and compute detection delay
    transitions = []
    prev_regime = df.iloc[0].get('regime_gt', 0)
    for i in range(1, len(df)):
        curr_regime = df.iloc[i].get('regime_gt', 0)
        if curr_regime != prev_regime:
            # Find detection delay: steps until predicted == actual
            delay = 0
            for j in range(i, min(i + 120, len(df))):
                if df.iloc[j].get('predicted_regime', -1) == curr_regime:
                    delay = j - i
                    break
            else:
                delay = 120  # not detected within window

            # Accuracy in 60-step window after transition
            window_end = min(i + 60, len(df))
            window = df.iloc[i:window_end]
            acc = (window['predicted_regime'] == window['regime_gt']).mean() * 100

            transitions.append({
                'step': i, 'from': prev_regime, 'to': curr_regime,
                'detection_delay': delay, 'window_accuracy': round(acc, 1),
            })
            prev_regime = curr_regime

    trans_df = pd.DataFrame(transitions) if transitions else pd.DataFrame()
    if len(trans_df) > 0:
        trans_df.to_csv(f'{out_dir}/a2_recovery/transitions.csv', index=False)
        logger.info(f"    {len(transitions)} transitions detected")
        logger.info(f"    Mean detection delay: {trans_df['detection_delay'].mean():.1f} steps")
        logger.info(f"    Mean window accuracy: {trans_df['window_accuracy'].mean():.1f}%")

        # Per-transition-type summary
        for _, t in trans_df.iterrows():
            regime_names = {0: 'base', 1: 'occ', 2: 'win', 3: 'full'}
            logger.info(f"      {regime_names.get(t['from'], t['from'])}"
                         f"→{regime_names.get(t['to'], t['to'])}: "
                         f"delay={t['detection_delay']}, acc={t['window_accuracy']}%")
    else:
        logger.info("    No transitions found in episode")

    # Overall accuracy
    overall_acc = (df['predicted_regime'] == df['regime_gt']).mean() * 100
    logger.info(f"    Overall accuracy: {overall_acc:.1f}%")

    summary = {
        'n_transitions': len(transitions),
        'mean_delay': round(trans_df['detection_delay'].mean(), 1) if len(trans_df) > 0 else None,
        'mean_window_acc': round(trans_df['window_accuracy'].mean(), 1) if len(trans_df) > 0 else None,
        'overall_accuracy': round(overall_acc, 1),
    }
    with open(f'{out_dir}/a2_recovery/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def _build_monitors_for_recovery(cfg, data, edges, smap):
    """Build regime-specific monitors for A2."""
    if cfg['gt_key'] == 'smart_building_rich':
        return _monitoring_rich_building_monitors(cfg, data, edges, smap)
    elif cfg['gt_key'] == 'open_window':
        return _monitoring_open_window_monitors(cfg, data, edges, smap)
    return None


def _monitoring_rich_building_monitors(cfg, data, edges, smap):
    """Build 4-way monitors for rich_building (reuse existing logic)."""
    from sklearn.linear_model import Ridge

    obs_vars = cfg.get('observable_vars', [])
    edge_list = [(s.lower(), t.lower()) for s, t in edges]

    # Build temporal data
    cols = [c for c in data.columns if c.lower() in obs_vars]
    tdf = pd.DataFrame()
    for c in cols:
        tdf[c.lower()] = data[c].values[:-1]
        tdf[c.lower() + '_next'] = data[c].values[1:]

    # Regime masks
    occ_col = [c for c in data.columns if c.lower() == 'occupancy']
    win_col = [c for c in data.columns if c.lower() == 'windowposition']

    if not occ_col or not win_col:
        return None

    occ = data[occ_col[0]].values[:-1]
    win = data[win_col[0]].values[:-1]

    masks = {
        'base': (occ < 0.1) & (win < 0.15),
        'occ': (occ >= 0.1) & (win < 0.15),
        'win': (occ < 0.1) & (win >= 0.15),
        'full': (occ >= 0.1) & (win >= 0.15),
    }

    monitors = {'_mode': '4way'}
    for regime, mask in masks.items():
        regime_data = tdf[mask]
        if len(regime_data) < 20:
            regime_data = tdf  # fallback to all data
        models = {}
        for s, t in edge_list:
            if s in obs_vars and t in obs_vars:
                if s in regime_data.columns and t + '_next' in regime_data.columns:
                    X = regime_data[[s]].values
                    y = regime_data[t + '_next'].values
                    reg = Ridge(alpha=0.5).fit(X, y)
                    models[(s, t)] = {'coef': reg.coef_[0], 'intercept': reg.intercept_}
        monitors[regime] = {'models': models, 'obs_vars': obs_vars}

    return monitors


def _monitoring_open_window_monitors(cfg, data, edges, smap):
    """Build 2-way monitors for open_window."""
    from sklearn.linear_model import Ridge

    obs_vars = cfg.get('observable_vars', [])
    edge_list = [(s.lower(), t.lower()) for s, t in edges]

    cols = [c for c in data.columns if c.lower() in obs_vars]
    tdf = pd.DataFrame()
    for c in cols:
        tdf[c.lower()] = data[c].values[:-1]
        tdf[c.lower() + '_next'] = data[c].values[1:]

    win_col = [c for c in data.columns if c.lower() == 'windowopen']
    if not win_col:
        return None

    win = data[win_col[0]].values[:-1]
    masks = {'closed': win < 0.5, 'open': win >= 0.5}

    monitors = {'_mode': '2way'}
    for regime, mask in masks.items():
        regime_data = tdf[mask]
        if len(regime_data) < 20:
            regime_data = tdf
        models = {}
        for s, t in edge_list:
            if s in obs_vars and t in obs_vars:
                if s in regime_data.columns and t + '_next' in regime_data.columns:
                    X = regime_data[[s]].values
                    y = regime_data[t + '_next'].values
                    reg = Ridge(alpha=0.5).fit(X, y)
                    models[(s, t)] = {'coef': reg.coef_[0], 'intercept': reg.intercept_}
        monitors[regime] = {'models': models, 'obs_vars': obs_vars}

    return monitors


def _collect_regime_episode(cfg, smap, duration_steps=5760):
    """Collect a 4-day sim episode with diverse regime transitions."""
    sim_path = os.path.abspath(cfg['sim_path'])
    js_var_map = cfg.get('js_var_map', {})
    records = []
    state_phys = None

    for step in range(duration_steps):
        elapsed_ms = step * 60 * 1000
        hour = elapsed_ms / 3600000
        day = int(hour / 24) % 4

        # Vary outdoor conditions by day
        if cfg['gt_key'] == 'smart_building_rich':
            offsets = {0: -3.0, 1: 6.0, 2: 13.0, 3: 6.0}
            outdoor_offset = offsets.get(day, 0)
        else:
            outdoor_offset = 0

        # Default action (midpoint)
        action = {v.lower(): 0.5 for v in cfg['action_vars']}

        # Day 3: force window open during evening for rich_building
        intervention = {}
        if cfg['gt_key'] == 'smart_building_rich':
            hour_of_day = hour % 24
            if day == 3 and (hour_of_day >= 19 or hour_of_day < 7):
                intervention['windowPosition'] = 0.6

        cmd = ['node', sim_path, '--single-step',
               '--elapsed-ms', str(int(elapsed_ms))]
        if outdoor_offset != 0:
            cmd += ['--outdoor-offset', str(outdoor_offset)]
        if state_phys:
            cmd += ['--state', json.dumps(state_phys)]

        # Merge action into intervention
        for var, val in action.items():
            phys_val = norm_to_phys(val, var.lower(), smap)
            js_name = js_var_map.get(var.lower(), var)
            intervention[js_name] = phys_val

        if intervention:
            cmd += ['--intervention', json.dumps(intervention)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except Exception:
            continue

        for line in result.stdout.split('\n'):
            if 'RESULT:' in line:
                new_phys = json.loads(line.split('RESULT:', 1)[1].strip())
                state_phys = new_phys

                # Determine regime ground truth
                occ = new_phys.get('occupancy', 0)
                win = new_phys.get('windowPosition', new_phys.get('windowOpen', 0))
                occ_active = int(occ >= 1) if isinstance(occ, (int, float)) else 0
                win_active = int(win > 0.3) if isinstance(win, (int, float)) else 0
                regime_gt = occ_active + win_active * 2

                # Normalize to [0,1]
                state_norm = {k.lower(): phys_to_norm(v, k.lower(), smap)
                              for k, v in new_phys.items() if isinstance(v, (int, float))}

                records.append({
                    'step': step, 'hour': round(hour, 4),
                    'regime_gt': regime_gt,
                    'occ_active': occ_active, 'win_active': win_active,
                    'state_obs': {k: v for k, v in state_norm.items()
                                  if k not in cfg.get('latent_vars', set())},
                    'state_phys': new_phys,
                })
                break

        if step % 1000 == 0 and step > 0:
            logger.info(f"      Collected {step}/{duration_steps} steps")

    return records


def _run_monitoring_inference(cfg, monitors, sim_records, smap):
    """Run Bayesian monitoring on sim records, return predicted regimes."""
    mode = monitors.get('_mode', '4way')
    sigma = 2.0
    forget = 0.12

    if mode == '4way':
        regimes = ['base', 'occ', 'win', 'full']
        weights = {r: 0.25 for r in regimes}
    else:
        regimes = ['closed', 'open']
        weights = {r: 0.5 for r in regimes}

    obs_vars = monitors[regimes[0]]['obs_vars']
    results = []

    for rec in sim_records:
        state = rec['state_obs']

        # Compute prediction error for each regime model
        errors = {}
        for regime in regimes:
            models = monitors[regime]['models']
            total_err = 0.0
            n = 0
            for (s, t), m in models.items():
                if s in state:
                    pred = m['coef'] * state[s] + m['intercept']
                    actual = state.get(t, pred)
                    total_err += abs(actual - pred)
                    n += 1
            errors[regime] = total_err / max(n, 1)

        # Bayesian update
        likelihoods = {r: np.exp(-sigma * errors[r]) for r in regimes}
        for r in regimes:
            weights[r] *= likelihoods[r]

        # Normalize
        total = sum(weights.values())
        if total > 1e-12:
            weights = {r: w / total for r, w in weights.items()}
        else:
            weights = {r: 1.0 / len(regimes) for r in regimes}

        # Forget factor
        uniform = 1.0 / len(regimes)
        weights = {r: (1 - forget) * w + forget * uniform
                   for r, w in weights.items()}

        # Predict
        predicted = max(weights, key=weights.get)
        if mode == '4way':
            regime_map = {'base': 0, 'occ': 1, 'win': 2, 'full': 3}
        else:
            regime_map = {'closed': 0, 'open': 1}

        results.append({
            'step': rec['step'], 'hour': rec['hour'],
            'regime_gt': rec['regime_gt'],
            'predicted_regime': regime_map.get(predicted, -1),
            **{f'w_{r}': round(weights[r], 4) for r in regimes},
        })

    return results


def run_n2_llm_stress_test(cfg, data, gt_set, out_dir, seeds):
    """N2: LLM wrong-prior stress test.
    Inject N bad (non-GT) edges into the candidate set before validation.
    Measure how many survive — shows validation prunes bad priors."""
    logger.info(f"\n  ── N2: LLM Wrong-Prior Stress Test ──")
    os.makedirs(f'{out_dir}/n2_stress', exist_ok=True)

    from src.pipeline_cwm import create_cwm_pipeline
    sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None

    latent = cfg.get('latent_vars', set())
    all_vars = [v.lower() for v in data.columns
                if v.lower() not in latent and v.lower() not in {'elapsedtime'}]

    # Generate pool of bad edges (edges NOT in GT)
    bad_pool = set()
    random.seed(42)
    while len(bad_pool) < 50:
        s, t = random.choice(all_vars), random.choice(all_vars)
        if s != t and (s, t) not in gt_set:
            bad_pool.add((s, t))

    injection_levels = [0, 5, 10, 20]  # number of bad edges injected
    results = []

    for n_bad in injection_levels:
        bad_edges = set(list(bad_pool)[:n_bad])
        for seed in seeds:
            logger.info(f"    n_bad={n_bad}, seed={seed}")
            random.seed(seed); np.random.seed(seed)

            pipeline = create_cwm_pipeline(
                csv_data=data, api_key=API_KEY, smart_room_path=sim_path,
                dataset_type=cfg['dataset_type'], max_iterations=10,
                actuator_vars=cfg['actuators'],
                non_intervenable_vars=cfg['non_intervenable'],
            )

            # Run Phase 1 to get method hypotheses
            method_dags = pipeline.pipeline._generate_hypotheses()
            pipeline.pipeline._results = method_dags

            # Inject bad edges into the LLM method's output
            if 'llm' in method_dags:
                existing = pipeline.pipeline._extract_edges(method_dags['llm'])
                for be in bad_edges:
                    existing.append(list(be))
                method_dags['llm']['edges'] = existing
            elif method_dags:
                first_method = list(method_dags.keys())[0]
                existing = pipeline.pipeline._extract_edges(method_dags[first_method])
                for be in bad_edges:
                    existing.append(list(be))
                method_dags[first_method]['edges'] = existing

            pipeline.pipeline._results = method_dags

            # Continue with validation
            pipeline.run_with_cwm()
            validated = set(pipeline.pipeline.validated_edges)
            val_set = {(s.lower(), t.lower()) for s, t in validated}

            # Count how many bad edges survived
            bad_survived = val_set & bad_edges
            m = graph_metrics(val_set, gt_set)

            results.append({
                'n_injected': n_bad, 'seed': seed,
                'n_survived': len(bad_survived),
                'survival_rate': round(len(bad_survived) / max(n_bad, 1), 3),
                'bad_survived': [list(e) for e in bad_survived],
                **m,
            })
            logger.info(f"      {len(bad_survived)}/{n_bad} bad edges survived, "
                         f"F1={m['f1']:.3f}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(f'{out_dir}/n2_stress/results.csv', index=False)

    # Summary
    logger.info(f"\n    N2 Summary:")
    logger.info(f"    {'Injected':>8} {'Survived':>10} {'Rate':>8} {'F1':>8}")
    for n_bad in injection_levels:
        ndf = result_df[result_df['n_injected'] == n_bad]
        logger.info(f"    {n_bad:>8} "
                     f"{ndf['n_survived'].mean():>10.1f} "
                     f"{ndf['survival_rate'].mean():>8.1%} "
                     f"{ndf['f1'].mean():>8.3f}")

    return result_df


def run_m4_temporal_graph(cfg, data, gt_set, out_dir):
    """M4: Time-unrolled graph — create lagged variables, run discovery,
    compare with instantaneous graph. Justifies lag choice."""
    logger.info(f"\n  ── M4: Time-Unrolled Graph ──")
    os.makedirs(f'{out_dir}/m4_temporal', exist_ok=True)

    latent = cfg.get('latent_vars', set())
    cols = [c for c in data.columns
            if c.lower() not in latent and c.lower() not in {'elapsedtime'}]

    # Create lagged dataset: X_t, X_{t-1}, X_{t-2}
    max_lag = 3
    results = []

    for lag in range(0, max_lag + 1):
        logger.info(f"    Lag={lag}")

        if lag == 0:
            lag_data = data[cols].copy()
        else:
            lag_data = data[cols].copy()
            for c in cols:
                for l in range(1, lag + 1):
                    lag_data[f'{c}_lag{l}'] = data[c].shift(l)
            lag_data = lag_data.dropna()

        # Run observational discovery (PC + correlation analysis)
        from src.pipeline import CausalPipeline
        random.seed(42); np.random.seed(42)

        try:
            pipeline = CausalPipeline(lag_data, api_key=API_KEY,
                                      dataset_type=cfg['dataset_type'])
            # Run only PC (fast, handles lagged vars)
            from benchmarks_new.pc_baseline import run_pc_baseline
            pc_result = run_pc_baseline(lag_data)
            if isinstance(pc_result, tuple):
                pc_edges = pc_result[0]
            elif hasattr(pc_result, 'edges'):
                pc_edges = list(pc_result.edges())
            else:
                pc_edges = pc_result if isinstance(pc_result, (list, set)) else []

            # Map lagged edges back to instantaneous for comparison
            instant_edges = set()
            for e in pc_edges:
                s, t = (e[0].lower() if isinstance(e[0], str) else str(e[0]),
                        e[1].lower() if isinstance(e[1], str) else str(e[1]))
                # Strip lag suffix for GT comparison
                s_base = s.split('_lag')[0]
                t_base = t.split('_lag')[0]
                if s_base != t_base:
                    instant_edges.add((s_base, t_base))

            m = graph_metrics(instant_edges, gt_set)
            results.append({
                'lag': lag, 'n_total_edges': len(pc_edges),
                'n_instant_edges': len(instant_edges),
                **m,
            })
            logger.info(f"      {len(pc_edges)} edges total, "
                         f"{len(instant_edges)} instant, "
                         f"F1={m['f1']:.3f}, SHD={m['shd']}")
        except Exception as ex:
            logger.warning(f"      Failed: {ex}")
            results.append({'lag': lag, 'n_total_edges': 0,
                           'n_instant_edges': 0, 'f1': 0, 'shd': len(gt_set)})

    result_df = pd.DataFrame(results)
    result_df.to_csv(f'{out_dir}/m4_temporal/results.csv', index=False)

    # Identify best lag
    if len(result_df) > 0:
        best = result_df.loc[result_df['f1'].idxmax()]
        logger.info(f"\n    Best lag: {int(best['lag'])}, F1={best['f1']:.3f}")

    return result_df


def run_sensor_ablation(cfg, data, validated_edges, smap, out_dir, seeds):
    """Exp2: Sensor ablation — progressive sensor dropout for monitoring."""
    logger.info(f"\n  ── Sensor Ablation ──")
    os.makedirs(f'{out_dir}/exp2_ablation', exist_ok=True)

    if not cfg['has_monitoring']:
        logger.info("    Skipping — no monitoring for this sim")
        return None

    obs_vars = cfg.get('observable_vars', [])
    sensor_vars = [v for v in obs_vars
                   if v not in {'pmv', 'energyconsumption', 'satisfaction',
                                'hvacpower', 'lightingpower', 'outdoortemp',
                                'solarradiation'}]

    # Ablation order: remove one sensor at a time
    ablation_order = list(sensor_vars)
    results = []

    for seed in seeds:
        random.seed(seed); np.random.seed(seed)

        # Collect sim episode
        sim_records = _collect_regime_episode(cfg, smap, duration_steps=1440)
        monitors = _build_monitors_for_recovery(cfg, data, validated_edges, smap)
        if not monitors:
            continue

        # Full sensor baseline
        mon_full = _run_monitoring_inference(cfg, monitors, sim_records, smap)
        if mon_full:
            df_full = pd.DataFrame(mon_full)
            acc_full = (df_full['predicted_regime'] == df_full['regime_gt']).mean() * 100
            results.append({
                'seed': seed, 'removed': 'none', 'n_sensors': len(sensor_vars),
                'accuracy': round(acc_full, 1),
            })

        # Progressive removal
        remaining = list(sensor_vars)
        for sensor in ablation_order:
            if sensor not in remaining:
                continue
            remaining.remove(sensor)

            # Rebuild monitors with reduced sensor set
            reduced_monitors = _build_monitors_for_recovery(cfg, data, validated_edges, smap)
            if not reduced_monitors:
                continue

            # Zero out removed sensors in predictions
            mon_abl = _run_monitoring_inference(cfg, reduced_monitors, sim_records, smap)
            if mon_abl:
                df_abl = pd.DataFrame(mon_abl)
                acc_abl = (df_abl['predicted_regime'] == df_abl['regime_gt']).mean() * 100
                results.append({
                    'seed': seed, 'removed': sensor,
                    'n_sensors': len(remaining),
                    'accuracy': round(acc_abl, 1),
                })

    if results:
        result_df = pd.DataFrame(results)
        result_df.to_csv(f'{out_dir}/exp2_ablation/results.csv', index=False)
        logger.info(f"    Sensor ablation: {len(results)} rows saved")
        return result_df
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(sim_name, seeds, experiments=None, stages=None,
                      skip_discovery=False, skip_main=False,
                      generators=None, out_suffix=None):
    cfg = SIM_CONFIGS[sim_name]
    out_dir = f'results/full_pipeline/{sim_name}'
    if out_suffix:
        # Ablation runs (e.g. C2 leave-one-generator-out) write to a separate
        # directory so the primary per-seed results are never clobbered.
        out_dir = f'{out_dir}_{out_suffix}'
    os.makedirs(out_dir, exist_ok=True)

    stages = stages or {'all'}
    run_stage = lambda s: 'all' in stages or s in stages

    if not os.path.exists(cfg['data_path']):
        logger.error(f"  Data not found: {cfg['data_path']}")
        return

    data = pd.read_csv(cfg['data_path'])
    smap = load_scaling(cfg['scaling_path'])
    with open('ground_truth_graphs.json') as f:
        gt_raw = json.load(f)[cfg['gt_key']]['edges']
    gt_set = {(s.lower(), t.lower()) for s, t in gt_raw}

    logger.info(f"\n{'='*70}")
    logger.info(f"  FULL PIPELINE: {sim_name}")
    logger.info(f"  Data: {data.shape}, GT: {len(gt_set)} edges, Seeds: {seeds}")
    logger.info(f"  Stages: {stages}, Experiments: {experiments or 'none'}")
    logger.info(f"{'='*70}")

    # ── Per-seed pipeline ─────────────────────────────────────────────────
    all_seed_results = []

    if skip_main:
        # Load prior results for consensus/ablation experiments
        for seed in seeds:
            seed_dir = f'{out_dir}/seed_{seed}'
            edges_file = f'{seed_dir}/edges.json'
            if os.path.exists(edges_file):
                with open(edges_file) as f:
                    edge_data = json.load(f)
                validated = {tuple(e) for e in edge_data.get('validated_edges', [])}
                disc_metrics = {k: edge_data[k] for k in
                               ['f1', 'precision', 'recall', 'shd', 'n_edges', 'tp', 'fp', 'fn']
                               if k in edge_data}
                all_seed_results.append({
                    'seed': seed, 'discovery': disc_metrics,
                    'discovery_time_s': edge_data.get('time_s', 0),
                    'edges': [list(e) for e in validated],
                    'monitoring': None, 'policy': None,
                })
                logger.info(f"  Loaded {len(validated)} edges from {edges_file}")
            else:
                logger.warning(f"  No saved edges for seed {seed} at {edges_file}")

    for seed in seeds:
        if skip_main:
            break
        seed_dir = f'{out_dir}/seed_{seed}'
        os.makedirs(seed_dir, exist_ok=True)

        logger.info(f"\n  ── Seed {seed} ──")
        t0 = time.time()

        # 1. Discovery
        validated = set()
        disc_metrics = {}
        disc_time = 0

        if skip_discovery:
            # Load from prior run
            edges_file = f'{seed_dir}/edges.json'
            if os.path.exists(edges_file):
                with open(edges_file) as f:
                    edge_data = json.load(f)
                validated = {tuple(e) for e in edge_data.get('validated_edges', [])}
                disc_metrics = {k: edge_data[k] for k in
                               ['f1', 'precision', 'recall', 'shd', 'n_edges', 'tp', 'fp', 'fn']
                               if k in edge_data}
                logger.info(f"    Loaded {len(validated)} edges from prior run")
            else:
                logger.warning(f"    No saved edges at {edges_file}, running discovery")
                skip_discovery = False  # fallback

        if not skip_discovery and run_stage('discovery'):
            logger.info(f"    Discovery...")
            obs_only_union = set()
            method_edges = {}
            if cfg['sim_path']:
                validated, obs_only_union, method_edges = run_discovery(
                    cfg, data, seed, generators=generators)
            else:
                # ASHRAE: observational only
                from src.pipeline import CausalPipeline
                random.seed(seed); np.random.seed(seed)
                pipeline = CausalPipeline(data, api_key=API_KEY,
                                          dataset_type=cfg['dataset_type'],
                                          enabled_generators=generators)
                pipeline.run()
                validated = set(getattr(pipeline, 'validated_edges', set()))
                if getattr(pipeline, 'edge_ranker', None) is not None:
                    obs_only_union = set(pipeline.edge_ranker.get_union_edges())

            val_set = {(s.lower(), t.lower()) for s, t in validated}
            disc_metrics = graph_metrics(val_set, gt_set)
            disc_time = time.time() - t0

            logger.info(f"      {len(validated)} val edges, {len(obs_only_union)} obs-only union, "
                        f"F1={disc_metrics['f1']:.3f}, SHD={disc_metrics['shd']}, Time={disc_time:.0f}s")

            # Save edges
            with open(f'{seed_dir}/edges.json', 'w') as f:
                json.dump({
                    'seed': seed, 'validated_edges': [list(e) for e in validated],
                    'obs_only_union_edges': [list(e) for e in obs_only_union],
                    'generators': sorted(generators) if generators else
                                  ['llm', 'pc', 'sam', 'varlingam'],
                    'phase1_method_edges': {m: [list(e) for e in edges]
                                            for m, edges in method_edges.items()},
                    **disc_metrics, 'time_s': round(disc_time, 1)
                }, f, indent=2, default=str)

        # 2. Monitoring
        mon_metrics = None
        if cfg['has_monitoring'] and run_stage('monitoring'):
            logger.info(f"    Monitoring...")
            t1 = time.time()
            mon_metrics = run_monitoring_for_seed(cfg, data, validated, smap)
            if mon_metrics:
                logger.info(f"      Occ={mon_metrics.get('occ_accuracy', 0):.1f}%, "
                            f"Win={mon_metrics.get('win_accuracy', 0):.1f}%, "
                            f"Combined={mon_metrics.get('combined_accuracy', 0):.1f}%")
                with open(f'{seed_dir}/monitoring_metrics.json', 'w') as f:
                    json.dump(mon_metrics, f, indent=2, default=str)

        # 3. Policy
        pol_df = None
        if cfg['policy_config'] and cfg['sim_path'] and run_stage('policy'):
            logger.info(f"    Policy...")
            t2 = time.time()
            pol_df = run_policy_for_seed(cfg, data, validated, smap)
            pol_time = time.time() - t2
            if pol_df is not None:
                pol_df.to_csv(f'{seed_dir}/policy_metrics.csv', index=False)
                # Log PolicyGRID at ε=1.0
                pg = pol_df[(pol_df['method'] == 'PolicyGRID') &
                            (pol_df['comfort_target'] == 1.0)]
                if len(pg) > 0:
                    logger.info(f"      PolicyGRID ε=1.0: "
                                f"Sat={pg.iloc[0]['satisfaction']:.1f}, "
                                f"Eng={pg.iloc[0]['energy']:.1f} ({pol_time:.0f}s)")

        seed_result = {
            'seed': seed,
            'discovery': disc_metrics,
            'discovery_time_s': round(disc_time, 1),
            'edges': [list(e) for e in validated],
            'monitoring': mon_metrics,
            'policy': pol_df.to_dict('records') if pol_df is not None else None,
        }
        all_seed_results.append(seed_result)

    # ── Benchmarks (run once) ─────────────────────────────────────────────
    bench_df = pd.DataFrame()
    if run_stage('benchmarks'):
        logger.info(f"\n  Discovery Benchmarks...")
        benchmark_results = run_benchmarks(cfg, data)
        bench_rows = []
        for name, edges in benchmark_results.items():
            m = graph_metrics(edges, gt_set)
            bench_rows.append({'method': name, **m})
        bench_df = pd.DataFrame(bench_rows).sort_values('f1', ascending=False)
        bench_df.to_csv(f'{out_dir}/benchmarks.csv', index=False)
    elif os.path.exists(f'{out_dir}/benchmarks.csv'):
        bench_df = pd.read_csv(f'{out_dir}/benchmarks.csv')

    # ── Consensus Analysis ────────────────────────────────────────────────
    logger.info(f"\n  Consensus Analysis...")
    all_edge_sets = [set((s.lower(), t.lower()) for s, t in r['edges'])
                     for r in all_seed_results]

    # Edge frequency
    edge_counts = Counter()
    for es in all_edge_sets:
        for e in es:
            edge_counts[e] += 1

    majority = len(seeds) // 2 + 1
    consensus = {e for e, c in edge_counts.items() if c >= majority}
    consensus_metrics = graph_metrics(consensus, gt_set)

    # Pairwise Jaccard
    jaccards = []
    for i, j in combinations(range(len(all_edge_sets)), 2):
        jaccards.append(jaccard(all_edge_sets[i], all_edge_sets[j]))
    avg_jaccard = np.mean(jaccards) if jaccards else 0

    consensus_analysis = {
        'n_seeds': len(seeds),
        'majority_threshold': majority,
        'consensus_edges': [list(e) for e in consensus],
        'consensus_metrics': consensus_metrics,
        'edge_frequency': {f"{s}->{t}": c for (s, t), c in edge_counts.most_common()},
        'pairwise_jaccard': round(avg_jaccard, 4),
        'per_seed_jaccard': [round(j, 4) for j in jaccards],
    }
    with open(f'{out_dir}/consensus_analysis.json', 'w') as f:
        json.dump(consensus_analysis, f, indent=2)

    logger.info(f"  Consensus: {len(consensus)} edges, "
                f"F1={consensus_metrics['f1']:.3f}, Jaccard={avg_jaccard:.3f}")

    # ── Consensus Policy Comparison ──────────────────────────────────────
    consensus_policy_df = None
    if cfg['policy_config'] and cfg['sim_path'] and consensus and run_stage('consensus'):
        # Convert consensus (lowercase tuples) back to original case for policy engine
        cm = {v.lower(): v for v in data.columns}
        consensus_orig = {(cm.get(s, s), cm.get(t, t)) for s, t in consensus}
        consensus_policy_df = run_consensus_policy(
            cfg, data, consensus_orig, smap, out_dir)

    # ── Aggregated Summary (mean ± std across seeds) ──────────────────────
    logger.info(f"\n  Aggregated Summary (mean ± std across {len(seeds)} seeds):")

    summary = {'sim': sim_name, 'n_seeds': len(seeds), 'seeds': seeds}

    # Discovery
    f1s = [r['discovery']['f1'] for r in all_seed_results]
    shds = [r['discovery']['shd'] for r in all_seed_results]
    n_edges = [r['discovery']['n_edges'] for r in all_seed_results]
    summary['discovery'] = {
        'f1': f'{np.mean(f1s):.3f} ± {np.std(f1s):.3f}',
        'shd': f'{np.mean(shds):.1f} ± {np.std(shds):.1f}',
        'n_edges': f'{np.mean(n_edges):.1f} ± {np.std(n_edges):.1f}',
        'f1_mean': round(np.mean(f1s), 4),
        'f1_std': round(np.std(f1s), 4),
    }
    logger.info(f"    Discovery: F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}, "
                f"SHD={np.mean(shds):.1f}±{np.std(shds):.1f}")

    # Monitoring
    if any(r['monitoring'] for r in all_seed_results):
        mon_results = [r['monitoring'] for r in all_seed_results if r['monitoring']]
        if mon_results:
            combs = [m['combined_accuracy'] for m in mon_results]
            summary['monitoring'] = {
                'combined': f'{np.mean(combs):.1f} ± {np.std(combs):.1f}',
                'combined_mean': round(np.mean(combs), 2),
                'combined_std': round(np.std(combs), 2),
            }
            logger.info(f"    Monitoring: {np.mean(combs):.1f}%±{np.std(combs):.1f}%")

    # Policy (aggregate PolicyGRID at each ε — per-seed graphs)
    if any(r['policy'] for r in all_seed_results):
        pol_summary = {}
        hv_points = []  # for per-seed HV
        for ct in COMFORT_TARGETS:
            seed_sats, seed_kwhs, seed_dhs, seed_mos = [], [], [], []
            for r in all_seed_results:
                if r['policy']:
                    pg = [p for p in r['policy']
                          if p['method'] == 'PolicyGRID' and p['comfort_target'] == ct]
                    if pg:
                        seed_sats.append(pg[0]['satisfaction'])
                        seed_kwhs.append(pg[0].get('kWh', 0))
                        seed_dhs.append(pg[0].get('degree_hours', 0))
                        seed_mos.append(pg[0].get('MO', 0))
            if seed_sats:
                pol_summary[f'eps_{ct}'] = {
                    'sat': f'{np.mean(seed_sats):.1f} ± {np.std(seed_sats):.1f}',
                    'sat_mean': round(np.mean(seed_sats), 2),
                    'sat_std': round(np.std(seed_sats), 2),
                    'kWh': f'{np.mean(seed_kwhs):.3f} ± {np.std(seed_kwhs):.3f}',
                    'kWh_mean': round(np.mean(seed_kwhs), 3),
                    'dh': f'{np.mean(seed_dhs):.3f} ± {np.std(seed_dhs):.3f}',
                    'MO': f'{np.mean(seed_mos):.4f} ± {np.std(seed_mos):.4f}',
                    'MO_mean': round(np.mean(seed_mos), 4),
                }
                hv_points.append((np.mean(seed_sats) / 100, np.mean(seed_kwhs)))
        pol_summary['hypervolume'] = compute_hypervolume(hv_points)
        summary['policy_per_seed'] = pol_summary
        logger.info(f"    Policy per-seed (PolicyGRID):")
        for ct, vals in pol_summary.items():
            if isinstance(vals, dict):
                logger.info(f"      {ct}: Sat={vals['sat']}, kWh={vals['kWh']}, MO={vals['MO']}")

    # Policy (consensus graph)
    if consensus_policy_df is not None:
        con_pol_summary = {}
        for method in consensus_policy_df['method'].unique():
            method_data = consensus_policy_df[consensus_policy_df['method'] == method]
            con_pol_summary[method] = {
                f'eps_{r["comfort_target"]}': {
                    'sat': r['satisfaction'], 'kWh': r['kWh'],
                    'MO': r['MO'], 'HV': r.get('HV', 0)
                } for _, r in method_data.iterrows()
            }
        summary['policy_consensus'] = con_pol_summary

    # Consensus
    summary['consensus'] = consensus_metrics
    summary['jaccard'] = round(avg_jaccard, 4)

    with open(f'{out_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Final Table ───────────────────────────────────────────────────────
    logger.info(f"\n  {'Method':<20} {'F1':>6} {'SHD':>5} {'Edges':>6}")
    logger.info(f"  {'-'*40}")
    logger.info(f"  {'PolicyGRID':<20} {np.mean(f1s):>6.3f} "
                f"{np.mean(shds):>5.1f} {np.mean(n_edges):>6.1f}")
    for _, r in bench_df.iterrows():
        logger.info(f"  {r['method']:<20} {r['f1']:>6.3f} {r['shd']:>5} {r['n_edges']:>6}")

    logger.info(f"\n  Results saved to {out_dir}/")

    # ── Ablation Experiments ─────────────────────────────────────────────
    run_exps = experiments or set()
    run_all = 'all' in run_exps

    # Use consensus edges for ablation experiments that need a single graph
    cm = {v.lower(): v for v in data.columns}
    consensus_orig = {(cm.get(s, s), cm.get(t, t)) for s, t in consensus} if consensus else set()

    if run_all or 'm3a' in run_exps:
        if cfg['policy_config'] and cfg['sim_path']:
            run_m3a_graph_attribution(cfg, data, consensus_orig, smap, out_dir, seeds)

    if run_all or 'm3b' in run_exps:
        if cfg['sim_path']:
            run_m3b_intervention_ordering(cfg, data, gt_set, out_dir, seeds)

    if run_all or 'a1' in run_exps:
        if cfg['sim_path']:
            run_a1_sample_efficiency(cfg, data, gt_set, out_dir, seeds)

    if run_all or 'a2' in run_exps:
        if cfg['has_monitoring']:
            run_a2_regime_recovery(cfg, data, consensus_orig, smap, out_dir)

    if run_all or 'n2' in run_exps:
        if cfg['sim_path']:
            run_n2_llm_stress_test(cfg, data, gt_set, out_dir, seeds)

    if run_all or 'm4' in run_exps:
        run_m4_temporal_graph(cfg, data, gt_set, out_dir)

    if run_all or 'ablation' in run_exps:
        if cfg['has_monitoring']:
            run_sensor_ablation(cfg, data, consensus_orig, smap, out_dir, seeds)

    if run_all or 'int_strategy' in run_exps:
        if cfg['sim_path']:
            run_intervention_strategy_ablation(cfg, data, gt_set, out_dir, seeds)

    if run_all or 'hyperparam' in run_exps:
        if cfg['sim_path']:
            run_hyperparam_ablation(cfg, data, gt_set, consensus_orig, smap, out_dir, seeds)


def run_intervention_strategy_ablation(cfg, data, gt_set, out_dir, seeds):
    """Intervention strategy ablation: test the 3+7 multi-strategy design.

    Ablates two axes:
      1. Screening-confirmatory split: 1+9, 3+7, 5+5, 10+0
      2. Strategy diversity: all-deterministic, all-paired, mixed (default)

    For each configuration, runs full discovery and reports F1, SHD, and
    the number of validated edges.
    """
    logger.info(f"\n  ── Intervention Strategy Ablation ──")
    abl_dir = f'{out_dir}/int_strategy_ablation'
    os.makedirs(abl_dir, exist_ok=True)

    from src.pipeline_cwm import create_cwm_pipeline

    sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None
    results = []

    # ── Axis 1: Split ratio ablation (with mixed strategy) ──
    split_configs = [
        {'name': 'split_1_9',  'initial': 1, 'extended': 9},
        {'name': 'split_3_7',  'initial': 3, 'extended': 7},  # default
        {'name': 'split_5_5',  'initial': 5, 'extended': 5},
        {'name': 'split_10_0', 'initial': 10, 'extended': 0},
    ]

    for split_cfg in split_configs:
        for seed in seeds:
            logger.info(f"    {split_cfg['name']}, seed={seed}")
            random.seed(seed); np.random.seed(seed)

            pipeline = create_cwm_pipeline(
                csv_data=data, api_key=API_KEY, smart_room_path=sim_path,
                dataset_type=cfg['dataset_type'], max_iterations=10,
                actuator_vars=cfg['actuators'],
                non_intervenable_vars=cfg['non_intervenable'],
            )

            # Override test counts
            pipeline.pipeline.tester.initial_test_count = split_cfg['initial']
            pipeline.pipeline.tester.extended_test_count = split_cfg['extended']

            pipeline.run_with_cwm()
            val = {(s.lower(), t.lower())
                   for s, t in pipeline.pipeline.validated_edges}
            m = graph_metrics(val, gt_set)
            results.append({
                'ablation': split_cfg['name'],
                'strategy': 'mixed',
                'seed': seed,
                'f1': m['f1'], 'shd': m['shd'],
                'n_validated': len(val),
            })
            logger.info(f"      F1={m['f1']:.3f}, SHD={m['shd']}, "
                         f"edges={len(val)}")

    # ── Axis 2: Strategy diversity ablation (with 3+7 split) ──
    # We control this by patching what additional_interventions are passed.
    # "all_deterministic" = no additional_interventions (only primary)
    # "all_paired" = only counterfactual as additional
    # "mixed" = already run above in split_3_7

    strategy_configs = [
        {'name': 'all_deterministic', 'use_additional': False},
        {'name': 'all_paired',        'use_paired_only': True},
    ]

    for strat_cfg in strategy_configs:
        for seed in seeds:
            logger.info(f"    {strat_cfg['name']}, seed={seed}")
            random.seed(seed); np.random.seed(seed)

            pipeline = create_cwm_pipeline(
                csv_data=data, api_key=API_KEY, smart_room_path=sim_path,
                dataset_type=cfg['dataset_type'], max_iterations=10,
                actuator_vars=cfg['actuators'],
                non_intervenable_vars=cfg['non_intervenable'],
            )

            # Patch: disable additional interventions
            if strat_cfg.get('use_additional') is False:
                # Monkey-patch tester to ignore additional_interventions
                orig_test = pipeline.pipeline.tester.test_edge_iteratively
                def single_strategy_test(edge, intervention, additional_interventions=None):
                    return orig_test(edge, intervention, additional_interventions=None)
                pipeline.pipeline.tester.test_edge_iteratively = single_strategy_test

            elif strat_cfg.get('use_paired_only'):
                # Monkey-patch: only pass counterfactual, no LLM
                orig_test = pipeline.pipeline.tester.test_edge_iteratively
                def paired_only_test(edge, intervention, additional_interventions=None):
                    if additional_interventions:
                        # Keep only paired counterfactual, drop LLM
                        paired = [i for i in additional_interventions
                                  if 'counterfactual' in i.get('strategy_type', '')]
                        return orig_test(edge, intervention,
                                        additional_interventions=paired or None)
                    return orig_test(edge, intervention, additional_interventions=None)
                pipeline.pipeline.tester.test_edge_iteratively = paired_only_test

            pipeline.run_with_cwm()
            val = {(s.lower(), t.lower())
                   for s, t in pipeline.pipeline.validated_edges}
            m = graph_metrics(val, gt_set)
            results.append({
                'ablation': 'split_3_7',
                'strategy': strat_cfg['name'],
                'seed': seed,
                'f1': m['f1'], 'shd': m['shd'],
                'n_validated': len(val),
            })
            logger.info(f"      F1={m['f1']:.3f}, SHD={m['shd']}, "
                         f"edges={len(val)}")

    # Save results
    result_df = pd.DataFrame(results)
    result_df.to_csv(f'{abl_dir}/results.csv', index=False)

    # Summary
    logger.info(f"\n    Intervention Strategy Ablation Summary:")
    logger.info(f"    {'Config':<25} {'Strategy':<20} {'F1':>8} {'SHD':>6} {'Edges':>6}")
    for (abl, strat), grp in result_df.groupby(['ablation', 'strategy']):
        logger.info(f"    {abl:<25} {strat:<20} "
                     f"{grp['f1'].mean():>8.3f} {grp['shd'].mean():>6.1f} "
                     f"{grp['n_validated'].mean():>6.1f}")

    return result_df


def run_hyperparam_ablation(cfg, data, gt_set, validated_edges, smap, out_dir, seeds):
    """Hyperparameter ablation sweep across 7 dimensions.

    Each ablation varies one parameter while holding others at default.
    Discovery ablations (K, alpha_prune, tau_r) re-run the discovery pipeline.
    Monitoring ablations (sigma, eta) re-run monitoring inference only.
    Policy ablations (ridge_alpha) refit the SEM and re-evaluate policy.
    """
    logger.info(f"\n  ── Hyperparameter Ablation Sweep ──")
    abl_dir = f'{out_dir}/hyperparam_ablation'
    os.makedirs(abl_dir, exist_ok=True)

    from src.pipeline_cwm import create_cwm_pipeline
    sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None
    results = []

    # ── 1. Intervention budget K ──
    logger.info("    Ablation 1/7: Intervention budget K")
    for K in [3, 5, 7, 10, 15]:
        for seed in seeds[:3]:  # use 3 seeds to save time
            logger.info(f"      K={K}, seed={seed}")
            random.seed(seed); np.random.seed(seed)
            pipeline = create_cwm_pipeline(
                csv_data=data, api_key=API_KEY, smart_room_path=sim_path,
                dataset_type=cfg['dataset_type'], max_iterations=10,
                actuator_vars=cfg['actuators'],
                non_intervenable_vars=cfg['non_intervenable'],
            )
            pipeline.pipeline.tester.initial_test_count = max(1, K // 3)
            pipeline.pipeline.tester.extended_test_count = K - max(1, K // 3)
            try:
                pipeline.run_with_cwm()
                val = {(s.lower(), t.lower()) for s, t in pipeline.pipeline.validated_edges}
                m = graph_metrics(val, gt_set)
                results.append({'param': 'K', 'value': K, 'seed': seed,
                                'f1': m['f1'], 'shd': m['shd'], 'n_edges': len(val)})
            except Exception as e:
                logger.error(f"      Failed: {e}")

    # ── 2. Pruning alpha ──
    logger.info("    Ablation 2/7: Pruning alpha")
    for alpha_p in [0.001, 0.01, 0.05, 0.10]:
        for seed in seeds[:3]:
            logger.info(f"      alpha_prune={alpha_p}, seed={seed}")
            random.seed(seed); np.random.seed(seed)
            pipeline = create_cwm_pipeline(
                csv_data=data, api_key=API_KEY, smart_room_path=sim_path,
                dataset_type=cfg['dataset_type'], max_iterations=10,
                actuator_vars=cfg['actuators'],
                non_intervenable_vars=cfg['non_intervenable'],
            )
            # Set pruning alpha on the pipeline
            if hasattr(pipeline, '_pruning_alpha'):
                pipeline._pruning_alpha = alpha_p
            try:
                pipeline.run_with_cwm()
                val = {(s.lower(), t.lower()) for s, t in pipeline.pipeline.validated_edges}
                m = graph_metrics(val, gt_set)
                results.append({'param': 'alpha_prune', 'value': alpha_p, 'seed': seed,
                                'f1': m['f1'], 'shd': m['shd'], 'n_edges': len(val)})
            except Exception as e:
                logger.error(f"      Failed: {e}")

    # ── 3. Rescue threshold tau_r ──
    logger.info("    Ablation 3/7: Rescue threshold tau_r")
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for seed in seeds[:3]:
            logger.info(f"      tau_r={tau}, seed={seed}")
            random.seed(seed); np.random.seed(seed)
            pipeline = create_cwm_pipeline(
                csv_data=data, api_key=API_KEY, smart_room_path=sim_path,
                dataset_type=cfg['dataset_type'], max_iterations=10,
                actuator_vars=cfg['actuators'],
                non_intervenable_vars=cfg['non_intervenable'],
            )
            if hasattr(pipeline, '_rescue_threshold'):
                pipeline._rescue_threshold = tau
            try:
                pipeline.run_with_cwm()
                val = {(s.lower(), t.lower()) for s, t in pipeline.pipeline.validated_edges}
                m = graph_metrics(val, gt_set)
                results.append({'param': 'tau_r', 'value': tau, 'seed': seed,
                                'f1': m['f1'], 'shd': m['shd'], 'n_edges': len(val)})
            except Exception as e:
                logger.error(f"      Failed: {e}")

    # ── 4. Hold period H ──
    logger.info("    Ablation 4/7: Hold period H")
    for H in [10, 20, 30, 50]:
        for seed in seeds[:3]:
            logger.info(f"      H={H}, seed={seed}")
            random.seed(seed); np.random.seed(seed)
            pipeline = create_cwm_pipeline(
                csv_data=data, api_key=API_KEY, smart_room_path=sim_path,
                dataset_type=cfg['dataset_type'], max_iterations=10,
                actuator_vars=cfg['actuators'],
                non_intervenable_vars=cfg['non_intervenable'],
            )
            # Hold period is set in the JS sim via --hold-steps; set on tester
            if hasattr(pipeline.pipeline.tester, 'hold_steps'):
                pipeline.pipeline.tester.hold_steps = H
            try:
                pipeline.run_with_cwm()
                val = {(s.lower(), t.lower()) for s, t in pipeline.pipeline.validated_edges}
                m = graph_metrics(val, gt_set)
                results.append({'param': 'H', 'value': H, 'seed': seed,
                                'f1': m['f1'], 'shd': m['shd'], 'n_edges': len(val)})
            except Exception as e:
                logger.error(f"      Failed: {e}")

    # ── 5-7: Monitoring ablations (sigma, eta, ridge_alpha) ──
    # These don't re-run discovery; they refit monitors on existing edges.
    if cfg['has_monitoring'] and validated_edges:
        # ── 5. Monitoring sigma ──
        logger.info("    Ablation 5/7: Monitoring sigma")
        sim_records = _collect_regime_episode(cfg, smap, duration_steps=2880)
        if sim_records:
            for sigma_val in [1.0, 2.0, 3.0, 5.0]:
                monitors = _build_monitors_for_recovery(cfg, data, validated_edges, smap)
                if not monitors:
                    continue
                # Override sigma in inference
                mon_records = _run_monitoring_inference_with_params(
                    cfg, monitors, sim_records, smap, sigma=sigma_val, forget=0.12)
                if mon_records:
                    df = pd.DataFrame(mon_records)
                    acc = (df['predicted_regime'] == df['regime_gt']).mean() * 100
                    results.append({'param': 'sigma', 'value': sigma_val, 'seed': 0,
                                    'accuracy': round(acc, 1)})
                    logger.info(f"      sigma={sigma_val}: acc={acc:.1f}%")

            # ── 6. Monitoring forget factor eta ──
            logger.info("    Ablation 6/7: Monitoring forget factor eta")
            for eta_val in [0.05, 0.10, 0.12, 0.15, 0.20]:
                monitors = _build_monitors_for_recovery(cfg, data, validated_edges, smap)
                if not monitors:
                    continue
                mon_records = _run_monitoring_inference_with_params(
                    cfg, monitors, sim_records, smap, sigma=2.0, forget=eta_val)
                if mon_records:
                    df = pd.DataFrame(mon_records)
                    acc = (df['predicted_regime'] == df['regime_gt']).mean() * 100
                    results.append({'param': 'eta', 'value': eta_val, 'seed': 0,
                                    'accuracy': round(acc, 1)})
                    logger.info(f"      eta={eta_val}: acc={acc:.1f}%")

        # ── 7. Ridge alpha for policy SEM ──
        logger.info("    Ablation 7/7: Ridge alpha")
        if cfg['policy_config'] and cfg['sim_path']:
            for ridge_a in [0.1, 0.5, 1.0, 5.0]:
                for seed in seeds[:3]:
                    logger.info(f"      ridge_alpha={ridge_a}, seed={seed}")
                    random.seed(seed); np.random.seed(seed)
                    pol_cfg = dict(cfg['policy_config'])
                    pol_df = run_policy_for_seed(cfg, data, validated_edges, smap,
                                                 ridge_alpha_override=ridge_a)
                    if pol_df is not None:
                        pg = pol_df[pol_df['method'] == 'PolicyGRID']
                        if len(pg) > 0:
                            results.append({
                                'param': 'ridge_alpha', 'value': ridge_a, 'seed': seed,
                                'mean_sat': round(pg['satisfaction'].mean(), 2),
                                'mean_eng': round(pg['energy'].mean(), 2),
                            })
    else:
        logger.info("    Skipping monitoring/policy ablations (no monitoring or no edges)")

    # Save
    result_df = pd.DataFrame(results)
    result_df.to_csv(f'{abl_dir}/results.csv', index=False)

    # Summary
    logger.info(f"\n    Hyperparameter Ablation Summary:")
    for param, grp in result_df.groupby('param'):
        logger.info(f"    {param}:")
        for _, row in grp.iterrows():
            metrics = {k: v for k, v in row.items() if k not in ['param', 'seed']}
            logger.info(f"      {metrics}")

    return result_df


def _run_monitoring_inference_with_params(cfg, monitors, sim_records, smap,
                                           sigma=2.0, forget=0.12):
    """Run monitoring inference with overridden sigma and forget factor."""
    mode = monitors.get('_mode', '4way')

    if mode == '4way':
        regimes = ['base', 'occ', 'win', 'full']
        weights = {r: 0.25 for r in regimes}
    else:
        regimes = ['closed', 'open']
        weights = {r: 0.5 for r in regimes}

    obs_vars = monitors[regimes[0]]['obs_vars']
    results = []

    for rec in sim_records:
        state = rec['state_obs']
        errors = {}
        for regime in regimes:
            models = monitors[regime]['models']
            total_err = 0.0
            n = 0
            for (s, t), m in models.items():
                if s in state:
                    pred = m['coef'] * state[s] + m['intercept']
                    actual = state.get(t, pred)
                    total_err += abs(actual - pred)
                    n += 1
            errors[regime] = total_err / max(n, 1)

        likelihoods = {r: np.exp(-sigma * errors[r]) for r in regimes}
        for r in regimes:
            weights[r] *= likelihoods[r]

        total = sum(weights.values())
        if total > 1e-12:
            weights = {r: w / total for r, w in weights.items()}
        else:
            weights = {r: 1.0 / len(regimes) for r in regimes}

        uniform = 1.0 / len(regimes)
        weights = {r: (1 - forget) * w + forget * uniform
                   for r, w in weights.items()}

        predicted = max(weights, key=weights.get)
        if mode == '4way':
            regime_map = {'base': 0, 'occ': 1, 'win': 2, 'full': 3}
        else:
            regime_map = {'closed': 0, 'open': 1}

        results.append({
            'step': rec['step'], 'hour': rec['hour'],
            'regime_gt': rec['regime_gt'],
            'predicted_regime': regime_map.get(predicted, -1),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

ALL_EXPERIMENTS = {'m3a', 'm3b', 'a1', 'a2', 'n2', 'm4', 'ablation',
                   'int_strategy', 'hyperparam'}


def main():
    parser = argparse.ArgumentParser(
        description='Full pipeline: discovery → monitoring → policy, per seed',
        epilog="""
Examples:
  # Full pipeline on rich_building, 3 seeds, all experiments:
  python run_full_pipeline.py --sim smart_building_rich --seeds 3

  # Discovery + monitoring only (no policy, no ablations):
  python run_full_pipeline.py --sim open_window --stages discovery,monitoring --experiments none

  # Only ablation experiments (skip main pipeline, reuse saved edges):
  python run_full_pipeline.py --sim smart_building_rich --skip-main --experiments m3a,a2,n2

  # Policy only with saved edges, 5 seeds:
  python run_full_pipeline.py --sim smart_building_rich --stages policy --skip-discovery --seeds 5

  # Just M3A + A1 ablations on all sims:
  python run_full_pipeline.py --sim all --skip-main --experiments m3a,a1
""",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Sim and seed selection
    parser.add_argument('--sim', default='all',
                        choices=list(SIM_CONFIGS.keys()) + ['all'],
                        help='Which simulator (default: all)')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of seeds (default: 3)')
    parser.add_argument('--seed-list', type=str, default=None,
                        help='Comma-separated seed list (overrides --seeds)')

    # Pipeline stage control
    parser.add_argument('--stages', type=str, default='all',
                        help='Comma-separated pipeline stages: '
                             'all,discovery,benchmarks,monitoring,policy,consensus '
                             '(default: all)')
    parser.add_argument('--skip-discovery', action='store_true',
                        help='Skip discovery, load edges from prior run in results/')
    parser.add_argument('--skip-main', action='store_true',
                        help='Skip entire main pipeline, only run --experiments')

    # Experiment/ablation selection
    parser.add_argument('--experiments', type=str, default='all',
                        help='Comma-separated experiments: '
                             'all,none,m3a,m3b,a1,a2,n2,m4,ablation (default: all)')

    # Generator ablation (rebuttal C2: leave-one-generator-out)
    parser.add_argument('--generators', type=str, default=None,
                        help='Comma-separated subset of pc,sam,llm,varlingam '
                             'to use in Phase 1 (default: all four)')
    parser.add_argument('--out-suffix', type=str, default=None,
                        help='Suffix for results dir, e.g. "no_llm" writes to '
                             'results/full_pipeline/<sim>_no_llm/ (protects '
                             'primary results)')

    args = parser.parse_args()

    if args.seed_list:
        seeds = [int(s) for s in args.seed_list.split(',')]
    else:
        seeds = DEFAULT_SEEDS[:args.seeds]

    experiments = set(args.experiments.split(','))
    if 'none' in experiments:
        experiments = set()

    stages = set(args.stages.split(','))

    t_start = time.time()

    generators = None
    if args.generators:
        generators = [g.strip().lower() for g in args.generators.split(',')]
        valid = {'pc', 'sam', 'llm', 'varlingam'}
        bad = set(generators) - valid
        if bad:
            parser.error(f"Unknown generators: {sorted(bad)}. Valid: {sorted(valid)}")
        if args.out_suffix is None:
            parser.error("--generators requires --out-suffix so ablation runs "
                         "don't overwrite primary results")

    sims = list(SIM_CONFIGS.keys()) if args.sim == 'all' else [args.sim]
    for sim_name in sims:
        run_full_pipeline(sim_name, seeds, experiments,
                          stages=stages,
                          skip_discovery=args.skip_discovery,
                          skip_main=args.skip_main,
                          generators=generators,
                          out_suffix=args.out_suffix)

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*70}")
    logger.info(f"  TOTAL TIME: {elapsed/60:.1f} minutes")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
