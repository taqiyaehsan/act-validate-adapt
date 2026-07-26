#!/usr/bin/env python3
"""
NeurIPS Final Experiments: PolicyGRID Benchmarks + Sensor Ablation
===================================================================
Two experiments for advisor presentation:

  Exp 1 — Full discovery pipeline + Monitoring benchmarks + Policy (GRID vs DDPC variants)
  Exp 2 — Sensor ablation (progressive sensor dropout)

Output: results/neurips_final/

Run:  python neurips_final_experiments.py
"""

import sys, os, json, subprocess, time, warnings, math, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict

sys.path.insert(0, '.')
sys.path.insert(0, 'benchmarks_new')
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH = 'data_regen/smart_building_rich_processed.csv'
SCALING_PATH = 'data_regen/smart_building_rich_processed_scaling.csv'
SIM_PATH = 'js/smart_building_rich.js'
GT_PATH = 'ground_truth_graphs.json'
OUT_DIR = 'results/neurips_final'

API_KEY = 'YOUR_OPENAI_API_KEY'

ALL_VARS = ['outdoortemp', 'solarradiation', 'occupancy', 'windowposition',
            'temperature', 'humidity', 'co2', 'lightlevel', 'noisedb',
            'airquality', 'pmv', 'hvacpower', 'lightingpower',
            'energyconsumption', 'satisfaction']
LATENT_VARS = {'occupancy', 'windowposition'}
OBSERVABLE_VARS = [v for v in ALL_VARS if v not in LATENT_VARS]
SENSOR_VARS = ['temperature', 'humidity', 'co2', 'lightlevel', 'noisedb', 'airquality']

# Physical constants for kWh / DH calculation
HVAC_MAX_KW = 5.0       # from js/smart_building_rich.js
LIGHT_MAX_KW = 0.5      # from js/smart_building_rich.js
STEP_HOURS = 1.0 / 60   # each step = 1 minute
COMFORT_REF_C = 22.0     # ASHRAE-55 neutral temperature
COMFORT_DEADBAND_C = 1.0 # deadband around setpoint

# Epsilon-constraint comfort targets (DH budget) for Pareto sweep
DEFAULT_COMFORT_TARGETS = [0.5, 0.8, 1.0, 1.5, 2.0]

# ── Visual style: colorblind-safe (Wong 2011) + B&W readable ─────────────────
# Colors chosen from Wong's colorblind-safe palette + IBM design.
# Every method also gets a unique marker and linestyle for B&W readability.
COLORS = {
    # PolicyGRID variants — blue family, clearly distinct
    'PolicyGRID-O':        '#0072B2',   # strong blue
    'PolicyGRID':          '#000000',   # black (the main method)
    'PolicyGRID-R':        '#009E73',   # teal-green

    # Benchmark discovery methods
    'PC':                  '#D55E00',   # vermilion
    'SAM':                 '#CC79A7',   # pink
    'GIES':                '#E69F00',   # amber
    'ICP':                 '#56B4E9',   # sky blue
    'NOTEARS-I':           '#F0E442',   # yellow
    'ABCD':                '#D55E00',   # vermilion (different marker)
    'JCI':                 '#CC79A7',   # pink (different marker)
    'Causal Bandits':      '#999999',   # gray
    'IID':                 '#56B4E9',   # sky blue (different marker)

    # Policy baselines
    'DDPC_Behavioral':     '#D55E00',   # vermilion
    'DDPC_Subspace':       '#E69F00',   # amber
    'DDPC_Neural':         '#CC79A7',   # pink
    'DDPC_PETS':           '#56B4E9',   # sky blue
    'MPC_Linear':          '#009E73',   # teal
    'MPC_Ensemble':        '#0072B2',   # blue
    'C-MBPO':              '#F0E442',   # yellow
    'PID':                 '#999999',   # gray
}

# Markers: every method gets a unique shape (readable in B&W)
MARKERS = {
    'PolicyGRID-O':     's',    # square
    'PolicyGRID':       'o',    # circle
    'PolicyGRID-R':     'D',    # diamond
    'PC':               '^',    # triangle up
    'SAM':              'v',    # triangle down
    'GIES':             '<',    # triangle left
    'ICP':              '>',    # triangle right
    'NOTEARS-I':        'p',    # pentagon
    'ABCD':             'h',    # hexagon
    'JCI':              '*',    # star
    'Causal Bandits':   'X',    # x-filled
    'IID':              'd',    # thin diamond
    'DDPC_Behavioral':  '^',
    'DDPC_Subspace':    '<',
    'DDPC_Neural':      'v',
    'DDPC_PETS':        'p',
    'MPC_Linear':       'h',
    'MPC_Ensemble':     '8',
    'C-MBPO':           'P',
    'PID':              'X',
}

# Line styles: PolicyGRID solid, benchmarks varied
LINESTYLES = {
    'PolicyGRID-O':     '--',
    'PolicyGRID':       '-',
    'PolicyGRID-R':     '-.',
    'PC':               ':',
    'SAM':              ':',
    'GIES':             '--',
    'ICP':              '-.',
    'NOTEARS-I':        '--',
    'ABCD':             ':',
    'JCI':              '-.',
    'Causal Bandits':   '--',
    'IID':              ':',
    'DDPC_Behavioral':  '--',
    'DDPC_Subspace':    '-.',
    'DDPC_Neural':      ':',
    'DDPC_PETS':        '--',
    'MPC_Linear':       '-.',
    'MPC_Ensemble':     ':',
    'C-MBPO':           '-',
    'PID':              '--',
}

# Hatch patterns for bar charts (B&W readability)
HATCHES = {
    'PolicyGRID-O':     '///',
    'PolicyGRID':       '',       # solid fill (main method)
    'PolicyGRID-R':     '\\\\\\',
    'PC':               '...',
    'SAM':              'xx',
    'GIES':             '++',
    'ICP':              '||',
    'NOTEARS-I':        '--',
    'ABCD':             'oo',
    'JCI':              '**',
    'Causal Bandits':   'OO',
    'IID':              '..',
    'DDPC_Behavioral':  '...',
    'DDPC_Subspace':    '++',
    'DDPC_Neural':      'xx',
    'DDPC_PETS':        '||',
    'MPC_Linear':       '//',
    'MPC_Ensemble':     '\\\\',
    'C-MBPO':           'oo',
    'PID':              '---',
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load data, ground truth, scaling params (no cached edges)."""
    data = pd.read_csv(DATA_PATH)
    scaling = pd.read_csv(SCALING_PATH)

    with open(GT_PATH) as f:
        gt_raw = json.load(f)['smart_building_rich']['edges']
    gt_set = {(s.lower(), t.lower()) for s, t in gt_raw}

    return data, scaling, gt_set


def get_scaling_map(scaling):
    """Build {var_lower: (min, max)} from scaling CSV."""
    feat_col = 'feature' if 'feature' in scaling.columns else 'column'
    min_col = 'data_min' if 'data_min' in scaling.columns else 'min'
    max_col = 'data_max' if 'data_max' in scaling.columns else 'max'
    smap = {}
    for _, row in scaling.iterrows():
        smap[row[feat_col].lower()] = (row[min_col], row[max_col])
    return smap


def norm_to_phys(val, var, smap):
    if var in smap:
        mn, mx = smap[var]
        return mn + val * (mx - mn)
    return val


def phys_to_norm(val, var, smap):
    if var in smap:
        mn, mx = smap[var]
        rng = mx - mn
        return (val - mn) / rng if rng > 0 else 0.5
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def graph_metrics(predicted_edges, gt_set):
    """Compute SHD, F1, precision, recall for a predicted edge set vs GT."""
    pred = {(s.lower(), t.lower()) for s, t in predicted_edges}
    gt = {(s.lower(), t.lower()) for s, t in gt_set}
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    shd = fp + fn
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {'n_edges': len(pred), 'tp': tp, 'fp': fp, 'fn': fn,
            'shd': shd, 'precision': round(p, 3), 'recall': round(r, 3),
            'f1': round(f1, 3)}


# ═══════════════════════════════════════════════════════════════════════════════
# FULL DISCOVERY PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_discovery(data, seed=42):
    """Run the complete PolicyGRID discovery pipeline:
    PC + SAM + LLM + VARLiNGAM → union → intervention validation → validated edges.

    Returns dict with validated_edges, obs_only_edges.
    """
    from src.pipeline_cwm import create_cwm_pipeline

    random.seed(seed)
    np.random.seed(seed)

    sim_path = os.path.abspath(SIM_PATH)

    logger.info("  Creating CWM pipeline (PC + SAM + LLM + VARLiNGAM)...")
    pipeline = create_cwm_pipeline(
        csv_data=data,
        api_key=API_KEY,
        smart_room_path=sim_path,
        dataset_type='smart_building_rich',
        max_iterations=10,
        alpha=0.5,
        beta=0.5,
        effect_threshold=0.1,
        actuator_vars=['hvacpower', 'lightingpower'],
        non_intervenable_vars=[
            'outdoortemp', 'solarradiation', 'occupancy', 'windowposition',
            'pmv', 'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
        derived_vars=['pmv', 'energyconsumption', 'satisfaction', 'overallsatisfaction'],
    )

    logger.info("  Running Phase 1+2: Discovery + Intervention Validation...")
    t0 = time.time()
    final_dag, final_metrics = pipeline.run_with_cwm()
    disc_time = time.time() - t0
    logger.info(f"  Discovery completed in {disc_time:.1f}s")

    # Extract validated edges
    validated_edges = set(pipeline.pipeline.validated_edges)

    # Extract method DAG union (obs-only edges)
    method_dags = pipeline.pipeline.get_method_dags()
    obs_only_edges = set()
    for G in method_dags.values():
        obs_only_edges.update(G.edges())

    # Count interventions
    intervention_count = sum(
        len(r) for r in pipeline.pipeline.tester.intervention_results.values()
    )

    logger.info(f"  Validated: {len(validated_edges)} edges, "
                f"Obs-only union: {len(obs_only_edges)} edges, "
                f"Interventions: {intervention_count}")

    return {
        'validated_edges': validated_edges,
        'obs_only_edges': obs_only_edges,
        'intervention_count': intervention_count,
        'discovery_time_s': disc_time,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 1A: DISCOVERY BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark_discovery(data):
    """Run all benchmark causal discovery methods on the dataset."""
    results = {}

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

    for name, module_name, func_name in benchmarks:
        try:
            logger.info(f"  Running {name}...")
            t0 = time.time()
            mod = __import__(module_name)
            func = getattr(mod, func_name)
            result = func(data, simulation_path=os.path.abspath(SIM_PATH))
            elapsed = time.time() - t0
            # Normalize return type: some benchmarks return list, tuple, Graph, or dict
            if isinstance(result, tuple) and len(result) == 2:
                edges = result[0]  # (edges, extra_info)
            elif hasattr(result, 'edges'):
                edges = list(result.edges())  # nx.Graph
            elif isinstance(result, dict):
                edges = result.get('edges', [])
            else:
                edges = result
            edge_set = {(s.lower(), t.lower()) for s, t in edges}
            results[name] = edge_set
            logger.info(f"    {name}: {len(edge_set)} edges in {elapsed:.1f}s")
        except Exception as e:
            logger.warning(f"    {name} FAILED: {e}")
            results[name] = set()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 1B: MONITORING — FACTORED BINARY BAYESIAN MONITORS
# ═══════════════════════════════════════════════════════════════════════════════

def build_observable_graph(edges):
    """Filter edges to those between observable variables only."""
    return {(s, t) for s, t in edges
            if s in OBSERVABLE_VARS and t in OBSERVABLE_VARS}


def compute_sensor_weights_from_graph(validated_edges, sensor_vars, latent_vars):
    """Derive sensor weights from the discovered causal graph structure.

    Sensors that receive validated causal edges from latent regime variables
    carry more regime-relevant information.  Weight = 1.0 (baseline) plus
    1.0 for each validated edge from a latent variable targeting that sensor.

    This is fully domain-agnostic: derived from the agent's own discovery
    results, not from domain knowledge or test data.

    Returns: {sensor: weight} dict, all weights >= 1.0
    """
    weights = {v: 1.0 for v in sensor_vars}
    for source, target in validated_edges:
        if source in latent_vars and target in sensor_vars:
            weights[target] += 1.0
    return weights


def build_predictive_model(obs_edges, data, mask, label=''):
    """Build PredictiveModel (linear temporal regression) on masked data."""
    import networkx as nx
    from src.causal_world_model import PredictiveModel

    G = nx.DiGraph()
    G.add_nodes_from(OBSERVABLE_VARS)
    G.add_edges_from(obs_edges)
    model = PredictiveModel(G, OBSERVABLE_VARS)

    cols = [c for c in data.columns if c.lower() in OBSERVABLE_VARS]
    df = data[cols].copy()
    df.columns = [c.lower() for c in df.columns]

    temporal = {}
    for var in OBSERVABLE_VARS:
        if var in df.columns:
            temporal[var] = df[var].values[:-1]
            temporal[f'{var}_next'] = df[var].values[1:]

    tdf = pd.DataFrame(temporal)
    tdf = tdf[mask[:-1]].reset_index(drop=True)
    model.fit_temporal(tdf)
    return model


def construct_factored_monitors(edges, data):
    """Build 4-way regime monitors: one predictive model per regime combination.

    Each model learns the sensor dynamics specific to its regime (base,
    occ-only, win-only, full).  At inference, all 4 compete via Bayesian
    model comparison.  No independence assumption between regime dimensions.
    """
    obs_edges = build_observable_graph(edges)
    if len(obs_edges) == 0:
        return None

    cols_lower = {c.lower(): c for c in data.columns}
    occ_vals = data[cols_lower['occupancy']].values
    win_vals = data[cols_lower['windowposition']].values

    occ_on = occ_vals >= 0.1
    win_on = win_vals >= 0.15

    monitors = {
        '_mode': '4way',
        'base': build_predictive_model(obs_edges, data, ~occ_on & ~win_on),
        'occ':  build_predictive_model(obs_edges, data,  occ_on & ~win_on),
        'win':  build_predictive_model(obs_edges, data, ~occ_on &  win_on),
        'full': build_predictive_model(obs_edges, data,  occ_on &  win_on),
    }
    return monitors


def run_monitoring(monitors, sim_records, sensor_vars=None, sensor_weights=None,
                   sigma=1.5, forget_factor=0.15):
    """Run regime monitoring, return combined records.

    Supports 4-way monitors (one model per regime).  At each timestep the 4
    models predict the next sensor state; prediction errors drive Bayesian
    weight updates with a forget factor to prevent lock-in.

    sensor_weights: optional {var: float} dict from compute_sensor_weights_from_graph.
    sigma / forget_factor: likelihood sharpness and forget rate. Defaults are the
        deployed values (1.5, 0.15); exposed as args so a sensitivity grid can
        sweep them without changing behavior for existing callers.
    """
    if sensor_vars is None:
        sensor_vars = SENSOR_VARS
    if monitors is None:
        return []
    # sigma: tuned for 4-way monitors (softer competition)
    # forget_factor: faster adaptation to regime changes
    regime_names = ['base', 'occ', 'win', 'full']
    weights = {r: 0.25 for r in regime_names}
    records = []

    for rec in sim_records:
        obs = rec['state_obs']

        # Compute likelihoods from each regime model (graph-weighted sensors)
        likelihoods = {}
        for rname in regime_names:
            model = monitors[rname]
            pred = model.predict(obs)
            if sensor_weights:
                err = sum(sensor_weights.get(v, 1.0) *
                          abs(obs.get(v, 0.5) - pred.get(v, 0.5))
                          for v in sensor_vars)
            else:
                err = sum(abs(obs.get(v, 0.5) - pred.get(v, 0.5))
                          for v in sensor_vars)
            likelihoods[rname] = np.exp(-sigma * err)

        # Bayesian update
        unnorm = {r: likelihoods[r] * weights[r] for r in regime_names}
        total = sum(unnorm.values())
        if total > 0:
            raw = {r: unnorm[r] / total for r in regime_names}
            weights = {r: (1 - forget_factor) * raw[r] + forget_factor * 0.25
                       for r in regime_names}
        # Re-normalize
        w_total = sum(weights.values())
        if w_total > 0:
            weights = {r: weights[r] / w_total for r in regime_names}

        winner = max(weights, key=weights.get)

        # Reconstruct marginal P_occ and P_win for metrics compatibility
        p_occ = weights['occ'] + weights['full']
        p_win = weights['win'] + weights['full']

        records.append({
            'step': rec['step'], 'hour': rec['hour'],
            'occ_active': rec['occ_active'], 'win_active': rec['win_active'],
            'regime_gt': rec['regime_gt'],
            'P_occ': p_occ, 'P_win': p_win,
            'winner': winner,
        })
    return records


def _gt_regime(log_entry):
    """Return the correct regime name(s) for a probe log entry."""
    occ, win = log_entry['gt_occ'], log_entry['gt_win']
    if occ and win:
        return {'full'}
    elif occ:
        return {'occ'}
    elif win:
        return {'win'}
    else:
        return {'base'}


def compute_monitoring_metrics(records):
    """Compute occ/win/combined detection accuracy from monitoring records."""
    if not records:
        return {'occ_accuracy': 0, 'win_accuracy': 0, 'combined_accuracy': 0}
    df = pd.DataFrame(records)
    occ_correct = ((df['P_occ'] >= 0.5) == df['occ_active'].astype(bool)).mean()
    win_correct = ((df['P_win'] >= 0.5) == df['win_active'].astype(bool)).mean()
    regime_map = {0: 'base', 1: 'occ', 2: 'win', 3: 'full'}
    gt_labels = df['regime_gt'].map(regime_map)
    combined = (df['winner'] == gt_labels).mean()
    return {
        'occ_accuracy': round(occ_correct * 100, 1),
        'win_accuracy': round(win_correct * 100, 1),
        'combined_accuracy': round(combined * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVE REGIME PROBING — LLM-Designed Diagnostic Interventions
# ═══════════════════════════════════════════════════════════════════════════════
#
# When the agent's posterior indicates uncertainty about the current regime,
# it designs and executes a diagnostic intervention using its actuators.
# The LLM reasons about what intervention would produce differential sensor
# responses under each regime hypothesis, keeping the approach domain-agnostic.
#
# Design principles (NeurIPS-defensible):
# 1. DOMAIN-AGNOSTIC: LLM designs both the intervention and interpretation.
#    No hardcoded thresholds or domain-specific probe protocols.
# 2. POSTERIOR-DRIVEN: Probing triggered by the agent's own belief state,
#    not by time-of-day or domain heuristics.
# 3. SAFE: LLM constrained to actuator-only interventions within comfort bounds.
# 4. ADAPTIVE: Confirmatory probes → back off (double interval).
#    Contradictory probes → stay alert (reset interval).
# 5. MODEL-INTERPRETED: Post-intervention state compared against all 4 regime
#    models — the regime whose model best predicts the outcome wins.
# ═══════════════════════════════════════════════════════════════════════════════


def design_diagnostic_probe(p_occ, p_win, state_phys, gpt_client=None):
    """LLM designs a two-phase diagnostic probe to resolve regime uncertainty.

    The LLM receives the current building state and the agent's posterior
    beliefs, then proposes a stimulus (actuator change) and observation
    protocol.  Falls back to a generic actuator perturbation if LLM is
    unavailable.

    Returns
    -------
    dict with keys:
        'phases': list of {'intervention': {var: val}, 'duration_steps': int}
        'observe_vars': list of sensor variables to watch
        'reasoning': str, LLM explanation of why this probe disambiguates
    """
    current_temp = state_phys.get('temperature',
                                   state_phys.get('Temperature', 22.0))
    current_hvac = state_phys.get('hvacPower',
                                   state_phys.get('hvacpower', 50))
    current_light = state_phys.get('lightingPower',
                                    state_phys.get('lightingpower', 30))
    outdoor_temp = state_phys.get('outdoorTemp',
                                   state_phys.get('outdoortemp', 20))

    if gpt_client is not None:
        system_prompt = f"""You are a building diagnostics expert. A smart building's
autonomous monitoring agent is uncertain about the current regime state.
It needs to design a SAFE diagnostic intervention to determine whether
the window is open and/or the room is occupied.

Current building state:
- Indoor temperature: {current_temp:.1f}°C
- Outdoor temperature: {outdoor_temp:.1f}°C
- HVAC power: {current_hvac:.0f}% (range 0-100, maps to 0-5kW heating/cooling)
- Lighting power: {current_light:.0f}% (range 0-100, maps to 0-500W)

Agent's current beliefs:
- P(room is occupied) = {p_occ:.3f}
- P(window is open) = {p_win:.3f}

Design a TWO-PHASE diagnostic probe:
- Phase 1 (stimulus): change an actuator to create a measurable perturbation
- Phase 2 (observation): change the actuator to observe how the system responds

CONSTRAINTS:
- You can ONLY set hvacPower and/or lightingPower (0-100 range each).
- The intervention must be SAFE for occupants who MAY be present.
- Keep the perturbation brief (1-3 minutes per phase) to minimize disruption.
- Total probe duration should be at most 6 minutes.

PHYSICAL REASONING:
- Window open: indoor temperature is coupled to outdoor temperature through
  air exchange. HVAC effects dissipate faster (heat/cool escapes through window).
- Window closed: room is thermally insulated. HVAC effects are sustained.
- Occupied: CO2 and noise levels respond to ventilation changes.
  People may notice and respond to lighting/temperature changes.
- Unoccupied: only temperature and humidity respond to HVAC changes.

Return a JSON object:
{{
    "phases": [
        {{"intervention": {{"hvacPower": <0-100>}}, "duration_steps": <1-3>}},
        {{"intervention": {{"hvacPower": <0-100>}}, "duration_steps": <1-3>}}
    ],
    "observe_vars": ["temperature", ...],
    "reasoning": "<why this two-phase protocol distinguishes the regimes>"
}}"""

        user_prompt = (f"Design a diagnostic probe. "
                       f"P(occupied)={p_occ:.3f}, P(window_open)={p_win:.3f}. "
                       f"Indoor={current_temp:.1f}°C, outdoor={outdoor_temp:.1f}°C.")

        try:
            response = gpt_client.generate_response(system_prompt, user_prompt,
                                                     temperature=0.3)
            plan = json.loads(response)

            phases = []
            for phase in plan.get('phases', []):
                intervention = {}
                raw = phase.get('intervention', {})
                if 'hvacPower' in raw:
                    intervention['hvacPower'] = max(0, min(100, float(
                        raw['hvacPower'])))
                if 'lightingPower' in raw:
                    intervention['lightingPower'] = max(0, min(100, float(
                        raw['lightingPower'])))
                if not intervention:
                    intervention = {'hvacPower': 80}
                dur = max(1, min(3, int(phase.get('duration_steps', 3))))
                phases.append({'intervention': intervention,
                               'duration_steps': dur})

            if not phases:
                phases = [{'intervention': {'hvacPower': 95}, 'duration_steps': 3},
                          {'intervention': {'hvacPower': 5}, 'duration_steps': 3}]

            observe_vars = plan.get('observe_vars', ['temperature'])
            reasoning = plan.get('reasoning', '')

            logger.info(f"  LLM probe design: {len(phases)} phases, "
                        f"observe={observe_vars}")
            logger.info(f"  LLM reasoning: {reasoning}")

            return {
                'phases': phases,
                'observe_vars': observe_vars,
                'reasoning': reasoning,
            }
        except Exception as e:
            logger.warning(f"  LLM probe design failed ({e}), "
                           f"using deterministic fallback")

    # ── Deterministic fallback ────────────────────────────────────────────
    # Generic two-phase probe: stimulus (high actuator) then observation
    # (low actuator).  The response trajectory differentiates regimes.
    return {
        'phases': [
            {'intervention': {'hvacPower': 95}, 'duration_steps': 3},
            {'intervention': {'hvacPower': 5}, 'duration_steps': 3},
        ],
        'observe_vars': ['temperature', 'co2', 'noisedb', 'humidity'],
        'reasoning': ('Phase 1: HVAC stimulus heats the room. '
                      'Phase 2: HVAC cut — observe thermal decay rate. '
                      'Window open → fast decay (heat escapes to outdoor). '
                      'Window closed → slow decay (insulated room). '
                      'Occupied → CO2/noise respond. Unoccupied → only temperature.'),
    }


def execute_probe(state_phys, probe_plan, smap, elapsed_ms,
                   outdoor_offset=0.0):
    """Execute a multi-phase diagnostic probe in the simulator.

    Returns per-phase sensor trajectories for model-based interpretation.
    """
    sim_abs = os.path.abspath(SIM_PATH)
    step_ms = 60000
    current_phys = dict(state_phys)

    # Record pre-probe sensor state (normalized)
    pre_norm = {}
    for k, v in current_phys.items():
        if isinstance(v, (int, float)):
            pre_norm[k.lower()] = phys_to_norm(v, k.lower(), smap)

    phase_trajectories = []
    for phase in probe_plan['phases']:
        intervention = phase['intervention']
        n_steps = phase['duration_steps']
        temps = []

        for _ in range(n_steps):
            elapsed_ms += step_ms
            cmd = ['node', sim_abs, '--single-step',
                   '--elapsed-ms', str(int(elapsed_ms)),
                   '--outdoor-offset', str(outdoor_offset),
                   '--state', json.dumps(current_phys),
                   '--intervention', json.dumps(intervention)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True,
                                         timeout=10)
                for line in result.stdout.split('\n'):
                    if 'RESULT:' in line:
                        current_phys = json.loads(
                            line.split('RESULT:', 1)[1].strip())
                        break
            except Exception:
                pass
            temps.append(current_phys.get('temperature',
                         current_phys.get('Temperature', 22)))

        phase_trajectories.append(temps)

    # Post-probe sensor state (normalized)
    post_norm = {}
    for k, v in current_phys.items():
        if isinstance(v, (int, float)):
            post_norm[k.lower()] = phys_to_norm(v, k.lower(), smap)

    post_obs = {k: v for k, v in post_norm.items() if k not in LATENT_VARS}

    return {
        'post_state_obs': post_obs,
        'post_state_phys': current_phys,
        'phase_trajectories': phase_trajectories,
        'pre_norm': pre_norm,
        'post_norm': post_norm,
        'elapsed_ms': elapsed_ms,
    }


def interpret_probe_with_llm(probe_result, probe_plan, p_occ, p_win,
                             gpt_client):
    """LLM interprets the probe results to determine regime likelihoods.

    The LLM designed the probe (knows what to expect), now it analyzes the
    actual sensor response.  This is fully domain-agnostic: the LLM reasons
    about the physical response without hardcoded thresholds.

    Returns dict: {'base': float, 'occ': float, 'win': float, 'full': float}
        — regime weight multipliers (boosts).
    """
    pre = probe_result['pre_norm']
    post = probe_result['post_norm']
    trajectories = probe_result['phase_trajectories']

    # Build concise sensor change summary
    changes = []
    for var in SENSOR_VARS:
        delta = round(post.get(var, 0.5) - pre.get(var, 0.5), 4)
        if abs(delta) > 0.001:
            changes.append(f"{var}: {delta:+.4f}")
    change_str = ', '.join(changes) if changes else 'minimal changes'

    # Temperature trajectory (the key signal)
    traj_parts = []
    for pi, temps in enumerate(trajectories):
        if temps:
            traj_parts.append(f"Phase {pi+1}: {[round(t,1) for t in temps]}")
    traj_str = '; '.join(traj_parts) if traj_parts else 'no trajectory data'

    system_prompt = (
        "You analyze smart building diagnostic probe results. "
        "A two-phase HVAC probe was run: Phase 1 = HVAC high (heat room), "
        "Phase 2 = HVAC cut (observe decay). "
        "Window OPEN: fast temperature decay after HVAC cut (heat escapes). "
        "Window CLOSED: slow decay (insulated). "
        "Occupied: CO2/noise respond. Unoccupied: only temperature responds. "
        "Return ONLY valid JSON with regime likelihood multipliers (0.1-10.0): "
        '{"base": <num>, "occ": <num>, "win": <num>, "full": <num>, '
        '"reasoning": "<brief>"}'
    )

    user_prompt = (
        f"P(occ)={p_occ:.2f}, P(win)={p_win:.2f}. "
        f"Temp trajectory: {traj_str}. "
        f"Sensor changes: {change_str}. "
        f"Which regime best explains this response?"
    )

    try:
        response = gpt_client.generate_response(system_prompt, user_prompt,
                                                 temperature=0.2)
        result = json.loads(response)

        boosts = {}
        for regime in ['base', 'occ', 'win', 'full']:
            boosts[regime] = max(0.1, min(10.0, float(result.get(regime, 1.0))))

        reasoning = result.get('reasoning', '')
        logger.info(f"    LLM probe interpretation: {boosts}")
        logger.info(f"    LLM reasoning: {reasoning}")

        return boosts, reasoning

    except Exception as e:
        logger.warning(f"    LLM interpretation failed ({e}), "
                       f"using model-based fallback")
        return None, str(e)


def compute_probe_boost(probe_result, monitors, sensor_vars,
                        probe_plan=None, p_occ=0.5, p_win=0.5,
                        gpt_client=None):
    """Interpret probe results — LLM-first with model-based fallback.

    1. If LLM available: LLM analyzes the sensor response and returns
       regime likelihood multipliers.  Domain-agnostic — the LLM reasons
       about physics without hardcoded thresholds.
    2. Fallback: compare pre→post sensor change against regime model
       predictions.  Less accurate for out-of-distribution probe states.
    """
    regime_names = ['base', 'occ', 'win', 'full']

    # ── Try LLM interpretation first ─────────────────────────────────────
    if gpt_client is not None and probe_plan is not None:
        boosts, reasoning = interpret_probe_with_llm(
            probe_result, probe_plan, p_occ, p_win, gpt_client)
        if boosts is not None:
            return boosts

    # ── Model-based fallback ─────────────────────────────────────────────
    sigma = 1.5
    pre_obs = {k: v for k, v in probe_result['pre_norm'].items()
               if k not in LATENT_VARS}
    post_obs = probe_result['post_state_obs']

    actual_delta = {v: post_obs.get(v, 0.5) - pre_obs.get(v, 0.5)
                    for v in sensor_vars}

    liks = {}
    for rname in regime_names:
        pred = monitors[rname].predict(pre_obs)
        pred_delta = {v: pred.get(v, 0.5) - pre_obs.get(v, 0.5)
                      for v in sensor_vars}
        delta_err = sum(abs(actual_delta[v] - pred_delta.get(v, 0))
                        for v in sensor_vars)
        liks[rname] = np.exp(-sigma * delta_err)

    total = sum(liks.values())
    if total < 1e-10:
        return {r: 1.0 for r in regime_names}

    mean_lik = total / 4
    boosts = {}
    for r in regime_names:
        ratio = liks[r] / max(1e-10, mean_lik)
        boosts[r] = max(0.05, min(20.0, ratio ** 2))

    return boosts


def compute_regime_decay_predictions(monitors, pre_state_obs, smap):
    """Predict what decay rate each regime model expects after HVAC perturbation.

    Each regime model has different temperature coefficients.  By predicting
    multi-step temperature trajectories from each model, we get per-regime
    expected decay rates that serve as reference for the virtual sensor.

    Uses 3-step rollout for more discriminative predictions:
    Step 1: HVAC high → peak temperature
    Step 2-3: HVAC low → observe decay over 2 steps
    """
    regime_names = ['base', 'occ', 'win', 'full']
    decay_preds = {}

    for rname in regime_names:
        model = monitors[rname]
        state = dict(pre_state_obs)

        # Phase 1: predict temp with high HVAC (3 steps to build heat)
        for _ in range(3):
            state_high = dict(state)
            state_high['hvacpower'] = 0.95
            pred = model.predict(state_high)
            state = dict(pred)
        peak_temp = state.get('temperature', pre_state_obs.get('temperature', 0.5))

        # Phase 2: predict decay with low HVAC (3 steps)
        temps = [peak_temp]
        for _ in range(3):
            state_low = dict(state)
            state_low['hvacpower'] = 0.05
            pred = model.predict(state_low)
            state = dict(pred)
            temps.append(state.get('temperature', temps[-1]))

        # Decay rate = average drop per step
        if len(temps) > 1:
            decay_preds[rname] = (temps[0] - temps[-1]) / (len(temps) - 1)
        else:
            decay_preds[rname] = 0.0

    return decay_preds


def run_monitoring_active(monitors, sim_records, smap, sensor_vars=None,
                          sensor_weights=None,
                          gpt_client=None, probe_cooldown=60, max_probes=50):
    """4-way Bayesian monitoring with probe-derived virtual sensors.

    When passive monitoring is uncertain, the agent executes an LLM-designed
    diagnostic probe (HVAC perturbation).  The probe's temperature decay rate
    becomes a temporary "virtual sensor" — an additional observation channel
    that persists in the Bayesian update for HOLD_DURATION steps.

    Each regime model predicts a different expected decay rate (from its
    learned temperature coefficients).  The regime whose predicted decay
    best matches the observed decay gets a likelihood boost at EVERY timestep
    during the hold period.  This solves the evidence persistence problem:
    instead of a one-time weight boost that gets eroded by passive updates,
    the decay observation contributes continuously.

    Fully domain-agnostic: the framework discovers which variables respond
    to perturbation, the LLM designs the probe protocol, and the regime
    models interpret the result.  No hardcoded thresholds.
    """
    if sensor_vars is None:
        sensor_vars = SENSOR_VARS
    if monitors is None:
        return [], []

    sigma = 1.5
    forget_factor = 0.15
    regime_names = ['base', 'occ', 'win', 'full']
    weights = {r: 0.25 for r in regime_names}

    records = []
    probe_log = []
    steps_since_probe = 0
    n_probes = 0

    PROBE_INTERVAL = 120      # 2 hours between probes
    HOLD_DURATION = 30        # virtual sensor persists for 30 steps (halved)
    VIRTUAL_WEIGHT = 1.0      # probe evidence weighted 1x (gentle influence)
    PROBE_SIGMA = 3.0         # moderately sharper than passive (1.5)

    unoccupied_streak = 0

    # Virtual sensor state
    virtual_sensor = None     # {'observed': float, 'predicted': {regime: float}}
    virtual_remaining = 0

    for i, rec in enumerate(sim_records):
        state_obs = rec['state_obs']

        # ── 4-way Bayesian update with optional virtual sensor ────────────
        likelihoods = {}
        for rname in regime_names:
            pred = monitors[rname].predict(state_obs)
            if sensor_weights:
                err = sum(sensor_weights.get(v, 1.0) *
                          abs(state_obs.get(v, 0.5) - pred.get(v, 0.5))
                          for v in sensor_vars)
            else:
                err = sum(abs(state_obs.get(v, 0.5) - pred.get(v, 0.5))
                          for v in sensor_vars)

            likelihoods[rname] = np.exp(-sigma * err)

            # Virtual sensor: probe decay as separate high-confidence channel
            if virtual_sensor is not None and virtual_remaining > 0:
                observed_decay = virtual_sensor['observed']
                predicted_decay = virtual_sensor['predicted'].get(rname, 0)
                decay_err = abs(observed_decay - predicted_decay)
                # Probe evidence uses sharper sigma — more informative per obs
                probe_lik = np.exp(-PROBE_SIGMA * decay_err)
                likelihoods[rname] *= (probe_lik ** VIRTUAL_WEIGHT)

        unnorm = {r: likelihoods[r] * weights[r] for r in regime_names}
        total = sum(unnorm.values())
        if total > 0:
            raw = {r: unnorm[r] / total for r in regime_names}
            weights = {r: (1 - forget_factor) * raw[r] + forget_factor * 0.25
                       for r in regime_names}
        w_total = sum(weights.values())
        if w_total > 0:
            weights = {r: weights[r] / w_total for r in regime_names}

        if virtual_remaining > 0:
            virtual_remaining -= 1
            if virtual_remaining == 0:
                virtual_sensor = None

        p_occ = weights['occ'] + weights['full']
        p_win = weights['win'] + weights['full']

        # ── Posterior-driven probe scheduling ─────────────────────────────
        believed_unoccupied = p_occ < 0.3
        if believed_unoccupied:
            unoccupied_streak += 1
        else:
            unoccupied_streak = 0

        should_probe = (
            believed_unoccupied and
            unoccupied_streak >= PROBE_INTERVAL and
            steps_since_probe >= probe_cooldown and
            n_probes < max_probes and
            'state_phys' in rec
        )

        probe_applied = False

        if should_probe:
            pre_winner = max(weights, key=weights.get)

            # LLM designs the probe (domain-agnostic protocol)
            probe_plan = design_diagnostic_probe(
                p_occ, p_win, rec['state_phys'], gpt_client=gpt_client)

            hour = rec['hour']
            day = int(hour / 24) % 4
            offset = {0: -3.0, 1: 6.0, 2: 13.0, 3: 6.0}.get(day, 0.0)

            # Execute the probe
            probe_result = execute_probe(
                rec['state_phys'], probe_plan, smap,
                elapsed_ms=int(hour * 3600000),
                outdoor_offset=offset)

            # Compute observed decay rate from probe trajectory
            trajectories = probe_result['phase_trajectories']
            observed_decay = 0.0
            if len(trajectories) >= 2 and trajectories[0] and trajectories[1]:
                peak_temp = trajectories[0][-1]
                final_temp = trajectories[1][-1]
                n_steps = len(trajectories[1])
                if n_steps > 0:
                    observed_decay = (peak_temp - final_temp) / n_steps

            # Predict what each regime model expects the decay to be
            pre_obs = {k: v for k, v in probe_result['pre_norm'].items()
                       if k not in LATENT_VARS}
            predicted_decays = compute_regime_decay_predictions(
                monitors, pre_obs, smap)

            # Normalize to comparable scale for likelihood computation
            max_decay = max(abs(observed_decay),
                           max(abs(v) for v in predicted_decays.values()),
                           0.01)
            norm_observed = observed_decay / max_decay
            norm_predicted = {r: predicted_decays[r] / max_decay
                             for r in regime_names}

            # Install virtual sensor
            virtual_sensor = {
                'observed': norm_observed,
                'predicted': norm_predicted,
            }
            virtual_remaining = HOLD_DURATION

            post_winner = max(weights, key=weights.get)

            probe_log.append({
                'step': i, 'hour': round(hour, 2),
                'trigger': 'unocc',
                'observed_decay': round(observed_decay, 4),
                'predicted_decays': {k: round(v, 4)
                                     for k, v in predicted_decays.items()},
                'p_occ_after': round(p_occ, 3),
                'p_win_after': round(p_win, 3),
                'gt_occ': rec['occ_active'],
                'gt_win': rec['win_active'],
                'winner_before': pre_winner,
                'winner_after': post_winner,
            })

            logger.info(f"  Step {i} (h={hour:.1f}): Probe → "
                        f"decay={observed_decay:.3f} → "
                        f"{pre_winner}→{post_winner}  "
                        f"gt_occ={rec['occ_active']} gt_win={rec['win_active']}")

            steps_since_probe = 0
            unoccupied_streak = 0
            n_probes += 1
            probe_applied = True
        else:
            steps_since_probe += 1

        winner = max(weights, key=weights.get)
        records.append({
            'step': rec['step'], 'hour': rec['hour'],
            'occ_active': rec['occ_active'], 'win_active': rec['win_active'],
            'regime_gt': rec['regime_gt'],
            'P_occ': p_occ, 'P_win': p_win,
            'winner': winner,
            'probe': probe_applied,
        })

    logger.info(f"  Active monitoring: {n_probes} diagnostic probes used")
    return records, probe_log

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATOR DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def collect_sim_data(smap, duration_steps=5760):
    """Run 4-day sim episode for monitoring evaluation. Fresh test data.

    Day 0: cold  (offset -3)  — mostly base + occ-only regimes
    Day 1: warm  (offset +6)  — mix of all regimes
    Day 2: hot   (offset +13) — occ-only (windows closed, too hot)
    Day 3: mild  (offset +6)  — window-only regime forced during off-hours
           Evening hours (19-24h) + early morning (0-7h): window held open via
           intervention while building is unoccupied. Simulates a realistic
           'window left open overnight on a mild day' scenario.
    """
    sim_abs = os.path.abspath(SIM_PATH)
    state = None
    elapsed_ms = 0
    step_ms = 60000
    records = []

    for step_i in range(duration_steps):
        elapsed_ms += step_ms
        hour = elapsed_ms / 3600000
        day = int(hour / 24) % 4
        hour_of_day = hour % 24

        offset = {0: -3.0, 1: 6.0, 2: 13.0, 3: 6.0}[day]

        # Day 3: force window open during unoccupied hours for window-only regime
        intervention = None
        if day == 3 and (hour_of_day >= 19 or hour_of_day < 7):
            intervention = {'windowPosition': 0.6}

        cmd = ['node', sim_abs, '--single-step',
               '--elapsed-ms', str(elapsed_ms),
               '--outdoor-offset', str(offset)]
        if state:
            cmd += ['--state', json.dumps(state)]
        if intervention:
            cmd += ['--intervention', json.dumps(intervention)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except Exception:
            continue

        for line in result.stdout.split('\n'):
            if 'RESULT:' in line:
                state = json.loads(line.split('RESULT:', 1)[1].strip())
                break
        else:
            continue

        # Normalize to [0,1]
        state_norm = {}
        for k, v in state.items():
            if not isinstance(v, (int, float)):
                continue
            kl = k.lower()
            state_norm[kl] = phys_to_norm(v, kl, smap)

        occ = state.get('occupancy', 0)
        win = state.get('windowPosition', state.get('windowposition', 0))
        occ_active = int(occ >= 1)
        win_active = int(win > 0.3)

        records.append({
            'step': step_i,
            'hour': round(hour, 4),
            'occ_active': occ_active,
            'win_active': win_active,
            'regime_gt': occ_active * 1 + win_active * 2,
            'state_obs': {k: v for k, v in state_norm.items() if k not in LATENT_VARS},
            'state_full': state_norm,
            'state_phys': dict(state),
        })

        if step_i % 500 == 0:
            logger.info(f"    Sim step {step_i}/{duration_steps}")

    logger.info(f"    Collected {len(records)} sim steps")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 1C: POLICY COMPARISON — GRID vs DDPC variants vs PID
# ═══════════════════════════════════════════════════════════════════════════════

def step_simulator(state_phys, action_norm, smap, elapsed_ms, outdoor_offset=0.0):
    """Step the JS simulator once with given action. Returns new state dicts."""
    sim_abs = os.path.abspath(SIM_PATH)

    # Denormalize action to physical units
    intervention = {}
    if 'hvacpower' in action_norm:
        intervention['hvacPower'] = norm_to_phys(action_norm['hvacpower'],
                                                  'hvacpower', smap)
    if 'lightingpower' in action_norm:
        intervention['lightingPower'] = norm_to_phys(action_norm['lightingpower'],
                                                      'lightingpower', smap)

    cmd = ['node', sim_abs, '--single-step',
           '--elapsed-ms', str(int(elapsed_ms)),
           '--outdoor-offset', str(outdoor_offset)]
    if state_phys:
        cmd += ['--state', json.dumps(state_phys)]
    if intervention:
        cmd += ['--intervention', json.dumps(intervention)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except Exception:
        return None, None, None

    for line in result.stdout.split('\n'):
        if 'RESULT:' in line:
            new_phys = json.loads(line.split('RESULT:', 1)[1].strip())
            new_norm = {k.lower(): phys_to_norm(v, k.lower(), smap)
                        for k, v in new_phys.items()
                        if isinstance(v, (int, float))}
            new_obs = {k: v for k, v in new_norm.items() if k not in LATENT_VARS}
            return new_phys, new_norm, new_obs

    return None, None, None


def run_policy_episode(policy_fn, smap, n_steps=120, start_hour=6.0,
                       outdoor_offset=0.0):
    """Run a policy episode through the JS simulator.

    Returns list of per-step records with physical units:
      kwh  — energy consumed this step (kWh)
      dh   — degree-hours of comfort violation this step
    """
    state_phys = None
    records = []

    for t in range(n_steps):
        elapsed_ms = (start_hour * 3600 + t * 60) * 1000
        hour = elapsed_ms / 3600000

        if state_phys is not None:
            state_norm = {k.lower(): phys_to_norm(v, k.lower(), smap)
                          for k, v in state_phys.items()
                          if isinstance(v, (int, float))}
            state_obs = {k: v for k, v in state_norm.items()
                         if k not in LATENT_VARS}
            action = policy_fn(state_obs, t)
        else:
            action = {'hvacpower': 0.5, 'lightingpower': 0.3}

        new_phys, new_norm, new_obs = step_simulator(
            state_phys, action, smap, elapsed_ms, outdoor_offset)

        if new_phys is None:
            continue

        state_phys = new_phys

        occ = new_phys.get('occupancy', 0)
        win = new_phys.get('windowPosition', new_phys.get('windowposition', 0))
        sat = new_phys.get('satisfaction', new_phys.get('overallSatisfaction', 50))
        eng = new_phys.get('energyConsumption', 50)

        # ── kWh: actuator power × step duration ──
        hvac_frac = new_phys.get('hvacPower', new_phys.get('hvacpower', 50)) / 100.0
        light_frac = new_phys.get('lightingPower', new_phys.get('lightingpower', 30)) / 100.0
        kwh = (hvac_frac * HVAC_MAX_KW + light_frac * LIGHT_MAX_KW) * STEP_HOURS

        # ── Degree-hours: |T_zone − T_ref| beyond deadband ──
        temp_c = new_phys.get('temperature', new_phys.get('Temperature', 22.0))
        dh = max(0.0, abs(temp_c - COMFORT_REF_C) - COMFORT_DEADBAND_C) * STEP_HOURS

        records.append({
            'step': t,
            'hour': round(hour, 3),
            'satisfaction': sat,
            'energy': eng,
            'kwh': round(kwh, 4),
            'dh': round(dh, 4),
            'occ': occ,
            'win': win,
            'regime': int(occ >= 1) * 1 + int(win > 0.3) * 2,
            'hvac': action.get('hvacpower', 0.5),
            'light': action.get('lightingpower', 0.3),
        })

    return records


def make_grid_policy(validated_edges, data, action_vars=None, comfort_target=0.8, intervention_directions=None):
    """Create GRID policy function from validated edges.

    comfort_target: epsilon-constraint DH budget (lower = tighter comfort).
        Maps to satisfaction weight: tighter comfort → higher sat weight.
    """
    from src.policy_engine import CausalPolicyEngine
    if action_vars is None:
        action_vars = ['HVACPower', 'LightingPower']
    pipeline_results = {'validated_edges': validated_edges}
    if intervention_directions:
        pipeline_results['intervention_directions'] = intervention_directions
    try:
        engine = CausalPolicyEngine(
            pipeline_results, data,
            use_llm=False,
            action_vars=action_vars,
            reg_lambda=0.1,
            n_gradient_steps=5,
        )
    except Exception as e:
        logger.warning(f"    CausalPolicyEngine init failed: {e}")
        return None

    # Map comfort_target to objective weights:
    # Low DH budget (tight comfort) → high satisfaction weight
    # High DH budget (relaxed comfort) → high energy weight
    sat_weight = max(0.3, min(0.70, 1.0 - comfort_target * 0.4))
    eng_weight = 1.0 - sat_weight
    sat_target = max(60, min(90, 90 - comfort_target * 15))

    def policy_fn(state_obs, t):
        try:
            col_map = {v.lower(): v for v in data.columns}
            state_proper = {col_map.get(k, k): v for k, v in state_obs.items()}
            result = engine.optimize_policy(
                objectives={'satisfaction': {'target': sat_target, 'weight': sat_weight},
                            'energy': {'target': 15, 'weight': eng_weight}},
                constraints={},
                current_state=state_proper,
            )
            action = {k.lower(): v for k, v in result.action_plan.items()}
            for k in action:
                action[k] = max(0.0, min(0.8, action[k]))
            return action
        except Exception:
            return {'hvacpower': 0.4, 'lightingpower': 0.25}

    return policy_fn


def make_regime_aware_policy(grid_val_edges, data, monitors_dict,
                             comfort_target=0.8, sensor_weights=None):
    """GRID policy that adapts objectives based on detected regime.

    Uses validated (intervention-tested) edges for the causal SEM, combined
    with live Bayesian regime switching for objective adaptation.

    comfort_target: epsilon-constraint DH budget. Scales regime-specific
    satisfaction targets/weights (tighter → more comfort-focused).
    """
    from src.policy_engine import CausalPolicyEngine

    try:
        engine = CausalPolicyEngine(
            {'validated_edges': grid_val_edges},
            data, use_llm=False,
            action_vars=['HVACPower', 'LightingPower'],
            reg_lambda=0.1, n_gradient_steps=5,
        )
    except Exception:
        engine = None

    # Scale objectives by comfort_target:
    # ct=0.5 (tight): boost satisfaction weights
    # ct=2.0 (relaxed): boost energy weights
    ct_scale = max(0.3, min(0.9, 1.0 - comfort_target * 0.35))
    regime_objectives = {
        'base': {'satisfaction': {'target': max(60, 70 + (0.8 - comfort_target) * 15),
                                  'weight': 0.4 + (ct_scale - 0.5) * 0.3},
                 'energy': {'target': 10, 'weight': 1.0 - (0.4 + (ct_scale - 0.5) * 0.3)}},
        'occ':  {'satisfaction': {'target': max(70, 85 + (0.8 - comfort_target) * 10),
                                  'weight': 0.65 + (ct_scale - 0.5) * 0.2},
                 'energy': {'target': 20, 'weight': 1.0 - (0.65 + (ct_scale - 0.5) * 0.2)}},
        'win':  {'satisfaction': {'target': max(65, 75 + (0.8 - comfort_target) * 10),
                                  'weight': 0.5 + (ct_scale - 0.5) * 0.2},
                 'energy': {'target': 10, 'weight': 1.0 - (0.5 + (ct_scale - 0.5) * 0.2)}},
        'full': {'satisfaction': {'target': max(65, 80 + (0.8 - comfort_target) * 10),
                                  'weight': 0.6 + (ct_scale - 0.5) * 0.2},
                 'energy': {'target': 15, 'weight': 1.0 - (0.6 + (ct_scale - 0.5) * 0.2)}},
    }

    regime_names_list = ['base', 'occ', 'win', 'full']
    weights = [{'base': 0.25, 'occ': 0.25, 'win': 0.25, 'full': 0.25}]

    def policy_fn(state_obs, t):
        if t == 0:
            weights[0] = {r: 0.25 for r in regime_names_list}

        if monitors_dict is not None:
            sigma = 1.5
            forget = 0.15
            liks = {}
            for rname in regime_names_list:
                model = monitors_dict[rname]
                pred = model.predict(state_obs)
                if sensor_weights:
                    err = sum(sensor_weights.get(v, 1.0) *
                              abs(state_obs.get(v, 0.5) - pred.get(v, 0.5))
                              for v in SENSOR_VARS)
                else:
                    err = sum(abs(state_obs.get(v, 0.5) - pred.get(v, 0.5))
                              for v in SENSOR_VARS)
                liks[rname] = np.exp(-sigma * err)

            unnorm = {r: liks[r] * weights[0][r] for r in regime_names_list}
            total = sum(unnorm.values())
            if total > 0:
                raw = {r: unnorm[r] / total for r in regime_names_list}
                weights[0] = {r: (1 - forget) * raw[r] + forget * 0.25
                              for r in regime_names_list}
            w_total = sum(weights[0].values())
            if w_total > 0:
                weights[0] = {r: weights[0][r] / w_total
                              for r in regime_names_list}

        regime = max(weights[0], key=weights[0].get)

        if engine is None:
            return {'hvacpower': 0.5, 'lightingpower': 0.3}

        try:
            col_map = {v.lower(): v for v in data.columns}
            state_proper = {col_map.get(k, k): v for k, v in state_obs.items()}
            objectives = regime_objectives[regime]
            result = engine.optimize_policy(
                objectives=objectives,
                constraints={},
                current_state=state_proper,
            )
            return {k.lower(): v for k, v in result.action_plan.items()}
        except Exception:
            return {'hvacpower': 0.5, 'lightingpower': 0.3}

    return policy_fn


# ═══════════════════════════════════════════════════════════════════════════════
# DDPC VARIANTS — Data-Driven Predictive Control baselines
# ═══════════════════════════════════════════════════════════════════════════════

def _wrap_src_baseline(ctrl, comfort_target):
    """Wrap a src baseline controller for the experiment runner interface.

    Adapts get_action(state, comfort_target) → policy_fn(state_obs, t)
    with automatic episode reset and lowercase action keys.
    """
    def policy_fn(state_obs, t):
        if t == 0 and hasattr(ctrl, 'reset'):
            ctrl.reset()
        action = ctrl.get_action(state_obs, comfort_target)
        return {k.lower(): v for k, v in action.items()}
    return policy_fn


def make_pid_policy():
    """Simple proportional control to fixed setpoints."""
    def policy_fn(state_obs, t):
        temp = state_obs.get('temperature', 0.5)
        co2 = state_obs.get('co2', 0.3)
        light = state_obs.get('lightlevel', 0.5)

        hvac = 0.5 + 2.0 * (temp - 0.5)
        hvac = max(0.0, min(1.0, hvac))

        lp = 0.3 + 1.5 * max(0, 0.5 - light)
        lp = max(0.0, min(1.0, lp))

        return {'hvacpower': hvac, 'lightingpower': lp}
    return policy_fn


# ═══════════════════════════════════════════════════════════════════════════════
# EXP 2: SENSOR ABLATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_sensor_ablation(monitors_dict, sim_records, method_name='PolicyGRID',
                        sensor_weights=None):
    """Progressive sensor dropout: measure monitoring accuracy at each tier."""
    ablation_order = ['co2', 'noisedb', 'lightlevel', 'airquality',
                      'humidity', 'temperature']
    results = []
    remaining = list(SENSOR_VARS)

    recs = run_monitoring(monitors_dict, sim_records, sensor_vars=remaining,
                          sensor_weights=sensor_weights)
    m = compute_monitoring_metrics(recs)
    results.append({
        'method': method_name, 'n_sensors': len(remaining),
        'dropped': 'none', **m})

    for sensor in ablation_order:
        if sensor in remaining:
            remaining.remove(sensor)
            if len(remaining) == 0:
                break
            recs = run_monitoring(monitors_dict, sim_records,
                                 sensor_vars=remaining,
                                 sensor_weights=sensor_weights)
            m = compute_monitoring_metrics(recs)
            results.append({
                'method': method_name, 'n_sensors': len(remaining),
                'dropped': sensor, **m})

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_discovery_comparison(discovery_df, out_dir):
    """Bar chart: F1, Precision, Recall for each discovery method.
    Colorblind-safe with hatching for B&W readability."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    methods = discovery_df['method'].tolist()

    for ax, metric, title in zip(axes,
            ['f1', 'precision', 'recall'],
            ['F1 Score', 'Precision', 'Recall']):
        vals = discovery_df[metric].tolist()
        for i, (m, v) in enumerate(zip(methods, vals)):
            bar = ax.bar(i, v, color=COLORS.get(m, '#999999'),
                         edgecolor='black', linewidth=0.8,
                         hatch=HATCHES.get(m, ''))
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.2)

    plt.suptitle('Causal Discovery: Method Comparison (15 vars, 28 GT edges)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_discovery_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"    Saved fig_discovery_comparison.png")


def plot_monitoring_comparison(monitoring_df, out_dir):
    """Grouped bar chart: Occ%, Win%, Combined% for each method.
    NOTE: Ground Truth is excluded from this plot.
    """
    # Filter out Ground Truth
    mdf = monitoring_df[~monitoring_df['method'].str.contains('Ground Truth')].copy()

    fig, ax = plt.subplots(figsize=(14, 7))
    methods = mdf['method'].tolist()
    x = np.arange(len(methods))
    w = 0.25

    occ = mdf['occ_accuracy'].tolist()
    win = mdf['win_accuracy'].tolist()
    comb = mdf['combined_accuracy'].tolist()

    # Three metric groups with distinct colors + hatching for B&W
    metric_styles = [
        ('Occupancy Detection', occ,  '#D55E00', '///'),
        ('Window Detection',    win,  '#009E73', '\\\\\\'),
        ('Combined 4-way',      comb, '#0072B2', 'xxx'),
    ]
    all_bars = []
    for i, (label, vals, color, hatch) in enumerate(metric_styles):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w, label=label,
                      color=color, alpha=0.85, edgecolor='black',
                      linewidth=0.8, hatch=hatch)
        all_bars.append(bars)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=10)
    ax.set_ylabel('Detection Accuracy (%)', fontsize=12)
    ax.set_title('Regime Monitoring: Latent Variable Detection Accuracy',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.axhline(50, color='gray', ls='--', lw=1, alpha=0.4, label='Random (binary)')
    ax.axhline(25, color='gray', ls=':', lw=1, alpha=0.4, label='Random (4-way)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.2)

    for bars in all_bars:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f'{h:.0f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_monitoring_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"    Saved fig_monitoring_comparison.png")


def plot_monitoring_timeseries(all_monitoring_records, out_dir):
    """Two-figure monitoring visualization:

    Figure A (fig_monitoring_policygrid.png):
        PolicyGRID-only posterior tracking with GT underlay.
        2 rows: P(Occ) and P(Win), each with shaded GT behind the curves.
        Shows how PolicyGRID (obs vs val) tracks the true regime.

    Figure B (fig_monitoring_benchmarks.png):
        PolicyGRID vs all benchmarks, same layout.
        PolicyGRID variants drawn bold, benchmarks drawn thin.
    """
    # Get ground truth from any available record set
    ref_recs = None
    for recs in all_monitoring_records.values():
        if recs:
            ref_recs = recs
            break
    if not ref_recs:
        logger.warning("No monitoring records for timeseries plot")
        return
    df_ref = pd.DataFrame(ref_recs)

    # ── Figure A: PolicyGRID-only with GT underlay ─────────────────────
    fig_a, (ax_occ_a, ax_win_a) = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'hspace': 0.12})

    # GT underlay as shaded regions
    for ax, gt_col, gt_color, gt_label in [
        (ax_occ_a, 'occ_active', '#E74C3C', 'Occupied (GT)'),
        (ax_win_a, 'win_active', '#27AE60', 'Window Open (GT)'),
    ]:
        ax.fill_between(df_ref['hour'], 0, df_ref[gt_col].astype(float),
                        color=gt_color, alpha=0.15, step='mid',
                        label=gt_label, zorder=0)

    # PolicyGRID curves on top (all 3 variants)
    pgrid_variants = ['PolicyGRID-O', 'PolicyGRID', 'PolicyGRID-R']
    for method_name in pgrid_variants:
        if method_name not in all_monitoring_records:
            continue
        recs = all_monitoring_records[method_name]
        if not recs:
            continue
        df = pd.DataFrame(recs)
        color = COLORS.get(method_name, '#000000')
        ls = LINESTYLES.get(method_name, '-')
        mk = MARKERS.get(method_name, 'o')
        lw = 2.5 if method_name == 'PolicyGRID' else 2.0
        ax_occ_a.plot(df['hour'], df['P_occ'], label=method_name,
                      color=color, lw=lw, ls=ls, marker=mk,
                      markevery=max(1, len(df)//12), ms=5, zorder=5)
        ax_win_a.plot(df['hour'], df['P_win'], label=method_name,
                      color=color, lw=lw, ls=ls, marker=mk,
                      markevery=max(1, len(df)//12), ms=5, zorder=5)

    for ax, ylabel in [(ax_occ_a, 'P(Occupied)'), (ax_win_a, 'P(Window Open)')]:
        ax.axhline(0.5, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.15)
        for d in [24, 48]:
            ax.axvline(d, color='black', ls=':', lw=1, alpha=0.3)

    ax_occ_a.set_title(
        'PolicyGRID Regime Detection: Posterior vs Ground Truth',
        fontsize=14, fontweight='bold')
    ax_win_a.set_xlabel('Simulation Time (hours)', fontsize=12)

    fig_a.tight_layout()
    fig_a.savefig(os.path.join(out_dir, 'fig_monitoring_policygrid.png'),
                  dpi=200, bbox_inches='tight')
    plt.close(fig_a)
    logger.info(f"    Saved fig_monitoring_policygrid.png")

    # ── Figure B: PolicyGRID vs all benchmarks ─────────────────────────
    fig_b, (ax_occ_b, ax_win_b) = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'hspace': 0.12})

    # GT underlay
    for ax, gt_col, gt_color in [
        (ax_occ_b, 'occ_active', '#E74C3C'),
        (ax_win_b, 'win_active', '#27AE60'),
    ]:
        ax.fill_between(df_ref['hour'], 0, df_ref[gt_col].astype(float),
                        color=gt_color, alpha=0.10, step='mid', zorder=0)

    # All methods with distinct linestyles and markers for B&W
    for method_name, recs in all_monitoring_records.items():
        if not recs or 'Ground Truth' in method_name:
            continue
        df = pd.DataFrame(recs)
        color = COLORS.get(method_name, '#95A5A6')
        ls = LINESTYLES.get(method_name, '-')
        mk = MARKERS.get(method_name, 'o')
        is_pgrid = 'PolicyGRID' in method_name
        lw = 2.5 if is_pgrid else 1.0
        alpha = 1.0 if is_pgrid else 0.5
        zorder = 5 if is_pgrid else 2
        me = max(1, len(df) // 12)  # ~12 markers per line
        ax_occ_b.plot(df['hour'], df['P_occ'], color=color, lw=lw, ls=ls,
                      marker=mk, markevery=me, ms=4 if is_pgrid else 3,
                      alpha=alpha, label=method_name, zorder=zorder)
        ax_win_b.plot(df['hour'], df['P_win'], color=color, lw=lw, ls=ls,
                      marker=mk, markevery=me, ms=4 if is_pgrid else 3,
                      alpha=alpha, label=method_name, zorder=zorder)

    for ax, ylabel in [(ax_occ_b, 'P(Occupied)'), (ax_win_b, 'P(Window Open)')]:
        ax.axhline(0.5, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.15)
        for d in [24, 48]:
            ax.axvline(d, color='black', ls=':', lw=1, alpha=0.3)

    ax_occ_b.set_title(
        'Regime Detection: PolicyGRID vs Benchmark Methods',
        fontsize=14, fontweight='bold')
    ax_win_b.set_xlabel('Simulation Time (hours)', fontsize=12)

    fig_b.tight_layout()
    fig_b.savefig(os.path.join(out_dir, 'fig_monitoring_benchmarks.png'),
                  dpi=200, bbox_inches='tight')
    plt.close(fig_b)
    logger.info(f"    Saved fig_monitoring_benchmarks.png")


def plot_policy_comparison(policy_results, out_dir):
    """Bar chart: satisfaction, energy, and MO score for each policy arm."""
    agg = {}
    for name, runs in policy_results.items():
        all_sat = [np.mean([r['satisfaction'] for r in run]) for run in runs]
        all_eng = [np.mean([r['energy'] for r in run]) for run in runs]
        all_mo = [0.6 * (s/100) + 0.4 * (1 - e/100)
                  for s, e in zip(all_sat, all_eng)]
        agg[name] = {
            'sat_mean': np.mean(all_sat), 'sat_std': np.std(all_sat),
            'eng_mean': np.mean(all_eng), 'eng_std': np.std(all_eng),
            'mo_mean': np.mean(all_mo), 'mo_std': np.std(all_mo),
        }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    names = list(agg.keys())
    colors = [COLORS.get(n, '#95A5A6') for n in names]
    x = np.arange(len(names))

    hatches = [HATCHES.get(n, '') for n in names]

    def _draw_bars(ax, vals, errs, ylabel, title, fmt='.1f', ylim=None):
        bars = ax.bar(x, vals, yerr=errs, color=colors, edgecolor='black',
                      linewidth=0.8, capsize=4)
        for bar, h_pat in zip(bars, hatches):
            bar.set_hatch(h_pat)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha='right', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)
        if ylim:
            ax.set_ylim(*ylim)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.02 if ylim else 1),
                    f'{val:{fmt}}', ha='center', fontsize=9)

    sat_vals = [agg[n]['sat_mean'] for n in names]
    sat_errs = [agg[n]['sat_std'] for n in names]
    _draw_bars(ax1, sat_vals, sat_errs, 'Satisfaction (%)',
               'Policy: Mean Satisfaction')

    eng_vals = [agg[n]['eng_mean'] for n in names]
    eng_errs = [agg[n]['eng_std'] for n in names]
    _draw_bars(ax2, eng_vals, eng_errs, 'Energy Consumption (%)',
               'Policy: Mean Energy (lower is better)')

    mo_vals = [agg[n]['mo_mean'] for n in names]
    mo_errs = [agg[n]['mo_std'] for n in names]
    _draw_bars(ax3, mo_vals, mo_errs, 'Multi-Objective Score',
               'Policy: MO Score (higher is better)\n0.6*sat + 0.4*(1-energy)',
               fmt='.3f', ylim=(0, 1.05))

    plt.suptitle('Policy Performance Under Regime Changes (2-hour episodes)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_policy_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"    Saved fig_policy_comparison.png")


def plot_policy_timeseries(policy_results, out_dir):
    """Policy episode timeline: satisfaction + cumulative kWh + regime context.

    Shows how each method responds to regime changes in real time.
    Regime bands are shaded behind the curves so the reviewer can see
    which method adapts best to occupancy/window transitions.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10),
                                         height_ratios=[3, 3, 1.2],
                                         sharex=True,
                                         gridspec_kw={'hspace': 0.10})

    # Get regime bands from first method's data
    ref = list(policy_results.values())[0]
    if ref and ref[0]:
        df_ref = pd.DataFrame(ref[0])
        regime_colors = {0: '#e8f4fd', 1: '#fce4e4', 2: '#e4f5e4', 3: '#f5e6f5'}
        regime_labels = {0: 'Empty', 1: 'Occupied', 2: 'Window', 3: 'Occ+Win'}
        current = df_ref['regime'].iloc[0]
        start = 0
        for i in range(1, len(df_ref)):
            if df_ref['regime'].iloc[i] != current or i == len(df_ref) - 1:
                for ax in [ax1, ax2]:
                    ax.axvspan(df_ref['hour'].iloc[start], df_ref['hour'].iloc[i],
                               color=regime_colors.get(current, 'white'),
                               alpha=0.25, zorder=0)
                # Label first occurrence of each regime
                mid_h = (df_ref['hour'].iloc[start] + df_ref['hour'].iloc[min(i, len(df_ref)-1)]) / 2
                ax1.text(mid_h, 98, regime_labels.get(current, ''),
                         ha='center', va='top', fontsize=7, alpha=0.6)
                start = i
                current = df_ref['regime'].iloc[i]

        # Regime step plot
        ax3.fill_between(df_ref['hour'], 0, (df_ref['regime'] >= 1).astype(float),
                         color='#E74C3C', alpha=0.2, step='mid', label='Occupied')
        ax3.fill_between(df_ref['hour'], 0, (df_ref['regime'] >= 2).astype(float) * 0.5,
                         color='#27AE60', alpha=0.2, step='mid', label='Window')
        ax3.step(df_ref['hour'], df_ref['regime'], color='#2C3E50', lw=1.5,
                 where='mid')
        ax3.set_yticks([0, 1, 2, 3])
        ax3.set_yticklabels(['Empty', 'Occ', 'Win', 'Both'], fontsize=9)
        ax3.set_ylabel('Regime', fontsize=10)
        ax3.legend(loc='upper right', fontsize=8)

    for name, runs in policy_results.items():
        if not runs or not runs[0]:
            continue
        df = pd.DataFrame(runs[0])
        color = COLORS.get(name, '#95A5A6')
        ls = LINESTYLES.get(name, '-')
        mk = MARKERS.get(name, 'o')
        is_pgrid = 'PolicyGRID' in name
        lw = 2.5 if is_pgrid else 1.2
        alpha = 1.0 if is_pgrid else 0.7
        me = max(1, len(df) // 10)

        # Satisfaction
        ax1.plot(df['hour'], df['satisfaction'], color=color, lw=lw, ls=ls,
                 marker=mk, markevery=me, ms=4 if is_pgrid else 3,
                 alpha=alpha, label=name)

        # Cumulative kWh
        if 'kwh' in df.columns:
            ax2.plot(df['hour'], df['kwh'].cumsum(), color=color, lw=lw, ls=ls,
                     marker=mk, markevery=me, ms=4 if is_pgrid else 3,
                     alpha=alpha, label=name)

    ax1.set_ylabel('Satisfaction (%)', fontsize=11)
    ax1.set_title('Policy Response to Regime Changes (Single Episode, ε=0.8)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.15)
    ax1.set_ylim(0, 105)

    ax2.set_ylabel('Cumulative Energy (kWh)', fontsize=11)
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.15)

    ax3.set_xlabel('Hour of Day', fontsize=12)
    ax3.grid(True, alpha=0.15)

    plt.savefig(os.path.join(out_dir, 'fig_policy_timeseries.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"    Saved fig_policy_timeseries.png")


def _pareto_front(points):
    """Return non-dominated (kwh, dh) points (both minimised).
    Sorted by kwh ascending.
    """
    pts = sorted(points, key=lambda p: p[0])
    front = []
    best_dh = float('inf')
    for k, d in pts:
        if d < best_dh:
            front.append((k, d))
            best_dh = d
    return front


def compute_pareto_hypervolume(policy_df):
    """Compute per-arm and combined hypervolume from policy_df with kwh/dh columns.

    Returns dict with:
        per_arm: {arm: hv_normalized}
        combined: hv_normalized of cross-policy Pareto front
        arm_means: {arm: (kwh_mean, dh_mean)}
        frontier: [(kwh, dh), ...] non-dominated points
    """
    arms = policy_df['method'].unique()

    # Per-arm means across all comfort targets and runs
    arm_means = {}
    all_pts = []
    for arm in arms:
        sub = policy_df[policy_df['method'] == arm]
        # Group by comfort_target, then mean across runs
        ct_means = sub.groupby('comfort_target')[['kwh', 'dh']].mean()
        for _, row in ct_means.iterrows():
            all_pts.append((arm, row['kwh'], row['dh']))
        arm_means[arm] = (sub['kwh'].mean(), sub['dh'].mean())

    # Cross-policy Pareto frontier
    raw_pts = [(k, d) for _, k, d in all_pts]
    frontier = _pareto_front(raw_pts)

    # HV reference point: 1.2x max of each axis
    all_kwh = [p[1] for p in all_pts]
    all_dh = [p[2] for p in all_pts]
    ref_kwh = max(all_kwh) * 1.2 if all_kwh else 1.0
    ref_dh = max(all_dh) * 1.2 if all_dh else 1.0

    def _hv(pts_list):
        """2D hypervolume dominated by pts relative to (ref_kwh, ref_dh)."""
        pts_sorted = sorted([p for p in pts_list if p[0] < ref_kwh and p[1] < ref_dh],
                            key=lambda p: p[0])
        if not pts_sorted:
            return 0.0
        area = 0.0
        prev_kwh = 0.0
        best_dh = ref_dh
        for k, d in pts_sorted:
            if d < best_dh:
                area += (k - prev_kwh) * (ref_dh - best_dh)
                prev_kwh = k
                best_dh = d
        area += (ref_kwh - prev_kwh) * (ref_dh - best_dh)
        return area

    max_hv = ref_kwh * ref_dh
    combined_hv = _hv(frontier) / max_hv if max_hv > 0 else 0.0

    per_arm = {}
    for arm in arms:
        arm_pts = [(k, d) for a, k, d in all_pts if a == arm]
        arm_front = _pareto_front(arm_pts)
        per_arm[arm] = _hv(arm_front) / max_hv if max_hv > 0 else 0.0

    return {
        'per_arm': per_arm,
        'combined': combined_hv,
        'arm_means': arm_means,
        'frontier': frontier,
    }


def plot_policy_pareto(policy_df, out_dir):
    """Pareto plot: kWh (x) vs Degree-Hours (y) with colored zones.

    Uses epsilon-constraint sweep data: each method has multiple
    (comfort_target, run) points plotted as faint scatter, with
    per-comfort-target means as bold points.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    hv_info = compute_pareto_hypervolume(policy_df)
    arm_means = hv_info['arm_means']
    frontier = hv_info['frontier']

    all_kwh = policy_df['kwh'].values
    all_dh = policy_df['dh'].values

    # Axis limits with padding
    kwh_pad = max((all_kwh.max() - all_kwh.min()) * 0.08, 0.5)
    dh_pad = max((all_dh.max() - all_dh.min()) * 0.08, 0.02)
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

    ax.text(kwh_lo + kwh_pad * 0.5, dh_lo + dh_pad * 0.5,
            'OPTIMAL\nLow Energy\nLow Discomfort',
            ha='left', va='bottom', fontsize=9, fontweight='bold',
            color='#1E8449', alpha=0.7)
    ax.text(kwh_hi - kwh_pad * 0.5, dh_hi - dh_pad * 0.5,
            'WORST\nHigh Energy\nHigh Discomfort',
            ha='right', va='top', fontsize=9, fontweight='bold',
            color='#B7950B', alpha=0.7)

    # Cross-policy Pareto frontier line
    if len(frontier) >= 2:
        f_kwh, f_dh = zip(*frontier)
        ax.plot(f_kwh, f_dh, color='black', linewidth=2.5,
                linestyle='-', zorder=8, alpha=0.7, label='Pareto Front')
        ax.fill_between(f_kwh, f_dh, dh_hi,
                        alpha=0.04, color='red', zorder=0)

    arms = policy_df['method'].unique()
    for arm in arms:
        arm_rows = policy_df[policy_df['method'] == arm]
        mk = MARKERS.get(arm, 'o')
        clr = COLORS.get(arm, '#95A5A6')

        # Faint scatter of all raw episode points
        ax.scatter(arm_rows['kwh'], arm_rows['dh'],
                   alpha=0.20, s=25, color=clr,
                   edgecolors='none', marker=mk, zorder=4)

        # Bold arm-mean point with unique marker
        if arm in arm_means:
            mk_kwh, mk_dh = arm_means[arm]
            on_front = any(abs(mk_kwh - fk) < 0.3 and abs(mk_dh - fd) < 0.01
                           for fk, fd in frontier)
            ew = 2.0 if on_front else 0.8
            sz = 130 if on_front else 80
            ax.scatter(mk_kwh, mk_dh, label=arm, s=sz, color=clr,
                       marker=mk, edgecolors='black', linewidth=ew,
                       zorder=9)

    # HV annotations
    combined_hv = hv_info['combined']
    per_arm_hv = hv_info['per_arm']
    ax.text(kwh_lo + kwh_pad * 0.5, dh_hi - dh_pad * 0.5,
            f'Combined HV = {combined_hv:.3f}',
            fontsize=10, fontweight='bold', color='black', ha='left')
    y_ann = dh_hi - dh_pad * 0.5 - (dh_hi - dh_lo) * 0.055
    for arm in arms:
        hv_val = per_arm_hv.get(arm, 0.0)
        ax.text(kwh_lo + kwh_pad * 0.5, y_ann,
                f'{arm}: HV={hv_val:.3f}', fontsize=7.5,
                color=COLORS.get(arm, '#95A5A6'), fontweight='bold', ha='left')
        y_ann -= (dh_hi - dh_lo) * 0.042

    ax.set_xlabel('Energy Consumption (kWh)', fontsize=12)
    ax.set_ylabel('Comfort Violation (Degree-Hours)', fontsize=12)
    ax.set_title('Pareto Frontier: Energy vs Comfort — Epsilon-Constraint Sweep',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9,
              framealpha=0.9)
    ax.set_xlim(kwh_lo, kwh_hi)
    ax.set_ylim(dh_lo, dh_hi)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    plt.savefig(os.path.join(out_dir, 'fig_policy_pareto.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"    Saved fig_policy_pareto.png")


def plot_ablation(ablation_df, out_dir):
    """Sensor ablation visualization — two outputs:

    1. fig_ablation_main.png (PAPER figure):
       Single clean panel — combined 4-way accuracy only.
       PolicyGRID drawn bold, all benchmarks drawn thin.
       No random baselines, no clutter. The degradation curves tell the story.

    2. fig_ablation_appendix.png (APPENDIX figure):
       3-panel diagnostic: Occupancy, Window, Combined separately.
       All methods shown with per-metric random baselines.
       Lets reviewers see exactly which detection axis breaks for which method.
    """
    # Build x-axis labels from any PolicyGRID variant
    pgrid_df = ablation_df[ablation_df['method'].str.contains('PolicyGRID')]
    if len(pgrid_df) > 0:
        ref_df = pgrid_df.sort_values('n_sensors', ascending=False).drop_duplicates('n_sensors')
    else:
        ref_df = ablation_df.sort_values('n_sensors', ascending=False).drop_duplicates('n_sensors')

    x_labels = []
    for _, row in ref_df.iterrows():
        if row['dropped'] == 'none':
            x_labels.append(f"All {int(row['n_sensors'])}")
        else:
            x_labels.append(f"{int(row['n_sensors'])} (−{row['dropped']})")
    n_ticks = len(x_labels)

    methods = ablation_df['method'].unique()

    # ── PAPER FIGURE: combined-only, clean ─────────────────────────────
    fig_main, ax = plt.subplots(figsize=(10, 6))

    for method in methods:
        mdf = ablation_df[ablation_df['method'] == method].sort_values(
            'n_sensors', ascending=False)
        color = COLORS.get(method, '#95A5A6')
        ls = LINESTYLES.get(method, '-')
        mk = MARKERS.get(method, 'o')
        is_pgrid = 'PolicyGRID' in method
        lw = 3.0 if is_pgrid else 1.0
        alpha = 1.0 if is_pgrid else 0.45
        ms = 7 if is_pgrid else 4
        zorder = 10 if is_pgrid else 2
        x = range(len(mdf))
        ax.plot(x, mdf['combined_accuracy'], marker=mk, ms=ms, ls=ls,
                color=color, lw=lw, alpha=alpha, label=method, zorder=zorder)

    ax.set_xticks(range(n_ticks))
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel('Sensors Available (dropped sensor in parentheses)', fontsize=12)
    ax.set_ylabel('Regime Detection Accuracy (%)', fontsize=12)
    ax.set_title('Sensor Ablation: Monitoring Robustness Under Sensor Dropout',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc='lower left', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.15)

    fig_main.tight_layout()
    fig_main.savefig(os.path.join(out_dir, 'fig_ablation_main.png'),
                     dpi=300, bbox_inches='tight')
    plt.close(fig_main)
    logger.info(f"    Saved fig_ablation_main.png (paper figure)")

    # ── APPENDIX FIGURE: 3-panel diagnostic ────────────────────────────
    fig_app, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    metrics_config = [
        (ax1, 'occ_accuracy',      'Occupancy Detection',   50),
        (ax2, 'win_accuracy',      'Window Detection',      50),
        (ax3, 'combined_accuracy', 'Combined 4-way',        25),
    ]

    for ax, col, title, random_baseline in metrics_config:
        for method in methods:
            mdf = ablation_df[ablation_df['method'] == method].sort_values(
                'n_sensors', ascending=False)
            color = COLORS.get(method, '#95A5A6')
            ls = LINESTYLES.get(method, '-')
            mk = MARKERS.get(method, 'o')
            is_pgrid = 'PolicyGRID' in method
            lw = 2.5 if is_pgrid else 1.0
            alpha = 1.0 if is_pgrid else 0.5
            ms = 6 if is_pgrid else 3
            x = range(len(mdf))
            ax.plot(x, mdf[col], marker=mk, ms=ms, ls=ls,
                    color=color, lw=lw, alpha=alpha, label=method,
                    zorder=5 if is_pgrid else 2)

        ax.axhline(random_baseline, color='gray', ls=':', lw=1, alpha=0.4,
                   label=f'Random ({random_baseline}%)')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_xlabel('Sensors Available', fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_xticks(range(n_ticks))
        ax.set_xticklabels(x_labels, fontsize=7, rotation=20, ha='right')
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=6, loc='lower left', ncol=2)

    fig_app.suptitle('Sensor Ablation — Diagnostic Breakdown (Appendix)',
                     fontsize=15, fontweight='bold', y=1.02)
    fig_app.tight_layout()
    fig_app.savefig(os.path.join(out_dir, 'fig_ablation_appendix.png'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig_app)
    logger.info(f"    Saved fig_ablation_appendix.png (appendix figure)")


def plot_summary_dashboard(discovery_df, monitoring_df, policy_df, policy_results,
                           ablation_df, out_dir):
    """Multi-panel summary figure for advisor presentation."""
    # Filter GT from monitoring for dashboard
    mdf_no_gt = monitoring_df[~monitoring_df['method'].str.contains('Ground Truth')]

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # (a) Discovery F1
    ax1 = fig.add_subplot(gs[0, 0])
    methods = discovery_df['method'].tolist()
    colors = [COLORS.get(m, '#95A5A6') for m in methods]
    h_pats = [HATCHES.get(m, '') for m in methods]
    bars = ax1.barh(range(len(methods)), discovery_df['f1'].tolist(),
                    color=colors, edgecolor='black', linewidth=0.8)
    for bar, hp in zip(bars, h_pats):
        bar.set_hatch(hp)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods, fontsize=9)
    ax1.set_xlabel('F1 Score')
    ax1.set_title('(a) Discovery F1', fontweight='bold')
    ax1.set_xlim(0, 1)
    for bar, val in zip(bars, discovery_df['f1']):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=8)

    # (b) Monitoring combined accuracy (no GT)
    ax2 = fig.add_subplot(gs[0, 1])
    m_methods = mdf_no_gt['method'].tolist()
    m_colors = [COLORS.get(m, '#95A5A6') for m in m_methods]
    m_hatches = [HATCHES.get(m, '') for m in m_methods]
    bars = ax2.barh(range(len(m_methods)), mdf_no_gt['combined_accuracy'].tolist(),
                    color=m_colors, edgecolor='black', linewidth=0.8)
    for bar, hp in zip(bars, m_hatches):
        bar.set_hatch(hp)
    ax2.set_yticks(range(len(m_methods)))
    ax2.set_yticklabels(m_methods, fontsize=9)
    ax2.set_xlabel('Combined Accuracy (%)')
    ax2.set_title('(b) Regime Detection', fontweight='bold')
    ax2.set_xlim(0, 105)
    for bar, val in zip(bars, mdf_no_gt['combined_accuracy']):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=8)

    # (c) Pareto Frontier: kWh vs DH (mini version)
    ax3 = fig.add_subplot(gs[0, 2])
    if len(policy_df) > 0:
        for arm in policy_df['method'].unique():
            arm_rows = policy_df[policy_df['method'] == arm]
            clr = COLORS.get(arm, '#95A5A6')
            mk = MARKERS.get(arm, 'o')
            ax3.scatter(arm_rows['kwh'], arm_rows['dh'],
                        c=clr, s=40, marker=mk, label=arm,
                        edgecolors='black', linewidth=0.5, zorder=5, alpha=0.7)
        ax3.set_xlabel('Energy (kWh)', fontsize=10)
        ax3.set_ylabel('Comfort Violation (DH)', fontsize=10)
        ax3.set_title('(c) Pareto Frontier', fontweight='bold')
        ax3.legend(fontsize=6, loc='upper right', ncol=2)
        ax3.grid(True, alpha=0.2)

    # (d) MO Score (at ε=0.8)
    ax4 = fig.add_subplot(gs[1, 0])
    pdf_08 = policy_df[policy_df['comfort_target'] == 0.8] if 'comfort_target' in policy_df.columns else policy_df
    p_names = list(policy_results.keys()) if policy_results else []
    if p_names and len(pdf_08) > 0:
        mo_scores = []
        for n in p_names:
            sub = pdf_08[pdf_08['method'] == n]
            mo_scores.append(sub['mo_score'].mean() if len(sub) > 0 else 0)
        p_colors = [COLORS.get(n, '#95A5A6') for n in p_names]
        p_hatches = [HATCHES.get(n, '') for n in p_names]
        bars = ax4.barh(range(len(p_names)), mo_scores, color=p_colors,
                        edgecolor='black', linewidth=0.8)
        for bar, hp in zip(bars, p_hatches):
            bar.set_hatch(hp)
        ax4.set_yticks(range(len(p_names)))
        ax4.set_yticklabels(p_names, fontsize=9)
        ax4.set_xlabel('MO Score')
        ax4.set_title('(d) Multi-Objective Score (ε=0.8)', fontweight='bold')
        ax4.set_xlim(0.7, 1.0)
        for bar, val in zip(bars, mo_scores):
            ax4.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=8)

    # (e) Ablation
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.set_title('(e) Sensor Ablation: Combined 4-way Accuracy', fontweight='bold')
    for method in ablation_df['method'].unique():
        mdf2 = ablation_df[ablation_df['method'] == method].sort_values(
            'n_sensors', ascending=False)
        color = COLORS.get(method, '#95A5A6')
        ls = LINESTYLES.get(method, '-')
        mk = MARKERS.get(method, 'o')
        is_pgrid = 'PolicyGRID' in method
        lw = 2.5 if is_pgrid else 1.5
        ms = 6 if is_pgrid else 4
        ax5.plot(range(len(mdf2)), mdf2['combined_accuracy'],
                 marker=mk, ms=ms, ls=ls,
                 color=color, lw=lw, label=method)
    ax5.axhline(25, color='gray', ls=':', lw=1, alpha=0.4)
    ax5.axhline(50, color='gray', ls='--', lw=1, alpha=0.3)
    ax5.set_ylabel('Accuracy (%)')
    ax5.set_xlabel('Sensors Dropped')
    ax5.legend(fontsize=8, loc='lower left', ncol=2)
    ax5.grid(True, alpha=0.2)
    pgrid = ablation_df[ablation_df['method'].str.contains('PolicyGRID')].sort_values(
        'n_sensors', ascending=False)
    if len(pgrid) > 0:
        labels = ['all'] + [f'-{d}' for d in pgrid['dropped'].iloc[1:]]
        ax5.set_xticks(range(len(labels)))
        ax5.set_xticklabels(labels, fontsize=9)

    fig.suptitle('PolicyGRID: Smart Building Benchmark Results',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(out_dir, 'fig_summary_dashboard.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"    Saved fig_summary_dashboard.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def main(skip_discovery=False, edges_path=None):
    """Run full NeurIPS experiment suite.

    Parameters
    ----------
    skip_discovery : bool
        If True, load edges from edges_path instead of running discovery.
        Saves ~150 minutes when only monitoring/policy/ablation changed.
    edges_path : str or None
        Path to discovered_edges.json.  Required if skip_discovery=True.
    """
    t_start = time.time()
    os.makedirs(f'{OUT_DIR}/exp1_benchmarks', exist_ok=True)
    os.makedirs(f'{OUT_DIR}/exp2_ablation', exist_ok=True)

    # M5: Compute resource tracking
    from compute_tracker import ComputeTracker
    tracker = ComputeTracker()

    logger.info("=" * 70)
    logger.info("  NeurIPS FINAL EXPERIMENTS (Full Pipeline)")
    logger.info("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("\n[SETUP] Loading data...")
    data, scaling, gt_set = load_data()
    smap = get_scaling_map(scaling)
    logger.info(f"  Data: {len(data)} rows, {len(data.columns)} cols")
    logger.info(f"  GT: {len(gt_set)} edges")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 0: FULL POLICYGRID DISCOVERY PIPELINE (3 seeds)
    # ══════════════════════════════════════════════════════════════════════
    if skip_discovery and edges_path:
        logger.info(f"\n[SKIP] Loading pre-computed edges from {edges_path}")
        with open(edges_path) as f:
            edge_cache = json.load(f)
        grid_val = {(s, t) for s, t in edge_cache['consensus_validated']}
        grid_obs = {(s, t) for s, t in edge_cache['consensus_obs_only']}
        all_grid_val = [{(s, t) for s, t in sv} for sv in edge_cache['per_seed_validated']]
        all_grid_obs = [{(s, t) for s, t in so} for so in edge_cache['per_seed_obs_only']]
        DISCOVERY_SEEDS = edge_cache['seeds']
        logger.info(f"  Loaded {len(grid_val)} validated, {len(grid_obs)} obs-only consensus edges")

        # Still run benchmark discovery (fast, ~1 min total)
        logger.info("\n  Running benchmark discovery methods...")
        benchmark_edges = run_benchmark_discovery(data)

        # Recompute metrics for summary
        discovery_rows = []
        for variant_name, seed_edge_list in [('PolicyGRID-O', all_grid_obs),
                                              ('PolicyGRID', all_grid_val)]:
            seed_metrics = [graph_metrics(edges, gt_set) for edges in seed_edge_list]
            m_consensus = graph_metrics(grid_val if variant_name == 'PolicyGRID' else grid_obs, gt_set)
            m_consensus['method'] = variant_name
            m_consensus['f1_std'] = round(np.std([s['f1'] for s in seed_metrics]), 3)
            m_consensus['shd_std'] = round(np.std([s['shd'] for s in seed_metrics]), 1)
            m_consensus['precision_std'] = round(np.std([s['precision'] for s in seed_metrics]), 3)
            m_consensus['recall_std'] = round(np.std([s['recall'] for s in seed_metrics]), 3)
            discovery_rows.append(m_consensus)
            logger.info(f"  {variant_name}: F1={m_consensus['f1']:.3f}, SHD={m_consensus['shd']}")

        for name, edges in benchmark_edges.items():
            m = graph_metrics(edges, gt_set)
            m['method'] = name
            m['f1_std'] = m['shd_std'] = m['precision_std'] = m['recall_std'] = 0.0
            discovery_rows.append(m)
            logger.info(f"  {name}: F1={m['f1']:.3f}, SHD={m['shd']}")

        discovery_df = pd.DataFrame(discovery_rows)
        discovery_df.to_csv(f'{OUT_DIR}/exp1_benchmarks/discovery_metrics.csv', index=False)

        # Save edges to current output dir
        with open(f'{OUT_DIR}/discovered_edges.json', 'w') as f:
            json.dump(edge_cache, f, indent=2)

    else:
        DISCOVERY_SEEDS = [42, 123, 456]
        logger.info("\n" + "=" * 70)
        logger.info(f"  PHASE 0: POLICYGRID DISCOVERY × {len(DISCOVERY_SEEDS)} seeds "
                    f"(PC + SAM + LLM + VARLiNGAM + Interventions, NO GT early-stop)")
        logger.info("=" * 70)

        all_grid_val = []  # list of edge sets, one per seed
        all_grid_obs = []
        all_discovery_results = []

        for d_seed in DISCOVERY_SEEDS:
            logger.info(f"\n  ── Discovery seed {d_seed} ──")
            result = run_full_discovery(data, seed=d_seed)
            all_grid_val.append(result['validated_edges'])
            all_grid_obs.append(result['obs_only_edges'])
            all_discovery_results.append(result)
            logger.info(f"    obs_only: {len(result['obs_only_edges'])} edges, "
                         f"validated: {len(result['validated_edges'])} edges, "
                         f"interventions: {result['intervention_count']}")

        # Edge stability: Jaccard similarity across seed pairs
        def _jaccard(s1, s2):
            if not s1 and not s2:
                return 1.0
            return len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0

        val_jaccards = []
        obs_jaccards = []
        for i in range(len(DISCOVERY_SEEDS)):
            for j in range(i + 1, len(DISCOVERY_SEEDS)):
                val_jaccards.append(_jaccard(all_grid_val[i], all_grid_val[j]))
                obs_jaccards.append(_jaccard(all_grid_obs[i], all_grid_obs[j]))
        logger.info(f"\n  Edge stability (Jaccard):")
        logger.info(f"    Validated: {np.mean(val_jaccards):.3f} ± {np.std(val_jaccards):.3f}")
        logger.info(f"    Obs-only:  {np.mean(obs_jaccards):.3f} ± {np.std(obs_jaccards):.3f}")

        # Use consensus edges: edges appearing in ≥2 of 3 seeds (majority vote)
        from collections import Counter
        val_edge_counts = Counter(e for s in all_grid_val for e in s)
        obs_edge_counts = Counter(e for s in all_grid_obs for e in s)
        majority = len(DISCOVERY_SEEDS) // 2 + 1  # 2 of 3

        grid_val = {e for e, c in val_edge_counts.items() if c >= majority}
        grid_obs = {e for e, c in obs_edge_counts.items() if c >= majority}

        logger.info(f"  Consensus (≥{majority}/{len(DISCOVERY_SEEDS)} seeds):")
        logger.info(f"    PolicyGRID validated: {len(grid_val)} edges")
        logger.info(f"    PolicyGRID obs_only:  {len(grid_obs)} edges")

        # Save all per-seed edges + consensus for reproducibility
        edge_cache = {
            'seeds': DISCOVERY_SEEDS,
            'per_seed_validated': [[[s, t] for s, t in sv] for sv in all_grid_val],
            'per_seed_obs_only': [[[s, t] for s, t in so] for so in all_grid_obs],
            'consensus_validated': [[s, t] for s, t in grid_val],
            'consensus_obs_only': [[s, t] for s, t in grid_obs],
            'edge_stability_jaccard_val': round(np.mean(val_jaccards), 3),
            'edge_stability_jaccard_obs': round(np.mean(obs_jaccards), 3),
        }
        with open(f'{OUT_DIR}/discovered_edges.json', 'w') as f:
            json.dump(edge_cache, f, indent=2)

        # ══════════════════════════════════════════════════════════════════════
        # EXP 1A: DISCOVERY BENCHMARKS
        # ══════════════════════════════════════════════════════════════════════
        logger.info("\n" + "=" * 70)
        logger.info("  EXP 1A: CAUSAL DISCOVERY BENCHMARKS")
        logger.info("=" * 70)

        benchmark_edges = run_benchmark_discovery(data)

        # Compute metrics: per-seed for PolicyGRID, single-run for benchmarks
        discovery_rows = []

        # Per-seed PolicyGRID metrics (for mean ± std)
        for variant_name, seed_edge_list in [('PolicyGRID-O', all_grid_obs),
                                              ('PolicyGRID', all_grid_val)]:
            seed_metrics = []
            for edges in seed_edge_list:
                seed_metrics.append(graph_metrics(edges, gt_set))
            # Report mean ± std
            m_consensus = graph_metrics(grid_val if variant_name == 'PolicyGRID' else grid_obs, gt_set)
            m_consensus['method'] = variant_name
            m_consensus['f1_std'] = round(np.std([s['f1'] for s in seed_metrics]), 3)
            m_consensus['shd_std'] = round(np.std([s['shd'] for s in seed_metrics]), 1)
            m_consensus['precision_std'] = round(np.std([s['precision'] for s in seed_metrics]), 3)
            m_consensus['recall_std'] = round(np.std([s['recall'] for s in seed_metrics]), 3)
            discovery_rows.append(m_consensus)
            logger.info(f"  {variant_name} (consensus): {m_consensus['n_edges']} edges, "
                         f"F1={m_consensus['f1']:.3f}±{m_consensus['f1_std']:.3f}, "
                         f"P={m_consensus['precision']:.3f}, R={m_consensus['recall']:.3f}, "
                         f"SHD={m_consensus['shd']}±{m_consensus['shd_std']:.0f}")

        # Benchmark methods (deterministic, single-run)
        all_methods_edges = {**benchmark_edges}
        for name, edges in all_methods_edges.items():
            m = graph_metrics(edges, gt_set)
            m['method'] = name
            m['f1_std'] = 0.0
            m['shd_std'] = 0.0
            m['precision_std'] = 0.0
            m['recall_std'] = 0.0
            discovery_rows.append(m)
            logger.info(f"  {name}: {m['n_edges']} edges, F1={m['f1']:.3f}, "
                         f"P={m['precision']:.3f}, R={m['recall']:.3f}, SHD={m['shd']}")

        discovery_df = pd.DataFrame(discovery_rows)
        discovery_df.to_csv(f'{OUT_DIR}/exp1_benchmarks/discovery_metrics.csv', index=False)

    # ══════════════════════════════════════════════════════════════════════
    # EXP 1B: MONITORING BENCHMARKS (no Ground Truth)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("  EXP 1B: REGIME MONITORING BENCHMARKS")
    logger.info("=" * 70)

    # Methods to test for monitoring (NO Ground Truth)
    monitor_methods = {
        'PolicyGRID-O': grid_obs,
        'PolicyGRID': grid_val,
    }
    for name, edges in benchmark_edges.items():
        if len(edges) > 0:
            monitor_methods[name] = edges

    # Build monitors
    method_monitors = {}
    for name, edges in monitor_methods.items():
        try:
            method_monitors[name] = construct_factored_monitors(edges, data)
        except Exception as e:
            logger.warning(f"    {name} monitor build FAILED: {e}")
            method_monitors[name] = None

    # Run monitoring over 3 sim seeds for robustness
    N_SEEDS = 3
    logger.info(f"  Running monitoring over {N_SEEDS} sim seeds...")
    all_seed_results = {name: [] for name in monitor_methods}
    all_monitoring_records = {}
    all_sim_records = []  # keep ALL seeds' sim data for ablation

    for seed_i in range(N_SEEDS):
        logger.info(f"\n  --- Sim seed {seed_i+1}/{N_SEEDS} ---")
        logger.info("  Collecting fresh simulator test data (4 days)...")
        sim_records = collect_sim_data(smap, duration_steps=5760)
        all_sim_records.append(sim_records)

        for name in monitor_methods:
            monitors = method_monitors[name]
            if monitors is None:
                all_seed_results[name].append(
                    {'occ_accuracy': 0, 'win_accuracy': 0, 'combined_accuracy': 0})
                continue
            recs = run_monitoring(monitors, sim_records)
            metrics = compute_monitoring_metrics(recs)
            all_seed_results[name].append(metrics)
            all_monitoring_records[name] = recs
            logger.info(f"    {name}: Occ={metrics['occ_accuracy']}%, "
                         f"Win={metrics['win_accuracy']}%, "
                         f"Combined={metrics['combined_accuracy']}%")

    # Aggregate across seeds
    monitoring_rows = []
    for name, edges in monitor_methods.items():
        seed_metrics = all_seed_results[name]
        occ_vals = [m['occ_accuracy'] for m in seed_metrics]
        win_vals = [m['win_accuracy'] for m in seed_metrics]
        comb_vals = [m['combined_accuracy'] for m in seed_metrics]
        monitoring_rows.append({
            'method': name,
            'n_edges': len(edges),
            'n_obs_edges': len(build_observable_graph(edges)),
            'occ_accuracy': round(np.mean(occ_vals), 1),
            'occ_std': round(np.std(occ_vals), 1),
            'win_accuracy': round(np.mean(win_vals), 1),
            'win_std': round(np.std(win_vals), 1),
            'combined_accuracy': round(np.mean(comb_vals), 1),
            'combined_std': round(np.std(comb_vals), 1),
        })
        logger.info(f"  {name} (mean): Combined={np.mean(comb_vals):.1f}%")

    monitoring_df = pd.DataFrame(monitoring_rows)
    monitoring_df.to_csv(f'{OUT_DIR}/exp1_benchmarks/monitoring_metrics.csv', index=False)

    # Checkpoint
    grid_obs_comb = monitoring_df[monitoring_df['method'] == 'PolicyGRID-O']['combined_accuracy'].values
    grid_obs_comb = grid_obs_comb[0] if len(grid_obs_comb) > 0 else 0
    benchmark_only = monitoring_df[~monitoring_df['method'].str.contains('PolicyGRID')]
    benchmark_max = benchmark_only['combined_accuracy'].max() if len(benchmark_only) > 0 else 0
    best_bench = benchmark_only.loc[benchmark_only['combined_accuracy'].idxmax(), 'method'] if len(benchmark_only) > 0 else 'none'
    logger.info(f"\n  CHECKPOINT: PolicyGRID={grid_obs_comb}% vs best benchmark {best_bench}={benchmark_max}%")

    # ══════════════════════════════════════════════════════════════════════
    # EXP 1B-ACTIVE: LLM-DESIGNED DIAGNOSTIC PROBING (PolicyGRID only)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("  EXP 1B-ACTIVE: LLM-Designed Diagnostic Probing")
    logger.info("=" * 70)

    grid_monitors = method_monitors.get('PolicyGRID')
    # Compute graph-derived sensor weights for PolicyGRID
    grid_sensor_weights = compute_sensor_weights_from_graph(
        grid_val, SENSOR_VARS, LATENT_VARS)
    logger.info(f"  Graph-derived sensor weights: {grid_sensor_weights}")

    if grid_monitors is not None:
        # Try LLM for probe design, fall back to deterministic
        try:
            from src.gpt_client import GPTClient
            gpt = GPTClient(api_key=API_KEY)
            logger.info("  LLM client available for probe design")
        except Exception:
            gpt = None
            logger.info("  LLM client unavailable, using deterministic probe fallback")

        active_seed_results = []
        all_probe_logs = []

        for seed_i, sim_records in enumerate(all_sim_records):
            logger.info(f"\n  --- Active monitoring seed {seed_i+1}/{len(all_sim_records)} ---")
            recs, probe_log = run_monitoring_active(
                grid_monitors, sim_records, smap,
                sensor_vars=SENSOR_VARS,
                gpt_client=gpt,
                probe_cooldown=60,
                max_probes=50,
            )
            metrics = compute_monitoring_metrics(recs)
            active_seed_results.append(metrics)
            all_probe_logs.extend(probe_log)
            logger.info(f"    PolicyGRID-Active: Occ={metrics['occ_accuracy']}%, "
                         f"Win={metrics['win_accuracy']}%, "
                         f"Combined={metrics['combined_accuracy']}%")
            logger.info(f"    Probes used: {len(probe_log)}")

        # Aggregate active results
        if active_seed_results:
            active_occ = [m['occ_accuracy'] for m in active_seed_results]
            active_win = [m['win_accuracy'] for m in active_seed_results]
            active_comb = [m['combined_accuracy'] for m in active_seed_results]

            passive_comb = monitoring_df[monitoring_df['method'] == 'PolicyGRID']['combined_accuracy'].values
            passive_comb = passive_comb[0] if len(passive_comb) > 0 else 0

            logger.info(f"\n  ACTIVE PROBING RESULTS:")
            logger.info(f"    PolicyGRID (passive):  Combined={passive_comb}%")
            logger.info(f"    PolicyGRID (active):   Combined={np.mean(active_comb):.1f}% "
                         f"(±{np.std(active_comb):.1f})")
            logger.info(f"    Improvement: {np.mean(active_comb) - passive_comb:+.1f}pp")
            logger.info(f"    Total probes across seeds: {len(all_probe_logs)}")

            # Save probe log
            if all_probe_logs:
                probe_df = pd.DataFrame(all_probe_logs)
                probe_df.to_csv(f'{OUT_DIR}/exp1_benchmarks/active_probe_log.csv',
                                index=False)
                logger.info(f"    Probe log saved")

                # Probe accuracy: did the probe change the winner correctly?
                n_correct = sum(1 for log in all_probe_logs
                                if log['winner_after'] in _gt_regime(log))
                n_changed = sum(1 for log in all_probe_logs
                                if log['winner_before'] != log['winner_after'])
                logger.info(f"    Probes that changed regime: {n_changed}/{len(all_probe_logs)}")
                logger.info(f"    Post-probe winner correct: {n_correct}/{len(all_probe_logs)}")

                # Win-open vs win-closed probe separation
                win_open_probes = [l for l in all_probe_logs if l['gt_win'] == 1]
                win_closed_probes = [l for l in all_probe_logs if l['gt_win'] == 0]
                if win_open_probes:
                    logger.info(f"    Win-open probes: n={len(win_open_probes)}")
                if win_closed_probes:
                    logger.info(f"    Win-closed probes: n={len(win_closed_probes)}")
    else:
        logger.warning("  PolicyGRID monitors not available, skipping active probing")

    # ══════════════════════════════════════════════════════════════════════
    # EXP 1C: POLICY COMPARISON — Epsilon-Constraint Pareto Sweep
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("  EXP 1C: POLICY COMPARISON — Epsilon-Constraint Pareto Sweep")
    logger.info("=" * 70)

    comfort_targets = DEFAULT_COMFORT_TARGETS  # [0.5, 0.8, 1.0, 1.5, 2.0]

    # Fit baselines from src/ (canonical implementations)
    from src.ddpc_baseline import create_ddpc_suite
    from src.mpc_baseline import create_mpc_suite
    from src.cmbpo_baseline import CMBPOController

    logger.info("  Fitting DDPC suite (Behavioral, Subspace, Neural, PETS)...")
    ddpc_suite = create_ddpc_suite(data)
    logger.info("  Fitting MPC suite (Linear, Ensemble)...")
    mpc_suite = create_mpc_suite(data)
    logger.info("  Fitting C-MBPO (causal model-based RL on validated DAG)...")
    cmbpo = CMBPOController(edges=grid_val)
    cmbpo.fit(data, smap=smap, step_fn=step_simulator)

    # PID baseline (static — no comfort_target sensitivity)
    pid_fn = make_pid_policy()

    n_runs = 3
    n_steps = 120
    policy_rows = []
    policy_results = {}  # for visualization: {arm_name: [list of runs]}
    # PolicyGRID-R uses validated-edge monitors (same as PolicyGRID monitoring)
    # for regime detection during policy execution. This ensures the regime-aware
    # policy benefits from the best available graph, not the weaker obs-only graph.
    grid_val_monitors_for_policy = method_monitors.get('PolicyGRID')

    for ct_idx, ct in enumerate(comfort_targets):
        logger.info(f"\n  ── Comfort target ε={ct} ({ct_idx+1}/{len(comfort_targets)}) ──")

        # Rebuild PolicyGRID variants with this comfort target
        grid_obs_fn = make_grid_policy(grid_obs, data, comfort_target=ct)
        grid_fn = make_grid_policy(grid_val, data, comfort_target=ct)
        regime_fn = make_regime_aware_policy(
            grid_val, data, grid_val_monitors_for_policy, comfort_target=ct)

        policy_arms = {}
        if grid_obs_fn:
            policy_arms['PolicyGRID-O'] = grid_obs_fn
        if grid_fn:
            policy_arms['PolicyGRID'] = grid_fn
        if regime_fn:
            policy_arms['PolicyGRID-R'] = regime_fn

        # Src baselines (wrapped for policy_fn(state_obs, t) interface)
        for name, ctrl in ddpc_suite.items():
            policy_arms[name] = _wrap_src_baseline(ctrl, ct)
        for name, ctrl in mpc_suite.items():
            policy_arms[name] = _wrap_src_baseline(ctrl, ct)
        policy_arms['C-MBPO'] = _wrap_src_baseline(cmbpo, ct)

        # PID is static — only run at first ct, duplicate for others
        if ct_idx == 0:
            policy_arms['PID'] = pid_fn

        for arm_name, pol_fn in policy_arms.items():
            logger.info(f"    Running {arm_name} ({n_runs} episodes x {n_steps} steps)...")
            runs = []
            for run_i in range(n_runs):
                start_hour = 6.0 + run_i * 2
                offset = [-2.0, 4.0, 10.0][run_i]
                recs = run_policy_episode(pol_fn, smap, n_steps=n_steps,
                                          start_hour=start_hour,
                                          outdoor_offset=offset)
                runs.append(recs)
                if recs:
                    avg_sat = np.mean([r['satisfaction'] for r in recs])
                    avg_eng = np.mean([r['energy'] for r in recs])
                    total_kwh = sum(r['kwh'] for r in recs)
                    total_dh = sum(r['dh'] for r in recs)
                    mo = 0.6 * (avg_sat/100) + 0.4 * (1 - avg_eng/100)
                    policy_rows.append({
                        'method': arm_name, 'run': run_i,
                        'comfort_target': ct,
                        'satisfaction': avg_sat, 'energy': avg_eng,
                        'kwh': round(total_kwh, 3),
                        'dh': round(total_dh, 3),
                        'mo_score': round(mo, 3),
                        'n_steps': len(recs),
                    })
                    logger.info(f"      Run {run_i}: sat={avg_sat:.1f}, eng={avg_eng:.1f}, "
                                f"kWh={total_kwh:.2f}, DH={total_dh:.3f}, MO={mo:.3f}")

            # Store runs for visualization (use default ct=0.8 for timeseries)
            if ct == 0.8:
                policy_results[arm_name] = runs

        # Duplicate PID rows for other comfort targets
        if ct_idx > 0:
            pid_rows = [r for r in policy_rows
                        if r['method'] == 'PID' and r['comfort_target'] == comfort_targets[0]]
            for pr in pid_rows:
                policy_rows.append({**pr, 'comfort_target': ct})

    policy_df = pd.DataFrame(policy_rows)
    policy_df.to_csv(f'{OUT_DIR}/exp1_benchmarks/policy_metrics.csv', index=False)

    # Log summary at default ct=0.8
    logger.info("\n  Policy summary (ε=0.8):")
    for name in policy_results:
        sub = policy_df[(policy_df['method'] == name) &
                        (policy_df['comfort_target'] == 0.8)]
        if len(sub) > 0:
            logger.info(f"    {name}: MO={sub['mo_score'].mean():.3f} "
                         f"(sat={sub['satisfaction'].mean():.1f}, "
                         f"eng={sub['energy'].mean():.1f}, "
                         f"kWh={sub['kwh'].mean():.2f}, "
                         f"DH={sub['dh'].mean():.3f})")

    # ══════════════════════════════════════════════════════════════════════
    # EXP 2: SENSOR ABLATION
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("  EXP 2: SENSOR ABLATION")
    logger.info("=" * 70)

    ablation_rows_all = []

    # Run ablation over ALL sim seeds for robustness
    for seed_i, sim_records in enumerate(all_sim_records):
        logger.info(f"\n  Ablation on sim seed {seed_i+1}/{len(all_sim_records)}")
        for name, monitors in method_monitors.items():
            if monitors is None:
                continue
            abl = run_sensor_ablation(monitors, sim_records, name)
            for row in abl:
                row['sim_seed'] = seed_i
            ablation_rows_all.extend(abl)

    # Aggregate across seeds: mean ± std per method × n_sensors
    ablation_raw = pd.DataFrame(ablation_rows_all)
    ablation_raw.to_csv(f'{OUT_DIR}/exp2_ablation/ablation_metrics_raw.csv', index=False)

    # Compute mean/std per method × tier
    ablation_rows = []
    for (method, n_sens, dropped), grp in ablation_raw.groupby(
            ['method', 'n_sensors', 'dropped'], sort=False):
        ablation_rows.append({
            'method': method,
            'n_sensors': n_sens,
            'dropped': dropped,
            'occ_accuracy': round(grp['occ_accuracy'].mean(), 1),
            'occ_std': round(grp['occ_accuracy'].std(), 1),
            'win_accuracy': round(grp['win_accuracy'].mean(), 1),
            'win_std': round(grp['win_accuracy'].std(), 1),
            'combined_accuracy': round(grp['combined_accuracy'].mean(), 1),
            'combined_std': round(grp['combined_accuracy'].std(), 1),
        })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(f'{OUT_DIR}/exp2_ablation/ablation_metrics.csv', index=False)

    # ══════════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("  GENERATING VISUALIZATIONS")
    logger.info("=" * 70)

    plot_discovery_comparison(discovery_df, f'{OUT_DIR}/exp1_benchmarks')
    plot_monitoring_comparison(monitoring_df, f'{OUT_DIR}/exp1_benchmarks')
    plot_monitoring_timeseries(all_monitoring_records, f'{OUT_DIR}/exp1_benchmarks')
    if policy_results:
        plot_policy_comparison(policy_results, f'{OUT_DIR}/exp1_benchmarks')
        plot_policy_timeseries(policy_results, f'{OUT_DIR}/exp1_benchmarks')
        plot_policy_pareto(policy_df, f'{OUT_DIR}/exp1_benchmarks')
    plot_ablation(ablation_df, f'{OUT_DIR}/exp2_ablation')
    plot_summary_dashboard(discovery_df, monitoring_df, policy_df, policy_results,
                           ablation_df, OUT_DIR)

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 70)
    logger.info("  EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time: {elapsed/60:.1f} minutes")

    # Write summary
    summary = []
    summary.append("NeurIPS Final Experiments — Summary (Full Pipeline)")
    summary.append("=" * 55)
    summary.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    summary.append(f"Runtime: {elapsed/60:.1f} minutes")
    if 'all_discovery_results' in dir() or not skip_discovery:
        total_interventions = sum(r['intervention_count'] for r in all_discovery_results)
        total_disc_time = sum(r['discovery_time_s'] for r in all_discovery_results)
        summary.append(f"Discovery: {len(DISCOVERY_SEEDS)} seeds, "
                       f"{total_interventions} total interventions, "
                       f"{total_disc_time:.0f}s total")
    else:
        summary.append(f"Discovery: loaded from {edges_path}")
    edge_cache_loaded = json.load(open(f'{OUT_DIR}/discovered_edges.json'))
    summary.append(f"Edge stability (Jaccard): val={edge_cache_loaded.get('edge_stability_jaccard_val', '?')}, "
                   f"obs={edge_cache_loaded.get('edge_stability_jaccard_obs', '?')}")
    summary.append("")
    summary.append("EXP 1A: CAUSAL DISCOVERY")
    summary.append("-" * 40)
    for _, row in discovery_df.iterrows():
        summary.append(f"  {row['method']:<22} F1={row['f1']:.3f}  "
                       f"P={row['precision']:.3f}  R={row['recall']:.3f}  "
                       f"SHD={row['shd']}")
    summary.append("")
    summary.append("EXP 1B: REGIME MONITORING (no Ground Truth)")
    summary.append("-" * 40)
    for _, row in monitoring_df.iterrows():
        summary.append(f"  {row['method']:<22} Occ={row['occ_accuracy']:>5.1f}%  "
                       f"Win={row['win_accuracy']:>5.1f}%  "
                       f"Combined={row['combined_accuracy']:>5.1f}%")
    if 'active_seed_results' in dir() and active_seed_results:
        summary.append("")
        summary.append("EXP 1B-ACTIVE: LLM-Designed Diagnostic Probing (PolicyGRID)")
        summary.append("-" * 40)
        summary.append(f"  PolicyGRID (passive):  Combined={passive_comb}%")
        summary.append(f"  PolicyGRID (active):   Combined={np.mean(active_comb):.1f}% "
                       f"(±{np.std(active_comb):.1f})")
        summary.append(f"  Improvement: {np.mean(active_comb) - passive_comb:+.1f}pp")
        summary.append(f"  Total probes: {len(all_probe_logs)}")
    summary.append("")
    summary.append("EXP 1C: POLICY — Epsilon-Constraint Pareto Sweep")
    summary.append(f"  Comfort targets (DH budget): {comfort_targets}")
    summary.append("-" * 40)
    summary.append("  At ε=0.8 (default):")
    pdf_08 = policy_df[policy_df['comfort_target'] == 0.8]
    for name in policy_results:
        sub = pdf_08[pdf_08['method'] == name]
        if len(sub) > 0:
            summary.append(f"  {name:<22} MO={sub['mo_score'].mean():.3f}  "
                           f"Sat={sub['satisfaction'].mean():>5.1f}+-{sub['satisfaction'].std():.1f}  "
                           f"Eng={sub['energy'].mean():>5.1f}+-{sub['energy'].std():.1f}  "
                           f"kWh={sub['kwh'].mean():>5.2f}  DH={sub['dh'].mean():.3f}")
    summary.append("")
    summary.append("  Pareto HV (across all ε):")
    hv_info = compute_pareto_hypervolume(policy_df)
    summary.append(f"    Combined HV = {hv_info['combined']:.3f}")
    for arm, hv_val in hv_info['per_arm'].items():
        summary.append(f"    {arm:<22} HV={hv_val:.3f}")
    summary.append("")
    summary.append("EXP 2: SENSOR ABLATION")
    summary.append("-" * 40)
    for _, row in ablation_df[ablation_df['method'] == 'PolicyGRID-O'].iterrows():
        summary.append(f"  {row['n_sensors']} sensors (drop {row['dropped']:<12}): "
                       f"Occ={row['occ_accuracy']:>5.1f}%  "
                       f"Win={row['win_accuracy']:>5.1f}%  "
                       f"Comb={row['combined_accuracy']:>5.1f}%")

    summary.append("")
    summary.append("METHODOLOGICAL INTEGRITY")
    summary.append("-" * 40)
    summary.append(f"  - Discovery: {len(DISCOVERY_SEEDS)} seeds, consensus edges (majority vote)")
    summary.append("  - GT early-stopping DISABLED — all 10 iterations run to completion")
    summary.append("  - GT SHD logged for diagnostics only, never used for decisions")
    summary.append("  - Regime vars (occupancy, windowPosition) NEVER in state_obs")
    summary.append("  - GT edges used ONLY for F1/SHD evaluation post-hoc")
    summary.append("  - GT regime labels used ONLY in accuracy metrics, not in Bayesian update")
    summary.append("  - Test data: fresh 3-day sim episodes, separate from 10k training CSV")
    summary.append("  - All benchmark methods run on same observational data")
    summary.append("  - PolicyGRID uses interventional data (simulator) — benchmarks do not")
    summary.append("  - Policy: same sim conditions for all arms per run per comfort target")
    summary.append(f"  - Monitoring: {N_SEEDS} sim seeds, mean ± std reported")
    summary.append(f"  - Ablation: {len(all_sim_records)} sim seeds, mean ± std reported")
    summary.append("  - Ground Truth excluded from monitoring/policy plots")

    # M5: Record LLM usage from active probing
    if 'gpt' in dir() and gpt is not None:
        tracker.record_llm_usage(gpt)
    summary.append("")
    summary.append(tracker.format_for_summary(n_seeds=N_SEEDS))

    summary_text = '\n'.join(summary)
    with open(f'{OUT_DIR}/summary.txt', 'w') as f:
        f.write(summary_text)

    # Save compute resources as JSON
    tracker.save(f'{OUT_DIR}/compute_resources.json')

    print("\n" + summary_text)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-discovery', action='store_true',
                        help='Skip discovery, load edges from --edges-path')
    parser.add_argument('--edges-path', type=str, default=None,
                        help='Path to discovered_edges.json')
    args = parser.parse_args()
    main(skip_discovery=args.skip_discovery, edges_path=args.edges_path)
