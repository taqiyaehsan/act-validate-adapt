#!/usr/bin/env python3
"""
PolicyGRID Physical Discovery
==============================
Runs iterative causal discovery on the physical room using the testbed's IoT drivers.
Matches the simulation pipeline's iterative loop: each iteration tests edges,
appends intervention data to D_obs, regenerates hypotheses, and repeats.

Phase 1: Hypothesis generation (PC, SAM, VARLiNGAM) on baseline CSV — no hardware
Phase 2: Iterative validation loop:
         test edges → append intervention data → regenerate hypotheses → repeat
Phase 3: Rescue non-intervenable edges via partial correlation

Usage:
    # Full iterative discovery (10 iterations, same as sim)
    python run_policygrid_discovery.py

    # Hypothesis gen only (no hardware, fast)
    python run_policygrid_discovery.py --skip-validation

    # Custom iteration count
    python run_policygrid_discovery.py --max-iterations 5

    # Multiple seeds (seed 1 does hardware, seeds 2+ reuse measurements)
    python run_policygrid_discovery.py --seeds 3

Prerequisites:
    - BME680 server running (python drivers/bme_server/server.py)
    - Room logger running (python drivers/room_logger/server.py)
    - Kasa devices on network
    - Baseline data at data/baseline_data.csv (from the data-preprocessing script)
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import atexit
import signal
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

# ── Setup paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, SCRIPT_DIR)        # for policygrid_adapter, drivers
sys.path.insert(0, PROJECT_ROOT)       # for src/ (PolicyGRID core)

from policygrid_adapter import (
    read_all_sensors, device_on, device_off, all_off, cleanup, get_device_states
)

logger = logging.getLogger(__name__)

# ── Config ──
BASELINE_CSV = os.path.join(SCRIPT_DIR, 'data', 'baseline_data.csv')
SCALING_CSV = os.path.join(SCRIPT_DIR, 'data', 'baseline_data_scaling.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'discovery')

ACTUATORS = ['Heater', 'Humidifier', 'Fan']
NON_INTERVENABLE = ['Temperature', 'Humidity', 'AirQuality',
                     'EnergyConsumption', 'Satisfaction',
                     'Occupancy', 'WindowState']
GRAPH_VARS = ACTUATORS + ['Temperature', 'Humidity', 'AirQuality',
                           'EnergyConsumption', 'Satisfaction',
                           'Occupancy', 'WindowState']

# Timing (seconds)
# Calibrated 2026-04-08: onset + 50% buffer from calibrate_actuators.py
HOLD_TIMES = {'Heater': 22 * 60, 'Humidifier': 31 * 60, 'Fan': 24 * 60}
WASHOUT_SEC = 15 * 60
MEASUREMENT_INTERVAL = 60  # 1 min between sensor reads

# Statistical thresholds
SCREENING_TESTS = 3
CONFIRMATORY_TESTS = 4
MAX_ITERATIONS = 10        # Outer loop iterations (same as sim pipeline)
VALIDATION_ALPHA = 0.10    # Liberal for small physical samples
RESCUE_TAU = 0.4           # Partial correlation threshold

# Effect size thresholds (in normalized [0,1] units)
# Based on sensor noise floor, NOT calibrated from observed effects.
# Within-session pre-measurements (devices OFF) show near-zero variation
# for Govee temp/humidity (resolution-limited at 0.1°C / 0.1%).
# Thresholds set to sensor resolution / data range — the minimum
# detectable change. The t-test across 3+4 trials decides significance;
# these thresholds only screen out zero-effect edges to save time.
#
# Sensor noise (from calibration pre-measurements, 2026-04-08):
#   Temperature:  within-session σ=0.00  (Govee resolution 0.1°C)
#   Humidity:     within-session σ=0.00  (Govee resolution 0.1%)
#   AirQuality:   within-session σ=0.18  (BME680 IAQ fluctuates)
#   Energy:       within-session σ=0.00  (Kasa: exact when OFF)
#   Satisfaction: within-session σ=0.00  (derived from T+H)
EFFECT_THRESHOLDS = {
    'temperature': 0.01,        # ~0.09°C — just above sensor resolution
    'humidity': 0.01,           # ~0.14%  — just above sensor resolution
    'airquality': 0.01,         # ~0.65 IAQ — above 3σ noise (0.008)
    'energyconsumption': 0.005, # ~2.8W — any device draws more
    'satisfaction': 0.01,       # ~0.5 points — derived, follows T+H
}

# Proposed GT for internal F1 (not used in decisions)
PROPOSED_GT = {
    ('Heater', 'Temperature'), ('Heater', 'EnergyConsumption'),
    ('Humidifier', 'Humidity'), ('Humidifier', 'EnergyConsumption'),
    ('Fan', 'Temperature'), ('Fan', 'AirQuality'), ('Fan', 'EnergyConsumption'),
    ('Occupancy', 'Temperature'), ('Occupancy', 'Humidity'),
    ('WindowState', 'Temperature'), ('WindowState', 'Humidity'),
    ('WindowState', 'AirQuality'),
    ('Temperature', 'Satisfaction'), ('Humidity', 'Satisfaction'),
}


# ── Safety: always turn everything off on exit ──
def _emergency_off():
    try:
        all_off()
    except:
        pass

atexit.register(_emergency_off)
signal.signal(signal.SIGINT, lambda s, f: (all_off(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (all_off(), sys.exit(0)))


# ── Scaling helpers ──
def load_scaling():
    if os.path.exists(SCALING_CSV):
        df = pd.read_csv(SCALING_CSV)
        return {row['column'].lower(): (row['data_min'], row['data_max'])
                for _, row in df.iterrows()}
    return {}

def to_normalized(value, var, smap):
    if var.lower() in smap:
        mn, mx = smap[var.lower()]
        return (value - mn) / (mx - mn + 1e-12)
    return value

def to_physical(value, var, smap):
    if var.lower() in smap:
        mn, mx = smap[var.lower()]
        return mn + value * (mx - mn)
    return value


# ── Hypothesis generation ──
def run_hypothesis_generation(data):
    """Run PC, SAM, VARLiNGAM on baseline CSV. Returns union candidates + scores."""
    from src.generators import PCGenerator

    cols = [c for c in data.columns if c in GRAPH_VARS]
    # Drop NaN rows (can occur in augmented data from incomplete intervention states)
    clean_data = data[cols].dropna()
    if len(clean_data) < len(data):
        logger.info(f"  Dropped {len(data) - len(clean_data)} NaN rows for hypothesis gen")
    logger.info(f"Hypothesis generation on {len(cols)} variables, {len(clean_data)} rows: {cols}")

    # Case-insensitive name normalization (VARLiNGAM lowercases everything)
    name_map = {c.lower(): c for c in cols}

    def _normalize_edges(raw_edges):
        """Map edge variable names back to canonical column names."""
        normalized = []
        for s, t in raw_edges:
            s_norm = name_map.get(s.lower(), s)
            t_norm = name_map.get(t.lower(), t)
            if s_norm in cols and t_norm in cols:
                normalized.append((s_norm, t_norm))
        return normalized

    all_edges = []
    method_edges = {}

    # PC
    try:
        logger.info("  Running PC...")
        pc = PCGenerator(relevant_columns=cols)
        result = pc.generate(clean_data)
        edges = _normalize_edges(result.get('edges', []))
        method_edges['PC'] = edges
        all_edges.extend(edges)
        logger.info(f"  PC: {len(edges)} edges")
    except Exception as e:
        logger.warning(f"  PC failed: {e}")

    # SAM
    try:
        logger.info("  Running SAM...")
        from src.generators import SAMGenerator
        sam = SAMGenerator(relevant_columns=cols, allow_cycles=True)
        result = sam.generate(clean_data)
        edges = _normalize_edges(result.get('edges', []))
        method_edges['SAM'] = edges
        all_edges.extend(edges)
        logger.info(f"  SAM: {len(edges)} edges")
    except Exception as e:
        logger.warning(f"  SAM failed: {e}")

    # VARLiNGAM
    try:
        logger.info("  Running VARLiNGAM...")
        from src.generators import VARLiNGAMGenerator
        vl = VARLiNGAMGenerator(max_lags=3, criterion='bic', threshold=0.05)
        result = vl.generate(clean_data)
        edges = _normalize_edges(result.get('edges', []))
        method_edges['VARLiNGAM'] = edges
        all_edges.extend(edges)
        logger.info(f"  VARLiNGAM: {len(edges)} edges")
    except Exception as e:
        logger.warning(f"  VARLiNGAM failed: {e}")

    # Confidence scores (cross-method agreement)
    unique_edges = list(set(all_edges))
    n_methods = len(method_edges)
    scores = {}
    for edge in unique_edges:
        count = sum(1 for m, edges in method_edges.items() if edge in edges)
        scores[edge] = count / max(n_methods, 1)

    # Sort ascending confidence (lowest first = test first, highest info gain)
    sorted_edges = sorted(scores.keys(), key=lambda e: scores[e])

    logger.info(f"Union: {len(sorted_edges)} candidate edges from {n_methods} methods")
    return sorted_edges, scores, method_edges


# ── Physical intervention ──
def run_physical_intervention(device, smap, turn_on=True):
    """
    Run one do-operator intervention on a physical device.

    Protocol (matches sim pipeline tester.py):
      1. Pre-measurement: 3 readings of ALL sensors → pre-state snapshot
      2. do(device=ON or OFF)
      3. Hold for device-specific duration
      4. Post-measurement: 3 readings of ALL sensors → post-state snapshot
      5. do(device=OFF)  [cleanup]
      6. Compute per-variable effect = (post_mean - pre_mean) for every sensor

    Returns:
        dict with keys:
        - 'pre_state': {var: normalized_mean, ...}  (all variables)
        - 'post_state': {var: normalized_mean, ...}  (all variables)
        - 'effects': {var: delta, ...}               (per-variable deltas)
        - 'device': str, 'action': 'on'/'off'
        Returns None on sensor failure.
    """
    action = 'on' if turn_on else 'off'

    # Pre-measurement (3 readings, 1 min apart)
    pre_states = []
    for _ in range(3):
        state = read_all_sensors()
        row = {}
        for var in GRAPH_VARS:
            val = state.get(var)
            if val is not None:
                row[var] = to_normalized(val, var, smap)
        if row:
            pre_states.append(row)
        time.sleep(MEASUREMENT_INTERVAL)

    if not pre_states:
        return None

    # Average pre-state
    pre_avg = {}
    for var in GRAPH_VARS:
        vals = [s[var] for s in pre_states if var in s]
        if vals:
            pre_avg[var] = float(np.mean(vals))

    # Intervention: do(device=ON) or do(device=OFF)
    if turn_on:
        device_on(device)
    else:
        device_off(device)
    hold = HOLD_TIMES.get(device, 15 * 60)
    logger.info(f"    do({device}={action.upper()}), hold {hold // 60} min...")
    time.sleep(hold)

    # Post-measurement (3 readings, 1 min apart)
    post_states = []
    for _ in range(3):
        state = read_all_sensors()
        row = {}
        for var in GRAPH_VARS:
            val = state.get(var)
            if val is not None:
                row[var] = to_normalized(val, var, smap)
        if row:
            post_states.append(row)
        time.sleep(MEASUREMENT_INTERVAL)

    # Cleanup: device OFF
    device_off(device)

    if not post_states:
        return {'pre_state': pre_avg, 'post_state': {}, 'effects': {},
                'device': device, 'action': action}

    post_avg = {}
    for var in GRAPH_VARS:
        vals = [s[var] for s in post_states if var in s]
        if vals:
            post_avg[var] = float(np.mean(vals))

    # Per-variable effects (matches sim: effect = |post - pre| for each target)
    effects = {}
    for var in GRAPH_VARS:
        if var in pre_avg and var in post_avg:
            effects[var] = post_avg[var] - pre_avg[var]

    return {
        'pre_state': pre_avg,
        'post_state': post_avg,
        'effects': effects,
        'device': device,
        'action': action,
    }


def test_device_edges(device, smap):
    """
    Test ALL edges from one actuator device simultaneously.

    Matches sim pipeline (tester.py test_edge_iteratively):
      - Phase 1: 3 screening tests with do(device=ON)  [deterministic MAX]
      - Phase 2: 4 confirmatory tests distributed across strategies:
                 do(ON) + paired counterfactual do(OFF)   [multi-strategy]
      - Per-variable effect sizes accumulated across all tests
      - One-sample t-test per target variable

    Key insight: one physical do(Heater=ON) gives evidence for ALL
    Heater→X edges simultaneously (X ∈ {Temperature, Humidity, ...}).
    The sim tests edges one-at-a-time because the simulator resets, but
    physical hardware doesn't reset — one intervention affects everything.

    Returns:
        (edge_results, intervention_records, pre_post_states)
        edge_results: {(device, target): evidence_dict, ...}
    """
    logger.info(f"\n{'─'*40}")
    logger.info(f"  Testing all edges from: {device}")
    logger.info(f"{'─'*40}")

    # Collect per-variable effect sizes across all tests
    # effects_by_var[var] = [(delta, strategy), ...]
    effects_by_var = {var: [] for var in GRAPH_VARS if var != device}
    intervention_records = []
    pre_post_states = []

    # Phase 1: Screening — 3× do(device=ON) [deterministic MAX, like sim]
    for k in range(SCREENING_TESTS):
        logger.info(f"  Screening {k+1}/{SCREENING_TESTS}: do({device}=ON)")
        result = run_physical_intervention(device, smap, turn_on=True)
        if result:
            for var, delta in result['effects'].items():
                if var != device and var in effects_by_var:
                    effects_by_var[var].append(delta)
            intervention_records.append({
                'timestamp': datetime.now().isoformat(),
                'device': device, 'phase': 'screening', 'test': k + 1,
                'strategy': 'deterministic_max',
                'effects': {v: round(d, 6) for v, d in result['effects'].items()},
            })
            pre_post_states.append(result['pre_state'])
            pre_post_states.append(result['post_state'])
        logger.info(f"    Washout {WASHOUT_SEC // 60} min...")
        time.sleep(WASHOUT_SEC)

    # Phase 2: Confirmatory — distributed across strategies (like sim)
    # Sim uses: deterministic_max + paired_counterfactual + LLM_designed
    # Physical: do(ON) + paired counterfactual do(OFF). No LLM needed for
    # binary devices (there's no intermediate operating point).
    confirm_strategies = [
        ('deterministic_max', True),    # do(ON) — 2 tests
        ('paired_counterfactual', False),  # do(OFF) — 2 tests
    ]
    tests_per_strategy = max(1, CONFIRMATORY_TESTS // len(confirm_strategies))
    remaining = CONFIRMATORY_TESTS - tests_per_strategy * len(confirm_strategies)

    test_idx = 0
    for strat_idx, (strategy_name, turn_on) in enumerate(confirm_strategies):
        n_tests = tests_per_strategy + (1 if strat_idx < remaining else 0)
        for k in range(n_tests):
            test_idx += 1
            action_str = 'ON' if turn_on else 'OFF'
            logger.info(f"  Confirmatory {test_idx}/{CONFIRMATORY_TESTS}: "
                        f"do({device}={action_str}) [{strategy_name}]")
            result = run_physical_intervention(device, smap, turn_on=turn_on)
            if result:
                for var, delta in result['effects'].items():
                    if var != device and var in effects_by_var:
                        # For do(OFF), effect direction is inverted
                        effects_by_var[var].append(
                            delta if turn_on else -delta)
                intervention_records.append({
                    'timestamp': datetime.now().isoformat(),
                    'device': device, 'phase': 'confirmatory',
                    'test': test_idx, 'strategy': strategy_name,
                    'effects': {v: round(d, 6)
                                for v, d in result['effects'].items()},
                })
                pre_post_states.append(result['pre_state'])
                pre_post_states.append(result['post_state'])
            logger.info(f"    Washout {WASHOUT_SEC // 60} min...")
            time.sleep(WASHOUT_SEC)

    # ── Per-edge statistical test (matches sim's _compute_statistical_confidence) ──
    edge_results = {}
    for var, deltas in effects_by_var.items():
        edge = (device, var)
        threshold = EFFECT_THRESHOLDS.get(var.lower(), 0.05)

        if len(deltas) < 3:
            edge_results[edge] = {
                'validated': False, 'reason': 'insufficient_measurements',
                'n_tests': len(deltas),
                'mean_delta': round(float(np.mean(deltas)), 6) if deltas else 0,
            }
            continue

        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=1))

        # Screening gate: if mean effect below threshold, reject early
        if abs(mean_delta) < threshold:
            edge_results[edge] = {
                'validated': False, 'phase': 'screening',
                'reason': 'below_threshold',
                'n_tests': len(deltas),
                'mean_delta': round(mean_delta, 6),
                'threshold': threshold,
            }
            continue

        # One-sample t-test (same as sim's _compute_statistical_confidence)
        t_stat, p_value = stats.ttest_1samp(deltas, 0)
        is_valid = p_value < VALIDATION_ALPHA

        edge_results[edge] = {
            'validated': bool(is_valid),
            'phase': 'confirmatory',
            'n_tests': len(deltas),
            'mean_delta': round(mean_delta, 6),
            'std_delta': round(std_delta, 6),
            't_stat': round(float(t_stat), 4),
            'p_value': round(float(p_value), 6),
            'effect_sizes': [round(float(d), 6) for d in deltas],
        }

        status = "VALIDATED" if is_valid else "Not significant"
        logger.info(f"    {device}→{var}: {status} "
                    f"(Δ={mean_delta:.4f}, p={p_value:.4f})")

    return edge_results, intervention_records, pre_post_states


def compute_regression_effect(edge, data):
    """
    Compute partial_corr(source, target | all other vars) on accumulated D_obs.
    Stage 2 of validation, matching pipeline_cwm.py _compute_regression_effect.
    """
    source, target = edge
    src_col = _find_col(data, source)
    tgt_col = _find_col(data, target)
    if not src_col or not tgt_col:
        return {'partial_r': 0, 'p_value': 1.0}

    covariate_cols = [c for c in data.columns
                      if c != src_col and c != tgt_col
                      and c.lower() not in {'timestamp'}]
    cols = [src_col, tgt_col] + covariate_cols
    sub = data[cols].dropna()
    n = len(sub)
    k = len(covariate_cols)

    if n < k + 5:
        return {'partial_r': 0, 'p_value': 1.0}

    try:
        C = sub.corr().values
        P = np.linalg.inv(C)
        partial_r = -P[0, 1] / np.sqrt(abs(P[0, 0] * P[1, 1]))
        partial_r = max(-1.0, min(1.0, partial_r))
    except np.linalg.LinAlgError:
        partial_r = float(data[src_col].corr(data[tgt_col]))
        if np.isnan(partial_r):
            partial_r = 0.0

    # Fisher z-transform for p-value
    from scipy.stats import norm as sp_norm
    dof = n - k - 2
    if dof > 0 and abs(partial_r) < 1.0:
        z = 0.5 * np.log((1 + partial_r) / (1 - partial_r))
        se = 1.0 / np.sqrt(max(dof, 1))
        p_value = float(2 * (1 - sp_norm.cdf(abs(z / se))))
    else:
        p_value = 1.0

    return {'partial_r': round(partial_r, 4), 'p_value': round(p_value, 6)}


def apply_fdr_correction(edge_evidence, data):
    """
    Benjamini-Hochberg FDR correction across all tested edges.
    Matches pipeline_cwm.py Phase 2 FDR logic.

    Also computes Stage 2 regression effect for each edge.
    """
    FDR_ALPHA = 0.10
    FDR_MIN_EDGES = 10  # Lower than sim (20) since we have fewer edges

    tested = [(edge_key, ev) for edge_key, ev in edge_evidence.items()
              if ev.get('phase') in ('screening', 'confirmatory')
              and 'p_value' in ev]

    n_tested = len(tested)
    if n_tested == 0:
        return

    # Add Stage 2 regression effect to each edge
    for edge_key, ev in tested:
        src, tgt = edge_key.split('->')
        reg = compute_regression_effect((src, tgt), data)
        ev['partial_r'] = reg['partial_r']
        ev['regression_p'] = reg['p_value']
        ev['method'] = 'intervention+regression'

    if n_tested < FDR_MIN_EDGES:
        logger.info(f"  FDR skipped: only {n_tested} edges tested "
                    f"(< {FDR_MIN_EDGES}), using per-edge threshold")
        return

    # BH procedure
    p_values = [(edge_key, ev['p_value']) for edge_key, ev in tested]
    p_values.sort(key=lambda x: x[1])

    max_reject_rank = 0
    for rank_idx, (edge_key, p) in enumerate(p_values):
        rank = rank_idx + 1
        if p <= (rank / n_tested) * FDR_ALPHA:
            max_reject_rank = rank

    accepted_keys = {edge_key for rank_idx, (edge_key, p) in enumerate(p_values)
                     if rank_idx + 1 <= max_reject_rank}

    logger.info(f"  FDR correction: {n_tested} edges tested, "
                f"{max_reject_rank} pass BH(alpha={FDR_ALPHA})")

    # Update evidence: FDR-rejected edges that were per-edge significant
    for edge_key, ev in tested:
        if ev.get('validated') and edge_key not in accepted_keys:
            # FDR-rescue if strong observational evidence (like sim)
            obs_r = abs(ev.get('partial_r', 0))
            p = ev.get('p_value', 1.0)
            if obs_r > 0.3 and p < 0.15:
                ev['fdr_status'] = 'fdr_rescued'
                logger.info(f"  FDR-rescued: {edge_key} "
                            f"(p={p:.4f}, |partial_r|={obs_r:.3f})")
            else:
                ev['validated'] = False
                ev['fdr_status'] = 'fdr_rejected'
                logger.info(f"  FDR-rejected: {edge_key} "
                            f"(p={p:.4f}, |partial_r|={obs_r:.3f})")
        elif ev.get('validated'):
            ev['fdr_status'] = 'fdr_accepted'


def get_devices_to_test(candidates, scores, validated, failed):
    """
    Group candidate edges by source device. Return devices that have
    untested edges, plus rescue candidates (non-actuator sources).
    """
    actuators_lower = {a.lower() for a in ACTUATORS}
    val_norm = {(s.lower(), t.lower()) for s, t in validated}
    fail_norm = {(s.lower(), t.lower()) for s, t in failed}

    rescue_candidates = []
    devices_with_untested = set()

    for edge in candidates:
        src, tgt = edge
        if src.lower() in actuators_lower:
            if ((src.lower(), tgt.lower()) not in val_norm
                    and (src.lower(), tgt.lower()) not in fail_norm):
                devices_with_untested.add(src)
        else:
            rescue_candidates.append(edge)

    # Also check proactive actuator→sensor edges
    for act in ACTUATORS:
        for var in GRAPH_VARS:
            if var in ACTUATORS:
                continue
            if ((act.lower(), var.lower()) not in val_norm
                    and (act.lower(), var.lower()) not in fail_norm):
                devices_with_untested.add(act)

    # Sort by lowest-confidence edge (test least-certain device first)
    def device_min_score(dev):
        dev_edges = [e for e in candidates
                     if e[0].lower() == dev.lower()
                     and (e[0].lower(), e[1].lower()) not in val_norm
                     and (e[0].lower(), e[1].lower()) not in fail_norm]
        return min((scores.get(e, 0) for e in dev_edges), default=0)

    sorted_devices = sorted(devices_with_untested, key=device_min_score)

    return sorted_devices, rescue_candidates


def augment_data(baseline, pre_post_states):
    """Append intervention pre/post sensor states to D_obs for re-generation."""
    if not pre_post_states:
        return baseline

    rows = []
    for state in pre_post_states:
        row = {}
        for col in baseline.columns:
            for k, v in state.items():
                if k.lower() == col.lower():
                    try:
                        row[col] = float(v)
                    except (ValueError, TypeError):
                        pass
                    break
        if len(row) >= len(baseline.columns) * 0.5:
            rows.append(row)

    if rows:
        intv_df = pd.DataFrame(rows)
        common = [c for c in baseline.columns if c in intv_df.columns]
        if common:
            augmented = pd.concat([baseline, intv_df[common]], ignore_index=True)
            logger.info(f"  Augmented data: {len(baseline)} obs + {len(rows)} intv "
                        f"= {len(augmented)} total")
            return augmented

    return baseline


def rescue_edges(candidates, data, validated_edges, method_edges, evidence):
    """
    Rescue non-intervenable edges via weighted method consensus +
    back-door adjusted partial correlation.

    Matches pipeline_cwm.py Phase 2.4:
      - Weighted method consensus >= dynamic threshold
      - |partial_r(src, tgt | non-descendants + time)| > RESCUE_TAU
      - Source must be non-intervenable (exogenous/regime variable)
    """
    actuators_lower = {a.lower() for a in ACTUATORS}
    # Derived outputs can't be rescue sources either
    derived_outputs = {'energyconsumption', 'satisfaction'}
    rescue_eligible = {v.lower() for v in NON_INTERVENABLE} - derived_outputs

    rescue_candidates = [e for e in candidates
                         if e[0].lower() in rescue_eligible]

    logger.info(f"\n=== Phase 2.4: Non-Intervenable Edge Rescue ===")
    logger.info(f"  Rescue-eligible sources: {sorted(rescue_eligible)}")
    logger.info(f"  Candidates: {len(rescue_candidates)}")

    # Weighted method consensus (like sim: weight = 1/n_edges per method)
    method_weights = {}
    for method, edges in method_edges.items():
        w = 1.0 / max(len(edges), 1)
        method_weights[method] = w

    total_weight = sum(method_weights.values())
    dynamic_threshold = 0.33 * total_weight  # ~33% of total possible weight

    # Use clean data for rescue — drop NaN rows from augmented data
    # and use only numeric graph columns (no timestamp)
    graph_cols = [c for c in data.columns if c in GRAPH_VARS]
    clean = data[graph_cols].dropna()
    logger.info(f"  Rescue data: {len(clean)} clean rows (dropped {len(data) - len(clean)} NaN)")

    # Pre-compute time features for back-door conditioning
    n = len(clean)
    hour = np.arange(n) / 60 % 24
    time_sin = np.sin(2 * np.pi * hour / 24)
    time_cos = np.cos(2 * np.pi * hour / 24)

    rescued = set()
    for edge in rescue_candidates:
        src, tgt = edge
        src_col = _find_col(clean, src)
        tgt_col = _find_col(clean, tgt)
        if src_col is None or tgt_col is None:
            continue

        # Check weighted method consensus
        weighted_score = sum(
            method_weights[m] for m, edges in method_edges.items()
            if edge in edges)
        if weighted_score < dynamic_threshold:
            continue

        try:
            # Back-door adjusted partial correlation:
            # Condition on non-descendants of source + time features
            # (Pearl 2009 Ch. 3.3 — same as sim Phase 2.4)
            non_desc_cols = [c for c in clean.columns
                            if c != src_col and c != tgt_col
                            and c.lower() not in derived_outputs]

            x = clean[src_col].values
            y = clean[tgt_col].values
            Z_cols = [clean[c].values for c in non_desc_cols]
            Z = np.column_stack(Z_cols + [time_sin, time_cos, np.ones(n)])

            x_r = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
            y_r = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
            partial_r, partial_p = stats.pearsonr(x_r, y_r)

            if abs(partial_r) > RESCUE_TAU:
                rescued.add(edge)
                logger.info(f"  RESCUED: {src}→{tgt} "
                            f"(|r|={abs(partial_r):.3f}, "
                            f"consensus={weighted_score:.2f})")

            evidence[f"{src}->{tgt}"] = {
                'validated': bool(abs(partial_r) > RESCUE_TAU),
                'method': 'rescue_phase2.4',
                'partial_r': round(float(partial_r), 4),
                'partial_p': round(float(partial_p), 6),
                'weighted_consensus': round(float(weighted_score), 3),
                'threshold': round(float(dynamic_threshold), 3),
            }
        except Exception as e:
            logger.warning(f"  Rescue failed for {src}→{tgt}: {e}")

    return rescued


def run_iterative_discovery(baseline, smap, output_dir, max_iterations=10):
    """
    Iterative discovery loop matching simulation pipeline (pipeline_cwm.py).

    Each iteration:
      1. Generate hypotheses on D_obs (baseline + accumulated intervention data)
      2. Identify devices with untested edges
      3. For each device: run physical do-operator interventions
         (3 screening + 4 confirmatory, multi-strategy)
         → one intervention tests ALL edges from that device simultaneously
      4. Per-edge: one-sample t-test on accumulated effect sizes
      5. Stage 2: partial_corr(src, tgt | covariates) on augmented D_obs
      6. BH FDR correction across all edges in batch
      7. Append pre/post states to D_obs, regenerate hypotheses
      8. Repeat until max_iterations or no new devices to test

    After iterations: Phase 2.4 rescue for non-intervenable edges.
    """
    validated = set()
    failed = set()
    evidence = {}
    all_intervention_log = []
    all_pre_post = []
    current_data = baseline.copy()
    latest_method_edges = {}

    for iteration in range(max_iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"  ITERATION {iteration + 1}/{max_iterations}")
        logger.info(f"  D_obs: {len(current_data)} rows | "
                    f"Validated: {len(validated)} | Failed: {len(failed)}")
        logger.info(f"{'='*60}")

        # Step 1: Generate hypotheses on current data
        candidates, scores, methods = run_hypothesis_generation(current_data)
        latest_method_edges = methods

        # Save iteration state
        iter_dir = os.path.join(output_dir, f'iteration_{iteration + 1}')
        os.makedirs(iter_dir, exist_ok=True)
        with open(os.path.join(iter_dir, 'candidate_edges.json'), 'w') as f:
            json.dump({
                'candidates': [list(e) for e in candidates],
                'scores': {f"{s}->{t}": sc for (s, t), sc in scores.items()},
                'methods': {m: [list(e) for e in edges]
                            for m, edges in methods.items()},
                'n_data_rows': len(current_data),
            }, f, indent=2)

        # Step 2: Identify devices with untested edges
        devices_to_test, rescue_cands = get_devices_to_test(
            candidates, scores, validated, failed)

        if not devices_to_test:
            logger.info("No more devices to test — stopping iterations")
            break

        logger.info(f"  Devices to test: {devices_to_test}")

        # Step 3: Test each device (tests ALL edges from that device at once)
        iteration_pre_post = []
        for device in devices_to_test:
            # Ensure clean baseline: all devices OFF before each device block
            all_off()
            time.sleep(5)

            edge_results, records, pre_post = test_device_edges(
                device, smap)

            # Merge per-edge results into evidence
            for edge, ev in edge_results.items():
                evidence[f"{edge[0]}->{edge[1]}"] = ev
                if ev.get('validated'):
                    validated.add(edge)
                else:
                    failed.add(edge)

            all_intervention_log.extend(records)
            iteration_pre_post.extend(pre_post)

        all_pre_post.extend(iteration_pre_post)

        # Step 4: Augment D_obs with intervention data
        current_data = augment_data(baseline, all_pre_post)

        # Step 5: Stage 2 regression + FDR correction on this iteration's batch
        apply_fdr_correction(evidence, current_data)

        # Sync validated/failed with FDR results
        for edge_key, ev in evidence.items():
            src, tgt = edge_key.split('->')
            edge = (src, tgt)
            if ev.get('fdr_status') == 'fdr_rejected':
                validated.discard(edge)
                failed.add(edge)

        logger.info(f"\n  Iteration {iteration + 1} summary:")
        logger.info(f"    Validated: {len(validated)} total")
        logger.info(f"    Failed: {len(failed)} total")
        logger.info(f"    D_obs: {len(current_data)} rows")

        # Save intermediate results
        with open(os.path.join(iter_dir, 'validated_edges.json'), 'w') as f:
            json.dump([list(e) for e in sorted(validated)], f, indent=2)
        with open(os.path.join(iter_dir, 'edge_evidence.json'), 'w') as f:
            json.dump(evidence, f, indent=2)

    # Step 6: Phase 2.4 — Rescue non-intervenable edges
    rescued = rescue_edges(candidates, current_data, validated,
                           latest_method_edges, evidence)
    all_validated = validated | rescued

    # ── Save final results ──
    with open(os.path.join(output_dir, 'validated_edges.json'), 'w') as f:
        json.dump([list(e) for e in sorted(all_validated)], f, indent=2)

    with open(os.path.join(output_dir, 'edge_evidence.json'), 'w') as f:
        json.dump(evidence, f, indent=2)

    if all_intervention_log:
        pd.DataFrame(all_intervention_log).to_csv(
            os.path.join(output_dir, 'intervention_log.csv'), index=False)

    # F1 against proposed GT (diagnostic only)
    gt_norm = {(s.lower(), t.lower()) for s, t in PROPOSED_GT}
    val_norm = {(s.lower(), t.lower()) for s, t in all_validated}
    tp = len(val_norm & gt_norm)
    fp = len(val_norm - gt_norm)
    fn = len(gt_norm - val_norm)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)

    metrics = {
        'n_validated': len(validated), 'n_rescued': len(rescued),
        'n_failed': len(failed), 'n_total': len(all_validated),
        'n_iterations': min(iteration + 1, max_iterations),
        'n_data_rows_final': len(current_data),
        'tests_per_device': SCREENING_TESTS + CONFIRMATORY_TESTS,
        'gt_f1': round(f1, 3), 'gt_shd': fp + fn,
    }
    with open(os.path.join(output_dir, 'discovery_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nDiscovery: {len(validated)} validated + {len(rescued)} rescued "
                f"= {len(all_validated)} total ({len(failed)} failed)")
    logger.info(f"Proposed GT: F1={f1:.3f}, SHD={fp+fn}")

    return all_validated, evidence


def _find_col(df, var_name):
    for c in df.columns:
        if c.lower() == var_name.lower():
            return c
    return None


# ── Main ──
def main():
    parser = argparse.ArgumentParser(description='PolicyGRID Physical Discovery')
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of discovery seeds (default: 1)')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip physical interventions (hypothesis gen + rescue only)')
    parser.add_argument('--max-iterations', type=int, default=MAX_ITERATIONS,
                        help=f'Outer loop iterations (default: {MAX_ITERATIONS})')
    parser.add_argument('--baseline', default=BASELINE_CSV,
                        help='Path to baseline CSV')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(f'logs/discovery_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
            logging.StreamHandler(),
        ]
    )

    logger.info("=" * 60)
    logger.info("  PolicyGRID Physical Discovery (Iterative)")
    logger.info(f"  Seeds: {args.seeds}")
    logger.info(f"  Physical validation: {'YES' if not args.skip_validation else 'NO'}")
    logger.info(f"  Max iterations: {args.max_iterations}")
    logger.info(f"  Tests per device: {SCREENING_TESTS} screening + {CONFIRMATORY_TESTS} confirmatory")
    logger.info(f"  Baseline: {args.baseline}")
    logger.info("=" * 60)

    # Load baseline
    if not os.path.exists(args.baseline):
        logger.error(f"No baseline data at {args.baseline}")
        logger.error("Preprocess the collected sensor data into a baseline CSV first (see data/)")
        return

    baseline = pd.read_csv(args.baseline)
    logger.info(f"Loaded {len(baseline)} rows, columns: {list(baseline.columns)}")

    smap = load_scaling()
    logger.info(f"Scaling params: {list(smap.keys())}")

    seeds = list(range(42, 42 + args.seeds))

    try:
        # Seed 1: full physical validation
        random.seed(seeds[0])
        np.random.seed(seeds[0])

        seed_dir = os.path.join(OUTPUT_DIR, f'seed_{seeds[0]}')
        os.makedirs(seed_dir, exist_ok=True)

        if args.skip_validation:
            logger.info("Skipping physical validation (--skip-validation)")
            candidates, scores, methods = run_hypothesis_generation(baseline)

            # Save candidates
            with open(os.path.join(seed_dir, 'candidate_edges.json'), 'w') as f:
                json.dump({
                    'candidates': [list(e) for e in candidates],
                    'scores': {f"{s}->{t}": sc for (s, t), sc in scores.items()},
                    'methods': {m: [list(e) for e in edges]
                                for m, edges in methods.items()},
                }, f, indent=2)

            # Rescue only (no hardware)
            evidence = {}
            rescued = rescue_edges(candidates, baseline, set(), methods, evidence)
            with open(os.path.join(seed_dir, 'validated_edges.json'), 'w') as f:
                json.dump([list(e) for e in sorted(rescued)], f, indent=2)
            with open(os.path.join(seed_dir, 'edge_evidence.json'), 'w') as f:
                json.dump(evidence, f, indent=2)
        else:
            # Preflight: check hardware
            logger.info("\nPreflight check...")
            state = read_all_sensors()
            available = [k for k, v in state.items() if v is not None]
            logger.info(f"  Sensors: {len(available)} available — {available}")
            devices = get_device_states()
            logger.info(f"  Devices: {devices}")

            if len(available) < 4:
                logger.error("Too few sensors available. Aborting.")
                return

            # Run iterative discovery
            validated, evidence = run_iterative_discovery(
                baseline, smap, seed_dir,
                max_iterations=args.max_iterations,
            )

        # Seeds 2+: reuse measurements, re-run hypothesis gen
        for seed in seeds[1:]:
            logger.info(f"\n  Seed {seed}: hypothesis gen only (reusing measurements)")
            random.seed(seed)
            np.random.seed(seed)

            s_dir = os.path.join(OUTPUT_DIR, f'seed_{seed}')
            os.makedirs(s_dir, exist_ok=True)

            cands, scr, meths = run_hypothesis_generation(baseline)
            with open(os.path.join(s_dir, 'candidate_edges.json'), 'w') as f:
                json.dump({
                    'candidates': [list(e) for e in cands],
                    'scores': {f"{s}->{t}": sc for (s, t), sc in scr.items()},
                }, f, indent=2)

            # Reuse seed 1's intervention evidence
            seed1_evidence = os.path.join(OUTPUT_DIR, f'seed_{seeds[0]}',
                                          'edge_evidence.json')
            if os.path.exists(seed1_evidence):
                with open(seed1_evidence) as f:
                    ev = json.load(f)
                revalidated = set()
                for edge in cands:
                    key = f"{edge[0]}->{edge[1]}"
                    if key in ev and ev[key].get('validated'):
                        revalidated.add(edge)
                with open(os.path.join(s_dir, 'validated_edges.json'), 'w') as f:
                    json.dump([list(e) for e in sorted(revalidated)], f, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("  DISCOVERY COMPLETE")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Discovery failed: {e}", exc_info=True)
    finally:
        all_off()
        cleanup()


if __name__ == '__main__':
    main()
