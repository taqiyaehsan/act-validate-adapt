#!/usr/bin/env python3
"""Compare 3 approaches for improving window-only regime detection.

Baseline: 4-way monitors with standard linear PredictiveModel → 0% win-only
Approach 1: Add physics-informed coupling features (outdoor-T, outdoor-H)
Approach 2: Active HVAC probing when window state is ambiguous
Approach 3: Combined (1 + 2)

All approaches use the same edge set, training data, test data.
"""

import sys, os, json, subprocess, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurips_final_experiments import (
    load_data, get_scaling_map, collect_sim_data,
    build_observable_graph, build_predictive_model,
    run_monitoring, compute_monitoring_metrics,
    SENSOR_VARS, OBSERVABLE_VARS, LATENT_VARS, SIM_PATH,
    phys_to_norm, norm_to_phys,
)

SIGMA = 2.0
FORGET = 0.12


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 1: Physics-informed coupling features
# ═══════════════════════════════════════════════════════════════════════════════

COUPLING_FEATURES = ['outdoor_indoor_temp_gap', 'outdoor_indoor_hum_gap']
EXTENDED_SENSOR_VARS = SENSOR_VARS + COUPLING_FEATURES


def add_coupling_features(state_obs):
    """Add (outdoor-indoor) coupling features to observable state."""
    ext = dict(state_obs)
    outdoor_t = state_obs.get('outdoortemp', 0.5)
    indoor_t = state_obs.get('temperature', 0.5)
    outdoor_h = 0.5  # outdoor humidity not directly available, use midpoint
    indoor_h = state_obs.get('humidity', 0.5)
    ext['outdoor_indoor_temp_gap'] = outdoor_t - indoor_t
    ext['outdoor_indoor_hum_gap'] = outdoor_h - indoor_h
    return ext


def build_coupling_predictive_model(obs_edges, data, mask):
    """Build PredictiveModel with coupling features added to data."""
    import networkx as nx
    from src.causal_world_model import PredictiveModel

    extended_vars = list(OBSERVABLE_VARS) + COUPLING_FEATURES

    G = nx.DiGraph()
    G.add_nodes_from(extended_vars)
    G.add_edges_from(obs_edges)
    # Add coupling features as parents of temperature and humidity
    for feat, targets in [('outdoor_indoor_temp_gap', ['temperature']),
                          ('outdoor_indoor_hum_gap', ['humidity'])]:
        for t in targets:
            if t in G.nodes:
                G.add_edge(feat, t)

    model = PredictiveModel(G, extended_vars)

    cols = [c for c in data.columns if c.lower() in OBSERVABLE_VARS]
    df = data[cols].copy()
    df.columns = [c.lower() for c in df.columns]

    # Add coupling features
    if 'outdoortemp' in df.columns and 'temperature' in df.columns:
        df['outdoor_indoor_temp_gap'] = df['outdoortemp'] - df['temperature']
    else:
        df['outdoor_indoor_temp_gap'] = 0.0
    df['outdoor_indoor_hum_gap'] = 0.5 - df.get('humidity', 0.5)

    temporal = {}
    for var in extended_vars:
        if var in df.columns:
            temporal[var] = df[var].values[:-1]
            temporal[f'{var}_next'] = df[var].values[1:]

    tdf = pd.DataFrame(temporal)
    tdf = tdf[mask[:-1]].reset_index(drop=True)
    model.fit_temporal(tdf)
    return model


def construct_coupling_monitors(edges, data):
    """4-way monitors with coupling features."""
    obs_edges = build_observable_graph(edges)
    if len(obs_edges) == 0:
        return None

    cols_lower = {c.lower(): c for c in data.columns}
    occ_vals = data[cols_lower['occupancy']].values
    win_vals = data[cols_lower['windowposition']].values
    occ_on = occ_vals >= 0.1
    win_on = win_vals >= 0.15

    return {
        '_mode': 'coupling',
        'base': build_coupling_predictive_model(obs_edges, data, ~occ_on & ~win_on),
        'occ':  build_coupling_predictive_model(obs_edges, data,  occ_on & ~win_on),
        'win':  build_coupling_predictive_model(obs_edges, data, ~occ_on &  win_on),
        'full': build_coupling_predictive_model(obs_edges, data,  occ_on &  win_on),
    }


def run_monitoring_coupling(monitors, sim_records, sensor_vars=None):
    """4-way monitoring with coupling features."""
    if sensor_vars is None:
        sensor_vars = EXTENDED_SENSOR_VARS
    if monitors is None:
        return []

    regime_names = ['base', 'occ', 'win', 'full']
    weights = {r: 0.25 for r in regime_names}
    records = []

    for rec in sim_records:
        obs = add_coupling_features(rec['state_obs'])

        likelihoods = {}
        for rname in regime_names:
            pred = monitors[rname].predict(obs)
            err = sum(abs(obs.get(v, 0.5) - pred.get(v, 0.5))
                      for v in sensor_vars)
            likelihoods[rname] = np.exp(-SIGMA * err)

        unnorm = {r: likelihoods[r] * weights[r] for r in regime_names}
        total = sum(unnorm.values())
        if total > 0:
            raw = {r: unnorm[r] / total for r in regime_names}
            weights = {r: (1 - FORGET) * raw[r] + FORGET * 0.25
                       for r in regime_names}
        w_total = sum(weights.values())
        if w_total > 0:
            weights = {r: weights[r] / w_total for r in regime_names}

        winner = max(weights, key=weights.get)
        p_occ = weights['occ'] + weights['full']
        p_win = weights['win'] + weights['full']

        records.append({
            'step': rec['step'], 'hour': rec['hour'],
            'occ_active': rec['occ_active'], 'win_active': rec['win_active'],
            'regime_gt': rec['regime_gt'],
            'P_occ': p_occ, 'P_win': p_win, 'winner': winner,
        })
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 2: Active HVAC probing for window detection
# ═══════════════════════════════════════════════════════════════════════════════

def run_hvac_probe(state_phys, smap, elapsed_ms, outdoor_offset=0.0):
    """Pulse HVAC to 95% for 3 steps, then cut to 5% for 3 steps.
    Measure temperature trajectory. Window-open → fast decay after cut.

    Returns: decay_rate (temp drop per step after HVAC cut), temp_response dict
    """
    sim_abs = os.path.abspath(SIM_PATH)
    step_ms = 60000

    temps_heat = []
    temps_cool = []
    current_phys = dict(state_phys)

    # Phase 1: HVAC pulse to 95% for 3 steps
    for i in range(3):
        elapsed_ms += step_ms
        cmd = ['node', sim_abs, '--single-step',
               '--elapsed-ms', str(int(elapsed_ms)),
               '--outdoor-offset', str(outdoor_offset),
               '--state', json.dumps(current_phys),
               '--intervention', json.dumps({'hvacPower': 95})]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            for line in result.stdout.split('\n'):
                if 'RESULT:' in line:
                    current_phys = json.loads(line.split('RESULT:', 1)[1].strip())
                    break
        except Exception:
            pass
        temps_heat.append(current_phys.get('temperature',
                          current_phys.get('Temperature', 22)))

    peak_temp = temps_heat[-1] if temps_heat else 22

    # Phase 2: Cut HVAC to 5% for 3 steps — observe decay
    for i in range(3):
        elapsed_ms += step_ms
        cmd = ['node', sim_abs, '--single-step',
               '--elapsed-ms', str(int(elapsed_ms)),
               '--outdoor-offset', str(outdoor_offset),
               '--state', json.dumps(current_phys),
               '--intervention', json.dumps({'hvacPower': 5})]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            for line in result.stdout.split('\n'):
                if 'RESULT:' in line:
                    current_phys = json.loads(line.split('RESULT:', 1)[1].strip())
                    break
        except Exception:
            pass
        temps_cool.append(current_phys.get('temperature',
                          current_phys.get('Temperature', 22)))

    # Decay rate: how fast temp drops after HVAC cut
    if len(temps_cool) >= 2:
        decay_rate = (peak_temp - temps_cool[-1]) / len(temps_cool)
    else:
        decay_rate = 0.0

    return {
        'decay_rate': decay_rate,
        'peak_temp': peak_temp,
        'final_temp': temps_cool[-1] if temps_cool else peak_temp,
        'temps_heat': temps_heat,
        'temps_cool': temps_cool,
        'elapsed_ms': elapsed_ms,
    }


def run_monitoring_active_probe(monitors, sim_records, smap,
                                probe_cooldown=60, max_probes=50):
    """4-way monitoring + active HVAC probing for window detection.

    Probe scheduling follows an information-theoretic principle:
    - When the agent believes occupancy is OFF (p_occ < 0.3), passive
      monitoring has the least power to distinguish base from window-only
      (both are unoccupied rooms with similar sensor profiles).
    - A verification timer ensures periodic probing during sustained
      unoccupied periods, preventing complacent lock-in to a wrong state.
    - When the agent believes occupancy is ON, passive monitoring is
      sufficient (occ-only vs full is easier to distinguish), so no probe.

    This is pure posterior-driven active sensing: the agent decides what to
    probe based on where its own world model is least informative.
    """
    if monitors is None:
        return [], []

    regime_names = ['base', 'occ', 'win', 'full']
    weights = {r: 0.25 for r in regime_names}
    records = []
    probe_log = []
    steps_since_probe = 0
    n_probes = 0

    # Decay threshold: midpoint between expected window-open (~0.5) and
    # window-closed (~0.1) decay rates from simulator physics.
    # T_next = T + win*0.15*(outdoor-T) + hvac*0.5*(22-T)
    # After HVAC cut: decay dominated by win*0.15*(outdoor-T)
    decay_threshold = 0.25

    # Adaptive verification interval.  Starts at BASE_INTERVAL; doubles
    # after a confirmatory probe (current belief was correct), resets to
    # BASE_INTERVAL after a contradictory probe.  Prevents wasting budget
    # on re-confirming stable regimes while staying responsive to changes.
    BASE_INTERVAL = 360  # 6 hours — reasonable for real building monitoring
    verification_interval = BASE_INTERVAL

    # Track how long we've been in believed-unoccupied state
    unoccupied_streak = 0

    for i, rec in enumerate(sim_records):
        obs = rec['state_obs']

        # Standard 4-way Bayesian update
        likelihoods = {}
        for rname in regime_names:
            pred = monitors[rname].predict(obs)
            err = sum(abs(obs.get(v, 0.5) - pred.get(v, 0.5))
                      for v in SENSOR_VARS)
            likelihoods[rname] = np.exp(-SIGMA * err)

        unnorm = {r: likelihoods[r] * weights[r] for r in regime_names}
        total = sum(unnorm.values())
        if total > 0:
            raw = {r: unnorm[r] / total for r in regime_names}
            weights = {r: (1 - FORGET) * raw[r] + FORGET * 0.25
                       for r in regime_names}
        w_total = sum(weights.values())
        if w_total > 0:
            weights = {r: weights[r] / w_total for r in regime_names}

        p_occ = weights['occ'] + weights['full']
        p_win = weights['win'] + weights['full']

        # ── Posterior-driven probe scheduling ─────────────────────────────
        # The agent probes when: (1) it believes the room is unoccupied,
        # AND (2) enough time has passed since last probe (verification).
        # Rationale: when p_occ is low, passive monitoring cannot distinguish
        # base from window-only because both produce similar sensor patterns.
        # An HVAC probe reveals the thermal coupling to outdoor air, which
        # differs by 5-6x between open and closed window.
        believed_unoccupied = p_occ < 0.3
        if believed_unoccupied:
            unoccupied_streak += 1
        else:
            unoccupied_streak = 0

        # Probe if: unoccupied for a while (verification timer) OR
        # window ambiguous while unoccupied (active disambiguation)
        # Single unified trigger: probe when unoccupied AND enough time has
        # passed (adaptive interval). The interval adapts: confirmatory probes
        # → back off (double), contradictory probes → reset (stay alert).
        should_probe = (
            believed_unoccupied and
            unoccupied_streak >= verification_interval and
            steps_since_probe >= probe_cooldown and
            n_probes < max_probes and
            'state_phys' in rec
        )

        if should_probe:
            hour = rec['hour']
            day = int(hour / 24) % 4
            offset = {0: -3.0, 1: 6.0, 2: 13.0, 3: 6.0}.get(day, 0.0)

            # Record pre-probe belief for adaptation
            pre_winner = max(weights, key=weights.get)

            probe_result = run_hvac_probe(
                rec['state_phys'], smap,
                elapsed_ms=int(hour * 3600000),
                outdoor_offset=offset)

            decay = probe_result['decay_rate']

            # High decay → window likely open (heat escapes fast)
            # Low decay → window likely closed (heat retained)
            if decay > decay_threshold:
                # Evidence for window open
                win_boost = 1.0 + min(5.0, (decay - decay_threshold) * 20)
                weights['win'] *= win_boost
                weights['full'] *= win_boost
                weights['base'] /= max(0.2, win_boost)
                weights['occ'] /= max(0.2, win_boost)
            else:
                # Evidence for window closed
                close_boost = 1.0 + min(5.0, (decay_threshold - decay) * 20)
                weights['base'] *= close_boost
                weights['occ'] *= close_boost
                weights['win'] /= max(0.2, close_boost)
                weights['full'] /= max(0.2, close_boost)

            # Renormalize
            w_total = sum(weights.values())
            if w_total > 0:
                weights = {r: max(0.01, weights[r] / w_total)
                           for r in regime_names}
                w_total = sum(weights.values())
                weights = {r: weights[r] / w_total for r in regime_names}

            p_occ = weights['occ'] + weights['full']
            p_win = weights['win'] + weights['full']

            # Adaptive interval: if probe confirmed current belief, back off
            # (double interval — we're probably right). If contradicted,
            # reset to base (something changed, stay alert).
            post_winner = max(weights, key=weights.get)
            if post_winner == pre_winner:
                # Confirmatory — extend interval (up to 4x base)
                verification_interval = min(BASE_INTERVAL * 4,
                                            verification_interval * 2)
            else:
                # Contradictory — reset to base interval
                verification_interval = BASE_INTERVAL

            probe_log.append({
                'step': i, 'hour': round(hour, 2),
                'decay_rate': round(decay, 4),
                'gt_occ': rec['occ_active'], 'gt_win': rec['win_active'],
                'p_win_after': round(p_win, 3),
                'correct': (decay > decay_threshold) == (rec['win_active'] == 1),
            })

            steps_since_probe = 0
            unoccupied_streak = 0  # reset verification timer
            n_probes += 1
        else:
            steps_since_probe += 1

        winner = max(weights, key=weights.get)
        records.append({
            'step': rec['step'], 'hour': rec['hour'],
            'occ_active': rec['occ_active'], 'win_active': rec['win_active'],
            'regime_gt': rec['regime_gt'],
            'P_occ': p_occ, 'P_win': p_win, 'winner': winner,
        })
    return records, probe_log


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 3: Combined (coupling features + active probing)
# ═══════════════════════════════════════════════════════════════════════════════

def run_monitoring_combined(monitors_coupling, monitors_standard,
                            sim_records, smap,
                            probe_cooldown=60, max_probes=50):
    """Coupling features for passive monitoring + active probing as fallback.

    Same posterior-driven probe scheduling as approach 2.
    """
    if monitors_coupling is None:
        return [], []

    regime_names = ['base', 'occ', 'win', 'full']
    weights = {r: 0.25 for r in regime_names}
    sensor_vars = EXTENDED_SENSOR_VARS
    records = []
    probe_log = []
    steps_since_probe = 0
    n_probes = 0
    decay_threshold = 0.25
    BASE_INTERVAL = 360
    verification_interval = BASE_INTERVAL
    unoccupied_streak = 0

    for i, rec in enumerate(sim_records):
        obs = add_coupling_features(rec['state_obs'])

        # Coupling-enhanced 4-way Bayesian update
        likelihoods = {}
        for rname in regime_names:
            pred = monitors_coupling[rname].predict(obs)
            err = sum(abs(obs.get(v, 0.5) - pred.get(v, 0.5))
                      for v in sensor_vars)
            likelihoods[rname] = np.exp(-SIGMA * err)

        unnorm = {r: likelihoods[r] * weights[r] for r in regime_names}
        total = sum(unnorm.values())
        if total > 0:
            raw = {r: unnorm[r] / total for r in regime_names}
            weights = {r: (1 - FORGET) * raw[r] + FORGET * 0.25
                       for r in regime_names}
        w_total = sum(weights.values())
        if w_total > 0:
            weights = {r: weights[r] / w_total for r in regime_names}

        p_occ = weights['occ'] + weights['full']
        p_win = weights['win'] + weights['full']

        # Posterior-driven probe scheduling (same as approach 2)
        believed_unoccupied = p_occ < 0.3
        if believed_unoccupied:
            unoccupied_streak += 1
        else:
            unoccupied_streak = 0

        # Single unified trigger: probe when unoccupied AND enough time has
        # passed (adaptive interval). The interval adapts: confirmatory probes
        # → back off (double), contradictory probes → reset (stay alert).
        should_probe = (
            believed_unoccupied and
            unoccupied_streak >= verification_interval and
            steps_since_probe >= probe_cooldown and
            n_probes < max_probes and
            'state_phys' in rec
        )

        if should_probe:
            hour = rec['hour']
            day = int(hour / 24) % 4
            offset = {0: -3.0, 1: 6.0, 2: 13.0, 3: 6.0}.get(day, 0.0)

            pre_winner = max(weights, key=weights.get)

            probe_result = run_hvac_probe(
                rec['state_phys'], smap,
                elapsed_ms=int(hour * 3600000),
                outdoor_offset=offset)

            decay = probe_result['decay_rate']

            if decay > decay_threshold:
                win_boost = 1.0 + min(5.0, (decay - decay_threshold) * 20)
                weights['win'] *= win_boost
                weights['full'] *= win_boost
                weights['base'] /= max(0.2, win_boost)
                weights['occ'] /= max(0.2, win_boost)
            else:
                close_boost = 1.0 + min(5.0, (decay_threshold - decay) * 20)
                weights['base'] *= close_boost
                weights['occ'] *= close_boost
                weights['win'] /= max(0.2, close_boost)
                weights['full'] /= max(0.2, close_boost)

            w_total = sum(weights.values())
            if w_total > 0:
                weights = {r: max(0.01, weights[r] / w_total)
                           for r in regime_names}
                w_total = sum(weights.values())
                weights = {r: weights[r] / w_total for r in regime_names}

            p_occ = weights['occ'] + weights['full']
            p_win = weights['win'] + weights['full']

            post_winner = max(weights, key=weights.get)
            if post_winner == pre_winner:
                verification_interval = min(BASE_INTERVAL * 4,
                                            verification_interval * 2)
            else:
                verification_interval = BASE_INTERVAL

            probe_log.append({
                'step': i, 'hour': round(hour, 2),
                'decay_rate': round(decay, 4),
                'gt_occ': rec['occ_active'], 'gt_win': rec['win_active'],
                'p_win_after': round(p_win, 3),
                'correct': (decay > decay_threshold) == (rec['win_active'] == 1),
            })

            steps_since_probe = 0
            unoccupied_streak = 0  # reset verification timer
            n_probes += 1
        else:
            steps_since_probe += 1

        winner = max(weights, key=weights.get)
        records.append({
            'step': rec['step'], 'hour': rec['hour'],
            'occ_active': rec['occ_active'], 'win_active': rec['win_active'],
            'regime_gt': rec['regime_gt'],
            'P_occ': p_occ, 'P_win': p_win, 'winner': winner,
        })
    return records, probe_log


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def detailed_metrics(records):
    if not records:
        return {}
    df = pd.DataFrame(records)
    regime_map = {0: 'base', 1: 'occ', 2: 'win', 3: 'full'}
    gt_labels = df['regime_gt'].map(regime_map)
    combined = (df['winner'] == gt_labels).mean() * 100
    occ_correct = ((df['P_occ'] >= 0.5) == df['occ_active'].astype(bool)).mean() * 100
    win_correct = ((df['P_win'] >= 0.5) == df['win_active'].astype(bool)).mean() * 100

    per_regime = {}
    for rid, rname in sorted(regime_map.items()):
        mask = df['regime_gt'] == rid
        if mask.sum() > 0:
            per_regime[rname] = round(
                (df.loc[mask, 'winner'] == rname).mean() * 100, 1)
    return {
        'combined': round(combined, 1),
        'occ': round(occ_correct, 1),
        'win': round(win_correct, 1),
        **per_regime,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    data, scaling, gt_set = load_data()
    smap = get_scaling_map(scaling)

    with open('results/neurips_final/discovered_edges.json') as f:
        d = json.load(f)
    grid_edges = set((e[0], e[1]) for e in d['consensus_validated'])
    obs_edges = build_observable_graph(grid_edges)

    cols_lower = {c.lower(): c for c in data.columns}
    occ_vals = data[cols_lower['occupancy']].values
    win_vals = data[cols_lower['windowposition']].values

    # Build monitors for each approach
    print("\nBuilding monitors...")

    print("  Baseline (4-way standard)...")
    from neurips_final_experiments import construct_factored_monitors
    monitors_baseline = construct_factored_monitors(grid_edges, data)

    print("  Approach 1 (coupling features)...")
    monitors_coupling = construct_coupling_monitors(grid_edges, data)

    # Approaches 2 and 3 use baseline/coupling monitors + active probing at runtime

    # Run on multiple seeds
    N_SEEDS = 3
    results = {
        'baseline': [], 'approach1_coupling': [],
        'approach2_active': [], 'approach3_combined': [],
    }
    all_probe_logs = {'approach2': [], 'approach3': []}

    for seed_i in range(N_SEEDS):
        print(f"\n--- Seed {seed_i + 1}/{N_SEEDS} ---")
        sim_records = collect_sim_data(smap, duration_steps=5760)

        from collections import Counter
        rc = Counter(r['regime_gt'] for r in sim_records)
        rn = {0: 'base', 1: 'occ', 2: 'win', 3: 'full'}
        print(f"  Regimes: {', '.join(f'{rn[k]}={rc.get(k,0)}' for k in sorted(rn))}")

        # Baseline
        recs = run_monitoring(monitors_baseline, sim_records)
        m = detailed_metrics(recs)
        results['baseline'].append(m)
        print(f"  Baseline:     Comb={m['combined']}  base={m.get('base','?')}  "
              f"occ={m.get('occ_regime', m.get('occ','?'))}  "
              f"win={m.get('win','?')}  full={m.get('full','?')}")

        # Approach 1: coupling
        recs = run_monitoring_coupling(monitors_coupling, sim_records)
        m = detailed_metrics(recs)
        results['approach1_coupling'].append(m)
        print(f"  1_coupling:   Comb={m['combined']}  base={m.get('base','?')}  "
              f"occ={m.get('occ_regime', m.get('occ','?'))}  "
              f"win={m.get('win','?')}  full={m.get('full','?')}")

        # Approach 2: active probing
        recs, plog = run_monitoring_active_probe(
            monitors_baseline, sim_records, smap,
            probe_cooldown=60, max_probes=20)
        m = detailed_metrics(recs)
        results['approach2_active'].append(m)
        all_probe_logs['approach2'].extend(plog)
        n_correct = sum(1 for p in plog if p['correct'])
        print(f"  2_active:     Comb={m['combined']}  base={m.get('base','?')}  "
              f"occ={m.get('occ_regime', m.get('occ','?'))}  "
              f"win={m.get('win','?')}  full={m.get('full','?')}  "
              f"probes={len(plog)} ({n_correct}/{len(plog)} correct)")

        # Approach 3: combined
        recs, plog = run_monitoring_combined(
            monitors_coupling, monitors_baseline, sim_records, smap,
            probe_cooldown=60, max_probes=20)
        m = detailed_metrics(recs)
        results['approach3_combined'].append(m)
        all_probe_logs['approach3'].extend(plog)
        n_correct = sum(1 for p in plog if p['correct'])
        print(f"  3_combined:   Comb={m['combined']}  base={m.get('base','?')}  "
              f"occ={m.get('occ_regime', m.get('occ','?'))}  "
              f"win={m.get('win','?')}  full={m.get('full','?')}  "
              f"probes={len(plog)} ({n_correct}/{len(plog)} correct)")

    # Summary
    print("\n" + "=" * 95)
    print(f"{'Approach':<22} {'Combined':>9} {'Base':>7} {'Occ':>7} {'Win':>7} {'Full':>7}  {'P(occ)':>7} {'P(win)':>7}")
    print("-" * 95)

    for name, label in [('baseline', 'Baseline (4-way)'),
                        ('approach1_coupling', '1: Coupling features'),
                        ('approach2_active', '2: Active HVAC probe'),
                        ('approach3_combined', '3: Combined (1+2)')]:
        seeds = results[name]
        comb = np.mean([s['combined'] for s in seeds])
        comb_std = np.std([s['combined'] for s in seeds])
        base = np.mean([s.get('base', 0) for s in seeds])
        occ_r = np.mean([s.get('occ', 0) for s in seeds])
        win_r = np.mean([s.get('win', 0) for s in seeds])
        full_r = np.mean([s.get('full', 0) for s in seeds])
        p_occ = np.mean([s.get('occ', 0) for s in seeds])
        p_win = np.mean([s.get('win', 0) for s in seeds])

        print(f"{label:<22} {comb:>6.1f}±{comb_std:.1f} {base:>6.1f}  "
              f"{occ_r:>6.1f}  {win_r:>6.1f}  {full_r:>6.1f}  {p_occ:>6.1f}  {p_win:>6.1f}")

    print("=" * 95)

    # Probe analysis
    for key in ('approach2', 'approach3'):
        plog = all_probe_logs[key]
        if plog:
            n_total = len(plog)
            n_correct = sum(1 for p in plog if p['correct'])
            win_probes = [p for p in plog if p['gt_win'] == 1]
            base_probes = [p for p in plog if p['gt_win'] == 0]
            print(f"\n{key} probe stats:")
            print(f"  Total probes: {n_total}, correct: {n_correct}/{n_total} ({100*n_correct/n_total:.0f}%)")
            if win_probes:
                decay_win = np.mean([p['decay_rate'] for p in win_probes])
                print(f"  Win-open probes: n={len(win_probes)}, avg decay={decay_win:.4f}")
            if base_probes:
                decay_base = np.mean([p['decay_rate'] for p in base_probes])
                print(f"  Win-closed probes: n={len(base_probes)}, avg decay={decay_base:.4f}")


if __name__ == '__main__':
    main()
