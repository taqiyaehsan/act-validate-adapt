#!/usr/bin/env python3
"""
Adaptive Regime Monitoring for smart_building_rich.

PolicyGRID discovers a causal graph where some variables are latent
(occupancy, windowPosition are never directly observed at runtime).
The framework:

  1. During discovery (training), all sensors are observed. PolicyGRID
     discovers edges including regime-variable→sensor edges.

  2. It identifies regime variables (non-intervenable sources with many
     children) and constructs FACTORED BINARY monitors:
       - Occupancy monitor: CG_occ_off vs CG_occ_on
         (trained on marginal occ splits regardless of window state)
       - Window monitor: CG_win_closed vs CG_win_open
         (trained on marginal window splits regardless of occupancy)

     Each monitor uses the SAME observable graph structure but different
     learned coefficients (intercepts, slopes shift between regimes).
     This avoids the data-imbalance problem of joint 4-way splits
     (e.g. "unoccupied + window open" is rare in realistic data).

  3. At runtime, regime variables are latent. Two independent Bayesian
     model comparisons run in parallel:
       - Occupancy: P(occupied | sensor readings)
       - Window: P(window open | sensor readings)
     Combined regime = product of the two binary decisions.
     The winning CG variant becomes the active world model for policy.

  4. Sensor ablation: additionally drop observable sensors and test how
     many can fail before regime detection breaks.

This is a factored switching linear dynamical system with Bayesian
model selection.
"""

import sys, os, json, subprocess, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import networkx as nx
from src.causal_world_model import PredictiveModel

# ── Ground truth (for evaluation only) ───────────────────────────────────────
with open('ground_truth_graphs.json') as f:
    GT = json.load(f)['smart_building_rich']['edges']
GT_SET = {(s.lower(), t.lower()) for s, t in GT}

ALL_VARS = ['outdoortemp', 'solarradiation', 'occupancy', 'windowposition',
            'temperature', 'humidity', 'co2', 'lightlevel', 'noisedb',
            'airquality', 'pmv', 'hvacpower', 'lightingpower',
            'energyconsumption', 'satisfaction']

LATENT_VARS = {'occupancy', 'windowposition'}
OBSERVABLE_VARS = [v for v in ALL_VARS if v not in LATENT_VARS]

# Sensor variables used for regime detection (downstream of regime vars)
SENSOR_VARS = ['temperature', 'humidity', 'co2', 'lightlevel', 'noisedb', 'airquality']


# ── CG Variant Construction ──────────────────────────────────────────────────

def build_observable_graph(discovered_edges):
    """Extract edges between observable variables only."""
    return {(s, t) for s, t in discovered_edges
            if s in OBSERVABLE_VARS and t in OBSERVABLE_VARS
            and s in ALL_VARS and t in ALL_VARS}


def build_variant_model(obs_edges, data, mask, label):
    """Build a PredictiveModel on observable variables, trained on masked data."""
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
    logger.info(f"  {label}: {len(obs_edges)} edges, {len(tdf)} training rows")
    return model


def construct_factored_monitors(discovered_edges, data):
    """
    From PolicyGRID's discovered edges, build FACTORED binary monitors.

    Instead of 4 joint variants (which suffer data imbalance when regimes
    are correlated), we train 2 independent binary monitors:
      - Occupancy: occ_off (marginal) vs occ_on (marginal)
      - Window:    win_closed (marginal) vs win_open (marginal)

    Each trained regardless of the other regime variable's state.
    This ensures ample training data for all variants.
    """
    obs_edges = build_observable_graph(discovered_edges)

    cols_lower = {c.lower(): c for c in data.columns}
    occ_vals = data[cols_lower['occupancy']].values
    win_vals = data[cols_lower['windowposition']].values

    occ_inactive = occ_vals < 0.1
    occ_active = occ_vals >= 0.1
    win_closed = win_vals < 0.15
    win_open = win_vals >= 0.15

    logger.info(f"Observable graph: {len(obs_edges)} edges")
    logger.info(f"Marginal splits:")
    logger.info(f"  Occupancy: off={occ_inactive.sum()}, on={occ_active.sum()}")
    logger.info(f"  Window: closed={win_closed.sum()}, open={win_open.sum()}")

    monitors = {}

    # Occupancy monitor: marginal split
    monitors['occ'] = {
        'off': build_variant_model(obs_edges, data, occ_inactive, "Occ_OFF (marginal)"),
        'on':  build_variant_model(obs_edges, data, occ_active, "Occ_ON (marginal)"),
    }

    # Window monitor: marginal split
    monitors['win'] = {
        'off': build_variant_model(obs_edges, data, win_closed, "Win_CLOSED (marginal)"),
        'on':  build_variant_model(obs_edges, data, win_open, "Win_OPEN (marginal)"),
    }

    return monitors


# ── Simulator Data Collection ────────────────────────────────────────────────

def get_outdoor_offset(hour_of_sim):
    day = int(hour_of_sim / 24) % 3
    return {0: -3.0, 1: 6.0, 2: 13.0}[day]


def collect_sim_data(sim_path, scaling_params, duration_steps=4320):
    """Run simulator once, collect normalized states."""
    sim_abs = os.path.abspath(sim_path)
    state = None
    elapsed_ms = 0
    step_ms = 60000
    records = []

    for step_i in range(duration_steps):
        elapsed_ms += step_ms
        hour_of_sim = elapsed_ms / 3600000
        offset = get_outdoor_offset(hour_of_sim)

        cmd = ['node', sim_abs, '--single-step',
               '--elapsed-ms', str(elapsed_ms),
               '--outdoor-offset', str(offset)]
        if state:
            cmd += ['--state', json.dumps(state)]

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

        # Normalize
        state_norm = {}
        for k, v in state.items():
            if not isinstance(v, (int, float)):
                continue
            k_lower = k.lower()
            if scaling_params is not None:
                feat_col = 'feature' if 'feature' in scaling_params.columns else 'column'
                min_col = 'data_min' if 'data_min' in scaling_params.columns else 'min'
                max_col = 'data_max' if 'data_max' in scaling_params.columns else 'max'
                param = scaling_params[scaling_params[feat_col].str.lower() == k_lower]
                if not param.empty:
                    mn, mx = param[min_col].values[0], param[max_col].values[0]
                    rng = mx - mn
                    state_norm[k_lower] = (v - mn) / rng if rng > 0 else 0.5
                else:
                    state_norm[k_lower] = v
            else:
                state_norm[k_lower] = v

        occ = state.get('occupancy', 0)
        win = state.get('windowPosition', state.get('windowposition', 0))

        # Ground truth regime (for evaluation only)
        occ_active = int(occ >= 1)
        win_active = int(win > 0.3)
        # Combined regime: 0=base, 1=occ_only, 2=win_only, 3=both
        regime_gt = occ_active * 1 + win_active * 2

        records.append({
            'step': step_i,
            'hour_of_sim': round(hour_of_sim, 4),
            'occupancy': occ,
            'occ_active': occ_active,
            'window_position': round(win, 3),
            'win_active': win_active,
            'regime_gt': regime_gt,
            'outdoor_temp': round(float(state.get('outdoorTemp', state.get('outdoortemp', 15))), 1),
            'state_obs': {k: v for k, v in state_norm.items() if k not in LATENT_VARS},
        })

        if step_i % 500 == 0:
            logger.info(f"  step {step_i}/{duration_steps}")

    logger.info(f"Collected {len(records)} states")
    return records


# ── Factored Bayesian Monitoring ─────────────────────────────────────────────


def _run_binary_monitor(model_off, model_on, sim_records, sensor_vars,
                         forget_factor=0.12, sigma=2.0):
    """
    Binary Bayesian monitor: P(on | sensor readings) over time.

    Returns list of dicts with 'P_on' posterior at each step.
    """
    p_on = 0.5  # uniform prior
    records = []

    for rec in sim_records:
        state_obs = rec['state_obs']

        # Each model predicts sensor readings
        pred_off = model_off.predict(state_obs)
        pred_on = model_on.predict(state_obs)

        err_off = sum(abs(state_obs.get(v, 0.5) - pred_off.get(v, 0.5))
                      for v in sensor_vars)
        err_on = sum(abs(state_obs.get(v, 0.5) - pred_on.get(v, 0.5))
                     for v in sensor_vars)

        # Bayesian update
        lik_off = np.exp(-sigma * err_off)
        lik_on = np.exp(-sigma * err_on)

        unnorm_off = lik_off * (1 - p_on)
        unnorm_on = lik_on * p_on
        total = unnorm_off + unnorm_on

        if total > 0:
            p_on_raw = unnorm_on / total
            p_on = (1 - forget_factor) * p_on_raw + forget_factor * 0.5
        # else keep p_on unchanged

        records.append({
            'P_on': round(p_on, 6),
            'err_off': round(err_off, 4),
            'err_on': round(err_on, 4),
        })

    return records


def run_factored_monitoring(monitors, sim_records, sensor_vars=None,
                            forget_factor=0.12, sigma=2.0):
    """
    Factored regime monitoring: two independent binary monitors.

    Returns combined records with P(occ_on), P(win_open), and derived
    4-way regime posterior.
    """
    if sensor_vars is None:
        sensor_vars = SENSOR_VARS

    # Run each binary monitor independently
    occ_records = _run_binary_monitor(
        monitors['occ']['off'], monitors['occ']['on'],
        sim_records, sensor_vars, forget_factor, sigma)

    win_records = _run_binary_monitor(
        monitors['win']['off'], monitors['win']['on'],
        sim_records, sensor_vars, forget_factor, sigma)

    # Combine into unified records
    records = []
    for i, rec in enumerate(sim_records):
        p_occ = occ_records[i]['P_on']
        p_win = win_records[i]['P_on']

        # 4-way combined (factored posterior)
        p_base = (1 - p_occ) * (1 - p_win)
        p_occ_only = p_occ * (1 - p_win)
        p_win_only = (1 - p_occ) * p_win
        p_both = p_occ * p_win

        combined = {'base': p_base, 'occ': p_occ_only,
                    'win': p_win_only, 'full': p_both}
        winner_4way = max(combined, key=combined.get)

        records.append({
            'step': rec['step'],
            'hour_of_sim': rec['hour_of_sim'],
            'occ_active': rec['occ_active'],
            'win_active': rec['win_active'],
            'regime_gt': rec['regime_gt'],
            'outdoor_temp': rec['outdoor_temp'],
            'P_occ': p_occ,
            'P_win': p_win,
            'P_base': round(p_base, 6),
            'P_occ_only': round(p_occ_only, 6),
            'P_win_only': round(p_win_only, 6),
            'P_both': round(p_both, 6),
            'winner_4way': winner_4way,
            'err_occ_off': occ_records[i]['err_off'],
            'err_occ_on': occ_records[i]['err_on'],
            'err_win_off': win_records[i]['err_off'],
            'err_win_on': win_records[i]['err_on'],
        })

    return records


def compute_detection_metrics(records):
    """Compute regime detection metrics from factored monitoring."""
    if not records:
        return {}

    df = pd.DataFrame(records)

    # Binary accuracy for each monitor
    occ_correct = ((df['P_occ'] >= 0.5) == df['occ_active'].astype(bool)).mean()
    win_correct = ((df['P_win'] >= 0.5) == df['win_active'].astype(bool)).mean()

    # 4-way combined accuracy
    regime_map = {0: 'base', 1: 'occ', 2: 'win', 3: 'full'}
    gt_labels = df['regime_gt'].map(regime_map)
    combined_accuracy = (df['winner_4way'] == gt_labels).mean()

    # Per-regime breakdown
    per_regime = {}
    for regime_id, regime_name in regime_map.items():
        subset = df[df['regime_gt'] == regime_id]
        if len(subset) == 0:
            continue
        winner_pct = (subset['winner_4way'] == regime_name).mean() * 100
        mean_p_occ = subset['P_occ'].mean()
        mean_p_win = subset['P_win'].mean()
        per_regime[regime_name] = {
            'n_steps': len(subset),
            'winner_pct': round(winner_pct, 1),
            'mean_P_occ': round(mean_p_occ, 3),
            'mean_P_win': round(mean_p_win, 3),
        }

    return {
        'occ_accuracy': round(occ_correct * 100, 1),
        'win_accuracy': round(win_correct * 100, 1),
        'combined_accuracy': round(combined_accuracy * 100, 1),
        'per_regime': per_regime,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_factored_monitoring(records, title, output_path):
    """Plot factored binary posteriors with regime ground truth."""
    df = pd.DataFrame(records)
    time_h = df['hour_of_sim']

    fig, axes = plt.subplots(4, 1, figsize=(16, 14),
                             height_ratios=[3, 3, 1.2, 1.2],
                             sharex=True, gridspec_kw={'hspace': 0.08})
    ax_occ_post, ax_win_post, ax_occ_gt, ax_win_gt = axes

    # Background regime shading on both posterior panels
    regime_colors = {0: '#e8f4fd', 1: '#fce4e4', 2: '#e4f5e4', 3: '#f5e6f5'}
    for ax_panel in [ax_occ_post, ax_win_post]:
        current = df['regime_gt'].iloc[0]
        start_idx = 0
        for i in range(1, len(df)):
            if df['regime_gt'].iloc[i] != current or i == len(df) - 1:
                end_idx = i if i < len(df) - 1 else i
                ax_panel.axvspan(time_h.iloc[start_idx], time_h.iloc[end_idx],
                                 color=regime_colors.get(current, 'white'),
                                 alpha=0.4, zorder=0)
                start_idx = i
                current = df['regime_gt'].iloc[i]

    # Day dividers
    for d in [24, 48]:
        for a in axes:
            a.axvline(d, color='black', ls=':', lw=1, alpha=0.4)

    # ── Panel 1: Occupancy posterior ──
    ax_occ_post.plot(time_h, df['P_occ'], color='#e74c3c', lw=1.8,
                     label='P(occupied)')
    ax_occ_post.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax_occ_post.set_ylabel('P(Occupied)', fontsize=11)
    ax_occ_post.set_ylim(-0.02, 1.02)
    ax_occ_post.set_title(title, fontsize=14, fontweight='bold')
    ax_occ_post.grid(True, alpha=0.15)

    # Day labels
    for d, label in [(12, 'Day 1: Cool'), (36, 'Day 2: Mild'), (60, 'Day 3: Hot')]:
        ax_occ_post.text(d, 0.97, label, ha='center', va='top', fontsize=9,
                         fontweight='bold', alpha=0.5)

    # Regime legend patches
    regime_labels = {0: 'Base', 1: 'Occupied', 2: 'Window Open', 3: 'Both'}
    patches = [mpatches.Patch(color=regime_colors[k], alpha=0.4,
                              label=regime_labels[k]) for k in [0, 1, 2, 3]]
    ax_occ_post.legend(handles=patches + ax_occ_post.get_legend_handles_labels()[0],
                       loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    # ── Panel 2: Window posterior ──
    ax_win_post.plot(time_h, df['P_win'], color='#2ecc71', lw=1.8,
                     label='P(window open)')
    ax_win_post.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax_win_post.set_ylabel('P(Window Open)', fontsize=11)
    ax_win_post.set_ylim(-0.02, 1.02)
    ax_win_post.grid(True, alpha=0.15)
    ax_win_post.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    # ── Panel 3: GT occupancy ──
    ax_occ_gt.fill_between(time_h, 0, df['occ_active'], color='#e74c3c',
                            alpha=0.3, step='mid', label='Occupied [GT]')
    ax_occ_gt.set_ylabel('Occ GT', fontsize=9)
    ax_occ_gt.set_ylim(-0.05, 1.1)
    ax_occ_gt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    # ── Panel 4: GT window ──
    ax_win_gt.fill_between(time_h, 0, df['win_active'], color='#2ecc71',
                            alpha=0.3, step='mid', label='Window Open [GT]')
    ax_win_gt.set_ylabel('Win GT', fontsize=9)
    ax_win_gt.set_ylim(-0.05, 1.1)
    ax_win_gt.set_xlabel('Simulation Time (hours)', fontsize=12)
    ax_win_gt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


# ── Sensor Ablation ──────────────────────────────────────────────────────────

def run_sensor_ablation(monitors, sim_records):
    """
    Test: how many observable sensors can fail before regime detection breaks?
    Progressively drop sensors and measure detection accuracy.
    """
    ablation_order = ['co2', 'noisedb', 'lightlevel', 'airquality', 'humidity', 'temperature']

    results = []
    remaining = list(SENSOR_VARS)

    # Full sensors
    records = run_factored_monitoring(monitors, sim_records, sensor_vars=remaining)
    metrics = compute_detection_metrics(records)
    results.append({
        'dropped': 'none',
        'n_sensors': len(remaining),
        'sensors': list(remaining),
        'occ_accuracy': metrics['occ_accuracy'],
        'win_accuracy': metrics['win_accuracy'],
        'combined_accuracy': metrics['combined_accuracy'],
    })
    logger.info(f"  Full ({len(remaining)} sensors): "
                f"occ={metrics['occ_accuracy']}%, win={metrics['win_accuracy']}%, "
                f"combined={metrics['combined_accuracy']}%")

    for sensor in ablation_order:
        if sensor in remaining:
            remaining.remove(sensor)
            if len(remaining) == 0:
                break
            records = run_factored_monitoring(monitors, sim_records,
                                              sensor_vars=remaining)
            metrics = compute_detection_metrics(records)
            results.append({
                'dropped': sensor,
                'n_sensors': len(remaining),
                'sensors': list(remaining),
                'occ_accuracy': metrics['occ_accuracy'],
                'win_accuracy': metrics['win_accuracy'],
                'combined_accuracy': metrics['combined_accuracy'],
            })
            logger.info(f"  Drop {sensor} ({len(remaining)} left): "
                        f"occ={metrics['occ_accuracy']}%, "
                        f"win={metrics['win_accuracy']}%, "
                        f"combined={metrics['combined_accuracy']}%")

    return results


def plot_sensor_ablation(ablation_results, output_path):
    """Plot sensor ablation curve with occ/win/combined lines."""
    fig, ax = plt.subplots(figsize=(12, 7))

    x = range(len(ablation_results))
    occ_acc = [r['occ_accuracy'] for r in ablation_results]
    win_acc = [r['win_accuracy'] for r in ablation_results]
    comb_acc = [r['combined_accuracy'] for r in ablation_results]
    labels = ['all'] + [f'−{r["dropped"]}' for r in ablation_results[1:]]

    ax.plot(x, occ_acc, 'o-', color='#e74c3c', lw=2.5, markersize=8,
            label='Occupancy detection')
    ax.plot(x, win_acc, 's-', color='#2ecc71', lw=2.5, markersize=8,
            label='Window detection')
    ax.plot(x, comb_acc, 'D-', color='#9b59b6', lw=2.5, markersize=8,
            label='Combined 4-way')
    ax.axhline(50, color='gray', ls='--', lw=1, alpha=0.5, label='Random (binary)')
    ax.axhline(25, color='gray', ls=':', lw=1, alpha=0.5, label='Random (4-way)')

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel('Sensor Dropped (cumulative)', fontsize=12)
    ax.set_ylabel('Detection Accuracy (%)', fontsize=12)
    ax.set_title('Sensor Ablation — Regime Detection vs. Observable Sensors',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='lower left', fontsize=10)

    # Annotate values on combined line
    for i, (ca, oa, wa) in enumerate(zip(comb_acc, occ_acc, win_acc)):
        ax.annotate(f'{ca:.0f}%', (i, ca), textcoords="offset points",
                    xytext=(0, -15), ha='center', fontsize=8, color='#9b59b6')

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def load_discovered_edges():
    """Load PolicyGRID discovered edges from cache."""
    cache_file = 'building_rich_grid_edges_cache.json'
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"{cache_file} not found. Run PolicyGRID discovery first:\n"
            "  python building_rich_monitoring.py --discover"
        )
    with open(cache_file) as f:
        cached = json.load(f)
    edges = {}
    for k, v in cached.items():
        edges[k] = {tuple(e) for e in v}
    return edges


def run_discovery():
    """Run PolicyGRID discovery and cache edges."""
    import random
    from src.pipeline_cwm import create_cwm_pipeline

    data = pd.read_csv('data_regen/smart_building_rich_processed.csv')
    random.seed(42); np.random.seed(42)

    logger.info("Running PolicyGRID discovery (Phase 1-2)...")
    pipeline = create_cwm_pipeline(
        csv_data=data,
        api_key='YOUR_OPENAI_API_KEY',
        smart_room_path=os.path.abspath('js/smart_building_rich.js'),
        dataset_type='smart_building_rich',
        max_iterations=10, alpha=0.5, beta=0.5, effect_threshold=0.1,
        actuator_vars=['hvacpower', 'lightingpower'],
        non_intervenable_vars=[
            'outdoortemp', 'solarradiation', 'occupancy', 'windowposition',
            'pmv', 'energyconsumption', 'satisfaction', 'overallsatisfaction',
        ],
    )

    final_dag, _ = pipeline.run_with_cwm()

    validated = {(s.lower(), t.lower()) for s, t in pipeline.pipeline.validated_edges}
    obs_only = set()
    for G in pipeline.pipeline.get_method_dags().values():
        obs_only.update(G.edges())
    obs_only = {(s.lower(), t.lower()) for s, t in obs_only}

    cache = {
        'GRID_validated': [list(e) for e in validated],
        'GRID_obs_only': [list(e) for e in obs_only],
    }
    with open('building_rich_grid_edges_cache.json', 'w') as f:
        json.dump(cache, f, indent=2)

    logger.info(f"Cached: validated={len(validated)}, obs_only={len(obs_only)}")
    return {'GRID_validated': validated, 'GRID_obs_only': obs_only}


def _graph_quality(edges, label):
    """Print graph quality metrics."""
    tp = len(edges & GT_SET)
    fp = len(edges - GT_SET)
    fn = len(GT_SET - edges)
    p = tp/(tp+fp) if (tp+fp)>0 else 0
    r = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*p*r/(p+r) if (p+r)>0 else 0
    logger.info(f"  {label}: {len(edges)} edges, TP={tp}, FP={fp}, FN={fn}, F1={f1:.3f}")
    return {'label': label, 'n_edges': len(edges), 'tp': tp, 'fp': fp, 'fn': fn, 'f1': round(f1, 3)}


def _run_monitoring_for_graph(edge_set, label, data, sim_records, out_dir):
    """Build monitors, run monitoring + ablation for one graph. Return metrics."""
    logger.info(f"\n{'─'*60}")
    logger.info(f"  GRAPH: {label}")
    logger.info(f"{'─'*60}")

    monitors = construct_factored_monitors(edge_set, data)

    records = run_factored_monitoring(monitors, sim_records)
    metrics = compute_detection_metrics(records)

    logger.info(f"  Occ={metrics['occ_accuracy']}%, "
                f"Win={metrics['win_accuracy']}%, "
                f"Combined={metrics['combined_accuracy']}%")

    # Plot
    safe_label = label.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plot_factored_monitoring(
        records,
        f'{label} — Factored Regime Detection',
        os.path.join(out_dir, f'monitor_{safe_label}.png'))

    # Ablation
    ablation = run_sensor_ablation(monitors, sim_records)
    plot_sensor_ablation(
        ablation,
        os.path.join(out_dir, f'ablation_{safe_label}.png'))

    # Save CSVs
    pd.DataFrame(records).to_csv(
        os.path.join(out_dir, f'monitoring_{safe_label}.csv'), index=False)
    pd.DataFrame(ablation).to_csv(
        os.path.join(out_dir, f'ablation_{safe_label}.csv'), index=False)

    return {
        'label': label,
        'occ_accuracy': metrics['occ_accuracy'],
        'win_accuracy': metrics['win_accuracy'],
        'combined_accuracy': metrics['combined_accuracy'],
        'ablation': ablation,
    }


def main():
    data_path = 'data_regen/smart_building_rich_processed.csv'
    sim_path = 'js/smart_building_rich.js'
    scaling_path = 'data_regen/smart_building_rich_processed_scaling.csv'
    out_dir = 'results/monitoring'
    os.makedirs(out_dir, exist_ok=True)

    data = pd.read_csv(data_path)
    try:
        scaling = pd.read_csv(scaling_path)
    except Exception:
        scaling = None

    # ── Step 1: Load discovered edges ────────────────────────────────────
    logger.info("="*70)
    logger.info("  STEP 1: LOAD EDGES")
    logger.info("="*70 + "\n")

    try:
        discovered = load_discovered_edges()
    except FileNotFoundError:
        logger.info("No cached edges — running discovery...")
        discovered = run_discovery()

    obs_only = discovered.get('GRID_obs_only', set())
    validated = discovered.get('GRID_validated', set())

    _graph_quality(obs_only, 'GRID_obs_only')
    _graph_quality(validated, 'GRID_validated')
    _graph_quality(GT_SET, 'Ground Truth')

    # ── Step 2: Collect simulator data (once, reuse for all graphs) ──────
    logger.info("\n" + "="*70)
    logger.info("  STEP 2: COLLECT SIMULATOR DATA (3 days)")
    logger.info("="*70 + "\n")

    sim_records = collect_sim_data(sim_path, scaling, duration_steps=4320)

    # ── Step 3: Run monitoring for each graph ────────────────────────────
    logger.info("\n" + "="*70)
    logger.info("  STEP 3: FACTORED MONITORING — ALL GRAPHS")
    logger.info("="*70)

    all_results = []

    # 3a: GRID obs_only (union of all methods)
    r = _run_monitoring_for_graph(obs_only, 'GRID obs_only', data, sim_records, out_dir)
    all_results.append(r)

    # 3b: GRID validated (intervention-confirmed edges only)
    r = _run_monitoring_for_graph(validated, 'GRID validated', data, sim_records, out_dir)
    all_results.append(r)

    # 3c: Ground truth (upper bound)
    r = _run_monitoring_for_graph(GT_SET, 'Ground Truth', data, sim_records, out_dir)
    all_results.append(r)

    # ── Step 4: Summary comparison ───────────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info("  COMPARISON SUMMARY")
    logger.info("="*70 + "\n")

    logger.info(f"{'Graph':<20} {'Occ%':>6} {'Win%':>6} {'Comb%':>6}")
    logger.info("-" * 42)
    summary_rows = []
    for r in all_results:
        logger.info(f"{r['label']:<20} {r['occ_accuracy']:>6.1f} "
                    f"{r['win_accuracy']:>6.1f} {r['combined_accuracy']:>6.1f}")
        summary_rows.append({
            'graph': r['label'],
            'occ_accuracy': r['occ_accuracy'],
            'win_accuracy': r['win_accuracy'],
            'combined_accuracy': r['combined_accuracy'],
        })

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, 'comparison_summary.csv'), index=False)

    logger.info(f"\nAll results saved to {out_dir}/")
    logger.info("\n" + "="*70)
    logger.info("  DONE")
    logger.info("="*70)


if __name__ == '__main__':
    import sys
    if '--discover' in sys.argv:
        run_discovery()
    else:
        main()
