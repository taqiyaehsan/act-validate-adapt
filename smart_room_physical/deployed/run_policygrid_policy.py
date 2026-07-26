#!/usr/bin/env python3
"""
PolicyGRID Physical Policy Evaluation
=======================================
Fits SEM on validated edges, runs PolicyGRID at multiple ε targets,
then runs control baselines. Each method gets a 30-min hardware episode.

Usage:
    # Full policy evaluation (PolicyGRID + all baselines)
    python run_policygrid_policy.py

    # PolicyGRID only (skip baselines)
    python run_policygrid_policy.py --skip-baselines

    # Use specific edge set
    python run_policygrid_policy.py --edges results/discovery/seed_42/validated_edges.json

Prerequisites:
    - Discovery completed (validated_edges.json exists)
    - BME680 server + room logger + Kasa devices running
    - Baseline data at data/baseline_data.csv
    - Someone nearby for safety
"""

import os
import sys
import json
import time
import logging
import argparse
import atexit
import signal
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from policygrid_adapter import (
    read_all_sensors, device_on, device_off, all_off, cleanup
)

logger = logging.getLogger(__name__)

# ── Config ──
BASELINE_CSV = os.path.join(SCRIPT_DIR, 'data', 'baseline_data.csv')
SCALING_CSV = os.path.join(SCRIPT_DIR, 'data', 'baseline_data_scaling.csv')
EDGES_PATH = os.path.join(SCRIPT_DIR, 'results', 'discovery', 'seed_42', 'validated_edges.json')
EVIDENCE_PATH = os.path.join(SCRIPT_DIR, 'results', 'discovery', 'seed_42', 'edge_evidence.json')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'policy')

ACTUATORS = ['Heater', 'Humidifier', 'Fan']
EPISODE_SEC = 30 * 60       # 30 min per method
WASHOUT_SEC = 15 * 60       # 15 min between episodes
MEASUREMENT_INTERVAL = 60   # 1 min
EPSILON_TARGETS = [0.5, 1.0, 2.0]
OPTIMIZER_STEPS = 50
RIDGE_ALPHA = 0.5


# ── Safety ──
def _emergency_off():
    try:
        all_off()
    except:
        pass

atexit.register(_emergency_off)
signal.signal(signal.SIGINT, lambda s, f: (all_off(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (all_off(), sys.exit(0)))


# ── Scaling ──
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


# ── SEM fitting ──
def fit_sem(data, validated_edges, evidence):
    """Fit Ridge SEM on validated edges with direction constraints."""
    target_parents = {}
    for src, tgt in validated_edges:
        if tgt not in target_parents:
            target_parents[tgt] = []
        target_parents[tgt].append(src)

    sem = {}
    for tgt, parents in target_parents.items():
        tgt_col = _find_col(data, tgt)
        parent_cols = [_find_col(data, p) for p in parents]
        parent_cols = [c for c in parent_cols if c is not None]
        if tgt_col is None or not parent_cols:
            continue

        X = data[parent_cols].values
        y = data[tgt_col].values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        if len(X) < 10:
            continue

        reg = Ridge(alpha=RIDGE_ALPHA).fit(X, y)
        coefs = list(reg.coef_)

        # Direction constraints from intervention evidence
        for i, parent in enumerate(parents):
            key = f"{parent}->{tgt}"
            ev = evidence.get(key, {})
            delta = ev.get('mean_delta')
            if delta is not None and abs(coefs[i]) < 0.01:
                coefs[i] = (1 if delta > 0 else -1) * 0.01

        sem[tgt] = {
            'parents': parents,
            'parent_cols': parent_cols,
            'coefs': [round(c, 6) for c in coefs],
            'intercept': round(float(reg.intercept_), 6),
            'r2': round(float(reg.score(X, y)), 4),
        }
        logger.info(f"  SEM: {' + '.join(parents)} → {tgt} (R²={sem[tgt]['r2']:.3f})")

    return sem


# ── PolicyGRID gradient policy ──
def policygrid_policy(sem, epsilon):
    """Gradient descent on SEM. Returns binary device commands."""
    lam = 1.0 / max(epsilon, 0.1)
    actions = {a.lower(): 0.5 for a in ACTUATORS}

    for _ in range(OPTIMIZER_STEPS):
        new_actions = dict(actions)
        for actuator in [a.lower() for a in ACTUATORS]:
            grad_sat, grad_eng = 0.0, 0.0
            for tgt, model in sem.items():
                parent_lower = [p.lower() for p in model['parents']]
                if actuator in parent_lower:
                    idx = parent_lower.index(actuator)
                    coef = model['coefs'][idx]
                    if tgt.lower() == 'satisfaction':
                        grad_sat += coef
                    elif tgt.lower() == 'energyconsumption':
                        grad_eng += coef
                    # Indirect paths
                    for tgt2, model2 in sem.items():
                        p2_lower = [p.lower() for p in model2['parents']]
                        if tgt.lower() in p2_lower:
                            idx2 = p2_lower.index(tgt.lower())
                            if tgt2.lower() == 'satisfaction':
                                grad_sat += coef * model2['coefs'][idx2]
                            elif tgt2.lower() == 'energyconsumption':
                                grad_eng += coef * model2['coefs'][idx2]
            grad = grad_sat - lam * grad_eng
            new_actions[actuator] = max(0.0, min(1.0, actions[actuator] + 0.05 * grad))
        actions = new_actions

    return {a: actions.get(a.lower(), 0.5) > 0.5 for a in ACTUATORS}


# ── Control baselines ──
def pid_policy(state):
    """Bang-bang thermostat: heater ON if T < 21.5°C, OFF if T > 22.5°C."""
    temp = state.get('Temperature', 22)
    return {'Heater': temp < 21.5, 'Humidifier': False, 'Fan': False}


# ── Episode runner ──
def run_episode(method_name, device_commands, duration_sec, smap,
                reactive_fn=None):
    """
    Run one control episode on real hardware.
    Returns (metrics_dict, episode_df).
    """
    logger.info(f"\n{'='*40}")
    logger.info(f"  Episode: {method_name} ({duration_sec // 60} min)")
    logger.info(f"{'='*40}")

    all_off()
    time.sleep(5)

    # Apply static commands
    if device_commands and not reactive_fn:
        for device, should_on in device_commands.items():
            if should_on:
                device_on(device)
            else:
                device_off(device)
        logger.info(f"  Commands: {device_commands}")

    steps = duration_sec // MEASUREMENT_INTERVAL
    records = []

    for step in range(steps):
        state = read_all_sensors()

        # Reactive control (PID)
        if reactive_fn:
            cmds = reactive_fn(state)
            for device, should_on in cmds.items():
                if should_on:
                    device_on(device)
                else:
                    device_off(device)

        state['step'] = step
        state['method'] = method_name
        records.append(state)

        if step % 5 == 0:
            logger.info(f"    Step {step}/{steps}: "
                        f"T={state.get('Temperature', '?')}, "
                        f"Sat={state.get('Satisfaction', '?')}, "
                        f"E={state.get('EnergyConsumption', '?')}W")

        time.sleep(MEASUREMENT_INTERVAL)

    all_off()
    df = pd.DataFrame(records)

    # Compute metrics in normalized units
    metrics = {'method': method_name}
    for var in ['Temperature', 'Satisfaction', 'EnergyConsumption']:
        if var in df.columns:
            raw_vals = df[var].dropna()
            if len(raw_vals) > 0:
                metrics[f'mean_{var.lower()}'] = round(float(raw_vals.mean()), 2)

    # Energy in Wh
    if 'EnergyConsumption' in df.columns:
        power_w = df['EnergyConsumption'].dropna()
        wh = float(power_w.sum() * MEASUREMENT_INTERVAL / 3600)
        metrics['total_energy_wh'] = round(wh, 2)

    metrics['n_steps'] = len(df)

    logger.info(f"  {method_name}: T={metrics.get('mean_temperature', '?')}, "
                f"Sat={metrics.get('mean_satisfaction', '?')}, "
                f"Wh={metrics.get('total_energy_wh', '?')}")

    return metrics, df


# ── Main ──
def main():
    parser = argparse.ArgumentParser(description='PolicyGRID Physical Policy')
    parser.add_argument('--edges', default=EDGES_PATH,
                        help='Path to validated_edges.json')
    parser.add_argument('--skip-baselines', action='store_true',
                        help='Skip control baselines, only run PolicyGRID')
    parser.add_argument('--episode-min', type=int, default=30,
                        help='Episode duration in minutes (default: 30)')
    parser.add_argument('--epsilon', type=str, default=None,
                        help='Comma-separated ε targets (default: 0.5,1.0,2.0)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(f'logs/policy_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
            logging.StreamHandler(),
        ]
    )

    episode_sec = args.episode_min * 60
    eps_targets = [float(e) for e in args.epsilon.split(',')] if args.epsilon else EPSILON_TARGETS

    logger.info("=" * 60)
    logger.info("  PolicyGRID Physical Policy Evaluation")
    logger.info(f"  Edges: {args.edges}")
    logger.info(f"  Episode: {args.episode_min} min")
    logger.info(f"  ε targets: {eps_targets}")
    logger.info(f"  Baselines: {'YES' if not args.skip_baselines else 'NO'}")
    logger.info("=" * 60)

    # Load edges + evidence
    if not os.path.exists(args.edges):
        logger.error(f"No edges at {args.edges}. Run discovery first.")
        return

    with open(args.edges) as f:
        validated = {tuple(e) for e in json.load(f)}
    logger.info(f"Loaded {len(validated)} validated edges")

    evidence = {}
    evidence_path = os.path.join(os.path.dirname(args.edges), 'edge_evidence.json')
    if os.path.exists(evidence_path):
        with open(evidence_path) as f:
            evidence = json.load(f)

    # Load baseline data
    baseline = pd.read_csv(BASELINE_CSV)
    smap = load_scaling()

    # Preflight
    logger.info("\nPreflight...")
    state = read_all_sensors()
    available = [k for k, v in state.items() if v is not None]
    logger.info(f"  Sensors: {available}")
    if len(available) < 4:
        logger.error("Too few sensors. Aborting.")
        return

    all_metrics = []
    all_logs = []

    try:
        # ── Fit SEM ──
        logger.info("\nFitting SEM on validated edges...")
        sem = fit_sem(baseline, validated, evidence)
        with open(os.path.join(OUTPUT_DIR, 'sem_coefficients.json'), 'w') as f:
            json.dump(sem, f, indent=2)
        logger.info(f"SEM: {len(sem)} target variables")

        if not sem:
            logger.error("Empty SEM — no edges to build policy from.")
            return

        # ── PolicyGRID at each ε ──
        for eps in eps_targets:
            cmds = policygrid_policy(sem, eps)
            logger.info(f"\nPolicyGRID ε={eps}: {cmds}")
            m, df = run_episode(f'PolicyGRID_eps{eps}', cmds, episode_sec, smap)
            m['epsilon'] = eps
            all_metrics.append(m)
            all_logs.append(df)

            logger.info(f"  Washout {WASHOUT_SEC // 60} min...")
            time.sleep(WASHOUT_SEC)

        # ── Baselines ──
        if not args.skip_baselines:
            # All-OFF
            m, df = run_episode('All-OFF',
                                {a: False for a in ACTUATORS},
                                episode_sec, smap)
            all_metrics.append(m)
            all_logs.append(df)
            time.sleep(WASHOUT_SEC)

            # PID
            m, df = run_episode('PID', None, episode_sec, smap,
                                reactive_fn=pid_policy)
            all_metrics.append(m)
            all_logs.append(df)
            time.sleep(WASHOUT_SEC)

            # All-ON (max comfort, max energy)
            m, df = run_episode('All-ON',
                                {a: True for a in ACTUATORS},
                                episode_sec, smap)
            all_metrics.append(m)
            all_logs.append(df)

        # ── Save ──
        pd.DataFrame(all_metrics).to_csv(
            os.path.join(OUTPUT_DIR, 'policy_metrics.csv'), index=False)

        if all_logs:
            pd.concat(all_logs, ignore_index=True).to_csv(
                os.path.join(OUTPUT_DIR, 'policy_log.csv'), index=False)

        logger.info("\n" + "=" * 60)
        logger.info("  POLICY EVALUATION COMPLETE")
        logger.info(f"  {'Method':<25} {'ε':>5} {'Sat':>8} {'Wh':>8}")
        for m in all_metrics:
            logger.info(f"  {m.get('method', '?'):<25} "
                        f"{str(m.get('epsilon', '-')):>5} "
                        f"{str(m.get('mean_satisfaction', '?')):>8} "
                        f"{str(m.get('total_energy_wh', '?')):>8}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Policy failed: {e}", exc_info=True)
    finally:
        all_off()
        cleanup()


def _find_col(df, var_name):
    for c in df.columns:
        if c.lower() == var_name.lower():
            return c
    return None


if __name__ == '__main__':
    main()
