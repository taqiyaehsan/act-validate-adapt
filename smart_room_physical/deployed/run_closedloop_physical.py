#!/usr/bin/env python3
"""
Physical Closed-Loop Policy with Regime Changes
================================================
Runs PolicyGRID (validated), PolicyGRID-O (obs-only), PID, and All-OFF
in 30-min episodes while a person walks in/out and opens/closes the window.

BEFORE RUNNING:
  1. Start BME680 server:    python drivers/bme_server/server.py
  2. Start room logger:      python drivers/room_logger/server.py
  3. Verify sensors:         python -c "from policygrid_adapter import read_all_sensors; print(read_all_sensors())"
  4. Open room logger in browser: http://localhost:5000
  5. Have a person ready to enter/exit and open/close window

REGIME PROTOCOL (same for EVERY episode):
  0-10 min:  Base (empty room, window closed)
  10-20 min: Person enters, logs on app → Occupied
  15-25 min: Open window, log on app → Full (occupied + window)
  20 min:    Person leaves, logs on app → Window-only
  25 min:    Close window, log on app → Base
  25-30 min: Base (recovery)

Usage:
    python run_closedloop_physical.py                  # Full run
    python run_closedloop_physical.py --method Validated --epsilon 1.0  # Single
    python run_closedloop_physical.py --episode-min 20  # Shorter episodes
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
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'closedloop_physical')

VALIDATED_EDGES_PATH = os.path.join(SCRIPT_DIR, 'results', 'discovery',
                                     'seed_42', 'validated_edges.json')
OBS_EDGES_PATH = os.path.join(SCRIPT_DIR, 'results', 'discovery',
                               'benchmark_edges', 'policygrid_o.json')
EVIDENCE_PATH = os.path.join(SCRIPT_DIR, 'results', 'discovery',
                              'seed_42', 'edge_evidence.json')

ACTUATORS = ['Heater', 'Humidifier', 'Fan']
EPISODE_SEC = 30 * 60
WASHOUT_SEC = 15 * 60
MEASUREMENT_INTERVAL = 60
RIDGE_ALPHA = 0.5
OPTIMIZER_STEPS = 50

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


# ── SEM fitting ──
from sklearn.linear_model import Ridge

def _find_col(df, var_name):
    for c in df.columns:
        if c.lower() == var_name.lower():
            return c
    return None


def fit_sem(data, edges, evidence=None):
    """Fit Ridge SEM. Same as run_policygrid_policy.py."""
    if evidence is None:
        evidence = {}
    target_parents = {}
    for src, tgt in edges:
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
        for i, parent in enumerate(parents):
            key = f"{parent}->{tgt}"
            ev = evidence.get(key, {})
            delta = ev.get('mean_delta')
            if delta is not None and abs(coefs[i]) < 0.01:
                coefs[i] = (1 if delta > 0 else -1) * 0.01
        sem[tgt] = {
            'parents': parents, 'parent_cols': parent_cols,
            'coefs': [round(c, 6) for c in coefs],
            'intercept': round(float(reg.intercept_), 6),
            'r2': round(float(reg.score(X, y)), 4),
        }
        logger.info(f"  SEM: {' + '.join(parents)} → {tgt} (R²={sem[tgt]['r2']:.3f})")
    return sem


def gradient_policy(sem, epsilon):
    """Gradient descent on SEM → binary device commands."""
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


def pid_policy(state):
    temp = state.get('Temperature', 22)
    return {'Heater': temp < 21.5, 'Humidifier': False, 'Fan': False}


# ── Episode runner with regime logging ──
def run_episode(method_name, policy_fn, duration_sec):
    """
    Run one closed-loop episode on real hardware.
    policy_fn(state) → {device: bool} at each step.
    Reads Occupancy and WindowState from the room logger app.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  Episode: {method_name} ({duration_sec // 60} min)")
    logger.info(f"  START REGIME PROTOCOL NOW")
    logger.info(f"  0-10min: Base | 10min: Person IN | 15min: Window OPEN")
    logger.info(f"  20min: Person OUT | 25min: Window CLOSE | 25-30min: Base")
    logger.info(f"{'='*60}")

    all_off()
    time.sleep(3)

    steps = duration_sec // MEASUREMENT_INTERVAL
    records = []

    for step in range(steps):
        # Read sensors + regime state from room logger
        state = read_all_sensors()

        # Apply policy
        cmds = policy_fn(state)
        for device, should_on in cmds.items():
            if should_on:
                device_on(device)
            else:
                device_off(device)

        # Determine regime from room logger
        occ = state.get('Occupancy', 0)
        win = state.get('WindowState', 0)
        occ_active = int(occ >= 1) if occ is not None else 0
        win_active = int(win >= 1) if win is not None else 0
        regime = {(0,0):'base', (1,0):'occ', (0,1):'win', (1,1):'full'
                 }[(occ_active, win_active)]

        record = {
            'timestamp': state.get('timestamp', datetime.now().isoformat()),
            'step': step,
            'minute': step,
            'method': method_name,
            'Temperature': state.get('Temperature'),
            'Humidity': state.get('Humidity'),
            'AirQuality': state.get('AirQuality'),
            'EnergyConsumption': state.get('EnergyConsumption'),
            'Satisfaction': state.get('Satisfaction'),
            'Occupancy': occ,
            'WindowState': win,
            'regime': regime,
            'Heater': cmds.get('Heater', False),
            'Humidifier': cmds.get('Humidifier', False),
            'Fan': cmds.get('Fan', False),
        }
        records.append(record)

        # Log every step
        devices_on = [d for d, v in cmds.items() if v]
        logger.info(f"  [{step:2d}/{steps}] {regime:5s} | "
                    f"T={state.get('Temperature', '?'):>5} "
                    f"Sat={state.get('Satisfaction', '?'):>5} "
                    f"E={state.get('EnergyConsumption', '?'):>6}W | "
                    f"ON: {devices_on if devices_on else 'none'}")

        time.sleep(MEASUREMENT_INTERVAL)

    all_off()
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default=None,
                        help='Run single method: Validated, Obs-Only, PID, All-OFF')
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--episode-min', type=int, default=30)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(f'logs/closedloop_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
            logging.StreamHandler(),
        ]
    )

    episode_sec = args.episode_min * 60
    eps = args.epsilon

    # Load data
    baseline = pd.read_csv(BASELINE_CSV)

    # Load edge sets
    with open(VALIDATED_EDGES_PATH) as f:
        val_edges_raw = json.load(f)
        # Handle both list-of-lists and dict formats
        if isinstance(val_edges_raw, dict):
            val_edges = {tuple(e) for e in val_edges_raw.get('validated_edges', [])}
        else:
            val_edges = {tuple(e) for e in val_edges_raw}
    logger.info(f"Validated edges: {len(val_edges)}")

    obs_edges = set()
    if os.path.exists(OBS_EDGES_PATH):
        with open(OBS_EDGES_PATH) as f:
            obs_raw = json.load(f)
            if isinstance(obs_raw, list):
                obs_edges = {tuple(e) for e in obs_raw}
            elif isinstance(obs_raw, dict):
                obs_edges = {tuple(e) for e in obs_raw.get('edges', [])}
        logger.info(f"Obs-only edges: {len(obs_edges)}")
    else:
        logger.warning(f"No obs-only edges at {OBS_EDGES_PATH}")

    # Load evidence for direction constraints
    evidence = {}
    if os.path.exists(EVIDENCE_PATH):
        with open(EVIDENCE_PATH) as f:
            evidence = json.load(f)

    # Fit SEMs
    logger.info("\nFitting Validated SEM...")
    val_sem = fit_sem(baseline, val_edges, evidence)
    logger.info(f"  {len(val_sem)} targets")

    obs_sem = {}
    if obs_edges:
        logger.info("Fitting Obs-Only SEM...")
        obs_sem = fit_sem(baseline, obs_edges)
        logger.info(f"  {len(obs_sem)} targets")

    # Define policy functions
    def make_grid_policy(sem, epsilon):
        cmds = gradient_policy(sem, epsilon)
        def fn(state):
            return cmds
        return fn

    methods = {
        f'Validated_eps{eps}': make_grid_policy(val_sem, eps),
        f'Obs-Only_eps{eps}': make_grid_policy(obs_sem, eps) if obs_sem else None,
        'PID': pid_policy,
        'All-OFF': lambda state: {a: False for a in ACTUATORS},
    }

    # Filter to single method if specified
    if args.method:
        key = next((k for k in methods if args.method.lower() in k.lower()), None)
        if key:
            methods = {key: methods[key]}
        else:
            logger.error(f"Unknown method: {args.method}. Available: {list(methods.keys())}")
            return

    methods = {k: v for k, v in methods.items() if v is not None}

    # Preflight
    logger.info("\nPreflight sensor check...")
    state = read_all_sensors()
    for k, v in state.items():
        logger.info(f"  {k}: {v}")
    if state.get('Temperature') is None:
        logger.error("Cannot read temperature. Check sensors.")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"  PHYSICAL CLOSED-LOOP POLICY EXPERIMENT")
    logger.info(f"  Methods: {list(methods.keys())}")
    logger.info(f"  Episode: {args.episode_min} min + {WASHOUT_SEC//60} min washout")
    logger.info(f"  Total time: ~{len(methods) * (args.episode_min + WASHOUT_SEC//60)} min")
    logger.info(f"  MAKE SURE ROOM LOGGER APP IS OPEN IN BROWSER")
    logger.info(f"{'='*60}")

    input("\nPress ENTER to start (or Ctrl+C to abort)...")

    all_dfs = []
    all_metrics = []

    try:
        for method_name, policy_fn in methods.items():
            logger.info(f"\n>>> STARTING: {method_name}")
            logger.info(f">>> Follow the regime protocol!")

            df = run_episode(method_name, policy_fn, episode_sec)
            all_dfs.append(df)

            # Per-regime metrics
            metrics = {'method': method_name}
            for regime in ['base', 'occ', 'win', 'full']:
                rdf = df[df['regime'] == regime]
                if len(rdf):
                    metrics[f'{regime}_steps'] = len(rdf)
                    metrics[f'{regime}_sat'] = round(rdf['Satisfaction'].mean(), 1) if rdf['Satisfaction'].notna().any() else None
                    metrics[f'{regime}_energy_w'] = round(rdf['EnergyConsumption'].mean(), 1) if rdf['EnergyConsumption'].notna().any() else None
            metrics['total_sat'] = round(df['Satisfaction'].mean(), 1) if df['Satisfaction'].notna().any() else None
            metrics['total_energy_wh'] = round(df['EnergyConsumption'].sum() * MEASUREMENT_INTERVAL / 3600, 1) if df['EnergyConsumption'].notna().any() else None

            all_metrics.append(metrics)
            logger.info(f"  Metrics: {metrics}")

            # Save incrementally
            pd.concat(all_dfs, ignore_index=True).to_csv(
                os.path.join(OUTPUT_DIR, 'closedloop_log.csv'), index=False)
            pd.DataFrame(all_metrics).to_csv(
                os.path.join(OUTPUT_DIR, 'closedloop_metrics.csv'), index=False)

            if method_name != list(methods.keys())[-1]:
                logger.info(f"\n  Washout {WASHOUT_SEC//60} min — all devices OFF")
                all_off()
                time.sleep(WASHOUT_SEC)

    except KeyboardInterrupt:
        logger.info("\nInterrupted")
    finally:
        all_off()
        cleanup()

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"  EXPERIMENT COMPLETE")
    logger.info(f"{'='*60}")
    for m in all_metrics:
        logger.info(f"  {m['method']}: sat={m.get('total_sat')}, Wh={m.get('total_energy_wh')}")
    logger.info(f"\n  Logs: {OUTPUT_DIR}/closedloop_log.csv")
    logger.info(f"  Metrics: {OUTPUT_DIR}/closedloop_metrics.csv")


if __name__ == '__main__':
    main()
