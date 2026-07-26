#!/usr/bin/env python3
"""
Physical testbed monitoring evaluation.

Tests two training strategies:
  A) Train on baseline (April 6-8), test on regime episode (April 9)
  B) Train on first half of regime data, test on second half

Both use the same edge set (16 edges from discovery) and the same
factored Bayesian monitoring pipeline as the simulation.

Usage:
    python run_physical_monitoring.py
"""
import os, sys, json, logging
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring.log'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ── Config ──
EDGES_PATH = 'results/discovery/seed_42/validated_edges.json'
BASELINE_CSV = 'data/baseline_data.csv'
REGIME_CSV = 'data/sensor_data_regime_monitoring.csv'
SCALING_CSV = 'data/baseline_data_scaling.csv'
OUTPUT_DIR = 'results/monitoring'

GRAPH_VARS = ['Heater', 'Humidifier', 'Fan', 'Temperature', 'Humidity',
              'AirQuality', 'EnergyConsumption', 'Satisfaction',
              'Occupancy', 'WindowState']

# Observable vars for monitoring (exclude regime vars — those are what we detect)
OBS_VARS = [v.lower() for v in GRAPH_VARS if v not in ('Occupancy', 'WindowState')]

# Monitoring hyperparams (same as sim)
SIGMA = 2.0
FORGET = 0.12

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)


def preprocess_regime_data(raw_csv, scaling_csv):
    """Preprocess raw regime CSV to normalized 1-min data matching baseline format."""
    raw = pd.read_csv(raw_csv)
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])

    # Load scaling
    sc = pd.read_csv(scaling_csv)
    smap = {row['column'].lower(): (row['data_min'], row['data_max']) for _, row in sc.iterrows()}

    # Downsample to 1 min
    raw = raw.set_index('timestamp').resample('1min').mean().reset_index()
    raw = raw.dropna(subset=['govee_temp_1'])

    rows = []
    for _, r in raw.iterrows():
        row = {'timestamp': str(r['timestamp'])}

        row['Heater'] = float(r.get('heater_on', 0))
        row['Humidifier'] = float(r.get('humidifier_on', 0))
        row['Fan'] = float(r.get('fan_on', 0))

        temps = [v for k, v in r.items() if 'govee_temp' in str(k) and pd.notna(v)]
        humids = [v for k, v in r.items() if 'govee_humidity' in str(k) and pd.notna(v)]
        raw_temp = np.mean(temps) if temps else None
        raw_humid = np.mean(humids) if humids else None
        raw_aq = r.get('bme_iaq', None)
        raw_energy = r.get('kasa_current_power_w', 0)

        for var, raw_val in [('Temperature', raw_temp), ('Humidity', raw_humid),
                              ('AirQuality', raw_aq), ('EnergyConsumption', raw_energy)]:
            if raw_val is not None and var.lower() in smap:
                mn, mx = smap[var.lower()]
                row[var] = (raw_val - mn) / (mx - mn + 1e-12)
            else:
                row[var] = np.nan

        # Satisfaction (ISO 7730)
        if raw_temp is not None and raw_humid is not None:
            met = 1.2
            pmv = 0.303 * np.exp(-0.036 * met * 58.15) + 0.028
            t_cl = 35.7 - 0.028 * met * 58.15
            h_c = max(2.38 * abs(t_cl - raw_temp) ** 0.25, 12.1 * np.sqrt(0.1))
            tl = (met * 58.15
                  - 3.05e-3 * (5733 - 6.99 * met * 58.15
                                - max(raw_humid * 0.01 * np.exp(16.6536 - 4030.183 / (raw_temp + 235)), 0))
                  - 0.42 * (met * 58.15 - 58.15)
                  - 1.7e-5 * met * 58.15 * (5867
                                - max(raw_humid * 0.01 * np.exp(16.6536 - 4030.183 / (raw_temp + 235)), 0))
                  - 0.0014 * met * (34 - raw_temp)
                  - 3.96e-8 * ((t_cl + 273) ** 4 - (raw_temp + 273) ** 4)
                  - h_c * (t_cl - raw_temp))
            pmv_val = max(-3, min(3, pmv * tl))
            sat_raw = max(0, 100 - (abs(pmv_val) / 3.0) * 100)
            if 'satisfaction' in smap:
                mn, mx = smap['satisfaction']
                row['Satisfaction'] = (sat_raw - mn) / (mx - mn + 1e-12)
            else:
                row['Satisfaction'] = sat_raw / 100.0
        else:
            row['Satisfaction'] = np.nan

        row['Occupancy'] = float(r.get('occupancy', 0))
        row['WindowState'] = float(r.get('window_open', 0))
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.dropna(subset=['Temperature', 'Humidity'])
    return df


def build_4way_monitors(train_data, edges):
    """Build factored Bayesian monitors: per-regime Ridge coefficients."""
    edge_list = [(s.lower(), t.lower()) for s, t in edges]

    # Temporal: predict X(t+1) from parents at t
    cols = [c for c in train_data.columns if c.lower() in OBS_VARS]
    tdf = pd.DataFrame()
    for c in cols:
        tdf[c.lower()] = train_data[c].values[:-1]
        tdf[c.lower() + '_next'] = train_data[c].values[1:]

    # Regime labels from training data
    occ = train_data['Occupancy'].values[:-1]
    win = train_data['WindowState'].values[:-1]

    masks = {
        'base': (occ < 0.5) & (win < 0.5),
        'occ': (occ >= 0.5) & (win < 0.5),
        'win': (occ < 0.5) & (win >= 0.5),
        'full': (occ >= 0.5) & (win >= 0.5),
    }

    monitors = {}
    for regime, mask in masks.items():
        regime_data = tdf[mask]
        n_regime = len(regime_data)
        if n_regime < 20:
            logger.warning(f"  {regime}: only {n_regime} rows, falling back to all data")
            regime_data = tdf

        models = {}
        for s, t in edge_list:
            if s in OBS_VARS and t in OBS_VARS:
                if s in regime_data.columns and t + '_next' in regime_data.columns:
                    X = regime_data[[s]].values
                    y = regime_data[t + '_next'].values
                    if len(X) > 5:
                        reg = Ridge(alpha=0.5).fit(X, y)
                        models[(s, t)] = {'coef': reg.coef_[0], 'intercept': reg.intercept_}

        monitors[regime] = {'models': models, 'n_train': n_regime}
        logger.info(f"  {regime}: {n_regime} rows, {len(models)} edge models")

    return monitors


def predict_error(monitors, regime, obs_row, obs_next):
    """Compute L1 prediction error for a regime's models."""
    models = monitors[regime]['models']
    error = 0.0
    n = 0
    for (s, t), m in models.items():
        if s in obs_row and t in obs_next:
            pred = m['coef'] * obs_row[s] + m['intercept']
            error += abs(obs_next[t] - pred)
            n += 1
    return error / max(n, 1)


def run_monitoring(monitors, test_data):
    """Run factored Bayesian monitoring on test episode."""
    # Initialize posteriors
    p_occ = 0.5
    p_win = 0.5

    records = []
    cols_lower = {c: c.lower() for c in test_data.columns}

    for i in range(len(test_data) - 1):
        row = {cols_lower[c]: test_data[c].values[i] for c in test_data.columns if c.lower() in OBS_VARS}
        row_next = {cols_lower[c]: test_data[c].values[i + 1] for c in test_data.columns if c.lower() in OBS_VARS}

        # GT regime
        gt_occ = test_data['Occupancy'].values[i] >= 0.5
        gt_win = test_data['WindowState'].values[i] >= 0.5

        # Compute errors for all 4 regimes
        errors = {}
        for regime in ['base', 'occ', 'win', 'full']:
            errors[regime] = predict_error(monitors, regime, row, row_next)

        # Factored Bayesian update
        # Occupancy dimension: compare (base+win) vs (occ+full)
        err_occ_off = min(errors['base'], errors['win'])
        err_occ_on = min(errors['occ'], errors['full'])
        ll_occ_on = np.exp(-SIGMA * err_occ_on)
        ll_occ_off = np.exp(-SIGMA * err_occ_off)

        p_occ = (ll_occ_on * p_occ) / (ll_occ_on * p_occ + ll_occ_off * (1 - p_occ) + 1e-12)

        # Window dimension: compare (base+occ) vs (win+full)
        err_win_off = min(errors['base'], errors['occ'])
        err_win_on = min(errors['win'], errors['full'])
        ll_win_on = np.exp(-SIGMA * err_win_on)
        ll_win_off = np.exp(-SIGMA * err_win_off)

        p_win = (ll_win_on * p_win) / (ll_win_on * p_win + ll_win_off * (1 - p_win) + 1e-12)

        # Forgetting
        p_occ = (1 - FORGET) * p_occ + FORGET * 0.5
        p_win = (1 - FORGET) * p_win + FORGET * 0.5

        # Predicted regime
        pred_occ = p_occ >= 0.5
        pred_win = p_win >= 0.5

        records.append({
            'step': i,
            'gt_occ': gt_occ, 'gt_win': gt_win,
            'pred_occ': pred_occ, 'pred_win': pred_win,
            'p_occ': round(p_occ, 4), 'p_win': round(p_win, 4),
            'err_base': round(errors['base'], 6),
            'err_occ': round(errors['occ'], 6),
            'err_win': round(errors['win'], 6),
            'err_full': round(errors['full'], 6),
        })

    return pd.DataFrame(records)


def compute_metrics(records):
    """Compute monitoring metrics from records."""
    df = records

    # Per-dimension accuracy
    occ_correct = (df['gt_occ'] == df['pred_occ']).mean()
    win_correct = (df['gt_win'] == df['pred_win']).mean()

    # 4-way accuracy
    combined = ((df['gt_occ'] == df['pred_occ']) & (df['gt_win'] == df['pred_win'])).mean()

    # Brier scores
    brier_occ = ((df['p_occ'] - df['gt_occ'].astype(float)) ** 2).mean()
    brier_win = ((df['p_win'] - df['gt_win'].astype(float)) ** 2).mean()

    # Discriminability: fraction of steps where correct regime model has lower error
    discrim_count = 0
    total = 0
    for _, r in df.iterrows():
        # True regime
        if r['gt_occ'] and r['gt_win']:
            correct_err = r['err_full']
            wrong_err = r['err_base']
        elif r['gt_occ']:
            correct_err = r['err_occ']
            wrong_err = r['err_base']
        elif r['gt_win']:
            correct_err = r['err_win']
            wrong_err = r['err_base']
        else:
            correct_err = r['err_base']
            wrong_err = min(r['err_occ'], r['err_win'], r['err_full'])

        total += 1
        if correct_err < wrong_err:
            discrim_count += 1

    discriminability = discrim_count / max(total, 1)

    metrics = {
        'occ_accuracy': round(occ_correct * 100, 1),
        'win_accuracy': round(win_correct * 100, 1),
        'combined_accuracy': round(combined * 100, 1),
        'brier_occ': round(brier_occ, 4),
        'brier_win': round(brier_win, 4),
        'discriminability': round(discriminability * 100, 1),
        'n_test_steps': len(df),
    }
    return metrics


def main():
    logger.info("=" * 60)
    logger.info("  Physical Testbed Monitoring Evaluation")
    logger.info("=" * 60)

    # Load edges
    with open(EDGES_PATH) as f:
        edges = [tuple(e) for e in json.load(f)]
    logger.info(f"Edges: {len(edges)}")

    # Load baseline
    baseline = pd.read_csv(BASELINE_CSV)
    logger.info(f"Baseline: {len(baseline)} rows")

    # Preprocess regime data
    if not os.path.exists(REGIME_CSV):
        logger.error(f"No regime data at {REGIME_CSV}")
        return
    regime = preprocess_regime_data(REGIME_CSV, SCALING_CSV)
    logger.info(f"Regime data: {len(regime)} rows (1-min)")

    # Regime distribution in test data
    for (o, w), label in {(0, 0): 'Base', (1, 0): 'Occ+Closed',
                           (0, 1): 'Win+Open', (1, 1): 'Full'}.items():
        n = ((regime['Occupancy'] == o) & (regime['WindowState'] == w)).sum()
        logger.info(f"  {label:15s}: {n:4d} rows")

    # ============================================================
    # Strategy A: Train on baseline, test on regime
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("  STRATEGY A: Train=baseline (Apr 6-8), Test=regime (Apr 9)")
    logger.info("=" * 60)

    monitors_a = build_4way_monitors(baseline, edges)
    records_a = run_monitoring(monitors_a, regime)
    metrics_a = compute_metrics(records_a)

    logger.info(f"\n  Results (Strategy A):")
    for k, v in metrics_a.items():
        logger.info(f"    {k}: {v}")

    records_a.to_csv(os.path.join(OUTPUT_DIR, 'strategy_a_records.csv'), index=False)

    # ============================================================
    # Strategy B: Train on first half of regime, test on second half
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("  STRATEGY B: Train=regime first half, Test=regime second half")
    logger.info("=" * 60)

    half = len(regime) // 2
    regime_train = regime.iloc[:half].reset_index(drop=True)
    regime_test = regime.iloc[half:].reset_index(drop=True)
    logger.info(f"  Train: {len(regime_train)} rows, Test: {len(regime_test)} rows")

    monitors_b = build_4way_monitors(regime_train, edges)
    records_b = run_monitoring(monitors_b, regime_test)
    metrics_b = compute_metrics(records_b)

    logger.info(f"\n  Results (Strategy B):")
    for k, v in metrics_b.items():
        logger.info(f"    {k}: {v}")

    records_b.to_csv(os.path.join(OUTPUT_DIR, 'strategy_b_records.csv'), index=False)

    # ============================================================
    # Strategy C: Train on baseline + first half regime, test on second half
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("  STRATEGY C: Train=baseline+regime first half, Test=regime second half")
    logger.info("=" * 60)

    combined_train = pd.concat([baseline, regime_train], ignore_index=True)
    logger.info(f"  Train: {len(combined_train)} rows, Test: {len(regime_test)} rows")

    monitors_c = build_4way_monitors(combined_train, edges)
    records_c = run_monitoring(monitors_c, regime_test)
    metrics_c = compute_metrics(records_c)

    logger.info(f"\n  Results (Strategy C):")
    for k, v in metrics_c.items():
        logger.info(f"    {k}: {v}")

    records_c.to_csv(os.path.join(OUTPUT_DIR, 'strategy_c_records.csv'), index=False)

    # ============================================================
    # Comparison
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("  COMPARISON")
    logger.info("=" * 60)
    logger.info(f"{'Strategy':12s} {'Occ%':>6s} {'Win%':>6s} {'Comb%':>6s} {'Discr%':>7s} {'Brier_O':>8s} {'Brier_W':>8s}")
    for name, m in [('A: baseline', metrics_a), ('B: regime½', metrics_b), ('C: base+reg½', metrics_c)]:
        logger.info(f"{name:12s} {m['occ_accuracy']:6.1f} {m['win_accuracy']:6.1f} "
                    f"{m['combined_accuracy']:6.1f} {m['discriminability']:7.1f} "
                    f"{m['brier_occ']:8.4f} {m['brier_win']:8.4f}")

    # Save all metrics
    all_metrics = {
        'strategy_a': metrics_a,
        'strategy_b': metrics_b,
        'strategy_c': metrics_c,
    }
    with open(os.path.join(OUTPUT_DIR, 'monitoring_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
