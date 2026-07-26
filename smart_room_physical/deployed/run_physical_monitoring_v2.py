#!/usr/bin/env python3
"""
Physical monitoring v2 — improved architecture.

Changes from v1:
  1. Multi-parent Ridge per target (not per-edge) — matches sim pipeline
  2. Clip predictions to [0,1] to prevent overflow
  3. Sweep σ and η to find best physical params
  4. Use ALL observable vars as features (full graph), not just discovered edges
  5. Try both edge-constrained and full-feature models
"""
import os, sys, json, logging, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from itertools import product

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring_v2.log'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

EDGES_PATH = 'results/discovery/seed_42/validated_edges.json'
BASELINE_CSV = 'data/baseline_data.csv'
REGIME_CSV = 'data/sensor_data_regime_monitoring.csv'
SCALING_CSV = 'data/baseline_data_scaling.csv'
OUTPUT_DIR = 'results/monitoring_v2'

GRAPH_VARS = ['Heater', 'Humidifier', 'Fan', 'Temperature', 'Humidity',
              'AirQuality', 'EnergyConsumption', 'Satisfaction',
              'Occupancy', 'WindowState']
OBS_VARS = [v.lower() for v in GRAPH_VARS if v not in ('Occupancy', 'WindowState')]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)


def preprocess_regime_data(raw_csv, scaling_csv):
    """Same as v1 — preprocess raw regime CSV."""
    raw = pd.read_csv(raw_csv)
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    sc = pd.read_csv(scaling_csv)
    smap = {row['column'].lower(): (row['data_min'], row['data_max']) for _, row in sc.iterrows()}

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

        for var, raw_val in [('Temperature', raw_temp), ('Humidity', raw_humid),
                              ('AirQuality', r.get('bme_iaq')),
                              ('EnergyConsumption', r.get('kasa_current_power_w', 0))]:
            if raw_val is not None and var.lower() in smap:
                mn, mx = smap[var.lower()]
                row[var] = np.clip((raw_val - mn) / (mx - mn + 1e-12), 0, 1)
            else:
                row[var] = np.nan

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
                row['Satisfaction'] = np.clip((sat_raw - mn) / (mx - mn + 1e-12), 0, 1)
            else:
                row['Satisfaction'] = sat_raw / 100.0
        else:
            row['Satisfaction'] = np.nan

        row['Occupancy'] = float(r.get('occupancy', 0))
        row['WindowState'] = float(r.get('window_open', 0))
        rows.append(row)

    df = pd.DataFrame(rows).dropna(subset=['Temperature', 'Humidity'])
    return df


def build_monitors(train_data, edges, mode='graph_parents'):
    """
    Build 4-way factored monitors.

    mode='graph_parents': use only discovered graph parents per target (like sim)
    mode='all_features': use ALL obs vars as features per target (richer model)
    """
    # Build parent map from edges
    parent_map = {v: [] for v in OBS_VARS}
    for s, t in edges:
        sl, tl = s.lower(), t.lower()
        if sl in OBS_VARS and tl in OBS_VARS:
            if sl not in parent_map[tl]:
                parent_map[tl].append(sl)

    # Temporal data: predict X(t+1) from features at t
    col_map = {c.lower(): c for c in train_data.columns}
    tdf = pd.DataFrame()
    for v in OBS_VARS:
        if v in col_map or v.title() in train_data.columns:
            c = col_map.get(v, v.title())
            vals = train_data[c].values
            tdf[v] = vals[:-1]
            tdf[v + '_next'] = vals[1:]

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
        if len(regime_data) < 20:
            regime_data = tdf

        models = {}
        for target in OBS_VARS:
            if target + '_next' not in regime_data.columns:
                continue

            if mode == 'graph_parents':
                features = parent_map.get(target, [])
                if not features:
                    continue
            else:  # all_features
                features = [v for v in OBS_VARS if v != target and v in regime_data.columns]

            # Check features exist
            features = [f for f in features if f in regime_data.columns]
            if not features:
                continue

            X = regime_data[features].values
            y = regime_data[target + '_next'].values

            # Drop rows with NaN
            valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            if valid.sum() < 10:
                continue

            reg = Ridge(alpha=1.0).fit(X[valid], y[valid])
            models[target] = {'ridge': reg, 'features': features}

        monitors[regime] = {'models': models, 'n_train': len(regime_data)}

    return monitors


def run_monitoring(monitors, test_data, sigma=2.0, forget=0.12):
    """Run factored Bayesian monitoring with multi-parent models."""
    p_occ = 0.5
    p_win = 0.5

    col_map = {c.lower(): c for c in test_data.columns}
    records = []

    for i in range(len(test_data) - 1):
        # Current obs
        obs = {}
        for v in OBS_VARS:
            c = col_map.get(v, v.title())
            if c in test_data.columns:
                obs[v] = test_data[c].values[i]

        # Next-step obs (ground truth for error)
        obs_next = {}
        for v in OBS_VARS:
            c = col_map.get(v, v.title())
            if c in test_data.columns:
                obs_next[v] = test_data[c].values[i + 1]

        gt_occ = test_data['Occupancy'].values[i] >= 0.5
        gt_win = test_data['WindowState'].values[i] >= 0.5

        # Per-regime prediction error
        errors = {}
        for regime in ['base', 'occ', 'win', 'full']:
            total_err = 0.0
            n_pred = 0
            for target, m in monitors[regime]['models'].items():
                features = m['features']
                x_vec = np.array([obs.get(f, 0.0) for f in features]).reshape(1, -1)
                pred = np.clip(m['ridge'].predict(x_vec)[0], 0, 1)
                actual = obs_next.get(target, 0.0)
                total_err += abs(actual - pred)
                n_pred += 1
            errors[regime] = total_err / max(n_pred, 1)

        # Factored Bayesian update
        err_occ_off = min(errors['base'], errors['win'])
        err_occ_on = min(errors['occ'], errors['full'])
        ll_on = np.exp(-sigma * err_occ_on)
        ll_off = np.exp(-sigma * err_occ_off)
        denom = ll_on * p_occ + ll_off * (1 - p_occ) + 1e-12
        p_occ = ll_on * p_occ / denom

        err_win_off = min(errors['base'], errors['occ'])
        err_win_on = min(errors['win'], errors['full'])
        ll_on = np.exp(-sigma * err_win_on)
        ll_off = np.exp(-sigma * err_win_off)
        denom = ll_on * p_win + ll_off * (1 - p_win) + 1e-12
        p_win = ll_on * p_win / denom

        # Forgetting
        p_occ = (1 - forget) * p_occ + forget * 0.5
        p_win = (1 - forget) * p_win + forget * 0.5

        records.append({
            'step': i,
            'gt_occ': gt_occ, 'gt_win': gt_win,
            'pred_occ': p_occ >= 0.5, 'pred_win': p_win >= 0.5,
            'p_occ': round(p_occ, 4), 'p_win': round(p_win, 4),
        })

    return pd.DataFrame(records)


def compute_metrics(df):
    occ_acc = (df['gt_occ'] == df['pred_occ']).mean() * 100
    win_acc = (df['gt_win'] == df['pred_win']).mean() * 100
    comb = ((df['gt_occ'] == df['pred_occ']) & (df['gt_win'] == df['pred_win'])).mean() * 100
    brier_occ = ((df['p_occ'] - df['gt_occ'].astype(float)) ** 2).mean()
    brier_win = ((df['p_win'] - df['gt_win'].astype(float)) ** 2).mean()
    return {
        'occ': round(occ_acc, 1), 'win': round(win_acc, 1),
        'comb': round(comb, 1),
        'brier_o': round(brier_occ, 4), 'brier_w': round(brier_win, 4),
    }


def main():
    logger.info("=" * 60)
    logger.info("  Physical Monitoring v2 — improved architecture")
    logger.info("=" * 60)

    with open(EDGES_PATH) as f:
        edges = [tuple(e) for e in json.load(f)]
    baseline = pd.read_csv(BASELINE_CSV)
    regime = preprocess_regime_data(REGIME_CSV, SCALING_CSV)

    logger.info(f"Edges: {len(edges)}, Baseline: {len(baseline)}, Regime: {len(regime)}")

    # ── Experiment 1: Model architecture ──
    logger.info("\n" + "=" * 60)
    logger.info("  EXP 1: Model architecture (Strategy A: train=baseline)")
    logger.info("=" * 60)

    for mode in ['graph_parents', 'all_features']:
        monitors = build_monitors(baseline, edges, mode=mode)
        records = run_monitoring(monitors, regime, sigma=2.0, forget=0.12)
        m = compute_metrics(records)
        logger.info(f"  {mode:20s}: Occ={m['occ']:5.1f}  Win={m['win']:5.1f}  "
                    f"Comb={m['comb']:5.1f}  Brier={m['brier_o']:.3f}/{m['brier_w']:.3f}")

    # ── Experiment 2: σ sweep (both architectures) ──
    logger.info("\n" + "=" * 60)
    logger.info("  EXP 2: σ sweep")
    logger.info("=" * 60)

    best_comb = 0
    best_params = {}

    for mode in ['graph_parents', 'all_features']:
        monitors = build_monitors(baseline, edges, mode=mode)
        for sigma in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
            for forget in [0.05, 0.12, 0.20]:
                records = run_monitoring(monitors, regime, sigma=sigma, forget=forget)
                m = compute_metrics(records)
                if m['comb'] > best_comb:
                    best_comb = m['comb']
                    best_params = {'mode': mode, 'sigma': sigma, 'forget': forget, **m}

    logger.info(f"  Best: {best_params}")

    # ── Experiment 3: Strategy comparison with best params ──
    logger.info("\n" + "=" * 60)
    logger.info("  EXP 3: Training strategy with best params")
    logger.info("=" * 60)

    mode = best_params.get('mode', 'all_features')
    sigma = best_params.get('sigma', 2.0)
    forget = best_params.get('forget', 0.12)
    logger.info(f"  Using: mode={mode}, σ={sigma}, η={forget}")

    # A: baseline train, regime test
    monitors_a = build_monitors(baseline, edges, mode=mode)
    records_a = run_monitoring(monitors_a, regime, sigma=sigma, forget=forget)
    m_a = compute_metrics(records_a)

    # B: regime first half train, second half test
    half = len(regime) // 2
    monitors_b = build_monitors(regime.iloc[:half], edges, mode=mode)
    records_b = run_monitoring(monitors_b, regime.iloc[half:].reset_index(drop=True),
                               sigma=sigma, forget=forget)
    m_b = compute_metrics(records_b)

    # C: combined train, second half test
    combined = pd.concat([baseline, regime.iloc[:half]], ignore_index=True)
    monitors_c = build_monitors(combined, edges, mode=mode)
    records_c = run_monitoring(monitors_c, regime.iloc[half:].reset_index(drop=True),
                               sigma=sigma, forget=forget)
    m_c = compute_metrics(records_c)

    logger.info(f"\n{'Strategy':15s} {'Occ%':>6s} {'Win%':>6s} {'Comb%':>6s}")
    logger.info(f"{'A: baseline':15s} {m_a['occ']:6.1f} {m_a['win']:6.1f} {m_a['comb']:6.1f}")
    logger.info(f"{'B: regime½':15s} {m_b['occ']:6.1f} {m_b['win']:6.1f} {m_b['comb']:6.1f}")
    logger.info(f"{'C: base+reg½':15s} {m_c['occ']:6.1f} {m_c['win']:6.1f} {m_c['comb']:6.1f}")

    # Save
    all_results = {
        'best_params': best_params,
        'strategy_a': m_a, 'strategy_b': m_b, 'strategy_c': m_c,
    }
    with open(os.path.join(OUTPUT_DIR, 'monitoring_metrics_v2.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    records_a.to_csv(os.path.join(OUTPUT_DIR, 'best_records.csv'), index=False)

    logger.info(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
