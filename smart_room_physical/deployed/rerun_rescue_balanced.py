#!/usr/bin/env python3
"""
Re-run rescue with balanced training data.

Strategy:
  1. Merge baseline (2065 rows) + first half of regime data (~365 rows)
  2. Oversample minority regimes to match Base count
  3. Re-run hypothesis gen + rescue on balanced data
  4. Keep the 10 validated edges from overnight discovery (no re-testing)
  5. Save results separately (original results untouched)

Usage:
    python rerun_rescue_balanced.py
"""
import os, sys, json, logging
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from run_policygrid_discovery import (
    run_hypothesis_generation, rescue_edges, GRAPH_VARS,
    PROPOSED_GT, RESCUE_TAU
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler('logs/rerun_rescue_balanced.log'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results/discovery/seed_42')
BASELINE_CSV = os.environ.get('BASELINE_CSV', 'data/baseline_data.csv')
REGIME_CSV = os.environ.get('REGIME_CSV', 'data/sensor_data_regime_monitoring.csv')
OUTPUT_DIR = 'results/discovery/seed_42_balanced'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ── Step 1: Load baseline + regime data ──
logger.info("=" * 60)
logger.info("  Balanced Rescue: merge + oversample + re-run")
logger.info("=" * 60)

baseline = pd.read_csv(BASELINE_CSV)
logger.info(f"Baseline: {len(baseline)} rows")

# Check if regime data exists and preprocess it
if os.path.exists(REGIME_CSV):
    logger.info(f"Regime data found: {REGIME_CSV}")
    regime_raw = pd.read_csv(REGIME_CSV)
    logger.info(f"Regime raw: {len(regime_raw)} rows")

    # Use first half for training augmentation
    half = len(regime_raw) // 2
    regime_train_raw = regime_raw.iloc[:half]
    logger.info(f"Using first half for training: {len(regime_train_raw)} rows")

    # Preprocess regime data same way as baseline
    # Downsample to 1min, normalize, map columns
    regime_train_raw['timestamp'] = pd.to_datetime(regime_train_raw['timestamp'])
    regime_train_raw = regime_train_raw.set_index('timestamp').resample('1min').mean().reset_index()
    regime_train_raw = regime_train_raw.dropna()

    # Map columns to PolicyGRID names using same scaling as baseline
    scaling_csv = BASELINE_CSV.replace('baseline_data.csv', 'baseline_data_scaling.csv')
    if os.path.exists(scaling_csv):
        sc = pd.read_csv(scaling_csv)
        smap = {row['column']: (row['data_min'], row['data_max']) for _, row in sc.iterrows()}
    else:
        smap = {}

    # Build normalized rows matching baseline columns
    regime_rows = []
    for _, row in regime_train_raw.iterrows():
        r = {}
        r['timestamp'] = str(row.get('timestamp', ''))

        # Actuators
        r['Heater'] = float(row.get('heater_on', 0))
        r['Humidifier'] = float(row.get('humidifier_on', 0))
        r['Fan'] = float(row.get('fan_on', 0))

        # Temperature: avg of govee sensors
        temps = [v for k, v in row.items() if 'govee_temp' in str(k) and pd.notna(v)]
        raw_temp = np.mean(temps) if temps else None

        # Humidity: avg of govee sensors
        humids = [v for k, v in row.items() if 'govee_humidity' in str(k) and pd.notna(v)]
        raw_humid = np.mean(humids) if humids else None

        # AirQuality: BME IAQ
        raw_aq = row.get('bme_iaq', None)

        # Energy: Kasa power
        raw_energy = row.get('kasa_current_power_w', 0)

        # Normalize using baseline scaling
        for var, raw in [('Temperature', raw_temp), ('Humidity', raw_humid),
                         ('AirQuality', raw_aq), ('EnergyConsumption', raw_energy)]:
            if raw is not None and var.lower() in smap:
                mn, mx = smap[var.lower()]
                r[var] = (raw - mn) / (mx - mn + 1e-12)
            elif raw is not None:
                r[var] = raw
            else:
                r[var] = np.nan

        # Satisfaction from temp + humidity
        if raw_temp is not None and raw_humid is not None:
            metabolic_rate = 1.2
            pmv = 0.303 * np.exp(-0.036 * metabolic_rate * 58.15) + 0.028
            t_cl = 35.7 - 0.028 * metabolic_rate * 58.15
            h_c = max(2.38 * abs(t_cl - raw_temp) ** 0.25, 12.1 * np.sqrt(0.1))
            thermal_load = (
                metabolic_rate * 58.15
                - 3.05e-3 * (5733 - 6.99 * metabolic_rate * 58.15
                              - max(raw_humid * 0.01 * np.exp(16.6536 - 4030.183 / (raw_temp + 235)), 0))
                - 0.42 * (metabolic_rate * 58.15 - 58.15)
                - 1.7e-5 * metabolic_rate * 58.15 * (5867
                              - max(raw_humid * 0.01 * np.exp(16.6536 - 4030.183 / (raw_temp + 235)), 0))
                - 0.0014 * metabolic_rate * (34 - raw_temp)
                - 3.96e-8 * ((t_cl + 273) ** 4 - (raw_temp + 273) ** 4)
                - h_c * (t_cl - raw_temp)
            )
            pmv_val = max(-3, min(3, pmv * thermal_load))
            sat_raw = max(0, 100 - (abs(pmv_val) / 3.0) * 100)
            if 'satisfaction' in smap:
                mn, mx = smap['satisfaction']
                r['Satisfaction'] = (sat_raw - mn) / (mx - mn + 1e-12)
            else:
                r['Satisfaction'] = sat_raw
        else:
            r['Satisfaction'] = np.nan

        # Regime vars
        r['Occupancy'] = float(row.get('occupancy', 0))
        r['WindowState'] = float(row.get('window_open', 0))

        regime_rows.append(r)

    regime_df = pd.DataFrame(regime_rows)
    # Keep only columns that exist in baseline
    common_cols = [c for c in baseline.columns if c in regime_df.columns]
    regime_df = regime_df[common_cols].dropna()
    logger.info(f"Regime preprocessed: {len(regime_df)} rows")

    # Merge
    merged = pd.concat([baseline, regime_df], ignore_index=True)
    logger.info(f"Merged: {len(merged)} rows")
else:
    logger.warning(f"No regime data at {REGIME_CSV} — using baseline only")
    merged = baseline.copy()

# ── Step 2: Oversample minority regimes ──
logger.info("\nRegime distribution before oversampling:")
for (o, w), label in {(0,0): 'Base', (1,0): 'Occ+Closed',
                       (0,1): 'Win+Open', (1,1): 'Full'}.items():
    n = ((merged['Occupancy'] == o) & (merged['WindowState'] == w)).sum()
    logger.info(f"  {label:15s}: {n:4d} rows")

# Oversample minority regimes to ~500 rows each (close to Base proportion)
target_per_regime = 500
balanced = merged.copy()
for (o, w), label in {(1,0): 'Occ+Closed', (0,1): 'Win+Open', (1,1): 'Full'}.items():
    mask = (merged['Occupancy'] == o) & (merged['WindowState'] == w)
    subset = merged[mask]
    n = len(subset)
    if 0 < n < target_per_regime:
        repeats = (target_per_regime // n)
        extra = target_per_regime - n * repeats
        oversampled = pd.concat([subset] * repeats + [subset.iloc[:extra]], ignore_index=True)
        balanced = pd.concat([balanced, oversampled.iloc[n:]], ignore_index=True)  # add only the new copies
        logger.info(f"  Oversampled {label}: {n} → {len(oversampled)} rows")

logger.info(f"\nBalanced total: {len(balanced)} rows")
logger.info("Regime distribution after oversampling:")
for (o, w), label in {(0,0): 'Base', (1,0): 'Occ+Closed',
                       (0,1): 'Win+Open', (1,1): 'Full'}.items():
    n = ((balanced['Occupancy'] == o) & (balanced['WindowState'] == w)).sum()
    logger.info(f"  {label:15s}: {n:4d} rows")

# ── Step 3: Load saved validated edges ──
with open(os.path.join(RESULTS_DIR, 'validated_edges.json')) as f:
    validated = {tuple(e) for e in json.load(f)}
logger.info(f"\nSaved validated edges: {len(validated)}")

# ── Step 4: Hypothesis gen on balanced data ──
candidates, scores, methods = run_hypothesis_generation(balanced)

# ── Step 5: Rescue on balanced data ──
evidence = {}
rescued = rescue_edges(candidates, balanced, validated, methods, evidence)

all_edges = validated | rescued
logger.info(f"\nFinal: {len(validated)} validated + {len(rescued)} rescued = {len(all_edges)} total")

# F1
gt_norm = {(s.lower(), t.lower()) for s, t in PROPOSED_GT}
val_norm = {(s.lower(), t.lower()) for s, t in all_edges}
tp = len(val_norm & gt_norm)
fp = len(val_norm - gt_norm)
fn = len(gt_norm - val_norm)
p = tp / max(tp + fp, 1)
r = tp / max(tp + fn, 1)
f1 = 2 * p * r / max(p + r, 1e-8)
logger.info(f"F1={f1:.3f} (P={p:.2f}, R={r:.2f}), TP={tp}, FP={fp}, FN={fn}")

logger.info(f"\nTP edges:")
for e in sorted(val_norm & gt_norm):
    logger.info(f"  {e[0]} → {e[1]}")
logger.info(f"\nFP edges:")
for e in sorted(val_norm - gt_norm):
    logger.info(f"  {e[0]} → {e[1]}")
logger.info(f"\nFN (missing):")
for e in sorted(gt_norm - val_norm):
    logger.info(f"  {e[0]} → {e[1]}")

# ── Step 6: Compare with unbalanced results ──
logger.info(f"\n{'='*60}")
logger.info(f"  COMPARISON: unbalanced vs balanced rescue")
logger.info(f"{'='*60}")

orig_path = os.path.join(RESULTS_DIR, 'validated_edges.json')
with open(orig_path) as f:
    orig_edges = {tuple(e) for e in json.load(f)}
orig_norm = {(s.lower(), t.lower()) for s, t in orig_edges}
new_norm = val_norm

new_edges = new_norm - orig_norm
lost_edges = orig_norm - new_norm
logger.info(f"New edges gained: {len(new_edges)}")
for e in sorted(new_edges):
    logger.info(f"  + {e[0]} → {e[1]}")
logger.info(f"Edges lost: {len(lost_edges)}")
for e in sorted(lost_edges):
    logger.info(f"  - {e[0]} → {e[1]}")

# Save to separate dir (original untouched)
with open(os.path.join(OUTPUT_DIR, 'validated_edges.json'), 'w') as f:
    json.dump([list(e) for e in sorted(all_edges)], f, indent=2)
with open(os.path.join(OUTPUT_DIR, 'edge_evidence.json'), 'w') as f:
    json.dump(evidence, f, indent=2)

metrics = {
    'n_validated': len(validated), 'n_rescued': len(rescued),
    'n_total': len(all_edges),
    'gt_f1': round(f1, 3), 'gt_precision': round(p, 3),
    'gt_recall': round(r, 3), 'gt_shd': fp + fn,
    'balanced': True,
    'training_rows': len(balanced),
}
with open(os.path.join(OUTPUT_DIR, 'discovery_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

logger.info(f"\nSaved balanced results to {OUTPUT_DIR}/")
logger.info(f"Original results preserved in {RESULTS_DIR}/")
