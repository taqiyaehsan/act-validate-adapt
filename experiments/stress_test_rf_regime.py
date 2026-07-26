#!/usr/bin/env python3
"""
RF Regime Detection Stress Test
================================
Answers: Is the RF signal strong enough and fast enough to reliably detect
regime shifts (window closed → open) for Experiment A2?

Step 0: Retrain RF on the processed (normalized) data and overwrite the pkl.
         The old model was trained on raw-scale data — unusable on [0,1] inputs.

Metrics reported:
  1. Classifier accuracy / precision / recall / F1 / AUC by class
  2. prob_open distribution under R1 vs R2 (separation quality)
  3. Detection delay: timesteps from C→O shift until prob_open crosses threshold
  4. False alarm rate: fraction of R1 timesteps where prob_open > threshold
  5. Per-run analysis: does detection succeed within each contiguous open window?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)

from regime_aware_validation import WindowStatePredictor

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/challenge1_data_10k_processed.csv"
MODEL_PATH = "window_predictor_model.pkl"
THRESHOLDS = [0.3, 0.5, 0.7]
MAX_DELAY  = 30   # cap on delay window (timesteps)

# CSV column name → RF feature name
COL_MAP = {
    'Temperature':        'temperature',
    'Humidity':           'humidity',
    'AirQuality':         'airQuality',
    'OutdoorTemperature': 'outdoorTemperature',
    'EnergyConsumption':  'energyConsumption',
    'Satisfaction':       'satisfaction',
    'WindowOpen':         'windowOpen',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def sep(char='─', n=60):
    print(char * n)

def load_and_remap():
    """Load processed CSV and rename columns to match RF feature expectations."""
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=COL_MAP)
    keep = list(COL_MAP.values())
    return df[[c for c in keep if c in df.columns]].copy()

def run_predictions(predictor, df):
    # Physical-state features only (no outcome variables)
    feature_cols = ['temperature', 'humidity', 'airQuality', 'outdoorTemperature']
    features  = predictor.extract_features(df[feature_cols])
    probs     = predictor.model.predict_proba(features)
    prob_open = probs[:, 1]
    pred      = (prob_open >= 0.5).astype(int)
    return pred, prob_open

def find_open_runs(window_series):
    """Return list of (start, end) indices for each contiguous open-window run."""
    runs, in_run, start = [], False, None
    for i, v in enumerate(window_series):
        if v == 1 and not in_run:
            in_run, start = True, i
        elif v == 0 and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(window_series) - 1))
    return runs

def detection_delay(prob_open, runs, threshold, max_delay=MAX_DELAY):
    delays = []
    for start, end in runs:
        cap = min(end - start + 1, max_delay)
        detected = next(
            (offset for offset in range(cap)
             if prob_open[start + offset] >= threshold),
            None
        )
        delays.append(detected)
    return delays

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sep('═')
    print("  RF REGIME DETECTION STRESS TEST")
    sep('═')
    print(f"\nData : {DATA_PATH}")
    print(f"Model: {MODEL_PATH}\n")

    df = load_and_remap()
    y_true = df['windowOpen'].values.astype(int)

    # ── Step 0: Retrain on processed data ─────────────────────────────────────
    sep()
    print("  0. RETRAINING RF ON PROCESSED (NORMALIZED) DATA")
    sep()
    print("\n  Old pkl was trained on raw-scale data → unusable on [0,1] inputs.")
    print("  Retraining now on challenge1_data_10k_processed.csv ...\n")

    predictor = WindowStatePredictor()
    accuracy  = predictor.train(df)          # prints sklearn report internally
    predictor.save(MODEL_PATH)
    print(f"\n  Retrained accuracy (held-out 20%): {accuracy:.4f}")

    # ── 1. Global classifier metrics ──────────────────────────────────────────
    pred, prob_open = run_predictions(predictor, df)

    sep()
    print("  1. CLASSIFIER PERFORMANCE (full dataset, in-sample)")
    sep()
    acc = accuracy_score(y_true, pred)
    auc = roc_auc_score(y_true, prob_open)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average=None, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, pred)

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'N':>8}")
    print(f"  {'─'*52}")
    for i, cls in enumerate(['Closed (R1)', 'Open   (R2)']):
        print(f"  {cls:<12}   {p[i]:>8.4f}   {r[i]:>8.4f}   {f1[i]:>8.4f}   {sum(y_true==i):>6}")

    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"               Pred-Closed  Pred-Open")
    print(f"  Actual-Closed   {cm[0,0]:>8}   {cm[0,1]:>8}")
    print(f"  Actual-Open     {cm[1,0]:>8}   {cm[1,1]:>8}")

    # ── 2. prob_open distribution by regime ───────────────────────────────────
    sep()
    print("  2. prob_open DISTRIBUTION BY REGIME")
    sep()
    p_closed = prob_open[y_true == 0]
    p_open   = prob_open[y_true == 1]

    def stats(arr, label):
        print(f"\n  {label}")
        pcts = np.percentile(arr, [5, 25, 50, 75, 95])
        print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"p5={pcts[0]:.4f}  p25={pcts[1]:.4f}  p50={pcts[2]:.4f}  "
              f"p75={pcts[3]:.4f}  p95={pcts[4]:.4f}")

    stats(p_closed, "R1 (window CLOSED) — want low prob_open")
    stats(p_open,   "R2 (window OPEN)   — want high prob_open")

    print(f"\n  {'Threshold':>10}  {'R1 above (FA)':>14}  {'R2 above (TPR)':>15}  {'Youden J':>10}")
    print(f"  {'─'*55}")
    for thr in THRESHOLDS:
        fa  = (p_closed >= thr).mean()
        tpr = (p_open   >= thr).mean()
        print(f"  {thr:>10.2f}  {fa:>14.4f}  {tpr:>15.4f}  {tpr-fa:>10.4f}")

    # ── 3. Detection delay ─────────────────────────────────────────────────────
    sep()
    print("  3. DETECTION DELAY  (C→O transitions)")
    sep()
    runs = find_open_runs(y_true)
    run_lengths = [e - s + 1 for s, e in runs]
    print(f"\n  {len(runs)} open-window runs  |  "
          f"lengths: min={min(run_lengths)}  median={np.median(run_lengths):.0f}  max={max(run_lengths)}")

    print(f"\n  {'Threshold':>10}  {'Detected':>9}  {'Missed':>7}  "
          f"{'Det%':>6}  {'Mean delay':>11}  {'p50':>6}  {'p90':>6}")
    print(f"  {'─'*68}")
    for thr in THRESHOLDS:
        delays   = detection_delay(prob_open, runs, thr)
        detected = [d for d in delays if d is not None]
        det_pct  = 100 * len(detected) / len(delays)
        if detected:
            m, p50, p90 = np.mean(detected), np.percentile(detected, 50), np.percentile(detected, 90)
        else:
            m = p50 = p90 = float('nan')
        print(f"  {thr:>10.2f}  {len(detected):>9}  {len(delays)-len(detected):>7}  "
              f"{det_pct:>5.1f}%  {m:>11.2f}  {p50:>6.2f}  {p90:>6.2f}")

    print(f"\n  Note: delay=0 means detected on the very first open timestep.")
    print(f"        Missed = not detected within first {MAX_DELAY} timesteps.")

    # ── 4. False alarm analysis ────────────────────────────────────────────────
    sep()
    print("  4. FALSE ALARM ANALYSIS  (R1 sustained sequences)")
    sep()
    closed_runs = find_open_runs(1 - y_true)
    closed_lengths = [e - s + 1 for s, e in closed_runs]
    print(f"\n  {len(closed_runs)} closed-window runs  |  "
          f"lengths: min={min(closed_lengths)}  median={np.median(closed_lengths):.0f}  max={max(closed_lengths)}")

    print(f"\n  {'Threshold':>10}  {'FA rate':>9}  {'Runs w/ ≥1 FA':>15}  {'Max FA streak':>14}")
    print(f"  {'─'*54}")
    for thr in THRESHOLDS:
        fa_rates, runs_with_fa, max_streak = [], 0, 0
        for s, e in closed_runs:
            run_p   = prob_open[s:e+1]
            fa_rate = (run_p >= thr).mean()
            fa_rates.append(fa_rate)
            if fa_rate > 0:
                runs_with_fa += 1
            streak = cur = 0
            for v in run_p:
                cur = cur + 1 if v >= thr else 0
                streak = max(streak, cur)
            max_streak = max(max_streak, streak)
        rp = 100 * runs_with_fa / len(closed_runs)
        print(f"  {thr:>10.2f}  {np.mean(fa_rates):>9.4f}  "
              f"{runs_with_fa:>7} ({rp:>4.1f}%)  {max_streak:>14}")

    # ── 5. Per-run detail ─────────────────────────────────────────────────────
    sep()
    print("  5. PER-RUN DETAIL  (10 longest open runs, threshold=0.5)")
    sep()
    top10 = sorted(runs, key=lambda x: -(x[1]-x[0]))[:10]
    print(f"\n  {'#':>3}  {'Start':>7}  {'End':>7}  {'Len':>5}  "
          f"{'Delay':>7}  {'MaxProb':>8}  {'MeanProb':>9}")
    print(f"  {'─'*52}")
    for i, (s, e) in enumerate(top10):
        rp = prob_open[s:e+1]
        delay = next((off for off, v in enumerate(rp[:MAX_DELAY]) if v >= 0.5), None)
        ds = str(delay) if delay is not None else "MISS"
        print(f"  {i+1:>3}  {s:>7}  {e:>7}  {e-s+1:>5}  "
              f"{ds:>7}  {rp.max():>8.4f}  {rp.mean():>9.4f}")

    # ── Summary verdict ───────────────────────────────────────────────────────
    sep('═')
    print("  VERDICT")
    sep('═')
    delays_05 = detection_delay(prob_open, runs, 0.5)
    detected_05 = [d for d in delays_05 if d is not None]
    det_rate   = 100 * len(detected_05) / len(delays_05)
    mean_delay = np.mean(detected_05) if detected_05 else float('nan')
    fa_rate_05 = (p_closed >= 0.5).mean()

    print(f"\n  At threshold=0.5:")
    print(f"    AUC-ROC       : {auc:.4f}   {'✓ strong' if auc > 0.85 else '⚠ weak'}")
    print(f"    Detection rate: {det_rate:.1f}%   {'✓' if det_rate > 80 else '⚠ weak'}")
    print(f"    Mean delay    : {mean_delay:.1f} steps  {'✓ fast' if not np.isnan(mean_delay) and mean_delay < 5 else '⚠ slow'}")
    print(f"    FA rate (R1)  : {fa_rate_05:.4f}   {'✓ low' if fa_rate_05 < 0.05 else '⚠ high'}")
    print(f"\n  A2 readiness: ", end="")
    if auc > 0.85 and det_rate > 80 and fa_rate_05 < 0.10:
        print("READY — signal is strong enough to drive regime switch in A2.")
    elif auc > 0.70:
        print("MARGINAL — signal works but consider tuning threshold or retraining RF.")
    else:
        print("NOT READY — RF signal too weak; investigate feature separability.")
    sep('═')
    print()


if __name__ == "__main__":
    main()
