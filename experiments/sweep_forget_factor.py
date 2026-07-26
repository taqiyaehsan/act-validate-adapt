#!/usr/bin/env python3
"""
Sensitivity analysis for the Bayesian monitoring forget factor (λ).

Replays recorded monitoring errors (err_H0, err_H1) from completed runs
with different λ values.  Computes detection latency, classification
accuracy, and F1 for each λ.

Output:
  neurips_results/forget_factor_sweep.csv   — per-λ metrics
  neurips_results/forget_factor_sweep.png   — summary plot
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────
LAMBDA_VALUES = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
RESULTS_DIRS = [
    'results/3runs_cwm_20260306_fixedPMV-works',
    'results/3runs_cwm_20260306_230959',
]
OUTPUT_DIR = 'neurips_results'
DECISION_THRESHOLD = 0.5   # P(H1) > threshold → predict "open"


def load_monitoring_runs(results_dirs):
    """Load all monitoring_probabilities.csv files with valid data."""
    runs = []
    for rdir in results_dirs:
        pattern = os.path.join(rdir, 'run_*', 'monitoring_probabilities.csv')
        for fpath in sorted(glob.glob(pattern)):
            df = pd.read_csv(fpath)
            if 'err_H0' not in df.columns or 'err_H1' not in df.columns:
                continue
            if len(df) < 10:
                continue
            # Tag with source
            df['source_run'] = fpath
            runs.append(df)
    return runs


def replay_bayesian(err_H0, err_H1, sigma_arr, lam):
    """Replay Bayesian updating with a given forget factor λ.

    Returns array of P(H1) at each timestep.
    """
    n = len(err_H0)
    P_H0 = np.full(n, 0.5)
    P_H1 = np.full(n, 0.5)

    p0, p1 = 0.5, 0.5
    for t in range(n):
        s = sigma_arr[t]
        lh0 = np.exp(-s * err_H0[t])
        lh1 = np.exp(-s * err_H1[t])
        unnorm0 = lh0 * p0
        unnorm1 = lh1 * p1
        total = unnorm0 + unnorm1
        if total > 0:
            p0 = (1 - lam) * (unnorm0 / total) + lam * 0.5
            p1 = (1 - lam) * (unnorm1 / total) + lam * 0.5
        P_H0[t] = p0
        P_H1[t] = p1
    return P_H1


def compute_metrics(P_H1, actual_window, threshold=DECISION_THRESHOLD):
    """Compute detection accuracy, F1, and mean latency."""
    n = len(P_H1)
    predicted = (P_H1 > threshold).astype(int)
    actual = np.array(actual_window).astype(int)

    # Accuracy
    accuracy = np.mean(predicted == actual)

    # Precision / Recall / F1
    tp = np.sum((predicted == 1) & (actual == 1))
    fp = np.sum((predicted == 1) & (actual == 0))
    fn = np.sum((predicted == 0) & (actual == 1))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # Detection latency: timesteps from regime change to correct detection
    latencies = []
    prev_window = actual[0]
    for t in range(1, n):
        if actual[t] != prev_window:
            # Regime switch at timestep t
            target_pred = actual[t]
            # Find first timestep where prediction matches new regime
            for dt in range(t, n):
                if predicted[dt] == target_pred:
                    latencies.append(dt - t)
                    break
            else:
                latencies.append(n - t)  # never detected
        prev_window = actual[t]

    mean_latency = np.mean(latencies) if latencies else 0.0
    n_transitions = len(latencies)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_latency_steps': mean_latency,
        'n_transitions': n_transitions,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    runs = load_monitoring_runs(RESULTS_DIRS)
    if not runs:
        print("ERROR: No monitoring data found. Run the pipeline first.")
        return

    print(f"Loaded {len(runs)} monitoring run(s)")
    for r in runs:
        print(f"  {r['source_run'].iloc[0]}: {len(r)} timesteps")

    # ── Sweep ────────────────────────────────────────────────────────
    all_rows = []
    trajectories = {}  # λ → list of P_H1 arrays (for plotting)

    for lam in LAMBDA_VALUES:
        run_metrics = []
        lam_trajectories = []

        for run_df in runs:
            err_H0 = run_df['err_H0'].values
            err_H1 = run_df['err_H1'].values
            sigma = run_df['sigma'].values
            actual = run_df['actual_window'].values

            P_H1 = replay_bayesian(err_H0, err_H1, sigma, lam)
            metrics = compute_metrics(P_H1, actual)
            metrics['lambda'] = lam
            metrics['source'] = run_df['source_run'].iloc[0]
            run_metrics.append(metrics)
            lam_trajectories.append((P_H1, actual, run_df['time'].values))

        all_rows.extend(run_metrics)
        trajectories[lam] = lam_trajectories

    results_df = pd.DataFrame(all_rows)

    # ── Aggregate across runs ────────────────────────────────────────
    summary = results_df.groupby('lambda').agg(
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        f1_mean=('f1', 'mean'),
        f1_std=('f1', 'std'),
        precision_mean=('precision', 'mean'),
        recall_mean=('recall', 'mean'),
        latency_mean=('mean_latency_steps', 'mean'),
        latency_std=('mean_latency_steps', 'std'),
        n_runs=('accuracy', 'count'),
    ).reset_index()

    # Fill NaN std with 0 (single run)
    summary = summary.fillna(0)

    # ── Save CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, 'forget_factor_sweep.csv')
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved sweep results to {csv_path}")
    print(summary.to_string(index=False))

    # ── Select optimal λ ─────────────────────────────────────────────
    # Criterion: highest F1; if tied, lowest latency
    best_idx = summary['f1_mean'].idxmax()
    best_row = summary.iloc[best_idx]
    best_lambda = best_row['lambda']
    print(f"\nOptimal λ = {best_lambda:.2f}  "
          f"(F1={best_row['f1_mean']:.3f}, "
          f"latency={best_row['latency_mean']:.1f} steps)")

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Forget Factor (λ) Sensitivity Analysis', fontsize=14, y=0.98)

    lambdas = summary['lambda'].values

    # (a) F1 Score
    ax = axes[0, 0]
    ax.errorbar(lambdas, summary['f1_mean'], yerr=summary['f1_std'],
                marker='o', capsize=4, color='#2196F3', linewidth=2)
    ax.axvline(best_lambda, color='red', linestyle='--', alpha=0.5,
               label=f'optimal λ={best_lambda:.2f}')
    ax.set_xlabel('λ (forget factor)')
    ax.set_ylabel('F1 Score')
    ax.set_title('(a) F1 Score vs λ')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Detection Latency
    ax = axes[0, 1]
    ax.errorbar(lambdas, summary['latency_mean'], yerr=summary['latency_std'],
                marker='s', capsize=4, color='#FF9800', linewidth=2)
    ax.axvline(best_lambda, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('λ (forget factor)')
    ax.set_ylabel('Mean Detection Latency (steps)')
    ax.set_title('(b) Detection Latency vs λ')
    ax.grid(True, alpha=0.3)

    # (c) Accuracy
    ax = axes[1, 0]
    ax.errorbar(lambdas, summary['accuracy_mean'], yerr=summary['accuracy_std'],
                marker='^', capsize=4, color='#4CAF50', linewidth=2)
    ax.axvline(best_lambda, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('λ (forget factor)')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('(c) Accuracy vs λ')
    ax.grid(True, alpha=0.3)

    # (d) Posterior trajectories for selected λ values
    ax = axes[1, 1]
    selected_lambdas = [0.01, 0.05, best_lambda, 0.20]
    colors = ['#9C27B0', '#2196F3', '#F44336', '#FF9800']
    for lam_val, color in zip(selected_lambdas, colors):
        if lam_val in trajectories and trajectories[lam_val]:
            P_H1, actual, time = trajectories[lam_val][0]
            label = f'λ={lam_val:.2f}'
            if lam_val == best_lambda:
                label += ' (optimal)'
            ax.plot(time, P_H1, label=label, color=color,
                    linewidth=2 if lam_val == best_lambda else 1, alpha=0.8)

    # Ground truth shading
    if trajectories and list(trajectories.values())[0]:
        _, actual, time = list(trajectories.values())[0][0]
        for i in range(len(actual)):
            if actual[i] > 0.5:
                t0 = time[max(0, i-1)] if i > 0 else time[0]
                t1 = time[min(i, len(time)-1)]
                ax.axvspan(t0, t1, alpha=0.1, color='red')

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('P(H1)')
    ax.set_title('(d) P(H1) Trajectory Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'forget_factor_sweep.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {plot_path}")

    return best_lambda


if __name__ == '__main__':
    best = main()
