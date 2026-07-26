"""
Calibration analysis for hypothesis tracking
Measures how well P(H1) probabilities match actual window states
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def compute_calibration_metrics(timestamps: List,
                                 P_H1: List[float],
                                 actual_states: List[int]) -> Dict:
    """
    Compute Brier score and log-loss for calibration quality.

    Args:
        timestamps: List of timesteps
        P_H1: Predicted probabilities that window is open
        actual_states: Ground truth (0=closed, 1=open)

    Returns:
        Dictionary with 'brier_score' and 'log_loss'
    """
    P_H1 = np.array(P_H1, dtype=float)
    actual = np.array(actual_states, dtype=float)

    # Brier score: mean squared error between predictions and actuals
    brier_score = np.mean((P_H1 - actual) ** 2)

    # Log-loss with numerical stability
    epsilon = 1e-10
    P_clipped = np.clip(P_H1, epsilon, 1 - epsilon)
    log_loss = -np.mean(actual * np.log(P_clipped) +
                        (1 - actual) * np.log(1 - P_clipped))

    return {
        'brier_score': float(brier_score),
        'log_loss': float(log_loss)
    }


def compute_suite_metrics(df: 'pd.DataFrame',
                          transition_buffer_min: float = 0.25) -> Dict:
    """
    Compute full suite of calibration metrics for all hypothesis signals,
    both on all timesteps and on steady-state timesteps only.

    Steady-state = timesteps more than `transition_buffer_min` minutes away
    from any ground-truth regime change. This isolates the system's performance
    once it has had time to converge, separate from detection latency.

    Methods evaluated (all as P(window open)):
        - CWM_binary       : binary H0/H1 Bayesian posterior  (primary)
        - RF               : Random Forest classifier          (baseline)
        - H1_pool          : H1_WindowOpen weight in CWM pool  (ablation)
        - H0_pool_inv      : 1 - H0_WindowClosed pool weight   (ablation)
        - VARLiNGAM_pool   : VARLiNGAM structural hypothesis   (appendix)
        - LLM_pool         : LLM structural hypothesis         (appendix)
        - PC_pool          : PC structural hypothesis          (appendix)

    Returns flat dict, e.g.:
        {
          'n_total': 59, 'n_steady': 36,
          'CWM_binary_brier': 0.27, 'CWM_binary_brier_ss': 0.18,
          'RF_brier': 0.13, 'RF_brier_ss': 0.09, ...
        }
    """
    import pandas as pd
    actual = df['actual_window'].values.astype(float)
    times  = df['time'].values.astype(float)

    # Identify transition times (where ground truth flips)
    transition_mask = np.abs(np.diff(actual, prepend=actual[0])) > 0.5
    transition_times = times[transition_mask]

    # Steady-state mask: >buffer_min minutes from every transition
    is_transition = np.zeros(len(df), dtype=bool)
    for t_change in transition_times:
        is_transition |= (np.abs(times - t_change) < transition_buffer_min)
    ss_mask = ~is_transition

    # Methods: (column_or_expression, label)
    # For pool weights that represent P(window closed) we flip with 1-x
    methods = [
        ('P_H1_binary',      'CWM_binary'),
        ('P_H1_RF',          'RF'),
        ('P_H1_WindowOpen',  'H1_pool'),
        ('H0_pool_inv',      'H0_pool_inv'),   # derived below
        ('P_VARLINGAM_graph','VARLiNGAM_pool'),
        ('P_LLM_graph',      'LLM_pool'),
        ('P_PC_graph',       'PC_pool'),
    ]

    # Build series dict (including derived columns)
    col_data = {col: df[col].values.astype(float)
                for col in df.columns if col != 'H0_pool_inv'}
    col_data['H0_pool_inv'] = 1.0 - df['P_H0_WindowClosed'].values.astype(float)

    result = {
        'n_total':  int(len(df)),
        'n_steady': int(ss_mask.sum()),
        'transition_buffer_min': transition_buffer_min,
    }

    epsilon = 1e-10
    for col, label in methods:
        preds = np.clip(col_data[col], epsilon, 1 - epsilon)

        # Full-window metrics
        result[f'{label}_brier']    = float(np.mean((preds - actual) ** 2))
        result[f'{label}_log_loss'] = float(-np.mean(
            actual * np.log(preds) + (1 - actual) * np.log(1 - preds)))

        # Steady-state metrics
        if ss_mask.sum() > 0:
            p_ss = preds[ss_mask]
            a_ss = actual[ss_mask]
            result[f'{label}_brier_ss']    = float(np.mean((p_ss - a_ss) ** 2))
            result[f'{label}_log_loss_ss'] = float(-np.mean(
                a_ss * np.log(p_ss) + (1 - a_ss) * np.log(1 - p_ss)))
        else:
            result[f'{label}_brier_ss']    = float('nan')
            result[f'{label}_log_loss_ss'] = float('nan')

    return result


def compute_calibration_curve(P_H1: List[float], 
                               actual_states: List[int],
                               n_bins: int = 10) -> Tuple[List, List]:
    """
    Compute calibration curve: predicted vs observed frequencies.
    
    Returns:
        (predicted_probs, observed_freqs) for each bin
    """
    P_H1 = np.array(P_H1)
    actual = np.array(actual_states)
    
    bins = np.linspace(0, 1, n_bins + 1)
    predicted_probs = []
    observed_freqs = []
    
    for i in range(n_bins):
        mask = (P_H1 >= bins[i]) & (P_H1 < bins[i+1])
        if mask.sum() > 0:
            observed_freqs.append(actual[mask].mean())
            predicted_probs.append(P_H1[mask].mean())
    
    return predicted_probs, observed_freqs


def plot_single_run_calibration(tracker_data: Dict, 
                                 save_path: str,
                                 window_open_time: float = 30):
    """
    Create 3-panel visualization for single run.
    
    Args:
        tracker_data: Dict with 'timestamps', 'P_H1', 'actual_window_state'
        save_path: Where to save figure
        window_open_time: When window opens (minutes)
    """
    timestamps = np.array(tracker_data['timestamps'])
    P_H1 = np.array(tracker_data['P_H1'])
    P_H0 = 1 - P_H1
    actual = np.array(tracker_data['actual_window_state'])
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Calibration Analysis - Single Run', fontsize=14, fontweight='bold')
    
    # Panel 1: Time series
    ax1 = axes[0]
    ax1.fill_between(timestamps, 0, 1, where=(actual==0), 
                     alpha=0.2, color='blue', label='Window Closed (Ground Truth)')
    ax1.fill_between(timestamps, 0, 1, where=(actual==1), 
                     alpha=0.2, color='red', label='Window Open (Ground Truth)')
    ax1.plot(timestamps, P_H0, 'b-', linewidth=2, label='P(H0|data) - Window Closed')
    ax1.plot(timestamps, P_H1, 'r-', linewidth=2, label='P(H1|data) - Window Open')
    ax1.axvline(x=window_open_time, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Probability')
    ax1.set_title('Hypothesis Posteriors Over Time')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Panel 2: Calibration curve
    ax2 = axes[1]
    pred_probs, obs_freqs = compute_calibration_curve(P_H1, actual)
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    ax2.plot(pred_probs, obs_freqs, 'ro-', linewidth=2, markersize=8, 
             label='Actual Calibration')
    ax2.set_xlabel('Predicted P(Window Open)')
    ax2.set_ylabel('Observed Frequency')
    ax2.set_title('Calibration Curve')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Panel 3: Rolling metrics
    ax3 = axes[2]
    window_size = min(20, len(timestamps) // 10)
    if window_size > 5:
        brier_scores = []
        log_losses = []
        time_windows = []
        
        for i in range(window_size, len(timestamps)):
            window_pred = P_H1[i-window_size:i]
            window_actual = actual[i-window_size:i]
            
            brier = np.mean((window_pred - window_actual)**2)
            brier_scores.append(brier)
            
            epsilon = 1e-10
            P_clipped = np.clip(window_pred, epsilon, 1-epsilon)
            log_loss = -np.mean(window_actual * np.log(P_clipped) + 
                                (1-window_actual) * np.log(1-P_clipped))
            log_losses.append(log_loss)
            time_windows.append(timestamps[i])
        
        ax3.plot(time_windows, brier_scores, 'g-', linewidth=2, label='Brier Score')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(time_windows, log_losses, 'orange', linewidth=2, label='Log-Loss')
        
        ax3.axvline(x=window_open_time, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Brier Score', color='g')
        ax3_twin.set_ylabel('Log-Loss', color='orange')
        ax3.set_title('Calibration Metrics Evolution')
        ax3.tick_params(axis='y', labelcolor='g')
        ax3_twin.tick_params(axis='y', labelcolor='orange')
        ax3.grid(alpha=0.3)
        
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for rolling metrics', 
                ha='center', va='center', transform=ax3.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Calibration plot saved: {save_path}")


def create_aggregate_table(all_metrics: List[Dict], save_path: str):
    """
    Create table with aggregate statistics across runs.
    
    Args:
        all_metrics: List of dicts with 'brier_score' and 'log_loss' for each run
        save_path: Where to save table image
    """
    brier_scores = [m['brier_score'] for m in all_metrics]
    log_losses = [m['log_loss'] for m in all_metrics]
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Mean', 'Std Dev', 'Min', 'Max', 'Better'],
        ['Brier Score', 
         f'{np.mean(brier_scores):.3f}', 
         f'{np.std(brier_scores):.3f}',
         f'{np.min(brier_scores):.3f}', 
         f'{np.max(brier_scores):.3f}',
         'Lower'],
        ['Log-Loss', 
         f'{np.mean(log_losses):.3f}', 
         f'{np.std(log_losses):.3f}',
         f'{np.min(log_losses):.3f}', 
         f'{np.max(log_losses):.3f}',
         'Lower']
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.22, 0.13, 0.13, 0.13, 0.13, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, 3):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title(f'Calibration Metrics Summary (N={len(all_metrics)} runs)', 
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Aggregate table saved: {save_path}")
    
    # Also print to console
    print("\n" + "="*60)
    print("CALIBRATION METRICS SUMMARY")
    print("="*60)
    print(f"Brier Score:  {np.mean(brier_scores):.3f} ± {np.std(brier_scores):.3f}  (lower is better)")
    print(f"Log-Loss:     {np.mean(log_losses):.3f} ± {np.std(log_losses):.3f}  (lower is better)")
    print("="*60 + "\n")