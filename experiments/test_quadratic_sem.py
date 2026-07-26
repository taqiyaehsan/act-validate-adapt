"""
Quadratic vs Linear SEM comparison for regime-aware monitoring.

Runs the open_window JS simulator once, then replays the trajectory
with both linear and quadratic H0/H1 models — identical data,
different SEM types — to check whether quadratic terms help, hurt,
or don't change regime detection.

Usage:
    python test_quadratic_sem.py
"""
import os, sys, json, time, subprocess, tempfile
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.causal_world_model import PredictiveModel

# ── Configuration ────────────────────────────────────────────────────────────
SMART_ROOM_PATH = './open_window.js'
DATASET_PATH    = 'data/challenge1_data_10k_processed.csv'
SCALING_PATH    = 'data/challenge1_data_10k_processed_scaling.csv'
DURATION        = 300   # seconds of simulated time
SAMPLE_INTERVAL = 5     # seconds between samples
OUTPUT_DIR      = 'results/quadratic_sem_comparison'

# Validated edges from a recent successful run (seed 42)
VALIDATED_EDGES = [
    ('temperature', 'satisfaction'),
    ('temperature', 'airquality'),
    ('energyconsumption', 'satisfaction'),
    ('satisfaction', 'pmv'),
    ('temperature', 'energyconsumption'),
    ('humidity', 'satisfaction'),
    ('humidity', 'airquality'),
    ('windowopen', 'energyconsumption'),
    ('pmv', 'temperature'),
    ('windowopen', 'temperature'),
    ('airquality', 'satisfaction'),
    ('temperature', 'windowopen'),
    ('satisfaction', 'energyconsumption'),
    ('energyconsumption', 'windowopen'),
    ('temperature', 'pmv'),
    ('satisfaction', 'temperature'),
    ('windowopen', 'pmv'),
    ('satisfaction', 'windowopen'),
    ('airquality', 'pmv'),
]

VARIABLES = ['temperature', 'humidity', 'airquality', 'pmv',
             'energyconsumption', 'satisfaction', 'windowopen',
             'outdoortemperature']


def build_h0_h1_graphs(validated_edges):
    """Build H0 (window-stripped) and H1 (window-augmented) graphs."""
    # H0: validated edges minus anything with 'window'
    G_H0 = nx.DiGraph()
    for src, tgt in validated_edges:
        if 'window' not in src and 'window' not in tgt:
            G_H0.add_edge(src, tgt)
    for v in VARIABLES:
        if v not in G_H0 and 'window' not in v:
            G_H0.add_node(v)

    # H1: H0 + outdoor → windowopen → temperature
    G_H1 = G_H0.copy()
    for node in ['windowopen', 'outdoortemperature']:
        if node not in G_H1:
            G_H1.add_node(node)
    G_H1.add_edge('outdoortemperature', 'windowopen')
    G_H1.add_edge('windowopen', 'temperature')

    logger.info(f"H0: {len(G_H0.edges())} edges | H1: {len(G_H1.edges())} edges")
    return G_H0, G_H1


def prepare_temporal_data(df, regime_filter=None):
    """Build (t, t+1) temporal pairs from dataset."""
    obs = df.copy()
    obs.columns = [c.lower() for c in obs.columns]

    if regime_filter == 'closed' and 'windowopen' in obs.columns:
        obs = obs[obs['windowopen'] == 0]
    elif regime_filter == 'open' and 'windowopen' in obs.columns:
        obs = obs[obs['windowopen'] == 1]

    # Build _next columns
    for col in obs.columns:
        obs[f'{col}_next'] = obs[col].shift(-1)
    obs = obs.iloc[:-1].reset_index(drop=True)
    return obs


def train_models(G_H0, G_H1, df, quadratic=False):
    """Train H0 and H1 PredictiveModels and compute sigma."""
    mode = "quadratic" if quadratic else "linear"
    logger.info(f"\n=== Training {mode.upper()} H0/H1 models ===")

    H0_model = PredictiveModel(G_H0, VARIABLES, quadratic=quadratic)
    H1_model = PredictiveModel(G_H1, VARIABLES, quadratic=quadratic)

    # H0 trained on closed-regime only (80/20 split for sigma)
    temporal_closed = prepare_temporal_data(df, regime_filter='closed')
    n_closed = len(temporal_closed)
    split = int(0.8 * n_closed)
    temporal_closed_train = temporal_closed.iloc[:split]
    temporal_closed_val   = temporal_closed.iloc[split:]
    if len(temporal_closed_train) < 10:
        temporal_closed_train = temporal_closed
        temporal_closed_val   = temporal_closed

    H0_model.fit_temporal(temporal_closed_train)

    # H1 trained on all data
    temporal_all = prepare_temporal_data(df, regime_filter=None)
    H1_model.fit_temporal(temporal_all)

    # Synchronise intercepts
    h0_temp = H0_model.parameters.get('temperature', {})
    h1_temp = H1_model.parameters.get('temperature', {})
    if h0_temp.get('weights'):
        h1_temp['intercept'] = h0_temp.get('intercept', 0)

    # Compute sigma from H0 validation residuals
    val_errs = []
    for _, row in temporal_closed_val.iterrows():
        row_dict = {k: v for k, v in dict(row).items() if not k.endswith('_next')}
        pred = H0_model.predict(row_dict)
        err = abs(pred.get('temperature', 0) - row_dict.get('temperature', 0))
        val_errs.append(err)
    mean_err = float(np.mean(val_errs)) if val_errs else 0.1
    sigma = min(1.0 / max(mean_err, 1e-3), 30.0)

    logger.info(f"  {mode} sigma = {sigma:.2f} (mean_val_err = {mean_err:.4f})")
    logger.info(f"  H0 temp weights: {h0_temp.get('weights', {})}")
    logger.info(f"  H1 temp weights: {h1_temp.get('weights', {})}")
    if quadratic:
        logger.info(f"  H0 temp quad: {h0_temp.get('quad_weights', {})}")
        logger.info(f"  H1 temp quad: {h1_temp.get('quad_weights', {})}")

    return H0_model, H1_model, sigma


def collect_trajectory():
    """Run the JS simulator and collect raw state trajectory."""
    logger.info(f"\n=== Collecting simulator trajectory ({DURATION}s) ===")

    scaling = pd.read_csv(SCALING_PATH)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write(f'''
        const {{ SmartRoom }} = require('{SMART_ROOM_PATH}');
        const room = new SmartRoom();
        room._initialize().then(() => {{
            room.start();
            const interval = setInterval(() => {{
                room.updateDependentVariables().then(() => {{
                    const state = room.getState();
                    console.log('STATE:' + JSON.stringify(state));
                }}).catch(err => console.error(err));
            }}, {SAMPLE_INTERVAL * 1000});
            setTimeout(() => {{
                clearInterval(interval);
                room.stop();
                room.cleanup().then(() => process.exit(0));
            }}, {DURATION * 1000});
        }});
        ''')
        temp_path = f.name

    trajectory = []
    start_time = time.time()
    try:
        proc = subprocess.Popen(
            ['node', temp_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1
        )
        for line in iter(proc.stdout.readline, ''):
            if 'STATE:' not in line:
                continue
            state_json = line.split('STATE:', 1)[1]
            state = json.loads(state_json)

            # Normalise
            state_norm = {}
            for k, v in state.items():
                if not isinstance(v, (int, float, bool)):
                    continue
                k_lower = k.lower()
                if isinstance(v, bool):
                    state_norm[k_lower] = float(v)
                    continue
                param = scaling[scaling['feature'].str.lower() == k_lower]
                if not param.empty:
                    mn = param['data_min'].values[0]
                    mx = param['data_max'].values[0]
                    state_norm[k_lower] = (v - mn) / (mx - mn)
                else:
                    state_norm[k_lower] = v

            state_norm['_time'] = time.time() - start_time
            trajectory.append(state_norm)

        proc.wait(timeout=60)
    finally:
        os.unlink(temp_path)

    logger.info(f"  Collected {len(trajectory)} timesteps")
    return trajectory


def replay_trajectory(trajectory, H0_model, H1_model, sigma):
    """Replay a pre-collected trajectory with given H0/H1 models."""
    P_H0, P_H1 = 0.5, 0.5
    FORGET = 0.05
    records = []

    for state_norm in trajectory:
        t_sec = state_norm['_time']

        # H1 gets windowopen masked (structural pathway only)
        state_for_h1 = {k: v for k, v in state_norm.items()
                        if k != 'windowopen' and k != '_time'}
        state_for_h0 = {k: v for k, v in state_norm.items() if k != '_time'}

        pred_H0 = H0_model.predict(state_for_h0)
        pred_H1 = H1_model.predict(state_for_h1)

        T_obs     = state_norm.get('temperature', 0)
        T_pred_H0 = pred_H0.get('temperature', 0)
        T_pred_H1 = pred_H1.get('temperature', 0)
        err_H0    = abs(T_obs - T_pred_H0)
        err_H1    = abs(T_obs - T_pred_H1)

        # Binary Bayesian update
        lh_H0 = np.exp(-sigma * err_H0)
        lh_H1 = np.exp(-sigma * err_H1)
        unnorm_H0 = lh_H0 * P_H0
        unnorm_H1 = lh_H1 * P_H1
        total = unnorm_H0 + unnorm_H1
        if total > 0:
            P_H0 = (1 - FORGET) * (unnorm_H0 / total) + FORGET * 0.5
            P_H1 = (1 - FORGET) * (unnorm_H1 / total) + FORGET * 0.5

        records.append({
            'time':          t_sec / 60.0,
            'actual_window': state_norm.get('windowopen', 0),
            'P_H0':          P_H0,
            'P_H1':          P_H1,
            'T_obs':         T_obs,
            'T_pred_H0':     T_pred_H0,
            'T_pred_H1':     T_pred_H1,
            'err_H0':        err_H0,
            'err_H1':        err_H1,
        })

    return pd.DataFrame(records)


def compute_brier(df):
    """Compute Brier score and steady-state Brier (excl. 15s around transitions)."""
    actual = df['actual_window'].values
    pred   = df['P_H1'].values
    brier_full = float(np.mean((pred - actual) ** 2))

    # Steady-state: exclude 15s buffer around transitions
    ss_mask = np.ones(len(df), dtype=bool)
    for i in range(1, len(df)):
        if actual[i] != actual[i-1]:
            # Mark ±3 steps (each ~5s) around transition
            lo = max(0, i - 3)
            hi = min(len(df), i + 3)
            ss_mask[lo:hi] = False
    brier_ss = float(np.mean((pred[ss_mask] - actual[ss_mask]) ** 2)) if ss_mask.sum() > 0 else brier_full
    return brier_full, brier_ss


def plot_comparison(df_linear, df_quad, output_path):
    """Side-by-side monitoring probability plots."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    for ax, df, title in [(axes[0], df_linear, 'LINEAR SEM'),
                           (axes[1], df_quad, 'QUADRATIC SEM')]:
        time_min = df['time']

        # Shade by ground truth
        current = df['actual_window'].iloc[0]
        start_idx = 0
        for i in range(1, len(df)):
            if df['actual_window'].iloc[i] != current:
                color = 'lightcoral' if current else 'lightblue'
                ax.axvspan(time_min.iloc[start_idx], time_min.iloc[i],
                          color=color, alpha=0.3, zorder=0)
                ax.axvline(time_min.iloc[i], color='gray', ls='--', lw=1.5, zorder=1)
                start_idx = i
                current = df['actual_window'].iloc[i]
        color = 'lightcoral' if current else 'lightblue'
        ax.axvspan(time_min.iloc[start_idx], time_min.iloc[-1],
                  color=color, alpha=0.3, zorder=0)

        ax.plot(time_min, df['P_H0'], color='royalblue', lw=2.5, label='P(H0) Window Closed')
        ax.plot(time_min, df['P_H1'], color='crimson', lw=2.5, label='P(H1) Window Open')
        ax.axhline(0.8, color='gray', ls=':', alpha=0.5)
        ax.axhline(0.2, color='gray', ls=':', alpha=0.5)

        brier_full, brier_ss = compute_brier(df)
        ax.set_title(f'{title}  |  Brier={brier_full:.4f}  Brier_SS={brier_ss:.4f}',
                     fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right', fontsize=10)

    axes[1].set_xlabel('Time (minutes)')

    # Legend patches
    blue_patch = mpatches.Patch(color='lightblue', alpha=0.3, label='Window Closed (GT)')
    red_patch  = mpatches.Patch(color='lightcoral', alpha=0.3, label='Window Open (GT)')
    fig.legend(handles=[blue_patch, red_patch], loc='upper center',
              ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison plot: {output_path}")


def plot_prediction_errors(df_linear, df_quad, output_path):
    """Side-by-side temperature prediction error plots."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for ax, df, title in [(axes[0], df_linear, 'LINEAR SEM'),
                           (axes[1], df_quad, 'QUADRATIC SEM')]:
        time_min = df['time']

        # Shade by ground truth
        current = df['actual_window'].iloc[0]
        start_idx = 0
        for i in range(1, len(df)):
            if df['actual_window'].iloc[i] != current:
                color = 'lightcoral' if current else 'lightblue'
                ax.axvspan(time_min.iloc[start_idx], time_min.iloc[i],
                          color=color, alpha=0.3, zorder=0)
                start_idx = i
                current = df['actual_window'].iloc[i]
        color = 'lightcoral' if current else 'lightblue'
        ax.axvspan(time_min.iloc[start_idx], time_min.iloc[-1],
                  color=color, alpha=0.3, zorder=0)

        ax.plot(time_min, df['err_H0'], color='royalblue', lw=1.5, alpha=0.7, label='|err| H0')
        ax.plot(time_min, df['err_H1'], color='crimson', lw=1.5, alpha=0.7, label='|err| H1')
        ax.set_title(f'{title} — Temperature Prediction Errors', fontsize=13, fontweight='bold')
        ax.set_ylabel('|T_obs - T_pred|')
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right')

    axes[1].set_xlabel('Time (minutes)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved error plot: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Dataset: {len(df)} rows, {list(df.columns)}")

    # Build graphs
    G_H0, G_H1 = build_h0_h1_graphs(VALIDATED_EDGES)

    # Train both model types
    H0_lin, H1_lin, sigma_lin = train_models(G_H0, G_H1, df, quadratic=False)
    H0_quad, H1_quad, sigma_quad = train_models(G_H0, G_H1, df, quadratic=True)

    # Collect ONE simulator trajectory
    trajectory = collect_trajectory()

    if len(trajectory) < 5:
        logger.error("Too few timesteps collected from simulator!")
        sys.exit(1)

    # Replay with linear models
    logger.info("\n=== Replaying with LINEAR H0/H1 ===")
    df_linear = replay_trajectory(trajectory, H0_lin, H1_lin, sigma_lin)

    # Replay with quadratic models
    logger.info("\n=== Replaying with QUADRATIC H0/H1 ===")
    df_quad = replay_trajectory(trajectory, H0_quad, H1_quad, sigma_quad)

    # Save CSVs
    df_linear.to_csv(os.path.join(OUTPUT_DIR, 'monitoring_linear.csv'), index=False)
    df_quad.to_csv(os.path.join(OUTPUT_DIR, 'monitoring_quadratic.csv'), index=False)

    # Compute metrics
    brier_lin, brier_lin_ss = compute_brier(df_linear)
    brier_quad, brier_quad_ss = compute_brier(df_quad)

    logger.info(f"\n{'='*60}")
    logger.info(f"  LINEAR:    Brier={brier_lin:.4f}  Brier_SS={brier_lin_ss:.4f}  sigma={sigma_lin:.2f}")
    logger.info(f"  QUADRATIC: Brier={brier_quad:.4f}  Brier_SS={brier_quad_ss:.4f}  sigma={sigma_quad:.2f}")
    logger.info(f"  Delta:     {brier_quad - brier_lin:+.4f} (negative = quadratic better)")
    logger.info(f"{'='*60}")

    # Save summary
    summary = {
        'linear_brier': brier_lin, 'linear_brier_ss': brier_lin_ss, 'linear_sigma': sigma_lin,
        'quadratic_brier': brier_quad, 'quadratic_brier_ss': brier_quad_ss, 'quadratic_sigma': sigma_quad,
        'delta_brier': brier_quad - brier_lin,
        'n_timesteps': len(trajectory),
        'n_window_open': int(df_linear['actual_window'].sum()),
        'n_window_closed': int((1 - df_linear['actual_window']).sum()),
    }
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate plots
    plot_comparison(df_linear, df_quad, os.path.join(OUTPUT_DIR, 'comparison_posteriors.png'))
    plot_prediction_errors(df_linear, df_quad, os.path.join(OUTPUT_DIR, 'comparison_errors.png'))

    logger.info(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
