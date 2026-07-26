#!/usr/bin/env python3
"""
Cross-Simulator Figures for NeurIPS 2026
========================================
Generates publication-quality figures from master-3seed-results/.
B&W compatible, colorblind friendly, clean and intuitive.

Usage:
    python scripts/cross_sim_figures.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results' / 'master-3seed-results'

# ─── Style ────────────────────────────────────────────────────────────────────
# Wong colorblind-safe palette + B&W markers/hatches
COLORS = {
    'PolicyGRID': '#D55E00',     # vermillion
    'PolicyGRID-O': '#E69F00',   # orange
    'PC': '#56B4E9',             # sky blue
    'SAM': '#009E73',            # bluish green
    'GIES': '#F0E442',           # yellow
    'ICP': '#0072B2',            # blue
    'NOTEARS-I': '#CC79A7',      # reddish purple
    'ABCD': '#666666',
    'JCI': '#999999',
    'Causal Bandits': '#BBBBBB',
    'IID': '#444444',
}
MARKERS = {
    'PolicyGRID': 'o', 'PolicyGRID-O': 's', 'PC': '^', 'SAM': 'D',
    'GIES': 'v', 'ICP': '<', 'NOTEARS-I': '>', 'ABCD': 'p',
    'JCI': 'h', 'Causal Bandits': '*', 'IID': 'X',
}
LINESTYLES = {
    'PolicyGRID': '-', 'PolicyGRID-O': '--', 'PC': ':', 'SAM': '-.',
    'GIES': '-', 'ICP': '--', 'NOTEARS-I': ':', 'ABCD': '-.',
    'JCI': '-', 'Causal Bandits': '--', 'IID': ':',
}
HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

# Sim ordering by complexity
SIM_ORDER = ['smart_room', 'smart_room_noise', 'smart_room_hidden_vars',
             'ashrae', 'open_window', 'smart_building_rich']
SIM_LABELS = {
    'smart_room': 'Base\n(5 vars)',
    'smart_room_noise': 'Noisy\n(5 vars)',
    'smart_room_hidden_vars': 'Hidden\n(6 vars)',
    'ashrae': 'ASHRAE\n(6 vars)',
    'open_window': 'Window\n(8 vars)',
    'smart_building_rich': 'Building\n(15 vars)',
}
SIM_NVARS = {
    'smart_room': 5, 'smart_room_noise': 5, 'smart_room_hidden_vars': 6,
    'ashrae': 6, 'open_window': 8, 'smart_building_rich': 15,
}

plt.rcParams.update({
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def load_cross_sim_data():
    """Load cross-sim discovery CSV."""
    csv_path = RESULTS / 'cross_sim' / 'cross_sim_discovery.csv'
    if not csv_path.exists():
        # Try to assemble from per-sim CSVs
        dfs = []
        for sim in SIM_ORDER:
            p = RESULTS / sim / 'discovery_metrics.csv'
            if p.exists():
                dfs.append(pd.read_csv(p))
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            (RESULTS / 'cross_sim').mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            return df
        return None
    return pd.read_csv(csv_path)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: COMPLEXITY vs F1 — THE INFLECTION POINT
# ═════════════════════════════════════════════════════════════════════════════

def fig_complexity_vs_f1(df, out_dir):
    """X: variable count, Y: F1. Lines for PolicyGRID, PolicyGRID-O, best benchmark."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Get mean F1 per method × sim
    summary = df.groupby(['sim', 'method']).agg(F1=('F1', 'mean')).reset_index()

    # PolicyGRID line
    for method, color, marker, ls, label in [
        ('PolicyGRID', '#D55E00', 'o', '-', 'PolicyGRID (validated)'),
        ('PolicyGRID-O', '#E69F00', 's', '--', 'PolicyGRID-O (obs-only)'),
    ]:
        vals = []
        xs = []
        for sim in SIM_ORDER:
            row = summary[(summary['sim'] == sim) & (summary['method'] == method)]
            if len(row) > 0:
                xs.append(SIM_NVARS[sim])
                vals.append(row['F1'].values[0])
        if xs:
            ax.plot(xs, vals, color=color, marker=marker, linestyle=ls,
                    linewidth=2.5, markersize=9, label=label, zorder=5)

    # Best benchmark per sim (excluding PolicyGRID variants)
    benchmark_methods = [m for m in summary['method'].unique()
                        if 'PolicyGRID' not in m]
    best_bench = []
    best_xs = []
    for sim in SIM_ORDER:
        sim_bench = summary[(summary['sim'] == sim) &
                           (summary['method'].isin(benchmark_methods))]
        if len(sim_bench) > 0:
            best = sim_bench.loc[sim_bench['F1'].idxmax()]
            best_xs.append(SIM_NVARS[sim])
            best_bench.append(best['F1'])
    if best_xs:
        ax.plot(best_xs, best_bench, color='#666666', marker='^',
                linestyle=':', linewidth=2, markersize=8,
                label='Best benchmark', zorder=4)

    # Add sim labels at bottom
    for sim in SIM_ORDER:
        nv = SIM_NVARS[sim]
        ax.annotate(SIM_LABELS[sim].replace('\n', ' '),
                    (nv, -0.08), ha='center', va='top', fontsize=8,
                    color='gray', rotation=0)

    # Annotate inflection region
    ax.axvspan(7, 16, alpha=0.08, color='red', zorder=0)
    ax.annotate('Complexity\nInflection', xy=(11.5, 0.85),
                fontsize=10, ha='center', color='#D55E00', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#D55E00', alpha=0.9))

    ax.set_xlabel('Number of Variables', fontsize=13)
    ax.set_ylabel('F1 Score', fontsize=13)
    ax.set_title('Causal Discovery: F1 vs Problem Complexity', fontsize=14,
                 fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(3.5, 17)
    ax.legend(loc='lower left', framealpha=0.9, fontsize=10)

    fig.savefig(out_dir / 'fig_complexity_vs_f1.png')
    fig.savefig(out_dir / 'fig_complexity_vs_f1.pdf')
    plt.close(fig)
    print(f"  Saved fig_complexity_vs_f1")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: NORMALIZED SHD vs COMPLEXITY
# ═════════════════════════════════════════════════════════════════════════════

def fig_complexity_vs_shd(df, out_dir):
    """X: variable count, Y: SHD / n(n-1) (normalized by possible edges)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    summary = df.groupby(['sim', 'method']).agg(SHD=('SHD', 'mean')).reset_index()

    for method, color, marker, ls, label in [
        ('PolicyGRID', '#D55E00', 'o', '-', 'PolicyGRID'),
        ('PolicyGRID-O', '#E69F00', 's', '--', 'PolicyGRID-O'),
    ]:
        vals = []
        xs = []
        for sim in SIM_ORDER:
            row = summary[(summary['sim'] == sim) & (summary['method'] == method)]
            if len(row) > 0:
                n = SIM_NVARS[sim]
                norm_shd = row['SHD'].values[0] / (n * (n - 1)) if n > 1 else 0
                xs.append(n)
                vals.append(norm_shd)
        if xs:
            ax.plot(xs, vals, color=color, marker=marker, linestyle=ls,
                    linewidth=2.5, markersize=9, label=label, zorder=5)

    # Best benchmark
    benchmark_methods = [m for m in summary['method'].unique()
                        if 'PolicyGRID' not in m]
    best_bench = []
    best_xs = []
    for sim in SIM_ORDER:
        sim_bench = summary[(summary['sim'] == sim) &
                           (summary['method'].isin(benchmark_methods))]
        if len(sim_bench) > 0:
            best = sim_bench.loc[sim_bench['SHD'].idxmin()]
            n = SIM_NVARS[sim]
            best_xs.append(n)
            best_bench.append(best['SHD'] / (n * (n - 1)) if n > 1 else 0)
    if best_xs:
        ax.plot(best_xs, best_bench, color='#666666', marker='^',
                linestyle=':', linewidth=2, markersize=8,
                label='Best benchmark', zorder=4)

    ax.set_xlabel('Number of Variables', fontsize=13)
    ax.set_ylabel('Normalized SHD (SHD / n(n-1))', fontsize=13)
    ax.set_title('Structural Hamming Distance vs Problem Complexity',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

    fig.savefig(out_dir / 'fig_complexity_vs_shd.png')
    fig.savefig(out_dir / 'fig_complexity_vs_shd.pdf')
    plt.close(fig)
    print(f"  Saved fig_complexity_vs_shd")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: DISCOVERY HEATMAP
# ═════════════════════════════════════════════════════════════════════════════

def fig_discovery_heatmap(df, out_dir):
    """Method × Sim F1 matrix with grayscale + annotations."""
    summary = df.groupby(['sim', 'method']).agg(F1=('F1', 'mean')).reset_index()
    pivot = summary.pivot(index='method', columns='sim', values='F1')

    # Reorder columns by complexity
    cols = [s for s in SIM_ORDER if s in pivot.columns]
    methods_order = ['PolicyGRID', 'PolicyGRID-O', 'PC', 'SAM', 'GIES', 'ICP',
                     'NOTEARS-I', 'ABCD', 'JCI', 'Causal Bandits', 'IID']
    rows = [m for m in methods_order if m in pivot.index]

    pivot = pivot.loc[rows, cols]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                   vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            # Bold if best in column
            col_vals = pivot.iloc[:, j].dropna()
            is_best = val == col_vals.max()
            weight = 'bold' if is_best else 'normal'
            color = 'white' if val < 0.3 or val > 0.8 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, fontweight=weight, color=color)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([SIM_LABELS[s] for s in cols], fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=10)
    ax.set_title('Discovery F1 Score: Method × Simulator', fontsize=14,
                 fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('F1 Score', fontsize=11)

    fig.savefig(out_dir / 'fig_discovery_heatmap.png')
    fig.savefig(out_dir / 'fig_discovery_heatmap.pdf')
    plt.close(fig)
    print(f"  Saved fig_discovery_heatmap")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: INTERVENTION VALUE-ADD
# ═════════════════════════════════════════════════════════════════════════════

def fig_intervention_value(df, out_dir):
    """Bar chart: F1(PolicyGRID) - F1(PolicyGRID-O) per sim."""
    summary = df.groupby(['sim', 'method']).agg(F1=('F1', 'mean')).reset_index()

    sims = [s for s in SIM_ORDER
            if s in summary['sim'].unique()]
    deltas = []
    labels = []
    for sim in sims:
        grid_val = summary[(summary['sim'] == sim) &
                          (summary['method'] == 'PolicyGRID')]
        grid_obs = summary[(summary['sim'] == sim) &
                          (summary['method'] == 'PolicyGRID-O')]
        if len(grid_val) > 0 and len(grid_obs) > 0:
            delta = grid_val['F1'].values[0] - grid_obs['F1'].values[0]
            deltas.append(delta)
            labels.append(SIM_LABELS[sim].replace('\n', ' '))

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#D55E00' if d > 0.01 else '#999999' for d in deltas]
    bars = ax.bar(range(len(deltas)), deltas, color=colors, edgecolor='black',
                  linewidth=0.8)

    # Add value labels
    for i, (bar, delta) in enumerate(zip(bars, deltas)):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.005,
                f'{delta:+.3f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold' if abs(delta) > 0.01 else 'normal')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('\u0394 F1 (Validated \u2212 Obs-only)', fontsize=12)
    ax.set_title('Value of Interventional Validation', fontsize=14,
                 fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Annotate
    ax.annotate('Validation helps\nwhen graph is complex',
                xy=(len(deltas) - 1, max(deltas) * 0.8),
                fontsize=9, ha='center', color='#D55E00',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#D55E00', alpha=0.8))

    fig.savefig(out_dir / 'fig_intervention_value.png')
    fig.savefig(out_dir / 'fig_intervention_value.pdf')
    plt.close(fig)
    print(f"  Saved fig_intervention_value")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5: METHOD RANKING STABILITY
# ═════════════════════════════════════════════════════════════════════════════

def fig_method_ranking(df, out_dir):
    """Bump chart: each method's rank per sim, connected by lines."""
    summary = df.groupby(['sim', 'method']).agg(F1=('F1', 'mean')).reset_index()

    sims = [s for s in SIM_ORDER if s in summary['sim'].unique()]

    # Compute ranks per sim
    ranks = {}
    for sim in sims:
        sim_data = summary[summary['sim'] == sim].sort_values('F1', ascending=False)
        for rank, (_, row) in enumerate(sim_data.iterrows(), 1):
            if row['method'] not in ranks:
                ranks[row['method']] = {}
            ranks[row['method']][sim] = rank

    fig, ax = plt.subplots(figsize=(10, 6))

    # Highlight PolicyGRID variants
    highlight = {'PolicyGRID', 'PolicyGRID-O'}
    for method, sim_ranks in ranks.items():
        xs = [i for i, s in enumerate(sims) if s in sim_ranks]
        ys = [sim_ranks[sims[i]] for i in xs]
        if not xs:
            continue

        is_highlight = method in highlight
        color = COLORS.get(method, '#AAAAAA')
        lw = 2.5 if is_highlight else 1.0
        alpha = 1.0 if is_highlight else 0.4
        marker = MARKERS.get(method, '.')
        ms = 8 if is_highlight else 5
        zorder = 5 if is_highlight else 2

        ax.plot(xs, ys, color=color, marker=marker, linewidth=lw,
                alpha=alpha, markersize=ms, label=method, zorder=zorder)

    ax.set_xticks(range(len(sims)))
    ax.set_xticklabels([SIM_LABELS[s] for s in sims], fontsize=9)
    ax.set_ylabel('Rank (1 = best)', fontsize=12)
    ax.set_title('Method Ranking Stability Across Simulators', fontsize=14,
                 fontweight='bold')
    ax.invert_yaxis()
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=8, framealpha=0.9)

    fig.savefig(out_dir / 'fig_method_ranking.png')
    fig.savefig(out_dir / 'fig_method_ranking.pdf')
    plt.close(fig)
    print(f"  Saved fig_method_ranking")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6: PER-SIM DISCOVERY BARS
# ═════════════════════════════════════════════════════════════════════════════

def fig_per_sim_discovery(df, out_dir):
    """Per-sim F1 bar chart saved in each sim's folder."""
    for sim in df['sim'].unique():
        sim_data = df[df['sim'] == sim].groupby('method').agg(
            F1_mean=('F1', 'mean'), F1_std=('F1', 'std')).reset_index()
        sim_data = sim_data.sort_values('F1_mean', ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(4, len(sim_data) * 0.4)))

        colors = [COLORS.get(m, '#AAAAAA') for m in sim_data['method']]
        bars = ax.barh(range(len(sim_data)), sim_data['F1_mean'],
                       xerr=sim_data['F1_std'].fillna(0),
                       color=colors, edgecolor='black', linewidth=0.5,
                       capsize=3)

        # Hatch for B&W
        for i, bar in enumerate(bars):
            bar.set_hatch(HATCHES[i % len(HATCHES)])

        ax.set_yticks(range(len(sim_data)))
        ax.set_yticklabels(sim_data['method'], fontsize=10)
        ax.set_xlabel('F1 Score', fontsize=12)
        ax.set_title(f'Discovery: {SIM_LABELS.get(sim, sim)}',
                     fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.05)

        sim_dir = RESULTS / sim
        sim_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(sim_dir / 'fig_discovery_bars.png')
        plt.close(fig)

    print(f"  Saved per-sim discovery bar charts")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("Cross-Simulator Figure Generation")
    print("=" * 50)

    df = load_cross_sim_data()
    if df is None:
        print("ERROR: No cross-sim discovery data found.")
        print("Run master_experiments.py first.")
        return

    print(f"Loaded {len(df)} rows across {df['sim'].nunique()} sims, "
          f"{df['method'].nunique()} methods")

    out_dir = RESULTS / 'cross_sim'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate all figures
    fig_complexity_vs_f1(df, out_dir)
    fig_complexity_vs_shd(df, out_dir)
    fig_discovery_heatmap(df, out_dir)
    fig_intervention_value(df, out_dir)
    fig_method_ranking(df, out_dir)
    fig_per_sim_discovery(df, out_dir)

    # Write text summary
    summary = df.groupby(['sim', 'method']).agg(
        F1_mean=('F1', 'mean'), F1_std=('F1', 'std'),
        SHD_mean=('SHD', 'mean'),
    ).reset_index()

    with open(out_dir / 'cross_sim_summary.txt', 'w') as f:
        f.write("Cross-Simulator Discovery Summary\n")
        f.write("=" * 50 + "\n\n")

        for sim in SIM_ORDER:
            sim_data = summary[summary['sim'] == sim]
            if len(sim_data) == 0:
                continue
            f.write(f"\n{sim} ({SIM_NVARS.get(sim, '?')} vars):\n")
            f.write("-" * 40 + "\n")
            best = sim_data.loc[sim_data['F1_mean'].idxmax()]
            f.write(f"  Best: {best['method']} (F1={best['F1_mean']:.3f})\n")
            for _, row in sim_data.sort_values('F1_mean', ascending=False).iterrows():
                std = f"±{row['F1_std']:.3f}" if pd.notna(row['F1_std']) else ""
                f.write(f"  {row['method']:20s} F1={row['F1_mean']:.3f}{std}  "
                        f"SHD={row['SHD_mean']:.1f}\n")

    print(f"\nAll figures saved to {out_dir}")


if __name__ == '__main__':
    main()
