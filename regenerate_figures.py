#!/usr/bin/env python3
"""
Regenerate all paper figures with consistent Okabe-Ito palette,
hatching patterns, and accessibility features.

Figures to regenerate:
1. fig_stationary_vs_closedloop.pdf — fix "22%" → "21%", keep hatching
2. fig_monitoring_comparison.pdf — add hatching + consistent colors
3. fig_physical_monitoring.pdf — add hatching + consistent colors
4. fig_discriminability.pdf — consistent colors + hatching
5. fig_regime_recovery.pdf — hatching for consistency
6. fig_wrong_prior.pdf — Okabe-Ito teal
7. fig_brier.pdf — Okabe-Ito palette

Usage:
    python regenerate_figures.py [--out-dir figures/]
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse, os

# ── Okabe-Ito palette (colorblind-safe) ──
COLORS = {
    'PolicyGRID':       '#CC0033',   # scarlet
    'PolicyGRID-O':     '#0072B2',   # blue
    'IID':              '#56B4E9',   # sky blue
    'ICP':              '#999999',   # gray
    'Causal Bandits':   '#666666',   # dark gray
    'JCI':              '#444444',   # charcoal
    'PC':               '#009E73',   # teal
    'GIES':             '#CC79A7',   # pink
    'NOTEARS-I':        '#888888',   # medium gray
    'ABCD':             '#AAAAAA',   # light gray
    'LLM-only':         '#000000',   # black
    'SAM':              '#E69F00',   # amber
    'VARLiNGAM':        '#D4A017',   # dark amber
    # Policy
    'Validated':        '#CC0033',
    'Obs-Only':         '#0072B2',
    'Random':           '#666666',
    'Empty':            '#E69F00',
    'PID':              '#D55E00',
    'MPC_Linear':       '#009E73',
    'DDPC_Behavioral':  '#56B4E9',
}

HATCHES = {
    'PolicyGRID':       '',
    'PolicyGRID-O':     '///',
    'IID':              '..',
    'ICP':              '||',
    'Causal Bandits':   'OO',
    'JCI':              '**',
    'PC':               '...',
    'GIES':             '++',
    'NOTEARS-I':        '--',
    'ABCD':             'oo',
    'LLM-only':         'xx',
    'SAM':              'xx',
    'VARLiNGAM':        '\\\\\\',
    'Validated':        '',
    'Obs-Only':         '///',
    'Random':           '...',
    'Empty':            'xxx',
    'PID':              '\\\\\\',
    'MPC_Linear':       '+++',
    'DDPC_Behavioral':  '---',
}

def _draw_bracket(ax, x, y_low, y_high, label, color, label_offset_x=0.10,
                  fontsize=15, lw=2.2, tick_w=0.05):
    """Measurement bracket (vertical line + serif ticks at both ends) with a
    labeled box to the right. Renders cleanly regardless of how close y_low and
    y_high are."""
    ax.plot([x, x], [y_low, y_high], color=color, linewidth=lw,
            solid_capstyle='round', zorder=10)
    for y in (y_low, y_high):
        ax.plot([x - tick_w, x + tick_w], [y, y],
                color=color, linewidth=lw,
                solid_capstyle='round', zorder=10)
    ax.text(x + label_offset_x, (y_low + y_high) / 2, label,
            fontsize=fontsize, fontweight='bold', color=color,
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.35', fc='white',
                      ec=color, linewidth=1.5), zorder=12)


# ── Global style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 15,
    'axes.labelsize': 16,
    'axes.titlesize': 17,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def fig_stationary_vs_closedloop(out_dir):
    """Figure 4: Stationary vs closed-loop comparison.

    Styled to match NE Agents Day slide 5 conventions: prominent percentage
    callout box with arrow pointing into the inter-bar gap at ε=1.0.
    Saved as PNG with tight bbox so LaTeX scaling preserves crisp text.
    """
    # Wide aspect (3:1) so when fit to \textwidth in two-col NeurIPS, the
    # rendered height is small and most page real estate is bar content,
    # not whitespace. Internal fonts are bumped to compensate.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5.6))

    # (a) Stationary
    eps_vals = [0.5, 0.8, 1.0, 1.5, 2.0]
    val_kwh  = [3.77, 2.73, 2.04, 0.54, 0.54]
    obs_kwh  = [3.94, 2.73, 2.36, 0.54, 0.54]
    rand_kwh = [5.35, 5.20, 5.15, 5.00, 5.00]
    empt_kwh = [5.33, 5.33, 5.33, 5.33, 5.33]

    x = np.arange(len(eps_vals))
    w = 0.2
    for i, (name, data) in enumerate([('Validated', val_kwh), ('Obs-Only', obs_kwh),
                                       ('Random', rand_kwh), ('Empty', empt_kwh)]):
        ax1.bar(x + i*w, data, w, color=COLORS[name], hatch=HATCHES[name],
                edgecolor='black', linewidth=0.9, label=name)

    ax1.set_xticks(x + 1.5*w)
    ax1.set_xticklabels([f'$\\varepsilon$={e}' for e in eps_vals], fontsize=18)
    ax1.set_ylabel('Energy (kWh)', fontsize=20)
    ax1.set_title('(a) Stationary Evaluation  (open-loop, no feedback)',
                  fontsize=21, fontweight='bold', pad=8)
    ax1.tick_params(axis='y', labelsize=17)
    # In-axes legend (top-left), full visible
    ax1.legend(loc='upper left', fontsize=16, frameon=True, framealpha=0.95,
               edgecolor='0.6', ncol=2, columnspacing=0.8, handlelength=1.5)
    # Annotation: text box near top-right
    ax1.text(3.5, 4.8, 'Both sweep ~7$\\times$ range',
             fontsize=18, color='#222222', ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', fc='#f0f0f0', ec='#999999', alpha=0.95))

    # (b) Closed-loop
    eps_cl = [0.5, 1.0, 2.0]
    methods_cl = ['Validated', 'Obs-Only', 'Random', 'Empty', 'PID']
    data_cl = {
        'Validated': [45.3, 24.6, 6.0],
        'Obs-Only':  [48.7, 31.1, 6.2],
        'Random':    [64.3, 62.6, 60.0],
        'Empty':     [66.0, 66.0, 66.0],
        'PID':       [66.0, 66.0, 66.0],
    }

    x2 = np.arange(len(eps_cl))
    w2 = 0.15
    for i, name in enumerate(methods_cl):
        ax2.bar(x2 + i*w2, data_cl[name], w2, color=COLORS[name],
                hatch=HATCHES[name], edgecolor='black', linewidth=0.9, label=name)

    ax2.set_xticks(x2 + 2*w2)
    ax2.set_xticklabels([f'$\\varepsilon$={e}' for e in eps_cl], fontsize=18)
    ax2.set_ylabel('Energy (kWh)', fontsize=20)
    ax2.set_ylim(0, 88)
    ax2.set_xlim(-0.25, 2.95)
    ax2.set_title('(b) Closed-Loop Deployment  (actions feed back, regime changes)',
                  fontsize=21, fontweight='bold', pad=8)
    ax2.tick_params(axis='y', labelsize=17)
    # In-axes legend on panel (b) too — top-right, doesn't overlap bars
    ax2.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=15,
               frameon=True, framealpha=0.95, edgecolor='0.6',
               ncol=3, columnspacing=0.6, handlelength=1.4)

    # ── Annotations ──
    # "Frozen" label centered above ε=2.0 cluster (kept well inside xlim).
    ax2.annotate('Frozen', xy=(2.0 + 3*w2, 66), fontsize=17, color='#555555',
                 fontweight='bold', ha='center', va='bottom',
                 xytext=(2.0 + 2*w2, 80),
                 arrowprops=dict(arrowstyle='->', color='#999999', lw=1.8))

    # −21% gap callout: red arrow + rounded box, mimicking NE slide 5 style.
    val_kwh_eps1 = data_cl['Validated'][1]
    obs_kwh_eps1 = data_cl['Obs-Only'][1]
    gap_pct = (obs_kwh_eps1 - val_kwh_eps1) / obs_kwh_eps1 * 100
    arrow_target = (1.0 + 0.5*w2, (val_kwh_eps1 + obs_kwh_eps1) / 2)
    label_pos = (1.55, 52)
    ax2.annotate(f'−{gap_pct:.0f}% energy\nvs Obs-Only',
                 xy=arrow_target, xytext=label_pos,
                 fontsize=18, fontweight='bold', color='#CC0033', ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white',
                           ec='#CC0033', linewidth=2.4),
                 arrowprops=dict(arrowstyle='-|>,head_width=0.4,head_length=0.55',
                                 color='#CC0033', lw=2.6,
                                 connectionstyle='arc3,rad=0.0',
                                 shrinkA=8, shrinkB=4),
                 zorder=15)

    # Tighten internal whitespace
    plt.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.10, wspace=0.18)

    # Save as both PNG (LaTeX uses this) and PDF (legacy/preview)
    png_path = os.path.join(out_dir, 'fig_stationary_vs_closedloop.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(os.path.join(out_dir, 'fig_stationary_vs_closedloop.pdf'),
                bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print('  Saved fig_stationary_vs_closedloop.{png,pdf}')


def fig_monitoring_comparison(out_dir):
    """Figure 3: Monitoring comparison - horizontal bars with hatching."""
    methods = ['PolicyGRID', 'PolicyGRID-O', 'IID', 'ICP', 'Causal Bandits',
               'JCI', 'PC', 'GIES', 'NOTEARS-I', 'ABCD', 'LLM-only', 'SAM', 'VARLiNGAM']
    occ  = [97, 99, 99, 98, 73, 65, 64, 64, 64, 59, 49, 50, 48]
    win  = [87, 76, 75, 69, 76, 76, 76, 76, 76, 76, 42, 37, 36]
    comb = [84, 74, 75, 68, 51, 52, 51, 51, 51, 51, 42, 15, 36]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    titles = ['Occupancy', 'Window', 'Combined (4-way)']
    data_sets = [occ, win, comb]

    y = np.arange(len(methods))

    for ax, title, data in zip(axes, titles, data_sets):
        for i, (m, v) in enumerate(zip(methods, data)):
            ax.barh(len(methods)-1-i, v, color=COLORS.get(m, '#888'),
                    hatch=HATCHES.get(m, ''), edgecolor='black', linewidth=0.6)
            ax.text(v + 1, len(methods)-1-i, str(v), va='center', fontsize=11)
        ax.set_xlim(0, 108)
        ax.axvline(50, color='magenta', linestyle='--', linewidth=1.5, alpha=0.7, label='Random (50%)')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Accuracy (%)')

    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels(list(reversed(methods)), fontsize=12)
    # Highlight PolicyGRID label
    labels = axes[0].get_yticklabels()
    labels[-1].set_color('#CC0033')
    labels[-1].set_fontweight('bold')

    axes[0].legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_monitoring_comparison.pdf'))
    plt.close()
    print('  Saved fig_monitoring_comparison.pdf')


def fig_discriminability(out_dir):
    """Discriminability + Brier combined figure."""
    methods = ['PolicyGRID', 'PolicyGRID-O', 'ICP', 'Causal Bandits', 'PC', 'GIES']
    occ_win = [99, 99, 98, 96, 65, 65]
    win_win = [89, 74, 68, 75, 75, 75]
    brier = [0.016, 0.086, 0.250]
    brier_labels = ['Occ', 'Win', 'Random']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5),
                                         gridspec_kw={'width_ratios': [1, 1, 0.6]})

    y = np.arange(len(methods))
    for ax, data, title in [(ax1, occ_win, 'Occupancy\nDiscriminability'),
                             (ax2, win_win, 'Window\nDiscriminability')]:
        for i, (m, v) in enumerate(zip(methods, data)):
            ax.barh(len(methods)-1-i, v, color=COLORS.get(m, '#888'),
                    hatch=HATCHES.get(m, ''), edgecolor='black', linewidth=0.6)
            ax.text(v + 1, len(methods)-1-i, f'{v}%', va='center', fontsize=10)
        ax.set_xlim(40, 108)
        ax.axvline(50, color='magenta', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Correct Model Wins (%)')

    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(list(reversed(methods)), fontsize=10)
    ax2.set_yticks([])

    # Brier
    brier_colors = ['#CC0033', '#0072B2', '#CC0033']
    brier_hatches = ['', '///', 'xxx']
    for i, (v, lbl) in enumerate(zip(brier, brier_labels)):
        ax3.bar(i, v, color=brier_colors[i], hatch=brier_hatches[i],
                edgecolor='black', linewidth=0.7)
        ax3.text(i, v + 0.008, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(brier_labels)
    ax3.set_ylabel('Brier Score')
    ax3.set_title('Calibration\n(lower = better)', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 0.30)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_discriminability.pdf'))
    plt.close()
    print('  Saved fig_discriminability.pdf')


def fig_brier(out_dir):
    """Standalone Brier score figure."""
    brier = [0.016, 0.086, 0.250]
    labels = ['Occupancy\n(PolicyGRID)', 'Window\n(PolicyGRID)', 'Random\nBaseline']
    colors = ['#0072B2', '#0072B2', '#CC0033']
    hatches = ['', '///', 'xxx']

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (v, lbl) in enumerate(zip(brier, labels)):
        ax.bar(i, v, color=colors[i], hatch=hatches[i],
               edgecolor='black', linewidth=0.7)
        ax.text(i, v + 0.008, f'{v:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Brier Score (lower = better calibrated)', fontsize=12)
    ax.set_title('Monitoring Calibration', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.30)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_brier.pdf'))
    plt.close()
    print('  Saved fig_brier.pdf')


def fig_regime_recovery(out_dir):
    """Regime-shift detection delay with hatching."""
    transitions = ['Base>Occ', 'Base>Win', 'Base>Full', 'Occ>Base', 'Full>Base']
    occ_delay = [25, 0, 27, 20, 16]
    win_delay = [0, 0, 0, 0, 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(transitions))
    w = 0.35

    ax.bar(x - w/2, occ_delay, w, color='#CC0033', hatch='', edgecolor='black',
           linewidth=0.7, label='Occupancy delay')
    ax.bar(x + w/2, win_delay, w, color='#0072B2', hatch='///', edgecolor='black',
           linewidth=0.7, label='Window delay')

    for i, (o, wi) in enumerate(zip(occ_delay, win_delay)):
        if o > 0:
            ax.text(i - w/2, o + 0.5, str(o), ha='center', va='bottom',
                    fontsize=11, color='#CC0033', fontweight='bold')
        ax.text(i + w/2, wi + 0.5, str(wi), ha='center', va='bottom',
                fontsize=11, color='#0072B2', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(transitions, fontsize=11)
    ax.set_ylabel('Steps to detection', fontsize=12)
    ax.set_title('Regime-Shift Detection Delay', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_regime_recovery.pdf'))
    plt.close()
    print('  Saved fig_regime_recovery.pdf')


def fig_wrong_prior(out_dir):
    """Wrong-prior recovery with Okabe-Ito color."""
    edges = ['HVAC>Solar', 'Light>Humid', 'HVAC>Noise', 'Light>OutT', 'HVAC>WinPos']
    one_minus_p = [0.38, 0.42, 0.67, 0.02, 0.52]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    y = np.arange(len(edges))

    ax.barh(y, one_minus_p, color='#009E73', hatch='', edgecolor='black', linewidth=0.7)
    for i, v in enumerate(one_minus_p):
        ax.text(max(v, 0.05) + 0.02, i, 'REJECTED', va='center', fontsize=11,
                color='#CC0033', fontweight='bold')

    ax.axvline(0.95, color='magenta', linestyle='--', linewidth=2, label='$\\alpha$=0.05 threshold')
    ax.set_yticks(y)
    ax.set_yticklabels(edges, fontsize=11)
    ax.set_xlabel('1 - p-value (higher = more "significant")', fontsize=12)
    ax.set_title('Injected False Edges: All Rejected by FDR', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_wrong_prior.pdf'))
    plt.close()
    print('  Saved fig_wrong_prior.pdf')


def fig_physical_monitoring(out_dir):
    """Physical monitoring - 4 panels with hatching. Span-2-col, bigger text,
    tight padding so PNG fills the page real estate it occupies."""
    methods = ['PolicyGRID', 'PolicyGRID-O', 'SAM', 'PC', 'GIES', 'VARLiNGAM',
               'LLM-only', 'IID', 'Causal Bandits', 'JCI', 'ICP', 'NOTEARS-I']

    data = {
        'Occupancy (OOD)': [60, 72, 76, 51, 38, 62, 40, 38, 46, 50, 38, 36],
        'Window (OOD)': [75, 78, 79, 44, 47, 75, 55, 50, 34, 42, 25, 37],
        'Occupancy (In-dist)': [82, 78, 68, 78, 71, 73, 78, 73, 73, 81, 68, 71],
        'Window (In-dist)': [91, 89, 89, 89, 84, 84, 85, 88, 85, 87, 84, 85],
    }

    # Very wide & shallow aspect (≈5:1) → minimal vertical real estate when
    # rendered at \textwidth. Bars stay readable because text scales with bbox.
    fig, axes = plt.subplots(1, 4, figsize=(28, 5.5), sharey=True)

    for ax, (title, vals) in zip(axes, data.items()):
        for i, (m, v) in enumerate(zip(methods, vals)):
            ax.barh(len(methods)-1-i, v, color=COLORS.get(m, '#888'),
                    hatch=HATCHES.get(m, ''), edgecolor='black', linewidth=0.9)
            ax.text(v + 1.2, len(methods)-1-i, str(v), va='center',
                    fontsize=14, fontweight='bold')
        ax.set_xlim(0, 112)
        ax.axvline(50, color='magenta', linestyle='--', linewidth=1.8, alpha=0.85)
        ax.set_title(title, fontsize=18, fontweight='bold', pad=6)
        ax.set_xlabel('Accuracy (%)', fontsize=15)
        ax.tick_params(axis='x', labelsize=13)

    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels(list(reversed(methods)), fontsize=14, fontweight='bold')
    labels = axes[0].get_yticklabels()
    labels[-1].set_color('#CC0033')
    labels[-1].set_fontweight('bold')

    plt.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.13, wspace=0.08)

    png_path = os.path.join(out_dir, 'fig_physical_monitoring.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(os.path.join(out_dir, 'fig_physical_monitoring.pdf'),
                bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print('  Saved fig_physical_monitoring.{png,pdf}')


def fig_discovery_summary(out_dir):
    """Section 5.1 main-paper figure: F1 across 12 methods, two panels.
    (a) primary env smart_building_rich (the headline);
    (b) cross-environment mean (breadth claim).  All fonts ≥ 13pt."""
    methods = ['PolicyGRID', 'LLM-only', 'IID', 'PC', 'GIES', 'SAM',
               'ABCD', 'Causal Bandits', 'JCI', 'VARLiNGAM', 'NOTEARS-I', 'ICP']
    # smart_building_rich F1 (from tab:discovery_full)
    sbr  = [0.376, 0.441, 0.433, 0.384, 0.352, 0.268,
            0.262, 0.231, 0.207, 0.175, 0.039, 0.032]
    sbr_e = [0.054, 0.038, 0.000, 0.017, 0.091, 0.027,
             0.025, 0.022, 0.044, 0.000, 0.000, 0.000]
    # mean across 6 envs (sbr, ow, sr, sr_noise, sr_hidden, ashrae);
    # recomputed from tab:discovery_full per-env values, May 2026.
    mean = [0.707, 0.510, 0.508, 0.436, 0.410, 0.497,
            0.194, 0.255, 0.330, 0.365, 0.240, 0.330]

    # Sort by sbr F1 descending so the bars stack from best → worst
    order = sorted(range(len(methods)), key=lambda i: -sbr[i])
    methods = [methods[i] for i in order]
    sbr     = [sbr[i]     for i in order]
    sbr_e   = [sbr_e[i]   for i in order]
    mean    = [mean[i]    for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    n = len(methods)

    for ax, vals, errs, title in [
        (ax1, sbr, sbr_e, '(a) Primary env: smart_building_rich (15V, 28E)'),
        (ax2, mean, None,  '(b) Mean F1 across 6 environments'),
    ]:
        for i, m in enumerate(methods):
            v = vals[i]
            e = errs[i] if errs else 0
            y = n - 1 - i
            # Asymmetric xerr: error bar extends LEFTWARD only (into the bar),
            # leaving the right side clean for the value label.
            xerr = ([[e], [0]] if errs else None)
            ax.barh(y, v, color=COLORS.get(m, '#888'),
                    hatch=HATCHES.get(m, ''), edgecolor='black', linewidth=0.9,
                    xerr=xerr,
                    error_kw={'ecolor': '0.15', 'capsize': 4, 'lw': 1.4} if errs else {})
            # Value label always to the right of the bar end, with extra padding
            ax.text(v + 0.018, y, f'{v:.3f}', va='center', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Directed-edge F1', fontsize=16)
        ax.set_title(title, fontsize=17, fontweight='bold', pad=8)
        ax.tick_params(axis='x', labelsize=13)
        ax.grid(axis='x', alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

    ax1.set_yticks(range(n))
    ax1.set_yticklabels(list(reversed(methods)), fontsize=14, fontweight='bold')
    # Highlight PolicyGRID (which is now somewhere in the middle of sbr ranking)
    pg_idx = methods.index('PolicyGRID')
    yticklabels = ax1.get_yticklabels()
    yticklabels[n - 1 - pg_idx].set_color('#CC0033')

    plt.subplots_adjust(left=0.13, right=0.99, top=0.91, bottom=0.11, wspace=0.06)
    plt.savefig(os.path.join(out_dir, 'fig_discovery_summary.png'),
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(os.path.join(out_dir, 'fig_discovery_summary.pdf'),
                bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print('  Saved fig_discovery_summary.{png,pdf}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='./figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Regenerating figures to {args.out_dir}...')

    fig_stationary_vs_closedloop(args.out_dir)
    fig_monitoring_comparison(args.out_dir)
    fig_discriminability(args.out_dir)
    fig_brier(args.out_dir)
    fig_regime_recovery(args.out_dir)
    fig_wrong_prior(args.out_dir)
    fig_physical_monitoring(args.out_dir)
    fig_discovery_summary(args.out_dir)

    print('\nDone. Figures not regenerated (already accessible):')
    print('  - fig_f1_*.png (6 panels + legend) — Okabe-Ito + hatching')
    print('  - fig_physical_timeseries.pdf — red/blue with shading')
    print('  - fig_confusion.pdf — sequential blue colormap with labels')
    print('  - fig_seed_stability.pdf — two-panel, values labeled')
    print('  - fig_sensor_ablation.pdf — line plot with distinct markers')
