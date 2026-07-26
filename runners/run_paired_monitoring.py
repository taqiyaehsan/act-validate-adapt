"""
Paired monitoring statistic for the advisor's §10 ask.

Computes 4-way regime classification accuracy per (method, sim-replicate seed)
using FROZEN edge sets, then runs paired t / Wilcoxon on (PolicyGRID vs Obs-Only)
and (PolicyGRID vs IID). Edges are held constant across seeds so that variation
reflects only test-data noise — the question is "is the gap robust to fresh
sim trajectories", not "is the gap stable across re-discovery."

Frozen edge sets:
  - Validated_31:  results/closedloop_server/edge_sets.json (PolicyGRID)
  - Obs-Only_47:   results/closedloop_server/edge_sets.json
  - IID_xxx:       generated once via run_iid, cached to results/paired_monitoring/iid_edges.json

Sim seeds: 5 fresh 4-day episodes (~12 min each ≈ 1h total on a Mac).

Outputs:
  results/paired_monitoring/per_seed.csv         (3 methods × 5 seeds = 15 rows)
  results/paired_monitoring/paired_stats.txt     (formatted stat summary)

Usage:
  python run_paired_monitoring.py                 # all 5 seeds
  python run_paired_monitoring.py --smoke         # 2 seeds, quick check
  python run_paired_monitoring.py --seeds 7       # custom seed count
"""
import sys, os, json, time, warnings, argparse
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import logging
from scipy.stats import ttest_rel, wilcoxon, t as tdist

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

from neurips_final_experiments import (
    construct_factored_monitors, run_monitoring, compute_monitoring_metrics,
    collect_sim_data, get_scaling_map, OBSERVABLE_VARS, SENSOR_VARS, LATENT_VARS,
    SIM_PATH,
)

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, default=5,
                    help='Number of sim-replicate seeds (default 5).')
parser.add_argument('--smoke', action='store_true',
                    help='Quick check: 2 seeds, 1440 steps each (~5 min).')
parser.add_argument('--steps', type=int, default=5760,
                    help='Sim steps per seed (default 5760 = 4 days).')
parser.add_argument('--regenerate-iid', action='store_true',
                    help='Force re-run of IID discovery to refresh edges.')
parser.add_argument('--workers', type=int, default=5,
                    help='Parallel sim-collection workers (default 5). '
                         'collect_sim_data spawns Node.js subprocesses that '
                         'release the GIL, so threads scale linearly. '
                         'Set to 1 for sequential.')
args = parser.parse_args()

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

if args.smoke:
    args.seeds = 2
    args.steps = 1440

OUT_DIR = 'results/paired_monitoring'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load training data + scaling ────────────────────────────────────────────
DATA_PATH = 'data_regen/smart_building_rich_processed.csv'
SCALING_PATH = 'data_regen/smart_building_rich_processed_scaling.csv'
data = pd.read_csv(DATA_PATH)
cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
data = data[cols]
scaling = pd.read_csv(SCALING_PATH)
smap = get_scaling_map(scaling)
logger.info(f"Loaded data: {data.shape}; scaling: {len(smap)} vars")

# ── Load frozen Validated and Obs-Only edge sets ────────────────────────────
EDGE_SETS_PATH = 'results/closedloop_server/edge_sets.json'
with open(EDGE_SETS_PATH) as f:
    edge_sets = json.load(f)
val_edges = {(s.lower(), t.lower()) for s, t in edge_sets['validated']}
obs_edges = {(s.lower(), t.lower()) for s, t in edge_sets['obs_only']}
logger.info(f"Validated: {len(val_edges)} edges; Obs-Only: {len(obs_edges)} edges")

# ── Load (or generate) IID edge set ─────────────────────────────────────────
IID_EDGES_PATH = f'{OUT_DIR}/iid_edges.json'
if args.regenerate_iid or not os.path.exists(IID_EDGES_PATH):
    logger.info("Generating IID edges via run_iid...")
    try:
        from iid_comparison import run_iid  # type: ignore
        result = run_iid(data, simulation_path=os.path.abspath(SIM_PATH))
        if isinstance(result, tuple) and len(result) == 2:
            iid_raw = result[0]
        elif hasattr(result, 'edges'):
            iid_raw = list(result.edges())
        elif isinstance(result, dict):
            iid_raw = result.get('edges', [])
        else:
            iid_raw = result
        iid_edges = {(s.lower(), t.lower()) for s, t in iid_raw}
        with open(IID_EDGES_PATH, 'w') as f:
            json.dump([list(e) for e in iid_edges], f)
        logger.info(f"  IID edges saved: {len(iid_edges)} → {IID_EDGES_PATH}")
    except Exception as e:
        logger.error(f"  IID discovery FAILED: {e}")
        logger.error(f"  You can hand-paste IID edges into {IID_EDGES_PATH} as JSON list of [src,tgt] pairs.")
        sys.exit(1)
else:
    with open(IID_EDGES_PATH) as f:
        iid_edges = {(s.lower(), t.lower()) for s, t in json.load(f)}
    logger.info(f"Loaded cached IID edges: {len(iid_edges)} → {IID_EDGES_PATH}")

# ── Build monitors for each method (one-time) ───────────────────────────────
METHOD_EDGES = {
    'PolicyGRID': val_edges,
    'Obs-Only': obs_edges,
    'IID': iid_edges,
}

logger.info("Building factored monitors for each method...")
method_monitors = {}
for name, edges in METHOD_EDGES.items():
    try:
        method_monitors[name] = construct_factored_monitors(edges, data)
        n_obs_edges = sum(1 for s, t in edges
                          if s in OBSERVABLE_VARS and t in OBSERVABLE_VARS)
        logger.info(f"  {name}: {len(edges)} edges total, {n_obs_edges} observable")
    except Exception as e:
        logger.error(f"  {name} monitor build FAILED: {e}")
        method_monitors[name] = None

# ── Collect monitoring per (method, sim seed) — parallel via threads ────────
per_seed_rows = []
results_lock = Lock()
t_start = time.time()


def run_one_seed(seed_i):
    """Generate fresh sim trajectory + run all method monitors. Returns rows."""
    t0 = time.time()
    sim_records = collect_sim_data(smap, duration_steps=args.steps)
    sim_t = (time.time() - t0) / 60

    rows = []
    for name, monitors in method_monitors.items():
        if monitors is None:
            continue
        recs = run_monitoring(monitors, sim_records)
        m = compute_monitoring_metrics(recs)
        rows.append({
            'method': name,
            'sim_seed': seed_i,
            'n_steps': len(sim_records),
            'occ_accuracy': m['occ_accuracy'],
            'win_accuracy': m['win_accuracy'],
            'combined_accuracy': m['combined_accuracy'],
        })
    return seed_i, rows, sim_t


with ThreadPoolExecutor(max_workers=args.workers) as executor:
    future_to_seed = {executor.submit(run_one_seed, i): i
                      for i in range(args.seeds)}
    completed = 0
    for fut in as_completed(future_to_seed):
        seed_i = future_to_seed[fut]
        try:
            seed_i, rows, sim_t = fut.result()
        except Exception as e:
            logger.warning(f"  Seed {seed_i} FAILED: {e}")
            continue
        with results_lock:
            per_seed_rows.extend(rows)
            completed += 1
            pd.DataFrame(per_seed_rows).to_csv(f'{OUT_DIR}/per_seed.csv', index=False)
            elapsed = (time.time() - t_start) / 60
            logger.info(f"  [{completed}/{args.seeds}] sim_seed={seed_i}: "
                        f"sim_collection={sim_t:.1f}min, total_elapsed={elapsed:.1f}min")
            for r in rows:
                logger.info(f"    {r['method']}: occ={r['occ_accuracy']:.1f}% "
                            f"win={r['win_accuracy']:.1f}% "
                            f"combined={r['combined_accuracy']:.1f}%")

per_seed = pd.DataFrame(per_seed_rows)
per_seed.to_csv(f'{OUT_DIR}/per_seed.csv', index=False)
logger.info(f"\nPer-seed CSV saved: {OUT_DIR}/per_seed.csv ({len(per_seed)} rows)")

# ── Paired stats ────────────────────────────────────────────────────────────
def paired_stats(a_vals, b_vals, a_label, b_label):
    """Compute paired t, Wilcoxon, mean Δ ± std, 95% CI on Δ."""
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    diff = a - b
    n = len(diff)
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1) if n > 1 else 0.0
    se = std_diff / np.sqrt(n) if n > 0 else 0.0
    ci_margin = tdist.ppf(0.975, n - 1) * se if n > 1 else 0.0

    # Paired t
    if n >= 2 and std_diff > 0:
        t_stat, p_t = ttest_rel(a, b)
    else:
        t_stat, p_t = float('nan'), float('nan')

    # Wilcoxon (non-parametric)
    try:
        if n >= 2 and not np.allclose(diff, 0):
            w_stat, p_w = wilcoxon(a, b, zero_method='wilcox')
        else:
            w_stat, p_w = float('nan'), float('nan')
    except Exception:
        w_stat, p_w = float('nan'), float('nan')

    # Direction-consistency: how many of n seeds favor A
    wins = int(np.sum(diff > 0))

    cohen_d = mean_diff / std_diff if std_diff > 0 else float('nan')

    lines = [
        f"  {a_label} vs {b_label} (n={n} paired sim seeds)",
        f"    {a_label}:  {a.mean():.2f} ± {a.std(ddof=1):.2f} %",
        f"    {b_label}:  {b.mean():.2f} ± {b.std(ddof=1):.2f} %",
        f"    Mean Δ:    {mean_diff:+.2f} ± {std_diff:.2f} pp",
        f"    95% CI:    [{mean_diff - ci_margin:+.2f}, {mean_diff + ci_margin:+.2f}] pp",
        f"    Direction: {wins}/{n} seeds favor {a_label}",
        f"    Paired t = {t_stat:.2f}, p = {p_t:.4f}, Cohen's d = {cohen_d:.2f}",
        f"    Wilcoxon W = {w_stat}, p = {p_w:.4f}",
    ]
    return '\n'.join(lines)


report_lines = [
    "═" * 72,
    f"PAIRED MONITORING STATS — 4-way regime classification accuracy",
    f"({args.seeds} sim-replicate seeds × {args.steps} steps each, frozen edge sets)",
    "═" * 72,
    "",
]

for metric in ['occ_accuracy', 'win_accuracy', 'combined_accuracy']:
    report_lines.append(f"\n── {metric} ──")
    p = per_seed.pivot_table(index='sim_seed', columns='method', values=metric)
    p = p[['PolicyGRID', 'Obs-Only', 'IID']]  # column order
    report_lines.append(p.round(2).to_string())
    report_lines.append("")
    if 'PolicyGRID' in p.columns and 'Obs-Only' in p.columns:
        report_lines.append(paired_stats(
            p['PolicyGRID'].values, p['Obs-Only'].values,
            'PolicyGRID', 'Obs-Only'))
        report_lines.append("")
    if 'PolicyGRID' in p.columns and 'IID' in p.columns:
        report_lines.append(paired_stats(
            p['PolicyGRID'].values, p['IID'].values,
            'PolicyGRID', 'IID'))
        report_lines.append("")

report = '\n'.join(report_lines)
print('\n' + report)
with open(f'{OUT_DIR}/paired_stats.txt', 'w') as f:
    f.write(report)
logger.info(f"\nReport saved: {OUT_DIR}/paired_stats.txt")
