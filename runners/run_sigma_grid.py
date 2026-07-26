"""S3.3 — Monitoring sigma/eta sensitivity grid (sbr only).

The paper sweeps sigma and the forget factor (eta) over a few values; the
deployed point is sigma=1.5, eta=0.15. This script sweeps a grid centered on the
deployed point and reports 4-way regime accuracy at each cell, so we can show the
monitor isn't brittle to these choices.

Method: build the factored monitors ONCE on the frozen validated graph, collect
ONE fresh sim trajectory, then re-run the Bayesian monitoring loop at every
(sigma, eta) cell on that same trajectory. Reusing one trajectory isolates the
hyperparameter effect (no sim-noise confound between cells). Seed robustness at
the deployed point is already covered by run_paired_monitoring.py.

Output: results/sigma_grid/grid.csv  (+ printed table, deployed + best cells marked)
"""
import sys, os, json, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from neurips_final_experiments import (
    construct_factored_monitors, run_monitoring, compute_monitoring_metrics,
    collect_sim_data, get_scaling_map,
)

DATA_PATH = 'data_regen/smart_building_rich_processed.csv'
SCALING_PATH = 'data_regen/smart_building_rich_processed_scaling.csv'
EDGE_SETS = 'results/closedloop_server/edge_sets.json'
OUT_DIR = 'results/sigma_grid'
DEPLOYED = (1.5, 0.15)

parser = argparse.ArgumentParser()
parser.add_argument('--sigmas', type=float, nargs='+',
                    default=[1.0, 1.5, 2.0, 2.5, 3.0])
parser.add_argument('--etas', type=float, nargs='+',
                    default=[0.05, 0.10, 0.15, 0.20, 0.25])
parser.add_argument('--steps', type=int, default=5760)
parser.add_argument('--edge-set', default='validated')
parser.add_argument('--smoke', action='store_true')
args = parser.parse_args()

if args.smoke:
    args.sigmas = [1.5, 2.0]
    args.etas = [0.15, 0.25]
    args.steps = 240

data = pd.read_csv(DATA_PATH)
scaling = pd.read_csv(SCALING_PATH)
smap = get_scaling_map(scaling)
edge_sets = json.load(open(EDGE_SETS))
edges = {(s.lower(), t.lower()) for s, t in edge_sets[args.edge_set]}

print(f"\nSigma/eta grid — edge_set={args.edge_set} ({len(edges)} edges), "
      f"{len(args.sigmas)}×{len(args.etas)} cells, {args.steps} steps/cell")
print("Building monitors + collecting one sim trajectory...")
monitors = construct_factored_monitors(edges, data)
sim_records = collect_sim_data(smap, duration_steps=args.steps)
print(f"  trajectory: {len(sim_records)} steps\n" + "=" * 64)

rows = []
print(f"{'sigma':>6} {'eta':>6} {'occ%':>7} {'win%':>7} {'comb%':>7}")
for s in args.sigmas:
    for e in args.etas:
        recs = run_monitoring(monitors, sim_records, sigma=s, forget_factor=e)
        m = compute_monitoring_metrics(recs)
        is_dep = abs(s - DEPLOYED[0]) < 1e-9 and abs(e - DEPLOYED[1]) < 1e-9
        rows.append({
            'sigma': s, 'eta': e,
            'occ_accuracy': m['occ_accuracy'],
            'win_accuracy': m['win_accuracy'],
            'combined_accuracy': m['combined_accuracy'],
            'deployed': is_dep,
        })
        tag = '  <-- deployed' if is_dep else ''
        print(f"{s:6.2f} {e:6.2f} {m['occ_accuracy']:7.1f} "
              f"{m['win_accuracy']:7.1f} {m['combined_accuracy']:7.1f}{tag}")

os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, 'grid.csv')
df = pd.DataFrame(rows)
df.to_csv(out, index=False)
print("=" * 64)

best = df.loc[df['combined_accuracy'].idxmax()]
dep = df[df['deployed']]
print(f"best cell:     sigma={best['sigma']}, eta={best['eta']} "
      f"→ combined {best['combined_accuracy']:.1f}%")
if len(dep):
    d = dep.iloc[0]
    print(f"deployed cell: sigma={d['sigma']}, eta={d['eta']} "
          f"→ combined {d['combined_accuracy']:.1f}%")
spread = df['combined_accuracy'].max() - df['combined_accuracy'].min()
print(f"combined-accuracy spread across grid: {spread:.1f} pp "
      f"(smaller = more robust)")
print(f"\nWrote {out}")
