"""
M3A: Graph Attribution — validated / obs-only / random / empty graphs, same policy engine.
Server run: 3 seeds, uses existing 5-seed consensus from server_runs/.
Saves after each (condition, seed) pair — resume-safe.
"""
import sys, os, json, warnings, random, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

from run_full_pipeline import (SIM_CONFIGS, graph_metrics, API_KEY,
                                load_scaling, run_policy_for_seed,
                                COMFORT_TARGETS)
from src.pipeline_cwm import create_cwm_pipeline

# ── Disable SAM on server (hangs due to PyTorch multiprocessing on Linux) ──
import src.pipeline as _pl
class _NoSAM:
    def __init__(self, **kwargs): pass
    def generate(self, data): return []
_pl.SAMGenerator = _NoSAM
logger.info("SAM disabled (no-op stub installed)")

SEEDS   = [42, 123, 456]
OUT_DIR = 'results/m3a_attribution_server'
os.makedirs(f'{OUT_DIR}/m3a_attribution', exist_ok=True)

cfg  = SIM_CONFIGS['smart_building_rich']
data = pd.read_csv(cfg['data_path'])
cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
data = data[cols]
cm   = {v.lower(): v for v in data.columns}

# Scaling map
smap = load_scaling(cfg['scaling_path'])

# ── Load validated (consensus) edges ──
# Check candidate paths in order; fall back to computing from per-seed results.
CANDIDATE_PATHS = [
    'server_runs/full_pipeline/smart_building_rich/consensus_analysis.json',
    'results/full_pipeline/smart_building_rich/consensus_analysis.json',
    'results/smart_building_rich/consensus_analysis.json',
]

consensus_path_found = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)

if consensus_path_found:
    with open(consensus_path_found) as f:
        ca = json.load(f)
    consensus_lower = {(s, t) for s, t in ca['consensus_edges']}
    logger.info(f"Loaded consensus from {consensus_path_found}")
else:
    # Compute consensus on the fly from per-seed edges.json files
    logger.info("No consensus file found — computing from per-seed edges.json files...")
    from collections import Counter
    seed_dirs = []
    for base in ['results/full_pipeline/smart_building_rich',
                 'results/smart_building_rich']:
        if os.path.isdir(base):
            seed_dirs = [os.path.join(base, d)
                         for d in os.listdir(base) if d.startswith('seed_')]
            if seed_dirs:
                break
    if not seed_dirs:
        raise FileNotFoundError(
            "Could not find per-seed results. Run the full pipeline first:\n"
            "  python run_full_pipeline.py --sim smart_building_rich --stages discovery"
        )
    edge_counts = Counter()
    for sd in seed_dirs:
        epath = os.path.join(sd, 'edges.json')
        if os.path.exists(epath):
            with open(epath) as f:
                edges = json.load(f)
            for e in edges:
                edge_counts[tuple(e)] += 1
    majority = len(seed_dirs) // 2 + 1
    consensus_lower = {e for e, c in edge_counts.items() if c >= majority}
    logger.info(f"Computed consensus from {len(seed_dirs)} seeds: "
                f"{len(consensus_lower)} edges (majority={majority}/{len(seed_dirs)})")

val_edges = {(cm.get(s, s), cm.get(t, t)) for s, t in consensus_lower}
logger.info(f"Validated edge set: {len(val_edges)} edges")

# ── Obs-only edges: run 4 discovery methods with max_iterations=0 ──
sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None

logger.info("Generating obs-only edges (max_iterations=0)...")
random.seed(42); np.random.seed(42)
try:
    obs_pipeline = create_cwm_pipeline(
        csv_data=data, api_key=API_KEY, smart_room_path=sim_path,
        dataset_type=cfg['dataset_type'], max_iterations=0,
        actuator_vars=cfg['actuators'],
        non_intervenable_vars=cfg['non_intervenable'],
    )
    method_dags = obs_pipeline.pipeline._generate_hypotheses()
    obs_lower   = set()
    for dag in method_dags.values():
        for e in obs_pipeline.pipeline._extract_edges(dag):
            obs_lower.add(tuple(e) if isinstance(e, list) else e)
    obs_edges = {(cm.get(s, s), cm.get(t, t)) for s, t in obs_lower}
    logger.info(f"Obs-only: {len(obs_edges)} edges")
except Exception as ex:
    logger.warning(f"Obs-only generation failed ({ex}), using validated as fallback")
    obs_edges = val_edges

# ── Random graph: same number of edges as validated ──
all_vars   = list(data.columns)
n_val      = len(val_edges)
random.seed(42)
rand_edges = set()
while len(rand_edges) < n_val:
    s, t = random.choice(all_vars), random.choice(all_vars)
    if s.lower() != t.lower():
        rand_edges.add((s, t))
logger.info(f"Random graph: {len(rand_edges)} edges")

# ── 4 conditions ──
CONDITIONS = {
    'Validated': val_edges,
    'Obs-Only':  obs_edges,
    'Random':    rand_edges,
    'Empty':     set(),
}

# Resume support
partial_csv = f'{OUT_DIR}/m3a_attribution/results.csv'
if os.path.exists(partial_csv):
    all_results_list = [pd.read_csv(partial_csv)]
    done = {(r['graph_condition'], r['seed'])
            for r in pd.read_csv(partial_csv).to_dict('records')}
    logger.info(f"Resuming — {sum(len(r) for r in all_results_list)} rows already saved")
else:
    all_results_list = []
    done = set()

t0 = time.time()

for cond_name, edges in CONDITIONS.items():
    logger.info(f"\n── Condition: {cond_name} ({len(edges)} edges) ──")
    for seed in SEEDS:
        if (cond_name, seed) in done:
            logger.info(f"  Skipping seed={seed} (already done)")
            continue

        logger.info(f"  seed={seed}")
        random.seed(seed); np.random.seed(seed)

        df = run_policy_for_seed(cfg, data, edges, smap)
        if df is not None:
            df['graph_condition'] = cond_name
            df['seed']            = seed
            all_results_list.append(df)

            # Save after every (condition, seed)
            combined = pd.concat(all_results_list, ignore_index=True)
            combined.to_csv(partial_csv, index=False)
            logger.info(f"  Saved {len(combined)} rows to {partial_csv}")

            # Quick summary for this seed
            for ct in [0.5, 1.0, 2.0]:
                pg = df[(df['method'] == 'PolicyGRID') & (df['comfort_target'] == ct)]
                if len(pg):
                    logger.info(f"    ε={ct}: MO={pg['MO'].mean():.3f}, "
                                f"kWh={pg['kWh'].mean():.3f}, "
                                f"Sat={pg['satisfaction'].mean():.1f}")

elapsed = (time.time() - t0) / 60
logger.info(f"\nTotal time: {elapsed:.1f} min")

if all_results_list:
    result_df = pd.concat(all_results_list, ignore_index=True)

    # Final summary table
    logger.info(f"\nM3(A) Summary — MO score (mean across seeds):")
    logger.info(f"  {'Condition':<12} {'ε=0.5':>8} {'ε=0.8':>8} {'ε=1.0':>8} {'ε=1.5':>8} {'ε=2.0':>8}  HV")
    for cond in CONDITIONS:
        cdf = result_df[result_df['graph_condition'] == cond]
        parts = []
        mo_vals = []
        for ct in COMFORT_TARGETS:
            pg = cdf[(cdf['method'] == 'PolicyGRID') & (cdf['comfort_target'] == ct)]
            if len(pg):
                parts.append(f"{pg['MO'].mean():.3f}")
                mo_vals.append(pg['MO'].mean())
            else:
                parts.append('  —  ')
        hv = result_df[result_df['graph_condition'] == cond].groupby('seed').apply(
            lambda x: x[x['method'] == 'PolicyGRID']['MO'].max()
        ).mean() if len(cdf) else 0
        logger.info(f"  {cond:<12} {'  '.join(parts):>45}  {hv:.3f}")

    # Save final summary
    summary_path = f'{OUT_DIR}/m3a_attribution/summary.csv'
    summary_rows = []
    for cond in CONDITIONS:
        cdf = result_df[result_df['graph_condition'] == cond]
        for ct in COMFORT_TARGETS:
            pg = cdf[(cdf['method'] == 'PolicyGRID') & (cdf['comfort_target'] == ct)]
            if len(pg):
                summary_rows.append({
                    'condition': cond, 'comfort_target': ct,
                    'MO_mean': round(pg['MO'].mean(), 4),
                    'MO_std':  round(pg['MO'].std(), 4),
                    'kWh_mean': round(pg['kWh'].mean(), 3),
                    'sat_mean': round(pg['satisfaction'].mean(), 2),
                })
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved to {summary_path}")
