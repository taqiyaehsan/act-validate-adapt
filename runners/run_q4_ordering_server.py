"""
Q4: Intervention Ordering — confidence-ranked vs random, fixed budget.
Server run: 3 seeds, max_iterations=10.
Saves after each (ordering, seed) pair — resume-safe.
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

from run_full_pipeline import SIM_CONFIGS, graph_metrics, API_KEY
from src.pipeline_cwm import create_cwm_pipeline

# ── Disable SAM on server (hangs due to PyTorch multiprocessing on Linux) ──
import src.pipeline as _pl
class _NoSAM:
    def __init__(self, **kwargs): pass
    def generate(self, data): return []
_pl.SAMGenerator = _NoSAM
logger.info("SAM disabled (no-op stub installed)")

MAX_ITER = 10
SEEDS    = [42, 123, 456]
OUT_DIR  = 'results/q4_ordering_server'
os.makedirs(f'{OUT_DIR}/m3b_ordering', exist_ok=True)

cfg      = SIM_CONFIGS['smart_building_rich']
data     = pd.read_csv(cfg['data_path'])
cols     = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
data     = data[cols]

with open('ground_truth_graphs.json') as f:
    gt_raw = json.load(f)
gt_set = {(s.lower(), t.lower()) for s, t in gt_raw['smart_building_rich']['edges']}

sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None

# Resume support
partial_csv = f'{OUT_DIR}/m3b_ordering/results.csv'
if os.path.exists(partial_csv):
    results = pd.read_csv(partial_csv).to_dict('records')
    done    = {(r['ordering'], r['seed']) for r in results}
    logger.info(f"Resuming — {len(results)} results already saved")
else:
    results = []
    done    = set()

t0 = time.time()

for ordering in ['ranked', 'random']:
    for seed in SEEDS:
        if (ordering, seed) in done:
            logger.info(f"  Skipping {ordering} seed={seed} (already done)")
            continue

        logger.info(f"\n── ordering={ordering}, seed={seed} ──")
        random.seed(seed); np.random.seed(seed)

        pipeline = create_cwm_pipeline(
            csv_data=data,
            api_key=API_KEY,
            smart_room_path=sim_path,
            dataset_type=cfg['dataset_type'],
            max_iterations=MAX_ITER,
            actuator_vars=cfg['actuators'],
            non_intervenable_vars=cfg['non_intervenable'],
        )

        if ordering == 'random':
            # Patch: shuffle AFTER topological ranking (line 451 in run_with_cwm
            # calls _rank_edges_topologically which overrides any earlier shuffle).
            orig_rank = pipeline._rank_edges_topologically
            def _random_rank(edges, orig=orig_rank, s=seed):
                result = list(orig(edges))   # run normal ranking first
                random.seed(s)
                random.shuffle(result)       # then discard the order
                return result
            pipeline._rank_edges_topologically = _random_rank

        pipeline.run_with_cwm()
        validated = pipeline.pipeline.validated_edges
        val_set   = {(s.lower(), t.lower()) for s, t in validated}
        m         = graph_metrics(val_set, gt_set)

        n_intv = getattr(pipeline.cwm, 'intervention_count', 0) \
                 if hasattr(pipeline, 'cwm') and pipeline.cwm else 0

        row = {
            'ordering': ordering, 'seed': seed,
            'f1': round(m['f1'], 3), 'shd': m['shd'],
            'n_validated': len(val_set),
            'n_interventions': n_intv,
        }
        results.append(row)
        logger.info(f"  → F1={m['f1']:.3f}, SHD={m['shd']}, "
                    f"edges={len(val_set)}, interventions={n_intv}")

        pd.DataFrame(results).to_csv(partial_csv, index=False)
        logger.info(f"  Saved to {partial_csv}")

elapsed = (time.time() - t0) / 60
logger.info(f"\nTotal time: {elapsed:.1f} min")

df = pd.DataFrame(results)
for ordering in ['ranked', 'random']:
    odf = df[df['ordering'] == ordering]
    if len(odf):
        logger.info(f"  {ordering}: F1={odf['f1'].mean():.3f}±{odf['f1'].std():.3f}, "
                    f"SHD={odf['shd'].mean():.1f}±{odf['shd'].std():.1f}")
print(df.to_string(index=False))
