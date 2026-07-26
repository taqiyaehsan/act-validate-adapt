"""
Q4 Corrected: Intervention Ordering with per-iteration edge budget.

The original Q4 tested ALL candidate edges per iteration, making ordering irrelevant.
This version caps edges tested per iteration so ordering actually determines
WHICH edges get their K=10 interventions first.

Budget: EDGES_PER_ITER edges per iteration × 10 interventions each.
Conditions: ranked (disagreement-first) vs random.
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

# Disable SAM (hangs on server due to PyTorch multiprocessing)
import src.pipeline as _pl
class _NoSAM:
    def __init__(self, **kwargs): pass
    def generate(self, data): return []
_pl.SAMGenerator = _NoSAM
logger.info("SAM disabled")

EDGES_PER_ITER = 5  # Cap: only test top-5 edges per iteration
MAX_ITER = 10
SEEDS = [42, 123, 456]
OUT_DIR = 'results/q4_ordering_corrected'
os.makedirs(OUT_DIR, exist_ok=True)

cfg = SIM_CONFIGS['smart_building_rich']
data = pd.read_csv(cfg['data_path'])
cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
data = data[cols]

with open('ground_truth_graphs.json') as f:
    gt_raw = json.load(f)
gt_set = {(s.lower(), t.lower()) for s, t in gt_raw['smart_building_rich']['edges']}

sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None

# Resume support
partial_csv = f'{OUT_DIR}/results.csv'
if os.path.exists(partial_csv):
    results = pd.read_csv(partial_csv).to_dict('records')
    done = {(r['ordering'], r['seed']) for r in results}
    logger.info(f"Resuming — {len(results)} results already saved")
else:
    results = []
    done = set()

t0 = time.time()

for ordering in ['ranked', 'random']:
    for seed in SEEDS:
        if (ordering, seed) in done:
            logger.info(f"  Skipping {ordering} seed={seed} (already done)")
            continue

        logger.info(f"\n── ordering={ordering}, seed={seed}, budget={EDGES_PER_ITER}/iter ──")
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

        # Patch: Apply edge budget cap + ordering
        orig_rank = pipeline._rank_edges_topologically

        def _budget_rank(edges, _orig=orig_rank, _ordering=ordering, _seed=seed):
            ranked = list(_orig(edges))
            if _ordering == 'random':
                random.shuffle(ranked)  # No reseed — different shuffle each call
            if len(ranked) > EDGES_PER_ITER:
                logger.info(f"  Budget cap: {len(ranked)} candidates -> testing top {EDGES_PER_ITER}")
            return ranked[:EDGES_PER_ITER]

        pipeline._rank_edges_topologically = _budget_rank

        # Track per-iteration metrics for efficiency curves
        iter_metrics = []
        cumulative_edges_tested = [0]
        orig_test = pipeline._test_edges_with_cwm_updates

        def _tracking_test(edges, _orig=orig_test):
            result = _orig(edges)
            cumulative_edges_tested[0] += len(edges)
            val_set = {(s.lower(), t.lower()) for s, t in pipeline.pipeline.validated_edges}
            tp = len(val_set & gt_set)
            fp = len(val_set - gt_set)
            fn = len(gt_set - val_set)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            iter_metrics.append({
                'iteration': len(iter_metrics) + 1,
                'edges_tested_this_iter': len(edges),
                'cumulative_edges_tested': cumulative_edges_tested[0],
                'cumulative_interventions': cumulative_edges_tested[0] * 10,
                'n_validated': len(val_set),
                'tp': tp, 'fp': fp, 'fn': fn,
                'precision': round(p, 3),
                'recall': round(r, 3),
                'f1': round(f1, 3),
            })
            logger.info(f"    Iter {len(iter_metrics)}: tested={len(edges)}, "
                        f"validated={len(val_set)}, F1={f1:.3f}, P={p:.3f}, R={r:.3f}")
            return result

        pipeline._test_edges_with_cwm_updates = _tracking_test

        pipeline.run_with_cwm()
        validated = pipeline.pipeline.validated_edges
        val_set = {(s.lower(), t.lower()) for s, t in validated}
        m = graph_metrics(val_set, gt_set)

        tp = len(val_set & gt_set)
        fp = len(val_set - gt_set)
        fn = len(gt_set - val_set)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0

        row = {
            'ordering': ordering, 'seed': seed,
            'f1': round(m['f1'], 3), 'shd': m['shd'],
            'precision': round(p, 3), 'recall': round(r, 3),
            'n_validated': len(val_set),
            'tp': tp, 'fp': fp, 'fn': fn,
            'total_edges_tested': cumulative_edges_tested[0],
            'total_interventions': cumulative_edges_tested[0] * 10,
        }
        results.append(row)
        logger.info(f"  FINAL: F1={m['f1']:.3f}, P={p:.3f}, R={r:.3f}, "
                    f"SHD={m['shd']}, edges_tested={cumulative_edges_tested[0]}, "
                    f"validated={len(val_set)}")

        # Save per-iteration curve
        curve_path = f'{OUT_DIR}/{ordering}_seed{seed}_curve.csv'
        pd.DataFrame(iter_metrics).to_csv(curve_path, index=False)
        logger.info(f"  Curve saved to {curve_path}")

        # Save after every (ordering, seed)
        pd.DataFrame(results).to_csv(partial_csv, index=False)

elapsed = (time.time() - t0) / 60
logger.info(f"\nTotal time: {elapsed:.1f} min")

df = pd.DataFrame(results)
logger.info(f"\n{'='*70}")
logger.info(f"Q4 Corrected Results (budget={EDGES_PER_ITER} edges/iter)")
logger.info(f"{'='*70}")
for ordering in ['ranked', 'random']:
    odf = df[df['ordering'] == ordering]
    if len(odf):
        logger.info(f"  {ordering}: F1={odf['f1'].mean():.3f}+/-{odf['f1'].std():.3f}, "
                    f"P={odf['precision'].mean():.3f}, R={odf['recall'].mean():.3f}, "
                    f"SHD={odf['shd'].mean():.1f}, "
                    f"edges_tested={odf['total_edges_tested'].mean():.0f}")
print(df.to_string(index=False))
