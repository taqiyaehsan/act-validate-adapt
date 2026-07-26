"""
A1: Sample efficiency curves — F1/SHD vs intervention count.
Tracks how quickly PolicyGRID recovers structure as interventions accumulate,
compared to ranked vs random ordering baselines.
Server run: 3 seeds, max_iterations=10.
Saves curves.csv after each seed — resume-safe.
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
OUT_DIR  = 'results/a1_efficiency_server'
os.makedirs(f'{OUT_DIR}/a1_efficiency', exist_ok=True)

cfg  = SIM_CONFIGS['smart_building_rich']
data = pd.read_csv(cfg['data_path'])
cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
data = data[cols]

with open('ground_truth_graphs.json') as f:
    gt_raw = json.load(f)
gt_set   = {(s.lower(), t.lower()) for s, t in gt_raw['smart_building_rich']['edges']}
sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None

# Resume support
curves_csv = f'{OUT_DIR}/a1_efficiency/curves.csv'
if os.path.exists(curves_csv):
    existing  = pd.read_csv(curves_csv)
    done_seeds = set(existing['seed'].unique())
    all_curves = existing.to_dict('records')
    logger.info(f"Resuming — seeds already done: {done_seeds}")
else:
    all_curves = []
    done_seeds = set()

t0 = time.time()

for ordering in ['ranked', 'random']:
    for seed in SEEDS:
        key = (ordering, seed)
        if key in {(r.get('ordering', 'ranked'), r['seed'])
                   for r in all_curves if r.get('ordering', 'ranked') == ordering}:
            logger.info(f"  Skipping ordering={ordering} seed={seed} (already done)")
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
            # Patch AFTER topological ranking — _get_edges_to_test shuffle is
            # overridden by _rank_edges_topologically on line 451 of run_with_cwm.
            orig_rank = pipeline._rank_edges_topologically
            def _random_rank(edges, orig=orig_rank, s=seed):
                result = list(orig(edges))
                random.seed(s)
                random.shuffle(result)
                return result
            pipeline._rank_edges_topologically = _random_rank

        # Hook tracking into _test_edges_with_cwm_updates (called each iteration).
        # NOTE: cannot hook validated_edges directly — run_with_cwm() resets it
        # to a plain set() internally before the iteration loop starts.
        f1_curve = [{'ordering': ordering, 'seed': seed,
                     'intervention': 0, 'f1': 0.0,
                     'shd': len(gt_set), 'n_validated': 0}]
        _counter   = [0]
        _tracking  = [set()]   # mirrors validated_edges for F1 computation

        orig_test  = pipeline._test_edges_with_cwm_updates
        def _tracking_test(edges_to_test, orig=orig_test):
            result = orig(edges_to_test)
            if result:
                for edge in result:
                    _tracking[0].add(edge)
                    _counter[0] += 1
                    val_set = {(s.lower(), t.lower()) for s, t in _tracking[0]}
                    m = graph_metrics(val_set, gt_set)
                    f1_curve.append({
                        'ordering': ordering, 'seed': seed,
                        'intervention': _counter[0],
                        'f1': m['f1'], 'shd': m['shd'],
                        'n_validated': len(_tracking[0]),
                    })
            return result
        pipeline._test_edges_with_cwm_updates = _tracking_test

        pipeline.run_with_cwm()
        all_curves.extend(f1_curve)

        logger.info(f"  Final: F1={f1_curve[-1]['f1']:.3f}, "
                    f"{f1_curve[-1]['n_validated']} validated, "
                    f"{_counter[0]} steps")

        # Save after each (ordering, seed)
        pd.DataFrame(all_curves).to_csv(curves_csv, index=False)
        logger.info(f"  Saved to {curves_csv}")

elapsed = (time.time() - t0) / 60
logger.info(f"\nTotal time: {elapsed:.1f} min")

# Summary table: F1 at key budgets
df = pd.DataFrame(all_curves)
logger.info(f"\nSample efficiency (mean F1 across {len(SEEDS)} seeds):")
logger.info(f"  {'Budget':>6}  {'ranked':>8}  {'random':>8}")
for budget in [5, 10, 15, 20, 25, 30]:
    row = {}
    for o in ['ranked', 'random']:
        f1s = []
        for s in SEEDS:
            sdf = df[(df['ordering'] == o) & (df['seed'] == s)]
            at  = sdf[sdf['intervention'] <= budget]
            if len(at):
                f1s.append(at.iloc[-1]['f1'])
        row[o] = f"{np.mean(f1s):.3f}" if f1s else "  —  "
    logger.info(f"  {budget:>6}  {row['ranked']:>8}  {row['random']:>8}")
