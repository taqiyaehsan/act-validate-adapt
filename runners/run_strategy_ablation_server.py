"""
Server full strategy ablation.
- max_iterations=10
- All 6 configs: 4 split ratios (mixed strategy) + 2 strategy variants (3+7 split)
- 3 seeds: 42, 123, 456
- Writes results after EACH config so partial output survives
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

MAX_ITER   = 10
SEEDS      = [42, 123, 456]
OUT_DIR    = 'results/int_strategy_server'
ABL_DIR    = f'{OUT_DIR}/int_strategy_ablation'
os.makedirs(ABL_DIR, exist_ok=True)

cfg = SIM_CONFIGS['smart_building_rich']
data = pd.read_csv(cfg['data_path'])
cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
data = data[cols]

with open('ground_truth_graphs.json') as f:
    gt_raw = json.load(f)
gt_set = {(s.lower(), t.lower()) for s, t in gt_raw['smart_building_rich']['edges']}

sim_path = os.path.abspath(cfg['sim_path']) if cfg['sim_path'] else None

# ── All 6 configs ──
CONFIGS = [
    # Axis 1: split ratio sweep (mixed strategy, 3+7 is default)
    {'name': 'split_1_9',        'ablation': 'split_1_9',  'strategy': 'mixed',            'patch': None,         'initial': 1,  'extended': 9},
    {'name': 'split_3_7',        'ablation': 'split_3_7',  'strategy': 'mixed',            'patch': None,         'initial': 3,  'extended': 7},
    {'name': 'split_5_5',        'ablation': 'split_5_5',  'strategy': 'mixed',            'patch': None,         'initial': 5,  'extended': 5},
    {'name': 'split_10_0',       'ablation': 'split_10_0', 'strategy': 'mixed',            'patch': None,         'initial': 10, 'extended': 0},
    # Axis 2: strategy diversity (3+7 split)
    {'name': 'all_deterministic','ablation': 'split_3_7',  'strategy': 'all_deterministic','patch': 'det_only',   'initial': 3,  'extended': 7},
    {'name': 'all_paired',       'ablation': 'split_3_7',  'strategy': 'all_paired',       'patch': 'paired_only','initial': 3,  'extended': 7},
]

# Load existing partial results if resuming
partial_csv = f'{ABL_DIR}/results_server.csv'
if os.path.exists(partial_csv):
    results = pd.read_csv(partial_csv).to_dict('records')
    done = {(r['strategy'], r['ablation'], r['seed']) for r in results}
    logger.info(f"Resuming — {len(results)} results already saved")
else:
    results = []
    done = set()

t0 = time.time()

for cfg_item in CONFIGS:
    for seed in SEEDS:
        key = (cfg_item['strategy'], cfg_item['ablation'], seed)
        if key in done:
            logger.info(f"  Skipping {cfg_item['name']} seed={seed} (already done)")
            continue

        logger.info(f"\n── {cfg_item['name']}, seed={seed}, max_iter={MAX_ITER} ──")
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

        # Override split counts
        pipeline.pipeline.tester.initial_test_count  = cfg_item['initial']
        pipeline.pipeline.tester.extended_test_count = cfg_item['extended']

        patch = cfg_item['patch']
        # Check whether tester supports additional_interventions (older server builds may not)
        import inspect as _inspect
        _tei_params = _inspect.signature(
            pipeline.pipeline.tester.test_edge_iteratively).parameters
        _has_addl = 'additional_interventions' in _tei_params

        if patch == 'det_only':
            orig = pipeline.pipeline.tester.test_edge_iteratively
            def _det_only(edge, intervention, additional_interventions=None, orig=orig):
                # Force deterministic-only: suppress all additional interventions
                if _has_addl:
                    return orig(edge, intervention, additional_interventions=None)
                return orig(edge, intervention)
            pipeline.pipeline.tester.test_edge_iteratively = _det_only

        elif patch == 'paired_only':
            orig = pipeline.pipeline.tester.test_edge_iteratively
            def _paired_only(edge, intervention, additional_interventions=None, orig=orig):
                # Force paired counterfactual only: filter to counterfactual strategies
                if _has_addl and additional_interventions:
                    paired = [i for i in additional_interventions
                              if 'counterfactual' in i.get('strategy_type', '')]
                    return orig(edge, intervention,
                                additional_interventions=paired or None)
                return orig(edge, intervention) if not _has_addl \
                    else orig(edge, intervention, additional_interventions=None)
            pipeline.pipeline.tester.test_edge_iteratively = _paired_only

        pipeline.run_with_cwm()
        val = {(s.lower(), t.lower()) for s, t in pipeline.pipeline.validated_edges}
        m   = graph_metrics(val, gt_set)

        row = {
            'ablation':    cfg_item['ablation'],
            'strategy':    cfg_item['strategy'],
            'seed':        seed,
            'max_iter':    MAX_ITER,
            'f1':          round(m['f1'], 3),
            'shd':         m['shd'],
            'n_validated': len(val),
        }
        results.append(row)
        logger.info(f"  → F1={m['f1']:.3f}, SHD={m['shd']}, edges={len(val)}")

        # Write after every (config, seed) pair — resume-safe
        pd.DataFrame(results).to_csv(partial_csv, index=False)
        logger.info(f"  Saved to {partial_csv}")

elapsed = (time.time() - t0) / 60
logger.info(f"\nTotal time: {elapsed:.1f} min")

df = pd.DataFrame(results)
summary = df.groupby(['ablation', 'strategy'])[['f1', 'shd', 'n_validated']].agg(
    ['mean', 'std']).round(3)
logger.info(f"\nSummary (mean±std across seeds):\n{summary.to_string()}")
print(df.to_string(index=False))
