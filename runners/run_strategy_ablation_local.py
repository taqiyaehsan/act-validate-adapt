"""
Local small-scale strategy ablation.
- max_iterations=3 (fast)
- 3 strategy configs only: mixed (3+7 default), all_deterministic, all_paired
- Single seed=42
- Writes results after EACH config so partial output survives early exit
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

MAX_ITER   = 2     # local: fast. Server script uses 10.
SEEDS      = [42]
OUT_DIR    = 'results/int_strategy_local'
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

# ── configs to run locally ──
# Axis 1: just the default split (3+7) — skip the 4-way split sweep (server only)
# Axis 2: all 3 strategy variants
CONFIGS = [
    {'name': 'split_3_7',       'ablation': 'split_3_7',  'strategy': 'mixed',           'patch': None},
    {'name': 'all_deterministic','ablation': 'split_3_7',  'strategy': 'all_deterministic','patch': 'det_only'},
    {'name': 'all_paired',       'ablation': 'split_3_7',  'strategy': 'all_paired',       'patch': 'paired_only'},
]

results = []
# Load existing partial results if resuming
partial_csv = f'{ABL_DIR}/results_local.csv'
if os.path.exists(partial_csv):
    results = pd.read_csv(partial_csv).to_dict('records')
    done = {(r['strategy'], r['seed']) for r in results}
    logger.info(f"Resuming — {len(results)} results already saved")
else:
    done = set()

t0 = time.time()

for cfg_item in CONFIGS:
    for seed in SEEDS:
        key = (cfg_item['strategy'], seed)
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

        patch = cfg_item['patch']
        if patch == 'det_only':
            orig = pipeline.pipeline.tester.test_edge_iteratively
            def _det_only(edge, intervention, additional_interventions=None, orig=orig):
                return orig(edge, intervention, additional_interventions=None)
            pipeline.pipeline.tester.test_edge_iteratively = _det_only

        elif patch == 'paired_only':
            orig = pipeline.pipeline.tester.test_edge_iteratively
            def _paired_only(edge, intervention, additional_interventions=None, orig=orig):
                if additional_interventions:
                    paired = [i for i in additional_interventions
                              if 'counterfactual' in i.get('strategy_type', '')]
                    return orig(edge, intervention,
                                additional_interventions=paired or None)
                return orig(edge, intervention, additional_interventions=None)
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

        # Write after every config so partial results survive
        pd.DataFrame(results).to_csv(partial_csv, index=False)
        logger.info(f"  Saved to {partial_csv}")

elapsed = (time.time() - t0) / 60
logger.info(f"\nTotal time: {elapsed:.1f} min")
logger.info(f"\nFinal results:")
print(pd.DataFrame(results).to_string(index=False))
