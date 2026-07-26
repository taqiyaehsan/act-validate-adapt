#!/usr/bin/env python3
"""Run 3-seed PolicyGRID discovery on open_window, save consensus edges."""
import sys, os, json, time, warnings, random
import numpy as np, pandas as pd
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

from src.pipeline_cwm import create_cwm_pipeline

API_KEY = 'YOUR_OPENAI_API_KEY'
SIM_PATH = os.path.abspath('js/open_window.js')
DATA_PATH = 'data_regen/open_window_varied_processed.csv'
GT_KEY = 'open_window'
SEEDS = [42, 123, 456]

data = pd.read_csv(DATA_PATH)
with open('ground_truth_graphs.json') as f:
    gt_raw = json.load(f)[GT_KEY]['edges']
gt_set = {(s.lower(), t.lower()) for s, t in gt_raw}

logger.info(f'Data: {data.shape}, GT: {len(gt_set)} edges, Seeds: {SEEDS}')

all_validated = []
all_obs = []

for seed in SEEDS:
    logger.info(f'\n{"="*60}')
    logger.info(f'  SEED {seed}')
    logger.info(f'{"="*60}')

    random.seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    pipeline = create_cwm_pipeline(
        csv_data=data, api_key=API_KEY, smart_room_path=SIM_PATH,
        dataset_type='open_window', max_iterations=3,
        actuator_vars=['temperature', 'humidity', 'airquality'],
        non_intervenable_vars=['windowopen', 'outdoortemperature', 'pmv',
                               'energyconsumption', 'satisfaction', 'elapsedtime'],
    )
    final_dag, final_metrics = pipeline.run_with_cwm()
    elapsed = time.time() - t0

    validated = set(pipeline.pipeline.validated_edges)
    val_set = {(s.lower(), t.lower()) for s, t in validated}
    tp = len(val_set & gt_set); fp = len(val_set - gt_set); fn = len(gt_set - val_set)
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1); f1 = 2*p*r/max(p+r,1e-10)

    logger.info(f'  Seed {seed}: {len(validated)} edges, F1={f1:.3f}, Time={elapsed:.0f}s')
    all_validated.append(validated)

# Consensus: majority vote (appear in ≥2 of 3 seeds)
from collections import Counter
edge_counts = Counter()
for edges in all_validated:
    for e in edges:
        edge_counts[(e[0].lower(), e[1].lower())] += 1

consensus = {e for e, count in edge_counts.items() if count >= 2}
val_set = consensus
tp = len(val_set & gt_set); fp = len(val_set - gt_set); fn = len(gt_set - val_set)
p = tp/max(tp+fp,1); r = tp/max(tp+fn,1); f1 = 2*p*r/max(p+r,1e-10); shd = fp+fn

logger.info(f'\n{"="*60}')
logger.info(f'  CONSENSUS (≥2/3 seeds): {len(consensus)} edges')
logger.info(f'  F1={f1:.3f}, P={p:.3f}, R={r:.3f}, SHD={shd}')
logger.info(f'{"="*60}')

for s, t in sorted(consensus):
    mark = '✓' if (s, t) in gt_set else '✗'
    logger.info(f'  {mark} {s} → {t}')

# Save
out = {
    'seeds': SEEDS,
    'per_seed_validated': [[list(e) for e in v] for v in all_validated],
    'consensus_validated': [list(e) for e in consensus],
    'f1': f1, 'precision': p, 'recall': r, 'shd': shd,
    'n_consensus': len(consensus),
}
with open('results/discovery_open_window_3seed.json', 'w') as f:
    json.dump(out, f, indent=2)
logger.info(f'Saved to results/discovery_open_window_3seed.json')
