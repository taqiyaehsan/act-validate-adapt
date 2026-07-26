#!/usr/bin/env python3
"""
Re-run Phase 2.4 rescue on saved discovery results.
Uses the already-validated edges + baseline data. No hardware needed.

Usage:
    python rerun_rescue.py
"""
import os, sys, json, logging
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from run_policygrid_discovery import (
    run_hypothesis_generation, rescue_edges, GRAPH_VARS,
    PROPOSED_GT, RESCUE_TAU
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = 'results/discovery/seed_42'
BASELINE_CSV = 'data/baseline_data.csv'

# Load baseline
baseline = pd.read_csv(BASELINE_CSV)
logger.info(f"Baseline: {len(baseline)} rows")

# Load saved validated edges
with open(os.path.join(RESULTS_DIR, 'validated_edges.json')) as f:
    validated = {tuple(e) for e in json.load(f)}
logger.info(f"Saved validated edges: {len(validated)}")
for e in sorted(validated):
    logger.info(f"  {e[0]} → {e[1]}")

# Re-run hypothesis gen on baseline (no augmented data — clean)
candidates, scores, methods = run_hypothesis_generation(baseline)

# Re-run rescue with the fix (dropna in rescue_edges)
evidence = {}
# Load existing evidence
ev_path = os.path.join(RESULTS_DIR, 'edge_evidence.json')
if os.path.exists(ev_path):
    with open(ev_path) as f:
        evidence = json.load(f)

rescued = rescue_edges(candidates, baseline, validated, methods, evidence)

all_edges = validated | rescued
logger.info(f"\nFinal: {len(validated)} validated + {len(rescued)} rescued = {len(all_edges)} total")

# F1
gt_norm = {(s.lower(), t.lower()) for s, t in PROPOSED_GT}
val_norm = {(s.lower(), t.lower()) for s, t in all_edges}
tp = len(val_norm & gt_norm)
fp = len(val_norm - gt_norm)
fn = len(gt_norm - val_norm)
p = tp / max(tp + fp, 1)
r = tp / max(tp + fn, 1)
f1 = 2 * p * r / max(p + r, 1e-8)
logger.info(f"F1={f1:.3f} (P={p:.2f}, R={r:.2f}), TP={tp}, FP={fp}, FN={fn}")

logger.info(f"\nTP edges:")
for e in sorted(val_norm & gt_norm):
    logger.info(f"  {e[0]} → {e[1]}")
logger.info(f"\nFP edges:")
for e in sorted(val_norm - gt_norm):
    logger.info(f"  {e[0]} → {e[1]}")
logger.info(f"\nFN (missing from GT):")
for e in sorted(gt_norm - val_norm):
    logger.info(f"  {e[0]} → {e[1]}")

# Save updated results
with open(os.path.join(RESULTS_DIR, 'validated_edges.json'), 'w') as f:
    json.dump([list(e) for e in sorted(all_edges)], f, indent=2)
with open(os.path.join(RESULTS_DIR, 'edge_evidence.json'), 'w') as f:
    json.dump(evidence, f, indent=2)

metrics = {
    'n_validated': len(validated), 'n_rescued': len(rescued),
    'n_total': len(all_edges),
    'gt_f1': round(f1, 3), 'gt_precision': round(p, 3),
    'gt_recall': round(r, 3), 'gt_shd': fp + fn,
}
with open(os.path.join(RESULTS_DIR, 'discovery_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

logger.info(f"\nSaved to {RESULTS_DIR}/")
