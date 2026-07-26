#!/usr/bin/env python3
"""
Run all discovery benchmarks on physical testbed data.
Uses PolicyGRID's real intervention results (21 physical do-operator tests)
as the shared intervention data for all methods.

This is the fairest comparison: every method gets the same observational
data AND the same real intervention evidence.
"""
import sys, os, json, ast
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)  # suppress benchmark spam

BASELINE_CSV = 'smart_room_physical/deployed/data/baseline_data.csv'
INTERVENTION_LOG = 'smart_room_physical/deployed/results/discovery/seed_42/intervention_log.csv'

PROPOSED_GT = {
    ('Heater', 'Temperature'), ('Heater', 'EnergyConsumption'),
    ('Humidifier', 'Humidity'), ('Humidifier', 'EnergyConsumption'),
    ('Fan', 'Temperature'), ('Fan', 'AirQuality'), ('Fan', 'EnergyConsumption'),
    ('Occupancy', 'Temperature'), ('Occupancy', 'Humidity'),
    ('WindowState', 'Temperature'), ('WindowState', 'Humidity'),
    ('WindowState', 'AirQuality'),
    ('Temperature', 'Satisfaction'), ('Humidity', 'Satisfaction'),
}
gt = {(s.lower(), t.lower()) for s, t in PROPOSED_GT}


EDGES_DIR = 'smart_room_physical/deployed/results/discovery/benchmark_edges'
os.makedirs(EDGES_DIR, exist_ok=True)

all_edge_sets = {}  # method_name -> list of (source, target) tuples

def score(edges, name):
    e_norm = {(s.lower(), t.lower()) for s, t in edges}
    tp = len(e_norm & gt)
    fp = len(e_norm - gt)
    fn = len(gt - e_norm)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    print(f'{name:25s}: F1={f1:.3f}  P={p:.2f}  R={r:.2f}  '
          f'edges={len(edges):2d}  TP={tp}  FP={fp}  FN={fn}')
    # Save edges
    edge_list = [list(e) for e in edges]
    all_edge_sets[name] = edge_list
    safe_name = name.replace(' ', '_').replace('-', '_').lower()
    with open(os.path.join(EDGES_DIR, f'{safe_name}.json'), 'w') as f:
        json.dump(edge_list, f, indent=2)
    return {'method': name, 'f1': f1, 'precision': p, 'recall': r,
            'edges': len(edges), 'tp': tp, 'fp': fp, 'fn': fn}


def build_intervention_data(baseline, intv_log):
    """
    Convert PolicyGRID's intervention log into augmented data
    that benchmarks can use. Each intervention becomes a row in the
    augmented dataset with the device forced to its intervention value.
    """
    cols = [c for c in baseline.columns if c != 'timestamp']
    base_mean = baseline[cols].mean()

    intv_rows = []
    for _, row in intv_log.iterrows():
        effects = ast.literal_eval(row['effects']) if isinstance(row['effects'], str) else row['effects']
        device = row['device']

        # Post-intervention state: baseline mean + measured effects
        post = base_mean.copy()
        for var, delta in effects.items():
            if var in post.index:
                post[var] = base_mean[var] + delta

        # Set the device to 1.0 (ON) for do(ON) or 0.0 for counterfactual
        if 'counterfactual' in str(row.get('strategy', '')):
            post[device] = 0.0
        else:
            post[device] = 1.0

        intv_rows.append(post.to_dict())

    intv_df = pd.DataFrame(intv_rows)
    augmented = pd.concat([baseline[cols], intv_df], ignore_index=True)
    return augmented


def extract_edges(result):
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, (list, set)):
        return list(result)
    if hasattr(result, 'edges'):
        return list(result.edges())
    return []


# ── Load data ──
baseline = pd.read_csv(BASELINE_CSV)
cols = [c for c in baseline.columns if c != 'timestamp']
intv_log = pd.read_csv(INTERVENTION_LOG)

print(f'Baseline: {len(baseline)} rows, Interventions: {len(intv_log)} physical tests')
print(f'GT: {len(gt)} edges')
print()

from benchmarks_new.utils import preprocess_benchmark_data, set_benchmark_data
preprocessed = preprocess_benchmark_data(baseline[cols])

# Build augmented dataset with real intervention data
augmented = build_intervention_data(baseline, intv_log)
augmented_preprocessed = preprocess_benchmark_data(augmented)

# Also set RF surrogate as fallback
set_benchmark_data(baseline[cols])

results = []

# ── PolicyGRID (from saved results) ──
with open('smart_room_physical/deployed/results/discovery/seed_42/validated_edges.json') as f:
    pg_edges = {tuple(e) for e in json.load(f)}
results.append(score(pg_edges, 'PolicyGRID'))

# ── PolicyGRID-O (union of PC+SAM+VARLiNGAM, no validation) ──
from src.generators import PCGenerator, SAMGenerator, VARLiNGAMGenerator

pc = PCGenerator(relevant_columns=cols)
pc_edges = [(s, t) for s, t in pc.generate(preprocessed).get('edges', [])]

sam = SAMGenerator(relevant_columns=cols, allow_cycles=True)
sam_edges = [(s, t) for s, t in sam.generate(preprocessed).get('edges', [])]

vl = VARLiNGAMGenerator(max_lags=3, criterion='bic', threshold=0.05)
vl_result = vl.generate(preprocessed)
# Normalize case
name_map = {c.lower(): c for c in cols}
vl_edges = []
for s, t in vl_result.get('edges', []):
    sn = name_map.get(s.lower(), s)
    tn = name_map.get(t.lower(), t)
    if sn in cols and tn in cols:
        vl_edges.append((sn, tn))

union = set(pc_edges) | set(sam_edges) | set(vl_edges)
results.append(score(union, 'PolicyGRID-O'))

# ── Individual obs methods ──
results.append(score(pc_edges, 'PC'))
results.append(score(sam_edges, 'SAM'))
results.append(score(vl_edges, 'VARLiNGAM'))

# ── LLM-only ──
try:
    from src.generators import LLMGenerator
    api_key = "YOUR_OPENAI_API_KEY"
    llm = LLMGenerator(api_key=api_key, relevant_columns=cols, dataset_type='smart_room')
    llm_result = llm.generate(preprocessed)
    llm_edges = [(s, t) for s, t in llm_result.get('edges', [])]
    results.append(score(llm_edges, 'LLM-only'))
except Exception as e:
    print(f'LLM-only failed: {e}')

# ── GIES (uses interventional data) ──
try:
    from benchmarks_new.gies_comparison import run_gies
    edges = run_gies(augmented_preprocessed, simulation_path=None)
    results.append(score(extract_edges(edges), 'GIES'))
except Exception as e:
    print(f'GIES failed: {e}')

# ── Methods that need intervention data — feed augmented dataset ──
# These methods will now see the real physical intervention data
# appended to the observational baseline.

# ICP: needs environments (intervention contexts)
try:
    from benchmarks_new.icp_comparison import run_icp
    edges = run_icp(augmented_preprocessed, simulation_path=None, max_interventions=5)
    results.append(score(extract_edges(edges), 'ICP'))
except Exception as e:
    print(f'ICP failed: {e}')

# NOTEARS-I: NOTEARS on augmented obs+intv data
try:
    from benchmarks_new.notears_i import run_notears_i
    edges = run_notears_i(augmented_preprocessed, simulation_path=None, max_interventions=5)
    results.append(score(extract_edges(edges), 'NOTEARS-I'))
except Exception as e:
    print(f'NOTEARS-I failed: {e}')

# JCI: joint causal inference on multiple contexts
try:
    from benchmarks_new.jci_comparison import run_jci
    edges = run_jci(augmented_preprocessed, simulation_path=None, max_interventions=5)
    results.append(score(extract_edges(edges), 'JCI'))
except Exception as e:
    print(f'JCI failed: {e}')

# Causal Bandits
try:
    from benchmarks_new.causal_bandits import run_causal_bandits
    edges = run_causal_bandits(augmented_preprocessed, simulation_path=None, num_iterations=5)
    results.append(score(extract_edges(edges), 'Causal Bandits'))
except Exception as e:
    print(f'Causal Bandits failed: {e}')

# IID: PC on pooled data
try:
    from benchmarks_new.iid_comparison import run_iid
    edges = run_iid(augmented_preprocessed, simulation_path=None, max_iterations=3)
    results.append(score(extract_edges(edges), 'IID'))
except Exception as e:
    print(f'IID failed: {e}')

# ── Save results ──
print()
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('f1', ascending=False)
print('=' * 80)
print('FINAL RANKING:')
print('=' * 80)
for _, r in results_df.iterrows():
    print(f'{r["method"]:25s}: F1={r["f1"]:.3f}  P={r["precision"]:.2f}  '
          f'R={r["recall"]:.2f}  edges={r["edges"]:2.0f}')

out_path = 'smart_room_physical/deployed/results/discovery/physical_benchmarks.csv'
results_df.to_csv(out_path, index=False)
print(f'\nSaved to {out_path}')
print(f'Edge sets saved to {EDGES_DIR}/')

# ── Run monitoring on ALL methods ──
print('\n' + '=' * 80)
print('  MONITORING ON ALL METHODS')
print('=' * 80)

from smart_room_physical.deployed.run_physical_monitoring_v2 import (
    preprocess_regime_data, build_monitors, run_monitoring, compute_metrics
)

BASELINE_MON = 'smart_room_physical/deployed/data/baseline_data.csv'
REGIME_CSV = 'smart_room_physical/deployed/data/sensor_data_regime_monitoring.csv'
SCALING_CSV = 'smart_room_physical/deployed/data/baseline_data_scaling.csv'

baseline_mon = pd.read_csv(BASELINE_MON)
regime = preprocess_regime_data(REGIME_CSV, SCALING_CSV)
half = len(regime) // 2

print(f'Baseline: {len(baseline_mon)} rows, Regime: {len(regime)} rows')

mon_results = []
for method_name, edge_list in all_edge_sets.items():
    edges = [tuple(e) for e in edge_list]
    try:
        # Strategy A: OOD (train=baseline, test=regime)
        monitors_a = build_monitors(baseline_mon, edges, mode='graph_parents')
        records_a = run_monitoring(monitors_a, regime, sigma=0.5, forget=0.2)
        m_a = compute_metrics(records_a)

        # Strategy B: In-distribution (train=regime first half, test=second half)
        monitors_b = build_monitors(regime.iloc[:half], edges, mode='graph_parents')
        records_b = run_monitoring(monitors_b, regime.iloc[half:].reset_index(drop=True),
                                   sigma=0.5, forget=0.2)
        m_b = compute_metrics(records_b)

        mon_results.append({
            'method': method_name,
            'ood_occ': m_a['occ'], 'ood_win': m_a['win'], 'ood_comb': m_a['comb'],
            'ind_occ': m_b['occ'], 'ind_win': m_b['win'], 'ind_comb': m_b['comb'],
        })
        print(f'{method_name:25s}: OOD={m_a["occ"]:.1f}/{m_a["win"]:.1f}/{m_a["comb"]:.1f}  '
              f'InD={m_b["occ"]:.1f}/{m_b["win"]:.1f}/{m_b["comb"]:.1f}')
    except Exception as e:
        print(f'{method_name:25s}: MONITORING FAILED - {e}')
        mon_results.append({
            'method': method_name,
            'ood_occ': None, 'ood_win': None, 'ood_comb': None,
            'ind_occ': None, 'ind_win': None, 'ind_comb': None,
        })

mon_df = pd.DataFrame(mon_results)
mon_path = 'smart_room_physical/deployed/results/monitoring_v2/all_methods_monitoring_full.csv'
mon_df.to_csv(mon_path, index=False)
print(f'\nMonitoring results saved to {mon_path}')
