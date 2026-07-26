#!/usr/bin/env python3
"""Run ALL benchmarks on ALL sims with correct GT, 5 seeds, directed + skeleton F1."""
import sys, os, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import pandas as pd, numpy as np

with open('ground_truth_graphs.json') as f:
    gts_raw = json.load(f)

def get_gt(sim):
    edges = gts_raw.get(sim, {}).get('edges', [])
    if not edges and 'hidden' in sim:
        edges = gts_raw.get('smart_room', {}).get('edges', [])
    return {(s.lower(), t.lower()) for s, t in edges}

def score_both(edges, gt):
    e = {(str(s).lower(), str(t).lower()) for s, t in edges}
    gt_skel = {frozenset(x) for x in gt}
    e_skel = {frozenset(x) for x in e}
    # Directed
    tp = len(e & gt); fp = len(e - gt); fn = len(gt - e)
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
    f1 = 2*p*r/max(p+r,1e-8)
    # Skeleton
    stp = len(e_skel & gt_skel); sfp = len(e_skel - gt_skel); sfn = len(gt_skel - e_skel)
    sp = stp/max(stp+sfp,1); sr = stp/max(stp+sfn,1)
    sf1 = 2*sp*sr/max(sp+sr,1e-8)
    return round(f1,3), round(sf1,3), round(p,3), round(r,3), len(e)

def extract(r):
    e = r[0] if isinstance(r, tuple) else r
    if hasattr(e, 'edges'): e = list(e.edges())
    return list(e) if isinstance(e, (list,set)) else []

from benchmarks_new.utils import preprocess_benchmark_data, set_benchmark_data
from benchmarks_new.pc_baseline import run_pc_baseline
from benchmarks_new.sam_baseline import run_sam_baseline
from benchmarks_new.gies_comparison import run_gies
from benchmarks_new.icp_comparison import run_icp
from benchmarks_new.notears_i import run_notears_i
from benchmarks_new.jci_comparison import run_jci
from benchmarks_new.causal_bandits import run_causal_bandits
from benchmarks_new.iid_comparison import run_iid
from src.generators import VARLiNGAMGenerator

sims = {
    'smart_room': ('data_regen/smart_room_processed.csv', 'js/smart_room.js'),
    'smart_room_noise': ('data_regen/smart_room_noise_processed.csv', 'js/smart_room_noise.js'),
    'smart_room_hidden_vars': ('data_regen/smart_room_hidden_vars_processed.csv', 'js/smart_room_hidden_vars.js'),
}

benchmarks = [
    ('PC', lambda p, sp: run_pc_baseline(p, sp)),
    ('SAM', lambda p, sp: run_sam_baseline(p, sp)),
    ('GIES', lambda p, sp: run_gies(p, sp)),
    ('ICP', lambda p, sp: run_icp(p, sp)),
    ('NOTEARS-I', lambda p, sp: run_notears_i(p, sp)),
    ('JCI', lambda p, sp: run_jci(p, sp)),
    ('Causal Bandits', lambda p, sp: run_causal_bandits(p, sp)),
    ('IID', lambda p, sp: run_iid(p, sp)),
]

seeds = [42, 123, 456, 789, 999]

for sim_name, (data_path, sim_js) in sims.items():
    gt = get_gt(sim_name)
    print(f'\n{"="*60}')
    print(f'  {sim_name} ({len(gt)} GT edges)')
    print(f'{"="*60}')

    data = pd.read_csv(data_path)
    cols = [c for c in data.columns if c.lower() not in {'timestamp','elapsedtime'}]
    data = data[cols]
    preprocessed = preprocess_benchmark_data(data)
    set_benchmark_data(data)
    sp = os.path.abspath(sim_js) if os.path.exists(sim_js) else None

    rows = []
    for method_name, func in benchmarks:
        f1s, sf1s = [], []
        for seed in seeds:
            np.random.seed(seed)
            try:
                edges = extract(func(preprocessed, sp))
                f1, sf1, p, r, n = score_both(edges, gt)
                f1s.append(f1); sf1s.append(sf1)
            except Exception as e:
                f1s.append(0.0); sf1s.append(0.0)
        rows.append({
            'method': method_name,
            'f1_mean': round(np.mean(f1s),3), 'f1_std': round(np.std(f1s),3),
            'skel_f1_mean': round(np.mean(sf1s),3), 'skel_f1_std': round(np.std(sf1s),3),
        })
        print(f'  {method_name:20s}: F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  skelF1={np.mean(sf1s):.3f}±{np.std(sf1s):.3f}')

    # VARLiNGAM
    f1s, sf1s = [], []
    for seed in seeds:
        np.random.seed(seed)
        try:
            vl = VARLiNGAMGenerator(max_lags=3, criterion='bic', threshold=0.05)
            vl_r = vl.generate(preprocessed)
            nm = {c.lower(): c for c in cols}
            vl_edges = [(nm.get(s.lower(),s), nm.get(t.lower(),t)) for s,t in vl_r.get('edges',[])]
            f1, sf1, p, r, n = score_both(vl_edges, gt)
            f1s.append(f1); sf1s.append(sf1)
        except:
            f1s.append(0.0); sf1s.append(0.0)
    rows.append({
        'method': 'VARLiNGAM',
        'f1_mean': round(np.mean(f1s),3), 'f1_std': round(np.std(f1s),3),
        'skel_f1_mean': round(np.mean(sf1s),3), 'skel_f1_std': round(np.std(sf1s),3),
    })
    print(f'  {"VARLiNGAM":20s}: F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  skelF1={np.mean(sf1s):.3f}±{np.std(sf1s):.3f}')

    df = pd.DataFrame(rows)
    out = f'server_runs/full_pipeline/{sim_name}/benchmarks_final.csv'
    df.to_csv(out, index=False)
    print(f'  → {out}')

print('\nALL DONE')
