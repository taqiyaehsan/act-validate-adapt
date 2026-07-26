#!/usr/bin/env python3
"""
Tune PolicyGRID per-sim: sweep lambda, SEM type, compare with fast baselines.
30 steps per config for speed.
"""
import sys, os, json, subprocess, warnings
import numpy as np, pandas as pd
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from src.policy_engine import CausalPolicyEngine

# ── Sim configs ──────────────────────────────────────────────────────────────
SIMS = {
    'smart_room': {
        'sim': 'js/smart_room.js',
        'data': 'data_regen/smart_room_varied_processed.csv',
        'scaling': 'data_regen/smart_room_varied_scaling.csv',
        'gt_key': 'smart_room',
        'action_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'js_map': {'temperature': 'temperature', 'humidity': 'humidity', 'airquality': 'airQuality'},
        'sat_key': 'overallSatisfaction', 'eng_key': 'energyConsumption',
    },
    'smart_room_noise': {
        'sim': 'js/smart_room_noise.js',
        'data': 'data_regen/smart_room_noise_varied_processed.csv',
        'scaling': 'data_regen/smart_room_noise_varied_scaling.csv',
        'gt_key': 'smart_room_noise',
        'action_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'js_map': {'temperature': 'temperature', 'humidity': 'humidity', 'airquality': 'airQuality'},
        'sat_key': 'overallSatisfaction', 'eng_key': 'energyConsumption',
    },
    'smart_room_hidden_vars': {
        'sim': 'js/smart_room_hidden_vars.js',
        'data': 'data_regen/smart_room_hidden_vars_varied_processed.csv',
        'scaling': 'data_regen/smart_room_hidden_vars_varied_scaling.csv',
        'gt_key': 'hidden_vars',
        'action_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'js_map': {'temperature': 'temperature', 'humidity': 'humidity', 'airquality': 'airQuality'},
        'sat_key': 'overallSatisfaction', 'eng_key': 'energyConsumption',
    },
    'open_window': {
        'sim': 'js/open_window.js',
        'data': 'data_regen/open_window_varied_processed.csv',
        'scaling': 'data_regen/open_window_varied_scaling.csv',
        'gt_key': 'open_window',
        'action_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'js_map': {'temperature': 'temperature', 'humidity': 'humidity', 'airquality': 'airQuality'},
        'sat_key': 'overallSatisfaction', 'eng_key': 'energyConsumption',
    },
    'smart_building': {
        'sim': 'js/smart_building.js',
        'data': 'data_regen/smart_building_varied_processed.csv',
        'scaling': 'data_regen/smart_building_varied_scaling.csv',
        'gt_key': 'smart_building',
        'action_vars': ['HVACSetpoint', 'LightingLevel'],
        'js_map': {'hvacsetpoint': 'HVACSetpoint', 'lightinglevel': 'LightingLevel'},
        'sat_key': 'OverallSatisfaction', 'eng_key': 'EnergyConsumption',
    },
}


def load_sim(cfg):
    data = pd.read_csv(cfg['data'])
    sp = pd.read_csv(cfg['scaling'])
    smap = {row['feature'].lower(): (row['data_min'], row['data_max'])
            for _, row in sp.iterrows()}
    with open('ground_truth_graphs.json') as f:
        gt = json.load(f)[cfg['gt_key']]['edges']
    return data, smap, {(s, t) for s, t in gt}


def n2p(v, var, smap):
    mn, mx = smap.get(var, (0, 1)); return mn + v * (mx - mn)

def p2n(v, var, smap):
    mn, mx = smap.get(var, (0, 1)); return (v - mn) / (mx - mn + 1e-12)


def step_sim(sim_path, sp, an, smap, js_map, ems):
    iv = {js_map.get(k, k): n2p(v, k, smap) for k, v in an.items()}
    cmd = ['node', os.path.abspath(sim_path), '--single-step',
           '--elapsed-ms', str(int(ems))]
    if sp: cmd += ['--state', json.dumps(sp)]
    if iv: cmd += ['--intervention', json.dumps(iv)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except:
        return None
    for l in r.stdout.split('\n'):
        if 'RESULT:' in l:
            return json.loads(l.split('RESULT:', 1)[1].strip())
    return None


def run_episode(policy_fn, cfg, smap, n_steps=30):
    sp = None; recs = []
    av = cfg['action_vars']
    for t in range(n_steps):
        ems = 8 * 3600000 + t * 60000
        if sp:
            sn = {k.lower(): p2n(v, k.lower(), smap)
                  for k, v in sp.items() if isinstance(v, (int, float))}
            a = policy_fn(sn, t)
        else:
            a = {v.lower(): 0.5 for v in av}
        np_ = step_sim(cfg['sim'], sp, a, smap, cfg['js_map'], ems)
        if np_:
            sp = np_
            recs.append({'s': np_.get(cfg['sat_key'], 50),
                         'e': np_.get(cfg['eng_key'], 90)})
    if recs:
        return np.mean([r['s'] for r in recs]), np.mean([r['e'] for r in recs])
    return 0, 0


def make_grid_fn(engine, sw, ew, data, av):
    cm = {v.lower(): v for v in data.columns}
    def fn(so, t):
        try:
            sp = {cm.get(k, k): v for k, v in so.items()}
            r = engine.optimize_policy(
                {'satisfaction': {'target': 70, 'weight': sw},
                 'energy': {'target': 80, 'weight': ew}}, {}, sp)
            return {k.lower(): max(0.0, min(1.0, v))
                    for k, v in r.action_plan.items()}
        except:
            return {v.lower(): 0.5 for v in av}
    return fn


def make_pid(av):
    def fn(so, t):
        return {v.lower(): max(0, min(1, 0.5 + 1.0 * (0.5 - so.get(v.lower(), 0.5))))
                for v in av}
    return fn


def make_baselines(data, cfg, gt_edges, smap):
    """Build fast baselines patched for this sim."""
    import src.ddpc_baseline as ddpc
    import src.cmbpo_baseline as cmbpo

    sv = list(data.columns)
    av = cfg['action_vars']
    ov = [c for c in data.columns if c not in av]

    ddpc.STATE_VARS = sv; ddpc.ACTION_VARS = av; ddpc.OUTPUT_VARS = ov[:2]
    ddpc.N_U = len(av); ddpc.N_Y = min(2, len(ov))
    cmbpo.ACTION_VARS = av; cmbpo.OUTPUT_VARS = ov[:2]
    cmbpo.SENSOR_VARS = av; cmbpo.N_U = len(av); cmbpo.N_Y = min(2, len(ov))

    from src.ddpc_baseline import create_ddpc_suite
    from src.cmbpo_baseline import CMBPOController

    suite = create_ddpc_suite(data)
    cm = CMBPOController(edges=gt_edges); cm.fit(data)

    baselines = {}
    for name in ['DDPC_Behavioral', 'DDPC_Subspace']:
        if name in suite:
            ctrl = suite[name]
            def wrap(c):
                def fn(so, t):
                    if t == 0 and hasattr(c, 'reset'): c.reset()
                    a = c.get_action(so, 0.8)
                    return {k.lower(): v for k, v in a.items()}
                return fn
            baselines[name] = wrap(ctrl)
    def wrap_cm(c):
        def fn(so, t):
            a = c.get_action(so, 0.8)
            return {k.lower(): v for k, v in a.items()}
        return fn
    baselines['C-MBPO'] = wrap_cm(cm)
    return baselines


def tune_sim(sim_name):
    cfg = SIMS[sim_name]
    if not os.path.exists(cfg['data']):
        print(f'  SKIP {sim_name}: no data at {cfg["data"]}')
        return

    data, smap, gt_edges = load_sim(cfg)
    av = cfg['action_vars']
    print(f'\n{"="*60}')
    print(f'  {sim_name}: {data.shape}, GT={len(gt_edges)} edges')
    print(f'{"="*60}')

    # ── PolicyGRID sweep ──
    print(f'\n  PolicyGRID sweep (λ × SEM type):')
    print(f'  {"Config":<25} {"ε=0.5":>10} {"ε=1.0":>10} {"ε=2.0":>10}')
    print(f'  {"-"*58}')

    best_config = None
    best_avg_sat = 0

    for linear_only in [True, False]:
        for lam in [1.0, 3.0, 5.0, 8.0]:
            label = f'{"lin" if linear_only else "quad"} λ={lam}'
            parts = []
            total_sat = 0
            for ct in [0.5, 1.0, 2.0]:
                eng = CausalPolicyEngine(
                    {'validated_edges': gt_edges}, data,
                    use_llm=False, action_vars=av,
                    reg_lambda=lam, n_gradient_steps=1,
                    linear_only=linear_only)
                sw = max(0.3, min(0.85, 1.0 - ct * 0.35)); ew = 1.0 - sw
                fn = make_grid_fn(eng, sw, ew, data, av)
                s, e = run_episode(fn, cfg, smap)
                parts.append(f'{s:.0f}/{e:.0f}')
                total_sat += s
            print(f'  {label:<25} {parts[0]:>10} {parts[1]:>10} {parts[2]:>10}')
            if total_sat > best_avg_sat:
                best_avg_sat = total_sat
                best_config = (linear_only, lam)

    lo, lam = best_config
    print(f'\n  Best: {"linear" if lo else "quadratic"} λ={lam} (total Sat={best_avg_sat:.0f})')

    # ── Compare with baselines ──
    print(f'\n  Baseline comparison (ε=0.5, 1.0, 2.0):')
    print(f'  {"Method":<25} {"ε=0.5":>10} {"ε=1.0":>10} {"ε=2.0":>10}')
    print(f'  {"-"*58}')

    # PolicyGRID with best config
    for ct in [0.5, 1.0, 2.0]:
        eng = CausalPolicyEngine(
            {'validated_edges': gt_edges}, data,
            use_llm=False, action_vars=av,
            reg_lambda=lam, n_gradient_steps=1, linear_only=lo)
        sw = max(0.3, min(0.85, 1.0 - ct * 0.35)); ew = 1.0 - sw
    # Print all at once
    parts = []
    for ct in [0.5, 1.0, 2.0]:
        eng = CausalPolicyEngine(
            {'validated_edges': gt_edges}, data,
            use_llm=False, action_vars=av,
            reg_lambda=lam, n_gradient_steps=1, linear_only=lo)
        sw = max(0.3, min(0.85, 1.0 - ct * 0.35)); ew = 1.0 - sw
        fn = make_grid_fn(eng, sw, ew, data, av)
        s, e = run_episode(fn, cfg, smap)
        parts.append(f'{s:.0f}/{e:.0f}')
    print(f'  {"PolicyGRID":<25} {parts[0]:>10} {parts[1]:>10} {parts[2]:>10}')

    # PID
    pid = make_pid(av)
    s, e = run_episode(pid, cfg, smap)
    print(f'  {"PID":<25} {s:.0f}/{e:.0f}')

    # Fast baselines
    baselines = make_baselines(data, cfg, gt_edges, smap)
    for name, fn in baselines.items():
        s, e = run_episode(fn, cfg, smap)
        print(f'  {name:<25} {s:.0f}/{e:.0f}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', default='all',
                        choices=list(SIMS.keys()) + ['all'])
    args = parser.parse_args()

    if args.sim == 'all':
        for name in SIMS:
            tune_sim(name)
    else:
        tune_sim(args.sim)
