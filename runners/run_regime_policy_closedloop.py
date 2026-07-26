"""
Closed-Loop Regime-Change Policy Experiment.

Actions feed back to the simulator. Wrong regime detection → wrong action
→ worse next state → compounds over time. This couples monitoring to policy.

Regime transitions happen NATURALLY via the simulator's internal time-based
schedule (elapsed_ms drives occupancy/window). The policy only controls
HVAC and lighting. Monitoring uses the proven construct_factored_monitors.

Each condition uses its OWN monitoring (from its edge set) and its OWN
regime-specific policy engines (from its edge set). This tests the full
system, not isolated components.
"""
import sys, os, json, warnings, random, time, subprocess
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

from run_full_pipeline import (SIM_CONFIGS, load_scaling, phys_to_norm,
                                norm_to_phys, SIM_POWER_SPECS, STEP_HOURS, API_KEY)
from neurips_final_experiments import (
    construct_factored_monitors, compute_sensor_weights_from_graph,
    collect_sim_data, get_scaling_map,
    OBSERVABLE_VARS, SENSOR_VARS, LATENT_VARS, ALL_VARS, SIM_PATH
)
from src.policy_engine import CausalPolicyEngine

SEEDS = [42, 123, 456]
N_STEPS = 1440  # 1 day
COMFORT_TARGETS = [0.5, 1.0, 2.0]
OUT_DIR = 'results/regime_policy_closedloop'
os.makedirs(OUT_DIR, exist_ok=True)

cfg = SIM_CONFIGS['smart_building_rich']
data = pd.read_csv(cfg['data_path'])
cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
data = data[cols]
cm = {v.lower(): v for v in data.columns}
pcfg = cfg['policy_config']
latent = cfg.get('latent_vars', set())

power_kw = SIM_POWER_SPECS.get(
    os.path.basename(cfg['sim_path']).replace('.js', ''), {})
scaling = pd.read_csv(cfg['scaling_path'])
smap = get_scaling_map(scaling)
sim_abs = os.path.abspath(cfg['sim_path'])

# ── Edge sets (M3A-matched) ──
with open('server_runs/full_pipeline/smart_building_rich/consensus_analysis.json') as f:
    ca = json.load(f)
val_edges_lower = {(s.lower(), t.lower()) for s, t in ca['consensus_edges']}
val_edges = {(cm.get(s, s), cm.get(t, t)) for s, t in ca['consensus_edges']}

import src.pipeline as _pl
class _NoSAM:
    def __init__(self, **kwargs): pass
    def generate(self, data): return []
_pl.SAMGenerator = _NoSAM

obs_cache = f'{OUT_DIR}/obs_edges_m3a_match.json'
if os.path.exists(obs_cache):
    with open(obs_cache) as f:
        cached = json.load(f)
    obs_edges = {(s, t) for s, t in cached}
    obs_edges_lower = {(s.lower(), t.lower()) for s, t in cached}
else:
    from src.pipeline_cwm import create_cwm_pipeline
    random.seed(42); np.random.seed(42)
    obs_pipeline = create_cwm_pipeline(
        csv_data=data, api_key=API_KEY, smart_room_path=sim_abs,
        dataset_type=cfg['dataset_type'], max_iterations=0,
        actuator_vars=cfg['actuators'],
        non_intervenable_vars=cfg['non_intervenable'],
    )
    method_dags = obs_pipeline.pipeline._generate_hypotheses()
    obs_lower = set()
    for dag in method_dags.values():
        for e in obs_pipeline.pipeline._extract_edges(dag):
            obs_lower.add(tuple(e) if isinstance(e, list) else e)
    obs_edges = {(cm.get(s, s), cm.get(t, t)) for s, t in obs_lower}
    obs_edges_lower = {(s.lower(), t.lower()) for s, t in obs_lower}
    with open(obs_cache, 'w') as f:
        json.dump(list(obs_edges), f)

all_vars = list(data.columns)
random.seed(42)
rand_edges = set()
while len(rand_edges) < len(val_edges):
    s, t = random.choice(all_vars), random.choice(all_vars)
    if s.lower() != t.lower():
        rand_edges.add((s, t))
rand_edges_lower = {(s.lower(), t.lower()) for s, t in rand_edges}

logger.info(f"Validated: {len(val_edges)}, Obs-Only: {len(obs_edges)}, Random: {len(rand_edges)}")

CONDITIONS = {
    'Validated': (val_edges, val_edges_lower),
    'Obs-Only': (obs_edges, obs_edges_lower),
    'Random': (rand_edges, rand_edges_lower),
    'Empty': (set(), set()),
}


def build_regime_engines(edges_mixed):
    if len(edges_mixed) == 0:
        return {r: None for r in ['base', 'occ', 'win', 'full']}

    cols_lower = {c.lower(): c for c in data.columns}
    occ_vals = data[cols_lower['occupancy']].values
    win_vals = data[cols_lower['windowposition']].values
    occ_on = occ_vals >= 0.1
    win_on = win_vals >= 0.15
    masks = {
        'base': ~occ_on & ~win_on, 'occ': occ_on & ~win_on,
        'win': ~occ_on & win_on, 'full': occ_on & win_on,
    }
    kwargs = {'use_llm': False, 'action_vars': cfg['action_vars'],
              'reg_lambda': pcfg['lambda'], 'n_gradient_steps': 1,
              'linear_only': pcfg['linear_only']}
    if pcfg.get('intervention_directions'):
        kwargs['intervention_directions'] = pcfg['intervention_directions']
    engines = {}
    for rname, mask in masks.items():
        data_r = data[mask]
        if len(data_r) < 20:
            data_r = data
        drop_cols = [c for c in data_r.columns
                     if c.lower() in ('occupancy', 'windowposition', 'windowopen')]
        data_r = data_r.drop(columns=drop_cols, errors='ignore')
        try:
            engines[rname] = CausalPolicyEngine(
                {'validated_edges': edges_mixed}, data_r, **kwargs)
        except:
            engines[rname] = None
    return engines


def step_simulator(state_phys, action, elapsed_ms, outdoor_offset=0):
    """Step the simulator with policy actions. Regime changes happen naturally."""
    intervention = {}
    for var, val in action.items():
        phys_val = norm_to_phys(val, var.lower(), smap)
        js_name = cfg['js_var_map'].get(var.lower(), var)
        intervention[js_name] = phys_val

    cmd = ['node', sim_abs, '--single-step',
           '--elapsed-ms', str(int(elapsed_ms))]
    if outdoor_offset:
        cmd += ['--outdoor-offset', str(outdoor_offset)]
    if state_phys:
        cmd += ['--state', json.dumps(state_phys)]
    if intervention:
        cmd += ['--intervention', json.dumps(intervention)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except:
        return None
    for line in result.stdout.split('\n'):
        if 'RESULT:' in line:
            return json.loads(line.split('RESULT:', 1)[1].strip())
    return None


def run_closedloop(cond_name, edges_mixed, edges_lower, seed, comfort_target):
    """Run closed-loop: action → sim → observe → monitor → select engine → action."""
    random.seed(seed); np.random.seed(seed)

    # Build monitoring from this condition's edges
    monitors = construct_factored_monitors(edges_lower, data)
    sensor_weights = compute_sensor_weights_from_graph(
        edges_lower, SENSOR_VARS, LATENT_VARS)

    # Build policy engines from this condition's edges
    engines = build_regime_engines(edges_mixed)

    sw = max(0.3, min(0.85, 1.0 - comfort_target * 0.35))
    ew = 1.0 - sw

    sigma = 1.5
    forget_factor = 0.15
    regime_names = ['base', 'occ', 'win', 'full']
    weights = {r: 0.25 for r in regime_names}

    # Use Day 1 (warm, offset +6) for regime diversity
    outdoor_offset = 6.0
    state_phys = None
    records = []
    ep_kwh = 0.0

    for t in range(N_STEPS):
        elapsed_ms = t * 60000

        if state_phys:
            # Normalize and get observable state
            sn = {k.lower(): phys_to_norm(v, k.lower(), smap)
                  for k, v in state_phys.items() if isinstance(v, (int, float))}
            so = {k: v for k, v in sn.items() if k not in latent}

            # Monitoring update (proven code)
            detected = 'base'
            if monitors is not None:
                likelihoods = {}
                for rn in regime_names:
                    pred = monitors[rn].predict(so)
                    err = sum(sensor_weights.get(v, 1.0) *
                              abs(so.get(v, 0.5) - pred.get(v, 0.5))
                              for v in SENSOR_VARS if v in so)
                    likelihoods[rn] = np.exp(-sigma * err)
                unnorm = {r: likelihoods[r] * weights[r] for r in regime_names}
                total = sum(unnorm.values())
                if total > 0:
                    raw = {r: unnorm[r] / total for r in regime_names}
                    weights = {r: (1 - forget_factor) * raw[r] + forget_factor * 0.25
                               for r in regime_names}
                w_total = sum(weights.values())
                if w_total > 0:
                    weights = {r: weights[r] / w_total for r in regime_names}
                detected = max(weights, key=weights.get)

            p_occ = weights['occ'] + weights['full']
            p_win = weights['win'] + weights['full']

            # Policy: use detected regime's engine
            engine = engines.get(detected)
            if engine is not None:
                try:
                    sp = {cm.get(k, k): v for k, v in so.items()}
                    r = engine.optimize_policy(
                        {'satisfaction': {'target': 70, 'weight': sw},
                         'energy': {'target': 15, 'weight': ew}}, {}, sp)
                    action = {k.lower(): max(0.0, min(1.0, v))
                              for k, v in r.action_plan.items()}
                except:
                    action = {v.lower(): 0.5 for v in cfg['action_vars']}
            else:
                action = {v.lower(): 0.5 for v in cfg['action_vars']}
        else:
            action = {v.lower(): 0.5 for v in cfg['action_vars']}
            p_occ, p_win, detected = 0.5, 0.5, 'base'

        # kWh
        step_kwh = sum(action.get(a, 0.5) * power_kw.get(a, 0) * STEP_HOURS
                       for a in action)
        ep_kwh += step_kwh

        # Step simulator with policy action (regime changes naturally)
        new_state = step_simulator(state_phys, action, elapsed_ms, outdoor_offset)
        if new_state:
            state_phys = new_state
            occ = state_phys.get('occupancy', 0)
            win = state_phys.get('windowPosition',
                                 state_phys.get('windowposition', 0))
            occ_active = int(occ >= 1)
            win_active = int(win > 0.3)
            gt_regime = {(0,0):'base',(1,0):'occ',(0,1):'win',(1,1):'full'
                        }[(occ_active, win_active)]

            sat = state_phys.get('satisfaction',
                    state_phys.get('OverallSatisfaction',
                    state_phys.get('overallSatisfaction', 50)))
            temp = state_phys.get('temperature',
                    state_phys.get('Temperature', 22))

            records.append({
                'step': t, 'hour': elapsed_ms / 3600000,
                'gt_regime': gt_regime, 'detected': detected,
                'correct': int(detected == gt_regime),
                'p_occ': round(p_occ, 3), 'p_win': round(p_win, 3),
                'satisfaction': sat, 'temperature': temp,
                'kwh_step': step_kwh,
                'hvac': action.get('hvacpower', 0.5),
                'light': action.get('lightingpower', 0.5),
            })

    return pd.DataFrame(records)


# ── Main ──
t0 = time.time()
all_results = []

for cond_name, (edges_mixed, edges_lower) in CONDITIONS.items():
    logger.info(f"\n{'='*60}")
    logger.info(f"Condition: {cond_name} ({len(edges_mixed)} edges)")
    logger.info(f"{'='*60}")

    for seed in SEEDS:
        for ct in COMFORT_TARGETS:
            logger.info(f"  seed={seed}, ε={ct}")
            df = run_closedloop(cond_name, edges_mixed, edges_lower, seed, ct)

            if df is not None and len(df) > 0:
                det_acc = df['correct'].mean() * 100
                total_kwh = df['kwh_step'].sum()
                mean_sat = df['satisfaction'].mean()
                kwh_max = sum(power_kw.values()) * N_STEPS * STEP_HOURS
                sw = max(0.3, min(0.85, 1.0 - ct * 0.35))
                ew = 1.0 - sw
                mo = sw * (mean_sat / 100) + ew * (1 - total_kwh / max(kwh_max, 1e-6))

                all_results.append({
                    'condition': cond_name, 'seed': seed,
                    'comfort_target': ct,
                    'det_acc': round(det_acc, 1),
                    'kWh': round(total_kwh, 2),
                    'satisfaction': round(mean_sat, 2),
                    'MO': round(mo, 4),
                    'mean_hvac': round(df['hvac'].mean(), 3),
                })
                logger.info(f"    det={det_acc:.1f}% kWh={total_kwh:.1f} sat={mean_sat:.1f} MO={mo:.3f}")

                # Save timeseries
                ts_path = f'{OUT_DIR}/{cond_name}_seed{seed}_ct{ct}_ts.csv'
                df.to_csv(ts_path, index=False)

result_df = pd.DataFrame(all_results)
result_df.to_csv(f'{OUT_DIR}/results.csv', index=False)

elapsed = (time.time() - t0) / 60
logger.info(f"\nTotal time: {elapsed:.1f} min")

logger.info(f"\n{'='*70}")
logger.info(f"Closed-Loop Regime Policy (mean±std, {len(SEEDS)} seeds)")
logger.info(f"{'='*70}")
logger.info(f"{'Cond':<12} {'ε':>4} {'Det%':>10} {'kWh':>14} {'Sat':>14} {'MO':>14}")

for cond in CONDITIONS:
    for ct in COMFORT_TARGETS:
        rows = result_df[(result_df['condition'] == cond) &
                         (result_df['comfort_target'] == ct)]
        if len(rows):
            logger.info(f"{cond:<12} {ct:>4.1f} "
                        f"{rows['det_acc'].mean():>5.1f}±{rows['det_acc'].std():>4.1f} "
                        f"{rows['kWh'].mean():>7.1f}±{rows['kWh'].std():>4.1f} "
                        f"{rows['satisfaction'].mean():>7.1f}±{rows['satisfaction'].std():>4.1f} "
                        f"{rows['MO'].mean():>7.3f}±{rows['MO'].std():>5.3f}")
    logger.info("")
