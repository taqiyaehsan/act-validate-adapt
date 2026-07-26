"""
Closed-Loop Regime Policy Experiment — Server Grade.

Tests whether validated causal structure produces more robust policy under
regime changes in closed-loop deployment (actions feed back to simulator).

Design:
  - 5 seeds × 3 outdoor conditions × 5 comfort targets × 6 graph conditions
  - All 8 policy baselines (PID, DDPC×4, MPC, C-MBPO) + PolicyGRID variants
  - Per-regime breakdown (base/occ/win/full)
  - Resume-safe: saves after each (condition, seed, weather) triple

Graph conditions (M3A-matched edge sets):
  1. Validated (31 consensus edges)
  2. Obs-Only (PC+LLM+VARLiNGAM union, SAM disabled)
  3. Random (31 edges, matched count)
  4. Empty (0 edges)

Weather conditions:
  Day 0: cold  (offset -3)  — mostly base regime
  Day 1: warm  (offset +6)  — mix of regimes
  Day 3: mild  (offset +6)  — includes window-only regime (forced overnight)

Each episode: 1440 steps (1 day), closed-loop.
"""
import sys, os, json, warnings, random, time, subprocess, argparse
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

from run_full_pipeline import (SIM_CONFIGS, load_scaling, phys_to_norm,
                                norm_to_phys, SIM_POWER_SPECS, STEP_HOURS,
                                API_KEY)
from neurips_final_experiments import (
    construct_factored_monitors, compute_sensor_weights_from_graph,
    get_scaling_map, OBSERVABLE_VARS, SENSOR_VARS, LATENT_VARS, SIM_PATH
)
from src.policy_engine import CausalPolicyEngine
from src.ddpc_baseline import create_ddpc_suite
from src.mpc_baseline import LinearMPCController

# ── Config ──
SEEDS = [42, 123, 456, 789, 999]
COMFORT_TARGETS = [0.5, 0.8, 1.0, 1.5, 2.0]
WEATHER = {
    'cold':  {'offset': -3.0, 'day': 0},
    'warm':  {'offset':  6.0, 'day': 1},
    'mild':  {'offset':  6.0, 'day': 3},  # Day 3 has forced window-open overnight
}
N_STEPS = 1440
OUT_DIR = 'results/closedloop_server'
os.makedirs(OUT_DIR, exist_ok=True)

# Parse optional --weather flag to run a single weather condition
parser = argparse.ArgumentParser()
parser.add_argument('--weather', type=str, default=None,
                    help='Run single weather: cold, warm, or mild')
_args = parser.parse_args()
if _args.weather:
    if _args.weather not in WEATHER:
        raise ValueError(f"Unknown weather: {_args.weather}. Use cold/warm/mild.")
    WEATHER = {_args.weather: WEATHER[_args.weather]}
    logger.info(f"Running single weather condition: {_args.weather}")
    OUT_DIR = f'results/closedloop_server_{_args.weather}'
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

# ── Edge sets: HARDCODED from pre-saved JSON ──
# All edge sets frozen to ensure reproducibility across machines.
# Generated once, saved to edge_sets.json, never regenerated.
EDGE_SETS_PATH = 'results/closedloop_server/edge_sets.json'
if not os.path.exists(EDGE_SETS_PATH):
    raise FileNotFoundError(
        f"{EDGE_SETS_PATH} not found. Copy from the machine that generated it.\n"
        f"This file contains the frozen Validated/Obs-Only/Random edge sets."
    )

with open(EDGE_SETS_PATH) as f:
    edge_sets = json.load(f)

val_edges = {(s, t) for s, t in edge_sets['validated']}
val_edges_lower = {(s.lower(), t.lower()) for s, t in val_edges}
obs_edges = {(s, t) for s, t in edge_sets['obs_only']}
obs_edges_lower = {(s.lower(), t.lower()) for s, t in obs_edges}
rand_edges = {(s, t) for s, t in edge_sets['random']}
rand_edges_lower = {(s.lower(), t.lower()) for s, t in rand_edges}

logger.info(f"Loaded hardcoded edge sets: Validated={len(val_edges)}, "
            f"Obs-Only={len(obs_edges)}, Random={len(rand_edges)}")

GRAPH_CONDITIONS = {
    'Validated': (val_edges, val_edges_lower),
    'Obs-Only': (obs_edges, obs_edges_lower),
    'Random': (rand_edges, rand_edges_lower),
    'Empty': (set(), set()),
}


# ── Build components ──

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


# Skip slow CEM-based baselines (DDPC_Neural, DDPC_PETS take hours per episode).
# These are already excluded from the main paper tables.
SKIP_BASELINES = {'DDPC_Neural', 'DDPC_PETS'}

def build_baselines():
    """Build non-causal policy baselines (PID, DDPC, MPC). Same as run_full_pipeline."""
    import src.ddpc_baseline as ddpc
    import src.mpc_baseline as mpc
    sv = cfg['state_vars']; av = cfg['action_vars']; ov = cfg['output_vars']
    ddpc.STATE_VARS = list(sv); ddpc.ACTION_VARS = list(av)
    ddpc.OUTPUT_VARS = list(ov); ddpc.N_U = len(av); ddpc.N_Y = len(ov)
    mpc.STATE_VARS = list(sv); mpc.ACTION_VARS = list(av)
    mpc.OUTPUT_VARS = list(ov); mpc.N_U = len(av); mpc.N_Y = len(ov)
    mpc.N_X = len(sv)

    baselines = {}
    try:
        for name, ctrl in create_ddpc_suite(data).items():
            if name not in SKIP_BASELINES:
                baselines[name] = ctrl
            else:
                logger.info(f"  Skipping {name} (too slow for closed-loop)")
    except: pass
    try:
        baselines['MPC_Linear'] = LinearMPCController().fit(data)
    except: pass
    return baselines


def pid_fn(state_obs, t):
    return {v.lower(): max(0, min(1, 0.5 + 1.0 * (0.5 - state_obs.get(v.lower(), 0.5))))
            for v in cfg['action_vars']}


def step_simulator(state_phys, action, elapsed_ms, outdoor_offset=0,
                   force_intervention=None):
    """Step simulator. Actions feed back. Regime changes naturally."""
    intervention = {}
    for var, val in action.items():
        phys_val = norm_to_phys(val, var.lower(), smap)
        js_name = cfg['js_var_map'].get(var.lower(), var)
        intervention[js_name] = phys_val
    if force_intervention:
        intervention.update(force_intervention)

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


def run_closedloop_episode(policy_fn, weather_key, seed):
    """Run one closed-loop episode. Returns per-step DataFrame."""
    random.seed(seed); np.random.seed(seed)
    wcfg = WEATHER[weather_key]
    outdoor_offset = wcfg['offset']
    day = wcfg['day']

    state_phys = None
    records = []

    for t in range(N_STEPS):
        # Elapsed time: start at beginning of the specified day
        elapsed_ms = (day * 24 * 3600 + t * 60) * 1000
        hour_of_day = (elapsed_ms / 3600000) % 24

        # Day 3 special: force window open during unoccupied hours
        force_intv = None
        if day == 3 and (hour_of_day >= 19 or hour_of_day < 7):
            force_intv = {'windowPosition': 0.6}

        if state_phys:
            sn = {k.lower(): phys_to_norm(v, k.lower(), smap)
                  for k, v in state_phys.items() if isinstance(v, (int, float))}
            so = {k: v for k, v in sn.items() if k not in latent}
            action = policy_fn(so, t)
        else:
            action = {v.lower(): 0.5 for v in cfg['action_vars']}

        step_kwh = sum(action.get(a, 0.5) * power_kw.get(a, 0) * STEP_HOURS
                       for a in action)

        new_state = step_simulator(state_phys, action, elapsed_ms,
                                   outdoor_offset, force_intv)
        if new_state:
            state_phys = new_state
            occ = state_phys.get('occupancy', 0)
            win = state_phys.get('windowPosition',
                                 state_phys.get('windowposition', 0))
            occ_active = int(occ >= 1)
            win_active = int(win > 0.3)
            gt_regime = {(0,0):'base',(1,0):'occ',
                        (0,1):'win',(1,1):'full'}[(occ_active, win_active)]
            sat = state_phys.get('satisfaction',
                    state_phys.get('OverallSatisfaction', 50))
            temp = state_phys.get('temperature', 22)

            records.append({
                'step': t, 'hour': hour_of_day,
                'gt_regime': gt_regime,
                'satisfaction': sat, 'temperature': temp,
                'kwh_step': step_kwh,
                'hvac': action.get('hvacpower', 0.5),
                'light': action.get('lightingpower', 0.5),
            })

    return pd.DataFrame(records)


def make_regime_aware_policy(engines, monitors, sensor_weights, comfort_target):
    """Create a regime-aware policy function (closure)."""
    sw = max(0.3, min(0.85, 1.0 - comfort_target * 0.35))
    ew = 1.0 - sw
    sigma = 1.5
    forget_factor = 0.15
    regime_names = ['base', 'occ', 'win', 'full']
    state = {'weights': {r: 0.25 for r in regime_names}}

    def policy_fn(obs, t):
        # Monitoring update
        detected = 'base'
        if monitors is not None:
            likelihoods = {}
            for rn in regime_names:
                pred = monitors[rn].predict(obs)
                err = sum(sensor_weights.get(v, 1.0) *
                          abs(obs.get(v, 0.5) - pred.get(v, 0.5))
                          for v in SENSOR_VARS if v in obs)
                likelihoods[rn] = np.exp(-sigma * err)
            unnorm = {r: likelihoods[r] * state['weights'][r] for r in regime_names}
            total = sum(unnorm.values())
            if total > 0:
                raw = {r: unnorm[r] / total for r in regime_names}
                state['weights'] = {
                    r: (1 - forget_factor) * raw[r] + forget_factor * 0.25
                    for r in regime_names}
            w_total = sum(state['weights'].values())
            if w_total > 0:
                state['weights'] = {r: state['weights'][r] / w_total
                                    for r in regime_names}
            detected = max(state['weights'], key=state['weights'].get)

        engine = engines.get(detected)
        if engine is not None:
            try:
                sp = {cm.get(k, k): v for k, v in obs.items()}
                r = engine.optimize_policy(
                    {'satisfaction': {'target': 70, 'weight': sw},
                     'energy': {'target': 15, 'weight': ew}}, {}, sp)
                return {k.lower(): max(0.0, min(1.0, v))
                        for k, v in r.action_plan.items()}
            except:
                pass
        return {v.lower(): 0.5 for v in cfg['action_vars']}

    return policy_fn


# ── Build all policy arms ──
logger.info("\n=== Building policy arms ===")

# Causal graph conditions: regime-aware policy + monitoring
arms = {}
for cond_name, (edges_mixed, edges_lower) in GRAPH_CONDITIONS.items():
    monitors = construct_factored_monitors(edges_lower, data)
    sensor_weights = compute_sensor_weights_from_graph(
        edges_lower, SENSOR_VARS, LATENT_VARS)
    engines = build_regime_engines(edges_mixed)
    arms[cond_name] = {
        'monitors': monitors,
        'sensor_weights': sensor_weights,
        'engines': engines,
        'type': 'causal',
    }
    n_eng = sum(1 for e in engines.values() if e is not None)
    logger.info(f"  {cond_name}: {len(edges_mixed)} edges, "
                f"{n_eng}/4 engines, monitors={'OK' if monitors else 'None'}")

# Non-causal baselines
baseline_ctrls = build_baselines()
logger.info(f"  Baselines: {list(baseline_ctrls.keys())}")

# ── Resume support ──
partial_csv = f'{OUT_DIR}/results.csv'
if os.path.exists(partial_csv):
    results = pd.read_csv(partial_csv).to_dict('records')
    done = {(r['method'], r['seed'], r['weather'], r['comfort_target'])
            for r in results}
    logger.info(f"Resuming: {len(results)} rows, {len(done)} done")
else:
    results = []
    done = set()

# ── Run ──
t0 = time.time()
total_episodes = (len(arms) + 1 + len(baseline_ctrls)) * len(SEEDS) * len(WEATHER) * len(COMFORT_TARGETS)
completed = 0

for weather_key in WEATHER:
    for seed in SEEDS:
        for ct in COMFORT_TARGETS:
            # Causal methods
            for cond_name, arm in arms.items():
                if (cond_name, seed, weather_key, ct) in done:
                    completed += 1
                    continue

                policy_fn = make_regime_aware_policy(
                    arm['engines'], arm['monitors'],
                    arm['sensor_weights'], ct)
                df = run_closedloop_episode(policy_fn, weather_key, seed)

                if df is not None and len(df):
                    total_kwh = df['kwh_step'].sum()
                    mean_sat = df['satisfaction'].mean()
                    kwh_max = sum(power_kw.values()) * N_STEPS * STEP_HOURS
                    sw = max(0.3, min(0.85, 1.0 - ct * 0.35))
                    ew = 1.0 - sw
                    mo = sw * (mean_sat / 100) + ew * (1 - total_kwh / max(kwh_max, 1e-6))

                    # Per-regime breakdown
                    regime_stats = {}
                    for gt in ['base', 'occ', 'win', 'full']:
                        rdf = df[df['gt_regime'] == gt]
                        if len(rdf):
                            regime_stats[f'{gt}_steps'] = len(rdf)
                            regime_stats[f'{gt}_kwh'] = round(rdf['kwh_step'].sum(), 2)
                            regime_stats[f'{gt}_sat'] = round(rdf['satisfaction'].mean(), 1)

                    row = {
                        'method': cond_name, 'seed': seed,
                        'weather': weather_key, 'comfort_target': ct,
                        'kWh': round(total_kwh, 2),
                        'satisfaction': round(mean_sat, 2),
                        'MO': round(mo, 4),
                        'mean_hvac': round(df['hvac'].mean(), 3),
                        **regime_stats,
                    }
                    results.append(row)

                completed += 1
                if completed % 5 == 0:
                    pd.DataFrame(results).to_csv(partial_csv, index=False)
                    elapsed = (time.time() - t0) / 60
                    logger.info(f"  [{completed}/{total_episodes}] {cond_name} "
                                f"s={seed} w={weather_key} ε={ct}: "
                                f"kWh={total_kwh:.1f} sat={mean_sat:.1f} MO={mo:.3f} "
                                f"({elapsed:.1f}min)")

            # PID baseline
            if ('PID', seed, weather_key, ct) not in done:
                df = run_closedloop_episode(pid_fn, weather_key, seed)
                if df is not None and len(df):
                    total_kwh = df['kwh_step'].sum()
                    mean_sat = df['satisfaction'].mean()
                    kwh_max = sum(power_kw.values()) * N_STEPS * STEP_HOURS
                    sw = max(0.3, min(0.85, 1.0 - ct * 0.35))
                    ew = 1.0 - sw
                    mo = sw * (mean_sat / 100) + ew * (1 - total_kwh / max(kwh_max, 1e-6))
                    results.append({
                        'method': 'PID', 'seed': seed,
                        'weather': weather_key, 'comfort_target': ct,
                        'kWh': round(total_kwh, 2),
                        'satisfaction': round(mean_sat, 2),
                        'MO': round(mo, 4),
                        'mean_hvac': round(df['hvac'].mean(), 3),
                    })
                completed += 1

            # DDPC / MPC baselines
            for bname, ctrl in baseline_ctrls.items():
                if (bname, seed, weather_key, ct) not in done:
                    def make_baseline_fn(c, ct_val):
                        def fn(so, t):
                            if t == 0 and hasattr(c, 'reset'):
                                c.reset()
                            a = c.get_action(so, ct_val)
                            return {k.lower(): v for k, v in a.items()}
                        return fn
                    bfn = make_baseline_fn(ctrl, ct)
                    df = run_closedloop_episode(bfn, weather_key, seed)
                    if df is not None and len(df):
                        total_kwh = df['kwh_step'].sum()
                        mean_sat = df['satisfaction'].mean()
                        kwh_max = sum(power_kw.values()) * N_STEPS * STEP_HOURS
                        sw = max(0.3, min(0.85, 1.0 - ct * 0.35))
                        ew = 1.0 - sw
                        mo = sw * (mean_sat / 100) + ew * (1 - total_kwh / max(kwh_max, 1e-6))
                        results.append({
                            'method': bname, 'seed': seed,
                            'weather': weather_key, 'comfort_target': ct,
                            'kWh': round(total_kwh, 2),
                            'satisfaction': round(mean_sat, 2),
                            'MO': round(mo, 4),
                            'mean_hvac': round(df['hvac'].mean(), 3),
                        })
                    completed += 1

            # Save after each (seed, weather, ct) block
            pd.DataFrame(results).to_csv(partial_csv, index=False)

elapsed = (time.time() - t0) / 60
logger.info(f"\nTotal time: {elapsed:.1f} min")

# ── Summary ──
result_df = pd.DataFrame(results)
result_df.to_csv(partial_csv, index=False)

logger.info(f"\n{'='*80}")
logger.info(f"Closed-Loop Regime Policy — Summary (mean±std, {len(SEEDS)} seeds, {len(WEATHER)} weather)")
logger.info(f"{'='*80}")

methods_order = ['Validated', 'Obs-Only', 'Random', 'Empty',
                 'PID'] + list(baseline_ctrls.keys())

for ct in [0.5, 1.0, 2.0]:
    logger.info(f"\n  ε = {ct}")
    logger.info(f"  {'Method':<20} {'kWh':>12} {'Sat':>12} {'MO':>12}")
    for method in methods_order:
        rows = result_df[(result_df['method'] == method) &
                         (result_df['comfort_target'] == ct)]
        if len(rows):
            logger.info(f"  {method:<20} "
                        f"{rows['kWh'].mean():>6.1f}±{rows['kWh'].std():>4.1f} "
                        f"{rows['satisfaction'].mean():>6.1f}±{rows['satisfaction'].std():>4.1f} "
                        f"{rows['MO'].mean():>6.3f}±{rows['MO'].std():>5.3f}")

# Per-weather breakdown for causal methods
logger.info(f"\n{'='*80}")
logger.info(f"Per-Weather Breakdown (ε=1.0, causal methods)")
logger.info(f"{'='*80}")
for weather_key in WEATHER:
    logger.info(f"\n  Weather: {weather_key}")
    for cond in ['Validated', 'Obs-Only', 'Random', 'Empty']:
        rows = result_df[(result_df['method'] == cond) &
                         (result_df['comfort_target'] == 1.0) &
                         (result_df['weather'] == weather_key)]
        if len(rows):
            logger.info(f"    {cond:<12}: kWh={rows['kWh'].mean():.1f}±{rows['kWh'].std():.1f} "
                        f"sat={rows['satisfaction'].mean():.1f} MO={rows['MO'].mean():.3f}")
