"""
W2: Closed-loop with Ridge SEM fitted on PID-driven training data.

Tests whether the existing 90/10 actuator split in our default training CSV
is operationalizing Proposition 1's deconfounding condition. Compare to the
existing Row B (Validated graph + 90/10 obs CSV) at ε=1.0 across the same
5 seeds × 3 weather paired conditions.

Setup:
  Row B (existing):  G_validated + Ridge on data_regen/smart_building_rich_processed.csv
                     (90/10 split: 90% steady-state + 10% extremes)
  Row B' (NEW W2):   G_validated + Ridge on data_regen/smart_building_rich_pid_processed.csv
                     (PID-driven: HVAC responds to temperature, lighting on schedule)

Expected outcomes:
  (a) B' worse than B → 90/10 doing real Prop-1 deconfounding work; story 1.
  (b) B' ≈ B          → Ridge robust to actuator-context correlation; story 2.
  (c) B' better than B → uncomfortable; flag and reconsider before submission.

Usage:
  python run_w2_truly_obs.py --weather cold --workers 5
  python run_w2_truly_obs.py                # all 3 weathers sequentially
  python run_w2_truly_obs.py --smoke        # 1 seed × 1 weather × ε=1.0
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
                                norm_to_phys, SIM_POWER_SPECS, STEP_HOURS)
from neurips_final_experiments import (
    construct_factored_monitors, compute_sensor_weights_from_graph,
    get_scaling_map, OBSERVABLE_VARS, SENSOR_VARS, LATENT_VARS
)
from src.policy_engine import CausalPolicyEngine
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── CLI ──
parser = argparse.ArgumentParser()
parser.add_argument('--weather', type=str, default=None,
                    choices=['cold', 'warm', 'mild'])
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--workers', type=int, default=5)
parser.add_argument('--training-csv',
                    default='data_regen/smart_building_rich_pid_processed.csv',
                    help='PID-driven training CSV (run generate_pid_training_data.py first).')
parser.add_argument('--training-scaling',
                    default='data_regen/smart_building_rich_pid_processed_scaling.csv')
parser.add_argument('--comfort-targets', nargs='+', type=float,
                    default=[1.0],
                    help='Comfort targets to evaluate. Default ε=1.0 (paired '
                         'comparison point); pass multiple for full sweep.')
args = parser.parse_args()

# ── Config (mirrors run_closedloop_server / run_rowA_ablation) ──
SEEDS = [42, 123, 456, 789, 999]
WEATHER = {
    'cold': {'offset': -3.0, 'day': 0},
    'warm': {'offset':  6.0, 'day': 1},
    'mild': {'offset':  6.0, 'day': 3},
}
N_STEPS = 1440
OUT_DIR = 'results/w2_truly_obs'
os.makedirs(OUT_DIR, exist_ok=True)

if args.smoke:
    SEEDS = [42]
    WEATHER = {'cold': WEATHER['cold']}
    args.comfort_targets = [1.0]
    logger.info("SMOKE: 1 seed × 1 weather × ε=1.0")
elif args.weather:
    WEATHER = {args.weather: WEATHER[args.weather]}
    OUT_DIR = f'{OUT_DIR}_{args.weather}'
    os.makedirs(OUT_DIR, exist_ok=True)
    logger.info(f"Single weather: {args.weather}")

cfg = SIM_CONFIGS['smart_building_rich']
pcfg = cfg['policy_config']
latent = cfg.get('latent_vars', set())

# Load PID training data + its scaling map
if not os.path.exists(args.training_csv):
    raise FileNotFoundError(
        f"{args.training_csv} not found.\n"
        f"Run: python generate_pid_training_data.py --steps 11520 --seed 42"
    )
data = pd.read_csv(args.training_csv)
cm = {v.lower(): v for v in data.columns}
logger.info(f"Loaded PID training data: {data.shape[0]} rows × {data.shape[1]} cols")

# IMPORTANT: closed-loop episodes need the simulator's *production* scaling
# (the standard one, derived from the original 10k CSV), because the simulator
# generates physical-unit observations that get normalized for the SCM. The
# *training* scaling (from PID CSV) is only used internally for the Ridge fit
# — but Ridge is fit on already-normalized [0,1] data so the scaling at fit
# time is just the identity. The closed-loop simulator-side normalization
# uses the standard scaling.
prod_scaling = pd.read_csv(cfg['scaling_path'])
smap = get_scaling_map(prod_scaling)

power_kw = SIM_POWER_SPECS.get('smart_building_rich', {})
sim_abs = os.path.abspath(cfg['sim_path'])

# ── Validated edges (same JSON as Row B) ──
EDGE_SETS_PATH = 'results/closedloop_server/edge_sets.json'
with open(EDGE_SETS_PATH) as f:
    edge_sets = json.load(f)
val_edges = {(s, t) for s, t in edge_sets['validated']}
logger.info(f"Validated edges: {len(val_edges)}")


# ── Build regime-specific engines on PID data ─────────────────────────────
def build_regime_engines(edges_mixed, dataset):
    cols_lower = {c.lower(): c for c in dataset.columns}
    occ_vals = dataset[cols_lower['occupancy']].values
    win_vals = dataset[cols_lower['windowposition']].values
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
        data_r = dataset[mask]
        if len(data_r) < 20:
            data_r = dataset
        drop_cols = [c for c in data_r.columns
                     if c.lower() in ('occupancy', 'windowposition', 'windowopen')]
        data_r = data_r.drop(columns=drop_cols, errors='ignore')
        try:
            engines[rname] = CausalPolicyEngine(
                {'validated_edges': edges_mixed}, data_r, **kwargs)
        except Exception as e:
            logger.warning(f"  Engine build failed for {rname}: {e}")
            engines[rname] = None
    return engines


# Build monitors and sensor weights on PID data, matching run_closedloop_server
val_edges_lower = {(s.lower(), t.lower()) for s, t in val_edges}
logger.info("Building monitors on PID data...")
monitors = construct_factored_monitors(val_edges_lower, data)
sensor_weights = compute_sensor_weights_from_graph(
    val_edges_lower, SENSOR_VARS, LATENT_VARS)
engines = build_regime_engines(val_edges, data)
n_eng = sum(1 for e in engines.values() if e is not None)
logger.info(f"  Engines: {n_eng}/4, monitors: {'OK' if monitors else 'None'}")


# ── Closed-loop machinery (mirrors run_closedloop_server.py) ──────────────
def step_simulator(state_phys, action, elapsed_ms, outdoor_offset=0,
                   force_intervention=None):
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
    except Exception:
        return None
    for line in result.stdout.split('\n'):
        if 'RESULT:' in line:
            return json.loads(line.split('RESULT:', 1)[1].strip())
    return None


def make_regime_aware_policy(engines, monitors, sensor_weights, comfort_target):
    sw = max(0.3, min(0.85, 1.0 - comfort_target * 0.35))
    ew = 1.0 - sw
    sigma = 1.5
    forget_factor = 0.15
    regime_names = ['base', 'occ', 'win', 'full']
    state = {'weights': {r: 0.25 for r in regime_names}}

    def policy_fn(obs, t):
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
            except Exception:
                pass
        return {v.lower(): 0.5 for v in cfg['action_vars']}

    return policy_fn


def run_episode(weather_key, seed, comfort_target):
    random.seed(seed); np.random.seed(seed)
    wcfg = WEATHER[weather_key]
    outdoor_offset = wcfg['offset']
    day = wcfg['day']
    policy_fn = make_regime_aware_policy(engines, monitors, sensor_weights,
                                          comfort_target)
    state_phys = None
    records = []
    for t in range(N_STEPS):
        elapsed_ms = (day * 24 * 3600 + t * 60) * 1000
        hour_of_day = (elapsed_ms / 3600000) % 24
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
            sat = state_phys.get('satisfaction',
                    state_phys.get('OverallSatisfaction', 50))
            records.append({
                'step': t, 'hour': hour_of_day,
                'satisfaction': sat,
                'kwh_step': step_kwh,
                'hvac': action.get('hvacpower', 0.5),
            })
    if not records:
        return None
    df = pd.DataFrame(records)
    return {
        'method': 'Validated_PID',
        'seed': seed, 'weather': weather_key,
        'comfort_target': comfort_target,
        'kWh': round(df['kwh_step'].sum(), 2),
        'satisfaction': round(df['satisfaction'].mean(), 2),
        'mean_hvac': round(df['hvac'].mean(), 3),
    }


# ── Resume support ──
results_path = f'{OUT_DIR}/results.csv'
if os.path.exists(results_path):
    results = pd.read_csv(results_path).to_dict('records')
    done = {(r['seed'], r['weather'], r['comfort_target']) for r in results}
    logger.info(f"Resuming: {len(results)} rows already done")
else:
    results = []
    done = set()


# ── Run all (seed, weather, ε) triples in parallel via threads ──
todos = []
for w in WEATHER:
    for s in SEEDS:
        for ct in args.comfort_targets:
            if (s, w, ct) not in done:
                todos.append((w, s, ct))
logger.info(f"Episodes to run: {len(todos)}")

t0 = time.time()
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futures = {ex.submit(run_episode, w, s, ct): (w, s, ct) for w, s, ct in todos}
    for fut in as_completed(futures):
        w, s, ct = futures[fut]
        try:
            r = fut.result()
        except Exception as e:
            logger.error(f"  Episode FAILED ({w}, seed {s}, ε{ct}): {e}")
            continue
        if r is None:
            continue
        results.append(r)
        # Save after each completion (resume safety)
        pd.DataFrame(results).to_csv(results_path, index=False)
        elapsed = time.time() - t0
        logger.info(f"  [{len(results)}/{len(todos)+len(done)}] "
                    f"{w} seed{s} ε{ct}: kWh={r['kWh']:.2f} sat={r['satisfaction']:.1f} "
                    f"({elapsed:.0f}s elapsed)")

logger.info(f"\nDone in {time.time()-t0:.0f}s. Results: {results_path}")

# ── Compare to Row B ──
df_w2 = pd.DataFrame(results)
df_w2_eps1 = df_w2[df_w2['comfort_target'] == 1.0]

if len(df_w2_eps1) > 0:
    logger.info("\n=== W2 (PID training) vs Row B (90/10 obs) at ε=1.0 ===")
    # Load Row B from existing closed-loop results
    rb_rows = []
    for w in ['cold', 'warm', 'mild']:
        rb_path = f'server_runs/closedloop_server/results_{w}.csv'
        if os.path.exists(rb_path):
            rb = pd.read_csv(rb_path)
            rb_rows.append(rb[(rb['method'] == 'Validated') &
                              (rb['comfort_target'] == 1.0)])
    if rb_rows:
        rb = pd.concat(rb_rows, ignore_index=True)
        merged = df_w2_eps1.merge(
            rb[['seed', 'weather', 'kWh']].rename(columns={'kWh': 'kWh_rowB'}),
            on=['seed', 'weather'])
        merged['delta'] = merged['kWh'] - merged['kWh_rowB']
        logger.info(f"Paired conditions: {len(merged)}")
        logger.info(f"  W2 (PID-trained):    {merged['kWh'].mean():.2f} ± {merged['kWh'].std():.2f}")
        logger.info(f"  Row B (90/10 obs):   {merged['kWh_rowB'].mean():.2f} ± {merged['kWh_rowB'].std():.2f}")
        logger.info(f"  Δ (W2 - B): {merged['delta'].mean():.3f} ± {merged['delta'].std():.3f}")
        from scipy import stats
        t_stat, p = stats.ttest_rel(merged['kWh'], merged['kWh_rowB'])
        logger.info(f"  Paired t = {t_stat:.2f}, p = {p:.2e}")
        n_favor_b = (merged['delta'] > 0).sum()
        logger.info(f"  Cells favoring Row B (90/10): {n_favor_b}/{len(merged)}")
