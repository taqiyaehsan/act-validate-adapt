"""
Row A coefficient-only ablation.

Tests whether do-operator interventions in the discovery loop produce better
SCM coefficients than purely observational fitting, holding the graph constant.

Setup (per the advisor's revision feedback, April 27):
  Row A (NEW):          G_validated + Ridge SEM fit on obs + intervention probes
  Row B (current):      G_validated + Ridge SEM fit on obs only — already in
                        results/closedloop_server/results_{weather}.csv as 'Validated'
  Row C (current):      G_obs-only  + Ridge SEM fit on obs only — already there as 'Obs-Only'

Intervention probe protocol:
  4 probe episodes of N_PROBE_STEPS each, occupancy and windowPosition forced
  to zero (base regime). Actuator pinned at extreme value:
    - HVAC=1.0, Light=0.5
    - HVAC=0.0, Light=0.5
    - HVAC=0.5, Light=1.0
    - HVAC=0.5, Light=0.0
  Each step recorded as a row with weight=PROBE_WEIGHT in the CSV. Concatenated
  to the 11520-row observational training CSV; the combined dataframe is
  passed to CausalPolicyEngine, whose Ridge fit uses the weight column as
  sample_weight (see src/policy_engine.py: _fit_sems).

Closed-loop evaluation matches run_closedloop_server.py exactly:
  5 seeds × 3 weather × 5 comfort targets, regime-aware policy, 1440 steps/episode.

Output: results/rowA_ablation/results_{weather}.csv

Paired comparison against existing Row B/Row C numbers happens offline via
analyze_rowA.py (same paired-stat machinery as reproduce_paired_stats.py).
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

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--weather', type=str, default=None,
                    choices=['cold', 'warm', 'mild'],
                    help='Run a single weather. Omit to run all three.')
parser.add_argument('--smoke', action='store_true',
                    help='Smoke test: 1 seed, 1 epsilon, 1 weather (~3min).')
parser.add_argument('--regenerate-probes', action='store_true',
                    help='Force re-generation of intervention probe CSV.')
parser.add_argument('--probe-weight', type=float, default=10.0,
                    help='Sample weight for intervention rows (default 10).')
parser.add_argument('--probe-steps', type=int, default=200,
                    help='Steps per probe episode (default 200, 4 episodes total).')
parser.add_argument('--workers', type=int, default=5,
                    help='Parallel episode workers via ThreadPoolExecutor. '
                         'Each worker spawns its own JS sim subprocess; '
                         'GIL releases during subprocess.run, so threads scale. '
                         '5 workers per weather thread × 3 weather threads = '
                         '15 concurrent JS subprocesses, fits in 18 cores. '
                         'Set to 1 for sequential.')
args = parser.parse_args()

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ── Config ────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 999]
COMFORT_TARGETS = [0.5, 0.8, 1.0, 1.5, 2.0]
WEATHER = {
    'cold': {'offset': -3.0, 'day': 0},
    'warm': {'offset':  6.0, 'day': 1},
    'mild': {'offset':  6.0, 'day': 3},
}
N_STEPS = 1440
OUT_DIR_BASE = 'results/rowA_ablation'
PROBE_CSV = f'{OUT_DIR_BASE}/intv_probes.csv'

if args.smoke:
    SEEDS = [42]
    COMFORT_TARGETS = [1.0]
    WEATHER = {'cold': WEATHER['cold']}
    OUT_DIR = f'{OUT_DIR_BASE}_smoke'
    logger.info("SMOKE TEST: 1 seed × 1 weather × 1 epsilon")
elif args.weather:
    WEATHER = {args.weather: WEATHER[args.weather]}
    OUT_DIR = f'{OUT_DIR_BASE}_{args.weather}'
else:
    OUT_DIR = OUT_DIR_BASE

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR_BASE, exist_ok=True)

# ── Load sim config and observational data ────────────────────────────────
cfg = SIM_CONFIGS['smart_building_rich']
data_obs = pd.read_csv(cfg['data_path'])
cols = [c for c in data_obs.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
data_obs = data_obs[cols]
cm = {v.lower(): v for v in data_obs.columns}
pcfg = cfg['policy_config']
latent = cfg.get('latent_vars', set())

power_kw = SIM_POWER_SPECS.get(
    os.path.basename(cfg['sim_path']).replace('.js', ''), {})
scaling = pd.read_csv(cfg['scaling_path'])
smap = get_scaling_map(scaling)
sim_abs = os.path.abspath(cfg['sim_path'])

# ── Validated edges: same JSON used by run_closedloop_server.py ───────────
EDGE_SETS_PATH = 'results/closedloop_server/edge_sets.json'
if not os.path.exists(EDGE_SETS_PATH):
    raise FileNotFoundError(
        f"{EDGE_SETS_PATH} not found. Required for Row A — must use the "
        f"same validated graph as Row B."
    )
with open(EDGE_SETS_PATH) as f:
    edge_sets = json.load(f)
val_edges = {(s, t) for s, t in edge_sets['validated']}
val_edges_lower = {(s.lower(), t.lower()) for s, t in val_edges}
logger.info(f"Loaded validated graph: {len(val_edges)} edges")


# ── Simulator step (mirrors run_closedloop_server.py) ─────────────────────
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


# ── Probe data generation ─────────────────────────────────────────────────
def generate_probes():
    """Run 4 fixed-action probe episodes with occupancy/window pinned to zero.

    Each episode contributes args.probe_steps rows; each row records the
    physical state vector after the do-operation takes effect. Resulting CSV
    matches the 15-column schema of data_regen/smart_building_rich_processed.csv,
    plus a 'weight' column.
    """
    probe_actions = [
        {'hvacpower': 1.0, 'lightingpower': 0.5},
        {'hvacpower': 0.0, 'lightingpower': 0.5},
        {'hvacpower': 0.5, 'lightingpower': 1.0},
        {'hvacpower': 0.5, 'lightingpower': 0.0},
    ]

    # Force base regime: occupancy=0, windowPosition=0
    force_intv = {'occupancy': 0, 'windowPosition': 0.0}

    # Use cold weather for probes (matches existing closed-loop generation)
    outdoor_offset = WEATHER.get('cold', {'offset': -3.0})['offset']
    if 'cold' not in WEATHER:
        # When user passes --weather warm, still generate probes once with cold offset
        outdoor_offset = -3.0

    rows = []
    for ep_idx, action in enumerate(probe_actions):
        logger.info(f"  Probe episode {ep_idx+1}/4: {action}")
        state_phys = None
        for t in range(args.probe_steps):
            elapsed_ms = (0 * 24 * 3600 + t * 60) * 1000  # day 0
            new_state = step_simulator(state_phys, action, elapsed_ms,
                                       outdoor_offset, force_intv)
            if new_state is None:
                continue
            state_phys = new_state

            # Convert physical state to normalized values matching CSV schema
            row_norm = {}
            for col in data_obs.columns:
                key = col.lower()
                # Map CSV col → JS sim key (reverse js_var_map)
                js_name = cfg['js_var_map'].get(key, col)
                phys = state_phys.get(js_name, state_phys.get(col,
                       state_phys.get(col[0].lower() + col[1:])))
                if phys is None:
                    # Try direct key match (some keys are PascalCase in sim)
                    for k, v in state_phys.items():
                        if k.lower() == key:
                            phys = v
                            break
                if phys is None:
                    continue
                if isinstance(phys, (int, float)):
                    row_norm[col] = phys_to_norm(float(phys), key, smap)
            if len(row_norm) == len(data_obs.columns):
                rows.append(row_norm)

    df_probe = pd.DataFrame(rows)
    df_probe['weight'] = args.probe_weight
    df_probe.to_csv(PROBE_CSV, index=False)
    logger.info(f"  Probe CSV written: {PROBE_CSV} "
                f"({len(df_probe)} rows, weight={args.probe_weight})")
    return df_probe


# ── Build augmented training data ─────────────────────────────────────────
if args.regenerate_probes or not os.path.exists(PROBE_CSV):
    logger.info("Generating intervention probe data...")
    df_probe = generate_probes()
else:
    df_probe = pd.read_csv(PROBE_CSV)
    logger.info(f"Loaded probes from {PROBE_CSV}: {len(df_probe)} rows, "
                f"mean weight={df_probe['weight'].mean():.1f}")

# Concatenate obs (weight=1) + probes (weight=probe_weight)
data_obs_w = data_obs.copy()
data_obs_w['weight'] = 1.0
data_aug = pd.concat([data_obs_w, df_probe], ignore_index=True)
logger.info(f"Augmented dataset: {len(data_obs_w)} obs (weight=1) + "
            f"{len(df_probe)} probes (weight={args.probe_weight}) = {len(data_aug)} rows")


# ── Build regime-specific engines from augmented data ─────────────────────
def build_regime_engines(edges_mixed, dataset):
    if len(edges_mixed) == 0:
        return {r: None for r in ['base', 'occ', 'win', 'full']}
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
        except Exception:
            engines[rname] = None
    return engines


# ── Closed-loop episode (mirrors run_closedloop_server.py) ────────────────
def run_closedloop_episode(policy_fn, weather_key, seed):
    random.seed(seed); np.random.seed(seed)
    wcfg = WEATHER[weather_key]
    outdoor_offset = wcfg['offset']
    day = wcfg['day']

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


# ── Build the single Row A arm ────────────────────────────────────────────
logger.info("\n=== Building Row A arm: G_validated + intervention-augmented Ridge SEM ===")

# Monitors are fit on OBSERVATIONAL data only — temporal monitor would be
# corrupted by short probe episodes (var → var_next jumps at boundaries).
# Row A specifically targets POLICY SEM coefficient quality, not monitor structure.
monitors = construct_factored_monitors(val_edges_lower, data_obs)
sensor_weights = compute_sensor_weights_from_graph(
    val_edges_lower, SENSOR_VARS, LATENT_VARS)

# Policy engines are fit on AUGMENTED data — this is what Row A is testing.
engines = build_regime_engines(val_edges, data_aug)
n_eng = sum(1 for e in engines.values() if e is not None)
logger.info(f"  Validated-IntvCoef: {len(val_edges)} edges, "
            f"{n_eng}/4 engines, monitors={'OK' if monitors else 'None'}")
for rname, eng in engines.items():
    if eng is not None:
        logger.info(f"    SEM({rname}): R²={eng._model_confidence:.3f}")


# ── Resume support ────────────────────────────────────────────────────────
partial_csv = f'{OUT_DIR}/results.csv'
if os.path.exists(partial_csv):
    results = pd.read_csv(partial_csv).to_dict('records')
    done = {(r['method'], r['seed'], r['weather'], r['comfort_target'])
            for r in results}
    logger.info(f"Resuming: {len(results)} rows, {len(done)} done")
else:
    results = []
    done = set()


# ── Run (parallel via ThreadPoolExecutor) ─────────────────────────────────
METHOD_NAME = 'Validated-IntvCoef'  # Row A
t0 = time.time()

# Build the work queue
tasks = []
for weather_key in WEATHER:
    for seed in SEEDS:
        for ct in COMFORT_TARGETS:
            if (METHOD_NAME, seed, weather_key, ct) in done:
                continue
            tasks.append((seed, weather_key, ct))
total_tasks = len(tasks)
logger.info(f"Total tasks: {total_tasks} (workers={args.workers})")

results_lock = Lock()
completed_count = [0]  # mutable counter via list

def run_one_episode(seed, weather_key, ct):
    """Run a single closed-loop episode and return the result row."""
    policy_fn = make_regime_aware_policy(
        engines, monitors, sensor_weights, ct)
    df = run_closedloop_episode(policy_fn, weather_key, seed)

    if df is None or len(df) == 0:
        return None

    total_kwh = df['kwh_step'].sum()
    mean_sat = df['satisfaction'].mean()
    kwh_max = sum(power_kw.values()) * N_STEPS * STEP_HOURS
    sw = max(0.3, min(0.85, 1.0 - ct * 0.35))
    ew = 1.0 - sw
    mo = sw * (mean_sat / 100) + ew * (1 - total_kwh / max(kwh_max, 1e-6))

    regime_stats = {}
    for gt in ['base', 'occ', 'win', 'full']:
        rdf = df[df['gt_regime'] == gt]
        if len(rdf):
            regime_stats[f'{gt}_steps'] = len(rdf)
            regime_stats[f'{gt}_kwh'] = round(rdf['kwh_step'].sum(), 2)
            regime_stats[f'{gt}_sat'] = round(rdf['satisfaction'].mean(), 1)

    return {
        'method': METHOD_NAME, 'seed': seed,
        'weather': weather_key, 'comfort_target': ct,
        'kWh': round(total_kwh, 2),
        'satisfaction': round(mean_sat, 2),
        'MO': round(mo, 4),
        'mean_hvac': round(df['hvac'].mean(), 3),
        **regime_stats,
    }


with ThreadPoolExecutor(max_workers=args.workers) as executor:
    future_to_task = {executor.submit(run_one_episode, s, w, c): (s, w, c)
                      for (s, w, c) in tasks}
    for fut in as_completed(future_to_task):
        s, w, c = future_to_task[fut]
        try:
            row = fut.result()
        except Exception as e:
            logger.warning(f"  TASK FAILED s={s} w={w} ε={c}: {e}")
            row = None
        with results_lock:
            if row:
                results.append(row)
            completed_count[0] += 1
            pd.DataFrame(results).to_csv(partial_csv, index=False)
            elapsed = (time.time() - t0) / 60
            tag = f"kWh={row['kWh']:.1f} sat={row['satisfaction']:.1f} MO={row['MO']:.3f}" if row else "FAILED"
            logger.info(f"  [{completed_count[0]}/{total_tasks}] s={s} w={w} ε={c}: "
                        f"{tag} ({elapsed:.1f}min)")

# ── Summary ───────────────────────────────────────────────────────────────
result_df = pd.DataFrame(results)
result_df.to_csv(partial_csv, index=False)
elapsed = (time.time() - t0) / 60

logger.info(f"\n{'='*72}")
logger.info(f"Row A summary ({len(SEEDS)} seeds × {len(WEATHER)} weather, {elapsed:.1f}min)")
logger.info(f"{'='*72}")
for ct in COMFORT_TARGETS:
    rows = result_df[result_df['comfort_target'] == ct]
    if len(rows):
        logger.info(f"  ε={ct}: kWh={rows['kWh'].mean():.2f}±{rows['kWh'].std():.2f} "
                    f"sat={rows['satisfaction'].mean():.1f}±{rows['satisfaction'].std():.1f} "
                    f"MO={rows['MO'].mean():.3f}±{rows['MO'].std():.3f} "
                    f"(n={len(rows)})")

logger.info(f"\nResults written to: {partial_csv}")
logger.info(f"Compare against Row B (current Validated) and Row C (Obs-Only) in:")
logger.info(f"  server_runs/closedloop_server/results_{{cold,warm,mild}}.csv")
