"""S1.2 — Tuned MPC baseline closed-loop runner (sbr only).

Runs ONLY the LinearMPC arm with `tuned=True` across the same (seed × weather × ε)
grid the canonical closedloop_server runs. Saves to results/mpc_tuned/results.csv,
schema-compatible with results/closedloop_server*/results.csv so the union is
analyzable downstream.

Why this script exists:
  - The canonical run_closedloop_server.py builds MPC_Linear with default args
    (tuned=False), preserving the paper number.
  - This runner builds it with tuned=True so we can compare:
      MPC_Linear (paper, fixed weights)  vs  MPC_Linear_Tuned (ε-dependent Q/R)
    holding everything else equal.

Output method label: 'MPC_Linear_Tuned' (distinct from 'MPC_Linear' in canonical
file). Resume-safe — appends per-episode rows; restart skips done jobs.
"""
import sys, os, json, warnings, random, time, subprocess, argparse
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

from run_full_pipeline import (SIM_CONFIGS, load_scaling, phys_to_norm,
                                norm_to_phys, SIM_POWER_SPECS, STEP_HOURS,
                                API_KEY)
from src.mpc_baseline import LinearMPCController

# ── CLI ──
parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, nargs='+',
                    default=[42, 123, 456, 789, 999])
parser.add_argument('--weathers', nargs='+', default=['cold', 'warm', 'mild'])
parser.add_argument('--epsilons', type=float, nargs='+',
                    default=[0.5, 0.8, 1.0, 1.5, 2.0])
parser.add_argument('--n-steps', type=int, default=1440,
                    help='Steps per episode (1440 = 1 day at 1 min/step)')
parser.add_argument('--workers', type=int, default=5)
parser.add_argument('--smoke', action='store_true',
                    help='1 seed, 1 weather, 1 ε, 60 steps')
args = parser.parse_args()

if args.smoke:
    args.seeds = [42]
    args.weathers = ['cold']
    args.epsilons = [1.0]
    args.n_steps = 60

# ── sbr config (only sbr has the full state space the MPC was built for) ──
SIM = 'smart_building_rich'
cfg = SIM_CONFIGS[SIM]
data = pd.read_csv(cfg['data_path'])
cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
data = data[cols]
cm = {v.lower(): v for v in data.columns}
power_kw = SIM_POWER_SPECS[SIM]
smap = load_scaling(cfg['scaling_path'])
sim_abs = os.path.abspath(cfg['sim_path'])

# Wire mpc_baseline module globals to sbr's vars (same pattern as
# run_closedloop_server.py:165)
import src.mpc_baseline as mpc_mod
sv = cfg['state_vars']; av = cfg['action_vars']; ov = cfg['output_vars']
mpc_mod.STATE_VARS = list(sv); mpc_mod.ACTION_VARS = list(av)
mpc_mod.OUTPUT_VARS = list(ov); mpc_mod.N_U = len(av); mpc_mod.N_Y = len(ov)
mpc_mod.N_X = len(sv)

WEATHER = {
    'cold':  {'offset': -3.0, 'day': 0},
    'warm':  {'offset':  6.0, 'day': 1},
    'mild':  {'offset':  6.0, 'day': 3},
}
N_STEPS = args.n_steps
OUT_DIR = 'results/mpc_tuned'
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_CSV = f'{OUT_DIR}/results.csv'

# ── Build ONE tuned MPC, share across all episodes (Ridge dynamics are
#    seed-independent, only the cost weights vary per ε via set_comfort_target) ──
logger.info("Fitting tuned LinearMPCController...")
mpc = LinearMPCController(tuned=True).fit(data)
logger.info(f"  fit done. action_vars={av}, state_vars={sv}")


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


def run_episode(weather_key, eps, seed):
    """One closed-loop episode. Returns a summary row matching closedloop_server schema."""
    random.seed(seed); np.random.seed(seed)
    wcfg = WEATHER[weather_key]
    outdoor_offset = wcfg['offset']
    day = wcfg['day']

    # NOTE: ε wires into _cost() via mpc.comfort_target, set per get_action call
    state_phys = None
    sats, hvacs, kwhs = [], [], []
    base_steps, base_kwh, base_sat = 0, 0.0, []
    occ_steps, occ_kwh, occ_sat = 0, 0.0, []
    win_steps, win_kwh, win_sat = 0, 0.0, []
    full_steps, full_kwh, full_sat = 0, 0.0, []
    t0 = time.time()

    for t in range(N_STEPS):
        elapsed_ms = (day * 24 * 3600 + t * 60) * 1000
        hour_of_day = (elapsed_ms / 3600000) % 24
        force_intv = None
        if day == 3 and (hour_of_day >= 19 or hour_of_day < 7):
            force_intv = {'windowPosition': 0.6}

        if state_phys:
            obs = {k.lower(): phys_to_norm(v, k.lower(), smap)
                   for k, v in state_phys.items() if isinstance(v, (int, float))}
            try:
                action = mpc.get_action(obs, comfort_target=eps)
            except Exception:
                action = {v.lower(): 0.5 for v in cfg['action_vars']}
        else:
            action = {v.lower(): 0.5 for v in cfg['action_vars']}

        # get_action returns CamelCase keys ('HVACPower') from ACTION_VARS,
        # but power_kw and the hvac lookup are lowercase — normalize for
        # accounting so kWh/mean_hvac aren't silently zeroed/defaulted.
        act_lc = {k.lower(): v for k, v in action.items()}
        step_kwh = sum(act_lc.get(a, 0.5) * power_kw.get(a, 0) * STEP_HOURS
                       for a in act_lc)
        kwhs.append(step_kwh)
        hvacs.append(act_lc.get('hvacpower', 0.5))

        new_state = step_simulator(state_phys, action, elapsed_ms,
                                   outdoor_offset, force_intv)
        if new_state:
            state_phys = new_state
            occ = state_phys.get('occupancy', 0)
            win = state_phys.get('windowPosition',
                                 state_phys.get('windowposition', 0))
            occ_active = int(occ >= 1)
            win_active = int(win > 0.3)
            sat = state_phys.get('satisfaction', 50)
            sats.append(sat)
            if not occ_active and not win_active:
                base_steps += 1; base_kwh += step_kwh; base_sat.append(sat)
            elif occ_active and not win_active:
                occ_steps += 1; occ_kwh += step_kwh; occ_sat.append(sat)
            elif not occ_active and win_active:
                win_steps += 1; win_kwh += step_kwh; win_sat.append(sat)
            else:
                full_steps += 1; full_kwh += step_kwh; full_sat.append(sat)

    total_kwh = float(np.sum(kwhs))
    mean_sat = float(np.mean(sats)) if sats else float('nan')
    # MO score is the same convention as run_closedloop_server.py
    MO = max(0.0, mean_sat / 100.0) - 0.01 * total_kwh / 30.0

    return {
        'method': 'MPC_Linear_Tuned',
        'seed': seed,
        'weather': weather_key,
        'comfort_target': eps,
        'kWh': round(total_kwh, 3),
        'satisfaction': round(mean_sat, 2),
        'MO': round(MO, 4),
        'mean_hvac': round(float(np.mean(hvacs)), 3),
        'base_steps': base_steps,
        'base_kwh': round(base_kwh, 2),
        'base_sat': round(float(np.mean(base_sat)) if base_sat else float('nan'), 1),
        'occ_steps': occ_steps,
        'occ_kwh': round(occ_kwh, 2),
        'occ_sat': round(float(np.mean(occ_sat)) if occ_sat else float('nan'), 1),
        'full_steps': full_steps,
        'full_kwh': round(full_kwh, 2),
        'full_sat': round(float(np.mean(full_sat)) if full_sat else float('nan'), 1),
        'wall_time_s': round(time.time() - t0, 1),
    }


# ── Job list w/ resume ──
all_jobs = [(w, e, s) for w in args.weathers for e in args.epsilons for s in args.seeds]
done_keys = set()
if os.path.exists(RESULTS_CSV):
    df_existing = pd.read_csv(RESULTS_CSV)
    done_keys = set(zip(df_existing['weather'], df_existing['comfort_target'],
                        df_existing['seed']))
    logger.info(f"Resuming: {len(done_keys)} episodes already in {RESULTS_CSV}")

todo = [j for j in all_jobs if j not in done_keys]
logger.info(f"Total jobs: {len(all_jobs)}, todo: {len(todo)}")
if not todo:
    logger.info("Nothing to do.")
    sys.exit(0)

with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futures = {ex.submit(run_episode, w, e, s): (w, e, s) for (w, e, s) in todo}
    for i, fut in enumerate(as_completed(futures), 1):
        job = futures[fut]
        try:
            row = fut.result()
        except Exception as e:
            logger.error(f"Episode {job} failed: {e}")
            continue
        df_row = pd.DataFrame([row])
        header = not os.path.exists(RESULTS_CSV)
        df_row.to_csv(RESULTS_CSV, mode='a', header=header, index=False)
        logger.info(f"[{i}/{len(todo)}] {row['weather']:<5} seed{row['seed']:<3} "
                    f"ε={row['comfort_target']:.1f}: "
                    f"kWh={row['kWh']:.2f} sat={row['satisfaction']:.1f} "
                    f"({row['wall_time_s']:.0f}s)")

logger.info(f"Wrote {RESULTS_CSV}")
print(f"DONE {os.path.basename(sys.argv[0])} {time.strftime('%Y-%m-%d %H:%M:%S')}")
