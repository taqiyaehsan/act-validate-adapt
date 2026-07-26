"""Cross-environment closed-loop policy comparison — Step 2c of summer plan.

Tests whether the Validated > Obs-Only closed-loop kWh result replicates on
non-sbr simulators (smart_room, smart_room_hidden_vars).

Static policy only (NO regime monitoring, NO Bayesian update). Rationale:
these sims have no hidden-regime structure to detect — adding a monitor that
never fires would dilute the graph-vs-graph signal. The clean experiment is
"validated graph → static policy" vs "obs-only graph → static policy."

Design:
  - 5 seeds × N comfort targets × 3 edge conditions (Validated / Obs-Only / Random)
  - Per (sim, ε, edge_set, seed): 1 closed-loop episode of N_STEPS, fresh policy
  - Resume-safe: rows appended after each episode to results.csv

Reads edge sets from results/closedloop_xenv/<sim>/edge_sets.json (built by
scripts/build_xenv_edge_sets.py from per-seed edges.json).
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
from src.policy_engine import CausalPolicyEngine

# ── CLI ──
parser = argparse.ArgumentParser()
parser.add_argument('--sim', type=str, required=True,
                    choices=['smart_room', 'smart_room_hidden_vars'])
parser.add_argument('--seeds', type=int, nargs='+',
                    default=[42, 123, 456, 789, 999])
parser.add_argument('--epsilons', type=float, nargs='+',
                    default=[0.5, 0.8, 1.0, 1.5, 2.0])
parser.add_argument('--n-steps', type=int, default=1440,
                    help='Steps per episode (1440 = 1 day at 1 min/step)')
parser.add_argument('--workers', type=int, default=5)
parser.add_argument('--smoke', action='store_true',
                    help='Quick test: 1 seed, 1 ε, 60 steps')
parser.add_argument('--out-dir', type=str, default=None,
                    help='Default: results/closedloop_xenv/<sim>/')
args = parser.parse_args()

SIM = args.sim
if args.smoke:
    args.seeds = [42]
    args.epsilons = [1.0]
    args.n_steps = 60

OUT_DIR = args.out_dir or f'results/closedloop_xenv/{SIM}'
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_CSV = f'{OUT_DIR}/results.csv'

# ── Sim config ──
cfg = SIM_CONFIGS[SIM]
data = pd.read_csv(cfg['data_path'])
cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
data = data[cols]
cm = {v.lower(): v for v in data.columns}
pcfg = cfg['policy_config']
power_kw = SIM_POWER_SPECS[SIM]
scaling = pd.read_csv(cfg['scaling_path'])
smap = load_scaling(cfg['scaling_path'])
sim_abs = os.path.abspath(cfg['sim_path'])

# ── Edge sets ──
EDGE_SETS_PATH = f'results/closedloop_xenv/{SIM}/edge_sets.json'
if not os.path.exists(EDGE_SETS_PATH):
    raise FileNotFoundError(
        f"{EDGE_SETS_PATH} not found. Run scripts/build_xenv_edge_sets.py "
        f"--sim {SIM} first (requires per-seed edges.json with the "
        f"obs_only_union_edges key from the June 1 patch).")

with open(EDGE_SETS_PATH) as f:
    edge_sets = json.load(f)

EDGE_CONDITIONS = {
    'Validated':  {(s, t) for s, t in edge_sets['validated']},
    'Obs-Only':   {(s, t) for s, t in edge_sets['obs_only']},
    'Random':     {(s, t) for s, t in edge_sets['random']},
}
logger.info(f"Edge sets: Validated={len(EDGE_CONDITIONS['Validated'])}, "
            f"Obs-Only={len(EDGE_CONDITIONS['Obs-Only'])}, "
            f"Random={len(EDGE_CONDITIONS['Random'])}")


def step_simulator(state_phys, action, elapsed_ms):
    """Step the JS simulator one tick. Returns new physical state or None."""
    intervention = {}
    for var, val in action.items():
        phys_val = norm_to_phys(val, var.lower(), smap)
        js_name = cfg['js_var_map'].get(var.lower(), var)
        intervention[js_name] = phys_val

    cmd = ['node', sim_abs, '--single-step',
           '--elapsed-ms', str(int(elapsed_ms))]
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


def build_static_engine(edges_mixed):
    """Single CausalPolicyEngine fit on full observational data. No regime split."""
    if len(edges_mixed) == 0:
        return None
    kwargs = {'use_llm': False, 'action_vars': cfg['action_vars'],
              'reg_lambda': pcfg['lambda'], 'n_gradient_steps': 1,
              'linear_only': pcfg['linear_only']}
    if pcfg.get('intervention_directions'):
        kwargs['intervention_directions'] = pcfg['intervention_directions']
    try:
        return CausalPolicyEngine(
            {'validated_edges': edges_mixed}, data, **kwargs)
    except Exception as e:
        logger.warning(f"Engine init failed: {e}")
        return None


def make_static_policy(engine, comfort_target):
    """Static policy closure. Weights derived from ε via the same ε→(w_s, w_e)
    map used in run_closedloop_server.py:277. No monitoring."""
    sw = max(0.3, min(0.85, 1.0 - comfort_target * 0.35))
    ew = 1.0 - sw

    def policy_fn(obs, t):
        if engine is None:
            return {v.lower(): 0.5 for v in cfg['action_vars']}
        try:
            sp = {cm.get(k, k): v for k, v in obs.items()}
            r = engine.optimize_policy(
                {'satisfaction': {'target': 70, 'weight': sw},
                 'energy':       {'target': 15, 'weight': ew}},
                {}, sp)
            return {k.lower(): max(0.0, min(1.0, v))
                    for k, v in r.action_plan.items()}
        except Exception:
            return {v.lower(): 0.5 for v in cfg['action_vars']}

    return policy_fn


def run_episode(condition, eps, seed, n_steps):
    """One closed-loop episode. Returns a summary dict."""
    edges_mixed = EDGE_CONDITIONS[condition]
    random.seed(seed); np.random.seed(seed)

    engine = build_static_engine(edges_mixed)
    policy_fn = make_static_policy(engine, eps)

    state_phys = None
    sats, kwhs = [], []
    t0 = time.time()

    for t in range(n_steps):
        elapsed_ms = (t * 60) * 1000  # 1 minute per step

        if state_phys:
            sn = {k.lower(): phys_to_norm(v, k.lower(), smap)
                  for k, v in state_phys.items() if isinstance(v, (int, float))}
            so = {k: v for k, v in sn.items()}
            action = policy_fn(so, t)
        else:
            action = {v.lower(): 0.5 for v in cfg['action_vars']}

        step_kwh = sum(action.get(a, 0.5) * power_kw.get(a, 0) * STEP_HOURS
                       for a in action)
        kwhs.append(step_kwh)

        new_state = step_simulator(state_phys, action, elapsed_ms)
        if new_state:
            state_phys = new_state
            sat = state_phys.get('satisfaction',
                    state_phys.get('OverallSatisfaction',
                    state_phys.get('overallSatisfaction', 50)))
            sats.append(sat)

    return {
        'sim': SIM,
        'condition': condition,
        'epsilon': eps,
        'seed': seed,
        'n_steps': n_steps,
        'energy_kwh_total': float(np.sum(kwhs)),
        'energy_kwh_per_step': float(np.mean(kwhs)),
        'satisfaction_mean': float(np.mean(sats)) if sats else float('nan'),
        'satisfaction_min': float(np.min(sats)) if sats else float('nan'),
        'n_completed_steps': len(sats),
        'wall_time_s': round(time.time() - t0, 1),
    }


# ── Build the work list, dropping anything already in results.csv ──
all_jobs = [(cond, eps, seed)
            for cond in EDGE_CONDITIONS
            for eps in args.epsilons
            for seed in args.seeds]

done_keys = set()
if os.path.exists(RESULTS_CSV):
    df_existing = pd.read_csv(RESULTS_CSV)
    done_keys = set(zip(df_existing['condition'], df_existing['epsilon'],
                        df_existing['seed']))
    logger.info(f"Resuming: {len(done_keys)} episodes already in {RESULTS_CSV}")

todo = [j for j in all_jobs if j not in done_keys]
logger.info(f"Total jobs: {len(all_jobs)}, todo: {len(todo)}")

if not todo:
    logger.info("Nothing to do. Exiting.")
    sys.exit(0)


# ── Run episodes with ThreadPoolExecutor (subprocess.run releases GIL) ──
def submit_episode(job):
    cond, eps, seed = job
    return run_episode(cond, eps, seed, args.n_steps)


with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futures = {ex.submit(submit_episode, j): j for j in todo}
    for i, fut in enumerate(as_completed(futures), 1):
        job = futures[fut]
        try:
            row = fut.result()
        except Exception as e:
            logger.error(f"Episode {job} failed: {e}")
            continue
        # Append-after-each-episode for resume safety
        df_row = pd.DataFrame([row])
        header = not os.path.exists(RESULTS_CSV)
        df_row.to_csv(RESULTS_CSV, mode='a', header=header, index=False)
        logger.info(f"[{i}/{len(todo)}] {row['condition']:<10} "
                    f"ε={row['epsilon']:.1f} seed={row['seed']:<3} "
                    f"kWh={row['energy_kwh_total']:.2f} "
                    f"sat={row['satisfaction_mean']:.1f} "
                    f"({row['wall_time_s']:.0f}s)")


# ── Paired stats ──
df = pd.read_csv(RESULTS_CSV)

stats_path = f'{OUT_DIR}/paired_stats.txt'
with open(stats_path, 'w') as f:
    from io import StringIO
    out = StringIO()
    out.write("=" * 72 + "\n")
    out.write(f"CROSS-ENV CLOSEDLOOP — {SIM} — STATIC POLICY (NO MONITORING)\n")
    out.write("=" * 72 + "\n\n")

    for eps in sorted(df['epsilon'].unique()):
        sub = df[df['epsilon'] == eps]
        pivot = sub.pivot_table(index='seed', columns='condition',
                                values='energy_kwh_total', aggfunc='mean')
        out.write(f"\n── ε = {eps} ── (kWh total per episode)\n")
        out.write(pivot.to_string() + "\n")

        if 'Validated' in pivot.columns and 'Obs-Only' in pivot.columns:
            from scipy import stats
            diffs = pivot['Obs-Only'] - pivot['Validated']
            t, p_t = stats.ttest_rel(pivot['Obs-Only'], pivot['Validated'])
            W, p_w = stats.wilcoxon(pivot['Obs-Only'], pivot['Validated'])
            n_fav = int((diffs > 0).sum())
            out.write(f"\n  Validated vs Obs-Only (n={len(pivot)} paired seeds):\n")
            out.write(f"    Validated kWh: {pivot['Validated'].mean():.3f} "
                      f"± {pivot['Validated'].std():.3f}\n")
            out.write(f"    Obs-Only kWh:  {pivot['Obs-Only'].mean():.3f} "
                      f"± {pivot['Obs-Only'].std():.3f}\n")
            out.write(f"    Mean Δ (Obs - Val): {diffs.mean():+.3f} kWh\n")
            out.write(f"    Direction: {n_fav}/{len(diffs)} seeds favor Validated\n")
            out.write(f"    Paired t={t:.2f}, p={p_t:.4f}\n")
            out.write(f"    Wilcoxon W={W:.1f}, p={p_w:.4f}\n")

    f.write(out.getvalue())

logger.info(f"Wrote {stats_path}")
print(f"DONE {os.path.basename(sys.argv[0])} {time.strftime('%Y-%m-%d %H:%M:%S')}")
