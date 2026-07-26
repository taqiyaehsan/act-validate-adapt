"""Regime-aware vs static control on the SAME validated graph (AC meta-review point 2).

Isolates the contribution of regime inference to closed-loop control. All four
conditions use the identical validated edge set, identical engine hyperparameters,
identical objectives — the ONLY difference is how the control engine is chosen:

  Static      — one engine fit on all training rows; never switches (= PolicyGRID).
  RegimeAware — Bayesian monitor infers the regime; the detected regime's engine
                acts (= PolicyGRID-R mechanics, run_regime_policy_closedloop.py).
  Oracle      — the TRUE simulator regime selects the engine (upper bound: what
                perfect regime inference would buy).
  RandomSwitch— uniformly random regime each step (floor: switching without
                information).

If RegimeAware > Static and captures most of the Oracle-Static gap, the inferred
regime demonstrably improves control. Harness copied verbatim from the proven
run_regime_policy_closedloop.py (same sim stepping, monitoring, MO formula);
regime engines drop occupancy/windowposition from their fit data, so the Static
engine does too — the comparison is regime-conditioned coefficients + switching,
nothing else.

Usage:
  python runners/run_regime_vs_static_policy.py --smoke     # 1 cell, 120 steps
  python runners/run_regime_vs_static_policy.py             # full 4x3x3 grid
Results: results/regime_vs_static/results.csv + stats.txt + per-episode ts CSVs.
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
    get_scaling_map, SENSOR_VARS, LATENT_VARS
)
from src.policy_engine import CausalPolicyEngine

ap = argparse.ArgumentParser()
ap.add_argument('--smoke', action='store_true', help='1 seed, eps=1.0, 120 steps')
ap.add_argument('--steps', type=int, default=1440)
ap.add_argument('--seeds', default='42,123,456')
ap.add_argument('--conditions', default='Static,RegimeAware,Oracle,RandomSwitch')
ap.add_argument('--out', default='results/regime_vs_static')
args = ap.parse_args()

SEEDS = [int(s) for s in args.seeds.split(',')]
N_STEPS = args.steps
COMFORT_TARGETS = [0.5, 1.0, 2.0]
CONDITIONS = args.conditions.split(',')
if args.smoke:
    SEEDS, COMFORT_TARGETS, N_STEPS = [42], [1.0], 120
OUT_DIR = args.out
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

# ── Validated consensus edges (same source as run_regime_policy_closedloop) ──
_EDGE_PATHS = [
    'server_runs/full_pipeline/smart_building_rich/consensus_analysis.json',
    'results/full_pipeline/smart_building_rich/consensus_analysis.json',
]
ca = None
for p in _EDGE_PATHS:
    if os.path.exists(p):
        with open(p) as f:
            ca = json.load(f)
        logger.info(f"Validated consensus edges from {p}")
        break
if ca is None:
    sys.exit(f"No consensus_analysis.json found in: {_EDGE_PATHS}")
val_edges_lower = {(s.lower(), t.lower()) for s, t in ca['consensus_edges']}
val_edges = {(cm.get(s, s), cm.get(t, t)) for s, t in ca['consensus_edges']}
logger.info(f"Validated edge set: {len(val_edges)} edges")

REGIME_NAMES = ['base', 'occ', 'win', 'full']
_ENGINE_KWARGS = {'use_llm': False, 'action_vars': cfg['action_vars'],
                  'reg_lambda': pcfg['lambda'], 'n_gradient_steps': 1,
                  'linear_only': pcfg['linear_only']}
if pcfg.get('intervention_directions'):
    _ENGINE_KWARGS['intervention_directions'] = pcfg['intervention_directions']


def _drop_regime_cols(df):
    drop_cols = [c for c in df.columns
                 if c.lower() in ('occupancy', 'windowposition', 'windowopen')]
    return df.drop(columns=drop_cols, errors='ignore')


def build_regime_engines():
    """Verbatim from run_regime_policy_closedloop.build_regime_engines (Validated)."""
    cols_lower = {c.lower(): c for c in data.columns}
    occ_vals = data[cols_lower['occupancy']].values
    win_vals = data[cols_lower['windowposition']].values
    occ_on = occ_vals >= 0.1
    win_on = win_vals >= 0.15
    masks = {
        'base': ~occ_on & ~win_on, 'occ': occ_on & ~win_on,
        'win': ~occ_on & win_on, 'full': occ_on & win_on,
    }
    engines = {}
    for rname, mask in masks.items():
        data_r = data[mask]
        if len(data_r) < 20:
            data_r = data
        data_r = _drop_regime_cols(data_r)
        try:
            engines[rname] = CausalPolicyEngine(
                {'validated_edges': val_edges}, data_r, **_ENGINE_KWARGS)
        except Exception:
            engines[rname] = None
    return engines


def build_static_engine():
    """One engine, all rows, identical kwargs/column treatment to regime engines."""
    try:
        return CausalPolicyEngine(
            {'validated_edges': val_edges}, _drop_regime_cols(data), **_ENGINE_KWARGS)
    except Exception as e:
        logger.error(f"Static engine failed: {e}")
        return None


def step_simulator(state_phys, action, elapsed_ms, outdoor_offset=0, sim_seed=None):
    intervention = {}
    for var, val in action.items():
        phys_val = norm_to_phys(val, var.lower(), smap)
        js_name = cfg['js_var_map'].get(var.lower(), var)
        intervention[js_name] = phys_val
    cmd = ['node', sim_abs, '--single-step',
           '--elapsed-ms', str(int(elapsed_ms))]
    if sim_seed is not None:
        # Common random numbers: the same (experiment seed, timestep)-derived
        # simulator seed is passed to every policy condition, so exogenous
        # disturbances are matched across conditions and only the policy differs.
        cmd += ['--seed', str(sim_seed)]
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


def _true_regime(state_phys):
    """Same thresholds the harness uses for gt_regime scoring."""
    occ = state_phys.get('occupancy', 0)
    win = state_phys.get('windowPosition', state_phys.get('windowposition', 0))
    return {(0, 0): 'base', (1, 0): 'occ', (0, 1): 'win', (1, 1): 'full'
            }[(int(occ >= 1), int(win > 0.3))]


def run_closedloop(mode, seed, comfort_target, engines, static_engine, monitors,
                   sensor_weights):
    """mode in {'Static','RegimeAware','Oracle','RandomSwitch'}."""
    random.seed(seed); np.random.seed(seed)
    rng = random.Random(seed)

    sw = max(0.3, min(0.85, 1.0 - comfort_target * 0.35))
    ew = 1.0 - sw
    sigma = 1.5
    forget_factor = 0.15
    weights = {r: 0.25 for r in REGIME_NAMES}

    outdoor_offset = 6.0
    state_phys = None
    records = []

    for t in range(N_STEPS):
        elapsed_ms = t * 60000

        if state_phys:
            sn = {k.lower(): phys_to_norm(v, k.lower(), smap)
                  for k, v in state_phys.items() if isinstance(v, (int, float))}
            so = {k: v for k, v in sn.items() if k not in latent}

            # Monitor update runs in ALL conditions (identical, from the same
            # validated edges) so det_acc is logged uniformly; only RegimeAware
            # uses it to pick the engine.
            monitor_detected = 'base'
            if monitors is not None:
                likelihoods = {}
                for rn in REGIME_NAMES:
                    pred = monitors[rn].predict(so)
                    err = sum(sensor_weights.get(v, 1.0) *
                              abs(so.get(v, 0.5) - pred.get(v, 0.5))
                              for v in SENSOR_VARS if v in so)
                    likelihoods[rn] = np.exp(-sigma * err)
                unnorm = {r: likelihoods[r] * weights[r] for r in REGIME_NAMES}
                total = sum(unnorm.values())
                if total > 0:
                    raw = {r: unnorm[r] / total for r in REGIME_NAMES}
                    weights = {r: (1 - forget_factor) * raw[r] + forget_factor * 0.25
                               for r in REGIME_NAMES}
                w_total = sum(weights.values())
                if w_total > 0:
                    weights = {r: weights[r] / w_total for r in REGIME_NAMES}
                monitor_detected = max(weights, key=weights.get)

            # Engine selection — the ONLY thing that differs between conditions
            if mode == 'Static':
                engine, engine_regime = static_engine, 'static'
            elif mode == 'RegimeAware':
                engine_regime = monitor_detected
                engine = engines.get(engine_regime)
            elif mode == 'Oracle':
                engine_regime = _true_regime(state_phys)
                engine = engines.get(engine_regime)
            else:  # RandomSwitch
                engine_regime = rng.choice(REGIME_NAMES)
                engine = engines.get(engine_regime)

            if engine is not None:
                try:
                    sp = {cm.get(k, k): v for k, v in so.items()}
                    r = engine.optimize_policy(
                        {'satisfaction': {'target': 70, 'weight': sw},
                         'energy': {'target': 15, 'weight': ew}}, {}, sp)
                    action = {k.lower(): max(0.0, min(1.0, v))
                              for k, v in r.action_plan.items()}
                except Exception:
                    action = {v.lower(): 0.5 for v in cfg['action_vars']}
            else:
                action = {v.lower(): 0.5 for v in cfg['action_vars']}
        else:
            action = {v.lower(): 0.5 for v in cfg['action_vars']}
            monitor_detected, engine_regime = 'base', 'base'

        step_kwh = sum(action.get(a, 0.5) * power_kw.get(a, 0) * STEP_HOURS
                       for a in action)

        new_state = step_simulator(state_phys, action, elapsed_ms, outdoor_offset,
                                   sim_seed=seed * 100003 + t)
        if new_state:
            state_phys = new_state
            gt_regime = _true_regime(state_phys)
            sat = state_phys.get('satisfaction',
                    state_phys.get('OverallSatisfaction',
                    state_phys.get('overallSatisfaction', 50)))
            records.append({
                'step': t, 'hour': elapsed_ms / 3600000,
                'gt_regime': gt_regime,
                'monitor_detected': monitor_detected,
                'engine_regime': engine_regime,
                'monitor_correct': int(monitor_detected == gt_regime),
                'satisfaction': sat,
                'kwh_step': step_kwh,
                'hvac': action.get('hvacpower', 0.5),
                'light': action.get('lightingpower', 0.5),
            })

    return pd.DataFrame(records)


# ── Main ──
t0 = time.time()
logger.info("Building shared components (monitors, regime engines, static engine)...")
monitors = construct_factored_monitors(val_edges_lower, data)
sensor_weights = compute_sensor_weights_from_graph(
    val_edges_lower, SENSOR_VARS, LATENT_VARS)
engines = build_regime_engines()
static_engine = build_static_engine()
logger.info(f"Engines ready: regime={[r for r, e in engines.items() if e]}, "
            f"static={'ok' if static_engine else 'FAILED'}")

all_results = []
for cond in CONDITIONS:
    logger.info(f"\n{'='*60}\nCondition: {cond}\n{'='*60}")
    for seed in SEEDS:
        for ct in COMFORT_TARGETS:
            logger.info(f"  seed={seed}, eps={ct}")
            df = run_closedloop(cond, seed, ct, engines, static_engine,
                                monitors, sensor_weights)
            if df is None or len(df) == 0:
                logger.warning("    EMPTY episode, skipping")
                continue
            det_acc = df['monitor_correct'].mean() * 100
            total_kwh = df['kwh_step'].sum()
            mean_sat = df['satisfaction'].mean()
            kwh_max = sum(power_kw.values()) * N_STEPS * STEP_HOURS
            sw = max(0.3, min(0.85, 1.0 - ct * 0.35))
            ew = 1.0 - sw
            mo = sw * (mean_sat / 100) + ew * (1 - total_kwh / max(kwh_max, 1e-6))
            all_results.append({
                'condition': cond, 'seed': seed, 'comfort_target': ct,
                'det_acc': round(det_acc, 1), 'kWh': round(total_kwh, 2),
                'satisfaction': round(mean_sat, 2), 'MO': round(mo, 4),
                'mean_hvac': round(df['hvac'].mean(), 3),
            })
            logger.info(f"    det={det_acc:.1f}% kWh={total_kwh:.1f} "
                        f"sat={mean_sat:.1f} MO={mo:.3f}")
            df.to_csv(f'{OUT_DIR}/{cond}_seed{seed}_ct{ct}_ts.csv', index=False)

result_df = pd.DataFrame(all_results)
result_df.to_csv(f'{OUT_DIR}/results.csv', index=False)
logger.info(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")

# ── Summary + paired stats ──
lines = []
lines.append(f"Regime-aware vs static control, SAME validated graph "
             f"({len(SEEDS)} seeds x {len(COMFORT_TARGETS)} eps, {N_STEPS} steps)")
lines.append(f"{'Cond':<14} {'eps':>4} {'Det%':>11} {'kWh':>13} {'Sat':>13} {'MO':>14}")
for cond in CONDITIONS:
    for ct in COMFORT_TARGETS:
        rows = result_df[(result_df['condition'] == cond) &
                         (result_df['comfort_target'] == ct)]
        if len(rows):
            lines.append(
                f"{cond:<14} {ct:>4.1f} "
                f"{rows['det_acc'].mean():>6.1f}±{rows['det_acc'].std():>4.1f} "
                f"{rows['kWh'].mean():>7.1f}±{rows['kWh'].std():>4.1f} "
                f"{rows['satisfaction'].mean():>7.1f}±{rows['satisfaction'].std():>4.1f} "
                f"{rows['MO'].mean():>8.4f}±{rows['MO'].std():>5.4f}")
    lines.append("")

pair_conds = [c for c in ('RegimeAware', 'Static', 'Oracle') if c in CONDITIONS]
if 'RegimeAware' in pair_conds and 'Static' in pair_conds and len(SEEDS) > 1:
    from scipy import stats as sps
    piv = result_df.pivot_table(index=['seed', 'comfort_target'],
                                columns='condition', values='MO')
    if {'RegimeAware', 'Static'}.issubset(piv.columns):
        d = (piv['RegimeAware'] - piv['Static']).dropna()
        t, p = sps.ttest_rel(piv.loc[d.index, 'RegimeAware'],
                             piv.loc[d.index, 'Static'])
        lines.append(f"Paired MO (RegimeAware - Static): "
                     f"Δ={d.mean():+.4f}±{d.std():.4f}, t={t:.2f}, p={p:.4g}, "
                     f"favor={int((d > 0).sum())}/{len(d)} cells")
    if 'Oracle' in piv.columns and {'RegimeAware', 'Static'}.issubset(piv.columns):
        gap_o = (piv['Oracle'] - piv['Static']).mean()
        gap_r = (piv['RegimeAware'] - piv['Static']).mean()
        if abs(gap_o) > 1e-9:
            lines.append(f"Oracle-gap capture: RegimeAware achieves "
                         f"{100 * gap_r / gap_o:.1f}% of the Oracle-Static MO gap")
    pivk = result_df.pivot_table(index=['seed', 'comfort_target'],
                                 columns='condition', values='kWh')
    if {'RegimeAware', 'Static'}.issubset(pivk.columns):
        dk = (pivk['RegimeAware'] - pivk['Static']).dropna()
        lines.append(f"Paired kWh (RegimeAware - Static): Δ={dk.mean():+.2f}±{dk.std():.2f}")

report = '\n'.join(lines)
logger.info('\n' + '=' * 70 + '\n' + report)
with open(f'{OUT_DIR}/stats.txt', 'w') as f:
    f.write(report + '\n')
logger.info(f"Saved {OUT_DIR}/results.csv + stats.txt")
