"""
P2 — Active-vs-passive probing ablation at regime transitions (rebuttal T7).

The paper (Appendix D.2) reports active probing changes nothing at steady state
(Δ=0.00 pp) and states the transition isolation "was not run for this
submission." This runner closes that gap: controlled regime-shift episodes
(pre-regime held N steps, post-regime held M steps, forced via per-step sim
interventions), monitored twice over the *same* records — once with the probe
mechanism enabled (virtual-sensor path of run_monitoring_active), once passive.
Arms differ only in the probe block; the Bayesian update is byte-identical.

Transitions mirror Appendix Table 19 (Base→Occ, Base→Win, Base→Full,
Occ→Base, Full→Base). The probe trigger (sustained believed-unoccupied ≥ 120
steps) can only fire while the posterior says "empty", so transitions into
occupied regimes exercise pre-shift probes and unoccupied-side transitions
(Base→Win, →Base recoveries) exercise post-shift probes — report both counts.

Metrics per (episode, σ/η config, arm):
  - joint argmax delay + joint 0.7-sustained lock-in delay (Table 18 style)
  - per-dimension argmax + lock-in delays (occ: ≥0.7 if occupied / ≤0.3 if not)
  - post-shift joint accuracy; probes fired pre/post shift

Outputs:
  results/probe_ablation/results.csv     one row per episode × config × arm
  results/probe_ablation/probe_log.csv   every probe fired
  results/probe_ablation/stats.txt       paired active-vs-passive summary

Usage:
  python runners/run_probe_ablation.py                # 5 transitions × 3 reps × 2 configs
  python runners/run_probe_ablation.py --smoke        # 1 transition, 1 rep, short phases
  python runners/run_probe_ablation.py --workers 5
"""
import sys, os, json, time, warnings, argparse, subprocess
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

from neurips_final_experiments import (
    construct_factored_monitors, design_diagnostic_probe, execute_probe,
    compute_regime_decay_predictions, get_scaling_map, phys_to_norm,
    LATENT_VARS, SENSOR_VARS, SIM_PATH,
)

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--replicates', type=int, default=3,
                    help='Sim-noise replicates per transition (default 3).')
parser.add_argument('--pre-steps', type=int, default=200,
                    help='Steps in the pre-shift regime (default 200, as Table 18).')
parser.add_argument('--post-steps', type=int, default=400,
                    help='Steps in the post-shift regime (default 400: room for '
                         'the 120-step probe trigger + 60-step cooldown).')
parser.add_argument('--configs', type=str, default='2.0:0.12,1.5:0.15',
                    help='Comma-separated sigma:eta pairs (default: Table-18 '
                         'default and deployed values).')
parser.add_argument('--transitions', type=str, default='all',
                    help='Comma-separated from>to pairs (e.g. base>win,full>base) '
                         'or "all" for the Table-19 five.')
parser.add_argument('--workers', type=int, default=5,
                    help='Parallel episode-collection workers (default 5).')
parser.add_argument('--smoke', action='store_true',
                    help='Quick check: base>win, 1 replicate, 150+150 steps, one config.')
args = parser.parse_args()

if args.smoke:
    args.transitions = 'base>win'
    args.replicates = 1
    args.pre_steps = 150
    args.post_steps = 150
    args.configs = '2.0:0.12'

OUT_DIR = 'results/probe_ablation'
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = [tuple(float(x) for x in c.split(':')) for c in args.configs.split(',')]

REGIME_FORCE = {
    'base': {'occupancy': 0, 'windowPosition': 0.05},
    'occ':  {'occupancy': 4, 'windowPosition': 0.05},
    'win':  {'occupancy': 0, 'windowPosition': 0.60},
    'full': {'occupancy': 4, 'windowPosition': 0.60},
}
TABLE19_TRANSITIONS = [('base', 'occ'), ('base', 'win'), ('base', 'full'),
                       ('occ', 'base'), ('full', 'base')]
if args.transitions == 'all':
    TRANSITIONS = TABLE19_TRANSITIONS
else:
    TRANSITIONS = [tuple(t.split('>')) for t in args.transitions.split(',')]

REGIME_NAMES = ['base', 'occ', 'win', 'full']
GT_NAME = {0: 'base', 1: 'occ', 2: 'win', 3: 'full'}

OUTDOOR_OFFSET = -3.0         # cold: open window has a thermal signature the
                              # monitor can observe (mild +6 makes indoor ≈
                              # outdoor and the window shift is undetectable
                              # for BOTH arms — verified in smoke)
START_HOUR = 9.0              # daytime start

# Probe-mechanism constants — verbatim from run_monitoring_active
PROBE_INTERVAL = 120
PROBE_COOLDOWN = 60
MAX_PROBES = 50
HOLD_DURATION = 30
VIRTUAL_WEIGHT = 1.0
PROBE_SIGMA = 3.0

# ── Load training data + frozen validated edges, build monitors ─────────────
DATA_PATH = 'data_regen/smart_building_rich_processed.csv'
SCALING_PATH = 'data_regen/smart_building_rich_processed_scaling.csv'
EDGE_SETS_PATH = 'results/closedloop_server/edge_sets.json'

data = pd.read_csv(DATA_PATH)
cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
data = data[cols]
smap = get_scaling_map(pd.read_csv(SCALING_PATH))

with open(EDGE_SETS_PATH) as f:
    val_edges = {(s.lower(), t.lower()) for s, t in json.load(f)['validated']}
logger.info(f"Frozen validated edges: {len(val_edges)}")

monitors = construct_factored_monitors(val_edges, data)
if monitors is None:
    logger.error("Monitor construction failed"); sys.exit(1)
logger.info("Monitors built (base/occ/win/full)")


# ── Controlled transition episode ───────────────────────────────────────────
def collect_transition_episode(pre_regime, post_regime, pre_steps, post_steps,
                               replicate):
    """Force pre_regime for pre_steps then post_regime for post_steps.

    Regime variables are pinned every step via the sim's do-operator; GT labels
    are read back from the realized state so any forcing slippage is reflected
    in the labels rather than hidden.
    """
    sim_abs = os.path.abspath(SIM_PATH)
    state = None
    elapsed_ms = int(START_HOUR * 3600000)
    step_ms = 60000
    records = []
    total = pre_steps + post_steps

    for step_i in range(total):
        elapsed_ms += step_ms
        hour = elapsed_ms / 3600000
        regime = pre_regime if step_i < pre_steps else post_regime
        intervention = dict(REGIME_FORCE[regime])

        cmd = ['node', sim_abs, '--single-step',
               '--elapsed-ms', str(elapsed_ms),
               '--outdoor-offset', str(OUTDOOR_OFFSET),
               '--intervention', json.dumps(intervention)]
        if state:
            cmd += ['--state', json.dumps(state)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except Exception:
            continue

        for line in result.stdout.split('\n'):
            if 'RESULT:' in line:
                state = json.loads(line.split('RESULT:', 1)[1].strip())
                break
        else:
            continue

        state_norm = {}
        for k, v in state.items():
            if not isinstance(v, (int, float)):
                continue
            kl = k.lower()
            state_norm[kl] = phys_to_norm(v, kl, smap)

        occ = state.get('occupancy', 0)
        win = state.get('windowPosition', state.get('windowposition', 0))
        occ_active = int(occ >= 1)
        win_active = int(win > 0.3)

        records.append({
            'step': step_i,
            'hour': round(hour, 4),
            'occ_active': occ_active,
            'win_active': win_active,
            'regime_gt': occ_active * 1 + win_active * 2,
            'state_obs': {k: v for k, v in state_norm.items()
                          if k not in LATENT_VARS},
            'state_phys': dict(state),
        })

    # Forcing sanity: realized regime must match intent in each phase
    for phase, lo, hi in [(pre_regime, 0, pre_steps),
                          (post_regime, pre_steps, total)]:
        want_occ = int(REGIME_FORCE[phase]['occupancy'] >= 1)
        want_win = int(REGIME_FORCE[phase]['windowPosition'] > 0.3)
        ph = [r for r in records if lo <= r['step'] < hi]
        if ph:
            slip = np.mean([(r['occ_active'] != want_occ) or
                            (r['win_active'] != want_win) for r in ph])
            if slip > 0.10:
                logger.warning(f"  {pre_regime}>{post_regime} rep{replicate}: "
                               f"{slip:.0%} forcing slippage in {phase} phase")
    return records


# ── Bayesian monitoring loop — verbatim port of run_monitoring_active with a
#    probe_enabled gate; passive arm is the identical loop minus the probe block.
def run_bayes(sim_records, sigma, eta, probe_enabled):
    weights = {r: 0.25 for r in REGIME_NAMES}
    records, probe_log = [], []
    steps_since_probe = 0
    n_probes = 0
    unoccupied_streak = 0
    virtual_sensor = None
    virtual_remaining = 0

    for i, rec in enumerate(sim_records):
        state_obs = rec['state_obs']

        likelihoods = {}
        for rname in REGIME_NAMES:
            pred = monitors[rname].predict(state_obs)
            err = sum(abs(state_obs.get(v, 0.5) - pred.get(v, 0.5))
                      for v in SENSOR_VARS)
            likelihoods[rname] = np.exp(-sigma * err)

            if virtual_sensor is not None and virtual_remaining > 0:
                decay_err = abs(virtual_sensor['observed'] -
                                virtual_sensor['predicted'].get(rname, 0))
                likelihoods[rname] *= (np.exp(-PROBE_SIGMA * decay_err)
                                       ** VIRTUAL_WEIGHT)

        unnorm = {r: likelihoods[r] * weights[r] for r in REGIME_NAMES}
        total = sum(unnorm.values())
        if total > 0:
            raw = {r: unnorm[r] / total for r in REGIME_NAMES}
            weights = {r: (1 - eta) * raw[r] + eta * 0.25 for r in REGIME_NAMES}
        w_total = sum(weights.values())
        if w_total > 0:
            weights = {r: weights[r] / w_total for r in REGIME_NAMES}

        if virtual_remaining > 0:
            virtual_remaining -= 1
            if virtual_remaining == 0:
                virtual_sensor = None

        p_occ = weights['occ'] + weights['full']
        p_win = weights['win'] + weights['full']

        believed_unoccupied = p_occ < 0.3
        unoccupied_streak = unoccupied_streak + 1 if believed_unoccupied else 0

        should_probe = (
            probe_enabled and
            believed_unoccupied and
            unoccupied_streak >= PROBE_INTERVAL and
            steps_since_probe >= PROBE_COOLDOWN and
            n_probes < MAX_PROBES and
            'state_phys' in rec
        )

        probe_applied = False
        if should_probe:
            probe_plan = design_diagnostic_probe(p_occ, p_win,
                                                 rec['state_phys'],
                                                 gpt_client=None)
            probe_result = execute_probe(
                rec['state_phys'], probe_plan, smap,
                elapsed_ms=int(rec['hour'] * 3600000),
                outdoor_offset=OUTDOOR_OFFSET)

            trajectories = probe_result['phase_trajectories']
            observed_decay = 0.0
            if len(trajectories) >= 2 and trajectories[0] and trajectories[1]:
                peak_temp = trajectories[0][-1]
                final_temp = trajectories[1][-1]
                n_steps = len(trajectories[1])
                if n_steps > 0:
                    observed_decay = (peak_temp - final_temp) / n_steps

            pre_obs = {k: v for k, v in probe_result['pre_norm'].items()
                       if k not in LATENT_VARS}
            predicted_decays = compute_regime_decay_predictions(
                monitors, pre_obs, smap)

            max_decay = max(abs(observed_decay),
                            max(abs(v) for v in predicted_decays.values()),
                            0.01)
            virtual_sensor = {
                'observed': observed_decay / max_decay,
                'predicted': {r: predicted_decays[r] / max_decay
                              for r in REGIME_NAMES},
            }
            virtual_remaining = HOLD_DURATION

            probe_log.append({
                'step': i,
                'observed_decay': round(observed_decay, 4),
                'gt_occ': rec['occ_active'], 'gt_win': rec['win_active'],
            })
            steps_since_probe = 0
            unoccupied_streak = 0
            n_probes += 1
            probe_applied = True
        else:
            steps_since_probe += 1

        records.append({
            'step': rec['step'],
            'occ_active': rec['occ_active'], 'win_active': rec['win_active'],
            'regime_gt': rec['regime_gt'],
            'P_occ': p_occ, 'P_win': p_win,
            'w_gt': weights[GT_NAME[rec['regime_gt']]],
            'winner': max(weights, key=weights.get),
            'probe': probe_applied,
        })

    return records, probe_log


# ── Metrics ─────────────────────────────────────────────────────────────────
def _sustained_first(flags, hold=30):
    """First index where flags is True and stays True for `hold` consecutive
    steps (truncated at the end of the sequence)."""
    n = len(flags)
    for i in range(n):
        window = flags[i:min(i + hold, n)]
        if len(window) > 0 and all(window):
            return i
    return None


def compute_metrics(mon_records, shift_step, post_steps):
    post = [r for r in mon_records if r['step'] >= shift_step]
    if not post:
        return {}
    gt = post[0]
    gt_name = GT_NAME[gt['regime_gt']]
    cap = post_steps

    def delay(cond):
        for j, r in enumerate(post):
            if cond(r):
                return j
        return cap

    m = {
        'joint_argmax_delay': delay(lambda r: r['winner'] == gt_name),
        'occ_argmax_delay': delay(lambda r: (r['P_occ'] > 0.5) == bool(gt['occ_active'])),
        'win_argmax_delay': delay(lambda r: (r['P_win'] > 0.5) == bool(gt['win_active'])),
        'post_shift_acc': float(np.mean(
            [r['winner'] == GT_NAME[r['regime_gt']] for r in post]) * 100),
        'post_occ_acc': float(np.mean(
            [(r['P_occ'] > 0.5) == bool(r['occ_active']) for r in post]) * 100),
        'post_win_acc': float(np.mean(
            [(r['P_win'] > 0.5) == bool(r['win_active']) for r in post]) * 100),
    }

    jl = _sustained_first([r['w_gt'] >= 0.7 for r in post])
    m['joint_lockin_delay'] = jl if jl is not None else cap

    occ_ok = [(r['P_occ'] >= 0.7) if gt['occ_active'] else (r['P_occ'] <= 0.3)
              for r in post]
    win_ok = [(r['P_win'] >= 0.7) if gt['win_active'] else (r['P_win'] <= 0.3)
              for r in post]
    ol = _sustained_first(occ_ok)
    wl = _sustained_first(win_ok)
    m['occ_lockin_delay'] = ol if ol is not None else cap
    m['win_lockin_delay'] = wl if wl is not None else cap
    return m


# ── Main loop ───────────────────────────────────────────────────────────────
episodes = {}     # (transition, rep) -> records
jobs = [(tr, rep) for tr in TRANSITIONS for rep in range(args.replicates)]

logger.info(f"Collecting {len(jobs)} episodes "
            f"({args.pre_steps}+{args.post_steps} steps each, "
            f"workers={args.workers})...")
t0 = time.time()
with ThreadPoolExecutor(max_workers=args.workers) as ex:
    futs = {ex.submit(collect_transition_episode, tr[0], tr[1],
                      args.pre_steps, args.post_steps, rep): (tr, rep)
            for tr, rep in jobs}
    for fut in as_completed(futs):
        tr, rep = futs[fut]
        episodes[(tr, rep)] = fut.result()
        logger.info(f"  collected {tr[0]}>{tr[1]} rep{rep} "
                    f"({len(episodes[(tr, rep)])} steps)")
logger.info(f"Collection done in {time.time()-t0:.0f}s")

rows, probe_rows = [], []
for (tr, rep), recs in sorted(episodes.items()):
    if not recs:
        logger.warning(f"  empty episode {tr} rep{rep}, skipping")
        continue
    for sigma, eta in CONFIGS:
        for arm in ['passive', 'active']:
            t1 = time.time()
            mon, plog = run_bayes(recs, sigma, eta,
                                  probe_enabled=(arm == 'active'))
            m = compute_metrics(mon, args.pre_steps, args.post_steps)
            n_pre = sum(1 for p in plog if p['step'] < args.pre_steps)
            row = {
                'transition': f"{tr[0]}>{tr[1]}", 'replicate': rep,
                'sigma': sigma, 'eta': eta, 'arm': arm,
                'probes_pre': n_pre, 'probes_post': len(plog) - n_pre,
                **m,
            }
            rows.append(row)
            for p in plog:
                probe_rows.append({'transition': row['transition'],
                                   'replicate': rep, 'sigma': sigma,
                                   'eta': eta, 'arm': arm, **p})
            logger.info(f"  {row['transition']} rep{rep} σ={sigma} η={eta} "
                        f"{arm:7s}: joint_lockin={row.get('joint_lockin_delay')} "
                        f"occ={row.get('occ_lockin_delay')} "
                        f"win={row.get('win_lockin_delay')} "
                        f"acc={row.get('post_shift_acc', 0):.1f}% "
                        f"probes={len(plog)} ({time.time()-t1:.0f}s)")

df = pd.DataFrame(rows)
df.to_csv(f'{OUT_DIR}/results.csv', index=False)
if probe_rows:
    pd.DataFrame(probe_rows).to_csv(f'{OUT_DIR}/probe_log.csv', index=False)

# ── Paired summary ──────────────────────────────────────────────────────────
lines = ['═' * 72,
         'ACTIVE-vs-PASSIVE PROBING AT REGIME TRANSITIONS (paired episodes)',
         f'pre={args.pre_steps} post={args.post_steps} steps; '
         f'replicates={args.replicates}; frozen validated edges',
         '═' * 72, '']
METRICS = ['joint_lockin_delay', 'occ_lockin_delay', 'win_lockin_delay',
           'joint_argmax_delay', 'post_shift_acc', 'post_occ_acc', 'post_win_acc']
for sigma, eta in CONFIGS:
    lines.append(f'── σ={sigma}, η={eta} ──')
    sub = df[(df.sigma == sigma) & (df.eta == eta)]
    for trn in sub.transition.unique():
        s = sub[sub.transition == trn]
        act = s[s.arm == 'active'].sort_values('replicate')
        pas = s[s.arm == 'passive'].sort_values('replicate')
        lines.append(f'  {trn}  (probes/eps active: '
                     f'pre={act.probes_pre.mean():.1f} '
                     f'post={act.probes_post.mean():.1f})')
        for met in METRICS:
            a, p = act[met].values, pas[met].values
            n = min(len(a), len(p))
            if n == 0:
                continue
            d = a[:n] - p[:n]
            lines.append(f'    {met:22s} active {np.mean(a):7.1f}  '
                         f'passive {np.mean(p):7.1f}  Δ(a−p) {np.mean(d):+7.1f}'
                         f'  per-rep Δ {[round(float(x), 1) for x in d]}')
        lines.append('')
stats_txt = '\n'.join(lines)
with open(f'{OUT_DIR}/stats.txt', 'w') as f:
    f.write(stats_txt)
print('\n' + stats_txt)
logger.info(f"Wrote {OUT_DIR}/results.csv, probe_log.csv, stats.txt")
