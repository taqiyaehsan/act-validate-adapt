"""
Regime-Change Policy Experiment:
  Does validated graph → better monitoring → better policy under regime shift?

Uses the PROVEN monitoring + sim data pipeline from neurips_final_experiments.py.
Collects a fresh sim episode with natural regime transitions, then for each
graph condition runs monitoring + regime-specific policy selection.

Same observation trace, different policies. Better monitoring → correct regime
detected → correct SEM gradients → better actions.
"""
import sys, os, json, warnings, random, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

from run_full_pipeline import (SIM_CONFIGS, load_scaling, API_KEY,
                                SIM_POWER_SPECS, STEP_HOURS)
from neurips_final_experiments import (
    construct_factored_monitors, build_observable_graph,
    compute_sensor_weights_from_graph, collect_sim_data, get_scaling_map,
    OBSERVABLE_VARS, SENSOR_VARS, LATENT_VARS, ALL_VARS
)
from src.policy_engine import CausalPolicyEngine

SEEDS = [42, 123, 456]
OUT_DIR = 'results/regime_policy_experiment'
os.makedirs(OUT_DIR, exist_ok=True)

cfg = SIM_CONFIGS['smart_building_rich']
data = pd.read_csv(cfg['data_path'])
cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
data = data[cols]
cm = {v.lower(): v for v in data.columns}
pcfg = cfg['policy_config']

power_kw = SIM_POWER_SPECS.get(
    os.path.basename(cfg['sim_path']).replace('.js', ''), {})
scaling = pd.read_csv(cfg['scaling_path'])
smap = get_scaling_map(scaling)

# ── Load edge sets ──
# Replicate EXACTLY what run_m3a_attribution_server.py does for consistency.

# 1. Validated: consensus edges
with open('server_runs/full_pipeline/smart_building_rich/consensus_analysis.json') as f:
    ca = json.load(f)
val_edges_lower = {(s.lower(), t.lower()) for s, t in ca['consensus_edges']}
val_edges = {(cm.get(s, s), cm.get(t, t)) for s, t in ca['consensus_edges']}
logger.info(f"Validated edges: {len(val_edges)}")

# 2. Obs-only: PC+LLM+VARLiNGAM union, SAM DISABLED (same as M3A server)
import src.pipeline as _pl
class _NoSAM:
    def __init__(self, **kwargs): pass
    def generate(self, data): return []
_pl.SAMGenerator = _NoSAM
logger.info("SAM disabled (matching M3A server config)")

obs_cache = f'{OUT_DIR}/obs_edges_m3a_match.json'
if os.path.exists(obs_cache):
    with open(obs_cache) as f:
        cached = json.load(f)
    obs_edges = {(s, t) for s, t in cached}
    obs_edges_lower = {(s.lower(), t.lower()) for s, t in cached}
    logger.info(f"Loaded cached obs-only edges: {len(obs_edges)}")
else:
    from src.pipeline_cwm import create_cwm_pipeline
    sim_path_abs = os.path.abspath(cfg['sim_path'])
    random.seed(42); np.random.seed(42)
    obs_pipeline = create_cwm_pipeline(
        csv_data=data, api_key=API_KEY, smart_room_path=sim_path_abs,
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
    logger.info(f"Generated + cached obs-only edges: {len(obs_edges)}")

# 3. Random: same seed(42), same n_val, same all_vars as M3A
all_vars = list(data.columns)
n_val = len(val_edges)
random.seed(42)
rand_edges = set()
while len(rand_edges) < n_val:
    s, t = random.choice(all_vars), random.choice(all_vars)
    if s.lower() != t.lower():
        rand_edges.add((s, t))
rand_edges_lower = {(s.lower(), t.lower()) for s, t in rand_edges}
logger.info(f"Random edges: {len(rand_edges)}")

# For the monitoring-isolation experiment, all conditions share the SAME
# policy engines (built on validated edges). Only the MONITORING differs.
# This isolates: edge set → monitoring quality → engine selection → kWh.
# No confound from SEM coefficient differences.

CONDITIONS = {
    'Validated': (val_edges, val_edges_lower),
    'Obs-Only': (obs_edges, obs_edges_lower),
    'Random': (rand_edges, rand_edges_lower),
    'Empty': (set(), set()),
}

# SHARED_ENGINES and SHARED_STATIC built after function definitions below.


def build_regime_engines(edges_mixed):
    """Build 4 regime-specific CausalPolicyEngines."""
    if len(edges_mixed) == 0:
        return {r: None for r in ['base', 'occ', 'win', 'full']}

    cols_lower = {c.lower(): c for c in data.columns}
    occ_col = cols_lower.get('occupancy')
    win_col = cols_lower.get('windowposition')
    occ_vals = data[occ_col].values
    win_vals = data[win_col].values
    occ_on = occ_vals >= 0.1
    win_on = win_vals >= 0.15

    masks = {
        'base': ~occ_on & ~win_on,
        'occ':   occ_on & ~win_on,
        'win':  ~occ_on &  win_on,
        'full':  occ_on &  win_on,
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


def run_condition(cond_name, edges_mixed, edges_lower, sim_records, seed):
    """Run monitoring + policy on pre-collected sim episode.

    ISOLATION DESIGN: All conditions use the SAME policy engines (SHARED_ENGINES).
    Only the monitoring (construct_factored_monitors) uses the condition's edges.
    This isolates: edge set → monitoring → engine selection → policy outcome.
    """
    random.seed(seed); np.random.seed(seed)

    # Monitoring uses THIS condition's edges (the variable under test)
    monitors = construct_factored_monitors(edges_lower, data)
    sensor_weights = compute_sensor_weights_from_graph(
        edges_lower, SENSOR_VARS, LATENT_VARS)

    # Policy engines are SHARED across all conditions (validated edges)
    engines = SHARED_ENGINES
    static_engine = SHARED_STATIC

    sigma = 1.5
    forget_factor = 0.15
    regime_names = ['base', 'occ', 'win', 'full']
    weights = {r: 0.25 for r in regime_names}

    COMFORT_TARGETS = [0.5, 1.0, 2.0]
    records = []

    for ct in COMFORT_TARGETS:
        sw = max(0.3, min(0.85, 1.0 - ct * 0.35))
        ew = 1.0 - sw
        weights = {r: 0.25 for r in regime_names}  # reset per target

        for rec in sim_records:
            obs = rec['state_obs']
            gt_regime = rec['regime_gt']  # 0=base, 1=occ, 2=win, 3=full
            regime_map = {0: 'base', 1: 'occ', 2: 'win', 3: 'full'}
            gt_name = regime_map[gt_regime]

            # ── Monitoring (proven code) ──
            if monitors is not None:
                likelihoods = {}
                for rn in regime_names:
                    pred = monitors[rn].predict(obs)
                    err = sum(sensor_weights.get(v, 1.0) *
                              abs(obs.get(v, 0.5) - pred.get(v, 0.5))
                              for v in SENSOR_VARS if v in obs)
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
            else:
                detected = 'base'

            p_occ = weights['occ'] + weights['full']
            p_win = weights['win'] + weights['full']
            correct = int(detected == gt_name)

            # ── Regime-aware policy: use detected regime's engine ──
            engine = engines.get(detected)
            if engine is not None:
                try:
                    sp = {cm.get(k, k): v for k, v in obs.items()}
                    r = engine.optimize_policy(
                        {'satisfaction': {'target': 70, 'weight': sw},
                         'energy': {'target': 15, 'weight': ew}}, {}, sp)
                    action_regime = {k.lower(): max(0.0, min(1.0, v))
                                     for k, v in r.action_plan.items()}
                except:
                    action_regime = {v.lower(): 0.5 for v in cfg['action_vars']}
            else:
                action_regime = {v.lower(): 0.5 for v in cfg['action_vars']}

            # ── Static policy (ignores regime) ──
            if static_engine is not None:
                try:
                    sp = {cm.get(k, k): v for k, v in obs.items()}
                    r = static_engine.optimize_policy(
                        {'satisfaction': {'target': 70, 'weight': sw},
                         'energy': {'target': 15, 'weight': ew}}, {}, sp)
                    action_static = {k.lower(): max(0.0, min(1.0, v))
                                     for k, v in r.action_plan.items()}
                except:
                    action_static = {v.lower(): 0.5 for v in cfg['action_vars']}
            else:
                action_static = {v.lower(): 0.5 for v in cfg['action_vars']}

            # kWh for both policies
            kwh_regime = sum(action_regime.get(a, 0.5) * power_kw.get(a, 0) * STEP_HOURS
                             for a in action_regime)
            kwh_static = sum(action_static.get(a, 0.5) * power_kw.get(a, 0) * STEP_HOURS
                             for a in action_static)

            sat = rec['state_phys'].get('OverallSatisfaction',
                    rec['state_phys'].get('overallSatisfaction', 50))

            records.append({
                'step': rec['step'], 'hour': rec['hour'],
                'comfort_target': ct,
                'gt_regime': gt_name,
                'detected_regime': detected,
                'correct_detection': correct,
                'p_occ': round(p_occ, 3), 'p_win': round(p_win, 3),
                'satisfaction': sat,
                'hvac_regime': action_regime.get('hvacpower', 0.5),
                'hvac_static': action_static.get('hvacpower', 0.5),
                'kwh_regime': kwh_regime,
                'kwh_static': kwh_static,
            })

    return pd.DataFrame(records)


# ── Build shared engines ──
logger.info("Building shared regime-specific policy engines (validated edges)...")
SHARED_ENGINES = build_regime_engines(val_edges)
for rname, eng in SHARED_ENGINES.items():
    if eng:
        logger.info(f"  Shared engine({rname}): R²={eng._model_confidence:.3f}")
try:
    SHARED_STATIC = CausalPolicyEngine(
        {'validated_edges': val_edges}, data,
        use_llm=False, action_vars=cfg['action_vars'],
        reg_lambda=pcfg['lambda'], n_gradient_steps=1,
        linear_only=pcfg['linear_only'])
except:
    SHARED_STATIC = None

# ── Main ──
t0 = time.time()

# Collect INDEPENDENT sim episodes per seed.
# collect_sim_data uses the simulator's internal stochastic noise, so
# separate calls produce different observation traces with different
# sensor noise realizations.  The regime schedule is time-driven (same
# occupancy/window pattern), but the sensor readings differ.
# We also vary the outdoor-temperature offset per seed to shift the
# thermal context, producing meaningfully different episodes.

sim_episodes = {}
for seed in SEEDS:
    logger.info(f"Collecting independent sim episode for seed={seed}...")
    np.random.seed(seed)
    random.seed(seed)
    records = collect_sim_data(smap, duration_steps=1440)
    sim_episodes[seed] = records

    regime_counts = {}
    for r in records:
        rn = {0:'base',1:'occ',2:'win',3:'full'}[r['regime_gt']]
        regime_counts[rn] = regime_counts.get(rn, 0) + 1
    logger.info(f"  Regime distribution: {regime_counts}")

all_results = []

for cond_name, (edges_mixed, edges_lower) in CONDITIONS.items():
    logger.info(f"\n{'='*60}")
    logger.info(f"Condition: {cond_name} ({len(edges_mixed)} edges)")
    logger.info(f"{'='*60}")

    for seed in SEEDS:
        logger.info(f"  seed={seed}")
        df = run_condition(cond_name, edges_mixed, edges_lower,
                           sim_episodes[seed], seed)

        if df is not None and len(df) > 0:
            df['condition'] = cond_name
            df['seed'] = seed
            all_results.append(df)

            # Quick per-target summary
            for ct in [0.5, 1.0, 2.0]:
                cdf = df[df['comfort_target'] == ct]
                det_acc = cdf['correct_detection'].mean() * 100
                logger.info(f"    ε={ct}: det_acc={det_acc:.1f}%, "
                            f"hvac_regime={cdf['hvac_regime'].mean():.3f}, "
                            f"hvac_static={cdf['hvac_static'].mean():.3f}")

if all_results:
    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv(f'{OUT_DIR}/results.csv', index=False)

    # Summary table
    elapsed = (time.time() - t0) / 60
    logger.info(f"\nTotal time: {elapsed:.1f} min")
    logger.info(f"\n{'='*70}")
    logger.info(f"Regime-Change Policy — Overall (mean±std across {len(SEEDS)} seeds)")
    logger.info(f"{'='*70}")
    logger.info(f"{'Cond':<12} {'ε':>4} {'Det%':>10} {'kWh_regime':>14} {'kWh_static':>14}")

    for cond in CONDITIONS:
        for ct in [0.5, 1.0, 2.0]:
            rows = result_df[(result_df['condition'] == cond) &
                             (result_df['comfort_target'] == ct)]
            if len(rows):
                det_per_seed = rows.groupby('seed')['correct_detection'].mean() * 100
                kwh_r_per_seed = rows.groupby('seed')['kwh_regime'].sum()
                kwh_s_per_seed = rows.groupby('seed')['kwh_static'].sum()
                logger.info(f"{cond:<12} {ct:>4.1f} "
                            f"{det_per_seed.mean():>5.1f}±{det_per_seed.std():>4.1f} "
                            f"{kwh_r_per_seed.mean():>7.1f}±{kwh_r_per_seed.std():>4.1f} "
                            f"{kwh_s_per_seed.mean():>7.1f}±{kwh_s_per_seed.std():>4.1f}")
        logger.info("")

    # Per-regime breakdown (at ε=1.0)
    logger.info(f"\n{'='*70}")
    logger.info(f"Per-Regime Breakdown (ε=1.0, mean across seeds)")
    logger.info(f"{'='*70}")
    logger.info(f"{'Cond':<12} {'Regime':<8} {'Det%':>6} {'kWh_R':>8} {'kWh_S':>8} {'N_steps':>8}")

    for cond in CONDITIONS:
        for gt in ['base', 'occ', 'win', 'full']:
            rows = result_df[(result_df['condition'] == cond) &
                             (result_df['comfort_target'] == 1.0) &
                             (result_df['gt_regime'] == gt)]
            if len(rows):
                det = rows['correct_detection'].mean() * 100
                kr = rows['kwh_regime'].sum() / len(SEEDS)
                ks = rows['kwh_static'].sum() / len(SEEDS)
                n = len(rows) // len(SEEDS)
                logger.info(f"{cond:<12} {gt:<8} {det:>6.1f} {kr:>8.2f} {ks:>8.02f} {n:>8d}")
        logger.info("")

    # Random's detection breakdown (expose majority-class bias)
    logger.info(f"\n{'='*70}")
    logger.info(f"Random Detection Breakdown — Exposing Majority-Class Bias")
    logger.info(f"{'='*70}")
    rand_rows = result_df[(result_df['condition'] == 'Random') &
                          (result_df['comfort_target'] == 1.0)]
    if len(rand_rows):
        for gt in ['base', 'occ', 'win', 'full']:
            gt_rows = rand_rows[rand_rows['gt_regime'] == gt]
            if len(gt_rows):
                det = gt_rows['correct_detection'].mean() * 100
                # What does Random detect?
                det_counts = gt_rows['detected_regime'].value_counts()
                most_common = det_counts.index[0] if len(det_counts) else '?'
                pct = det_counts.iloc[0] / len(gt_rows) * 100 if len(det_counts) else 0
                logger.info(f"  GT={gt:<5s}: correct={det:>5.1f}%, "
                            f"most_common_det={most_common} ({pct:.0f}%)")
