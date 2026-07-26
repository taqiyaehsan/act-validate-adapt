#!/usr/bin/env python3
"""
Quick A/B/C/D comparison: signed-only vs OLS back-door vs RF back-door vs intervention-regression.
Runs smart_building_rich with 1 seed, 5 iterations. Monkey-patches _calculate_effect_size
for each method so the full pipeline runs identically except for the effect estimator.
"""

import sys, os, json, time, random, copy
import numpy as np
import pandas as pd
import logging
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline_cwm import create_cwm_pipeline

logging.basicConfig(
    level=logging.WARNING,  # quiet — we only want the summary
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_BASE_DIR, 'ground_truth_graphs.json')) as f:
    ALL_GT = json.load(f)

SIM_CFG = {
    'sim_path': 'js/smart_building_rich.js',
    'obs_data_path': 'data_regen/smart_building_rich_processed.csv',
    'dataset_type': 'smart_building_rich',
    'gt_key': 'smart_building_rich',
    'actuator_vars': ['hvacpower', 'lightingpower'],
    'non_intervenable_vars': [
        'outdoortemp', 'solarradiation', 'occupancy', 'windowposition',
        'pmv', 'energyconsumption', 'satisfaction', 'overallsatisfaction',
    ],
}

API_KEY = 'YOUR_OPENAI_API_KEY'
SEED = 42
MAX_ITER = 5


def compute_metrics(discovered_edges, gt_key):
    gt_edges = {(s.lower(), t.lower()) for s, t in ALL_GT[gt_key]['edges']}
    disc = {(str(s).lower(), str(t).lower()) for s, t in discovered_edges}
    tp = len(disc & gt_edges)
    fp = len(disc - gt_edges)
    fn = len(gt_edges - disc)
    shd = fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'shd': shd, 'n_gt': len(gt_edges), 'n_disc': len(disc),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1}


# ── Effect-size methods ──────────────────────────────────────────────

def signed_only(tester, edge, result):
    """Method A: raw signed effect (current working approach)."""
    source, target = edge
    sv = tester.variable_mapping.get(source.lower(), source)
    tv = tester.variable_mapping.get(target.lower(), target)
    return tester.edge_validator.validate_edge_changes(
        result['pre_state'], result['post_state'], sv, tv)


def ols_backdoor(tester, edge, result):
    """Method B: OLS back-door adjustment."""
    source, target = edge
    sv = tester.variable_mapping.get(source.lower(), source)
    tv = tester.variable_mapping.get(target.lower(), target)
    obs_data = getattr(tester, '_obs_data', None)
    if obs_data is not None:
        adj = _ols_adjusted(tester, sv, tv,
                            result['pre_state'], result['post_state'], obs_data)
        if adj is not None:
            return adj
    return tester.edge_validator.validate_edge_changes(
        result['pre_state'], result['post_state'], sv, tv)


def rf_backdoor(tester, edge, result):
    """Method C: Random Forest back-door (Double ML style)."""
    source, target = edge
    sv = tester.variable_mapping.get(source.lower(), source)
    tv = tester.variable_mapping.get(target.lower(), target)
    obs_data = getattr(tester, '_obs_data', None)
    if obs_data is not None:
        adj = _rf_adjusted(tester, sv, tv,
                           result['pre_state'], result['post_state'], obs_data)
        if adj is not None:
            return adj
    return tester.edge_validator.validate_edge_changes(
        result['pre_state'], result['post_state'], sv, tv)


def rf_plus_signed(tester, edge, result):
    """Method E: RF back-door + raw signed, averaged.
    Uses obs data (RF residualization) AND intervention data (raw signed)
    together. RF removes confounder bias, raw captures full signal.
    When RF can't compute (missing keys etc), falls back to raw only."""
    source, target = edge
    sv = tester.variable_mapping.get(source.lower(), source)
    tv = tester.variable_mapping.get(target.lower(), target)
    raw = tester.edge_validator.validate_edge_changes(
        result['pre_state'], result['post_state'], sv, tv)
    obs_data = getattr(tester, '_obs_data', None)
    if obs_data is not None:
        adj = _rf_adjusted(tester, sv, tv,
                           result['pre_state'], result['post_state'], obs_data)
        if adj is not None and raw is not None:
            # Average: RF-adjusted (obs-based deconfounding) + raw signed (intervention signal)
            return 0.5 * adj + 0.5 * raw
    return raw


def intervention_regression(tester, edge, result):
    """Method D: collect raw changes, do regression at confidence-computation time.
    For the per-run call, just return the raw signed effect. The regression
    happens in a patched _compute_statistical_confidence."""
    source, target = edge
    sv = tester.variable_mapping.get(source.lower(), source)
    tv = tester.variable_mapping.get(target.lower(), target)
    raw = tester.edge_validator.validate_edge_changes(
        result['pre_state'], result['post_state'], sv, tv)

    # Also stash confounder changes for later regression
    find_key = tester.edge_validator._find_key
    pre, post = result['pre_state'], result['post_state']
    confounder_names = ['occupancy', 'windowposition']
    conf_deltas = {}
    for cn in confounder_names:
        k = find_key(pre, cn)
        if k:
            rng = tester.edge_validator._RANGES.get(cn, 1.0)
            mn = tester.edge_validator._MINS.get(cn, 0.0)
            pre_n = (float(pre[k]) - mn) / max(rng, 1e-10)
            post_n = (float(post[k]) - mn) / max(rng, 1e-10)
            conf_deltas[cn] = post_n - pre_n
    if not hasattr(tester, '_interv_run_data'):
        tester._interv_run_data = {}
    tester._interv_run_data.setdefault(edge, []).append({
        'target_effect': raw,
        'conf_deltas': conf_deltas,
    })
    return raw  # per-run: same as signed-only; regression done at confidence step


def confounder_conditioned(tester, edge, result):
    """Method F: Confounder-conditioned back-door (ANCOVA on intervention data).

    Real-world applicable: during each intervention experiment, OBSERVE the
    confounder states (occupancy sensors, window sensors, outdoor weather).
    At the statistical test stage, use ANCOVA to control for confounder
    changes — isolating the true causal effect.

    Per-run: returns raw signed effect (same as signed_only).
    Stashes observed confounder changes for ANCOVA at confidence time.

    Scientific basis: Pearl (2009) back-door criterion via subclassification /
    regression adjustment on OBSERVED confounders during intervention.
    Angrist & Pischke (2009) — ANCOVA for causal identification.
    """
    source, target = edge
    sv = tester.variable_mapping.get(source.lower(), source)
    tv = tester.variable_mapping.get(target.lower(), target)
    raw = tester.edge_validator.validate_edge_changes(
        result['pre_state'], result['post_state'], sv, tv)

    # Record observed confounder states during this intervention run
    find_key = tester.edge_validator._find_key
    pre, post = result['pre_state'], result['post_state']
    # All observable confounders (regime + exogenous)
    confounder_names = ['occupancy', 'windowposition', 'outdoortemp', 'solarradiation']
    conf_deltas = {}
    for cn in confounder_names:
        k = find_key(pre, cn)
        if k:
            rng = tester.edge_validator._RANGES.get(cn, 1.0)
            mn = tester.edge_validator._MINS.get(cn, 0.0)
            pre_n = (float(pre[k]) - mn) / max(rng, 1e-10)
            post_n = (float(post[k]) - mn) / max(rng, 1e-10)
            conf_deltas[cn] = post_n - pre_n
    if not hasattr(tester, '_ancova_data'):
        tester._ancova_data = {}
    tester._ancova_data.setdefault(edge, []).append({
        'effect': raw,
        'conf_deltas': conf_deltas,
    })
    return raw


def rf_plus_ancova(tester, edge, result):
    """Method G: RF-residualized effect + ANCOVA confidence.

    Combines obs-data RF model (DML residualization) with confounder-conditioned
    ANCOVA on intervention data. Two-stage deconfounding:
      1. RF removes confounder bias from per-run effect estimate (obs data)
      2. ANCOVA removes residual confounder correlation at test time (interv data)

    Real-world: train confounder model on historical sensor logs,
    record confounder states during experiments, apply both adjustments.
    """
    source, target = edge
    sv = tester.variable_mapping.get(source.lower(), source)
    tv = tester.variable_mapping.get(target.lower(), target)

    # RF-adjusted effect (from obs data)
    obs_data = getattr(tester, '_obs_data', None)
    raw = tester.edge_validator.validate_edge_changes(
        result['pre_state'], result['post_state'], sv, tv)
    effect = raw
    if obs_data is not None:
        adj = _rf_adjusted(tester, sv, tv,
                           result['pre_state'], result['post_state'], obs_data)
        if adj is not None and raw is not None:
            effect = 0.5 * adj + 0.5 * raw  # Same as Method E

    # Stash confounder changes for ANCOVA
    find_key = tester.edge_validator._find_key
    pre, post = result['pre_state'], result['post_state']
    confounder_names = ['occupancy', 'windowposition', 'outdoortemp', 'solarradiation']
    conf_deltas = {}
    for cn in confounder_names:
        k = find_key(pre, cn)
        if k:
            rng = tester.edge_validator._RANGES.get(cn, 1.0)
            mn = tester.edge_validator._MINS.get(cn, 0.0)
            pre_n = (float(pre[k]) - mn) / max(rng, 1e-10)
            post_n = (float(post[k]) - mn) / max(rng, 1e-10)
            conf_deltas[cn] = post_n - pre_n
    if not hasattr(tester, '_ancova_data'):
        tester._ancova_data = {}
    tester._ancova_data.setdefault(edge, []).append({
        'effect': effect,
        'conf_deltas': conf_deltas,
    })
    return effect


# ── Helper: OLS back-door (existing approach) ────────────────────────

def _ols_adjusted(tester, source_var, target_var, pre_state, post_state, obs_data):
    """OLS counterfactual — same as existing _backdoor_adjusted_effect."""
    find_key = tester.edge_validator._find_key
    def find_col(df, vn):
        vl = vn.lower()
        for c in df.columns:
            if c.lower() == vl:
                return c
            if vl == 'satisfaction' and c.lower() == 'overallsatisfaction':
                return c
        return None

    src_col = find_col(obs_data, source_var)
    tgt_col = find_col(obs_data, target_var)
    if not src_col or not tgt_col:
        return None

    skip = {src_col.lower(), tgt_col.lower(), 'timestamp', 'elapsedtime', 'interventionapplied'}
    conf_cols = [c for c in obs_data.columns
                 if c.lower() not in skip and np.issubdtype(obs_data[c].dtype, np.number)]
    if not conf_cols:
        return None

    feat_cols = [src_col] + conf_cols
    sub = obs_data[[tgt_col] + feat_cols].dropna()
    if len(sub) < len(feat_cols) + 5:
        return None

    try:
        X = sub[feat_cols].values
        y = sub[tgt_col].values
        X_aug = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    except (np.linalg.LinAlgError, ValueError):
        return None

    def to_norm(state, col):
        key = find_key(state, col)
        if key is None:
            return None
        raw = float(state[key])
        vl = col.lower()
        rng = tester.edge_validator._RANGES.get(vl, None)
        mn = tester.edge_validator._MINS.get(vl, None)
        if rng and rng > 1e-10 and mn is not None:
            return max(0.0, min(1.0, (raw - mn) / rng))
        cmin, cmax = obs_data[col].min(), obs_data[col].max()
        span = cmax - cmin
        return max(0.0, min(1.0, (raw - cmin) / span)) if span > 1e-10 else 0.5

    src_pre = to_norm(pre_state, src_col)
    src_post = to_norm(post_state, src_col)
    if src_pre is None or src_post is None:
        return None
    source_change = src_post - src_pre
    if abs(source_change) < 0.001:
        return None

    x_cf = [1.0]
    for col in feat_cols:
        val = to_norm(pre_state if col == src_col else post_state, col)
        if val is None:
            return None
        x_cf.append(val)
    target_cf = float(np.array(x_cf) @ beta)
    tgt_post = to_norm(post_state, tgt_col)
    if tgt_post is None:
        return None
    adjusted_delta = tgt_post - target_cf
    return adjusted_delta / abs(source_change)


# ── Helper: RF back-door (Double ML residualization) ─────────────────

_rf_cache = {}  # cache fitted RF models across runs

def _rf_adjusted(tester, source_var, target_var, pre_state, post_state, obs_data):
    """RF residualization: fit target ~ confounders (no source), subtract predicted change."""
    from sklearn.ensemble import RandomForestRegressor

    find_key = tester.edge_validator._find_key

    def find_col(df, vn):
        vl = vn.lower()
        for c in df.columns:
            if c.lower() == vl:
                return c
            if vl == 'satisfaction' and c.lower() == 'overallsatisfaction':
                return c
        return None

    src_col = find_col(obs_data, source_var)
    tgt_col = find_col(obs_data, target_var)
    if not src_col or not tgt_col:
        return None

    # Confounders = everything except source, target, non-numeric
    skip = {src_col.lower(), tgt_col.lower(), 'timestamp', 'elapsedtime', 'interventionapplied'}
    conf_cols = [c for c in obs_data.columns
                 if c.lower() not in skip and np.issubdtype(obs_data[c].dtype, np.number)]
    if not conf_cols:
        return None

    # Cache key
    cache_key = (tgt_col, tuple(conf_cols))
    if cache_key not in _rf_cache:
        sub = obs_data[[tgt_col] + conf_cols].dropna()
        if len(sub) < 50:
            return None
        X = sub[conf_cols].values
        y = sub[tgt_col].values
        rf = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)
        rf.fit(X, y)
        _rf_cache[cache_key] = (rf, conf_cols)

    rf, conf_cols = _rf_cache[cache_key]

    def to_norm(state, col):
        key = find_key(state, col)
        if key is None:
            return None
        raw = float(state[key])
        vl = col.lower()
        rng = tester.edge_validator._RANGES.get(vl, None)
        mn = tester.edge_validator._MINS.get(vl, None)
        if rng and rng > 1e-10 and mn is not None:
            return max(0.0, min(1.0, (raw - mn) / rng))
        cmin, cmax = obs_data[col].min(), obs_data[col].max()
        span = cmax - cmin
        return max(0.0, min(1.0, (raw - cmin) / span)) if span > 1e-10 else 0.5

    # Source change
    src_pre = to_norm(pre_state, src_col)
    src_post = to_norm(post_state, src_col)
    if src_pre is None or src_post is None:
        return None
    source_change = src_post - src_pre
    if abs(source_change) < 0.001:
        return None

    # RF prediction: what would target be given confounders at pre vs post?
    x_pre = []
    x_post = []
    for col in conf_cols:
        vp = to_norm(pre_state, col)
        vq = to_norm(post_state, col)
        if vp is None or vq is None:
            return None
        x_pre.append(vp)
        x_post.append(vq)

    target_hat_pre = rf.predict([x_pre])[0]
    target_hat_post = rf.predict([x_post])[0]
    confounder_explained = target_hat_post - target_hat_pre

    # Actual target change
    tgt_pre = to_norm(pre_state, tgt_col)
    tgt_post = to_norm(post_state, tgt_col)
    if tgt_pre is None or tgt_post is None:
        return None
    actual_change = tgt_post - tgt_pre

    # Residual = what confounders can't explain = causal effect of source
    adjusted_delta = actual_change - confounder_explained
    return adjusted_delta / abs(source_change)


# ── Patched confidence: ANCOVA (confounder-conditioned) ──────────────

def _ancova_confidence(tester_self, edge, effect_sizes):
    """ANCOVA-based confidence: test causal effect controlling for observed
    confounder changes during intervention.

    Model: effect_i = β0 + β1*Δocc_i + β2*Δwin_i + β3*Δoutdoor_i + ε_i
    H0: β0 = 0 (no causal effect after controlling for confounders)

    Real-world: β0 is the effect you'd see if confounders held still.
    """
    run_data = getattr(tester_self, '_ancova_data', {}).get(edge, [])
    valid = [r for r in run_data if r['effect'] is not None]

    if len(valid) < 5:
        return tester_self._orig_compute_statistical_confidence(edge, effect_sizes)

    Y = np.array([r['effect'] for r in valid])

    # Build confounder change matrix (exclude source/target if they're confounders)
    source, target = edge
    src_l = source.lower()
    tgt_l = target.lower()
    conf_names = sorted(set(
        k for r in valid for k in r['conf_deltas']
        if k != src_l and k != tgt_l  # don't control for the variables being tested
    ))

    # Need at least 3 more observations than parameters for meaningful test
    if not conf_names or len(Y) < len(conf_names) + 3:
        # Not enough confounder variation or data; fall back to standard t-test
        return tester_self._orig_compute_statistical_confidence(edge, effect_sizes)

    # Check: do confounders actually vary? If not, ANCOVA degenerates.
    conf_matrix = np.column_stack([
        np.array([r['conf_deltas'].get(cn, 0.0) for r in valid])
        for cn in conf_names
    ])
    # Drop confounders with near-zero variance (no leverage)
    conf_std = conf_matrix.std(axis=0)
    active_mask = conf_std > 1e-6
    if not np.any(active_mask):
        # No confounder variation → ANCOVA can't help, use standard t-test
        return tester_self._orig_compute_statistical_confidence(edge, effect_sizes)
    conf_matrix = conf_matrix[:, active_mask]
    active_names = [cn for cn, m in zip(conf_names, active_mask) if m]

    X = np.column_stack([np.ones(len(Y)), conf_matrix])
    n, k = X.shape
    if n <= k:
        return tester_self._orig_compute_statistical_confidence(edge, effect_sizes)

    try:
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return tester_self._orig_compute_statistical_confidence(edge, effect_sizes)

    intercept = beta[0]  # causal effect after controlling confounders
    y_hat = X @ beta
    mse = np.sum((Y - y_hat)**2) / (n - k)
    try:
        cov_beta = mse * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return tester_self._orig_compute_statistical_confidence(edge, effect_sizes)

    se_intercept = np.sqrt(max(cov_beta[0, 0], 1e-20))
    t_stat = intercept / se_intercept
    p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=n - k))

    confidence = (1.0 - p_value) if p_value < 0.1 else 0.0

    return {
        'confidence': confidence,
        'effect_size': intercept,
        'p_value': p_value,
        'num_interventions': len(Y),
        'effect_sizes': Y.tolist(),
        'ancova_confounders': active_names,
        'ancova_betas': {cn: float(b) for cn, b in zip(active_names, beta[1:])},
    }


# ── Diverse-timing simulation wrapper ────────────────────────────────

# Spread interventions across different times of day so confounders
# (occupancy, window position) take diverse states across runs.
# Real-world analogy: "run your thermostat experiments at different
# times — morning, afternoon, evening — to distinguish true effects
# from occupancy-correlated artifacts."
_DIVERSE_TIMES_MS = [
    100,         # ~00:00 — night, occ=0, win=0
    25200000,    # 07:00 — morning ramp, occ rising
    36000000,    # 10:00 — full office, occ~6
    43200000,    # 12:00 — noon, occ~6
    46800000,    # 13:00 — lunch dip, occ~3
    57600000,    # 16:00 — afternoon, occ~6
    68400000,    # 19:00 — evening ramp-down
    7200000,     # 02:00 — deep night
    32400000,    # 09:00 — morning work
    54000000,    # 15:00 — afternoon
]

# Hybrid timing: mostly midnight (low confounding) + a few daytime runs
# for regime-dependent edges. 7 of 10 runs at night, 3 at daytime.
_HYBRID_TIMES_MS = [
    100,         # midnight
    100,         # midnight
    100,         # midnight
    100,         # midnight
    100,         # midnight
    100,         # midnight
    100,         # midnight
    36000000,    # 10:00 — daytime (occ~6, win varies)
    46800000,    # 13:00 — lunch (occ~3)
    68400000,    # 19:00 — evening (occ dropping)
]


def _patch_diverse_timing(tester, time_schedule=None):
    """Replace _run_simulation with a version that varies intervention timing.

    Each call uses the next time slot from the schedule, cycling through.
    time_schedule: list of elapsedMs values. Defaults to _DIVERSE_TIMES_MS.
    """
    import tempfile, subprocess, json as _json

    schedule = time_schedule or _DIVERSE_TIMES_MS
    original_run_sim = tester._run_simulation
    tester._diverse_time_counter = 0

    def _diverse_run_simulation(edge, intervention, max_retries=3, retry_delay=2):
        """_run_simulation with rotating intervention time-of-day."""
        idx = tester._diverse_time_counter % len(schedule)
        tester._diverse_time_counter += 1
        interv_time = schedule[idx]
        duration = interv_time + 120000  # 2 minutes post-intervention

        if tester.smart_room_path is None:
            return tester._simulate_ashrae_intervention(edge, intervention)

        results = []
        sim_intervention = intervention['variables']

        for attempt in range(max_retries):
            process = None
            temp_js_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_js:
                    temp_js_path = temp_js.name
                    temp_js.write(f'''
                    const {{ simulateAndGetLatestData }} = require('{tester.smart_room_path}');
                    async function runSimulation() {{
                        try {{
                            const result = await simulateAndGetLatestData(
                                {duration},
                                {_json.dumps(sim_intervention)},
                                {interv_time}
                            );
                            if (!result || !result.preInterventionData || !result.postInterventionData) {{
                                throw new Error('Invalid simulation result');
                            }}
                            console.log("SIMULATION_RESULT:" + JSON.stringify(result));
                            process.exit(0);
                        }} catch (error) {{
                            console.error('SIMULATION_ERROR:', error.message);
                            process.exit(1);
                        }}
                    }}
                    process.on('unhandledRejection', (error) => {{
                        console.error('SIMULATION_ERROR:', error);
                        process.exit(1);
                    }});
                    runSimulation();
                    ''')

                process = subprocess.Popen(
                    ['node', '--max-old-space-size=512', temp_js_path],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, bufsize=1, universal_newlines=True
                )
                output, stderr = process.communicate(timeout=60)

                simulation_result = None
                for line in output.splitlines():
                    if 'SIMULATION_RESULT:' in line:
                        json_str = line.split('SIMULATION_RESULT:')[1].strip()
                        simulation_result = _json.loads(json_str)
                        break

                if simulation_result:
                    results.append({
                        'pre_state': simulation_result.get('preInterventionData', {}),
                        'post_state': simulation_result.get('postInterventionData', {}),
                    })
                    return results

            except Exception as e:
                logger.warning(f"Diverse-time sim attempt {attempt+1} failed: {e}")
            finally:
                if process and process.poll() is None:
                    try: process.kill()
                    except: pass
                if temp_js_path:
                    try: os.unlink(temp_js_path)
                    except: pass

            if attempt < max_retries - 1 and not results:
                time.sleep(retry_delay)

        return results

    tester._run_simulation = _diverse_run_simulation


# ── Patched confidence for intervention regression ───────────────────

def _interv_reg_confidence(tester_self, edge, effect_sizes):
    """Replace standard t-test with regression-based coefficient test."""
    run_data = getattr(tester_self, '_interv_run_data', {}).get(edge, [])

    if len(run_data) < 4:
        # Not enough data for regression, fall back to standard t-test
        return tester_self._orig_compute_statistical_confidence(edge, effect_sizes)

    # Build regression: target_effect ~ 1 + conf1_delta + conf2_delta
    Y = np.array([r['target_effect'] for r in run_data if r['target_effect'] is not None])
    conf_names = sorted(set(k for r in run_data for k in r['conf_deltas']))
    if not conf_names or len(Y) < len(conf_names) + 2:
        return tester_self._orig_compute_statistical_confidence(edge, effect_sizes)

    X = np.column_stack([
        np.ones(len(Y)),
        *[np.array([r['conf_deltas'].get(cn, 0) for r in run_data
                     if r['target_effect'] is not None]) for cn in conf_names]
    ])

    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return tester_self._orig_compute_statistical_confidence(edge, effect_sizes)

    intercept = beta[0]  # This is the effect after controlling for confounders
    # Compute SE of intercept
    n, k = X.shape
    if n <= k:
        return tester_self._orig_compute_statistical_confidence(edge, effect_sizes)
    y_hat = X @ beta
    mse = np.sum((Y - y_hat)**2) / (n - k)
    try:
        cov_beta = mse * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return tester_self._orig_compute_statistical_confidence(edge, effect_sizes)
    se_intercept = np.sqrt(max(cov_beta[0, 0], 1e-20))
    t_stat = intercept / se_intercept
    p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=n - k))

    if p_value < 0.05:
        confidence = 1.0 - p_value
    elif p_value < 0.1:
        confidence = 1.0 - p_value
    else:
        confidence = 0.0

    return {
        'confidence': confidence,
        'effect_size': intercept,
        'p_value': p_value,
        'num_interventions': len(Y),
        'effect_sizes': Y.tolist(),
    }


# ── Runner ───────────────────────────────────────────────────────────

def run_with_method(method_name, effect_fn, patch_confidence=False,
                    diverse_timing=False, extra_runs=False, ancova=False,
                    max_iter_override=None, hybrid_timing=False):
    """Run full pipeline with a specific effect-size method."""
    random.seed(SEED)
    np.random.seed(SEED)
    _rf_cache.clear()

    iters = max_iter_override or MAX_ITER

    obs_path = os.path.join(_BASE_DIR, SIM_CFG['obs_data_path'])
    obs_df = pd.read_csv(obs_path)
    sim_path = os.path.join(_BASE_DIR, SIM_CFG['sim_path'])

    pipeline = create_cwm_pipeline(
        csv_data=obs_df,
        api_key=API_KEY,
        smart_room_path=sim_path,
        dataset_type=SIM_CFG['dataset_type'],
        max_iterations=iters,
        alpha=0.5, beta=0.5,
        effect_threshold=0.1,
        actuator_vars=SIM_CFG.get('actuator_vars', []),
        non_intervenable_vars=SIM_CFG.get('non_intervenable_vars', []),
    )

    # Monkey-patch the effect-size method
    tester = pipeline.pipeline.tester
    tester._calculate_effect_size = lambda edge, result, _t=tester: effect_fn(_t, edge, result)

    # Diverse timing: vary time-of-day for intervention runs
    if hybrid_timing:
        _patch_diverse_timing(tester, time_schedule=_HYBRID_TIMES_MS)
    elif diverse_timing:
        _patch_diverse_timing(tester)

    # More runs: increase intervention count for better statistical power
    if extra_runs:
        tester.initial_test_count = 4
        tester.extended_test_count = 6  # total ~10 runs per edge

    # ANCOVA confidence: control for observed confounder changes
    if ancova:
        tester._orig_compute_statistical_confidence = tester._compute_statistical_confidence
        tester._compute_statistical_confidence = lambda edge, es, _t=tester: _ancova_confidence(_t, edge, es)

    if patch_confidence:
        tester._orig_compute_statistical_confidence = tester._compute_statistical_confidence
        tester._compute_statistical_confidence = lambda edge, es, _t=tester: _interv_reg_confidence(_t, edge, es)

    print(f"\n{'='*60}")
    print(f"  METHOD: {method_name}")
    print(f"{'='*60}")

    t0 = time.time()
    final_dag, final_metrics = pipeline.run_with_cwm()
    elapsed = time.time() - t0

    validated = set(pipeline.pipeline.validated_edges)
    m = compute_metrics(validated, SIM_CFG['gt_key'])

    print(f"  Time: {elapsed:.0f}s")
    print(f"  GT={m['n_gt']}  Disc={m['n_disc']}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    print(f"  SHD={m['shd']}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}")

    # Show edges
    gt_edges = {(s.lower(), t.lower()) for s, t in ALL_GT[SIM_CFG['gt_key']]['edges']}
    for e in sorted(validated):
        mark = "T" if (e[0].lower(), e[1].lower()) in gt_edges else "F"
        print(f"    [{mark}] {e[0]}→{e[1]}")

    return {'method': method_name, 'elapsed': elapsed, 'metrics': m}


if __name__ == '__main__':
    results = []

    # Each entry: (name, effect_fn, kwargs_dict)
    methods = [
        ('A: Signed-only 5iter', signed_only, {}),
        ('H: Signed-only 10iter', signed_only,
         {'max_iter_override': 10}),
        ('I: Hybrid-time + ANCOVA', confounder_conditioned,
         {'hybrid_timing': True, 'extra_runs': True, 'ancova': True}),
    ]

    for name, fn, kwargs in methods:
        try:
            r = run_with_method(name, fn, **kwargs)
            results.append(r)
        except Exception as e:
            print(f"\n  {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({'method': name, 'elapsed': 0, 'metrics': {'shd': -1, 'precision': 0, 'recall': 0, 'f1': 0}})

    # Summary
    print(f"\n{'='*80}")
    print(f"  COMPARISON SUMMARY — smart_building_rich (seed={SEED}, iter={MAX_ITER})")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'SHD':>5} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'Time':>7}")
    print('-'*80)
    for r in results:
        m = r['metrics']
        print(f"{r['method']:<30} {m['shd']:>5} {m['precision']:>6.2f} {m['recall']:>6.2f} "
              f"{m['f1']:>6.2f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {r['elapsed']:>6.0f}s")
    print('='*80)
