#!/usr/bin/env python3
"""Supervised-classifier baseline vs. Bayesian regime monitor (rebuttal item 1).

The requested comparison: logistic regression + small decision tree against the
factored Bayesian monitor, with identical labels, features, training data, and test
episodes. All three arms are scored on the SAME fresh simulator episode per seed, so
every comparison is paired.

Protocol:
  - Training: data_regen/smart_building_rich_processed.csv (the monitor's training CSV).
    Labels: occ = occupancy >= 0.1, win = windowposition >= 0.15 (the exact partition
    thresholds construct_factored_monitors uses). 4-way label = occ*1 + win*2.
  - Features: observable variables only (13 vars), current step + previous step
    (the monitor's likelihood at step t uses its temporal model on the same window).
  - Test: fresh 4-day sim episode (collect_sim_data, 5760 steps) per seed; the monitor
    uses that seed's validated edges (lowercased; sensor weights from the graph,
    sigma/forget at deployed defaults), classifiers are seed-independent but scored on
    the same episode.
  - Metrics per arm: 4-way accuracy, occ/win marginal accuracy, occ/win Brier,
    transition delay (steps to first correct 4-way call after each GT transition,
    censored at segment length).

Output: results/classifier_baseline/seed_<s>/{metrics.json,records.csv.gz},
        results/classifier_baseline/summary.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from neurips_final_experiments import (  # noqa: E402
    DATA_PATH, SCALING_PATH, OBSERVABLE_VARS, SENSOR_VARS, LATENT_VARS,
    get_scaling_map, construct_factored_monitors, collect_sim_data,
    run_monitoring, compute_sensor_weights_from_graph,
)

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

import logging
logger = logging.getLogger(__name__)

EDGES_DIR = 'results/server_pull_jul23/smart_building_rich_abl_full'
OUT_DIR = 'results/classifier_baseline'
REGIME_NAMES = ['base', 'occ', 'win', 'full']


FEATURE_VARS = None  # set in main(): OBSERVABLE_VARS or SENSOR_VARS (--sensor-only)


def load_training_matrix():
    """Training features/labels with the monitor's exact label thresholds."""
    data = pd.read_csv(DATA_PATH)
    cols_lower = {c.lower(): c for c in data.columns}
    occ_on = data[cols_lower['occupancy']].values >= 0.1
    win_on = data[cols_lower['windowposition']].values >= 0.15
    y4 = occ_on.astype(int) * 1 + win_on.astype(int) * 2

    obs_cols = [cols_lower[v] for v in FEATURE_VARS]
    X = data[obs_cols].values.astype(float)
    # prev-step + current-step features; label at current step. Consecutive-row
    # pairing matches the monitor's fit_temporal (values[:-1] -> values[1:]).
    X_feat = np.concatenate([X[:-1], X[1:]], axis=1)
    y_feat = y4[1:]
    return data, X_feat, y_feat


def episode_features(records):
    """Per-step feature rows for the classifiers, aligned to records[1:]."""
    vecs = np.array([[r['state_obs'].get(v, 0.5) for v in FEATURE_VARS]
                     for r in records], dtype=float)
    return np.concatenate([vecs[:-1], vecs[1:]], axis=1)


def transition_delays(pred4, gt4):
    """Steps after each GT 4-way transition until the first correct call.

    Censored at segment length when the arm never gets the new regime right.
    Returns (delays, n_censored).
    """
    delays, censored = [], 0
    n = len(gt4)
    i = 1
    while i < n:
        if gt4[i] != gt4[i - 1]:
            seg_end = i
            while seg_end < n and gt4[seg_end] == gt4[i]:
                seg_end += 1
            d = None
            for k in range(i, seg_end):
                if pred4[k] == gt4[k]:
                    d = k - i
                    break
            if d is None:
                d = seg_end - i
                censored += 1
            delays.append(d)
            i = seg_end
        else:
            i += 1
    return delays, censored


def score_arm(pred4, p_occ, p_win, gt4, occ_active, win_active):
    pred4 = np.asarray(pred4)
    gt4 = np.asarray(gt4)
    p_occ = np.asarray(p_occ, dtype=float)
    p_win = np.asarray(p_win, dtype=float)
    occ_active = np.asarray(occ_active, dtype=float)
    win_active = np.asarray(win_active, dtype=float)

    delays, censored = transition_delays(list(pred4), list(gt4))
    return {
        'combined_accuracy': round(float((pred4 == gt4).mean()) * 100, 1),
        'occ_accuracy': round(float(((p_occ >= 0.5) == (occ_active >= 0.5)).mean()) * 100, 1),
        'win_accuracy': round(float(((p_win >= 0.5) == (win_active >= 0.5)).mean()) * 100, 1),
        'brier_occ': round(float(((p_occ - occ_active) ** 2).mean()), 4),
        'brier_win': round(float(((p_win - win_active) ** 2).mean()), 4),
        'mean_transition_delay': round(float(np.mean(delays)), 1) if delays else None,
        'median_transition_delay': round(float(np.median(delays)), 1) if delays else None,
        'n_transitions': len(delays),
        'n_censored': censored,
    }


def run_seed(seed, data, clfs, smap, duration_steps):
    edges_path = f'{EDGES_DIR}/seed_{seed}/edges.json'
    with open(edges_path) as f:
        payload = json.load(f)
    validated = {(s.lower(), t.lower()) for s, t in payload['validated_edges']}
    logger.info(f'[seed {seed}] {len(validated)} validated edges')

    monitors = construct_factored_monitors(validated, data)
    if monitors is None:
        raise RuntimeError(f'seed {seed}: no observable edges, monitor unbuildable')
    weights = compute_sensor_weights_from_graph(validated, SENSOR_VARS, LATENT_VARS)

    t0 = time.time()
    records = collect_sim_data(smap, duration_steps=duration_steps)
    logger.info(f'[seed {seed}] episode collected: {len(records)} steps '
                f'in {time.time() - t0:.0f}s')
    if len(records) < 100:
        raise RuntimeError(f'seed {seed}: episode too short ({len(records)} steps)')

    mon_records = run_monitoring(monitors, records, sensor_weights=weights)

    # Alignment: classifiers need a previous step, so all arms are scored on
    # records[1:] (monitor record 0 dropped identically).
    gt4 = [r['regime_gt'] for r in records[1:]]
    occ_active = [r['occ_active'] for r in records[1:]]
    win_active = [r['win_active'] for r in records[1:]]
    name_to_int = {n: i for i, n in enumerate(REGIME_NAMES)}
    mon_pred4 = [name_to_int[m['winner']] for m in mon_records[1:]]
    mon_p_occ = [m['P_occ'] for m in mon_records[1:]]
    mon_p_win = [m['P_win'] for m in mon_records[1:]]

    X_ep = episode_features(records)

    arms = {'monitor': score_arm(mon_pred4, mon_p_occ, mon_p_win,
                                 gt4, occ_active, win_active)}
    per_step = {
        'gt4': gt4, 'occ_active': occ_active, 'win_active': win_active,
        'monitor_pred4': mon_pred4, 'monitor_p_occ': mon_p_occ,
        'monitor_p_win': mon_p_win,
    }

    for name, clf in clfs.items():
        proba = clf.predict_proba(X_ep)
        # map clf.classes_ onto the 4-way space (a class can be absent in training)
        p4 = np.zeros((len(X_ep), 4))
        for ci, c in enumerate(clf.classes_):
            p4[:, int(c)] = proba[:, ci]
        pred4 = p4.argmax(axis=1)
        p_occ = p4[:, 1] + p4[:, 3]
        p_win = p4[:, 2] + p4[:, 3]
        arms[name] = score_arm(pred4, p_occ, p_win, gt4, occ_active, win_active)
        per_step[f'{name}_pred4'] = pred4.tolist()
        per_step[f'{name}_p_occ'] = p_occ.round(4).tolist()
        per_step[f'{name}_p_win'] = p_win.round(4).tolist()

    regime_counts = {REGIME_NAMES[k]: int((np.asarray(gt4) == k).sum())
                     for k in range(4)}
    seed_dir = f'{OUT_DIR}/seed_{seed}'
    os.makedirs(seed_dir, exist_ok=True)
    with open(f'{seed_dir}/metrics.json', 'w') as f:
        json.dump({'seed': seed, 'duration_steps': duration_steps,
                   'n_scored_steps': len(gt4), 'regime_step_counts': regime_counts,
                   'n_validated_edges': len(validated), 'arms': arms}, f, indent=2)
    pd.DataFrame(per_step).to_csv(f'{seed_dir}/records.csv.gz', index=False,
                                  compression='gzip')
    logger.info(f'[seed {seed}] regime steps: {regime_counts}')
    for arm, m in arms.items():
        logger.info(f'[seed {seed}] {arm:>8}: 4way={m["combined_accuracy"]:.1f}% '
                    f'occ={m["occ_accuracy"]:.1f}% win={m["win_accuracy"]:.1f}% '
                    f'brier={m["brier_occ"]:.3f}/{m["brier_win"]:.3f} '
                    f'delay={m["mean_transition_delay"]} '
                    f'(n={m["n_transitions"]}, cens={m["n_censored"]})')
    return arms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='42,123,456')
    parser.add_argument('--sensor-only', action='store_true',
                        help="Restrict classifier features to the monitor's six "
                             "sensor variables (feature-matched comparison)")
    parser.add_argument('--steps', type=int, default=5760,
                        help='episode length (5760 = 4-day paper protocol)')
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    global FEATURE_VARS, OUT_DIR
    FEATURE_VARS = SENSOR_VARS if args.sensor_only else OBSERVABLE_VARS
    if args.sensor_only:
        OUT_DIR = 'results/classifier_baseline_sensor6'
    os.makedirs(OUT_DIR, exist_ok=True)

    data, X_feat, y_feat = load_training_matrix()
    logger.info(f'training matrix: {X_feat.shape}, '
                f'label counts: {np.bincount(y_feat, minlength=4).tolist()} '
                f'(base/occ/win/full)')

    np.random.seed(0)
    clfs = {
        'logreg': LogisticRegression(max_iter=2000, C=1.0),
        'tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=50,
                                       random_state=0),
    }
    for name, clf in clfs.items():
        clf.fit(X_feat, y_feat)
        train_acc = clf.score(X_feat, y_feat)
        logger.info(f'{name}: train accuracy {train_acc:.3f}')

    scaling = pd.read_csv(SCALING_PATH)
    smap = get_scaling_map(scaling)

    summary = {'protocol': {
        'steps': args.steps, 'seeds': seeds,
        'label_thresholds': 'occ>=0.1, win>=0.15 (monitor partition thresholds)',
        'features': (f'{"SENSOR_VARS" if args.sensor_only else "OBSERVABLE_VARS"} '
                     f'at t-1 and t ({2 * len(FEATURE_VARS)} dims)'),
        'monitor_config': 'lowercased validated edges, graph sensor weights, '
                          'sigma/forget deployed defaults (1.5/0.15)',
        'training_labels_4way_counts': np.bincount(y_feat, minlength=4).tolist(),
    }, 'seeds': {}}

    for seed in seeds:
        try:
            summary['seeds'][seed] = run_seed(seed, data, clfs, smap, args.steps)
        except Exception as e:
            logger.error(f'[seed {seed}] FAILED: {e}')
            summary['seeds'][seed] = {'error': str(e)}
        with open(f'{OUT_DIR}/summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    # aggregate across seeds
    ok = [s for s in summary['seeds'].values() if 'error' not in s]
    if ok:
        agg = {}
        for arm in ok[0]:
            agg[arm] = {}
            for metric in ('combined_accuracy', 'occ_accuracy', 'win_accuracy',
                           'brier_occ', 'brier_win', 'mean_transition_delay'):
                vals = [s[arm][metric] for s in ok if s[arm][metric] is not None]
                if vals:
                    agg[arm][metric] = {
                        'mean': round(float(np.mean(vals)), 3),
                        'std': round(float(np.std(vals)), 3),
                        'per_seed': vals,
                    }
        summary['aggregate'] = agg
        with open(f'{OUT_DIR}/summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info('=== AGGREGATE (mean +- std over seeds) ===')
        for arm, metrics in agg.items():
            line = ' '.join(f'{k}={v["mean"]}+-{v["std"]}'
                            for k, v in metrics.items())
            logger.info(f'{arm:>8}: {line}')


if __name__ == '__main__':
    main()
