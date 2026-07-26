#!/usr/bin/env python3
"""Operating characteristic of the deployed physical validation gate's per-edge stage.

The per-edge gate (deployed runner: smart_room_physical/deployed/run_policygrid_discovery.py)
is:
  1. 3 screening trials; discard edge unless |mean of 3 deltas| >= EFFECT_THRESHOLDS[tgt]
     (resolution-level thresholds, ~0.01 in NORMALIZED units — see the runner)
  2. 4 confirmatory trials
  3. one-sample t-test on all 7 deltas; accept iff p < 0.10 (two-sided)
There is NO all-K sign-consistency gate on hardware. The deployed runner additionally
applies BH-FDR (alpha=0.10) ACROSS edges when >=10 reach the t-test (apply_fdr_correction;
this fired in the realized run, which tested exactly 10), so the per-edge analysis here,
which omits FDR, is a conservative upper bound on the deployed gate's Type I. This differs
from the simulation's composite gate (K=10, alpha=0.05, sign gate, BH-FDR): 2^-(K-1)
sign-gate arithmetic does not describe the hardware procedure; its Type I is governed by
the effect-size screen relative to hold-horizon sensor noise, with a selection effect from
screening feeding the pooled t-test.

This script (a) estimates hold-horizon noise sigma per sensor from the testbed's own
baseline data, (b) Monte-Carlos the exact gate for Type I (d=0) and power (d=2).
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
N_MC = 400_000
ALPHA = 0.10                      # validation alpha (deployed runner)
N_SCREEN, N_CONF = 3, 4           # tests_per_edge_screening / _confirmatory
# Deployed screen thresholds, NORMALIZED [0,1] units (EFFECT_THRESHOLDS in
# deployed/run_policygrid_discovery.py) — resolution-level, sized to sensor
# resolution, not to observed effects. The baseline CSV is normalized too, so
# theta and sigma are in the same units.
THETA = {'Temperature': 0.01, 'Humidity': 0.01, 'AirQuality': 0.01}

from scipy import stats


def gate_mc(theta_over_sigma, d, n_mc=N_MC, rng=RNG):
    """P(edge accepted) under per-trial standardized effect d, threshold theta/sigma."""
    # sigma = 1 wlog; theta and d in sigma units
    screen = rng.normal(d, 1.0, size=(n_mc, N_SCREEN))
    passed = np.abs(screen.mean(axis=1)) >= theta_over_sigma
    n_pass = int(passed.sum())
    if n_pass == 0:
        return 0.0, 0.0
    conf = rng.normal(d, 1.0, size=(n_pass, N_CONF))
    seven = np.concatenate([screen[passed], conf], axis=1)
    m = seven.mean(axis=1)
    s = seven.std(axis=1, ddof=1)
    t = m / (s / np.sqrt(seven.shape[1]))
    p = 2 * stats.t.sf(np.abs(t), df=seven.shape[1] - 1)
    accept = p < ALPHA
    return float(passed.mean()), float(accept.mean() * passed.mean())


def hold_lag_sigma(df, col, lag_min, quiet_mask):
    """Std of x(t+lag) - x(t) on rows where both endpoints are 'quiet'."""
    x = df[col].values
    d = x[lag_min:] - x[:-lag_min]
    ok = quiet_mask[lag_min:] & quiet_mask[:-lag_min]
    return float(np.std(d[ok], ddof=1)), int(ok.sum())


def main():
    import os
    data_dir = next(d for d in ('smart_room_physical/deployed/data',)
                    if os.path.exists(f'{d}/baseline_data.csv'))
    df = pd.read_csv(f'{data_dir}/baseline_data.csv')
    # baseline_data.csv is already NORMALIZED [0,1] — the same units as the
    # deployed EFFECT_THRESHOLDS and the edge_evidence deltas. Do NOT rescale.

    quiet = (df['Occupancy'].values < 0.5) & (df['WindowState'].values < 0.5)
    # also require actuators off so baseline deltas are intervention-free
    for a in ('Heater', 'Humidifier', 'Fan'):
        quiet &= df[a].values < 0.5

    print(f'baseline rows: {len(df)}, quiet rows: {int(quiet.sum())}')
    print(f'MC reps: {N_MC}, gate: |mean3|>=theta screen -> 7-trial t-test '
          f'(two-sided alpha={ALPHA}), no sign gate; per-edge stage only '
          f'(deployed runner adds BH-FDR across edges -> this is conservative)\n')

    rows = []
    for col, theta in THETA.items():
        for lag in (20, 25, 30):
            sigma, n = hold_lag_sigma(df, col, lag, quiet)
            r = theta / sigma
            p_screen0, type1 = gate_mc(r, 0.0)
            _, power2 = gate_mc(r, 2.0)
            rows.append({'sensor': col, 'lag_min': lag, 'n_deltas': n,
                         'sigma_hat': round(sigma, 3), 'theta': theta,
                         'theta_over_sigma': round(r, 2),
                         'P_pass_screen_null': round(p_screen0, 5),
                         'type_I': round(type1, 5),
                         'power_d2': round(power2, 4)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv('results/c2_analysis/physical_gate_oc.csv', index=False)

    print('\nGeneric theta/sigma grid (sensor-agnostic):')
    grid = []
    for r in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        _, t1 = gate_mc(r, 0.0)
        _, pw = gate_mc(r, 2.0)
        grid.append({'theta_over_sigma': r, 'type_I': round(t1, 5),
                     'power_d2': round(pw, 4)})
    print(pd.DataFrame(grid).to_string(index=False))

    # ── Power at REALIZED actuator effect sizes ─────────────────────────────
    # The screen demands |mean3| >= theta with theta many sigma above hold-
    # horizon noise, so power at a hypothetical d=2 is ~0 by design. The
    # relevant question is power at the effects actuators actually produce.
    # Compute realized d = |mean_delta| / sigma_hat for every gate-tested edge
    # of the deployed run and the gate's power at those effects.
    import json
    ev_path = 'smart_room_physical/deployed/results/discovery/seed_42/edge_evidence.json'
    try:
        ev = json.load(open(ev_path))
        ev = ev.get('edge_evidence', ev)
    except FileNotFoundError:
        print(f'\n[realized-effect section skipped: {ev_path} not found]')
        return
    sig25 = {c: hold_lag_sigma(df, c, 25, quiet)[0] for c in THETA}
    real = []
    for key, e in ev.items():
        if not isinstance(e, dict) or e.get('phase') != 'confirmatory':
            continue
        tgt = key.split('->')[-1]
        if tgt not in THETA or 'mean_delta' not in e or not e.get('std_delta'):
            continue
        # Realized per-trial standardized effect: the edge's own trial spread
        # (std_delta) is the noise the 7-trial t-test actually faces.
        d_trial = abs(e['mean_delta']) / e['std_delta']
        r = THETA[tgt] / sig25[tgt]
        _, pw = gate_mc(r, min(d_trial, 500.0))
        real.append({'edge': key, 'target': tgt,
                     'mean_delta': e['mean_delta'],
                     'std_delta_trials': e['std_delta'],
                     'realized_d_trial': round(d_trial, 2),
                     'p_value': e.get('p_value'),
                     'power_at_realized_d': round(pw, 4),
                     'validated': e.get('validated', False)})
    if real:
        rdf = pd.DataFrame(real).sort_values('realized_d_trial', ascending=False)
        print('\nRealized actuator effects (deployed run, sigma at 25-min hold):')
        print(rdf.to_string(index=False))
        rdf.to_csv('results/c2_analysis/physical_gate_realized_effects.csv',
                   index=False)
        print('\nReading: the deployed screen is resolution-level '
              '(theta ~ 0.01 normalized), so it filters only zero-effect '
              'edges; per-edge Type I is governed by the 7-trial t-test '
              '(~alpha pre-FDR), then tightened by BH-FDR across edges. '
              'Power is evaluated at the realized per-trial effects above; '
              'the simulation gate\'s d~2 criterion and 0.2% Type I are '
              'simulation-scoped and do not transfer.')


if __name__ == '__main__':
    main()
