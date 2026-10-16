#!/usr/bin/env python3
"""Operating characteristic of the SIMULATION edge-validation gate (camera-ready App. B).

Re-implements the default building-path acceptance rule of
src/tester.py (EdgeValidator.compute_confidence, ~L1150-1200) on synthetic per-trial effects:
  K per-trial effects  ->  one-sample t-test vs 0
  accept iff p < alpha  AND  |mean| > tau_eff  AND  all K effects share one sign
with alpha = 0.05 (paper's stated gate) and alpha = 0.10 (the code's secondary branch).
Effects are N(d, 1): d is the per-trial standardized effect, tau_eff = 0.1 in the same units.
The 3-trial screening pass and BH-FDR are not modelled (both only make acceptance stricter).

Outputs results/c2_analysis/sim_gate_oc.csv (power by d at K=10, both alphas) and
results/c2_analysis/sim_gate_oc_by_K.csv (Type I and power at d=2 by K).
"""

import numpy as np
import pandas as pd
from scipy import stats

N_MC = 200_000
TAU_EFF = 0.1


def accept_rate(d, K, alpha, n_mc=N_MC):
    # one fixed draw per (K, d): identical cells agree across tables, alphas are paired
    rng = np.random.default_rng([K, int(round(d * 100))])
    x = rng.normal(d, 1.0, size=(n_mc, K))
    m = x.mean(axis=1)
    s = x.std(axis=1, ddof=1)
    t = m / (s / np.sqrt(K))
    p = 2 * stats.t.sf(np.abs(t), df=K - 1)
    sign = (x > 0).all(axis=1) | (x < 0).all(axis=1)
    return float(((p < alpha) & (np.abs(m) > TAU_EFF) & sign).mean())


def main():
    rows = []
    for d in (0.0, 0.5, 1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0):
        rows.append({'K': 10, 'd': d,
                     'accept_alpha05': round(accept_rate(d, 10, 0.05), 4),
                     'accept_alpha10': round(accept_rate(d, 10, 0.10), 4)})
    by_d = pd.DataFrame(rows)
    by_d.to_csv('results/c2_analysis/sim_gate_oc.csv', index=False)
    print(by_d.to_string(index=False))

    rows = []
    for K in (4, 6, 8, 10, 12, 15):
        rows.append({'K': K,
                     'type_I_alpha05': round(accept_rate(0.0, K, 0.05), 5),
                     'sign_gate_bound': 2.0 ** -(K - 1),
                     'power_d2_alpha05': round(accept_rate(2.0, K, 0.05), 4)})
    by_k = pd.DataFrame(rows)
    by_k.to_csv('results/c2_analysis/sim_gate_oc_by_K.csv', index=False)
    print()
    print(by_k.to_string(index=False))


if __name__ == '__main__':
    main()
