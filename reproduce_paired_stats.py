#!/usr/bin/env python3
"""Reproduces the paired-difference statistics reported in §5.3 of the paper.

Inputs: server_runs/closedloop_server/results_{cold,warm,mild}.csv
Outputs: paired t, p-value, Cohen's d, 95% CI, leave-one-seed-out range.

Run from repo root:
    python reproduce_paired_stats.py
"""
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, t as tdist

CSV_DIR = "server_runs/closedloop_server"
WEATHERS = ["cold", "warm", "mild"]
EPS = 1.0
METHOD_A = "Validated"
METHOD_B = "Obs-Only"


def main():
    dfs = []
    for w in WEATHERS:
        d = pd.read_csv(f"{CSV_DIR}/results_{w}.csv")
        d["weather"] = w
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)

    e = df[df.comfort_target == EPS]
    a = e[e.method == METHOD_A].sort_values(["seed", "weather"]).reset_index(drop=True)
    b = e[e.method == METHOD_B].sort_values(["seed", "weather"]).reset_index(drop=True)
    assert len(a) == len(b), f"unmatched: {len(a)} vs {len(b)}"

    a_kwh, b_kwh = a.kWh.values, b.kWh.values
    diff = b_kwh - a_kwh
    n = len(diff)

    print(f"n = {n} paired (seed, weather) conditions at ε={EPS}")
    print(f"{METHOD_A:10s}: {a_kwh.mean():.2f} ± {a_kwh.std(ddof=1):.2f} kWh")
    print(f"{METHOD_B:10s}: {b_kwh.mean():.2f} ± {b_kwh.std(ddof=1):.2f} kWh")

    print(f"\nPaired difference: {diff.mean():.2f} ± {diff.std(ddof=1):.2f} kWh")
    t, p = ttest_rel(b_kwh, a_kwh)
    print(f"Paired t = {t:.2f}, df = {n-1}, p = {p:.3e}")

    d = diff.mean() / diff.std(ddof=1)
    print(f"Cohen's d = {d:.2f}")

    se = diff.std(ddof=1) / np.sqrt(n)
    margin = tdist.ppf(0.975, n - 1) * se
    ci_lo, ci_hi = diff.mean() - margin, diff.mean() + margin
    pct_lo = ci_lo / b_kwh.mean() * 100
    pct_hi = ci_hi / b_kwh.mean() * 100
    print(f"95% CI on absolute gap: [{ci_lo:.2f}, {ci_hi:.2f}] kWh")
    print(f"95% CI on relative gap: [{pct_lo:.1f}%, {pct_hi:.1f}%]")

    print(f"\nLeave-one-seed-out (relative gap):")
    seeds = sorted(a.seed.unique())
    for s in seeds:
        mask = a.seed != s
        sub_a = a_kwh[mask]
        sub_b = b_kwh[mask]
        gap_pct = (sub_b.mean() - sub_a.mean()) / sub_b.mean() * 100
        print(f"  hold out seed {s}: {gap_pct:.2f}%")

    print(f"\nPer-weather breakdown:")
    for w in WEATHERS:
        wa = a[a.weather == w].kWh.mean()
        wb = b[b.weather == w].kWh.mean()
        print(f"  {w:5s}: {METHOD_A}={wa:.2f}, {METHOD_B}={wb:.2f}, gap={(wb-wa)/wb*100:.1f}%")


if __name__ == "__main__":
    main()
