"""C2 leave-one-generator-out ablation: aggregate per-config results into the rebuttal table.

Reads results/server_pull_jul23/<sim>_<config>/seed_*/{edges.json,monitoring_metrics.json}
(the run_full_pipeline.py --generators/--out-suffix outputs pulled from the server) and emits:
  1. Main ablation table (F1, validated/union edge counts, monitoring occ/win/combined) with
     deltas vs the paired abl_full reference.
  2. Generator complementarity from abl_full's phase1_method_edges: per-generator candidate
     counts, GT-true candidates, unique contributions, and coverage of the final validated set.
Markdown + LaTeX rows are written to --out and printed to stdout.

Usage:
  python scripts/c2_aggregate.py
  python scripts/c2_aggregate.py --sim open_window --root results/full_pipeline --configs abl_full,no_pc,no_sam,no_llm,no_varlingam,pc_only
"""

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULT_CONFIGS = ["abl_full", "no_pc", "no_sam", "no_llm", "no_varlingam", "pc_only"]

CONFIG_LABELS = {
    "abl_full": "Full (PC+SAM+LLM+VARLiNGAM)",
    "no_pc": "-- PC",
    "no_sam": "-- SAM",
    "no_llm": "-- LLM",
    "no_varlingam": "-- VARLiNGAM",
    "pc_only": "PC only",
}


def norm_edge(e):
    return (e[0].lower(), e[1].lower())


def load_gt(sim):
    gt = json.load(open(REPO / "ground_truth_graphs.json"))[sim]["edges"]
    return {norm_edge(e) for e in gt}


def mean_std(vals):
    if not vals:
        return None
    m = statistics.mean(vals)
    # population std (ddof=0) to match run_full_pipeline.py's summary.json convention
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def fmt(ms, prec=3):
    if ms is None:
        return "--"
    return f"{ms[0]:.{prec}f} ± {ms[1]:.{prec}f}"


def load_config(root, sim, config):
    d = root / f"{sim}_{config}"
    if not d.is_dir():
        return None
    seeds = []
    for sd in sorted(d.glob("seed_*")):
        edges_f = sd / "edges.json"
        if not edges_f.exists():
            continue
        e = json.load(open(edges_f))
        row = {
            "seed": e["seed"],
            "f1": e["f1"],
            "precision": e["precision"],
            "recall": e["recall"],
            "shd": e["shd"],
            "n_val": len(e["validated_edges"]),
            "n_union": len(e["obs_only_union_edges"]),
            "validated": {tuple(norm_edge(x)) for x in e["validated_edges"]},
            "phase1": {g: [norm_edge(x) for x in eds] for g, eds in e.get("phase1_method_edges", {}).items()},
            "generators": e.get("generators"),
        }
        mon_f = sd / "monitoring_metrics.json"
        if mon_f.exists():
            m = json.load(open(mon_f))
            # open_window has no occupancy regime — occ_accuracy absent there
            row.update(win=m["win_accuracy"], combined=m["combined_accuracy"])
            if "occ_accuracy" in m:
                row["occ"] = m["occ_accuracy"]
        seeds.append(row)
    return {"dir": d, "seeds": seeds} if seeds else None


def aggregate(cfg):
    seeds = cfg["seeds"]
    get = lambda k: [s[k] for s in seeds if k in s]
    return {
        "n_seeds": len(seeds),
        "seed_ids": [s["seed"] for s in seeds],
        "f1": mean_std(get("f1")),
        "precision": mean_std(get("precision")),
        "recall": mean_std(get("recall")),
        "shd": mean_std(get("shd")),
        "n_val": mean_std(get("n_val")),
        "n_union": mean_std(get("n_union")),
        "occ": mean_std(get("occ")),
        "win": mean_std(get("win")),
        "combined": mean_std(get("combined")),
        "f1_per_seed": get("f1"),
        "combined_per_seed": get("combined"),
    }


def complementarity(full_cfg, gt):
    """Per-generator stats from the full run's phase1 candidates, averaged over seeds."""
    gens = sorted({g for s in full_cfg["seeds"] for g in s["phase1"]})
    out = {}
    for g in gens:
        n_cand, n_true, n_uniq, n_uniq_true, n_val_cov = [], [], [], [], []
        for s in full_cfg["seeds"]:
            cands = {tuple(e) for e in s["phase1"].get(g, [])}
            others = {tuple(e) for og, eds in s["phase1"].items() if og != g for e in eds}
            uniq = cands - others
            n_cand.append(len(cands))
            n_true.append(len(cands & gt))
            n_uniq.append(len(uniq))
            n_uniq_true.append(len(uniq & gt))
            n_val_cov.append(len(cands & s["validated"]))
        out[g] = {
            "candidates": mean_std(n_cand),
            "gt_true": mean_std(n_true),
            "unique": mean_std(n_uniq),
            "unique_gt_true": mean_std(n_uniq_true),
            "validated_coverage": mean_std(n_val_cov),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/server_pull_jul23")
    ap.add_argument("--sim", default="smart_building_rich")
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--out", default="results/c2_analysis")
    args = ap.parse_args()

    root = (REPO / args.root) if not Path(args.root).is_absolute() else Path(args.root)
    configs = args.configs.split(",")
    gt = load_gt(args.sim)

    data, missing = {}, []
    for c in configs:
        cfg = load_config(root, args.sim, c)
        if cfg is None:
            missing.append(c)
        else:
            data[c] = (cfg, aggregate(cfg))

    lines = [f"# C2 generator ablation — {args.sim}", ""]
    if missing:
        lines.append(f"**Missing/incomplete (not yet pulled or still running): {', '.join(missing)}**")
        lines.append("")

    ref = data.get("abl_full", (None, None))[1]

    # ---- main table (markdown) ----
    hdr = "| Config | seeds | F1 | Prec | Rec | Val edges | Union | Occ % | Win % | Combined % | ΔF1 | ΔComb |"
    lines += [hdr, "|" + "---|" * 12]
    for c in configs:
        if c not in data:
            lines.append(f"| {CONFIG_LABELS.get(c, c)} | -- | (pending) |" + " -- |" * 9)
            continue
        a = data[c][1]
        dF1 = f"{a['f1'][0] - ref['f1'][0]:+.3f}" if ref and a["f1"] else "--"
        dC = f"{a['combined'][0] - ref['combined'][0]:+.1f}" if ref and ref["combined"] and a["combined"] else "--"
        if c == "abl_full":
            dF1 = dC = "ref"
        lines.append(
            f"| {CONFIG_LABELS.get(c, c)} | {a['n_seeds']} | {fmt(a['f1'])} | {fmt(a['precision'], 2)} | "
            f"{fmt(a['recall'], 2)} | {fmt(a['n_val'], 1)} | {fmt(a['n_union'], 1)} | {fmt(a['occ'], 1)} | "
            f"{fmt(a['win'], 1)} | {fmt(a['combined'], 1)} | {dF1} | {dC} |"
        )
    lines.append("")

    # ---- per-seed detail ----
    lines.append("## Per-seed detail")
    for c in configs:
        if c not in data:
            continue
        cfg, a = data[c]
        for s in cfg["seeds"]:
            lines.append(
                f"- {c} seed {s['seed']}: F1={s['f1']:.3f} val={s['n_val']} union={s['n_union']}"
                + (f" occ={s.get('occ')} win={s.get('win')} comb={s.get('combined')}" if "combined" in s else " (no monitoring)")
            )
    lines.append("")

    # ---- complementarity ----
    if "abl_full" in data:
        lines.append("## Generator complementarity (from abl_full phase-1 candidates, mean ± std over seeds)")
        lines.append("| Generator | Candidates | GT-true | Unique | Unique GT-true | In final validated set |")
        lines.append("|" + "---|" * 6)
        for g, st in complementarity(data["abl_full"][0], gt).items():
            lines.append(
                f"| {g} | {fmt(st['candidates'], 1)} | {fmt(st['gt_true'], 1)} | {fmt(st['unique'], 1)} | "
                f"{fmt(st['unique_gt_true'], 1)} | {fmt(st['validated_coverage'], 1)} |"
            )
        lines.append("")

    # ---- LaTeX rows for the rebuttal ----
    lines.append("## LaTeX rows (rebuttal table)")
    lines.append("```latex")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Configuration & Discovery F1 & Validated edges & Regime monitoring (\%) \\")
    lines.append(r"\midrule")
    for c in configs:
        if c not in data:
            lines.append(f"% {c}: PENDING")
            continue
        a = data[c][1]
        lab = CONFIG_LABELS.get(c, c).replace("--", r"$-$")
        f1s = f"${a['f1'][0]:.3f} \\pm {a['f1'][1]:.3f}$" if a["f1"] else "--"
        nv = f"${a['n_val'][0]:.0f}$" if a["n_val"] else "--"
        cb = f"${a['combined'][0]:.1f} \\pm {a['combined'][1]:.1f}$" if a["combined"] else "--"
        lines.append(f"{lab} & {f1s} & {nv} & {cb} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("```")

    out_dir = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_f = out_dir / f"c2_table_{args.sim}.md"
    out_f.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[written to {out_f}]")


if __name__ == "__main__":
    main()
