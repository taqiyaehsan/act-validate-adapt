# C2 generator ablation — open_window

**Missing/incomplete (not yet pulled or still running): pc_only**

| Config | seeds | F1 | Prec | Rec | Val edges | Union | Occ % | Win % | Combined % | ΔF1 | ΔComb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Full (PC+SAM+LLM+VARLiNGAM) | 3 | 0.218 ± 0.095 | 0.38 ± 0.19 | 0.15 ± 0.06 | 5.7 ± 0.9 | 41.3 ± 0.5 | -- | 72.8 ± 33.5 | 72.8 ± 33.5 | ref | ref |
| -- PC | 3 | 0.296 ± 0.000 | 0.29 ± 0.00 | 0.31 ± 0.00 | 14.0 ± 0.0 | 38.3 ± 1.2 | -- | 37.2 ± 8.7 | 37.2 ± 8.7 | +0.078 | -35.5 |
| -- SAM | 3 | 0.323 ± 0.035 | 0.38 ± 0.03 | 0.28 ± 0.04 | 9.7 ± 0.5 | 36.7 ± 2.1 | -- | 25.0 ± 0.0 | 25.0 ± 0.0 | +0.104 | -47.8 |
| -- LLM | 3 | 0.201 ± 0.029 | 0.20 ± 0.02 | 0.21 ± 0.04 | 13.3 ± 0.9 | 35.7 ± 0.5 | -- | 29.6 ± 5.6 | 29.6 ± 5.6 | -0.017 | -43.2 |
| -- VARLiNGAM | 3 | 0.235 ± 0.037 | 0.27 ± 0.04 | 0.21 ± 0.04 | 9.7 ± 0.5 | 30.7 ± 2.5 | -- | 25.0 ± 0.0 | 25.0 ± 0.0 | +0.016 | -47.8 |
| PC only | -- | (pending) | -- | -- | -- | -- | -- | -- | -- | -- | -- |

## Per-seed detail
- abl_full seed 123: F1=0.222 val=5 union=42 occ=None win=25.4 comb=25.4
- abl_full seed 42: F1=0.100 val=7 union=41 occ=None win=96.2 comb=96.2
- abl_full seed 456: F1=0.333 val=5 union=41 occ=None win=96.7 comb=96.7
- no_pc seed 123: F1=0.296 val=14 union=40 occ=None win=42.1 comb=42.1
- no_pc seed 42: F1=0.296 val=14 union=37 occ=None win=25.0 comb=25.0
- no_pc seed 456: F1=0.296 val=14 union=38 occ=None win=44.6 comb=44.6
- no_sam seed 123: F1=0.348 val=10 union=34 occ=None win=25.0 comb=25.0
- no_sam seed 42: F1=0.348 val=10 union=39 occ=None win=25.0 comb=25.0
- no_sam seed 456: F1=0.273 val=9 union=37 occ=None win=25.0 comb=25.0
- no_llm seed 123: F1=0.222 val=14 union=36 occ=None win=37.5 comb=37.5
- no_llm seed 42: F1=0.222 val=14 union=36 occ=None win=25.4 comb=25.4
- no_llm seed 456: F1=0.160 val=12 union=35 occ=None win=25.8 comb=25.8
- no_varlingam seed 123: F1=0.261 val=10 union=34 occ=None win=25.0 comb=25.0
- no_varlingam seed 42: F1=0.182 val=9 union=28 occ=None win=25.0 comb=25.0
- no_varlingam seed 456: F1=0.261 val=10 union=30 occ=None win=25.0 comb=25.0

## Generator complementarity (from abl_full phase-1 candidates, mean ± std over seeds)
| Generator | Candidates | GT-true | Unique | Unique GT-true | In final validated set |
|---|---|---|---|---|---|
| llm | 12.7 ± 4.1 | 5.3 ± 1.7 | 4.3 ± 0.5 | 1.7 ± 0.5 | 1.3 ± 0.5 |
| pc | 14.0 ± 0.0 | 7.0 ± 0.0 | 1.7 ± 0.5 | 0.0 ± 0.0 | 1.3 ± 0.5 |
| sam | 16.0 ± 0.0 | 5.0 ± 0.0 | 4.7 ± 0.5 | 0.0 ± 0.0 | 4.0 ± 1.4 |
| varlingam | 26.7 ± 1.2 | 7.0 ± 0.0 | 12.7 ± 0.5 | 1.0 ± 0.0 | 3.0 ± 0.0 |

## LaTeX rows (rebuttal table)
```latex
\begin{tabular}{lccc}
\toprule
Configuration & Discovery F1 & Validated edges & Regime monitoring (\%) \\
\midrule
Full (PC+SAM+LLM+VARLiNGAM) & $0.218 \pm 0.095$ & $6$ & $72.8 \pm 33.5$ \\
$-$ PC & $0.296 \pm 0.000$ & $14$ & $37.2 \pm 8.7$ \\
$-$ SAM & $0.323 \pm 0.035$ & $10$ & $25.0 \pm 0.0$ \\
$-$ LLM & $0.201 \pm 0.029$ & $13$ & $29.6 \pm 5.6$ \\
$-$ VARLiNGAM & $0.235 \pm 0.037$ & $10$ & $25.0 \pm 0.0$ \\
% pc_only: PENDING
\bottomrule
\end{tabular}
```
