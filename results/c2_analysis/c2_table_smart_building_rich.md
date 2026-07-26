# C2 generator ablation — smart_building_rich

| Config | seeds | F1 | Prec | Rec | Val edges | Union | Occ % | Win % | Combined % | ΔF1 | ΔComb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Full (PC+SAM+LLM+VARLiNGAM) | 3 | 0.431 ± 0.033 | 0.42 ± 0.04 | 0.45 ± 0.03 | 31.3 ± 1.2 | 133.0 ± 1.6 | 96.8 ± 1.5 | 70.3 ± 10.9 | 67.0 ± 10.6 | ref | ref |
| -- PC | 3 | 0.346 ± 0.015 | 0.28 ± 0.02 | 0.45 ± 0.00 | 46.3 ± 3.3 | 133.0 ± 1.4 | 96.0 ± 0.3 | 67.5 ± 6.6 | 63.6 ± 7.0 | -0.086 | -3.4 |
| -- SAM | 3 | 0.393 ± 0.009 | 0.37 ± 0.00 | 0.43 ± 0.02 | 33.7 ± 0.9 | 109.7 ± 0.5 | 98.5 ± 0.5 | 69.8 ± 12.1 | 68.2 ± 11.8 | -0.038 | +1.2 |
| -- LLM | 3 | 0.286 ± 0.027 | 0.24 ± 0.03 | 0.34 ± 0.03 | 41.0 ± 1.6 | 127.0 ± 2.9 | 96.4 ± 1.1 | 62.7 ± 0.2 | 59.1 ± 1.3 | -0.145 | -7.9 |
| -- VARLiNGAM | 3 | 0.373 ± 0.024 | 0.30 ± 0.02 | 0.48 ± 0.03 | 46.0 ± 0.8 | 103.7 ± 1.7 | 96.6 ± 0.7 | 65.5 ± 3.8 | 62.1 ± 4.5 | -0.058 | -4.9 |
| PC only | 3 | 0.262 ± 0.037 | 0.55 ± 0.04 | 0.17 ± 0.03 | 9.0 ± 0.8 | 24.7 ± 0.9 | 96.6 ± 4.4 | 83.9 ± 8.7 | 74.2 ± 21.8 | -0.169 | +7.2 |

## Per-seed detail
- abl_full seed 123: F1=0.467 val=31 union=131 occ=98.8 win=62.7 comb=61.5
- abl_full seed 42: F1=0.387 val=33 union=133 occ=95.2 win=62.6 comb=57.7
- abl_full seed 456: F1=0.441 val=30 union=135 occ=96.3 win=85.7 comb=81.8
- no_pc seed 123: F1=0.325 val=51 union=134 occ=95.8 win=63.0 comb=58.8
- no_pc seed 42: F1=0.356 val=44 union=134 occ=95.8 win=62.6 comb=58.5
- no_pc seed 456: F1=0.356 val=44 union=131 occ=96.5 win=76.9 comb=73.6
- no_sam seed 123: F1=0.387 val=33 union=110 occ=99.2 win=59.8 comb=58.7
- no_sam seed 42: F1=0.387 val=33 union=110 occ=98.2 win=62.8 comb=61.0
- no_sam seed 456: F1=0.406 val=35 union=109 occ=98.0 win=86.9 comb=84.9
- no_llm seed 123: F1=0.294 val=39 union=128 occ=96.7 win=62.8 comb=59.5
- no_llm seed 42: F1=0.314 val=41 union=123 occ=95.0 win=62.4 comb=57.4
- no_llm seed 456: F1=0.250 val=43 union=130 occ=97.6 win=62.8 comb=60.5
- no_varlingam seed 123: F1=0.342 val=47 union=103 occ=97.3 win=70.9 comb=68.5
- no_varlingam seed 42: F1=0.378 val=45 union=106 occ=95.7 win=62.8 comb=58.5
- no_varlingam seed 456: F1=0.400 val=46 union=102 occ=96.7 win=62.7 comb=59.4
- pc_only seed 123: F1=0.308 val=10 union=24 occ=90.3 win=71.6 comb=43.4
- pc_only seed 42: F1=0.216 val=8 union=24 occ=99.7 win=89.6 comb=89.0
- pc_only seed 456: F1=0.263 val=9 union=26 occ=99.7 win=90.6 comb=90.3

## Generator complementarity (from abl_full phase-1 candidates, mean ± std over seeds)
| Generator | Candidates | GT-true | Unique | Unique GT-true | In final validated set |
|---|---|---|---|---|---|
| llm | 28.0 ± 2.2 | 13.3 ± 0.9 | 8.3 ± 0.5 | 5.3 ± 0.5 | 13.0 ± 0.0 |
| pc | 23.3 ± 1.2 | 6.0 ± 0.0 | 2.3 ± 1.7 | 0.7 ± 0.5 | 11.3 ± 1.2 |
| sam | 74.7 ± 0.9 | 14.3 ± 0.5 | 24.7 ± 1.7 | 4.7 ± 0.5 | 22.3 ± 0.5 |
| varlingam | 80.3 ± 2.5 | 9.0 ± 0.8 | 37.3 ± 2.5 | 2.3 ± 0.9 | 10.0 ± 0.8 |

## LaTeX rows (rebuttal table)
```latex
\begin{tabular}{lccc}
\toprule
Configuration & Discovery F1 & Validated edges & Regime monitoring (\%) \\
\midrule
Full (PC+SAM+LLM+VARLiNGAM) & $0.431 \pm 0.033$ & $31$ & $67.0 \pm 10.6$ \\
$-$ PC & $0.346 \pm 0.015$ & $46$ & $63.6 \pm 7.0$ \\
$-$ SAM & $0.393 \pm 0.009$ & $34$ & $68.2 \pm 11.8$ \\
$-$ LLM & $0.286 \pm 0.027$ & $41$ & $59.1 \pm 1.3$ \\
$-$ VARLiNGAM & $0.373 \pm 0.024$ & $46$ & $62.1 \pm 4.5$ \\
PC only & $0.262 \pm 0.037$ & $9$ & $74.2 \pm 21.8$ \\
\bottomrule
\end{tabular}
```
