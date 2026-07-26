#!/bin/bash
# C2 leave-one-generator-out ablation on open_window — LOCAL run.
# All four generators are live on open_window (no collinearity degeneracy),
# so this grid gives VARLiNGAM a real marginal-value row.
#
# Usage:  bash runners/run_c2_openwindow_local.sh
# Runs 6 configs sequentially (~10-15 min each, ~1-1.5 h total).
# Results: results/full_pipeline/open_window_<variant>/
# Logs:    logs/c2_ablation_ow/<variant>.log

cd "$(dirname "$0")/.." || exit 1
mkdir -p logs/c2_ablation_ow

run_variant () {
  local name="$1"; local gens="$2"
  echo "=== [$(date +%H:%M:%S)] C2 open_window: ${name} (generators: ${gens:-all four}) ==="
  if [ -n "$gens" ]; then
    python -u run_full_pipeline.py --sim open_window --seed-list 42,123,456 \
      --stages discovery,monitoring --experiments none \
      --generators "$gens" --out-suffix "$name" \
      > "logs/c2_ablation_ow/${name}.log" 2>&1
  else
    python -u run_full_pipeline.py --sim open_window --seed-list 42,123,456 \
      --stages discovery,monitoring --experiments none \
      --out-suffix "$name" \
      > "logs/c2_ablation_ow/${name}.log" 2>&1
  fi
  local rc=$?
  # Completion check: 3 seeds x edges.json
  local n_edges_files
  n_edges_files=$(ls "results/full_pipeline/open_window_${name}"/seed_*/edges.json 2>/dev/null | wc -l | tr -d ' ')
  echo "    exit=${rc}, edges.json files: ${n_edges_files}/3"
  grep -h "F1=\|Combined=\|Win=" "logs/c2_ablation_ow/${name}.log" | tail -6
}

run_variant abl_full      ""
run_variant no_pc         "sam,llm,varlingam"
run_variant no_sam        "pc,llm,varlingam"
run_variant no_llm        "pc,sam,varlingam"
run_variant no_varlingam  "pc,sam,llm"
run_variant pc_only       "pc"

echo "=== [$(date +%H:%M:%S)] C2 open_window grid COMPLETE ==="
echo "Summary of headline numbers:"
grep -H "Discovery: F1=" logs/c2_ablation_ow/*.log
