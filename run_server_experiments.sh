#!/bin/bash
# Recovery: run remaining batches (ashrae + smart_room already done)
# Usage: screen -S experiments && bash run_server_experiments.sh
set -e
cd /home/user/machine_causation
source /home/user/machine_causation/causal-env/bin/activate
mkdir -p logs

echo "=========================================="
echo "  PolicyGRID Server Experiments (remaining)"
echo "  Started: $(date)"
echo "=========================================="

# ── Pre-flight checks ────────────────────────────────────
echo ""
echo "=== Pre-flight checks ==="
echo ""
echo "-- Existing python/node processes --"
ps aux | grep -E 'python.*run_full_pipeline|node.*js/' | grep -v grep || echo "  (none)"
echo ""
echo "-- Server load --"
uptime
echo ""
echo "-- Top CPU consumers --"
ps aux --sort=-%cpu | head -6
echo ""
echo "-- Memory --"
free -h
echo ""
echo "Press Enter to continue or Ctrl-C to abort..."
read

# ── BATCH 1: open_window + smart_building_rich ────────────
echo ""
echo "[BATCH 1/3] open_window + smart_building_rich"
echo "  Started: $(date)"

python run_full_pipeline.py --sim open_window --seeds 5 --experiments none \
  > logs/open_window.log 2>&1 &
PID1=$!

python run_full_pipeline.py --sim smart_building_rich --seeds 5 --experiments none \
  > logs/rich_building.log 2>&1 &
PID2=$!

echo "  PIDs: $PID1, $PID2"
wait $PID1 $PID2
echo "  BATCH 1 DONE: $(date)"

# ── BATCH 2: smart_room_noise + hidden_vars ───────────────
echo ""
echo "[BATCH 2/3] smart_room_noise + hidden_vars"
echo "  Started: $(date)"

python run_full_pipeline.py --sim smart_room_noise --seeds 5 --experiments none \
  > logs/smart_room_noise.log 2>&1 &
PID1=$!

python run_full_pipeline.py --sim smart_room_hidden_vars --seeds 5 --experiments none \
  > logs/hidden_vars.log 2>&1 &
PID2=$!

echo "  PIDs: $PID1, $PID2"
wait $PID1 $PID2
echo "  BATCH 2 DONE: $(date)"

# ── BATCH 3: ablations on smart_building_rich ─────────────
echo ""
echo "[BATCH 3/3] Ablations (smart_building_rich)"
echo "  Started: $(date)"

python run_full_pipeline.py --sim smart_building_rich --seeds 5 \
  --skip-main --experiments m3a,a1,n2,a2,m4,ablation,int_strategy,hyperparam \
  > logs/ablations.log 2>&1

echo "  BATCH 3 DONE: $(date)"

# ── SUMMARY ───────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Finished: $(date)"
echo "=========================================="
echo ""
echo "Check results:"
echo "  grep -E 'F1=|edges|Monitoring|Policy' logs/*.log"
echo "  ls results/full_pipeline/*/summary.json"
