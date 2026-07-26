#!/bin/bash
# ============================================================
# Recovery script for interrupted server experiments
# Run inside tmux: tmux new -s recovery
# ============================================================

set -e
cd /home/user/machine_causation
source /home/user/machine_causation/causal-env/bin/activate

echo "============================================================"
echo "  PolicyGRID Server Recovery — $(date)"
echo "============================================================"

# ── 1. open_window: seed 789 needs policy, seed 999 needs full run ──
LOGDIR="logs/recovery_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOGDIR"

echo ""
echo "[1/4] open_window — seed 789 policy only"
python run_full_pipeline.py \
    --sim open_window \
    --seed-list 789 \
    --stages policy \
    --skip-discovery \
    --experiments none \
    2>&1 | tee "$LOGDIR/1_open_window_789_policy.log"

echo ""
echo "[2/4] open_window — seed 999 full pipeline"
python run_full_pipeline.py \
    --sim open_window \
    --seed-list 999 \
    --experiments none \
    2>&1 | tee "$LOGDIR/2_open_window_999_full.log"

# ── 2. smart_building_rich: seed 123 needs policy, seeds 456,789,999 need full ──
echo ""
echo "[3/4] smart_building_rich — seed 123 policy only"
python run_full_pipeline.py \
    --sim smart_building_rich \
    --seed-list 123 \
    --stages policy \
    --skip-discovery \
    --experiments none \
    2>&1 | tee "$LOGDIR/3_sbr_123_policy.log"

echo ""
echo "[4/4] smart_building_rich — seeds 456,789,999 full pipeline"
python run_full_pipeline.py \
    --sim smart_building_rich \
    --seed-list 456,789,999 \
    --experiments none \
    2>&1 | tee "$LOGDIR/4_sbr_456_789_999_full.log"

echo ""
echo "============================================================"
echo "  Recovery complete — $(date)"
echo "============================================================"
