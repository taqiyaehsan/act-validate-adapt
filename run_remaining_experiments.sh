#!/bin/bash
# Run all remaining server experiments in parallel tmux sessions.
# Launch from: ~/machine_causation/
#
# Usage:
#   chmod +x run_remaining_experiments.sh
#   ./run_remaining_experiments.sh

LOGDIR="logs/remaining_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOGDIR"

echo "============================================"
echo "  Remaining Experiments Launcher"
echo "  Log dir: $LOGDIR"
echo "============================================"

# Kill any existing session with this name
tmux kill-session -t experiments 2>/dev/null

# Create new tmux session
tmux new-session -d -s experiments -n main

# ── 1. smart_room_noise: 5 seeds, full pipeline ──
tmux new-window -t experiments -n noise
tmux send-keys -t experiments:noise \
  "python run_full_pipeline.py --sim smart_room_noise --seed-list 42,123,456,789,999 --experiments none 2>&1 | tee $LOGDIR/noise_pipeline.log" Enter

# ── 2. smart_room_hidden_vars: 5 seeds, full pipeline ──
tmux new-window -t experiments -n hidden
tmux send-keys -t experiments:hidden \
  "python run_full_pipeline.py --sim smart_room_hidden_vars --seed-list 42,123,456,789,999 --experiments none 2>&1 | tee $LOGDIR/hidden_pipeline.log" Enter

# ── 3. Ablations on smart_building_rich (primary sim, already has 5-seed edges) ──
tmux new-window -t experiments -n ablation_sbr
tmux send-keys -t experiments:ablation_sbr \
  "python run_full_pipeline.py --sim smart_building_rich --skip-main --seed-list 42,123,456,789,999 --experiments m3a,a1,a2,n2 2>&1 | tee $LOGDIR/ablation_sbr.log" Enter

# ── 4. Ablations on open_window (already has 5-seed edges) ──
tmux new-window -t experiments -n ablation_ow
tmux send-keys -t experiments:ablation_ow \
  "python run_full_pipeline.py --sim open_window --skip-main --seed-list 42,123,456,789,999 --experiments m3a,a1,a2,n2 2>&1 | tee $LOGDIR/ablation_ow.log" Enter

# ── 5. Ablations on smart_room (already has 5-seed edges) ──
tmux new-window -t experiments -n ablation_sr
tmux send-keys -t experiments:ablation_sr \
  "python run_full_pipeline.py --sim smart_room --skip-main --seed-list 42,123,456,789,999 --experiments m3a,a1,a2,n2 2>&1 | tee $LOGDIR/ablation_sr.log" Enter

echo ""
echo "Launched 5 parallel jobs in tmux session 'experiments'"
echo ""
echo "Monitor:"
echo "  tmux attach -t experiments          # attach to session"
echo "  tmux select-window -t experiments:noise   # switch window"
echo "  tail -f $LOGDIR/noise_pipeline.log  # follow a specific log"
echo ""
echo "Check status:"
echo "  for f in $LOGDIR/*.log; do echo \"\$f:\"; tail -1 \"\$f\"; echo; done"
