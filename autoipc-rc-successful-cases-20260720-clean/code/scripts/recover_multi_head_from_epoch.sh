#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:?OUT required}"
REF_PC1="${REF_PC1:?REF_PC1 required}"
REF_PC2="${REF_PC2:?REF_PC2 required}"
RESUME_EPOCH="${RESUME_EPOCH:-1500}"
REMAINING_EPOCHS="${REMAINING_EPOCHS:-3000}"
MANIFEST_JSONL="${MANIFEST_JSONL:-outputs/preflight/data_gate_20260703/frames.jsonl}"
LOCK_FILE="${TRAIN_EVAL_LOCK_FILE:-$OUT/train-eval.lock}"

cd "$ROOT"
source /home/dammerung/miniconda3/etc/profile.d/conda.sh
conda activate tensorflow

test -s "$OUT/checkpoints/tf_epoch_$(printf '%04d' "$RESUME_EPOCH").index"
flock "$LOCK_FILE" env \
  OUT="$OUT" \
  REF_PC1="$REF_PC1" \
  REF_PC2="$REF_PC2" \
  MANIFEST_JSONL="$MANIFEST_JSONL" \
  TRAIN_EVAL_LOCK_FILE="$LOCK_FILE" \
  bash "$ROOT/scripts/eval_multi_head_epoch.sh" "$RESUME_EPOCH" \
  > "$OUT/recovery_eval_epoch_${RESUME_EPOCH}.log" 2>&1

test -s "$OUT/fig2d_eval/epoch_${RESUME_EPOCH}/fig2d_metrics_summary.json"
export OUT REF_PC1 REF_PC2 MANIFEST_JSONL
export EPOCHS="$REMAINING_EPOCHS"
export INITIAL_EPOCH="$RESUME_EPOCH"
export RESUME_CHECKPOINT="$OUT/checkpoints/tf_epoch_$(printf '%04d' "$RESUME_EPOCH")"
nohup bash "$ROOT/scripts/run_paper_multi_head_parity_4500.sh" \
  > "$OUT/train_resume_launcher.log" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$OUT/launcher_shell.pid"
nohup env \
  OUT="$OUT" \
  REF_PC1="$REF_PC1" \
  REF_PC2="$REF_PC2" \
  MANIFEST_JSONL="$MANIFEST_JSONL" \
  TRAIN_PID="$TRAIN_PID" \
  TRAIN_EVAL_LOCK_FILE="$LOCK_FILE" \
  bash "$ROOT/scripts/watch_multi_head_eval.sh" \
  > "$OUT/watcher_resume_launcher.log" 2>&1 < /dev/null &
WATCH_PID=$!
echo "$WATCH_PID" > "$OUT/watcher.pid"
echo "recovery submitted: train_pid=$TRAIN_PID watcher_pid=$WATCH_PID resume_epoch=$RESUME_EPOCH"
wait "$TRAIN_PID"
