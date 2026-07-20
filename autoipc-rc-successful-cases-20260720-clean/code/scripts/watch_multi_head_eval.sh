#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:?OUT required}"
REF_PC1="${REF_PC1:?REF_PC1 required}"
REF_PC2="${REF_PC2:?REF_PC2 required}"
MANIFEST_JSONL="${MANIFEST_JSONL:-outputs/preflight/data_gate_20260703/frames.jsonl}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
LOCK_FILE="${TRAIN_EVAL_LOCK_FILE:-$OUT/train-eval.lock}"
TRAIN_PID="${TRAIN_PID:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$OUT/multi_head_watch.log"
TARGETS=(100 500 1000 1500 2000 3000 4000 4500)

mkdir -p "$OUT"
cd "$ROOT"
source /home/dammerung/miniconda3/etc/profile.d/conda.sh
conda activate tensorflow

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"
}

for epoch in "${TARGETS[@]}"; do
  eval_dir="$OUT/fig2d_eval/epoch_${epoch}"
  summary="$eval_dir/fig2d_metrics_summary.json"
  ckpt="$OUT/checkpoints/epoch_$(printf '%04d' "$epoch").weights.h5"
  if [ -s "$summary" ]; then
    touch "$eval_dir/evaluation.done"
    log "already evaluated epoch=$epoch"
    continue
  fi
  while [ ! -s "$summary" ]; do
    if [ -s "$ckpt" ]; then
      log "checkpoint ready epoch=$epoch; starting evaluation"
      flock "$LOCK_FILE" env \
        OUT="$OUT" \
        REF_PC1="$REF_PC1" \
        REF_PC2="$REF_PC2" \
        MANIFEST_JSONL="$MANIFEST_JSONL" \
        TRAIN_EVAL_LOCK_FILE="$LOCK_FILE" \
        bash "$ROOT/scripts/eval_multi_head_epoch.sh" "$epoch" >> "$LOG" 2>&1
      log "evaluation complete epoch=$epoch"
      break
    fi
    if [ -n "$TRAIN_PID" ] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
      log "training pid=$TRAIN_PID exited before checkpoint epoch=$epoch"
      exit 1
    fi
    log "waiting for checkpoint epoch=$epoch"
    sleep "$SLEEP_SECONDS"
  done
done
log "all multi-head Fig2d milestones evaluated"
