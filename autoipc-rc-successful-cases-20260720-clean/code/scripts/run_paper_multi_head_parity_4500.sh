#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:?OUT required}"
MANIFEST_JSONL="${MANIFEST_JSONL:-outputs/preflight/data_gate_20260703/frames.jsonl}"
NORMALIZATION_JSON="${NORMALIZATION_JSON:-outputs/preflight/fhi47_paper_20260706/normalization.json}"
EPOCHS="${EPOCHS:-4500}"
INITIAL_EPOCH="${INITIAL_EPOCH:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

mkdir -p "$OUT"
cd "$ROOT"
source /home/dammerung/miniconda3/etc/profile.d/conda.sh
conda activate tensorflow

# Keep a separate submission lock for the long-running trainer. The
# train-eval.lock file is reserved for short checkpoint/evaluation transactions
# so the watcher can evaluate while training continues.
exec 9>"$OUT/training-submit.lock"
if ! flock -n 9; then
  echo "another training submission already holds $OUT/training-submit.lock" >&2
  exit 2
fi
echo "$$" > "$OUT/launcher.pid"
export TRAIN_EVAL_LOCK_FILE="$OUT/train-eval.lock"
export TRAIN_EVAL_BARRIER_DIR="$OUT/fig2d_eval"
export TRAIN_EVAL_BARRIER_EPOCHS="100,500,1000,1500,2000,3000,4000,4500"

resume_args=()
if [ -n "$RESUME_CHECKPOINT" ]; then
  resume_args=(--resume-checkpoint "$RESUME_CHECKPOINT" --initial-epoch "$INITIAL_EPOCH")
fi

PYTHONPATH=src python scripts/train.py \
  --manifest-jsonl "$MANIFEST_JSONL" \
  --output-dir "$OUT" \
  --epochs "$EPOCHS" \
  --batch-size 300 \
  --sample-frames-per-epoch 20000 \
  --repeat-size 2 \
  --shuffle-buffer 400 \
  --normalization-json "$NORMALIZATION_JSON" \
  --inner-dim 250 \
  --dropout 0.1 \
  --descriptor-dropout 0.1 \
  --m1 100 \
  --m2 100 \
  --learning-rate 1e-4 \
  --rank-weight 0 \
  --head-weights 1,1 \
  --pc-spec PC1:0.2:0.455 \
  --pc-spec PC2:0.4:0.490 \
  --log-every 500 \
  --checkpoint-every 100 \
  --checkpoint-keep 5 \
  --seed 20260716 \
  "${resume_args[@]}"
