#!/usr/bin/env bash
set -euo pipefail

EPOCH="${1:?epoch required}"
OUT="${OUT:?OUT required}"
REF_PC1="${REF_PC1:?REF_PC1 required}"
REF_PC2="${REF_PC2:?REF_PC2 required}"
MANIFEST_JSONL="${MANIFEST_JSONL:-outputs/preflight/data_gate_20260703/frames.jsonl}"
EXPECTED_ROWS="${EXPECTED_ROWS:-599700}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-32}"
PREDICT_CUDA_VISIBLE_DEVICES="${PREDICT_CUDA_VISIBLE_DEVICES:--1}"
PREDICT_TF_XLA_FLAGS="${PREDICT_TF_XLA_FLAGS:---tf_xla_auto_jit=0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"
source /home/dammerung/miniconda3/etc/profile.d/conda.sh
conda activate tensorflow

epoch_tag="$(printf '%04d' "$EPOCH")"
ckpt="$OUT/checkpoints/epoch_${epoch_tag}.weights.h5"
eval_dir="$OUT/fig2d_eval/epoch_${EPOCH}"
pred_dir="$eval_dir/p1800_t188_predictions"
eval_ckpt="$eval_dir/epoch_${epoch_tag}.weights.h5"
mkdir -p "$pred_dir"
rm -f "$eval_dir/evaluation.failed" "$eval_dir/evaluation.done"
trap 'touch "$eval_dir/evaluation.failed"' ERR
test -s "$ckpt"
test -s "$OUT/normalization.json"

# The training writer uses an atomic replace, and the shared lock covers the
# copy plus evaluation bookkeeping so another evaluator cannot prune/copy over it.
cp -f "$ckpt" "$eval_ckpt"
CUDA_VISIBLE_DEVICES="$PREDICT_CUDA_VISIBLE_DEVICES" \
  TF_XLA_FLAGS="$PREDICT_TF_XLA_FLAGS" \
  TF_CPP_MIN_LOG_LEVEL=2 PYTHONPATH=src python scripts/predict_distribution.py \
  --manifest-jsonl "$MANIFEST_JSONL" \
  --weights "$eval_ckpt" \
  --normalization "$OUT/normalization.json" \
  --output-dir "$pred_dir" \
  --condition P1800_T188 \
  --batch-size "$PREDICT_BATCH_SIZE" \
  --inner-dim 250 \
  --dropout 0.1 \
  --descriptor-dropout 0.1 \
  --m1 100 \
  --m2 100 \
  --head-names PC1,PC2 \
  --mc-dropout \
  --allow-cpu

for spec in \
  "PC1|$REF_PC1" \
  "PC2|$REF_PC2"; do
  IFS='|' read -r pc ref <<< "$spec"
  rows="$pred_dir/${pc}_predicted_rows.txt"
  metrics="$eval_dir/${pc}_vs_ref_metrics.json"
  gate="$eval_dir/${pc}_fig2d_gate.json"
  plot="$eval_dir/${pc}_fig2d_compare.png"
  test -s "$rows"
  PYTHONPATH=src python scripts/evaluate_fig2d.py \
    --predicted "$rows" \
    --reference "$ref" \
    --expected-rows "$EXPECTED_ROWS" \
    --output "$metrics" \
    --bins 100
  PYTHONPATH=src python scripts/plot_fig2d_eval.py \
    --predicted "$rows" \
    --reference "$ref" \
    --output "$plot" \
    --title "$pc epoch $EPOCH"
  PYTHONPATH=src python scripts/evaluate_fig2d_gate.py \
    --rows "$rows" \
    --reference "$ref" \
    --output "$gate" \
    --bins 120 \
    --prominence-fraction 0.08 \
    --min-peak-distance-bins 10 \
    --max-valley-fraction 0.25
done

EVAL_DIR="$eval_dir" EPOCH="$EPOCH" python - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["EVAL_DIR"])
epoch = int(os.environ["EPOCH"])
payload = {"epoch": epoch}
for pc in ("PC1", "PC2"):
    payload[pc] = json.loads((base / f"{pc}_vs_ref_metrics.json").read_text())
    payload[f"{pc}_gate"] = json.loads((base / f"{pc}_fig2d_gate.json").read_text())
(base / "fig2d_metrics_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    "epoch": epoch,
    "PC1_passed": payload["PC1_gate"].get("passed"),
    "PC2_passed": payload["PC2_gate"].get("passed"),
    "PC1_pc_wasserstein": payload["PC1"].get("pc", {}).get("wasserstein"),
    "PC2_pc_wasserstein": payload["PC2"].get("pc", {}).get("wasserstein"),
}, sort_keys=True))
PY
trap - ERR
touch "$eval_dir/evaluation.done"
