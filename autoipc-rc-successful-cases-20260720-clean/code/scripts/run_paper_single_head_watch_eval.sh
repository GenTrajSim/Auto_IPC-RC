#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTOIPC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MANIFEST_JSONL="${MANIFEST_JSONL:-$ROOT/../data_contract/frames.jsonl}"
NORMALIZATION_JSON="${NORMALIZATION_JSON:-$ROOT/../data_contract/normalization.json}"
if [[ ! -s "$MANIFEST_JSONL" && -s "$ROOT/outputs/preflight/data_gate_20260703/frames.jsonl" ]]; then
  MANIFEST_JSONL="$ROOT/outputs/preflight/data_gate_20260703/frames.jsonl"
fi
if [[ ! -s "$NORMALIZATION_JSON" && -s "$ROOT/outputs/preflight/fhi47_paper_20260706/normalization.json" ]]; then
  NORMALIZATION_JSON="$ROOT/outputs/preflight/fhi47_paper_20260706/normalization.json"
fi
[[ -s "$MANIFEST_JSONL" ]] || { echo "manifest not found: $MANIFEST_JSONL" >&2; exit 2; }
[[ -s "$NORMALIZATION_JSON" ]] || { echo "normalization JSON not found: $NORMALIZATION_JSON" >&2; exit 2; }
PC_NAME="${PC_NAME:-PC1}"
PC_ALPHA="${PC_ALPHA:-0.2}"
PC_PHI="${PC_PHI:-0.455}"
REF_DEFAULT="/home/dammerung/workstation/train/ALL_Map_PT_ulta_smallNewtwork/Linear_0.2/slope_455/logtest/xe_1800_188.txt"
REF="${REF:-$REF_DEFAULT}"
OUT="${OUT:-$ROOT/../runs/paper_single_head_${PC_NAME}_a${PC_ALPHA}_p${PC_PHI}_m1_100_m2_100_sample20k}"
LOG="$OUT/single_head_train_eval.log"
TRAIN_LOG="$OUT/train.log"
TARGETS=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)
POLL_SECONDS="${POLL_SECONDS:-180}"

cd "$ROOT"
mkdir -p "$OUT"

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG"
}

cleanup() {
  local status=$?
  if [[ -n "${TRAIN_PID:-}" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    log "received stop signal; terminating single-head training pid=$TRAIN_PID"
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup INT TERM

log "paper single-head training/eval loop start"
log "config: pc=${PC_NAME} alpha=${PC_ALPHA} phi=${PC_PHI} m1=100 m2=100 batch_size=300 sample_frames_per_epoch=20000 repeat_size=2 dropout=0.1 descriptor_dropout=0.1 lr=1e-4 checkpoint_every=10 checkpoint_keep=12"
log "reference: $REF"
log "manifest: $MANIFEST_JSONL"
log "normalization: $NORMALIZATION_JSON"

TF_CPP_MIN_LOG_LEVEL=2 PYTHONPATH=src python scripts/train.py \
  --manifest-jsonl "$MANIFEST_JSONL" \
  --output-dir "$OUT" \
  --epochs 1000 \
  --batch-size 300 \
  --sample-frames-per-epoch 20000 \
  --repeat-size 2 \
  --inner-dim 250 \
  --descriptor-dropout 0.1 \
  --dropout 0.1 \
  --m1 100 \
  --m2 100 \
  --normalization-json "$NORMALIZATION_JSON" \
  --learning-rate 1.0e-4 \
  --rank-weight 0.0 \
  --pc-spec "${PC_NAME}:${PC_ALPHA}:${PC_PHI}" \
  --head-weights 1 \
  --checkpoint-every 10 \
  --checkpoint-keep 12 \
  --log-every 100 \
  > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$OUT/train.pid"
log "started train pid=$TRAIN_PID"

wait_for_checkpoint() {
  local target="$1"
  local ckpt="$OUT/checkpoints/epoch_$(printf '%04d' "$target").weights.h5"
  while [[ ! -f "$ckpt" ]]; do
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
      log "train pid=$TRAIN_PID exited before checkpoint epoch_$target appeared"
      wait "$TRAIN_PID"
      return 1
    fi
    sleep "$POLL_SECONDS"
  done
  return 0
}

evaluate_target() {
  local target="$1"
  local ckpt="$OUT/checkpoints/epoch_$(printf '%04d' "$target").weights.h5"
  local eval_dir="$OUT/fig2d_eval/epoch_${target}"
  local pred_dir="$eval_dir/p1800_t188_predictions"
  local eval_ckpt="$eval_dir/epoch_$(printf '%04d' "$target").weights.h5"
  mkdir -p "$pred_dir"
  if [[ -f "$eval_dir/fig2d_metrics_summary.json" ]]; then
    log "target=$target evaluation already complete; skipping"
    return 0
  fi
  cp -f "$ckpt" "$eval_ckpt"
  log "target=$target checkpoint copied for CPU evaluation"

  CUDA_VISIBLE_DEVICES="-1" TF_XLA_FLAGS="--tf_xla_auto_jit=0" TF_CPP_MIN_LOG_LEVEL=2 PYTHONPATH=src python scripts/predict_distribution.py \
    --manifest-jsonl "$MANIFEST_JSONL" \
    --weights "$eval_ckpt" \
    --normalization "$OUT/normalization.json" \
    --output-dir "$pred_dir" \
    --condition P1800_T188 \
    --batch-size 32 \
    --inner-dim 250 \
    --descriptor-dropout 0.1 \
    --dropout 0.1 \
    --m1 100 \
    --m2 100 \
    --head-names "$PC_NAME" \
    --mc-dropout \
    --allow-cpu 2>&1 | tee -a "$LOG"

  local rows="$pred_dir/${PC_NAME}_predicted_rows.txt"
  PYTHONPATH=src python scripts/evaluate_fig2d.py \
    --predicted "$rows" \
    --reference "$REF" \
    --expected-rows 599700 \
    --output "$eval_dir/${PC_NAME}_vs_ref_metrics.json" \
    --bins 100
  python scripts/plot_fig2d_eval.py \
    --predicted "$rows" \
    --reference "$REF" \
    --output "$eval_dir/${PC_NAME}_fig2d_compare.png" \
    --title "${PC_NAME} epoch $target alpha=${PC_ALPHA} phi=${PC_PHI}"
  PYTHONPATH=src python scripts/evaluate_fig2d_gate.py \
    --rows "$rows" \
    --reference "$REF" \
    --output "$eval_dir/${PC_NAME}_fig2d_gate.json" \
    --bins 120 \
    --prominence-fraction 0.08 \
    --min-peak-distance-bins 10 \
    --max-valley-fraction 0.25
  EVAL_DIR="$eval_dir" TARGET_EPOCH="$target" PC_NAME="$PC_NAME" PYTHONPATH=src python - <<'PY2' | tee -a "$LOG"
import os
import json
import pathlib
base = pathlib.Path(os.environ["EVAL_DIR"])
pc_name = os.environ["PC_NAME"]
p = json.loads((base / f"{pc_name}_vs_ref_metrics.json").read_text())
summary = {
    "epoch": int(os.environ["TARGET_EPOCH"]),
    "pc_name": pc_name,
    "pc_js": p["pc"]["js_divergence"],
    "pc_wasserstein": p["pc"]["wasserstein"],
    "rho_pc_2d_js": p["rho_pc_2d"]["js_divergence_2d"],
    "potential_pc_2d_js": p["potential_pc_2d"]["js_divergence_2d"],
    "pred_pc_mean": p["pc"]["pred_mean"],
    "ref_pc_mean": p["pc"]["ref_mean"],
}
gate_path = base / f"{pc_name}_fig2d_gate.json"
if gate_path.exists():
    gate = json.loads(gate_path.read_text())
    summary["dual_mode_passed"] = gate["passed"]
    summary["dual_mode_reason"] = gate["reason"]
    summary["distance_ratio"] = gate.get("distance_ratio")
    summary["allowed_valley_fraction"] = gate.get("allowed_valley_fraction")
    summary["pred_peak_count"] = gate["predicted"]["peak_count"]
    summary["pred_peak_distance_bins"] = gate["predicted"]["best_peak_distance_bins"]
    summary["pred_valley_fraction"] = gate["predicted"]["best_valley_fraction"]
    summary["ref_peak_distance_bins"] = gate["reference"]["best_peak_distance_bins"]
    summary["ref_valley_fraction"] = gate["reference"]["best_valley_fraction"]
(base / "fig2d_metrics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY2
  log "target=$target evaluation complete"
}

for target in "${TARGETS[@]}"; do
  log "waiting for checkpoint epoch_$target"
  wait_for_checkpoint "$target"
  evaluate_target "$target"
done

if wait "$TRAIN_PID"; then
  log "single-head training process completed"
else
  status=$?
  log "single-head training process failed with status=$status"
  exit "$status"
fi
log "paper single-head training/eval loop complete"
