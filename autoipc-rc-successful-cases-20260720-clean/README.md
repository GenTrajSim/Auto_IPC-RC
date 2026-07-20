# AutoIPC-RC: Clean Successful Cases

## Included Cases

### PC1 single-head

- Constraint: `alpha=0.2`, `phi/pi=0.455`
- Epochs 1-4500: `learning_rate=1e-4`
- Epochs 4501-5000: resumed from the full TensorFlow checkpoint at `8e-5`
- Fig2d gate passed at epochs 4500 and 5000
- Epoch 4500: PC Wasserstein `3.60939`, rho-PC 2D JS `0.073998`
- Epoch 5000: PC Wasserstein `3.90834`, rho-PC 2D JS `0.072226`

### PC2 single-head

- Constraint: `alpha=0.4`, `phi/pi=0.490`
- Epochs 1-2000: `learning_rate=1e-4`
- Epochs 2001-3000: resumed from the full TensorFlow checkpoint at `8e-5`
- Fig2d gate passed at epochs 2000 and 3000
- Epoch 2000: PC Wasserstein `13.07950`, rho-PC 2D JS `0.137508`
- Epoch 3000: PC Wasserstein `14.12728`, rho-PC 2D JS `0.153335`

### Two-head paper-parity run

- Shared encoder with independent PC1 and PC2 decoder heads
- `M1=M2=100`, `batch_size=300`, `inner_dim=250`
- 20,000 unique frames sampled independently for each of two repeats
- Tail batch retained; shuffle buffer `400`
- Dropout and descriptor dropout `0.1`
- `learning_rate=1e-4`, `head_weights=1,1`, rank surrogate disabled
- MC-dropout used during Fig2d evaluation
- Both heads passed the Fig2d gate from epoch 1000 through epoch 4500

| Epoch | PC1 Wasserstein | PC1 rho-PC JS | PC2 Wasserstein | PC2 rho-PC JS |
|---:|---:|---:|---:|---:|
| 1000 | 2.8344 | 0.120951 | 10.6897 | 0.303953 |
| 1500 | 3.5388 | 0.108147 | 7.7332 | 0.230542 |
| 2000 | 3.2236 | 0.071910 | 6.1720 | 0.164873 |
| 3000 | 4.6177 | 0.086516 | 3.2938 | 0.079115 |
| 4000 | 3.3117 | 0.054427 | 1.6917 | 0.051242 |
| 4500 | 2.7527 | 0.039786 | 2.6521 | 0.047679 |

## Fixed Reproducibility Contract

The successful cases use the following contract:

1. The manifest represents 290,635 frames.
2. Each epoch samples 20,000 unique frames twice independently.
3. The final incomplete batch is retained.
4. The fixed normalization JSON is used for training and evaluation.
5. The encoder receives gradients only from rho/potential losses.
6. PC decoders receive gradients only from PC geometry losses.
7. PC loss uses Pearson, the original Spearman rank-difference formula, and the per-system slope/phi constraint.
8. PC head weights are absolute multipliers `1,1`.
9. Fig2d prediction uses MC-dropout and CPU isolation.
10. Full TensorFlow checkpoints preserve model weights, both Adam optimizers, optimizer slots, epoch, and global step.

## Release Layout

```text
code/                    Source, launchers, tests, and original model reference
data_contract/           Compressed manifest and fixed normalization JSON
results/                 Compact plots, gates, metrics, configs, and summaries
full_results/            Complete successful output trees, including checkpoints
release_manifest.json    Package metadata and endpoint metrics
SHA256SUMS.txt           Checksums for all clean-release files
```

The original frame files are not duplicated here. `data_contract/frames.jsonl.gz` contains the manifest paths and must be connected to the original frame storage or regenerated on the target workstation.

## Setup

```bash
cd autoipc-rc-successful-cases-20260720-clean
gzip -dk data_contract/frames.jsonl.gz
export MANIFEST_JSONL="$PWD/data_contract/frames.jsonl"
export NORMALIZATION_JSON="$PWD/data_contract/normalization.json"
export REF_PC1=/path/to/Linear_0.2/slope_455/logtest/xe_1800_188.txt
export REF_PC2=/path/to/Linear_0.4/slope_490/logtest/xe_1800_188.txt
```

Use the existing `tensorflow` conda environment or an equivalent Python environment containing TensorFlow, NumPy, SciPy, Matplotlib, and pytest.

## Tests

```bash
PYTHONPATH=code/src pytest -q \
  code/tests/test_losses_gradient.py \
  code/tests/test_training_step.py \
  code/tests/test_multi_head_model.py \
  code/tests/test_train_cli.py \
  code/tests/test_predict_distribution_cli.py \
  code/tests/test_paper_multi_head_launcher.py
```

## Single-Head Usage

For a 1,000-epoch milestone run:

```bash
cd code
PC_NAME=PC1 PC_ALPHA=0.2 PC_PHI=0.455 REF="$REF_PC1" \
  MANIFEST_JSONL="$MANIFEST_JSONL" NORMALIZATION_JSON="$NORMALIZATION_JSON" \
  OUT="$PWD/../runs/pc1_single_head" \
  bash scripts/run_paper_single_head_watch_eval.sh
```

Use `PC_NAME=PC2`, `PC_ALPHA=0.4`, `PC_PHI=0.490`, and `REF="$REF_PC2"` for PC2.

For a longer continuation, always use the full TensorFlow checkpoint prefix rather than a weights-only file:

```bash
PYTHONPATH=src python scripts/train.py \
  --manifest-jsonl "$MANIFEST_JSONL" --normalization-json "$NORMALIZATION_JSON" \
  --output-dir "$PWD/../runs/pc1_long" --epochs 4500 --initial-epoch 0 \
  --batch-size 300 --sample-frames-per-epoch 20000 --repeat-size 2 \
  --shuffle-buffer 400 --inner-dim 250 --m1 100 --m2 100 \
  --dropout 0.1 --descriptor-dropout 0.1 --learning-rate 1e-4 \
  --rank-weight 0 --head-weights 1 --pc-spec PC1:0.2:0.455 \
  --checkpoint-every 100 --checkpoint-keep 5

PYTHONPATH=src python scripts/train.py \
  --manifest-jsonl "$MANIFEST_JSONL" --normalization-json "$NORMALIZATION_JSON" \
  --output-dir "$PWD/../runs/pc1_long" --epochs 500 --initial-epoch 4500 \
  --resume-checkpoint "$PWD/../runs/pc1_long/checkpoints/tf_epoch_4500" \
  --batch-size 300 --sample-frames-per-epoch 20000 --repeat-size 2 \
  --shuffle-buffer 400 --inner-dim 250 --m1 100 --m2 100 \
  --dropout 0.1 --descriptor-dropout 0.1 --learning-rate 8e-5 \
  --rank-weight 0 --head-weights 1 --pc-spec PC1:0.2:0.455 \
  --checkpoint-every 100 --checkpoint-keep 5
```

For PC2, use `PC2:0.4:0.490`, `--epochs 2000` followed by `--epochs 1000`, and resume from `tf_epoch_2000`.

## Two-Head Usage

```bash
cd code
OUT="$PWD/../runs/multi_head" \
MANIFEST_JSONL="$MANIFEST_JSONL" \
NORMALIZATION_JSON="$NORMALIZATION_JSON" \
  bash scripts/run_paper_multi_head_parity_4500.sh
```

The launcher uses `M1=M2=100`, batch size `300`, independent repeat sampling, both PC heads, and the fixed paper constraints. Milestone evaluation can be run separately:

```bash
OUT="$OUT" REF_PC1="$REF_PC1" REF_PC2="$REF_PC2" \
MANIFEST_JSONL="$MANIFEST_JSONL" \
  bash scripts/watch_multi_head_eval.sh
```

To evaluate a completed epoch:

```bash
OUT="$OUT" REF_PC1="$REF_PC1" REF_PC2="$REF_PC2" \
MANIFEST_JSONL="$MANIFEST_JSONL" \
  bash scripts/eval_multi_head_epoch.sh 4500
```

## Publication Scope

Commit `code/`, `data_contract/`, `results/`, `README.md`, `release_manifest.json`, and `SHA256SUMS.txt` to a normal GitHub repository. Keep `full_results/` as a release asset or use Git LFS because it contains the complete checkpoints, raw prediction rows, and per-step metrics.
