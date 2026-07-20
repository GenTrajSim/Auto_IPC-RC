# PC1 Fig2d Reproduction Template

## Status

This is the required baseline for subsequent AutoIPC-RC development. The paper-parity PC1 run passed the Fig2d dual-mode gate at epoch 4500 on 2026-07-12. A final low-learning-rate stage continued the same model and both Adam optimizers from epoch 4501 through 5000 at `8e-5`; epoch 5000 also passed the gate.

## Reference And Naming

- Constraint name: v8_plus `PC1`, corresponding to the current single output and the alpha `0.2`, phi/pi `0.455` reference.
- Reference rows: `/home/dammerung/workstation/train/ALL_Map_PT_ulta_smallNewtwork/Linear_0.2/slope_455/logtest/xe_1800_188.txt`
- Run directory: `/home/dammerung/workstation/train/workspace/project01-AutoIPC-RC/output/paper_single_head_PC1_a0.2_p0.455_lr1e4_indrepeat2_tail_mcdo_4500_20260710`
- Epoch-4500 comparison: `<run>/fig2d_eval/epoch_4500/PC1_fig2d_compare.png`
- Epoch-5000 comparison: `<run>/fig2d_eval/epoch_5000/PC1_fig2d_compare.png`

## Non-Negotiable Model And Loss Contract

- Single independent PC decoder head before any multi-head extension.
- Shared encoder for rho and potential; PC decoder is optimized separately.
- `M1=M2=100`, inner dimension 250, 30 neighbors, 4 descriptor features.
- Encoder optimizer receives only rho/potential loss gradients.
- Decoder optimizer receives only PC geometry loss gradients.
- PC loss uses Pearson alpha target `0.2`, original Spearman rank-difference reporting, and phi/pi slope target `0.455`; no added rank surrogate.
- Dropout `0.1` and descriptor dropout `0.1`.

## Non-Negotiable Data Contract

- Manifest contains 290,635 frames from Dataset and BalanceDataset.
- Sample 20,000 unique frames independently for each of two repeat passes.
- Expected epoch exposure is 40,000 rows; the fixed-seed smoke check covered 38,676 distinct frames.
- Batch size is 300 and the final 100-frame tail batch is retained, giving 134 steps per epoch.
- Shuffle buffer is 400.
- Use the fixed paper normalization JSON at `outputs/preflight/fhi47_paper_20260706/normalization.json`.

## Training Schedule

### Stage 1

- Epochs 1-4500 from scratch.
- Learning rate `1e-4`, seed `2026`.
- Checkpoint every 100 epochs; retain the latest five.

### Stage 2

- Restore full TensorFlow checkpoint `checkpoints/tf_epoch_4500`.
- Preserve both Adam iteration counters and moment slots.
- Explicitly override both restored learning-rate variables to `8e-5`.
- Train epochs 4501-5000, seed `2027`.
- Do not use weights-only resume.

## Epoch-4500 Evidence

- Gate: passed.
- Predicted PC standard deviation `14.21299`; reference `14.69779`.
- PC Wasserstein distance `3.60939`.
- rho-PC 2D JS divergence `0.073998`.
- Predicted dual-peak distance ratio `0.82905` relative to reference.
- Predicted valley fraction `0.13673`; reference `0.14407`.
- Rho Wasserstein distance `0.005503`; potential Wasserstein distance `0.00004656`.

## Epoch-5000 Evidence

- Training completed at global step and both optimizer iterations `670000`; both checkpointed learning rates are `8e-5`.
- Gate: passed.
- Predicted PC standard deviation `14.16652`; reference `14.69779`.
- PC Wasserstein distance `3.90834`.
- rho-PC 2D JS divergence `0.072226`.
- Predicted dual-peak distance ratio `0.83889` relative to reference.
- Predicted valley fraction `0.13609`; reference `0.14407`.
- Rho Wasserstein distance `0.008023`; potential Wasserstein distance `0.00006241`.
- Predicted peak count is three while the gate's reference detector finds two; the extra central/right ridge remains a residual topology difference.

## Checkpoint Selection Lesson

- Epoch 5000 slightly improved relative structural metrics over epoch 4500: rho-PC 2D JS `0.073998 -> 0.072226`, peak-distance ratio `0.82905 -> 0.83889`, and valley fraction `0.13673 -> 0.13609`.
- It did not uniformly improve global alignment: PC Wasserstein changed `3.60939 -> 3.90834`, rho Wasserstein `0.005503 -> 0.008023`, and potential Wasserstein `0.00004656 -> 0.00006241`.
- Preserve both checkpoints. Use epoch 4500 as the stronger global-distribution alignment baseline and epoch 5000 as the completed low-LR structural-refinement endpoint. Do not assume later epochs dominate every criterion.
- MC-dropout evaluation is stochastic, so small differences should not be overinterpreted without a fixed evaluation seed or repeated uncertainty estimate.

## Evaluation And Reliability Contract

- Fig2d prediction must use model `training=True` via `--mc-dropout`.
- Training and full CPU evaluation share `<run>/train-eval.lock`; never overlap them.
- Do not use `nvidia-smi` as a monitor on this workstation.
- Monitor using process state, `metrics.csv`, checkpoint mtimes, and evaluation summaries.

## Development Gate

PC2 must first reproduce its own single-head Fig2d result under this contract. Multi-head work may begin only after both single-head baselines pass. Multi-head changes must be minimal additions and must not alter sampling, normalization, encoder/decoder optimizer ownership, checkpoint restore semantics, or MC-dropout evaluation.

## Failed Attempts And Anti-Patterns

### Confirmed Semantic Failures

1. **Sampling once and tiling the same 20,000 frames twice.** This produced only 20,000 distinct frames per epoch instead of the roughly 38,600 distinct frames produced by two independent generator invocations. It materially reduced training diversity even though the exposure counter still said 40,000.
2. **Dropping the final incomplete batch.** The refactor used 133 full batches while the paper path retained the final 100 frames and used 134 steps. Do not use `--drop-remainder` in paper-parity runs.
3. **Deterministic Fig2d inference.** The original test activates dropout with `training=True`. Evaluating with `training=False` narrowed and distorted the PC distribution, so all reproduction gates must use `--mc-dropout`.
4. **Changing several scientific semantics at once.** Earlier attempts removed descriptor dropout or original Spearman reporting, added a rank surrogate, changed head-loss scale, and altered head weighting. Those runs could not distinguish an implementation regression from an optimization effect. Add one feature only after the unchanged single-head baseline passes.

### Training-Loop And Resource Failures

1. **Eager per-step training and repeated optimizer bookkeeping.** This caused Python/TensorFlow object growth and poor throughput. Keep one compiled `@tf.function` train step and build optimizer slots once.
2. **Every-epoch HDF5 checkpoint pairs.** Excessive checkpoint I/O contributed to stalls and made the long run fragile. Save every 100 epochs and retain only the latest five, while preserving explicitly named lineage checkpoints outside the pruning directory.
3. **Concurrent full CPU prediction and GPU training.** The previous watcher began a 599,700-row evaluation while training continued; host memory pressure then coincided with the training process disappearing. Training and evaluation must share `train-eval.lock`.
4. **Target-loop process restarts.** Training separately to targets 50, 100, and later targets restarted process state and made optimizer continuity ambiguous. One uninterrupted stage is preferred; any planned LR transition must resume a full TensorFlow checkpoint and verify optimizer iterations.

### Failed Batch-Size Diagnosis

- Batch-size reductions from 300 to 250, 200, and 150 did not address the semantic root cause. Observed failures included an uninterruptible-I/O stall near epoch 49, exit status 137 with host memory near 90 percent, and continued memory growth. Do not treat batch size as the first tuning lever unless a reproducible single-step memory limit proves it is necessary.
- Descriptor-dropout/rank variants at batch sizes 300 and 250 also failed with OOM. Those experiments combined architecture/loss changes with memory changes and therefore are not valid Fig2d baselines.

### Misleading Diagnostic Paths

1. **Blaming learning rate before parity checks.** The four historical successful cases formed their main topology at `1e-4`; their lower learning rates were short late-stage refinements. Check data traversal, evaluation mode, and optimizer ownership before changing LR.
2. **Comparing raw alpha loss to weighted historical output.** Current `PC1_alpha_loss` is raw, while historical `loss_correlation` includes the factor 100. Compare `PC1_alpha_loss * 100` with the historical field.
3. **Monitoring with `nvidia-smi`.** It has hung on this workstation. Use process state, metrics growth, checkpoint mtimes, and lock state instead.
4. **Confusing host/service failures with model failures.** OpenVPN restart loops and workstation reboots were infrastructure incidents; preserve training checkpoints and diagnose journal/process evidence separately rather than changing model parameters.
5. **Assuming low-LR continuation must improve every metric.** Epoch 5000 retained the gate and slightly improved structural metrics but worsened several global marginal distances relative to epoch 4500. Keep milestone checkpoints and select by the downstream criterion.

## Required Failure-Prevention Checklist

- Verify independent repeat sampling and 40,000 exposures before every new baseline.
- Verify `drop_remainder=false` and 134 steps per epoch.
- Verify full checkpoint contents include both optimizers, slots, epoch, and global step.
- After resume, print and check both optimizer iterations and both effective learning rates before accepting the run.
- Never overlap training and full Fig2d prediction.
- Require single-head PC1 and PC2 parity before enabling multi-head changes.
