# Multi-Head Paper-Parity Training Template

## Current Run

- Output: `/home/dammerung/workstation/train/workspace/project01-AutoIPC-RC/output/multi_head_paper_parity_m1_100_m2_100_lr1e4_4500_barrier_20260716`
- Launcher: `scripts/run_paper_multi_head_parity_4500.sh`
- Watcher: `scripts/watch_multi_head_eval.sh`
- Milestones: `100, 500, 1000, 1500, 2000, 3000, 4000, 4500`

## Fixed Contract

- Shared encoder for rho/potential; two independent PC decoder heads.
- `M1=M2=100`, inner dimension 250, batch size 300, 30 neighbors, four descriptor channels.
- Each epoch samples 20,000 unique frames twice independently, retains the final 100-frame tail, and uses shuffle buffer 400.
- Dropout and descriptor dropout are both `0.1`; learning rate is `1e-4`; rank surrogate is disabled.
- PC1 is `alpha=0.2`, `phi/pi=0.455`; PC2 is `alpha=0.4`, `phi/pi=0.490`.
- `head_weights=1,1` are absolute multipliers. They are not normalized to `0.5,0.5`.
- Encoder gradients come only from rho/potential loss; decoder gradients come only from PC geometry loss.
- Fig2d prediction uses MC-dropout (`training=True`) and the fixed paper normalization JSON.

## Checkpoint And Evaluation Safety

- Save every 100 epochs and retain the newest five training checkpoints.
- Each milestone is copied into its own `fig2d_eval/epoch_N` directory before pruning can remove it.
- Training pauses at a milestone until the full CPU evaluation writes `evaluation.done`.
- The checkpoint lock covers checkpoint writes and evaluation; full prediction never overlaps GPU training.
- A failed evaluation writes `evaluation.failed` and stops the trainer instead of silently continuing.
- Use process state, `metrics.csv`, checkpoint mtimes, and evaluation summaries for monitoring; do not use `nvidia-smi` on this workstation.

## Acceptance Evidence

At each milestone retain `PC1_fig2d_compare.png`, `PC2_fig2d_compare.png`, both distribution metric JSON files, both gate JSON files, and `fig2d_metrics_summary.json`. Judge both absolute marginals and relative dual-mode structure; do not accept a result from only one head or from raw coordinate alignment alone.
