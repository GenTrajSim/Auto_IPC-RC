from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multi_head_launcher_pins_paper_parity_contract() -> None:
    text = (ROOT / "scripts" / "run_paper_multi_head_parity_4500.sh").read_text(encoding="utf-8")

    for fragment in (
        "--m1 100",
        "--m2 100",
        "--batch-size 300",
        "--sample-frames-per-epoch 20000",
        "--repeat-size 2",
        "--shuffle-buffer 400",
        "--dropout 0.1",
        "--descriptor-dropout 0.1",
        "--learning-rate 1e-4",
        "--head-weights 1,1",
        "--pc-spec PC1:0.2:0.455",
        "--pc-spec PC2:0.4:0.490",
        "--checkpoint-every 100",
        "--checkpoint-keep 5",
        "TRAIN_EVAL_LOCK_FILE",
        "TRAIN_EVAL_BARRIER_DIR",
        "TRAIN_EVAL_BARRIER_EPOCHS",
    ):
        assert fragment in text
    assert "--drop-remainder" not in text


def test_multi_head_evaluator_pins_mc_dropout_and_both_reference_heads() -> None:
    text = (ROOT / "scripts" / "eval_multi_head_epoch.sh").read_text(encoding="utf-8")
    assert 'PREDICT_CUDA_VISIBLE_DEVICES:--1' in text
    assert 'CUDA_VISIBLE_DEVICES="$PREDICT_CUDA_VISIBLE_DEVICES"' in text
    assert "--head-names PC1,PC2" in text
    assert "--mc-dropout" in text
    assert "--allow-cpu" in text
    assert '"PC1|$REF_PC1"' in text
    assert '"PC2|$REF_PC2"' in text


def test_recovery_launcher_restores_full_checkpoint_before_resuming() -> None:
    text = (ROOT / "scripts" / "recover_multi_head_from_epoch.sh").read_text(encoding="utf-8")
    assert "tf_epoch_" in text
    assert "eval_multi_head_epoch.sh" in text
    assert "RESUME_CHECKPOINT" in text
    assert "REMAINING_EPOCHS" in text
