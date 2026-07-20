from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from auto_ipc_rc.data_manifest import build_manifest, write_manifest_artifacts
from auto_ipc_rc.normalization import fit_target_normalizer, write_normalizer
from auto_ipc_rc.splits import create_holdout_split, write_split_artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build fail-closed AutoIPC-RC data manifest artifacts.")
    parser.add_argument("roots", nargs="+", help="Dataset roots such as BalanceDataset and Dataset")
    parser.add_argument("--output-dir", required=True, help="Directory for frames/rejections/summary artifacts")
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--exclude-condition", default="P1800_T188")
    parser.add_argument("--allow-rejections", action="store_true", help="Write rejection report instead of failing closed")
    args = parser.parse_args(argv)

    manifest = build_manifest(args.roots, fail_closed=not args.allow_rejections)
    output_dir = Path(args.output_dir)
    write_manifest_artifacts(manifest, output_dir)
    split = create_holdout_split(
        manifest,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        exclude_condition=args.exclude_condition,
    )
    write_split_artifact(split, output_dir / "split.json")
    normalizer = fit_target_normalizer(manifest, split.train)
    write_normalizer(normalizer, output_dir / "normalization.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
