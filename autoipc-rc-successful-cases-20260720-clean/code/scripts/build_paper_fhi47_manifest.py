#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter frames.jsonl to the 47 source data dirs used by dp_LDL_simple_fhi47_100_linear.py.")
    parser.add_argument("--reference", default="references/dp_LDL_simple_fhi47_100_linear.py")
    parser.add_argument("--input", required=True, help="Input frames.jsonl")
    parser.add_argument("--output", required=True, help="Filtered output frames.jsonl")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path")
    args = parser.parse_args(argv)

    ref_text = Path(args.reference).read_text(encoding="utf-8", errors="ignore")
    allowed = set(
        re.findall(
            r"pd\.read_csv\('../../../dp_LDL/([^/]+)/([^/]+)/(data[^/]+)/boxdata\.csv'\)",
            ref_text,
        )
    )
    if not allowed:
        raise RuntimeError("no fhi47 boxdata.csv paths found in reference script")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    kept = 0
    by_key: dict[str, int] = {}
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            total += 1
            row = json.loads(line)
            key = (row["dataset"], row["condition"], row["data_dir"])
            if key not in allowed:
                continue
            kept += 1
            by_key["/".join(key)] = by_key.get("/".join(key), 0) + 1
            dst.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "reference": str(args.reference),
        "input": str(input_path),
        "output": str(output_path),
        "input_frames": total,
        "kept_frames": kept,
        "allowed_data_dirs": len(allowed),
        "kept_data_dirs": len(by_key),
        "counts_by_data_dir": dict(sorted(by_key.items())),
    }
    if kept != 51835:
        summary["warning"] = "kept frame count differs from the original fhi47 boxdata row count observed in the reference script"
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
