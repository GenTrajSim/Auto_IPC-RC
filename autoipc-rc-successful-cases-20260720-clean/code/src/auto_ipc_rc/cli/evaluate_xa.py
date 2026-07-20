from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from auto_ipc_rc.xa import ABPlane, compute_frame_xa, summarize_xa


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute frame-level xA from PCI/PCII potential/rho/PC rows.")
    parser.add_argument("--pcii-rows", required=True, help="Rows from alpha=0.2/phi=455 head: potential rho PC")
    parser.add_argument("--pci-rows", required=True, help="Rows from alpha=0.4/phi=490 head: potential rho PC")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--c1", type=float, required=True)
    parser.add_argument("--c2", type=float, required=True)
    parser.add_argument("--c3", type=float, required=True)
    parser.add_argument("--m-cut", type=float, required=True)
    parser.add_argument("--particles-per-frame", type=int, default=300)
    args = parser.parse_args(argv)

    plane = ABPlane(c1=args.c1, c2=args.c2, c3=args.c3, m_cut=args.m_cut)
    pcii = np.loadtxt(args.pcii_rows, dtype=np.float64)
    pci = np.loadtxt(args.pci_rows, dtype=np.float64)
    result = compute_frame_xa(pcii, pci, plane=plane, particles_per_frame=args.particles_per_frame)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_dir / "xA_timeseries.csv",
        result.as_rows(),
        delimiter=",",
        header="mean_RC,mean_Q1_PCII_PC,mean_Q2_PCI_rho,mean_Q3_PCI_PC,xA",
        comments="",
        fmt="%.10g",
    )
    summary = summarize_xa(result, plane=plane, particles_per_frame=args.particles_per_frame)
    (output_dir / "xA_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
