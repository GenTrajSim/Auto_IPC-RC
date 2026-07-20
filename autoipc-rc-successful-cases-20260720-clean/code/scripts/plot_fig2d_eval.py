#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot predicted/reference Fig2d-style distributions and xA histograms.")
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Fig2d distribution")
    parser.add_argument("--bins", type=int, default=120)
    args = parser.parse_args(argv)

    pred = np.loadtxt(args.predicted, dtype=np.float64)
    ref = np.loadtxt(args.reference, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, rows, title in [(axes[0], ref, "reference"), (axes[1], pred, "predicted")]:
        hist = ax.hist2d(rows[:, 1], rows[:, 2], bins=args.bins, cmap="magma", cmin=1)
        ax.set_title(title)
        ax.set_xlabel("rho / Q2")
        ax.set_ylabel("PC")
        fig.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(args.title)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
