#!/usr/bin/env python3
"""Create a lightweight H5AD while preserving graph slots for scICEpy runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scICEpy.large_h5ad import create_light_h5ad


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .h5ad file")
    parser.add_argument("--output", required=True, help="Output .h5ad file")
    parser.add_argument("--n-vars", type=int, default=1, help="Number of feature columns to keep")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    create_light_h5ad(input_path=args.input, output_path=args.output, n_vars=args.n_vars)


if __name__ == "__main__":
    main()
