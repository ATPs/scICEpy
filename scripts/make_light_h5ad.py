#!/usr/bin/env python3
"""Create a lightweight H5AD while preserving graph slots for scICEpy runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input .h5ad file")
    parser.add_argument("--output", required=True, help="Output .h5ad file")
    parser.add_argument("--n-vars", type=int, default=1, help="Number of feature columns to keep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_path)
    n_keep = max(1, min(int(args.n_vars), int(adata.n_vars)))
    light = adata[:, :n_keep].copy()
    light.uns.setdefault("scICEpy_light_h5ad", {})
    light.uns["scICEpy_light_h5ad"] = {
        "source_path": str(input_path),
        "n_obs": int(adata.n_obs),
        "n_vars_original": int(adata.n_vars),
        "n_vars_kept": int(n_keep),
        "obsp_keys": sorted(map(str, adata.obsp.keys())),
    }
    light.write_h5ad(output_path)


if __name__ == "__main__":
    main()
