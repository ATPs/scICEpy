#!/usr/bin/env python3
"""Run scICEpy on a lightweight H5AD copy and write results back to the source file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scICEpy.large_h5ad import create_light_h5ad, run_scice_on_light_h5ad, write_scice_results_back


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scICEpy on a lightweight H5AD copy and write results back to the original H5AD.",
    )
    parser.add_argument("--input", required=True, help="Original input .h5ad file")
    parser.add_argument("--light-output", required=True, help="Output path for the lightweight .h5ad file")
    parser.add_argument("--n-vars", type=int, default=1, help="Number of feature columns to keep in the light file")
    parser.add_argument("--graph-key", default="connectivities", help="Graph key in adata.obsp")
    parser.add_argument("--n-workers", type=int, default=10, help="Top-level worker budget")
    parser.add_argument("--outer-workers", type=int, default=None, help="Optional outer worker cap")
    parser.add_argument("--inner-workers", type=int, default=None, help="Optional inner worker cap")
    parser.add_argument("--n-trials", type=int, default=15, help="Leiden trials per gamma")
    parser.add_argument("--n-bootstrap", type=int, default=100, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter retained for compatibility")
    parser.add_argument("--n-iterations", type=int, default=10, help="Leiden iterations per trial")
    parser.add_argument("--max-iterations", type=int, default=150, help="Maximum optimization iterations")
    parser.add_argument("--ic-threshold", type=float, default=float("inf"), help="IC threshold for summary fields")
    parser.add_argument(
        "--objective-function",
        choices=["CPM", "modularity"],
        default="CPM",
        help="Leiden objective function",
    )
    parser.add_argument("--remove-threshold", type=float, default=1.15, help="Cluster-range pre-filter threshold")
    parser.add_argument("--min-cluster-size", type=int, default=2, help="Minimum final cluster size")
    parser.add_argument(
        "--resolution-tolerance",
        type=float,
        default=1e-8,
        help="Search tolerance used in cluster-range mode",
    )
    parser.add_argument("--scratch-dir", default=None, help="Optional temp root for runtime files")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--cluster-range",
        nargs="+",
        type=int,
        help="Requested target cluster counts for cluster-range mode",
    )
    mode_group.add_argument(
        "--resolution",
        nargs="+",
        type=float,
        help="Manual gamma values for resolution mode",
    )

    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument("--verbose", dest="verbose", action="store_true", help="Enable verbose logging")
    verbosity_group.add_argument("--quiet", dest="verbose", action="store_false", help="Disable verbose logging")
    parser.set_defaults(verbose=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    create_light_h5ad(input_path=args.input, output_path=args.light_output, n_vars=args.n_vars)
    results, obs_names = run_scice_on_light_h5ad(
        light_h5ad_path=args.light_output,
        graph_key=args.graph_key,
        cluster_range=args.cluster_range,
        resolution=args.resolution,
        n_workers=args.n_workers,
        outer_workers=args.outer_workers,
        inner_workers=args.inner_workers,
        n_trials=args.n_trials,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        beta=args.beta,
        n_iterations=args.n_iterations,
        max_iterations=args.max_iterations,
        ic_threshold=args.ic_threshold,
        objective_function=args.objective_function,
        remove_threshold=args.remove_threshold,
        min_cluster_size=args.min_cluster_size,
        resolution_tolerance=args.resolution_tolerance,
        verbose=args.verbose,
        scratch_dir=args.scratch_dir,
    )
    write_scice_results_back(
        input_path=args.input,
        results=results,
        expected_obs_names=obs_names,
    )


if __name__ == "__main__":
    main()
