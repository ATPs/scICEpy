"""Helpers for running scICEpy workflows on lightweight H5AD copies."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd

from .api import scICE_clustering
from .results import serialize_results_for_h5ad


def create_light_h5ad(input_path: str | Path, output_path: str | Path, n_vars: int = 1) -> Path:
    """Create a smaller H5AD that preserves graph and AnnData metadata."""

    source_path = Path(input_path)
    light_path = Path(output_path)
    light_path.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(source_path)
    n_keep = max(1, min(int(n_vars), int(adata.n_vars)))
    light = adata[:, :n_keep].copy()
    light.uns["scICEpy_light_h5ad"] = {
        "source_path": str(source_path),
        "n_obs": int(adata.n_obs),
        "n_vars_original": int(adata.n_vars),
        "n_vars_kept": int(n_keep),
        "obsp_keys": sorted(map(str, adata.obsp.keys())),
    }
    light.write_h5ad(light_path)
    return light_path


def _validate_mode_inputs(cluster_range: Sequence[int] | None, resolution: Sequence[float] | None) -> None:
    if cluster_range is None and resolution is None:
        raise ValueError("Provide either cluster_range or resolution.")
    if cluster_range is not None and resolution is not None:
        raise ValueError("cluster_range and resolution are mutually exclusive.")


def run_scice_on_light_h5ad(
    light_h5ad_path: str | Path,
    graph_key: str = "connectivities",
    cluster_range: Sequence[int] | None = None,
    resolution: Sequence[float] | None = None,
    n_workers: int = 10,
    outer_workers: int | None = None,
    inner_workers: int | None = None,
    n_trials: int = 15,
    n_bootstrap: int = 100,
    seed: int | None = None,
    beta: float = 0.1,
    n_iterations: int = 10,
    max_iterations: int = 150,
    ic_threshold: float = np.inf,
    objective_function: str = "CPM",
    remove_threshold: float = 1.15,
    min_cluster_size: int = 2,
    resolution_tolerance: float = 1e-8,
    verbose: bool = True,
    scratch_dir: str | None = None,
) -> tuple[dict, pd.Index]:
    """Run scICEpy on a lightweight H5AD and persist results into that file."""

    _validate_mode_inputs(cluster_range=cluster_range, resolution=resolution)

    light_path = Path(light_h5ad_path)
    adata = ad.read_h5ad(light_path)
    scICE_clustering(
        adata,
        graph_key=graph_key,
        cluster_range=None if cluster_range is None else list(cluster_range),
        n_workers=n_workers,
        outer_workers=outer_workers,
        inner_workers=inner_workers,
        n_trials=n_trials,
        n_bootstrap=n_bootstrap,
        seed=seed,
        beta=beta,
        n_iterations=n_iterations,
        max_iterations=max_iterations,
        ic_threshold=ic_threshold,
        objective_function=objective_function,
        remove_threshold=remove_threshold,
        min_cluster_size=min_cluster_size,
        resolution_tolerance=resolution_tolerance,
        verbose=verbose,
        resolution=None if resolution is None else list(resolution),
        scratch_dir=scratch_dir,
    )
    results = dict(adata.uns["scICE"])
    obs_names = pd.Index(adata.obs_names)
    adata.uns["scICE"] = serialize_results_for_h5ad(results)
    adata.write_h5ad(light_path)
    return results, obs_names


def write_scice_results_back(
    input_path: str | Path,
    results: dict,
    expected_obs_names: Sequence[str],
) -> Path:
    """Write scICE results into the original H5AD after validating cell order."""

    source_path = Path(input_path)
    expected_index = pd.Index(expected_obs_names)
    adata = ad.read_h5ad(source_path, backed="r+")
    try:
        if not pd.Index(adata.obs_names).equals(expected_index):
            raise ValueError("Original H5AD obs_names do not match the analyzed light H5AD.")
        adata.uns["scICE"] = serialize_results_for_h5ad(results)
        adata.write()
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    return source_path
