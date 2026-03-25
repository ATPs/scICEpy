"""Public AnnData-facing API for scICEpy."""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

from .clustering_inputs import (
    _extract_graph,
    _format_cluster_values,
    _normalize_cluster_range,
    _normalize_resolution_values,
    _validate_common_inputs,
)
from .clustering_modes import _build_manual_resolution_results, _run_cluster_range_mode
from .clustering_reporting import _log_results_summary
from .leiden_wrapper import beta_support_status
from .results import attach_summary_fields
from .runtime import (
    cleanup_runtime_spill,
    clear_clustering_cache,
    create_runtime_context,
    logger,
    resolve_effective_workers,
)

def scICE_clustering(
    adata,
    graph_key: str = "connectivities",
    cluster_range=None,
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
    resolution=None,
    copy: bool = False,
    scratch_dir: str | None = None,
):
    """Run scICE clustering on an AnnData object and store the results in `adata.uns["scICE"]`."""
    total_start = time.time()
    if copy:
        adata = adata.copy()

    requested_workers = int(n_workers)
    min_cluster_size = int(min_cluster_size)
    _validate_common_inputs(adata, graph_key, requested_workers, min_cluster_size, objective_function)
    if outer_workers is not None and int(outer_workers) < 1:
        raise ValueError("outer_workers must be >= 1 when provided.")
    if inner_workers is not None and int(inner_workers) < 1:
        raise ValueError("inner_workers must be >= 1 when provided.")

    resolution_mode = resolution is not None
    requested_cluster_range = None if resolution_mode else _normalize_cluster_range(cluster_range)
    resolution_values = _normalize_resolution_values(resolution) if resolution_mode else None
    worker_layout = resolve_effective_workers(requested_workers)
    n_workers = int(worker_layout["effective"])
    runtime_context = create_runtime_context(scratch_dir=scratch_dir)
    beta_status = beta_support_status()
    if (
        not bool(beta_status["supported"])
        and np.isfinite(beta)
        and not np.isclose(float(beta), float(beta_status["default"]))
    ):
        warnings.warn(
            (
                f"scICEpy received beta={float(beta):.6g}, but {beta_status['reason']} "
                "The value is retained for API compatibility and cache keys only."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    if verbose:
        logger.info("=" * 80)
        logger.info("Starting scICE clustering analysis...")
        logger.info("Timestamp: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("Python: %s", sys.version.split()[0])
        logger.info("Platform: %s", sys.platform)
        logger.info("Process ID: %s", os.getpid())
        logger.info("-" * 80)
        logger.info("INPUT PARAMETERS:")
        logger.info("  Using graph: %s", graph_key)
        if resolution_mode:
            logger.info("  Analysis mode: manual resolution")
            logger.info("  Manual resolutions: %s", _format_cluster_values(resolution_values))
            logger.info("  Resolution count: %s", len(resolution_values))
        else:
            logger.info("  Analysis mode: cluster range search")
            logger.info("  Testing cluster range: %s", _format_cluster_values(requested_cluster_range))
            logger.info(
                "  Range: %s-%s (%s values)",
                int(requested_cluster_range.min()),
                int(requested_cluster_range.max()),
                len(requested_cluster_range),
            )
        logger.info("  Requested workers: %s", worker_layout["requested"])
        logger.info("  Effective workers: %s", worker_layout["effective"])
        logger.info(
            "  Requested outer workers: %s",
            "auto" if outer_workers is None else int(outer_workers),
        )
        logger.info(
            "  Requested inner workers: %s",
            "auto" if inner_workers is None else int(inner_workers),
        )
        logger.info(
            "  Internal memory budget (bytes): %s",
            format(int(runtime_context.memory_budget_bytes), ","),
        )
        logger.info("  Runtime temp root: %s", runtime_context.scratch_root)
        logger.info("  Runtime temp dir: %s", runtime_context.runtime_dir)
        logger.info("  Number of trials per resolution: %s", n_trials)
        logger.info("  Number of bootstrap iterations: %s", n_bootstrap)
        logger.info("  Random seed: %s", "NULL (random)" if seed is None else seed)
        logger.info("  Beta parameter: %s", beta)
        logger.info("  Beta supported by current Python backend: %s", bool(beta_status["supported"]))
        if not bool(beta_status["supported"]):
            logger.info("  Beta backend note: %s", str(beta_status["reason"]))
        logger.info("  Leiden iterations: %s", n_iterations)
        logger.info("  Maximum optimization iterations: %s", max_iterations)
        logger.info("  IC threshold: %s", ic_threshold)
        logger.info("  Objective function: %s", objective_function)
        if resolution_mode:
            logger.info("  Remove threshold: ignored in manual resolution mode")
        else:
            logger.info("  Remove threshold: %s", remove_threshold)
        logger.info("  Minimum cluster size: %s", min_cluster_size)
        if min_cluster_size > 1:
            logger.info("  min_cluster_size semantics: counting uses effective clusters; final best_labels are merged")
        if resolution_mode:
            logger.info("  Resolution tolerance: not used in manual resolution mode")
        else:
            logger.info("  Resolution tolerance: %s", resolution_tolerance)
        logger.info("-" * 80)
        logger.info("PARALLEL PROCESSING SETUP:")
        logger.info("  Detected cores: %s", worker_layout["detected"])
        logger.info("  Requested workers: %s", worker_layout["requested"])
        logger.info("  Effective workers: %s", worker_layout["effective"])
        if worker_layout["effective"] == 1:
            logger.info("  Running in sequential mode (effective n_workers = 1)")
        else:
            logger.info("  Using multiprocessing + nested thread parallelism")

    if resolution_mode and cluster_range is not None and verbose:
        logger.info("resolution provided; cluster_range will be ignored.")
    if resolution_mode and resolution_values is not None and np.asarray(resolution, dtype=float).size != resolution_values.size and verbose:
        logger.info("removed duplicated manual resolution values before evaluation.")

    clear_clustering_cache()
    adjacency, graph = _extract_graph(adata, graph_key=graph_key, verbose=verbose)

    try:
        clustering_start = time.time()
        if verbose:
            logger.info("-" * 80)
            logger.info("CLUSTERING ANALYSIS:")
            logger.info("  Starting clustering analysis")
            logger.info("  Thread context: main process (PID: %s)", os.getpid())
            if n_workers > 1:
                logger.info("  Parallel workers will be spawned for sub-tasks")
        if resolution_mode:
            if verbose:
                logger.info(
                    "  Manual resolution mode selected - skipping cluster_range search and evaluating supplied gamma values directly."
                )
            results = _build_manual_resolution_results(
                graph=graph,
                resolution_values=resolution_values,
                n_workers=n_workers,
                outer_workers=outer_workers,
                inner_workers=inner_workers,
                n_trials=n_trials,
                n_bootstrap=n_bootstrap,
                seed=seed,
                beta=beta,
                n_iterations=n_iterations,
                objective_function=objective_function,
                snn_graph=adjacency,
                min_cluster_size=min_cluster_size,
                verbose=verbose,
                runtime_context=runtime_context,
            )
            results["analysis_mode"] = "resolution"
            results["resolution_input"] = resolution_values
            results["resolution_search_diagnostics"] = None
            results["requested_cluster_range"] = None
            results["searched_target_cluster_range"] = None
            results["search_coverage_complete"] = True
            results["coverage_complete"] = True
            results["target_diagnostics"] = None
            results["plateau_stop"] = False
            results["search_uncovered_targets"] = np.asarray([], dtype=int)
            results["uncovered_targets"] = np.asarray([], dtype=int)
            results["discovered_upper_gamma"] = np.nan
            results["upper_cap_stop_reason"] = None
            results["coarse_probe_count"] = np.nan
            results["target_gamma_seeds"] = {}
            results["target_interval_details"] = {}
        else:
            results = _run_cluster_range_mode(
                graph=graph,
                requested_cluster_range=requested_cluster_range,
                n_workers=n_workers,
                outer_workers=outer_workers,
                inner_workers=inner_workers,
                n_trials=n_trials,
                n_bootstrap=n_bootstrap,
                seed=seed,
                beta=beta,
                n_iterations=n_iterations,
                max_iterations=max_iterations,
                objective_function=objective_function,
                remove_threshold=remove_threshold,
                snn_graph=adjacency,
                min_cluster_size=min_cluster_size,
                resolution_tolerance=resolution_tolerance,
                verbose=verbose,
                runtime_context=runtime_context,
            )
            results["analysis_mode"] = "cluster_range"
            results["resolution_input"] = None
            results["resolution_diagnostics"] = None

        if verbose:
            clustering_time = time.time() - clustering_start
            logger.info("  Clustering analysis completed in %.3f seconds", clustering_time)

        results["min_cluster_size"] = int(min_cluster_size)
        results["cell_names"] = np.asarray(adata.obs_names, dtype=object)
        results["graph_key"] = graph_key
        results["graph_name"] = graph_key
        results["beta"] = float(beta)
        results["beta_supported"] = bool(beta_status["supported"])
        results["beta_applied"] = bool(beta_status["applied"])
        results["beta_support_reason"] = str(beta_status["reason"])
        results.setdefault(
            "parallel_layout",
            {
                "total_workers": int(n_workers),
                "outer_workers": 1,
                "inner_workers": int(n_workers),
            },
        )
        results = attach_summary_fields(results, ic_threshold=float(ic_threshold))
        adata.uns["scICE"] = results
        if verbose:
            _log_results_summary(
                results=results,
                resolution_mode=resolution_mode,
                requested_cluster_range=requested_cluster_range,
                resolution_values=resolution_values,
                ic_threshold=float(ic_threshold),
                total_time=time.time() - total_start,
            )
    finally:
        clear_clustering_cache()
        cleanup_runtime_spill(runtime_context)

    return adata if copy else None
