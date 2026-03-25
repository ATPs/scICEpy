"""Top-level execution modes for the public scICE clustering entry point."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .clustering_dispatch import _filter_cluster_targets, _map_manual_resolutions, _map_optimized_targets
from .clustering_inputs import _format_cluster_values
from .resolution_search import find_resolution_ranges
from .results import (
    build_target_result_record,
    cluster_results_to_dict,
    finalize_cluster_range_results,
    rekey_target_results_by_final_cluster,
)
from .runtime import (
    logger,
    resolve_nested_worker_layout,
)

def _build_manual_resolution_results(
    graph,
    resolution_values: np.ndarray,
    n_workers: int,
    outer_workers: int | None,
    inner_workers: int | None,
    n_trials: int,
    n_bootstrap: int,
    seed: int | None,
    beta: float,
    n_iterations: int,
    objective_function: str,
    snn_graph,
    min_cluster_size: int,
    verbose: bool,
    runtime_context,
) -> dict[str, Any]:
    """Evaluate fixed user-provided resolutions and retain the best result per final cluster count."""
    resolution_values = np.asarray(resolution_values, dtype=float)
    n_vertices = graph.vcount()
    worker_layout = resolve_nested_worker_layout(
        total_workers=n_workers,
        task_count=int(resolution_values.size),
        n_cells=n_vertices,
        n_trials=n_trials,
        n_bootstrap=n_bootstrap,
        runtime_context=runtime_context,
        outer_workers=outer_workers,
        inner_workers=inner_workers,
        expected_gamma_count=1,
    )
    active_resolution_workers = int(worker_layout["outer_workers"])
    per_resolution_worker_budget = int(worker_layout["inner_workers"])
    if verbose:
        logger.info("CLUSTERING_MAIN: Using manual resolution mode.")
        logger.info(
            "CLUSTERING_MAIN: Resolution worker layout - %s resolution workers x %s trial/bootstrap workers",
            active_resolution_workers,
            per_resolution_worker_budget,
        )

    state = {
        "graph": graph,
        "n_workers": per_resolution_worker_budget,
        "n_trials": n_trials,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "beta": beta,
        "n_iterations": n_iterations,
        "objective_function": objective_function,
        "snn_graph": snn_graph,
        "min_cluster_size": min_cluster_size,
        "verbose": verbose,
        "runtime_context": runtime_context,
        "in_parallel_context": active_resolution_workers > 1,
    }
    resolution_results = _map_manual_resolutions(
        resolution_values,
        state,
        active_workers=active_resolution_workers,
    )
    resolution_results = [result for result in resolution_results if result is not None]
    if not resolution_results:
        final_results = cluster_results_to_dict([])
        final_results["resolution_diagnostics"] = pd.DataFrame(
            columns=[
                "resolution",
                "cluster_number",
                "ic",
                "effective_cluster_median",
                "raw_cluster_median",
                "final_cluster_median",
                "best_labels_raw_cluster_count",
                "best_labels_final_cluster_count",
                "n_iter",
                "selected",
            ]
        )
        return final_results

    manual_target_results = [
        build_target_result_record(
            int(result.get("best_labels_final_cluster_count", -1)),
            result=result,
            source_target_cluster=np.nan,
        )
        for result in resolution_results
    ]
    selected_results, full_resolution_results = rekey_target_results_by_final_cluster(
        manual_target_results,
        require_matching_source_target=False,
    )
    cluster_numbers = np.asarray(
        [int(result["best_labels_final_cluster_count"]) for result in full_resolution_results],
        dtype=int,
    )
    ic_scores = np.asarray([float(result["ic_median"]) for result in full_resolution_results], dtype=float)
    gamma_values = np.asarray([float(result["gamma"]) for result in full_resolution_results], dtype=float)
    selected_mask = np.asarray(
        [bool(result.get("selected_main_result", False)) for result in full_resolution_results],
        dtype=bool,
    )

    resolution_diagnostics = pd.DataFrame(
        {
            "resolution": gamma_values,
            "cluster_number": cluster_numbers,
            "ic": ic_scores,
            "effective_cluster_median": np.asarray(
                [float(result["effective_cluster_median"]) for result in full_resolution_results],
                dtype=float,
            ),
            "raw_cluster_median": np.asarray(
                [float(result["raw_cluster_median"]) for result in full_resolution_results],
                dtype=float,
            ),
            "final_cluster_median": np.asarray(
                [float(result["final_cluster_median"]) for result in full_resolution_results],
                dtype=float,
            ),
            "best_labels_raw_cluster_count": np.asarray(
                [int(result["best_labels_raw_cluster_count"]) for result in full_resolution_results],
                dtype=int,
            ),
            "best_labels_final_cluster_count": np.asarray(
                [int(result["best_labels_final_cluster_count"]) for result in full_resolution_results],
                dtype=int,
            ),
            "n_iter": np.asarray(
                [int(result.get("n_iterations", 0)) for result in full_resolution_results],
                dtype=int,
            ),
            "selected": selected_mask,
        }
    )

    final_results = cluster_results_to_dict(selected_results)
    final_results["resolution_diagnostics"] = resolution_diagnostics
    final_results["parallel_layout"] = worker_layout
    return final_results

def _run_cluster_range_mode(
    graph,
    requested_cluster_range: np.ndarray,
    n_workers: int,
    outer_workers: int | None,
    inner_workers: int | None,
    n_trials: int,
    n_bootstrap: int,
    seed: int | None,
    beta: float,
    n_iterations: int,
    max_iterations: int,
    objective_function: str,
    remove_threshold: float,
    snn_graph,
    min_cluster_size: int,
    resolution_tolerance: float,
    verbose: bool,
    runtime_context,
) -> dict[str, Any]:
    """Resolve target-specific gamma intervals, optimize each feasible cluster count, and build the output tables."""
    search_start_g = -13.0 if objective_function == "modularity" else max(float(np.log(resolution_tolerance)), -20.0)
    search_end_g = 20.0

    gamma_search = find_resolution_ranges(
        graph=graph,
        cluster_range=requested_cluster_range,
        start_g=search_start_g,
        end_g=search_end_g,
        objective_function=objective_function,
        resolution_tolerance=resolution_tolerance,
        n_workers=n_workers,
        verbose=verbose,
        seed=seed,
        snn_graph=snn_graph,
        min_cluster_size=min_cluster_size,
        in_parallel_context=False,
        runtime_context=runtime_context,
    )
    search_attrs = gamma_search.pop("_attrs", {})
    gamma_dict = {int(key): value for key, value in gamma_search.items()}
    resolution_search_diagnostics = search_attrs.get("resolution_search_diagnostics", pd.DataFrame())
    search_coverage_complete = bool(search_attrs.get("coverage_complete", False))
    plateau_stop = bool(search_attrs.get("plateau_stop", False))
    search_uncovered_targets = np.asarray(search_attrs.get("uncovered_targets", []), dtype=int)
    target_gamma_seeds = search_attrs.get("target_gamma_seeds", {})
    target_interval_details = search_attrs.get("target_interval_details", {})
    discovered_upper_gamma = search_attrs.get("discovered_upper_gamma", np.nan)
    upper_cap_stop_reason = search_attrs.get("upper_cap_stop_reason")
    coarse_probe_count = search_attrs.get("coarse_probe_count", np.nan)

    cluster_filter_results = _filter_cluster_targets(
        graph=graph,
        cluster_range=requested_cluster_range,
        gamma_dict=gamma_dict,
        remove_threshold=remove_threshold,
        objective_function=objective_function,
        seed=seed,
        verbose=verbose,
    )
    target_results: list[dict[str, Any]] = []
    valid_clusters: list[int] = []
    for filter_result in cluster_filter_results:
        cluster_num = int(filter_result["cluster_num"])
        if filter_result["excluded"]:
            reason = str(filter_result["reason"])
            target_results.append(
                build_target_result_record(
                    cluster_num,
                    excluded=True,
                    exclusion_reason=reason,
                    result_status=reason,
                )
            )
        else:
            valid_clusters.append(cluster_num)

    n_vertices = graph.vcount()
    worker_layout = resolve_nested_worker_layout(
        total_workers=n_workers,
        task_count=max(1, len(valid_clusters) if valid_clusters else 1),
        n_cells=n_vertices,
        n_trials=n_trials,
        n_bootstrap=n_bootstrap,
        runtime_context=runtime_context,
        outer_workers=outer_workers,
        inner_workers=inner_workers,
        expected_gamma_count=11,
    )
    active_cluster_workers = int(worker_layout["outer_workers"])
    per_cluster_worker_budget = int(worker_layout["inner_workers"])
    if verbose:
        logger.info(
            "CLUSTERING_MAIN: Resolution search returned %s optimization-ready target(s) with %s diagnostic probes",
            len(gamma_dict),
            len(resolution_search_diagnostics),
        )
        logger.info(
            "CLUSTERING_MAIN: Filtering retained %s target(s) and excluded %s target(s)",
            len(valid_clusters),
            len(cluster_filter_results) - len(valid_clusters),
        )
        if math.isinf(remove_threshold):
            logger.info("CLUSTERING_MAIN: Filtering skipped because remove_threshold=Inf")
        if search_uncovered_targets.size:
            logger.info(
                "CLUSTERING_MAIN: Search uncovered targets: %s",
                _format_cluster_values(search_uncovered_targets),
            )
        logger.info("CLUSTERING_MAIN: Starting clustering optimization...")
        logger.info("CLUSTERING_MAIN: Active cluster workers: %s", active_cluster_workers)
        logger.info("CLUSTERING_MAIN: Per-cluster worker budget: %s", per_cluster_worker_budget)
        logger.info(
            "CLUSTERING_MAIN: Estimated bytes per outer worker: %s",
            format(int(worker_layout["estimated_bytes_per_outer_worker"]), ","),
        )

    optimization_state = {
        "graph": graph,
        "gamma_dict": gamma_dict,
        "objective_function": objective_function,
        "n_trials": n_trials,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "beta": beta,
        "n_iterations": n_iterations,
        "max_iterations": max_iterations,
        "resolution_tolerance": resolution_tolerance,
        "n_workers": per_cluster_worker_budget,
        "total_workers_requested": int(n_workers),
        "snn_graph": snn_graph,
        "target_gamma_seeds": target_gamma_seeds,
        "target_interval_details": target_interval_details,
        "resolution_search_diagnostics": resolution_search_diagnostics,
        "min_cluster_size": min_cluster_size,
        "verbose": verbose,
        "runtime_context": runtime_context,
        "in_parallel_context": active_cluster_workers > 1,
    }
    target_results.extend(
        _map_optimized_targets(
            valid_clusters,
            optimization_state,
            active_workers=active_cluster_workers,
        )
    )

    final_results = finalize_cluster_range_results(
        target_results=target_results,
        requested_cluster_range=requested_cluster_range,
        searched_target_cluster_range=requested_cluster_range,
        search_coverage_complete=search_coverage_complete,
        gamma_dict=gamma_dict,
        resolution_search_diagnostics=resolution_search_diagnostics,
        plateau_stop=plateau_stop,
        search_uncovered_targets=search_uncovered_targets,
        discovered_upper_gamma=discovered_upper_gamma,
        upper_cap_stop_reason=upper_cap_stop_reason,
        coarse_probe_count=coarse_probe_count,
        target_gamma_seeds=target_gamma_seeds,
        target_interval_details=target_interval_details,
    )
    final_results["parallel_layout"] = worker_layout
    return final_results
