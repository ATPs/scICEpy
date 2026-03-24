"""Public AnnData-facing API for scICEpy."""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import sys
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

from .leiden_wrapper import beta_support_status, graph_to_igraph, leiden_clustering
from .metrics import calculate_ic_from_extracted, extract_clustering_array
from .optimization import evaluate_fixed_resolution, optimize_clustering
from .resolution_search import find_resolution_ranges, global_resolution_search_midpoint
from .results import attach_summary_fields, cluster_results_to_dict, finalize_cluster_range_results
from .runtime import (
    cleanup_runtime_spill,
    clear_clustering_cache,
    create_runtime_context,
    logger,
    resolve_nested_worker_layout,
    resolve_effective_workers,
    summarize_adjacency_matrix,
)
from .visualization import get_robust_labels, plot_ic

_PARALLEL_STATE: dict[str, Any] = {}


def _get_parallel_context():
    if os.name == "nt":
        return None
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context()


def _init_parallel_state(state: dict[str, Any]) -> None:
    _PARALLEL_STATE.clear()
    _PARALLEL_STATE.update(state)
    clear_clustering_cache()


def _normalize_cluster_range(cluster_range: Any) -> np.ndarray:
    if cluster_range is None:
        return np.arange(2, 21, dtype=int)
    values = np.asarray(cluster_range, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("cluster_range must contain at least one finite value.")
    rounded = np.rint(values)
    if np.any(np.abs(values - rounded) > np.sqrt(np.finfo(float).eps)):
        raise ValueError("cluster_range must contain only integers >= 1.")
    cluster_values = np.unique(rounded.astype(int))
    if np.any(cluster_values < 1):
        raise ValueError("cluster_range must contain only integers >= 1.")
    return np.sort(cluster_values.astype(int))


def _normalize_resolution_values(resolution: Any) -> np.ndarray:
    values = np.asarray(resolution, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("resolution must contain at least one finite numeric value.")
    ordered_unique: list[float] = []
    seen: set[float] = set()
    for value in values.tolist():
        key = float(value)
        if key in seen:
            continue
        seen.add(key)
        ordered_unique.append(key)
    return np.asarray(ordered_unique, dtype=float)


def _validate_common_inputs(
    adata,
    graph_key: str,
    n_workers: int,
    min_cluster_size: int,
    objective_function: str,
) -> None:
    if not hasattr(adata, "obsp") or not hasattr(adata, "obs_names") or not hasattr(adata, "uns"):
        raise TypeError("adata must be an AnnData-like object with .obsp, .obs_names, and .uns.")
    if graph_key not in adata.obsp:
        raise ValueError(
            f"Graph '{graph_key}' not found in adata.obsp. Available keys: {list(adata.obsp.keys())}"
        )
    if int(n_workers) < 1:
        raise ValueError("n_workers must be >= 1.")
    if int(min_cluster_size) < 1:
        raise ValueError("min_cluster_size must be >= 1.")
    if objective_function not in {"CPM", "modularity"}:
        raise ValueError("objective_function must be either 'CPM' or 'modularity'.")


def _extract_graph(adata, graph_key: str, verbose: bool):
    adjacency = adata.obsp[graph_key]
    if verbose:
        graph_summary = summarize_adjacency_matrix(adjacency)
        logger.info("-" * 80)
        logger.info("GRAPH EXTRACTION:")
        logger.info("  Accessing graph: %s", graph_key)
        logger.info("  Available graphs in object: %s", ", ".join(map(str, adata.obsp.keys())))
        logger.info("  Graph extraction successful")
        if graph_summary["shape"][0] is not None and graph_summary["shape"][1] is not None:
            logger.info("  Graph dimensions: %s x %s", graph_summary["shape"][0], graph_summary["shape"][1])
        logger.info("  Graph storage type: %s", graph_summary["dtype"])
        if graph_summary["nnz"] is not None:
            logger.info("  Non-zero entries: %s", graph_summary["nnz"])
        if np.isfinite(graph_summary["sparsity_percent"]):
            logger.info("  Sparsity: %.2f%%", graph_summary["sparsity_percent"])
        if np.isfinite(graph_summary["weight_min"]) and np.isfinite(graph_summary["weight_max"]):
            logger.info(
                "  Weight range: [%.4f, %.4f]",
                graph_summary["weight_min"],
                graph_summary["weight_max"],
            )
            logger.info("  Mean weight: %.4f", graph_summary["weight_mean"])
        logger.info("-" * 80)
        logger.info("GRAPH CONVERSION:")
        conversion_start = time.time()
        logger.info("  Starting graph conversion")
    graph = graph_to_igraph(adjacency)
    if verbose:
        conversion_time = time.time() - conversion_start
        logger.info("  Graph conversion completed in %.3f seconds", conversion_time)
        logger.info("  Converted graph vertices: %s", graph.vcount())
        logger.info("  Converted graph edges: %s", graph.ecount())
        logger.info("  Graph is weighted: %s", graph.is_weighted())
        if graph.is_weighted() and graph.ecount() > 0:
            weights = np.asarray(graph.es["weight"], dtype=float)
            finite_weights = weights[np.isfinite(weights)]
            if finite_weights.size:
                logger.info(
                    "  Edge weight range: [%.4f, %.4f]",
                    float(np.min(finite_weights)),
                    float(np.max(finite_weights)),
                )
    return adjacency, graph


def _safe_len(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(len(value))
    except TypeError:
        return 0


def _format_cluster_values(values: Any) -> str:
    if values is None:
        return "none"
    arr = np.asarray(values)
    if arr.size == 0:
        return "none"
    return ", ".join(map(str, arr.tolist()))


def _log_results_summary(
    results: dict[str, Any],
    resolution_mode: bool,
    requested_cluster_range: np.ndarray | None,
    resolution_values: np.ndarray | None,
    ic_threshold: float,
    total_time: float,
) -> None:
    logger.info("-" * 80)
    logger.info("RESULTS ANALYSIS:")
    logger.info("  IC threshold for consistency: %s", ic_threshold)
    logger.info("  Results structure:")
    logger.info("    - gamma length: %s", _safe_len(results.get("gamma")))
    logger.info("    - labels length: %s", _safe_len(results.get("labels")))
    logger.info("    - ic length: %s", _safe_len(results.get("ic")))
    logger.info("    - n_cluster length: %s", _safe_len(results.get("n_cluster")))
    logger.info(
        "    - effective_cluster_median length: %s",
        _safe_len(results.get("effective_cluster_median")),
    )
    logger.info("    - raw_cluster_median length: %s", _safe_len(results.get("raw_cluster_median")))
    logger.info("    - final_cluster_median length: %s", _safe_len(results.get("final_cluster_median")))
    logger.info("    - admission_mode length: %s", _safe_len(results.get("admission_mode")))
    logger.info(
        "    - best_labels_raw_cluster_count length: %s",
        _safe_len(results.get("best_labels_raw_cluster_count")),
    )
    logger.info(
        "    - best_labels_final_cluster_count length: %s",
        _safe_len(results.get("best_labels_final_cluster_count")),
    )
    logger.info("    - source_target_cluster length: %s", _safe_len(results.get("source_target_cluster")))

    resolution_diag = results.get("resolution_diagnostics")
    search_diag = results.get("resolution_search_diagnostics")
    target_diag = results.get("target_diagnostics")
    if isinstance(target_diag, pd.DataFrame):
        logger.info("    - target diagnostics rows: %s", len(target_diag))
    if isinstance(search_diag, pd.DataFrame):
        logger.info("    - resolution search diagnostics rows: %s", len(search_diag))
        logger.info("    - search coverage complete: %s", bool(results.get("search_coverage_complete", False)))
    if isinstance(resolution_diag, pd.DataFrame):
        logger.info("    - manual resolution diagnostics rows: %s", len(resolution_diag))

    consistent_clusters = np.asarray(results.get("consistent_clusters", []), dtype=int)
    logger.info("  Returned final clusters: %s", _format_cluster_values(results.get("n_cluster")))
    logger.info("  Consistent clusters: %s", _format_cluster_values(consistent_clusters))

    if resolution_mode:
        logger.info("  Analysis mode: manual resolution")
        logger.info("  Manual resolutions evaluated: %s", _safe_len(resolution_values))
        if isinstance(resolution_diag, pd.DataFrame):
            superseded = int((~resolution_diag["selected"]).sum()) if "selected" in resolution_diag.columns else 0
            logger.info("  Manual resolutions retained after per-cluster IC selection: %s", _safe_len(results.get("n_cluster")))
            if superseded > 0:
                logger.info("  Manual resolutions superseded by lower-IC matches: %s", superseded)
    else:
        logger.info("  Analysis mode: cluster range search")
        logger.info("  Requested final cluster targets: %s", _format_cluster_values(requested_cluster_range))
        logger.info(
            "  Searched target clusters: %s",
            _format_cluster_values(results.get("searched_target_cluster_range")),
        )
        logger.info("  Search coverage complete: %s", bool(results.get("search_coverage_complete", False)))
        logger.info("  Coverage complete: %s", bool(results.get("coverage_complete", False)))
        uncovered_targets = np.asarray(results.get("uncovered_targets", []), dtype=int)
        if uncovered_targets.size:
            logger.info("  Uncovered targets: %s", _format_cluster_values(uncovered_targets))

    best_cluster = results.get("best_cluster", np.nan)
    best_resolution = results.get("best_resolution", np.nan)
    if np.isfinite(best_cluster):
        logger.info("  Best cluster: %s", int(best_cluster))
    if np.isfinite(best_resolution):
        logger.info("  Best resolution: %.6g", float(best_resolution))

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("  Total execution time: %.3f seconds", total_time)


def _empty_mei() -> np.ndarray:
    return np.asarray([], dtype=float)


def _build_excluded_target_result(
    cluster_num: int,
    reason: str,
    gamma: float = np.nan,
    effective_cluster_median: float = np.nan,
    raw_cluster_median: float = np.nan,
    final_cluster_median: float = np.nan,
    admission_mode: str | None = None,
    best_labels_raw_cluster_count: int = -1,
    best_labels_final_cluster_count: int = -1,
    n_iterations: int = 0,
    k: int = 0,
    phase1_primary_gamma_count: int = 0,
    phase1_secondary_gamma_count: int = 0,
    phase1_total_gamma_count: int = 0,
    phase1_elapsed_sec: float = 0.0,
    phase1_leiden_runs: int = 0,
    secondary_phase1_used: bool = False,
    exact_hit_gamma_count: int = 0,
    phase4_iterations: int = 0,
    phase4_elapsed_sec: float = 0.0,
    phase5_elapsed_sec: float = 0.0,
    optimization_elapsed_sec: float = 0.0,
    optimization_diagnostics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    return {
        "cluster_number": int(cluster_num),
        "gamma": float(gamma),
        "labels": None,
        "ic_median": np.nan,
        "ic_bootstrap": np.asarray([], dtype=float),
        "best_labels": None,
        "effective_cluster_median": float(effective_cluster_median),
        "raw_cluster_median": float(raw_cluster_median),
        "final_cluster_median": float(final_cluster_median),
        "admission_mode": admission_mode if admission_mode is not None else reason,
        "best_labels_raw_cluster_count": int(best_labels_raw_cluster_count),
        "best_labels_final_cluster_count": int(best_labels_final_cluster_count),
        "n_iterations": int(n_iterations),
        "mei": _empty_mei(),
        "k": int(k),
        "source_target_cluster": int(cluster_num),
        "excluded": True,
        "exclusion_reason": str(reason),
        "selected_main_result": False,
        "result_status": str(reason),
        "phase1_primary_gamma_count": int(phase1_primary_gamma_count),
        "phase1_secondary_gamma_count": int(phase1_secondary_gamma_count),
        "phase1_total_gamma_count": int(phase1_total_gamma_count),
        "phase1_elapsed_sec": float(phase1_elapsed_sec),
        "phase1_leiden_runs": int(phase1_leiden_runs),
        "secondary_phase1_used": bool(secondary_phase1_used),
        "exact_hit_gamma_count": int(exact_hit_gamma_count),
        "phase4_iterations": int(phase4_iterations),
        "phase4_elapsed_sec": float(phase4_elapsed_sec),
        "phase5_elapsed_sec": float(phase5_elapsed_sec),
        "optimization_elapsed_sec": float(optimization_elapsed_sec),
        "optimization_diagnostics": optimization_diagnostics if optimization_diagnostics is not None else pd.DataFrame(),
    }


def _build_successful_target_result(cluster_num: int, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "cluster_number": int(cluster_num),
        "gamma": float(result["gamma"]),
        "labels": result["labels"],
        "ic_median": float(result["ic_median"]),
        "ic_bootstrap": np.asarray(result["ic_bootstrap"], dtype=float),
        "best_labels": np.asarray(result["best_labels"], dtype=np.int32),
        "effective_cluster_median": float(result.get("effective_cluster_median", np.nan)),
        "raw_cluster_median": float(result.get("raw_cluster_median", np.nan)),
        "final_cluster_median": float(result.get("final_cluster_median", np.nan)),
        "admission_mode": str(result.get("admission_mode", "unknown")),
        "best_labels_raw_cluster_count": int(result.get("best_labels_raw_cluster_count", -1)),
        "best_labels_final_cluster_count": int(result.get("best_labels_final_cluster_count", -1)),
        "n_iterations": int(result.get("n_iterations", 0)),
        "mei": np.asarray(result.get("mei", _empty_mei()), dtype=float),
        "k": int(result.get("k", 0)),
        "source_target_cluster": int(cluster_num),
        "excluded": False,
        "exclusion_reason": "none",
        "selected_main_result": False,
        "result_status": "candidate",
        "phase1_primary_gamma_count": int(result.get("phase1_primary_gamma_count", 0)),
        "phase1_secondary_gamma_count": int(result.get("phase1_secondary_gamma_count", 0)),
        "phase1_total_gamma_count": int(result.get("phase1_total_gamma_count", 0)),
        "phase1_elapsed_sec": float(result.get("phase1_elapsed_sec", 0.0)),
        "phase1_leiden_runs": int(result.get("phase1_leiden_runs", 0)),
        "secondary_phase1_used": bool(result.get("secondary_phase1_used", False)),
        "exact_hit_gamma_count": int(result.get("exact_hit_gamma_count", 0)),
        "phase4_iterations": int(result.get("phase4_iterations", 0)),
        "phase4_elapsed_sec": float(result.get("phase4_elapsed_sec", 0.0)),
        "phase5_elapsed_sec": float(result.get("phase5_elapsed_sec", 0.0)),
        "optimization_elapsed_sec": float(result.get("optimization_elapsed_sec", 0.0)),
        "optimization_diagnostics": (
            result.get("optimization_diagnostics")
            if isinstance(result.get("optimization_diagnostics"), pd.DataFrame)
            else pd.DataFrame()
        ),
    }


def _match_resolution_counts(
    seed_table: pd.DataFrame,
    resolution_search_diagnostics: pd.DataFrame | None,
    gamma_left: float,
    gamma_right: float,
) -> pd.DataFrame:
    if resolution_search_diagnostics is None or resolution_search_diagnostics.empty:
        return seed_table
    required = {"gamma", "final_cluster_count", "raw_cluster_count"}
    if not required.issubset(resolution_search_diagnostics.columns):
        return seed_table

    diagnostics = resolution_search_diagnostics.loc[:, ["gamma", "final_cluster_count", "raw_cluster_count"]].copy()
    diagnostics["gamma"] = diagnostics["gamma"].astype(float)
    tolerance = max(np.sqrt(np.finfo(float).eps), abs(gamma_right - gamma_left) * 1e-8, 1e-12)
    final_counts: list[float] = []
    raw_counts: list[float] = []
    for gamma_value in seed_table["gamma"].astype(float).tolist():
        delta = np.abs(diagnostics["gamma"].to_numpy(dtype=float) - float(gamma_value))
        if delta.size == 0 or np.all(~np.isfinite(delta)):
            final_counts.append(np.nan)
            raw_counts.append(np.nan)
            continue
        idx = int(np.nanargmin(delta))
        if not np.isfinite(delta[idx]) or delta[idx] > tolerance:
            final_counts.append(np.nan)
            raw_counts.append(np.nan)
            continue
        final_counts.append(float(diagnostics.iloc[idx]["final_cluster_count"]))
        raw_counts.append(float(diagnostics.iloc[idx]["raw_cluster_count"]))
    seed_table["final_cluster_count"] = final_counts
    seed_table["raw_cluster_count"] = raw_counts
    return seed_table


def _build_target_gamma_seed_table(
    target_cluster: int,
    gamma_dict: dict[int, tuple[float, float]],
    objective_function: str,
    target_gamma_seeds: dict[str, list[float]] | None = None,
    target_interval_details: dict[str, dict[str, Any]] | None = None,
    resolution_search_diagnostics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    target_key = str(int(target_cluster))
    interval_detail = (target_interval_details or {}).get(target_key, {})
    gamma_bounds = gamma_dict.get(int(target_cluster))
    gamma_left = float(interval_detail.get("gamma_left", gamma_bounds[0] if gamma_bounds else np.nan))
    gamma_right = float(interval_detail.get("gamma_right", gamma_bounds[1] if gamma_bounds else np.nan))

    seed_values = sorted(
        {
            float(value)
            for value in (target_gamma_seeds or {}).get(target_key, [])
            if np.isfinite(value)
        }
    )
    exact_probe_values = sorted(
        {
            float(value)
            for value in interval_detail.get("exact_probe_values", []) or []
            if np.isfinite(value)
        }
    )
    near_probe_values = sorted(
        {
            float(value)
            for value in interval_detail.get("near_probe_values", []) or []
            if np.isfinite(value) and float(value) not in set(exact_probe_values)
        }
    )

    selected_gamma = np.nan
    if seed_values and np.isfinite(gamma_left) and np.isfinite(gamma_right):
        midpoint = global_resolution_search_midpoint(
            gamma_left,
            gamma_right,
            objective_function=objective_function,
        )
        selected_gamma = min(seed_values, key=lambda value: abs(value - midpoint))
    elif seed_values:
        selected_gamma = seed_values[0]

    rows: list[dict[str, Any]] = []
    for gamma, seed_role in [
        (gamma_left, "left"),
        (gamma_right, "right"),
        (selected_gamma, "selected"),
    ]:
        if np.isfinite(gamma):
            rows.append({"gamma": float(gamma), "seed_role": seed_role})
    rows.extend({"gamma": float(gamma), "seed_role": "exact"} for gamma in exact_probe_values)
    rows.extend({"gamma": float(gamma), "seed_role": "near"} for gamma in near_probe_values)

    covered = {row["gamma"] for row in rows}
    rows.extend(
        {"gamma": float(gamma), "seed_role": "seed"}
        for gamma in seed_values
        if float(gamma) not in covered
    )
    if not rows:
        return pd.DataFrame(
            columns=["gamma", "seed_role", "final_cluster_count", "raw_cluster_count"]
        )

    seed_table = pd.DataFrame(rows).drop_duplicates(subset=["gamma", "seed_role"]).sort_values(
        ["gamma", "seed_role"]
    ).reset_index(drop=True)
    seed_table["final_cluster_count"] = np.nan
    seed_table["raw_cluster_count"] = np.nan
    return _match_resolution_counts(
        seed_table,
        resolution_search_diagnostics=resolution_search_diagnostics,
        gamma_left=gamma_left,
        gamma_right=gamma_right,
    )


def _filter_cluster_targets(
    graph,
    cluster_range: np.ndarray,
    gamma_dict: dict[int, tuple[float, float]],
    remove_threshold: float,
    objective_function: str,
    seed: int | None,
    verbose: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if math.isinf(remove_threshold):
        for cluster_num in cluster_range:
            if int(cluster_num) not in gamma_dict:
                results.append(
                    {"cluster_num": int(cluster_num), "excluded": True, "reason": "resolution_search_failed"}
                )
            else:
                results.append(
                    {
                        "cluster_num": int(cluster_num),
                        "excluded": False,
                        "reason": "filtering_skipped_inf_threshold",
                    }
                )
        return results

    for cluster_num in cluster_range:
        cluster_num = int(cluster_num)
        if cluster_num not in gamma_dict:
            results.append({"cluster_num": cluster_num, "excluded": True, "reason": "resolution_search_failed"})
            continue

        gamma_left, gamma_right = gamma_dict[cluster_num]
        if objective_function == "CPM" and gamma_left > 0 and gamma_right > 0:
            gamma_test = np.exp(np.linspace(np.log(gamma_left), np.log(gamma_right), num=5))
        else:
            gamma_test = np.linspace(gamma_left, gamma_right, num=5)

        ic_scores: list[float] = []
        for gamma_idx, gamma_val in enumerate(np.unique(gamma_test)):
            cluster_matrix = np.zeros((10, graph.vcount()), dtype=np.int32)
            for trial_idx in range(10):
                trial_seed = None
                if seed is not None:
                    trial_seed = int(
                        (seed + cluster_num * 1000 + gamma_idx * 100 + trial_idx + 1) % (2**31 - 1)
                    ) or 1
                cluster_matrix[trial_idx] = leiden_clustering(
                    graph=graph,
                    resolution=float(gamma_val),
                    objective_function=objective_function,
                    n_iterations=5,
                    beta=0.01,
                    seed=trial_seed,
                )
            ic_scores.append(calculate_ic_from_extracted(extract_clustering_array(cluster_matrix), n_workers=1))

        excluded = bool(np.nanmin(np.asarray(ic_scores, dtype=float)) >= float(remove_threshold))
        results.append(
            {
                "cluster_num": cluster_num,
                "excluded": excluded,
                "reason": "high_inconsistency" if excluded else "passed_filtering",
            }
        )

    if verbose:
        excluded_targets = [str(item["cluster_num"]) for item in results if item["excluded"]]
        if excluded_targets:
            logger.info("Filtering excluded targets: %s", ", ".join(excluded_targets))
    return results


def _select_lowest_ic_indices(cluster_numbers: np.ndarray, ic_scores: np.ndarray, gamma_values: np.ndarray) -> list[int]:
    selected_indices: list[int] = []
    for cluster_num in sorted(set(cluster_numbers.tolist())):
        indices = np.where(cluster_numbers == cluster_num)[0].tolist()
        finite_indices = [idx for idx in indices if np.isfinite(ic_scores[idx])]
        choice_pool = finite_indices if finite_indices else indices
        chosen = min(choice_pool, key=lambda idx: (float(ic_scores[idx]), float(gamma_values[idx])))
        selected_indices.append(int(chosen))
    selected_indices.sort(key=lambda idx: (int(cluster_numbers[idx]), float(gamma_values[idx])))
    return selected_indices


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

    cluster_numbers = np.asarray(
        [int(result["best_labels_final_cluster_count"]) for result in resolution_results],
        dtype=int,
    )
    ic_scores = np.asarray([float(result["ic_median"]) for result in resolution_results], dtype=float)
    gamma_values = np.asarray([float(result["gamma"]) for result in resolution_results], dtype=float)
    selected_indices = _select_lowest_ic_indices(cluster_numbers, ic_scores, gamma_values)
    selected_mask = np.zeros(len(resolution_results), dtype=bool)
    selected_mask[selected_indices] = True

    resolution_diagnostics = pd.DataFrame(
        {
            "resolution": gamma_values,
            "cluster_number": cluster_numbers,
            "ic": ic_scores,
            "effective_cluster_median": np.asarray(
                [float(result["effective_cluster_median"]) for result in resolution_results],
                dtype=float,
            ),
            "raw_cluster_median": np.asarray(
                [float(result["raw_cluster_median"]) for result in resolution_results],
                dtype=float,
            ),
            "final_cluster_median": np.asarray(
                [float(result["final_cluster_median"]) for result in resolution_results],
                dtype=float,
            ),
            "best_labels_raw_cluster_count": np.asarray(
                [int(result["best_labels_raw_cluster_count"]) for result in resolution_results],
                dtype=int,
            ),
            "best_labels_final_cluster_count": np.asarray(
                [int(result["best_labels_final_cluster_count"]) for result in resolution_results],
                dtype=int,
            ),
            "n_iter": np.asarray(
                [int(result.get("n_iterations", 0)) for result in resolution_results],
                dtype=int,
            ),
            "selected": selected_mask,
        }
    )

    selected_results = []
    for idx in selected_indices:
        result = dict(resolution_results[idx])
        result["cluster_number"] = int(result["best_labels_final_cluster_count"])
        result["source_target_cluster"] = np.nan
        result["excluded"] = False
        result["exclusion_reason"] = "none"
        result["selected_main_result"] = True
        result["result_status"] = "selected_main_result"
        selected_results.append(result)

    final_results = cluster_results_to_dict(selected_results)
    final_results["resolution_diagnostics"] = resolution_diagnostics
    final_results["parallel_layout"] = worker_layout
    return final_results


def _evaluate_manual_resolution_impl(resolution_value: float, state: dict[str, Any]) -> dict[str, Any]:
    return evaluate_fixed_resolution(
        graph=state["graph"],
        resolution=float(resolution_value),
        objective_function=state["objective_function"],
        n_trials=state["n_trials"],
        n_bootstrap=state["n_bootstrap"],
        seed=state["seed"],
        beta=state["beta"],
        n_iterations=state["n_iterations"],
        n_workers=state["n_workers"],
        snn_graph=state["snn_graph"],
        min_cluster_size=state["min_cluster_size"],
        verbose=state["verbose"],
        worker_id=f"RESOLUTION {float(resolution_value):.6g}",
        runtime_context=state["runtime_context"],
        in_parallel_context=bool(state.get("in_parallel_context", False)),
    )


def _evaluate_manual_resolution_worker(resolution_value: float) -> dict[str, Any]:
    return _evaluate_manual_resolution_impl(float(resolution_value), _PARALLEL_STATE)


def _map_manual_resolutions(
    resolution_values: np.ndarray,
    state: dict[str, Any],
    active_workers: int,
) -> list[dict[str, Any]]:
    resolution_values = np.asarray(resolution_values, dtype=float)
    active_workers = max(1, min(int(active_workers), int(resolution_values.size)))
    context = _get_parallel_context()
    if context is None or active_workers <= 1 or resolution_values.size <= 1:
        return [_evaluate_manual_resolution_impl(float(value), state) for value in resolution_values]

    with context.Pool(
        processes=active_workers,
        initializer=_init_parallel_state,
        initargs=(state,),
    ) as pool:
        return pool.map(_evaluate_manual_resolution_worker, resolution_values.tolist())


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
            target_results.append(_build_excluded_target_result(cluster_num, reason=str(filter_result["reason"])))
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


def _optimize_target_cluster_impl(cluster_num: int, state: dict[str, Any]) -> dict[str, Any]:
    cluster_num = int(cluster_num)
    gamma_range = state["gamma_dict"].get(cluster_num)
    if gamma_range is None:
        return _build_excluded_target_result(cluster_num, reason="resolution_search_failed")

    if state.get("verbose", False):
        logger.info("WORKER %s: Starting optimization for target k = %s", cluster_num, cluster_num)
        logger.info(
            "WORKER %s: Gamma range [%.6g, %.6g]",
            cluster_num,
            float(gamma_range[0]),
            float(gamma_range[1]),
        )

    gamma_seed_table = _build_target_gamma_seed_table(
        target_cluster=cluster_num,
        gamma_dict=state["gamma_dict"],
        objective_function=state["objective_function"],
        target_gamma_seeds=state["target_gamma_seeds"],
        target_interval_details=state["target_interval_details"],
        resolution_search_diagnostics=state["resolution_search_diagnostics"],
    )
    optimization_result = optimize_clustering(
        graph=state["graph"],
        target_clusters=cluster_num,
        gamma_range=gamma_range,
        objective_function=state["objective_function"],
        n_trials=state["n_trials"],
        n_bootstrap=state["n_bootstrap"],
        seed=state["seed"],
        beta=state["beta"],
        n_iterations=state["n_iterations"],
        max_iterations=state["max_iterations"],
        resolution_tolerance=state["resolution_tolerance"],
        n_workers=state["n_workers"],
        snn_graph=state["snn_graph"],
        gamma_seed_values=gamma_seed_table,
        min_cluster_size=state["min_cluster_size"],
        verbose=state["verbose"],
        worker_id=f"WORKER {cluster_num}",
        runtime_context=state["runtime_context"],
        in_parallel_context=bool(state.get("in_parallel_context", False)),
    )
    if state.get("verbose", False):
        logger.info(
            "WORKER %s: Optimization completed in %.3f seconds",
            cluster_num,
            float(optimization_result.get("optimization_elapsed_sec", 0.0)),
        )
    if not optimization_result.get("success", False):
        return _build_excluded_target_result(
            cluster_num=cluster_num,
            reason=str(optimization_result.get("failure_reason", "optimization_failed")),
            gamma=float(optimization_result.get("gamma", np.nan)),
            effective_cluster_median=float(optimization_result.get("effective_cluster_median", np.nan)),
            raw_cluster_median=float(optimization_result.get("raw_cluster_median", np.nan)),
            final_cluster_median=float(optimization_result.get("final_cluster_median", np.nan)),
            admission_mode=str(optimization_result.get("admission_mode", "optimization_failed")),
            best_labels_raw_cluster_count=int(optimization_result.get("best_labels_raw_cluster_count", -1)),
            best_labels_final_cluster_count=int(optimization_result.get("best_labels_final_cluster_count", -1)),
            n_iterations=int(optimization_result.get("n_iterations", 0)),
            k=int(optimization_result.get("k", 0)),
            phase1_primary_gamma_count=int(optimization_result.get("phase1_primary_gamma_count", 0)),
            phase1_secondary_gamma_count=int(optimization_result.get("phase1_secondary_gamma_count", 0)),
            phase1_total_gamma_count=int(optimization_result.get("phase1_total_gamma_count", 0)),
            phase1_elapsed_sec=float(optimization_result.get("phase1_elapsed_sec", 0.0)),
            phase1_leiden_runs=int(optimization_result.get("phase1_leiden_runs", 0)),
            secondary_phase1_used=bool(optimization_result.get("secondary_phase1_used", False)),
            exact_hit_gamma_count=int(optimization_result.get("exact_hit_gamma_count", 0)),
            phase4_iterations=int(optimization_result.get("phase4_iterations", 0)),
            phase4_elapsed_sec=float(optimization_result.get("phase4_elapsed_sec", 0.0)),
            phase5_elapsed_sec=float(optimization_result.get("phase5_elapsed_sec", 0.0)),
            optimization_elapsed_sec=float(optimization_result.get("optimization_elapsed_sec", 0.0)),
            optimization_diagnostics=(
                optimization_result.get("optimization_diagnostics")
                if isinstance(optimization_result.get("optimization_diagnostics"), pd.DataFrame)
                else pd.DataFrame()
            ),
        )

    return _build_successful_target_result(cluster_num, optimization_result)


def _optimize_target_cluster_worker(cluster_num: int) -> dict[str, Any]:
    return _optimize_target_cluster_impl(int(cluster_num), _PARALLEL_STATE)


def _map_optimized_targets(
    valid_clusters: list[int],
    state: dict[str, Any],
    active_workers: int,
) -> list[dict[str, Any]]:
    if not valid_clusters:
        return []
    active_workers = max(1, min(int(active_workers), len(valid_clusters)))
    context = _get_parallel_context()
    if context is None or active_workers <= 1 or len(valid_clusters) <= 1:
        return [_optimize_target_cluster_impl(int(cluster_num), state) for cluster_num in valid_clusters]

    with context.Pool(
        processes=active_workers,
        initializer=_init_parallel_state,
        initargs=(state,),
    ) as pool:
        return pool.map(_optimize_target_cluster_worker, [int(cluster_num) for cluster_num in valid_clusters])


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
):
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
    runtime_context = create_runtime_context()
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


__all__ = ["scICE_clustering", "get_robust_labels", "plot_ic"]
