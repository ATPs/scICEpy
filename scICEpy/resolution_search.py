"""Resolution search helpers for scICEpy."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import pandas as pd

from .cluster_utils import (
    classify_resolution_search_state,
    count_effective_clusters,
    merge_small_clusters_to_neighbors,
    raw_cluster_guard_limits,
    raw_cluster_search_upper,
)
from .leiden_wrapper import cached_leiden_clustering
from .runtime import (
    cap_workers_by_memory,
    estimate_trial_matrix_bytes,
    get_parallel_context,
    initialize_parallel_state,
    logger,
    parallel_map_threads,
)
from .search_bounds import (
    build_gamma_sequence_for_range,
    build_refinement_probe_plan,
    clamp_gamma_range_to_raw_plateau,
    derive_shared_gamma_intervals,
    global_resolution_search_midpoint,
    stabilize_monotone_probe_counts,
    stabilize_probe_raw_medians,
)

_SEARCH_PROBE_STATE: dict[str, Any] = {}

def _init_search_probe_state(state: dict[str, Any]) -> None:
    """Initialize search workers with the shared probe state."""
    initialize_parallel_state(_SEARCH_PROBE_STATE, state)

def _run_single_probe_impl(
    probe_index: int,
    gamma: float,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Run one preliminary search probe and summarize its cluster-count diagnostics."""
    started = pd.Timestamp.utcnow()
    seed = state["seed"]
    sweep_round = int(state["sweep_round"])
    gamma_seed = None
    if seed is not None:
        gamma_seed = int((seed + int(abs(gamma) * 100000) + sweep_round * 1000) % (2**31 - 1))
        if gamma_seed <= 0:
            gamma_seed = 1
    labels = cached_leiden_clustering(
        state["graph"],
        resolution=float(gamma),
        objective_function=state["objective_function"],
        n_iterations=int(state["n_iter_preliminary"]),
        beta=float(state["beta_preliminary"]),
        use_cache=True,
        cache_key_suffix=f"shared_probe_{sweep_round}_{int(probe_index)}",
        seed=gamma_seed,
    )
    effective_cluster_count = count_effective_clusters(labels, min_cluster_size=state["min_cluster_size"])
    raw_cluster_count = int(np.unique(labels).size)
    if state["min_cluster_size"] > 1:
        merged = merge_small_clusters_to_neighbors(
            labels,
            snn_graph=state["snn_graph"],
            min_cluster_size=state["min_cluster_size"],
        )
        final_cluster_count = int(np.unique(merged).size)
    else:
        final_cluster_count = raw_cluster_count
    raw_state = classify_resolution_search_state(
        raw_cluster_median=raw_cluster_count,
        effective_cluster_median=effective_cluster_count,
        target_clusters=int(state["requested_max"]),
        min_cluster_size=state["min_cluster_size"],
    )
    meta = state["metadata_lookup"].get(float(gamma), {})
    return {
        "sweep_round": sweep_round,
        "discovery_round": int(state["discovery_round"]) if state["discovery_round"] is not None else np.nan,
        "probe_stage": state["probe_stage"],
        "probe_index": int(probe_index),
        "probe_pid": int(os.getpid()),
        "probe_elapsed_sec": (pd.Timestamp.utcnow() - started).total_seconds(),
        "gamma": float(gamma),
        "upper_cap_discovery_gamma": float(gamma) if state["probe_stage"] == "upper_cap_discovery" else np.nan,
        "degenerate_high_gamma": is_high_gamma_degenerate_probe(
            effective_cluster_count,
            raw_cluster_count,
            final_cluster_count,
            state["graph"].vcount(),
        ),
        "scheduled_probe_workers": int(state["active_probe_workers"]),
        "coarse_probe_count": int(state["coarse_probe_count"]) if state["coarse_probe_count"] is not None else np.nan,
        "discovered_upper_gamma": np.nan,
        "upper_cap_stop_reason": None,
        "refinement_interval_width": meta.get("refinement_interval_width", np.nan),
        "refinement_interval_id": meta.get("refinement_interval_id", np.nan),
        "refinement_points_per_interval": meta.get("refinement_points_per_interval", np.nan),
        "effective_cluster_count": float(effective_cluster_count),
        "raw_cluster_count": float(raw_cluster_count),
        "final_cluster_count": float(final_cluster_count),
        "raw_class": raw_state["raw_class"],
        "over_fragmented": bool(raw_state["over_fragmented"]),
        "selected_for_refinement": False,
        "selected_for_target_interval": False,
        "plateau_round": np.nan,
    }

def _run_single_probe_worker(task: tuple[int, tuple[int, float]]) -> tuple[int, dict[str, Any]]:
    """Process-pool wrapper for a single preliminary search probe."""
    task_index, probe_item = task
    return int(task_index), _run_single_probe_impl(int(probe_item[0]), float(probe_item[1]), _SEARCH_PROBE_STATE)

def is_high_gamma_degenerate_probe(
    effective_cluster_count: float,
    raw_cluster_count: float,
    final_cluster_count: float,
    n_vertices: int,
) -> bool:
    """Detect the high-gamma regime where raw clustering becomes nearly singleton."""
    return bool(
        np.isfinite(effective_cluster_count)
        and np.isfinite(raw_cluster_count)
        and np.isfinite(final_cluster_count)
        and effective_cluster_count == 0
        and final_cluster_count == 1
        and raw_cluster_count >= (0.98 * float(n_vertices))
    )

def build_cpm_discovery_batch_gamma_values(
    current_gamma: float,
    hard_cap_gamma: float,
    batch_size: int,
    step_ratio: float = 4.0,
) -> np.ndarray:
    """Build the next CPM upper-cap discovery batch in geometric steps."""
    current_gamma = max(float(current_gamma), np.finfo(float).tiny)
    hard_cap_gamma = max(float(hard_cap_gamma), current_gamma)
    batch_size = max(0, int(batch_size))
    if batch_size <= 0:
        return np.asarray([], dtype=float)
    powers = step_ratio ** np.arange(batch_size)
    return np.sort(np.unique(np.minimum(current_gamma * powers, hard_cap_gamma)))

def derive_cpm_discovery_batch_plan(
    active_probe_workers: int,
    requested_max: int,
    frontier_final_cluster_count: float = np.nan,
) -> dict[str, Any]:
    """Tune CPM upper-cap discovery aggressiveness based on current search coverage."""
    max_batch_size = min(max(1, int(active_probe_workers)), 6)
    default_plan = {"batch_size": max_batch_size, "step_ratio": 4.0}
    if not np.isfinite(frontier_final_cluster_count) or not np.isfinite(requested_max) or requested_max <= 0:
        return default_plan

    coverage_ratio = float(frontier_final_cluster_count) / float(requested_max)
    if coverage_ratio >= 0.9:
        return {"batch_size": min(max_batch_size, 2), "step_ratio": 1.5}
    if coverage_ratio >= 0.75:
        return {"batch_size": min(max_batch_size, 2), "step_ratio": 2.0}
    if coverage_ratio >= 0.5:
        return {"batch_size": min(max_batch_size, 3), "step_ratio": 2.0}
    return default_plan

def estimate_search_probe_bytes(
    n_vertices: int,
    n_preliminary_trials: int,
    min_cluster_size: int = 1,
) -> float:
    """Estimate the memory cost of one search probe batch."""
    n_vertices = max(1, int(n_vertices))
    n_preliminary_trials = max(1, int(n_preliminary_trials))
    min_cluster_size = max(1, int(min_cluster_size))
    base_bytes = estimate_trial_matrix_bytes(n_vertices, n_preliminary_trials, 1)
    if n_vertices >= 200000:
        graph_replication_factor = 12.0 if min_cluster_size > 1 else 9.0
    elif n_vertices >= 100000:
        graph_replication_factor = 8.0 if min_cluster_size > 1 else 6.0
    elif n_vertices >= 50000:
        graph_replication_factor = 4.5 if min_cluster_size > 1 else 3.0
    else:
        graph_replication_factor = 2.0 if min_cluster_size > 1 else 1.5
    return float(base_bytes * graph_replication_factor)

def resolve_search_worker_capacity(
    requested_workers: int,
    n_vertices: int,
    n_preliminary_trials: int,
    min_cluster_size: int,
    target_count: int,
    runtime_context=None,
) -> int:
    """Cap coarse-search workers by memory pressure and target-count scaling."""
    workers = max(1, int(requested_workers))
    workers = cap_workers_by_memory(
        workers,
        estimate_search_probe_bytes(
            n_vertices=n_vertices,
            n_preliminary_trials=n_preliminary_trials,
            min_cluster_size=min_cluster_size,
        ),
        runtime_context,
    )
    target_count = max(1, int(target_count))
    if int(n_vertices) >= 200000:
        target_parallel_scale = 0.6
    elif int(n_vertices) >= 100000:
        target_parallel_scale = 0.8
    elif int(n_vertices) >= 50000:
        target_parallel_scale = 1.0
    else:
        target_parallel_scale = 1.5
    target_limited_workers = max(1, int(np.ceil(target_count * target_parallel_scale)))
    return max(1, min(workers, target_limited_workers))

def resolve_search_probe_workers(
    requested_workers: int,
    n_vertices: int,
    n_preliminary_trials: int,
    min_cluster_size: int,
    target_count: int,
    planned_probe_count: int | None,
    runtime_context=None,
) -> int:
    """Resolve the actual worker count for the next probe batch."""
    workers = resolve_search_worker_capacity(
        requested_workers=requested_workers,
        n_vertices=n_vertices,
        n_preliminary_trials=n_preliminary_trials,
        min_cluster_size=min_cluster_size,
        target_count=target_count,
        runtime_context=runtime_context,
    )
    if planned_probe_count is not None:
        workers = min(workers, max(1, int(planned_probe_count)))
    return max(1, int(workers))

def global_resolution_search_interval_small(
    left: float,
    right: float,
    objective_function: str,
    resolution_tolerance: float,
) -> bool:
    """Return whether an interval is already below the refinement tolerance."""
    if not np.isfinite(left) or not np.isfinite(right) or left >= right:
        return True
    tolerance = max(resolution_tolerance, np.finfo(float).eps * 100)
    if objective_function == "CPM":
        return bool(abs(np.log(right) - np.log(left)) <= tolerance)
    return bool(abs(right - left) <= tolerance)

def global_resolution_search_interval_width(left: float, right: float, objective_function: str) -> float:
    """Measure interval width in either linear or log-gamma space."""
    if not np.isfinite(left) or not np.isfinite(right) or left >= right:
        return float("nan")
    if objective_function == "CPM":
        return float(np.log(right) - np.log(left))
    return float(right - left)

def global_resolution_search_internal_points(
    left: float,
    right: float,
    objective_function: str,
    n_points: int,
) -> np.ndarray:
    """Generate evenly spaced internal probe points for one interval."""
    n_points = max(0, int(n_points))
    if n_points <= 0 or not np.isfinite(left) or not np.isfinite(right) or left >= right:
        return np.asarray([], dtype=float)
    fractions = np.arange(1, n_points + 1, dtype=float) / float(n_points + 1)
    if objective_function == "CPM":
        return np.exp(np.log(left) + fractions * (np.log(right) - np.log(left)))
    return left + fractions * (right - left)

def global_resolution_search_probe_batch(
    graph,
    gamma_values: np.ndarray,
    sweep_round: int,
    objective_function: str,
    n_iter_preliminary: int,
    beta_preliminary: float,
    requested_max: int,
    min_cluster_size: int,
    snn_graph,
    active_probe_workers: int,
    verbose: bool,
    seed: int | None,
    probe_stage: str,
    coarse_probe_count: int | None = None,
    discovery_round: int | None = None,
    probe_metadata: pd.DataFrame | None = None,
    runtime_context=None,
) -> pd.DataFrame:
    """Evaluate a batch of search probes and return their diagnostics as a DataFrame."""
    gamma_values = np.asarray(gamma_values, dtype=float)
    metadata_lookup = {}
    if probe_metadata is not None and not probe_metadata.empty:
        metadata_lookup = probe_metadata.set_index("gamma").to_dict(orient="index")
    probe_items = [(probe_index, float(gamma)) for probe_index, gamma in enumerate(gamma_values.tolist(), start=1)]
    scheduled_workers = max(1, min(int(active_probe_workers), len(probe_items))) if probe_items else 1
    probe_state = {
        "graph": graph,
        "sweep_round": int(sweep_round),
        "objective_function": objective_function,
        "n_iter_preliminary": int(n_iter_preliminary),
        "beta_preliminary": float(beta_preliminary),
        "requested_max": int(requested_max),
        "min_cluster_size": int(min_cluster_size),
        "snn_graph": snn_graph,
        "active_probe_workers": int(scheduled_workers),
        "seed": seed,
        "probe_stage": probe_stage,
        "coarse_probe_count": coarse_probe_count,
        "discovery_round": discovery_round,
        "metadata_lookup": metadata_lookup,
        "runtime_context": runtime_context,
    }

    context = get_parallel_context()
    if context is not None and int(scheduled_workers) > 1 and len(probe_items) > 1:
        rows_by_index: list[dict[str, Any] | None] = [None] * len(probe_items)
        with context.Pool(
            processes=int(scheduled_workers),
            initializer=_init_search_probe_state,
            initargs=(probe_state,),
        ) as pool:
            for task_index, row in pool.imap_unordered(
                _run_single_probe_worker,
                list(enumerate(probe_items)),
                chunksize=1,
            ):
                rows_by_index[int(task_index)] = row
        rows = [row for row in rows_by_index if row is not None]
    else:
        rows = parallel_map_threads(
            probe_items,
            lambda item: _run_single_probe_impl(int(item[0]), float(item[1]), probe_state),
            max_workers=max(1, int(scheduled_workers)),
        )
    batch = pd.DataFrame(rows)
    if verbose and not batch.empty:
        logger.info(
            "RESOLUTION_SEARCH: probed %s gamma values in %s stage; final-count range [%s, %s]",
            len(batch),
            probe_stage,
            int(np.nanmin(batch["final_cluster_count"])),
            int(np.nanmax(batch["final_cluster_count"])),
        )
    return batch

def discover_cpm_upper_gamma(
    graph,
    gamma_bounds: tuple[float, float],
    requested_max: int,
    n_iter_preliminary: int,
    beta_preliminary: float,
    min_cluster_size: int,
    snn_graph,
    active_probe_workers: int,
    verbose: bool,
    seed: int | None,
    runtime_context=None,
    target_count: int = 1,
) -> dict[str, Any]:
    """Increase CPM gamma until target coverage or a high-gamma stop condition is reached."""
    current_gamma = max(float(gamma_bounds[0]), np.finfo(float).tiny)
    hard_cap_gamma = max(float(gamma_bounds[1]), current_gamma)
    frontier_final_cluster_count = np.nan
    probe_results = pd.DataFrame()
    discovered_upper_gamma = hard_cap_gamma
    coverage_upper_gamma = np.nan
    upper_cap_stop_reason = "hard_cap"
    nondegenerate_seen = False
    consecutive_high_degenerate = 0
    discovery_round = 0
    n_vertices = int(graph.vcount())
    if int(min_cluster_size) > 1 and int(requested_max) >= 10 and n_vertices >= 200000:
        post_coverage_rounds = 2
    else:
        post_coverage_rounds = 0
    coverage_round: int | None = None

    while current_gamma <= hard_cap_gamma:
        discovery_round += 1
        plan = derive_cpm_discovery_batch_plan(active_probe_workers, requested_max, frontier_final_cluster_count)
        batch_gamma_values = build_cpm_discovery_batch_gamma_values(
            current_gamma=current_gamma,
            hard_cap_gamma=hard_cap_gamma,
            batch_size=plan["batch_size"],
            step_ratio=plan["step_ratio"],
        )
        if batch_gamma_values.size == 0:
            break
        stage_workers = resolve_search_probe_workers(
            requested_workers=active_probe_workers,
            n_vertices=n_vertices,
            n_preliminary_trials=n_iter_preliminary,
            min_cluster_size=min_cluster_size,
            target_count=target_count,
            planned_probe_count=int(batch_gamma_values.size),
            runtime_context=runtime_context,
        )

        batch = global_resolution_search_probe_batch(
            graph=graph,
            gamma_values=batch_gamma_values,
            sweep_round=0,
            objective_function="CPM",
            n_iter_preliminary=n_iter_preliminary,
            beta_preliminary=beta_preliminary,
            requested_max=requested_max,
            min_cluster_size=min_cluster_size,
            snn_graph=snn_graph,
            active_probe_workers=stage_workers,
            verbose=verbose,
            seed=seed,
            probe_stage="upper_cap_discovery",
            discovery_round=discovery_round,
            runtime_context=runtime_context,
        )
        probe_results = pd.concat([probe_results, batch], ignore_index=True).drop_duplicates(subset=["gamma"]).sort_values("gamma").reset_index(drop=True)
        discovered_upper_gamma = float(batch_gamma_values.max())
        stabilized_final = stabilize_monotone_probe_counts(probe_results["final_cluster_count"].to_numpy())
        frontier_final_cluster_count = float(np.nanmax(stabilized_final)) if stabilized_final.size else np.nan
        coverage_mask = np.isfinite(stabilized_final) & (stabilized_final >= requested_max)
        current_coverage_gamma = (
            float(probe_results.loc[coverage_mask, "gamma"].min())
            if np.any(coverage_mask)
            else np.nan
        )

        batch_degenerate = batch["degenerate_high_gamma"].to_numpy(dtype=bool)
        if np.any(~batch_degenerate):
            nondegenerate_seen = True
            consecutive_high_degenerate = 0
        else:
            consecutive_high_degenerate += 1

        if np.isfinite(frontier_final_cluster_count) and frontier_final_cluster_count >= requested_max:
            if coverage_round is None:
                coverage_round = discovery_round
                coverage_upper_gamma = (
                    float(current_coverage_gamma)
                    if np.isfinite(current_coverage_gamma)
                    else float(discovered_upper_gamma)
                )
                if post_coverage_rounds <= 0:
                    upper_cap_stop_reason = "target_covered"
                    break
            elif discovery_round - coverage_round >= post_coverage_rounds:
                upper_cap_stop_reason = "post_coverage_buffer" if post_coverage_rounds > 0 else "target_covered"
                break
        if nondegenerate_seen and consecutive_high_degenerate >= 2:
            upper_cap_stop_reason = "high_gamma_degenerate"
            break
        if discovered_upper_gamma >= hard_cap_gamma:
            upper_cap_stop_reason = "hard_cap"
            break
        current_gamma = min(hard_cap_gamma, float(batch_gamma_values.max()) * float(plan["step_ratio"]))

    return {
        "probe_results": probe_results,
        "discovered_upper_gamma": discovered_upper_gamma,
        "coverage_upper_gamma": coverage_upper_gamma,
        "upper_cap_stop_reason": upper_cap_stop_reason,
    }

def find_resolution_ranges(
    graph,
    cluster_range: np.ndarray,
    start_g: float,
    end_g: float,
    objective_function: str,
    resolution_tolerance: float,
    n_workers: int,
    verbose: bool,
    seed: int | None = None,
    snn_graph=None,
    min_cluster_size: int = 1,
    in_parallel_context: bool = False,
    runtime_context=None,
) -> dict[int, tuple[float, float]]:
    """Run the shared resolution search and derive optimization-ready gamma intervals for each requested target."""
    min_cluster_size = max(1, int(min_cluster_size))
    cluster_range = np.unique(np.asarray(cluster_range, dtype=int))
    if cluster_range.size == 0:
        return {}

    n_vertices = graph.vcount()
    n_preliminary_trials = 3 if n_vertices >= 200000 else (5 if n_vertices >= 100000 else 15)
    n_iter_preliminary = 3 if n_vertices >= 200000 else 5
    beta_preliminary = 0.01
    requested_search_workers = max(1, int(n_workers))
    search_worker_budget = 1 if in_parallel_context else requested_search_workers
    available_probe_workers = resolve_search_worker_capacity(
        requested_workers=search_worker_budget,
        n_vertices=n_vertices,
        n_preliminary_trials=n_preliminary_trials,
        min_cluster_size=min_cluster_size,
        target_count=int(cluster_range.size),
        runtime_context=runtime_context,
    )
    gamma_bounds = (
        (float(np.exp(start_g)), float(np.exp(end_g)))
        if objective_function == "CPM"
        else (float(start_g), float(end_g))
    )
    requested_max = int(cluster_range.max())
    discovered_upper_gamma = gamma_bounds[1]
    coverage_upper_gamma = np.nan
    coarse_upper_gamma = gamma_bounds[1]
    upper_cap_stop_reason = None
    discovery_probe_results = pd.DataFrame()
    coarse_probe_count = int(min(max(2 * int(cluster_range.size), 3 * available_probe_workers, 12), 30))

    if verbose:
        logger.info(
            "RESOLUTION_SEARCH: Worker allocation - requested: %s available probe workers: %s preliminary trials per step: %s graph vertices: %s targets: %s",
            requested_search_workers,
            available_probe_workers,
            n_preliminary_trials,
            n_vertices,
            int(cluster_range.size),
        )
        logger.info(
            "RESOLUTION_SEARCH: Search bounds [%.6g, %.6g] objective=%s coarse_probe_count=%s",
            float(gamma_bounds[0]),
            float(gamma_bounds[1]),
            objective_function,
            coarse_probe_count,
        )

    if objective_function == "CPM":
        discovery_state = discover_cpm_upper_gamma(
            graph=graph,
            gamma_bounds=gamma_bounds,
            requested_max=requested_max,
            n_iter_preliminary=n_iter_preliminary,
            beta_preliminary=beta_preliminary,
            min_cluster_size=min_cluster_size,
            snn_graph=snn_graph,
            active_probe_workers=available_probe_workers,
            verbose=verbose,
            seed=seed,
            runtime_context=runtime_context,
            target_count=int(cluster_range.size),
        )
        discovery_probe_results = discovery_state["probe_results"]
        discovered_upper_gamma = discovery_state["discovered_upper_gamma"]
        coverage_upper_gamma = discovery_state.get("coverage_upper_gamma", np.nan)
        upper_cap_stop_reason = discovery_state["upper_cap_stop_reason"]
        if objective_function == "CPM" and np.isfinite(coverage_upper_gamma):
            coarse_upper_gamma = float(min(discovered_upper_gamma, coverage_upper_gamma))
        else:
            coarse_upper_gamma = float(discovered_upper_gamma)
    else:
        coarse_upper_gamma = float(discovered_upper_gamma)

    if verbose and objective_function == "CPM":
        logger.info(
            "RESOLUTION_SEARCH: discovery upper gamma = %.6g, coverage upper gamma = %.6g, coarse upper gamma = %.6g",
            float(discovered_upper_gamma),
            float(coverage_upper_gamma) if np.isfinite(coverage_upper_gamma) else float("nan"),
            float(coarse_upper_gamma),
        )

    coarse_gamma_values = build_gamma_sequence_for_range(
        gamma_range=(gamma_bounds[0], coarse_upper_gamma),
        objective_function=objective_function,
        resolution_tolerance=resolution_tolerance,
        n_vertices=n_vertices,
        n_steps=coarse_probe_count,
    )
    if not discovery_probe_results.empty:
        coarse_gamma_values = coarse_gamma_values[~np.isin(coarse_gamma_values, discovery_probe_results["gamma"].to_numpy(dtype=float))]
    all_probe_results = global_resolution_search_probe_batch(
        graph=graph,
        gamma_values=coarse_gamma_values,
        sweep_round=1,
        objective_function=objective_function,
        n_iter_preliminary=n_iter_preliminary,
        beta_preliminary=beta_preliminary,
        requested_max=requested_max,
        min_cluster_size=min_cluster_size,
        snn_graph=snn_graph,
        active_probe_workers=resolve_search_probe_workers(
            requested_workers=available_probe_workers,
            n_vertices=n_vertices,
            n_preliminary_trials=n_preliminary_trials,
            min_cluster_size=min_cluster_size,
            target_count=int(cluster_range.size),
            planned_probe_count=int(coarse_gamma_values.size),
            runtime_context=runtime_context,
        ),
        verbose=verbose,
        seed=seed,
        probe_stage="coarse",
        coarse_probe_count=coarse_probe_count,
        runtime_context=runtime_context,
    )
    all_probe_results = (
        pd.concat([discovery_probe_results, all_probe_results], ignore_index=True)
        .drop_duplicates(subset=["gamma"])
        .sort_values("gamma")
        .reset_index(drop=True)
    )
    all_probe_results["discovered_upper_gamma"] = float(discovered_upper_gamma)
    all_probe_results["coverage_upper_gamma"] = float(coverage_upper_gamma) if np.isfinite(coverage_upper_gamma) else np.nan
    all_probe_results["coarse_upper_gamma"] = float(coarse_upper_gamma)
    all_probe_results["upper_cap_stop_reason"] = upper_cap_stop_reason
    all_probe_results["coarse_probe_count"] = int(coarse_probe_count)

    interval_state = derive_shared_gamma_intervals(
        all_probe_results,
        cluster_range,
        gamma_bounds,
        objective_function=objective_function,
    )
    all_probe_results = interval_state.get("annotated_probes_df", all_probe_results)
    previous_max_final = float(np.nanmax(stabilize_monotone_probe_counts(all_probe_results["final_cluster_count"].to_numpy(dtype=float))))
    previous_ready_count = len(interval_state["optimization_ready_targets"])
    plateau_count = 0
    plateau_stop = False
    max_search_iterations = 30 if n_vertices >= 200000 else 50

    for sweep_round in range(2, max_search_iterations + 2):
        if not interval_state["unresolved_targets"]:
            break

        refinement_plan = build_refinement_probe_plan(
            unresolved_intervals=interval_state["unresolved_intervals"],
            objective_function=objective_function,
            resolution_tolerance=resolution_tolerance,
            active_probe_workers=available_probe_workers,
            existing_gamma_values=all_probe_results["gamma"].to_numpy(dtype=float),
        )
        next_probe_metadata = refinement_plan["probe_metadata"]
        next_probe_values = next_probe_metadata["gamma"].to_numpy(dtype=float) if not next_probe_metadata.empty else np.asarray([], dtype=float)
        if next_probe_values.size == 0:
            break

        new_probe_results = global_resolution_search_probe_batch(
            graph=graph,
            gamma_values=next_probe_values,
            sweep_round=sweep_round,
            objective_function=objective_function,
            n_iter_preliminary=n_iter_preliminary,
            beta_preliminary=beta_preliminary,
            requested_max=requested_max,
            min_cluster_size=min_cluster_size,
            snn_graph=snn_graph,
            active_probe_workers=resolve_search_probe_workers(
                requested_workers=available_probe_workers,
                n_vertices=n_vertices,
                n_preliminary_trials=n_preliminary_trials,
                min_cluster_size=min_cluster_size,
                target_count=int(len(interval_state["unresolved_targets"])),
                planned_probe_count=int(next_probe_values.size),
                runtime_context=runtime_context,
            ),
            verbose=verbose,
            seed=seed,
            probe_stage="refinement",
            coarse_probe_count=coarse_probe_count,
            probe_metadata=next_probe_metadata,
            runtime_context=runtime_context,
        )
        all_probe_results = (
            pd.concat([all_probe_results, new_probe_results], ignore_index=True)
            .drop_duplicates(subset=["gamma"])
            .sort_values("gamma")
            .reset_index(drop=True)
        )
        all_probe_results["discovered_upper_gamma"] = float(discovered_upper_gamma)
        all_probe_results["coverage_upper_gamma"] = float(coverage_upper_gamma) if np.isfinite(coverage_upper_gamma) else np.nan
        all_probe_results["coarse_upper_gamma"] = float(coarse_upper_gamma)
        all_probe_results["upper_cap_stop_reason"] = upper_cap_stop_reason
        all_probe_results["coarse_probe_count"] = int(coarse_probe_count)

        interval_state = derive_shared_gamma_intervals(
            all_probe_results,
            cluster_range,
            gamma_bounds,
            objective_function=objective_function,
        )
        all_probe_results = interval_state.get("annotated_probes_df", all_probe_results)
        current_max_final = float(np.nanmax(stabilize_monotone_probe_counts(all_probe_results["final_cluster_count"].to_numpy(dtype=float))))
        current_ready_count = len(interval_state["optimization_ready_targets"])
        no_growth = (not np.isfinite(previous_max_final) and not np.isfinite(current_max_final)) or (
            np.isfinite(previous_max_final) and np.isfinite(current_max_final) and current_max_final <= previous_max_final
        )
        no_new_ready = current_ready_count <= previous_ready_count
        if no_growth and no_new_ready:
            plateau_count += 1
            all_probe_results.loc[all_probe_results["sweep_round"] == sweep_round, "plateau_round"] = plateau_count
        else:
            plateau_count = 0
        previous_max_final = current_max_final
        previous_ready_count = current_ready_count
        if plateau_count >= 2:
            plateau_stop = True
            break

    all_probe_results["selected_for_target_interval"] = all_probe_results["gamma"].isin(interval_state["selected_gamma_values"])
    stabilized_final = stabilize_monotone_probe_counts(all_probe_results["final_cluster_count"].to_numpy(dtype=float))
    coverage_complete = (
        len(interval_state["unresolved_targets"]) == 0
        and np.any(np.isfinite(stabilized_final))
        and float(np.nanmax(stabilized_final)) >= requested_max
    )
    uncovered_targets = np.asarray(
        sorted(set(map(int, cluster_range.tolist())) - set(map(int, interval_state["optimization_ready_targets"]))),
        dtype=int,
    )

    gamma_dict = interval_state["gamma_dict"]
    setattr_obj = {
        "resolution_search_diagnostics": all_probe_results.reset_index(drop=True),
        "coverage_complete": bool(coverage_complete),
        "plateau_stop": bool(plateau_stop),
        "uncovered_targets": uncovered_targets,
        "target_gamma_seeds": interval_state["target_gamma_seeds"],
        "target_interval_details": interval_state["target_interval_details"],
        "discovered_upper_gamma": float(discovered_upper_gamma),
        "coverage_upper_gamma": float(coverage_upper_gamma) if np.isfinite(coverage_upper_gamma) else np.nan,
        "coarse_upper_gamma": float(coarse_upper_gamma),
        "active_probe_workers": int(available_probe_workers),
        "upper_cap_stop_reason": upper_cap_stop_reason,
        "coarse_probe_count": int(coarse_probe_count),
    }
    if verbose:
        logger.info(
            "RESOLUTION_SEARCH: completed with %s optimization-ready targets, %s uncovered targets, %s total probes",
            len(interval_state["optimization_ready_targets"]),
            len(uncovered_targets),
            len(all_probe_results),
        )
        if uncovered_targets.size:
            logger.info(
                "RESOLUTION_SEARCH: uncovered targets after shared sweep: %s",
                ", ".join(map(str, uncovered_targets.tolist())),
            )
    gamma_dict_with_attrs = dict(gamma_dict)
    gamma_dict_with_attrs["_attrs"] = setattr_obj
    return gamma_dict_with_attrs
