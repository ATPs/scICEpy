"""Parallel dispatch and target scheduling helpers for the public scICE entry point."""

from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from .leiden_wrapper import leiden_clustering
from .metrics import calculate_ic_from_extracted, extract_clustering_array
from .gamma_candidates import (
    build_optimization_gamma_batches,
    derive_gamma_admission_state,
    should_expand_phase1_secondary,
)
from .gamma_execution import _evaluate_gamma
from .target_optimizer import (
    evaluate_fixed_resolution,
    optimize_clustering,
)
from .resolution_search import global_resolution_search_midpoint
from .results import (
    build_target_result_record,
)
from .runtime import (
    cap_workers_by_memory,
    estimate_trial_matrix_bytes,
    get_parallel_context,
    initialize_parallel_state,
    logger,
)

_PARALLEL_STATE: dict[str, Any] = {}

def _init_parallel_state(state: dict[str, Any]) -> None:
    """Initialize process-pool workers with the shared API execution state."""
    initialize_parallel_state(_PARALLEL_STATE, state)

def _match_resolution_counts(
    seed_table: pd.DataFrame,
    resolution_search_diagnostics: pd.DataFrame | None,
    gamma_left: float,
    gamma_right: float,
) -> pd.DataFrame:
    """Attach search-time raw and final cluster counts to the target seed table when exact gamma matches exist."""
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
    """Build the seed gamma table that guides optimization for one requested target cluster."""
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
    """Filter requested target clusters by probing whether their shared-search interval is already too inconsistent."""
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

def _evaluate_manual_resolution_impl(resolution_value: float, state: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one manual resolution value using the shared manual-resolution execution state."""
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

def _evaluate_manual_resolution_worker(task: tuple[int, float]) -> tuple[int, dict[str, Any]]:
    """Process-pool wrapper for manual-resolution evaluation."""
    task_index, resolution_value = task
    return int(task_index), _evaluate_manual_resolution_impl(float(resolution_value), _PARALLEL_STATE)

def _map_manual_resolutions(
    resolution_values: np.ndarray,
    state: dict[str, Any],
    active_workers: int,
) -> list[dict[str, Any]]:
    """Evaluate all manual resolutions with either sequential execution or a process pool."""
    resolution_values = np.asarray(resolution_values, dtype=float)
    active_workers = max(1, min(int(active_workers), int(resolution_values.size)))
    context = get_parallel_context()
    if context is None or active_workers <= 1 or resolution_values.size <= 1:
        return [_evaluate_manual_resolution_impl(float(value), state) for value in resolution_values]

    ordered_results: list[dict[str, Any] | None] = [None] * int(resolution_values.size)
    with context.Pool(
        processes=active_workers,
        initializer=_init_parallel_state,
        initargs=(state,),
    ) as pool:
        for task_index, result in pool.imap_unordered(
            _evaluate_manual_resolution_worker,
            list(enumerate(resolution_values.tolist())),
            chunksize=1,
        ):
            ordered_results[int(task_index)] = result
    return [result for result in ordered_results if result is not None]

def _optimize_target_cluster_impl(cluster_num: int, state: dict[str, Any]) -> dict[str, Any]:
    """Optimize one requested target cluster using its shared-search gamma interval and worker budget."""
    cluster_num = int(cluster_num)
    gamma_range = state["gamma_dict"].get(cluster_num)
    if gamma_range is None:
        return build_target_result_record(
            cluster_num,
            excluded=True,
            exclusion_reason="resolution_search_failed",
            result_status="resolution_search_failed",
        )
    target_worker_budget = int(
        state.get("finalize_worker_budgets", {}).get(
            cluster_num,
            state.get("target_worker_budgets", {}).get(cluster_num, state["n_workers"]),
        )
    )
    precomputed_phase1 = None
    precomputed_phase1_by_target = state.get("precomputed_phase1_by_target")
    if isinstance(precomputed_phase1_by_target, dict):
        precomputed_phase1 = precomputed_phase1_by_target.get(cluster_num)

    if state.get("verbose", False):
        logger.info("WORKER %s: Starting optimization for target k = %s", cluster_num, cluster_num)
        logger.info(
            "WORKER %s: Gamma range [%.6g, %.6g]",
            cluster_num,
            float(gamma_range[0]),
            float(gamma_range[1]),
        )
        logger.info("WORKER %s: Assigned per-target worker budget = %s", cluster_num, target_worker_budget)
        if isinstance(precomputed_phase1, dict):
            logger.info(
                "WORKER %s: Reusing precomputed global Phase 1 results from process pool (%s worker(s))",
                cluster_num,
                int(precomputed_phase1.get("phase1_pool_workers", 1)),
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
        n_workers=target_worker_budget,
        snn_graph=state["snn_graph"],
        gamma_seed_values=gamma_seed_table,
        min_cluster_size=state["min_cluster_size"],
        verbose=state["verbose"],
        worker_id=f"WORKER {cluster_num}",
        runtime_context=state["runtime_context"],
        in_parallel_context=bool(state.get("in_parallel_context", False)),
        precomputed_phase1=precomputed_phase1,
    )
    if state.get("verbose", False):
        logger.info(
            "WORKER %s: Optimization completed in %.3f seconds",
            cluster_num,
            float(optimization_result.get("optimization_elapsed_sec", 0.0)),
        )
    if not optimization_result.get("success", False):
        failure_reason = str(optimization_result.get("failure_reason", "optimization_failed"))
        return build_target_result_record(
            cluster_num,
            result=optimization_result,
            excluded=True,
            exclusion_reason=failure_reason,
            result_status=failure_reason,
            admission_mode=str(optimization_result.get("admission_mode", failure_reason)),
        )

    return build_target_result_record(cluster_num, result=optimization_result)

def _optimize_target_cluster_worker(task: tuple[int, int]) -> tuple[int, dict[str, Any]]:
    """Process-pool wrapper for per-target optimization."""
    task_index, cluster_num = task
    return int(task_index), _optimize_target_cluster_impl(int(cluster_num), _PARALLEL_STATE)

def _estimate_target_cost(cluster_num: int, state: dict[str, Any]) -> tuple[float, float, int]:
    """Estimate per-target optimization cost so expensive targets can be scheduled first."""
    detail = state.get("target_interval_details", {}).get(str(int(cluster_num)), {})
    gamma_range = state["gamma_dict"].get(int(cluster_num), (np.nan, np.nan))
    left, right = sorted((float(gamma_range[0]), float(gamma_range[1])))
    if state.get("objective_function") == "CPM" and left > 0 and right > 0:
        interval_width = float(abs(np.log(right) - np.log(left)))
    else:
        interval_width = float(abs(right - left))

    mode = str(detail.get("mode", "missing"))
    exact_probe_count = int(len(detail.get("exact_probe_values", []) or []))
    near_probe_count = int(len(detail.get("near_probe_values", []) or []))
    seed_count = int(len(detail.get("seed_gamma_values", []) or []))
    recovery_risk = 1.0
    if exact_probe_count == 0:
        recovery_risk += 2.0
    if near_probe_count > 0:
        recovery_risk += 0.5
    if "near" in mode or "bracket" in mode:
        recovery_risk += 0.5
    recovery_risk += min(seed_count, 8) * 0.05
    return (recovery_risk, interval_width, int(cluster_num))

def _should_use_global_phase1_process_pool(
    scheduled_clusters: list[int],
    state: dict[str, Any],
    active_workers: int,
) -> bool:
    """Choose whether large runs should precompute Phase 1 gamma evaluations in a shared process pool."""
    if os.name == "nt":
        return False
    if len(scheduled_clusters) <= 1:
        return False
    if int(state.get("graph").vcount()) < 200000:
        return False
    total_workers = int(state.get("total_workers_requested", state.get("n_workers", 1)))
    return total_workers >= 2

def _build_phase1_log_every(primary_count: int, secondary_count: int) -> int:
    """Return the logging stride for shared Phase 1 progress messages."""
    return max(1, int(math.floor(max(int(primary_count), int(secondary_count), 1) / 5)))

def _should_log_phase1_step(step_idx: int, log_every: int) -> bool:
    """Return whether the current shared Phase 1 task should emit a progress log."""
    return int(step_idx) == 1 or (int(step_idx) % max(1, int(log_every))) == 0

def _build_target_phase1_plan(cluster_num: int, state: dict[str, Any]) -> dict[str, Any]:
    """Build the Phase 1 gamma batches and logging metadata for one target cluster."""
    gamma_range = state["gamma_dict"].get(int(cluster_num))
    gamma_seed_table = _build_target_gamma_seed_table(
        target_cluster=int(cluster_num),
        gamma_dict=state["gamma_dict"],
        objective_function=state["objective_function"],
        target_gamma_seeds=state["target_gamma_seeds"],
        target_interval_details=state["target_interval_details"],
        resolution_search_diagnostics=state["resolution_search_diagnostics"],
    )
    gamma_batches = build_optimization_gamma_batches(
        gamma_range=gamma_range,
        gamma_seed_values=gamma_seed_table,
        target_clusters=int(cluster_num),
        objective_function=state["objective_function"],
        resolution_tolerance=state["resolution_tolerance"],
        n_vertices=int(state["graph"].vcount()),
        primary_budget=8,
        secondary_budget=4,
    )
    primary_gamma_sequence = np.asarray(gamma_batches["primary_gammas"], dtype=float)
    secondary_gamma_sequence = np.asarray(gamma_batches["secondary_gammas"], dtype=float)
    return {
        "target_clusters": int(cluster_num),
        "worker_id": f"WORKER {int(cluster_num)}",
        "gamma_range": gamma_range,
        "gamma_seed_table": gamma_seed_table,
        "primary_gamma_sequence": primary_gamma_sequence,
        "secondary_gamma_sequence": secondary_gamma_sequence,
        "phase1_log_every": _build_phase1_log_every(
            len(primary_gamma_sequence),
            len(secondary_gamma_sequence),
        ),
        "phase1_expected_runs": int(
            (len(primary_gamma_sequence) + len(secondary_gamma_sequence)) * max(1, int(state["n_trials"]))
        ),
    }

def _evaluate_global_phase1_task(task: tuple[int, dict[str, Any]]) -> tuple[int, int, str, int, float, dict[str, Any]]:
    """Evaluate one shared Phase 1 gamma task and return enough metadata to reassemble per-target batches."""
    task_index, spec = task
    target_clusters = int(spec["target_clusters"])
    cluster_seed = None if _PARALLEL_STATE.get("seed") is None else int(_PARALLEL_STATE["seed"] + target_clusters * 1000)
    result = _evaluate_gamma(
        graph=_PARALLEL_STATE["graph"],
        gamma_val=float(spec["gamma"]),
        target_clusters=target_clusters,
        objective_function=_PARALLEL_STATE["objective_function"],
        n_trials=_PARALLEL_STATE["n_trials"],
        beta=_PARALLEL_STATE["beta"],
        n_iterations=_PARALLEL_STATE["n_iterations"],
        seed=cluster_seed,
        snn_graph=_PARALLEL_STATE["snn_graph"],
        min_cluster_size=_PARALLEL_STATE["min_cluster_size"],
        worker_id=str(spec["worker_id"]),
        verbose=bool(_PARALLEL_STATE["verbose"]),
        runtime_context=_PARALLEL_STATE["runtime_context"],
        gamma_idx=int(spec["gamma_idx"]),
        gamma_total=int(spec["gamma_total"]),
        log_this_gamma=bool(spec["log_this_gamma"]),
    )
    result["_gamma_batch"] = str(spec["batch_label"])
    result["_phase_name"] = str(spec["phase_name"])
    return (
        int(task_index),
        target_clusters,
        str(spec["batch_kind"]),
        int(spec["gamma_idx"]),
        float(time.time()),
        result,
    )

def _resolve_global_phase1_workers(
    task_specs: list[dict[str, Any]],
    state: dict[str, Any],
) -> int:
    """Resolve the worker count for the shared Phase 1 process pool."""
    total_workers = max(1, int(state.get("total_workers_requested", state.get("n_workers", 1))))
    requested = min(total_workers, max(1, len(task_specs)))
    return cap_workers_by_memory(
        requested_workers=requested,
        bytes_per_task=estimate_trial_matrix_bytes(int(state["graph"].vcount()), int(state["n_trials"]), 1),
        runtime_context=state.get("runtime_context"),
    )

def _execute_global_phase1_tasks(
    task_specs: list[dict[str, Any]],
    state: dict[str, Any],
    phase1_workers: int,
) -> dict[int, dict[str, Any]]:
    """Execute shared Phase 1 gamma tasks and regroup the outputs by target cluster."""
    if not task_specs:
        return {}

    batch_start = time.time()
    results_by_target: dict[int, dict[int, dict[str, Any] | None]] = {}
    completion_times: dict[int, list[float]] = {}
    context = get_parallel_context()

    def _record_output(output: tuple[int, int, str, int, float, dict[str, Any]]) -> None:
        """Store one shared Phase 1 worker result under its target-cluster/gamma slot."""
        _, target_clusters, _, gamma_idx, completed_at, result = output
        results_by_target.setdefault(int(target_clusters), {})[int(gamma_idx)] = result
        completion_times.setdefault(int(target_clusters), []).append(float(completed_at))

    if context is None or int(phase1_workers) <= 1 or len(task_specs) <= 1:
        for task_index, spec in enumerate(task_specs):
            _record_output(_evaluate_global_phase1_task((int(task_index), spec)))
    else:
        with context.Pool(
            processes=int(phase1_workers),
            initializer=_init_parallel_state,
            initargs=(state,),
        ) as pool:
            for output in pool.imap_unordered(
                _evaluate_global_phase1_task,
                list(enumerate(task_specs)),
                chunksize=1,
            ):
                _record_output(output)

    batch_results: dict[int, dict[str, Any]] = {}
    for target_clusters, ordered_results in results_by_target.items():
        ordered_gamma_indices = sorted(ordered_results)
        collected = [ordered_results[idx] for idx in ordered_gamma_indices if ordered_results[idx] is not None]
        batch_elapsed = 0.0
        if completion_times.get(int(target_clusters)):
            batch_elapsed = max(completion_times[int(target_clusters)]) - batch_start
        batch_results[int(target_clusters)] = {
            "results": collected,
            "elapsed_sec": float(max(0.0, batch_elapsed)),
            "gamma_count": int(len(collected)),
            "leiden_runs": int(len(collected) * max(1, int(state["n_trials"]))),
            "nested_workers": 1,
        }
    return batch_results

def _build_global_phase1_precomputed(
    scheduled_clusters: list[int],
    state: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Precompute reusable Phase 1 gamma evaluations for all scheduled target clusters."""
    phase1_plans = {
        int(cluster_num): _build_target_phase1_plan(int(cluster_num), state)
        for cluster_num in scheduled_clusters
    }

    primary_task_specs: list[dict[str, Any]] = []
    for cluster_num in scheduled_clusters:
        plan = phase1_plans[int(cluster_num)]
        for gamma_idx, gamma_val in enumerate(plan["primary_gamma_sequence"].tolist(), start=1):
            primary_task_specs.append(
                {
                    "target_clusters": int(cluster_num),
                    "worker_id": plan["worker_id"],
                    "gamma": float(gamma_val),
                    "gamma_idx": int(gamma_idx),
                    "gamma_total": int(len(plan["primary_gamma_sequence"])),
                    "log_this_gamma": bool(
                        state.get("verbose", False)
                        and _should_log_phase1_step(int(gamma_idx), int(plan["phase1_log_every"]))
                    ),
                    "batch_kind": "primary",
                    "batch_label": "Primary Phase 1",
                    "phase_name": "phase1_primary",
                }
            )

    phase1_workers = _resolve_global_phase1_workers(primary_task_specs, state)
    if state.get("verbose", False):
        logger.info(
            "CLUSTERING_MAIN: Using global Phase 1 process pool with %s worker(s) across %s primary gamma task(s)",
            phase1_workers,
            len(primary_task_specs),
        )

    primary_phase1 = _execute_global_phase1_tasks(primary_task_specs, state, phase1_workers=phase1_workers)

    secondary_task_specs: list[dict[str, Any]] = []
    secondary_phase1_used_by_target: dict[int, bool] = {}
    for cluster_num in scheduled_clusters:
        plan = phase1_plans[int(cluster_num)]
        primary_results = primary_phase1.get(int(cluster_num), {}).get("results", [])
        primary_admission = derive_gamma_admission_state(
            primary_results,
            int(cluster_num),
            min_cluster_size=state["min_cluster_size"],
            verbose=False,
            worker_id=plan["worker_id"],
        )
        secondary_used = should_expand_phase1_secondary(
            primary_admission["valid_indices"],
            primary_admission["admission_mode"],
            primary_admission["exact_hit_gamma_count"],
        )
        secondary_phase1_used_by_target[int(cluster_num)] = bool(secondary_used)
        if not secondary_used or len(plan["secondary_gamma_sequence"]) == 0:
            continue
        if state.get("verbose", False):
            if not primary_admission["valid_indices"]:
                logger.info(
                    "%s: Secondary batch triggered because primary batch produced no admitted gamma candidates",
                    plan["worker_id"],
                )
            else:
                logger.info(
                    "%s: Secondary batch triggered because primary batch ended at %s without exact final-hit gamma support",
                    plan["worker_id"],
                    primary_admission["admission_mode"],
                )
        for gamma_idx, gamma_val in enumerate(plan["secondary_gamma_sequence"].tolist(), start=1):
            secondary_task_specs.append(
                {
                    "target_clusters": int(cluster_num),
                    "worker_id": plan["worker_id"],
                    "gamma": float(gamma_val),
                    "gamma_idx": int(gamma_idx),
                    "gamma_total": int(len(plan["secondary_gamma_sequence"])),
                    "log_this_gamma": bool(
                        state.get("verbose", False)
                        and _should_log_phase1_step(int(gamma_idx), int(plan["phase1_log_every"]))
                    ),
                    "batch_kind": "secondary",
                    "batch_label": "Secondary Phase 1",
                    "phase_name": "phase1_secondary",
                }
            )

    secondary_phase1 = _execute_global_phase1_tasks(secondary_task_specs, state, phase1_workers=phase1_workers)

    precomputed: dict[int, dict[str, Any]] = {}
    for cluster_num in scheduled_clusters:
        plan = phase1_plans[int(cluster_num)]
        precomputed[int(cluster_num)] = {
            "primary_gamma_sequence": plan["primary_gamma_sequence"],
            "secondary_gamma_sequence": plan["secondary_gamma_sequence"],
            "gamma_seed_table": plan["gamma_seed_table"],
            "phase1_expected_runs": int(plan["phase1_expected_runs"]),
            "primary_phase1": primary_phase1.get(
                int(cluster_num),
                {"results": [], "elapsed_sec": 0.0, "gamma_count": 0, "leiden_runs": 0, "nested_workers": 1},
            ),
            "secondary_phase1": secondary_phase1.get(
                int(cluster_num),
                {"results": [], "elapsed_sec": 0.0, "gamma_count": 0, "leiden_runs": 0, "nested_workers": 1},
            ),
            "secondary_phase1_used": bool(secondary_phase1_used_by_target.get(int(cluster_num), False)),
            "phase1_pool_workers": int(phase1_workers),
        }
    return precomputed

def _resolve_target_worker_cap(
    scheduled_cluster_count: int,
    active_workers: int,
    total_workers: int,
    state: dict[str, Any],
) -> int:
    """Cap per-target inner workers based on graph size, trial counts, and available total workers."""
    max_parallel_from_work = max(1, min(int(total_workers), max(int(state["n_trials"]), int(state["n_bootstrap"]))))
    load_factor = float(total_workers) / float(max(1, active_workers))
    return min(max_parallel_from_work, max(1, int(math.ceil(load_factor))))

def _build_target_worker_budgets(
    scheduled_clusters: list[int],
    state: dict[str, Any],
    active_workers: int,
) -> dict[int, int]:
    """Allocate per-target worker budgets for concurrent optimization tasks."""
    if not scheduled_clusters:
        return {}

    default_inner = max(1, int(state["n_workers"]))
    total_workers = max(default_inner * max(1, int(active_workers)), int(state.get("total_workers_requested", default_inner)))
    concurrent_clusters = [int(cluster_num) for cluster_num in scheduled_clusters[: max(1, int(active_workers))]]
    budgets = {int(cluster_num): default_inner for cluster_num in scheduled_clusters}
    if int(state["graph"].vcount()) >= 200000 and concurrent_clusters:
        max_target_workers = _resolve_target_worker_cap(
            scheduled_cluster_count=len(scheduled_clusters),
            active_workers=active_workers,
            total_workers=total_workers,
            state=state,
        )
        worker_pool = min(int(total_workers), int(max_target_workers) * len(concurrent_clusters))
        base_budget, extra_workers = divmod(worker_pool, len(concurrent_clusters))
        base_budget = max(default_inner, min(int(base_budget), int(max_target_workers)))
        for cluster_num in concurrent_clusters:
            budgets[int(cluster_num)] = int(base_budget)
        remaining_headroom = sum(max(0, int(max_target_workers) - int(budgets[int(cluster_num)])) for cluster_num in concurrent_clusters)
        extra_workers = min(int(extra_workers), int(remaining_headroom))
        for cluster_num in concurrent_clusters:
            if extra_workers <= 0:
                break
            if budgets[int(cluster_num)] >= int(max_target_workers):
                continue
            budgets[int(cluster_num)] += 1
            extra_workers -= 1
        return budgets

    reserved_workers = max(1, int(active_workers)) * default_inner
    remaining = max(0, int(total_workers - reserved_workers))
    max_target_workers = _resolve_target_worker_cap(
        scheduled_cluster_count=len(scheduled_clusters),
        active_workers=active_workers,
        total_workers=total_workers,
        state=state,
    )
    if remaining <= 0 or max_target_workers <= default_inner or not concurrent_clusters:
        return budgets

    candidate_rounds: list[list[int]] = [concurrent_clusters]
    current_size = len(concurrent_clusters)
    while current_size > 1 and len(candidate_rounds) < max_target_workers - default_inner:
        current_size = max(1, int(math.ceil(current_size / 2.0)))
        candidate_rounds.append(concurrent_clusters[:current_size])

    for clusters_for_round in candidate_rounds:
        if remaining <= 0:
            break
        for cluster_num in clusters_for_round:
            if remaining <= 0:
                break
            if budgets[int(cluster_num)] >= max_target_workers:
                continue
            budgets[int(cluster_num)] += 1
            remaining -= 1
    return budgets

def _map_optimized_targets(
    valid_clusters: list[int],
    state: dict[str, Any],
    active_workers: int,
) -> list[dict[str, Any]]:
    """Optimize all retained target clusters, optionally reusing shared Phase 1 precomputations."""
    if not valid_clusters:
        return []
    active_workers = max(1, min(int(active_workers), len(valid_clusters)))
    scheduled_clusters = sorted(
        [int(cluster_num) for cluster_num in valid_clusters],
        key=lambda cluster_num: _estimate_target_cost(cluster_num, state),
        reverse=True,
    )
    target_worker_budgets = _build_target_worker_budgets(
        scheduled_clusters=scheduled_clusters,
        state=state,
        active_workers=active_workers,
    )
    state = dict(state)
    state["finalize_worker_budgets"] = target_worker_budgets
    if _should_use_global_phase1_process_pool(
        scheduled_clusters=scheduled_clusters,
        state=state,
        active_workers=active_workers,
    ):
        precomputed_phase1_by_target = _build_global_phase1_precomputed(
            scheduled_clusters=scheduled_clusters,
            state=state,
        )
        state["precomputed_phase1_by_target"] = precomputed_phase1_by_target
        if state.get("verbose", False):
            logger.info(
                "CLUSTERING_MAIN: Global Phase 1 precompute ready for %s target(s)",
                len(precomputed_phase1_by_target),
            )
    if state.get("verbose", False):
        budget_preview = ", ".join(
            f"k{int(cluster_num)}->{int(target_worker_budgets.get(int(cluster_num), state['n_workers']))}"
            for cluster_num in scheduled_clusters[: min(len(scheduled_clusters), max(1, active_workers))]
        )
        logger.info(
            "CLUSTERING_MAIN: Per-target worker budget preview for active frontier: %s",
            budget_preview if budget_preview else "none",
        )
    context = get_parallel_context()
    if context is None or active_workers <= 1 or len(valid_clusters) <= 1:
        ordered = [_optimize_target_cluster_impl(int(cluster_num), state) for cluster_num in scheduled_clusters]
        return sorted(
            ordered,
            key=lambda item: int(item.get("source_target_cluster", item.get("cluster_number", -1))),
        )

    ordered_results: list[dict[str, Any] | None] = [None] * len(scheduled_clusters)
    with context.Pool(
        processes=active_workers,
        initializer=_init_parallel_state,
        initargs=(state,),
    ) as pool:
        for task_index, result in pool.imap_unordered(
            _optimize_target_cluster_worker,
            list(enumerate(scheduled_clusters)),
            chunksize=1,
        ):
            ordered_results[int(task_index)] = result
    return sorted(
        [result for result in ordered_results if result is not None],
        key=lambda item: int(item.get("source_target_cluster", item.get("cluster_number", -1))),
    )
