"""Per-target optimization and fixed-resolution execution helpers."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd

from .cluster_utils import raw_cluster_guard_limits
from .leiden_wrapper import leiden_clustering
from .metrics import calculate_ic_from_extracted, extract_clustering_array
from .runtime import (
    cap_workers_by_memory,
    create_heartbeat_logger,
    estimate_trial_matrix_bytes,
    load_cluster_matrix,
    logger,
    parallel_map_threads,
    release_cluster_matrix_refs,
    store_cluster_matrix,
)
from .gamma_candidates import (
    build_optimization_gamma_batches,
    build_local_recovery_gamma_points,
    derive_gamma_admission_state,
    order_gamma_candidate_indices,
    phase4_iteration_cap_for_mode,
    preferred_trial_flags,
    should_expand_phase1_secondary,
    should_skip_phase4_refinement,
)
from .gamma_execution import (
    _evaluate_gamma,
    build_gamma_diagnostic_row,
    finalize_selected_clustering,
    summarize_trial_matrix,
)

def optimize_clustering(
    graph,
    target_clusters: int,
    gamma_range: tuple[float, float],
    objective_function: str,
    n_trials: int,
    n_bootstrap: int,
    seed: int | None,
    beta: float,
    n_iterations: int,
    max_iterations: int,
    resolution_tolerance: float,
    n_workers: int,
    snn_graph=None,
    gamma_seed_values=None,
    min_cluster_size: int = 1,
    verbose: bool = False,
    worker_id: str = "OPTIMIZER",
    runtime_context=None,
    in_parallel_context: bool = False,
    precomputed_phase1: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Optimize one target cluster count across staged gamma batches and bootstrap the selected clustering."""
    optimization_start = time.time()
    cluster_seed = None if seed is None else int(seed + int(target_clusters) * 1000)
    gamma_diagnostics_rows: list[dict[str, Any]] = []
    if verbose:
        logger.info("%s: Optimization parameters:", worker_id)
        logger.info("%s:   Target clusters: %s", worker_id, target_clusters)
        logger.info("%s:   Trials per gamma: %s", worker_id, n_trials)
        logger.info("%s:   Bootstrap iterations: %s", worker_id, n_bootstrap)
        logger.info("%s:   Max iterations: %s", worker_id, max_iterations)
        logger.info("%s:   Beta: %s", worker_id, beta)
        logger.info("%s:   Leiden iterations: %s", worker_id, n_iterations)
        if min_cluster_size > 1:
            limits = raw_cluster_guard_limits(target_clusters)
            logger.info("%s: Raw-cluster guard soft<=%s hard<=%s", worker_id, limits["soft"], limits["hard"])

    if precomputed_phase1 is None:
        gamma_batches = build_optimization_gamma_batches(
            gamma_range=gamma_range,
            gamma_seed_values=gamma_seed_values,
            target_clusters=target_clusters,
            objective_function=objective_function,
            resolution_tolerance=resolution_tolerance,
            n_vertices=graph.vcount(),
            primary_budget=8,
            secondary_budget=4,
        )
        primary_gamma_sequence = gamma_batches["primary_gammas"]
        secondary_gamma_sequence = gamma_batches["secondary_gammas"]
        gamma_seed_table = gamma_batches["seed_table"]
    else:
        primary_gamma_sequence = np.asarray(precomputed_phase1.get("primary_gamma_sequence", []), dtype=float)
        secondary_gamma_sequence = np.asarray(precomputed_phase1.get("secondary_gamma_sequence", []), dtype=float)
        gamma_seed_table = precomputed_phase1.get("gamma_seed_table")
    if verbose:
        logger.info(
            "%s: Primary Phase 1 gamma batch (%s): %s",
            worker_id,
            len(primary_gamma_sequence),
            ", ".join(map(lambda value: f"{float(value):.6g}", primary_gamma_sequence.tolist())) if primary_gamma_sequence.size else "none",
        )
        if secondary_gamma_sequence.size:
            logger.info(
                "%s: Secondary Phase 1 gamma batch (%s): %s",
                worker_id,
                len(secondary_gamma_sequence),
                ", ".join(map(lambda value: f"{float(value):.6g}", secondary_gamma_sequence.tolist())),
            )
        if gamma_seed_table is not None and len(gamma_seed_table) > 0:
            logger.info(
                "%s: Included %s target-specific gamma seeds from shared search diagnostics",
                worker_id,
                len(gamma_seed_table),
            )
        if precomputed_phase1 is not None:
            logger.info(
                "%s: Reusing precomputed global Phase 1 gamma evaluations from %s-worker process pool",
                worker_id,
                int(precomputed_phase1.get("phase1_pool_workers", 1)),
            )

    def compute_phase1_nested_workers(batch_gamma_count: int) -> int:
        """Resolve the nested worker budget for one Phase 1 gamma batch."""
        nested_workers = max(1, int(n_workers))
        nested_workers = min(nested_workers, max(1, int(batch_gamma_count)))
        return cap_workers_by_memory(
            nested_workers,
            estimate_trial_matrix_bytes(graph.vcount(), n_trials, 1),
            runtime_context,
        )

    phase1_log_every = max(
        1,
        int(math.floor(max(len(primary_gamma_sequence), len(secondary_gamma_sequence), 1) / 5)),
    )

    def should_log_phase1_step(step_idx: int) -> bool:
        """Decide whether the current Phase 1 gamma step should emit a progress log."""
        return step_idx == 1 or (step_idx % phase1_log_every) == 0

    def evaluate_gamma_batch(gamma_sequence: np.ndarray, batch_label: str) -> dict[str, Any]:
        """Evaluate one labeled Phase 1 gamma batch and return timing plus diagnostics."""
        gamma_sequence = np.asarray(gamma_sequence, dtype=float)
        if gamma_sequence.size == 0:
            return {
                "results": [],
                "elapsed_sec": 0.0,
                "gamma_count": 0,
                "leiden_runs": 0,
                "nested_workers": 1,
            }

        estimated_phase1_bytes = estimate_trial_matrix_bytes(graph.vcount(), n_trials, int(gamma_sequence.size))
        nested_workers = compute_phase1_nested_workers(int(gamma_sequence.size))
        if verbose:
            logger.info(
                "%s: %s - Testing %s gamma values with %s trials each",
                worker_id,
                batch_label,
                len(gamma_sequence),
                n_trials,
            )
            if in_parallel_context:
                logger.info(
                    "%s: Running in parallel context with per-cluster worker budget %s",
                    worker_id,
                    n_workers,
                )
            logger.info(
                "%s: Worker budget for this cluster: %s - using %s workers for %s gamma evaluation",
                worker_id,
                n_workers,
                nested_workers,
                batch_label,
            )
        batch_start = time.time()

        def run_gamma(gamma_item: tuple[int, float]) -> dict[str, Any]:
            """Evaluate one gamma value from the current Phase 1 batch."""
            gamma_idx, gamma_val = gamma_item
            return _evaluate_gamma(
                graph=graph,
                gamma_val=float(gamma_val),
                target_clusters=target_clusters,
                objective_function=objective_function,
                n_trials=n_trials,
                beta=beta,
                n_iterations=n_iterations,
                seed=cluster_seed,
                snn_graph=snn_graph,
                min_cluster_size=min_cluster_size,
                worker_id=worker_id,
                verbose=verbose,
                runtime_context=runtime_context,
                gamma_idx=int(gamma_idx),
                gamma_total=int(gamma_sequence.size),
                log_this_gamma=bool(verbose and should_log_phase1_step(int(gamma_idx))),
            )

        gamma_results = parallel_map_threads(
            [(idx + 1, float(gamma_val)) for idx, gamma_val in enumerate(gamma_sequence.tolist())],
            run_gamma,
            max_workers=nested_workers,
        )
        if "Primary" in batch_label:
            phase_name = "phase1_primary"
        elif "Secondary" in batch_label:
            phase_name = "phase1_secondary"
        else:
            phase_name = "phase1_recovery"
        for result in gamma_results:
            result["_gamma_batch"] = batch_label
            gamma_diagnostics_rows.append(
                build_gamma_diagnostic_row(
                    result,
                    phase=phase_name,
                    gamma_batch=batch_label,
                )
            )
        elapsed_sec = time.time() - batch_start
        if verbose:
            logger.info("%s: %s completed in %.3f seconds", worker_id, batch_label, elapsed_sec)
        return {
            "results": gamma_results,
            "elapsed_sec": elapsed_sec,
            "gamma_count": int(gamma_sequence.size),
            "leiden_runs": int(gamma_sequence.size) * max(1, int(n_trials)),
            "nested_workers": nested_workers,
        }

    if precomputed_phase1 is None:
        phase1_expected_runs = int((len(primary_gamma_sequence) + len(secondary_gamma_sequence)) * max(1, int(n_trials)))
    else:
        phase1_expected_runs = int(
            precomputed_phase1.get(
                "phase1_expected_runs",
                (len(primary_gamma_sequence) + len(secondary_gamma_sequence)) * max(1, int(n_trials)),
            )
        )
    if verbose:
        logger.info("%s: Phase 1 maximum expected Leiden runs: %s", worker_id, f"{phase1_expected_runs:,}")
    if precomputed_phase1 is None:
        primary_phase1 = evaluate_gamma_batch(primary_gamma_sequence, "Primary Phase 1")
        primary_results = primary_phase1["results"]
        primary_admission = derive_gamma_admission_state(primary_results, target_clusters, min_cluster_size=min_cluster_size, verbose=False, worker_id=worker_id)
        secondary_phase1_used = should_expand_phase1_secondary(primary_admission["valid_indices"], primary_admission["admission_mode"], primary_admission["exact_hit_gamma_count"])
        secondary_results: list[dict[str, Any]] = []
        secondary_phase1 = {
            "results": [],
            "elapsed_sec": 0.0,
            "gamma_count": 0,
            "leiden_runs": 0,
            "nested_workers": 1,
        }
        if secondary_phase1_used and secondary_gamma_sequence.size:
            if verbose:
                if not primary_admission["valid_indices"]:
                    logger.info("%s: Secondary batch triggered because primary batch produced no admitted gamma candidates", worker_id)
                else:
                    logger.info(
                        "%s: Secondary batch triggered because primary batch ended at %s without exact final-hit gamma support",
                        worker_id,
                        primary_admission["admission_mode"],
                    )
            secondary_phase1 = evaluate_gamma_batch(secondary_gamma_sequence, "Secondary Phase 1")
            secondary_results = secondary_phase1["results"]
    else:
        primary_phase1 = dict(precomputed_phase1.get("primary_phase1", {}))
        secondary_phase1 = dict(precomputed_phase1.get("secondary_phase1", {}))
        primary_phase1.setdefault("results", [])
        primary_phase1.setdefault("elapsed_sec", 0.0)
        primary_phase1.setdefault("gamma_count", len(primary_gamma_sequence))
        primary_phase1.setdefault("leiden_runs", len(primary_gamma_sequence) * max(1, int(n_trials)))
        primary_phase1.setdefault("nested_workers", 1)
        secondary_phase1.setdefault("results", [])
        secondary_phase1.setdefault("elapsed_sec", 0.0)
        secondary_phase1.setdefault("gamma_count", len(secondary_gamma_sequence))
        secondary_phase1.setdefault("leiden_runs", len(secondary_gamma_sequence) * max(1, int(n_trials)))
        secondary_phase1.setdefault("nested_workers", 1)
        primary_results = list(primary_phase1["results"])
        secondary_results = list(secondary_phase1["results"])
        secondary_phase1_used = bool(precomputed_phase1.get("secondary_phase1_used", False))
        for result in primary_results + secondary_results:
            gamma_diagnostics_rows.append(
                build_gamma_diagnostic_row(
                    result,
                    phase=str(result.get("_phase_name", "phase1_primary")),
                    gamma_batch=result.get("_gamma_batch"),
                )
            )
    gamma_results = primary_results + secondary_results
    recovery_phase1 = {
        "results": [],
        "elapsed_sec": 0.0,
        "gamma_count": 0,
        "leiden_runs": 0,
        "nested_workers": 1,
    }
    phase1_primary_gamma_count = int(primary_phase1["gamma_count"])
    phase1_secondary_gamma_count = int(secondary_phase1["gamma_count"])
    phase1_total_gamma_count = phase1_primary_gamma_count + phase1_secondary_gamma_count
    phase1_elapsed_sec = float(primary_phase1["elapsed_sec"] + secondary_phase1["elapsed_sec"])
    phase1_leiden_runs = int(primary_phase1["leiden_runs"] + secondary_phase1["leiden_runs"])
    phase1_nested_workers = max(1, int(max(primary_phase1["nested_workers"], secondary_phase1["nested_workers"])))
    if verbose:
        logger.info(
            "%s: Phase 1 completed in %.3f seconds - primary gammas = %s - secondary gammas = %s - total runs = %s",
            worker_id,
            phase1_elapsed_sec,
            phase1_primary_gamma_count,
            phase1_secondary_gamma_count,
            f"{phase1_leiden_runs:,}",
        )

    admission_state = derive_gamma_admission_state(gamma_results, target_clusters, min_cluster_size=min_cluster_size, verbose=verbose, worker_id=worker_id)
    valid_indices = admission_state["valid_indices"]
    admission_mode = admission_state["admission_mode"]
    exact_hit_gamma_count = int(admission_state["exact_hit_gamma_count"])
    if (not valid_indices or exact_hit_gamma_count <= 0) and gamma_results:
        recovery_gamma_sequence = build_local_recovery_gamma_points(
            gamma_results=gamma_results,
            gamma_range=gamma_range,
            objective_function=objective_function,
            target_clusters=target_clusters,
            resolution_tolerance=resolution_tolerance,
            n_points=4,
        )
        if recovery_gamma_sequence.size:
            if verbose:
                logger.info(
                    "%s: Recovery batch triggered with %s local gamma value(s) because admitted exact-hit support is still insufficient",
                    worker_id,
                    int(recovery_gamma_sequence.size),
                )
            recovery_phase1 = evaluate_gamma_batch(recovery_gamma_sequence, "Recovery Phase 1")
            gamma_results = gamma_results + recovery_phase1["results"]
            phase1_total_gamma_count += int(recovery_phase1["gamma_count"])
            phase1_elapsed_sec += float(recovery_phase1["elapsed_sec"])
            phase1_leiden_runs += int(recovery_phase1["leiden_runs"])
            phase1_nested_workers = max(phase1_nested_workers, int(recovery_phase1["nested_workers"]))
            admission_state = derive_gamma_admission_state(
                gamma_results,
                target_clusters,
                min_cluster_size=min_cluster_size,
                verbose=verbose,
                worker_id=worker_id,
            )
            valid_indices = admission_state["valid_indices"]
            admission_mode = admission_state["admission_mode"]
            exact_hit_gamma_count = int(admission_state["exact_hit_gamma_count"])

    if not valid_indices:
        release_cluster_matrix_refs([result.get("matrix_ref") for result in gamma_results if result.get("matrix_ref") is not None])
        failure_result = min(
            gamma_results,
            key=lambda x: (float(x.get("median_gap", np.inf)), float(x.get("raw_median_gap", np.inf)), float(x.get("gamma", np.inf))),
        )
        for idx, result in enumerate(gamma_results):
            gamma_diagnostics_rows.append(
                build_gamma_diagnostic_row(
                    result,
                    phase="admission",
                    gamma_batch=result.get("_gamma_batch"),
                    admission_selected=False,
                    admission_mode=admission_mode,
                )
            )
        return {
            "success": False,
            "failure_reason": "optimization_admission_failed",
            "gamma": float(failure_result.get("gamma", np.nan)),
            "ic_median": np.nan,
            "ic_bootstrap": np.asarray([], dtype=float),
            "best_labels": None,
            "effective_cluster_median": float(failure_result.get("median_effective_clusters", np.nan)),
            "raw_cluster_median": float(failure_result.get("raw_cluster_median", np.nan)),
            "final_cluster_median": float(failure_result.get("final_cluster_median", np.nan)),
            "admission_mode": "optimization_admission_failed",
            "best_labels_raw_cluster_count": -1,
            "best_labels_final_cluster_count": -1,
            "phase1_primary_gamma_count": phase1_primary_gamma_count,
            "phase1_secondary_gamma_count": phase1_secondary_gamma_count,
            "phase1_total_gamma_count": phase1_total_gamma_count,
            "phase1_elapsed_sec": phase1_elapsed_sec,
            "phase1_leiden_runs": phase1_leiden_runs,
            "secondary_phase1_used": bool(secondary_phase1_used),
            "exact_hit_gamma_count": exact_hit_gamma_count,
            "phase4_iterations": 0,
            "phase4_elapsed_sec": 0.0,
            "phase5_elapsed_sec": 0.0,
            "optimization_elapsed_sec": time.time() - optimization_start,
            "n_iterations": int(n_iterations),
            "k": int(n_iterations),
            "mei": np.asarray([], dtype=float),
            "labels": None,
            "optimization_diagnostics": pd.DataFrame(gamma_diagnostics_rows),
        }

    valid_results = [gamma_results[idx] for idx in valid_indices]
    exact_hit_gamma_flags = np.asarray([int(result.get("hit_count", 0)) > 0 for result in valid_results], dtype=bool)
    exact_hit_priority_enabled = not admission_mode.startswith("raw_")
    if exact_hit_priority_enabled and np.any(exact_hit_gamma_flags):
        valid_indices = [valid_indices[idx] for idx in np.where(exact_hit_gamma_flags)[0]]
        valid_results = [gamma_results[idx] for idx in valid_indices]

    for idx, result in enumerate(gamma_results):
        gamma_diagnostics_rows.append(
            build_gamma_diagnostic_row(
                result,
                phase="admission",
                gamma_batch=result.get("_gamma_batch"),
                admission_selected=idx in set(valid_indices),
                admission_mode=admission_mode,
            )
        )

    discarded_refs = [
        gamma_results[idx].get("matrix_ref")
        for idx in range(len(gamma_results))
        if idx not in valid_indices and gamma_results[idx].get("matrix_ref") is not None
    ]
    release_cluster_matrix_refs(discarded_refs)

    gamma_sequence = np.asarray([result["gamma"] for result in valid_results], dtype=float)
    ic_scores = np.asarray([result["ic"] for result in valid_results], dtype=float)
    clustering_refs = [result["matrix_ref"] for result in valid_results]
    final_cluster_count_vectors = [
        np.asarray(result.get("final_cluster_counts", []), dtype=int)
        for result in valid_results
    ]
    effective_cluster_medians = np.asarray([result.get("median_effective_clusters", np.nan) for result in valid_results], dtype=float)
    final_cluster_medians = np.asarray([result.get("final_cluster_median", np.nan) for result in valid_results], dtype=float)
    raw_cluster_medians = np.asarray([result.get("raw_cluster_median", np.nan) for result in valid_results], dtype=float)
    preferred_hit_trials = [result.get("hit_trials", []) for result in valid_results]
    current_finalizable_flags = preferred_trial_flags(preferred_hit_trials, size=len(valid_results))
    initial_rank_order = order_gamma_candidate_indices(
        valid_results,
        target_clusters,
        exact_support_flags=exact_hit_gamma_flags,
        prefer_right_exact_hits=exact_hit_priority_enabled,
        finalizable_flags=current_finalizable_flags,
    )

    best_index = int(initial_rank_order[0]) if initial_rank_order else 0
    best_gamma = float(gamma_sequence[best_index])
    best_ref = clustering_refs[best_index]
    best_preferred_trials = preferred_hit_trials[best_index]
    best_final_cluster_counts = final_cluster_count_vectors[best_index]
    k = int(n_iterations)
    phase4_iterations = 0
    phase4_elapsed_sec = 0.0

    if ic_scores[best_index] != 1.0 and len(gamma_sequence) > 1:
        if should_skip_phase4_refinement(len(gamma_sequence), float(ic_scores[best_index]), exact_hit_gamma_count):
            release_cluster_matrix_refs([ref for idx, ref in enumerate(clustering_refs) if idx != best_index])
        else:
            iterative_start = time.time()
            current_refs = clustering_refs
            current_gammas = gamma_sequence
            current_ic = ic_scores
            current_preferred_trials = preferred_hit_trials
            current_final_cluster_counts = final_cluster_count_vectors
            ic_history = np.tile(current_ic[:, None], (1, 10))
            current_exact_support_flags = exact_hit_gamma_flags.copy()
            current_finalizable_flags = preferred_trial_flags(current_preferred_trials, size=len(current_gammas))
            converged = False
            delta_n = 2
            phase4_limit = phase4_iteration_cap_for_mode(admission_mode)
            prefer_exact_hits = exact_hit_priority_enabled and np.any(exact_hit_gamma_flags)

            while k < int(max_iterations) and phase4_iterations < phase4_limit:
                k += delta_n
                phase4_iterations += 1
                new_results = []
                for gamma_idx, gamma_val in enumerate(current_gammas):
                    current_matrix = load_cluster_matrix(current_refs[gamma_idx])
                    new_matrix = np.zeros_like(current_matrix)
                    for trial_idx in range(current_matrix.shape[0]):
                        if cluster_seed is not None:
                            np.random.seed(int((cluster_seed + k * 100 + gamma_idx + trial_idx + 1) % (2**31 - 1)) or 1)
                        init_membership = current_matrix[np.random.randint(current_matrix.shape[0])].tolist()
                        new_matrix[trial_idx] = leiden_clustering(
                            graph=graph,
                            resolution=float(gamma_val),
                            objective_function=objective_function,
                            n_iterations=delta_n,
                            beta=beta,
                            initial_membership=init_membership,
                        )
                    trial_summary = summarize_trial_matrix(
                        new_matrix,
                        snn_graph=snn_graph,
                        min_cluster_size=min_cluster_size,
                        target_clusters=target_clusters,
                    )
                    extracted = extract_clustering_array(new_matrix)
                    new_results.append(
                        {
                            "matrix_ref": store_cluster_matrix(new_matrix, runtime_context=runtime_context, prefix=f"k{target_clusters}_iter{k}_g{gamma_idx}"),
                            "ic": calculate_ic_from_extracted(extracted, n_workers=1),
                            "exact_hit_count": int(trial_summary.hit_count),
                            "preferred_trials": trial_summary.final_hit_trials.tolist(),
                            "final_cluster_counts": trial_summary.final_cluster_counts.copy(),
                        }
                    )
                new_refs = [item["matrix_ref"] for item in new_results]
                new_ic = np.asarray([item["ic"] for item in new_results], dtype=float)
                new_exact_hit_counts = np.asarray([item["exact_hit_count"] for item in new_results], dtype=int)
                phase4_rows = [
                    build_gamma_diagnostic_row(
                        {
                            "gamma": float(current_gammas[idx]),
                            "ic": float(new_results[idx]["ic"]),
                            "final_cluster_median": float(target_clusters if new_exact_hit_counts[idx] > 0 else np.nan),
                            "hit_count": int(new_exact_hit_counts[idx]),
                        },
                        phase="phase4",
                        phase4_iteration=int(phase4_iterations),
                        preferred_trial_count=len(new_results[idx]["preferred_trials"]),
                    )
                    for idx in range(len(new_results))
                ]
                candidate_refs = new_refs
                candidate_ic = new_ic
                candidate_gammas = current_gammas
                candidate_preferred_trials = [item["preferred_trials"] for item in new_results]
                candidate_final_cluster_counts = [item["final_cluster_counts"] for item in new_results]
                candidate_finalizable_flags = preferred_trial_flags(candidate_preferred_trials, size=len(new_results))
                candidate_history = np.concatenate([ic_history[:, 1:], new_ic[:, None]], axis=1)
                candidate_exact_hit_flags = new_exact_hit_counts > 0
                candidate_exact_support_flags = candidate_exact_hit_flags | current_exact_support_flags
                candidate_origin_indices = np.arange(len(new_results), dtype=int)
                candidate_results = [
                    {
                        "gamma": float(candidate_gammas[idx]),
                        "ic": float(candidate_ic[idx]),
                        "hit_count": int(new_exact_hit_counts[idx]),
                        "final_cluster_median": float(target_clusters if new_exact_hit_counts[idx] > 0 else np.nan),
                        "relaxed_valid": bool(new_exact_hit_counts[idx] > 0),
                    }
                    for idx in range(len(new_results))
                ]

                if prefer_exact_hits:
                    if not np.any(candidate_finalizable_flags) and np.any(current_finalizable_flags):
                        for row_idx in range(len(phase4_rows)):
                            phase4_rows[row_idx]["phase4_keep"] = False
                            phase4_rows[row_idx]["phase4_prune_reason"] = "lost_final_target_support"
                        gamma_diagnostics_rows.extend(phase4_rows)
                        release_cluster_matrix_refs(new_refs)
                        current_results = [
                            {
                                "gamma": float(current_gammas[idx]),
                                "ic": float(current_ic[idx]),
                                "hit_count": int(current_finalizable_flags[idx]),
                                "final_cluster_median": float(target_clusters if current_finalizable_flags[idx] else np.nan),
                                "relaxed_valid": bool(current_finalizable_flags[idx]),
                            }
                            for idx in range(len(current_gammas))
                        ]
                        current_rank_order = order_gamma_candidate_indices(
                            current_results,
                            target_clusters,
                            exact_support_flags=current_exact_support_flags,
                            prefer_right_exact_hits=prefer_exact_hits,
                            finalizable_flags=current_finalizable_flags,
                        )
                        best_index = int(current_rank_order[0]) if current_rank_order else 0
                        best_gamma = float(current_gammas[best_index])
                        best_ref = current_refs[best_index]
                        best_preferred_trials = current_preferred_trials[best_index]
                        best_final_cluster_counts = current_final_cluster_counts[best_index]
                        release_cluster_matrix_refs([ref for idx, ref in enumerate(current_refs) if idx != best_index])
                        converged = True
                        break
                    if not np.any(candidate_exact_support_flags):
                        for row_idx in range(len(phase4_rows)):
                            phase4_rows[row_idx]["phase4_keep"] = False
                            phase4_rows[row_idx]["phase4_prune_reason"] = "missing_exact_hit_support"
                        gamma_diagnostics_rows.extend(phase4_rows)
                        release_cluster_matrix_refs(new_refs)
                        current_results = [
                            {
                                "gamma": float(current_gammas[idx]),
                                "ic": float(current_ic[idx]),
                                "hit_count": int(current_exact_support_flags[idx]),
                                "final_cluster_median": float(target_clusters if current_exact_support_flags[idx] else np.nan),
                                "relaxed_valid": bool(current_exact_support_flags[idx]),
                            }
                            for idx in range(len(current_gammas))
                        ]
                        current_rank_order = order_gamma_candidate_indices(
                            current_results,
                            target_clusters,
                            exact_support_flags=current_exact_support_flags,
                            prefer_right_exact_hits=prefer_exact_hits,
                            finalizable_flags=current_finalizable_flags,
                        )
                        best_index = int(current_rank_order[0]) if current_rank_order else 0
                        best_gamma = float(current_gammas[best_index])
                        best_ref = current_refs[best_index]
                        best_preferred_trials = current_preferred_trials[best_index]
                        best_final_cluster_counts = current_final_cluster_counts[best_index]
                        release_cluster_matrix_refs([ref for idx, ref in enumerate(current_refs) if idx != best_index])
                        converged = True
                        break
                    if np.any(candidate_finalizable_flags) and not np.all(candidate_finalizable_flags):
                        release_cluster_matrix_refs([ref for idx, ref in enumerate(candidate_refs) if not candidate_finalizable_flags[idx]])
                        for row_idx, keep_flag in enumerate(candidate_finalizable_flags.tolist()):
                            if not keep_flag:
                                phase4_rows[row_idx]["phase4_keep"] = False
                                phase4_rows[row_idx]["phase4_prune_reason"] = "lost_final_target_support"
                        keep_idx = np.where(candidate_finalizable_flags)[0]
                        candidate_exact_hit_flags = candidate_exact_hit_flags[keep_idx]
                        candidate_refs = [candidate_refs[idx] for idx in keep_idx]
                        candidate_ic = candidate_ic[keep_idx]
                        candidate_gammas = candidate_gammas[keep_idx]
                        candidate_history = candidate_history[keep_idx]
                        candidate_preferred_trials = [candidate_preferred_trials[idx] for idx in keep_idx]
                        candidate_final_cluster_counts = [candidate_final_cluster_counts[idx] for idx in keep_idx]
                        candidate_results = [candidate_results[idx] for idx in keep_idx]
                        candidate_finalizable_flags = candidate_finalizable_flags[keep_idx]
                        candidate_exact_support_flags = candidate_exact_support_flags[keep_idx]
                        candidate_origin_indices = candidate_origin_indices[keep_idx]
                    elif not np.all(candidate_exact_support_flags):
                        release_cluster_matrix_refs([ref for idx, ref in enumerate(candidate_refs) if not candidate_exact_support_flags[idx]])
                        for row_idx, keep_flag in enumerate(candidate_exact_support_flags.tolist()):
                            if not keep_flag:
                                phase4_rows[row_idx]["phase4_keep"] = False
                                phase4_rows[row_idx]["phase4_prune_reason"] = "missing_exact_hit_support"
                        keep_idx = np.where(candidate_exact_support_flags)[0]
                        candidate_finalizable_flags = candidate_finalizable_flags[keep_idx]
                        candidate_refs = [candidate_refs[idx] for idx in keep_idx]
                        candidate_ic = candidate_ic[keep_idx]
                        candidate_gammas = candidate_gammas[keep_idx]
                        candidate_history = candidate_history[keep_idx]
                        candidate_preferred_trials = [candidate_preferred_trials[idx] for idx in keep_idx]
                        candidate_final_cluster_counts = [candidate_final_cluster_counts[idx] for idx in keep_idx]
                        candidate_results = [candidate_results[idx] for idx in keep_idx]
                        candidate_exact_support_flags = candidate_exact_support_flags[keep_idx]
                        candidate_origin_indices = candidate_origin_indices[keep_idx]

                release_cluster_matrix_refs(current_refs)
                stable_indices = np.asarray([len(np.unique(row)) == 1 for row in candidate_history], dtype=bool)
                perfect_indices = np.where(candidate_ic == 1.0)[0]

                if perfect_indices.size:
                    perfect_rank_order = order_gamma_candidate_indices(
                        [candidate_results[idx] for idx in perfect_indices.tolist()],
                        target_clusters,
                        exact_support_flags=candidate_exact_support_flags[perfect_indices],
                        prefer_right_exact_hits=prefer_exact_hits,
                        finalizable_flags=candidate_finalizable_flags[perfect_indices],
                    )
                    best_index = int(perfect_indices[int(perfect_rank_order[0])]) if perfect_rank_order else int(perfect_indices[0])
                    origin_best = int(candidate_origin_indices[best_index])
                    best_gamma = float(candidate_gammas[best_index])
                    best_ref = candidate_refs[best_index]
                    best_preferred_trials = candidate_preferred_trials[best_index]
                    best_final_cluster_counts = candidate_final_cluster_counts[best_index]
                    for row_idx in candidate_origin_indices.tolist():
                        phase4_rows[int(row_idx)]["phase4_keep"] = int(row_idx) == origin_best
                        phase4_rows[int(row_idx)]["phase4_prune_reason"] = None if int(row_idx) == origin_best else "perfect_ic_superseded"
                    release_cluster_matrix_refs([ref for idx, ref in enumerate(candidate_refs) if idx != best_index])
                    gamma_diagnostics_rows.extend(phase4_rows)
                    converged = True
                    break
                if np.all(stable_indices):
                    stable_rank_order = order_gamma_candidate_indices(
                        candidate_results,
                        target_clusters,
                        exact_support_flags=candidate_exact_support_flags,
                        prefer_right_exact_hits=prefer_exact_hits,
                        finalizable_flags=candidate_finalizable_flags,
                    )
                    best_index = int(stable_rank_order[0]) if stable_rank_order else int(np.argmin(candidate_ic))
                    origin_best = int(candidate_origin_indices[best_index])
                    best_gamma = float(candidate_gammas[best_index])
                    best_ref = candidate_refs[best_index]
                    best_preferred_trials = candidate_preferred_trials[best_index]
                    best_final_cluster_counts = candidate_final_cluster_counts[best_index]
                    for row_idx in candidate_origin_indices.tolist():
                        phase4_rows[int(row_idx)]["phase4_keep"] = int(row_idx) == origin_best
                        phase4_rows[int(row_idx)]["phase4_prune_reason"] = None if int(row_idx) == origin_best else "stable_ic_superseded"
                    release_cluster_matrix_refs([ref for idx, ref in enumerate(candidate_refs) if idx != best_index])
                    gamma_diagnostics_rows.extend(phase4_rows)
                    converged = True
                    break

                keep_indices = (candidate_ic <= np.quantile(candidate_ic, 0.5)) | stable_indices
                keep_indices[int(np.argmin(candidate_ic))] = True
                keep_pool = np.where(keep_indices)[0].tolist()
                keep_limit = 4 if phase4_iterations == 1 else 2
                ranked_pool = order_gamma_candidate_indices(
                    candidate_results,
                    target_clusters,
                    exact_support_flags=candidate_exact_support_flags,
                    prefer_right_exact_hits=prefer_exact_hits,
                    finalizable_flags=candidate_finalizable_flags,
                )
                if len(keep_pool) > keep_limit:
                    best_global_idx = int(ranked_pool[0]) if ranked_pool else int(np.argmin(candidate_ic))
                    stable_pool = np.where(stable_indices)[0]
                    stable_best_idx = int(
                        stable_pool[
                            order_gamma_candidate_indices(
                                [candidate_results[idx] for idx in stable_pool.tolist()],
                                target_clusters,
                                exact_support_flags=candidate_exact_support_flags[stable_pool],
                                prefer_right_exact_hits=prefer_exact_hits,
                                finalizable_flags=candidate_finalizable_flags[stable_pool],
                            )[0]
                        ]
                    ) if stable_pool.size else None
                    ordered_pool = [idx for idx in ranked_pool if idx in keep_pool]
                    merged_pool = [best_global_idx]
                    if stable_best_idx is not None:
                        merged_pool.append(stable_best_idx)
                    merged_pool.extend(ordered_pool)
                    keep_pool = list(dict.fromkeys(merged_pool))[:keep_limit]
                else:
                    keep_pool = list(dict.fromkeys([*ranked_pool[:keep_limit], *keep_pool]))[:keep_limit]
                keep_mask = np.asarray([idx in keep_pool for idx in range(len(candidate_ic))], dtype=bool)
                for pos, origin_idx in enumerate(candidate_origin_indices.tolist()):
                    keep_flag = bool(keep_mask[pos])
                    phase4_rows[int(origin_idx)]["phase4_keep"] = keep_flag
                    if not keep_flag and phase4_rows[int(origin_idx)]["phase4_prune_reason"] is None:
                        phase4_rows[int(origin_idx)]["phase4_prune_reason"] = "phase4_pruned_by_ic"
                if np.sum(keep_mask) == 1:
                    best_index = int(np.where(keep_mask)[0][0])
                    origin_best = int(candidate_origin_indices[best_index])
                    best_gamma = float(candidate_gammas[best_index])
                    best_ref = candidate_refs[best_index]
                    best_preferred_trials = candidate_preferred_trials[best_index]
                    best_final_cluster_counts = candidate_final_cluster_counts[best_index]
                    phase4_rows[origin_best]["phase4_keep"] = True
                    phase4_rows[origin_best]["phase4_prune_reason"] = None
                    release_cluster_matrix_refs([ref for idx, ref in enumerate(candidate_refs) if idx != best_index])
                    gamma_diagnostics_rows.extend(phase4_rows)
                    converged = True
                    break

                release_cluster_matrix_refs([ref for idx, ref in enumerate(candidate_refs) if not keep_mask[idx]])
                gamma_diagnostics_rows.extend(phase4_rows)
                current_gammas = candidate_gammas[keep_mask]
                current_refs = [candidate_refs[idx] for idx in np.where(keep_mask)[0]]
                current_ic = candidate_ic[keep_mask]
                current_preferred_trials = [candidate_preferred_trials[idx] for idx in np.where(keep_mask)[0]]
                current_final_cluster_counts = [candidate_final_cluster_counts[idx] for idx in np.where(keep_mask)[0]]
                ic_history = candidate_history[keep_mask]
                current_exact_support_flags = candidate_exact_support_flags[keep_mask]
                current_finalizable_flags = candidate_finalizable_flags[keep_mask]

            if not converged:
                current_results = [
                    {
                        "gamma": float(current_gammas[idx]),
                        "ic": float(current_ic[idx]),
                        "hit_count": int(current_exact_support_flags[idx]),
                        "final_cluster_median": float(target_clusters if current_exact_support_flags[idx] else np.nan),
                        "relaxed_valid": bool(current_exact_support_flags[idx]),
                    }
                    for idx in range(len(current_gammas))
                ]
                current_rank_order = order_gamma_candidate_indices(
                    current_results,
                    target_clusters,
                    exact_support_flags=current_exact_support_flags,
                    prefer_right_exact_hits=prefer_exact_hits,
                    finalizable_flags=current_finalizable_flags,
                )
                best_index = int(current_rank_order[0]) if current_rank_order else int(np.argmin(current_ic))
                best_gamma = float(current_gammas[best_index])
                best_ref = current_refs[best_index]
                best_preferred_trials = current_preferred_trials[best_index]
                best_final_cluster_counts = current_final_cluster_counts[best_index]
                release_cluster_matrix_refs([ref for idx, ref in enumerate(current_refs) if idx != best_index])

            phase4_elapsed_sec = time.time() - iterative_start
    else:
        release_cluster_matrix_refs([ref for idx, ref in enumerate(clustering_refs) if idx != best_index])

    best_gamma_diag_index = int(np.argmin(np.abs(gamma_sequence - best_gamma)))
    finalize_worker_budget = phase1_nested_workers if precomputed_phase1 is None else max(1, int(n_workers))
    finalized = finalize_selected_clustering(
        matrix_ref=best_ref,
        gamma=best_gamma,
        effective_cluster_median=float(effective_cluster_medians[best_gamma_diag_index]),
        raw_cluster_median=float(raw_cluster_medians[best_gamma_diag_index]),
        final_cluster_median=float(final_cluster_medians[best_gamma_diag_index]),
        admission_mode=admission_mode,
        cluster_seed=cluster_seed,
        n_bootstrap=n_bootstrap,
        n_workers=finalize_worker_budget,
        snn_graph=snn_graph,
        target_clusters=target_clusters,
        preferred_trial_indices=best_preferred_trials,
        final_cluster_counts=best_final_cluster_counts,
        min_cluster_size=min_cluster_size,
        verbose=verbose,
        worker_id=worker_id,
        runtime_context=runtime_context,
    )
    gamma_diagnostics_rows.append(
        {
            "phase": "finalize",
            "gamma_batch": None,
            "phase4_iteration": int(phase4_iterations),
            "gamma": float(best_gamma),
            "ic": float(finalized.get("ic_median", np.nan)),
            "effective_cluster_median": float(effective_cluster_medians[best_gamma_diag_index]),
            "raw_cluster_median": float(raw_cluster_medians[best_gamma_diag_index]),
            "final_cluster_median": float(final_cluster_medians[best_gamma_diag_index]),
            "hit_count": int(exact_hit_gamma_count),
            "raw_hit_count": np.nan,
            "strict_valid": np.nan,
            "relaxed_valid": np.nan,
            "raw_strict_valid": np.nan,
            "raw_relaxed_valid": np.nan,
            "raw_guard_soft": np.nan,
            "raw_guard_hard": np.nan,
            "admission_selected": True,
            "admission_mode": admission_mode,
            "phase4_keep": True,
            "phase4_prune_reason": None,
            "selected_best_gamma": True,
            "preferred_trial_count": int(finalized.get("preferred_trial_count", 0)),
            "best_labels_raw_cluster_count": int(finalized.get("best_labels_raw_cluster_count", -1)),
            "best_labels_final_cluster_count": int(finalized.get("best_labels_final_cluster_count", -1)),
        }
    )
    if int(finalized.get("best_labels_final_cluster_count", -1)) != int(target_clusters):
        finalized.update(
            {
                "success": False,
                "failure_reason": "final_cluster_mismatch",
                "phase1_primary_gamma_count": phase1_primary_gamma_count,
                "phase1_secondary_gamma_count": phase1_secondary_gamma_count,
                "phase1_total_gamma_count": phase1_total_gamma_count,
                "phase1_elapsed_sec": phase1_elapsed_sec,
                "phase1_leiden_runs": phase1_leiden_runs,
                "secondary_phase1_used": bool(secondary_phase1_used and phase1_secondary_gamma_count > 0),
                "exact_hit_gamma_count": exact_hit_gamma_count,
                "phase4_iterations": int(phase4_iterations),
                "phase4_elapsed_sec": float(phase4_elapsed_sec),
                "optimization_elapsed_sec": time.time() - optimization_start,
                "n_iterations": int(k),
                "k": int(k),
                "optimization_diagnostics": pd.DataFrame(gamma_diagnostics_rows),
            }
        )
        return finalized
    finalized.update(
        {
            "success": True,
            "phase1_primary_gamma_count": phase1_primary_gamma_count,
            "phase1_secondary_gamma_count": phase1_secondary_gamma_count,
            "phase1_total_gamma_count": phase1_total_gamma_count,
            "phase1_elapsed_sec": phase1_elapsed_sec,
            "phase1_leiden_runs": phase1_leiden_runs,
            "secondary_phase1_used": bool(secondary_phase1_used and phase1_secondary_gamma_count > 0),
            "exact_hit_gamma_count": exact_hit_gamma_count,
            "phase4_iterations": int(phase4_iterations),
            "phase4_elapsed_sec": float(phase4_elapsed_sec),
            "optimization_elapsed_sec": time.time() - optimization_start,
            "n_iterations": int(k),
            "k": int(k),
            "optimization_diagnostics": pd.DataFrame(gamma_diagnostics_rows),
        }
    )
    return finalized

def evaluate_fixed_resolution(
    graph,
    resolution: float,
    objective_function: str,
    n_trials: int,
    n_bootstrap: int,
    seed: int | None,
    beta: float,
    n_iterations: int,
    n_workers: int,
    snn_graph=None,
    min_cluster_size: int = 1,
    verbose: bool = False,
    worker_id: str = "RESOLUTION",
    runtime_context=None,
    in_parallel_context: bool = False,
) -> dict[str, Any]:
    """Evaluate one fixed resolution across repeated trials and bootstrap the selected representative clustering."""
    cluster_seed = None if seed is None else int((seed + int(abs(float(resolution)) * 100000)) % (2**31 - 1) or 1)
    heartbeat = create_heartbeat_logger(verbose=verbose, context=worker_id)
    if verbose:
        logger.info("%s: Fixed-resolution parameters:", worker_id)
        logger.info("%s:   Resolution: %.6g", worker_id, float(resolution))
        logger.info("%s:   Trials: %s", worker_id, n_trials)
        logger.info("%s:   Bootstrap iterations: %s", worker_id, n_bootstrap)
        logger.info("%s:   Beta: %s", worker_id, beta)
        logger.info("%s:   Leiden iterations: %s", worker_id, n_iterations)
        if min_cluster_size > 1:
            logger.info(
                "%s:   Counting uses effective clusters (size >= %s); final merge applied only on best_labels",
                worker_id,
                min_cluster_size,
            )

    n_vertices = graph.vcount()
    estimated_phase1_bytes = estimate_trial_matrix_bytes(n_vertices, n_trials, 1)
    trial_workers = max(1, int(n_workers))
    trial_workers = min(trial_workers, max(1, int(n_trials)))
    trial_workers = cap_workers_by_memory(
        trial_workers,
        estimate_trial_matrix_bytes(n_vertices, 1, 1),
        runtime_context,
    )
    phase1_log_every = max(1, int(math.floor(max(1, int(n_trials)) / 5)))

    def should_log_trial_step(step_idx: int) -> bool:
        """Decide whether the current fixed-resolution trial should emit a progress log."""
        return step_idx == 1 or step_idx == int(n_trials) or (step_idx % phase1_log_every) == 0

    if verbose:
        logger.info("%s: Phase 1 - Evaluating fixed resolution with %s trials", worker_id, n_trials)
        if in_parallel_context:
            logger.info("%s: Running in parallel context with worker budget %s", worker_id, n_workers)
        logger.info("%s: Trial worker budget: %s", worker_id, trial_workers)
    phase1_start = time.time()

    def run_single_trial(trial_idx: int) -> np.ndarray:
        """Run one Leiden trial for the requested fixed resolution."""
        trial_idx = int(trial_idx)
        trial_seed = None if cluster_seed is None else int((cluster_seed + trial_idx + 1) % (2**31 - 1) or 1)
        labels = leiden_clustering(
            graph=graph,
            resolution=float(resolution),
            objective_function=objective_function,
            n_iterations=n_iterations,
            beta=beta,
            seed=trial_seed,
        )
        heartbeat(lambda: f"phase1 running - fixed resolution {resolution:.6g} - trial {trial_idx + 1}/{n_trials}")
        if verbose and should_log_trial_step(trial_idx + 1):
            logger.info(
                "%s: Phase 1 progress trial %s/%s (resolution = %.6g)",
                worker_id,
                trial_idx + 1,
                n_trials,
                float(resolution),
            )
        return np.asarray(labels, dtype=np.int32)

    cluster_matrix = np.asarray(
        parallel_map_threads(range(int(n_trials)), run_single_trial, max_workers=trial_workers),
        dtype=np.int32,
    )

    trial_summary = summarize_trial_matrix(
        cluster_matrix,
        snn_graph=snn_graph,
        min_cluster_size=min_cluster_size,
        target_clusters=None,
    )
    raw_clusters_vec = trial_summary.raw_cluster_counts
    effective_clusters_vec = trial_summary.effective_cluster_counts
    final_clusters_vec = trial_summary.final_cluster_counts

    extracted = extract_clustering_array(cluster_matrix)
    phase1_ic = calculate_ic_from_extracted(extracted, n_workers=1)
    matrix_ref = store_cluster_matrix(cluster_matrix, runtime_context=runtime_context, prefix=f"fixed_resolution_{abs(hash(float(resolution))) % 100000}")
    if verbose:
        phase1_time = time.time() - phase1_start
        logger.info("%s: Phase 1 completed in %.3f seconds", worker_id, phase1_time)
        logger.info(
            "%s: Phase 1 diagnostics - resolution = %.6g - median_effective = %.6g - median_raw = %.6g - median_final = %.6g - IC (all trials) = %.4f",
            worker_id,
            float(resolution),
            float(np.median(effective_clusters_vec)),
            float(np.median(raw_clusters_vec)),
            float(np.median(final_clusters_vec)),
            float(phase1_ic),
        )
    finalized = finalize_selected_clustering(
        matrix_ref=matrix_ref,
        gamma=float(resolution),
        effective_cluster_median=float(np.median(effective_clusters_vec)),
        raw_cluster_median=float(np.median(raw_clusters_vec)),
        final_cluster_median=float(np.median(final_clusters_vec)),
        admission_mode="manual_resolution",
        cluster_seed=cluster_seed,
        n_bootstrap=n_bootstrap,
        n_workers=trial_workers,
        snn_graph=snn_graph,
        target_clusters=None,
        preferred_trial_indices=None,
        final_cluster_counts=final_clusters_vec,
        min_cluster_size=min_cluster_size,
        verbose=verbose,
        worker_id=worker_id,
        runtime_context=runtime_context,
    )
    finalized["phase1_ic"] = phase1_ic
    finalized["n_iterations"] = int(n_iterations)
    finalized["k"] = int(n_iterations)
    return finalized
