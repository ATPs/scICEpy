"""Gamma evaluation and clustering finalization helpers for optimization."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cluster_utils import (
    merge_small_clusters_to_neighbors,
    passes_raw_cluster_guard,
    summarize_cluster_labels,
)
from .leiden_wrapper import leiden_clustering
from .metrics import calculate_ic_from_extracted, calculate_mei_from_array, extract_clustering_array, get_best_clustering
from .runtime import (
    cap_workers_by_memory,
    create_heartbeat_logger,
    estimate_trial_matrix_bytes,
    load_cluster_matrix,
    logger,
    parallel_map_threads,
    release_cluster_matrix,
    store_cluster_matrix,
)


@dataclass
class TrialMatrixSummary:
    """Summarize raw, effective, and final cluster-count diagnostics for one trial matrix."""

    raw_cluster_counts: np.ndarray
    effective_cluster_counts: np.ndarray
    final_cluster_counts: np.ndarray
    final_hit_trials: np.ndarray
    raw_hit_trials: np.ndarray

    @property
    def raw_cluster_median(self) -> float:
        """Return the median raw cluster count across all trials."""
        return float(np.median(self.raw_cluster_counts)) if self.raw_cluster_counts.size else np.nan

    @property
    def effective_cluster_median(self) -> float:
        """Return the median effective cluster count across all trials."""
        return float(np.median(self.effective_cluster_counts)) if self.effective_cluster_counts.size else np.nan

    @property
    def final_cluster_median(self) -> float:
        """Return the median final cluster count across all trials."""
        return float(np.median(self.final_cluster_counts)) if self.final_cluster_counts.size else np.nan

    @property
    def hit_count(self) -> int:
        """Return how many trials hit the requested final cluster count exactly."""
        return int(self.final_hit_trials.size)

    @property
    def raw_hit_count(self) -> int:
        """Return how many trials hit the requested raw cluster count exactly."""
        return int(self.raw_hit_trials.size)


def summarize_trial_cluster_counts(cluster_labels: np.ndarray, min_cluster_size: int) -> tuple[int, int]:
    """Return the raw and effective cluster counts for a single trial label vector."""
    counts = summarize_cluster_labels(cluster_labels, min_cluster_size=min_cluster_size)
    return int(counts["raw_cluster_count"]), int(counts["effective_cluster_count"])


def summarize_trial_matrix(
    cluster_matrix: np.ndarray,
    snn_graph,
    min_cluster_size: int,
    target_clusters: int | None = None,
) -> TrialMatrixSummary:
    """Aggregate per-trial cluster counts and exact-hit indices for one trial matrix."""
    cluster_matrix = np.asarray(cluster_matrix, dtype=np.int32)
    min_cluster_size = max(1, int(min_cluster_size))
    n_trials = int(cluster_matrix.shape[0]) if cluster_matrix.ndim >= 2 else 0
    raw_cluster_counts = np.zeros(n_trials, dtype=int)
    effective_cluster_counts = np.zeros(n_trials, dtype=int)
    final_cluster_counts = np.zeros(n_trials, dtype=int)

    for trial_idx in range(n_trials):
        raw_count, effective_count = summarize_trial_cluster_counts(
            cluster_matrix[trial_idx],
            min_cluster_size=min_cluster_size,
        )
        raw_cluster_counts[trial_idx] = int(raw_count)
        effective_cluster_counts[trial_idx] = int(effective_count)
        if min_cluster_size > 1:
            merged_labels = merge_small_clusters_to_neighbors(
                cluster_matrix[trial_idx],
                snn_graph=snn_graph,
                min_cluster_size=min_cluster_size,
            )
            final_cluster_counts[trial_idx] = int(np.unique(merged_labels).size)
        else:
            final_cluster_counts[trial_idx] = int(raw_count)

    if target_clusters is None:
        final_hit_trials = np.asarray([], dtype=int)
        raw_hit_trials = np.asarray([], dtype=int)
    else:
        final_hit_trials = np.where(final_cluster_counts == int(target_clusters))[0].astype(int, copy=False)
        raw_hit_trials = np.where(raw_cluster_counts == int(target_clusters))[0].astype(int, copy=False)

    return TrialMatrixSummary(
        raw_cluster_counts=raw_cluster_counts,
        effective_cluster_counts=effective_cluster_counts,
        final_cluster_counts=final_cluster_counts,
        final_hit_trials=final_hit_trials,
        raw_hit_trials=raw_hit_trials,
    )


def build_gamma_diagnostic_row(
    result: dict[str, Any],
    phase: str,
    gamma_batch: str | None = None,
    phase4_iteration: int = 0,
    admission_selected: bool = False,
    admission_mode: str = "none",
    phase4_keep: bool | None = None,
    phase4_prune_reason: str | None = None,
    selected_best_gamma: bool = False,
    preferred_trial_count: int = 0,
) -> dict[str, Any]:
    """Convert one gamma-evaluation result into a flat diagnostics row."""
    return {
        "phase": phase,
        "gamma_batch": gamma_batch,
        "phase4_iteration": int(phase4_iteration),
        "gamma": float(result.get("gamma", np.nan)),
        "ic": float(result.get("ic", np.nan)),
        "effective_cluster_median": float(result.get("median_effective_clusters", np.nan)),
        "raw_cluster_median": float(result.get("raw_cluster_median", np.nan)),
        "final_cluster_median": float(result.get("final_cluster_median", np.nan)),
        "hit_count": int(result.get("hit_count", 0)),
        "raw_hit_count": int(result.get("raw_hit_count", 0)),
        "strict_valid": bool(result.get("strict_valid", False)),
        "relaxed_valid": bool(result.get("relaxed_valid", False)),
        "raw_strict_valid": bool(result.get("raw_strict_valid", False)),
        "raw_relaxed_valid": bool(result.get("raw_relaxed_valid", False)),
        "raw_guard_soft": bool(result.get("raw_guard_soft", True)),
        "raw_guard_hard": bool(result.get("raw_guard_hard", True)),
        "admission_selected": bool(admission_selected),
        "admission_mode": str(admission_mode),
        "phase4_keep": phase4_keep,
        "phase4_prune_reason": phase4_prune_reason,
        "selected_best_gamma": bool(selected_best_gamma),
        "preferred_trial_count": int(preferred_trial_count),
    }


def finalize_selected_clustering(
    matrix_ref: dict[str, Any],
    gamma: float,
    effective_cluster_median: float,
    raw_cluster_median: float,
    final_cluster_median: float,
    admission_mode: str,
    cluster_seed: int | None,
    n_bootstrap: int,
    n_workers: int,
    snn_graph,
    target_clusters: int | None = None,
    preferred_trial_indices: list[int] | None = None,
    final_cluster_counts: np.ndarray | None = None,
    min_cluster_size: int = 1,
    verbose: bool = False,
    worker_id: str = "OPTIMIZER",
    runtime_context=None,
) -> dict[str, Any]:
    """Finalize one selected gamma by bootstrapping IC, choosing representative labels, and attaching summary metrics."""
    best_clustering = load_cluster_matrix(matrix_ref)
    heartbeat = create_heartbeat_logger(verbose=verbose, context=worker_id)
    n_trials = best_clustering.shape[0]
    preferred_trial_indices = sorted(
        set(int(idx) for idx in (preferred_trial_indices or []) if 0 <= int(idx) < n_trials)
    )
    final_cluster_counts = None if final_cluster_counts is None else np.asarray(final_cluster_counts, dtype=int)
    if not preferred_trial_indices and target_clusters is not None:
        if final_cluster_counts is None or final_cluster_counts.size != n_trials:
            final_cluster_counts = summarize_trial_matrix(
                best_clustering,
                snn_graph=snn_graph,
                min_cluster_size=min_cluster_size,
                target_clusters=target_clusters,
            ).final_cluster_counts
        preferred_trial_indices = np.where(final_cluster_counts == int(target_clusters))[0].tolist()

    bootstrap_start = time.time()
    if verbose:
        logger.info("%s: Phase 5 - Bootstrap analysis with %s iterations", worker_id, n_bootstrap)
    bootstrap_workers = cap_workers_by_memory(
        max(1, int(n_workers)),
        estimate_trial_matrix_bytes(best_clustering.shape[1], n_trials, 1),
        runtime_context,
    )
    bootstrap_workers = min(bootstrap_workers, max(1, int(n_bootstrap)))
    bootstrap_log_every = max(1, int(math.floor(max(1, int(n_bootstrap)) / 5)))

    def should_log_bootstrap_step(step_idx: int) -> bool:
        """Decide whether the current bootstrap iteration should emit a progress log."""
        return step_idx == 1 or step_idx == int(n_bootstrap) or (step_idx % bootstrap_log_every) == 0

    def run_single_bootstrap(bootstrap_idx: int) -> float:
        """Evaluate one bootstrap resample and return its IC score."""
        bootstrap_idx = int(bootstrap_idx)
        bootstrap_seed = None if cluster_seed is None else int(cluster_seed + 10000 + bootstrap_idx + 1)
        rng = np.random.default_rng(bootstrap_seed)
        sample_indices = rng.integers(0, n_trials, size=n_trials, endpoint=False)
        bootstrap_matrix = best_clustering[sample_indices]
        ic_value = float(calculate_ic_from_extracted(extract_clustering_array(bootstrap_matrix), n_workers=1))
        heartbeat(lambda: f"phase5 running - bootstrap {bootstrap_idx + 1}/{n_bootstrap} - latest IC = {ic_value:.4f}")
        if verbose and should_log_bootstrap_step(bootstrap_idx + 1):
            logger.info(
                "%s: Phase 5 progress %s/%s - IC = %.4f",
                worker_id,
                bootstrap_idx + 1,
                n_bootstrap,
                ic_value,
            )
        return ic_value

    ic_bootstrap = np.asarray(
        parallel_map_threads(range(int(n_bootstrap)), run_single_bootstrap, max_workers=bootstrap_workers),
        dtype=float,
    )
    phase5_elapsed_sec = time.time() - bootstrap_start
    ic_median = float(np.median(ic_bootstrap))
    if verbose:
        logger.info("%s: Bootstrap analysis completed in %.3f seconds", worker_id, phase5_elapsed_sec)
        logger.info("%s: Bootstrap IC median: %.4f", worker_id, ic_median)
        if ic_bootstrap.size:
            logger.info(
                "%s: Bootstrap IC range: [%.4f, %.4f]",
                worker_id,
                float(np.min(ic_bootstrap)),
                float(np.max(ic_bootstrap)),
            )

    extracted_all = extract_clustering_array(best_clustering)
    selection_matrix = best_clustering if not preferred_trial_indices else best_clustering[preferred_trial_indices]
    if verbose and preferred_trial_indices:
        logger.info(
            "%s: Selecting representative best_labels from %s exact final-hit trial(s)%s",
            worker_id,
            len(preferred_trial_indices),
            "" if target_clusters is None else f" for target {int(target_clusters)}",
        )
    best_labels_raw = get_best_clustering(extract_clustering_array(selection_matrix))
    best_labels = (
        merge_small_clusters_to_neighbors(best_labels_raw, snn_graph=snn_graph, min_cluster_size=min_cluster_size)
        if min_cluster_size > 1
        else best_labels_raw
    )
    best_labels_raw_cluster_count = summarize_trial_cluster_counts(best_labels_raw, min_cluster_size=1)[0]
    best_labels_final_cluster_count = summarize_trial_cluster_counts(best_labels, min_cluster_size=1)[0]
    if (
        target_clusters is not None
        and preferred_trial_indices
        and int(best_labels_final_cluster_count) != int(target_clusters)
    ):
        fallback_labels_raw = np.asarray(best_clustering[int(preferred_trial_indices[0])], dtype=np.int32)
        fallback_labels = (
            merge_small_clusters_to_neighbors(fallback_labels_raw, snn_graph=snn_graph, min_cluster_size=min_cluster_size)
            if min_cluster_size > 1
            else fallback_labels_raw
        )
        fallback_final_cluster_count = summarize_trial_cluster_counts(fallback_labels, min_cluster_size=1)[0]
        if int(fallback_final_cluster_count) == int(target_clusters):
            best_labels_raw = fallback_labels_raw
            best_labels = fallback_labels
            best_labels_raw_cluster_count = summarize_trial_cluster_counts(best_labels_raw, min_cluster_size=1)[0]
            best_labels_final_cluster_count = fallback_final_cluster_count
    release_cluster_matrix(matrix_ref)
    if verbose and min_cluster_size > 1:
        logger.info(
            "%s: Final best_labels merged small clusters to satisfy min_cluster_size (value = %s; final clusters = %s)",
            worker_id,
            min_cluster_size,
            best_labels_final_cluster_count,
        )
    if verbose:
        logger.info(
            "%s: Selected diagnostics - gamma = %.6g - effective_median = %.6g - raw_median = %.6g - final_median = %.6g - admission_mode = %s - best_labels_raw_clusters = %s - best_labels_final_clusters = %s",
            worker_id,
            float(gamma),
            float(effective_cluster_median),
            float(raw_cluster_median),
            float(final_cluster_median),
            admission_mode,
            best_labels_raw_cluster_count,
            best_labels_final_cluster_count,
        )

    return {
        "gamma": float(gamma),
        "labels": extracted_all,
        "ic_median": ic_median,
        "ic_bootstrap": ic_bootstrap,
        "best_labels": best_labels,
        "effective_cluster_median": float(effective_cluster_median),
        "raw_cluster_median": float(raw_cluster_median),
        "final_cluster_median": float(final_cluster_median),
        "admission_mode": admission_mode,
        "best_labels_raw_cluster_count": best_labels_raw_cluster_count,
        "best_labels_final_cluster_count": best_labels_final_cluster_count,
        "preferred_trial_count": int(len(preferred_trial_indices)),
        "phase5_elapsed_sec": phase5_elapsed_sec,
        "mei": calculate_mei_from_array(extracted_all),
    }


def _evaluate_gamma(
    graph,
    gamma_val: float,
    target_clusters: int,
    objective_function: str,
    n_trials: int,
    beta: float,
    n_iterations: int,
    seed: int | None,
    snn_graph,
    min_cluster_size: int,
    worker_id: str,
    verbose: bool,
    runtime_context,
    gamma_idx: int | None = None,
    gamma_total: int | None = None,
    log_this_gamma: bool | None = None,
) -> dict[str, Any]:
    """Run Phase 1 Leiden trials for one gamma and return admission diagnostics plus cached trial-matrix state."""
    gamma_start = time.time()
    cluster_matrix = np.zeros((int(n_trials), graph.vcount()), dtype=np.int32)
    heartbeat = create_heartbeat_logger(verbose=verbose, context=worker_id)
    if log_this_gamma is None:
        log_this_gamma = bool(verbose)
    if log_this_gamma and gamma_idx is not None and gamma_total is not None:
        logger.info(
            "%s: Phase 1 progress gamma %s/%s started (gamma = %.6g)",
            worker_id,
            gamma_idx,
            gamma_total,
            float(gamma_val),
        )
    for trial_idx in range(int(n_trials)):
        trial_seed = None
        if seed is not None:
            gamma_component = int(abs(gamma_val) * 10000) % 100000
            trial_seed = int((seed + gamma_component + trial_idx + 1) % (2**31 - 1)) or 1
        cluster_matrix[trial_idx] = leiden_clustering(
            graph=graph,
            resolution=float(gamma_val),
            objective_function=objective_function,
            n_iterations=n_iterations,
            beta=beta,
            seed=trial_seed,
        )
        heartbeat(lambda: f"phase1 running - gamma {gamma_val:.6g} - trial {trial_idx + 1}/{n_trials}")

    trial_summary = summarize_trial_matrix(
        cluster_matrix,
        snn_graph=snn_graph,
        min_cluster_size=min_cluster_size,
        target_clusters=target_clusters,
    )
    raw_cluster_median = float(trial_summary.raw_cluster_median)
    final_cluster_median = float(trial_summary.final_cluster_median)
    median_effective_clusters = float(trial_summary.effective_cluster_median)
    hit_count = int(trial_summary.hit_count)
    raw_hit_count = int(trial_summary.raw_hit_count)
    median_gap = abs(final_cluster_median - float(target_clusters))
    raw_median_gap = abs(raw_cluster_median - float(target_clusters))
    within_median_window = median_gap <= 1
    strict_valid = int(final_cluster_median) == int(target_clusters)
    relaxed_valid = hit_count >= 1 and within_median_window
    raw_within_median_window = raw_median_gap <= 1
    raw_strict_valid = int(raw_cluster_median) == int(target_clusters)
    raw_relaxed_valid = raw_hit_count >= 1 and raw_within_median_window
    raw_guard_soft = bool(
        passes_raw_cluster_guard(raw_cluster_median, target_clusters, min_cluster_size=min_cluster_size, level="soft")
    )
    raw_guard_hard = bool(
        passes_raw_cluster_guard(raw_cluster_median, target_clusters, min_cluster_size=min_cluster_size, level="hard")
    )
    gamma_admitted = strict_valid or relaxed_valid or raw_strict_valid or raw_relaxed_valid
    exact_hit_supported = hit_count > 0

    result = {
        "valid": gamma_admitted or exact_hit_supported,
        "gamma": float(gamma_val),
        "mean_clusters": final_cluster_median,
        "median_effective_clusters": median_effective_clusters,
        "final_cluster_median": final_cluster_median,
        "raw_cluster_median": raw_cluster_median,
        "median_gap": median_gap,
        "raw_median_gap": raw_median_gap,
        "within_median_window": within_median_window,
        "strict_valid": strict_valid,
        "relaxed_valid": relaxed_valid,
        "raw_strict_valid": raw_strict_valid,
        "raw_relaxed_valid": raw_relaxed_valid,
        "hit_count": hit_count,
        "raw_hit_count": raw_hit_count,
        "raw_guard_soft": raw_guard_soft,
        "raw_guard_hard": raw_guard_hard,
        "effective_hit_count": hit_count,
        "hit_trials": trial_summary.final_hit_trials.tolist(),
        "final_cluster_counts": trial_summary.final_cluster_counts.copy(),
    }
    if not (gamma_admitted or exact_hit_supported):
        if log_this_gamma and gamma_idx is not None and gamma_total is not None:
            logger.info(
                "%s: Phase 1 progress gamma %s/%s completed in %.3f seconds - median_effective = %.6g - median_final = %.6g - median_raw = %.6g - median gap = %.3f - final hit trials = %s/%s - raw hit trials = %s/%s - strict_valid = %s - relaxed_valid = %s - raw_strict_valid = %s - raw_relaxed_valid = %s - raw_guard_soft = %s - raw_guard_hard = %s (target = %s; IC skipped)",
                worker_id,
                gamma_idx,
                gamma_total,
                time.time() - gamma_start,
                median_effective_clusters,
                final_cluster_median,
                raw_cluster_median,
                median_gap,
                hit_count,
                n_trials,
                raw_hit_count,
                n_trials,
                strict_valid,
                relaxed_valid,
                raw_strict_valid,
                raw_relaxed_valid,
                raw_guard_soft,
                raw_guard_hard,
                target_clusters,
            )
        return result

    extracted = extract_clustering_array(cluster_matrix)
    result["ic"] = calculate_ic_from_extracted(extracted, n_workers=1)
    result["matrix_ref"] = store_cluster_matrix(
        cluster_matrix,
        runtime_context=runtime_context,
        prefix=f"k{target_clusters}_g{abs(hash((target_clusters, gamma_val))) % 100000}",
    )
    if log_this_gamma and gamma_idx is not None and gamma_total is not None:
        ic_note = ""
        if exact_hit_supported and not gamma_admitted:
            ic_note = " - IC retained for exact-hit-supported candidate"
        logger.info(
            "%s: Phase 1 progress gamma %s/%s completed in %.3f seconds - median_effective = %.6g - median_final = %.6g - median_raw = %.6g - median gap = %.3f - final hit trials = %s/%s - raw hit trials = %s/%s - strict_valid = %s - relaxed_valid = %s - raw_strict_valid = %s - raw_relaxed_valid = %s - raw_guard_soft = %s - raw_guard_hard = %s - IC (all trials) = %.4f%s",
            worker_id,
            gamma_idx,
            gamma_total,
            time.time() - gamma_start,
            median_effective_clusters,
            final_cluster_median,
            raw_cluster_median,
            median_gap,
            hit_count,
            n_trials,
            raw_hit_count,
            n_trials,
            strict_valid,
            relaxed_valid,
            raw_strict_valid,
            raw_relaxed_valid,
            raw_guard_soft,
            raw_guard_hard,
            float(result["ic"]),
            ic_note,
        )
    return result
