"""Optimization helpers for scICEpy."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd

from .leiden_wrapper import leiden_clustering
from .metrics import calculate_ic_from_extracted, calculate_mei_from_array, extract_clustering_array, get_best_clustering
from .resolution_search import (
    build_gamma_sequence_for_range,
    passes_raw_cluster_guard,
    raw_cluster_guard_limits,
    reindex_cluster_labels,
    summarize_cluster_labels,
)
from .runtime import (
    cap_workers_by_memory,
    create_heartbeat_logger,
    estimate_trial_matrix_bytes,
    load_cluster_matrix,
    logger,
    parallel_map_threads,
    release_cluster_matrix,
    release_cluster_matrix_refs,
    store_cluster_matrix,
)


def select_gamma_admission(
    strict_flags: np.ndarray,
    relaxed_flags: np.ndarray,
    soft_guard_flags: np.ndarray,
    hard_guard_flags: np.ndarray,
    raw_strict_flags: np.ndarray | None = None,
    raw_relaxed_flags: np.ndarray | None = None,
) -> dict[str, Any]:
    strict_flags = np.asarray(strict_flags, dtype=bool)
    relaxed_flags = np.asarray(relaxed_flags, dtype=bool)
    soft_guard_flags = np.asarray(soft_guard_flags, dtype=bool)
    hard_guard_flags = np.asarray(hard_guard_flags, dtype=bool)
    raw_strict_flags = np.zeros_like(strict_flags) if raw_strict_flags is None else np.asarray(raw_strict_flags, dtype=bool)
    raw_relaxed_flags = np.zeros_like(strict_flags) if raw_relaxed_flags is None else np.asarray(raw_relaxed_flags, dtype=bool)

    candidate_sets = {
        "raw_strict_soft": np.where(raw_strict_flags & soft_guard_flags)[0],
        "strict_soft": np.where(strict_flags & soft_guard_flags)[0],
        "relaxed_soft": np.where(relaxed_flags & soft_guard_flags)[0],
        "strict_hard": np.where(strict_flags & hard_guard_flags)[0],
        "relaxed_hard": np.where(relaxed_flags & hard_guard_flags)[0],
        "relaxed_unguarded": np.where(relaxed_flags)[0],
        "raw_relaxed_soft": np.where(raw_relaxed_flags & soft_guard_flags)[0],
        "raw_relaxed_hard": np.where(raw_relaxed_flags & hard_guard_flags)[0],
        "raw_relaxed_unguarded": np.where(raw_relaxed_flags)[0],
    }
    for mode, indices in candidate_sets.items():
        if indices.size:
            return {"indices": indices.tolist(), "mode": mode}
    return {"indices": [], "mode": "none"}


def extract_raw_median_gap(result: dict[str, Any], target_clusters: int) -> float:
    if "raw_median_gap" not in result or not np.isfinite(result["raw_median_gap"]):
        return abs(float(result.get("mean_clusters", np.nan)) - float(target_clusters))
    return float(result["raw_median_gap"])


def refine_gamma_candidates_by_raw_gap(
    valid_indices: list[int],
    admission_mode: str,
    gamma_results: list[dict[str, Any]],
    target_clusters: int,
    min_cluster_size: int = 1,
) -> dict[str, Any]:
    if not valid_indices or int(min_cluster_size) <= 1:
        return {"indices": valid_indices, "mode": admission_mode, "raw_gaps": np.asarray([], dtype=float), "best_raw_gap": math.inf}

    if not str(admission_mode).startswith("raw_"):
        exact_hit_indices = [
            int(idx)
            for idx in valid_indices
            if int(gamma_results[idx].get("hit_count", 0)) > 0
        ]
        if len(exact_hit_indices) > 1:
            exact_hit_raw_gaps = np.asarray(
                [extract_raw_median_gap(gamma_results[idx], target_clusters) for idx in exact_hit_indices],
                dtype=float,
            )
            best_raw_gap = (
                float(np.min(exact_hit_raw_gaps[np.isfinite(exact_hit_raw_gaps)]))
                if np.any(np.isfinite(exact_hit_raw_gaps))
                else math.inf
            )
            return {
                "indices": exact_hit_indices,
                "mode": admission_mode,
                "raw_gaps": exact_hit_raw_gaps,
                "best_raw_gap": best_raw_gap,
            }

    selected_raw_gaps = np.asarray(
        [extract_raw_median_gap(gamma_results[idx], target_clusters) for idx in valid_indices],
        dtype=float,
    )
    best_raw_gap = float(np.min(selected_raw_gaps[np.isfinite(selected_raw_gaps)])) if np.any(np.isfinite(selected_raw_gaps)) else math.inf
    if len(valid_indices) > 1 and np.any(np.isfinite(selected_raw_gaps)):
        keep_mask = np.isfinite(selected_raw_gaps) & (selected_raw_gaps == best_raw_gap)
        if np.any(keep_mask) and int(np.sum(keep_mask)) < len(valid_indices):
            valid_indices = [valid_indices[idx] for idx in np.where(keep_mask)[0]]
            selected_raw_gaps = selected_raw_gaps[keep_mask]
    return {"indices": valid_indices, "mode": admission_mode, "raw_gaps": selected_raw_gaps, "best_raw_gap": best_raw_gap}


def merge_small_clusters_to_neighbors(labels: np.ndarray, snn_graph, min_cluster_size: int = 1) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    min_cluster_size = max(1, int(min_cluster_size))
    if min_cluster_size <= 1 or snn_graph is None or labels.size == 0:
        return labels
    if snn_graph.shape[0] != labels.size or snn_graph.shape[1] != labels.size:
        raise ValueError("snn_graph dimensions must match label length when min_cluster_size > 1.")

    base_labels = reindex_cluster_labels(labels)
    sizes = np.bincount(base_labels + 1)[1:]
    cluster_ids = np.where(sizes > 0)[0].astype(np.int32)
    if cluster_ids.size <= 1:
        return base_labels

    large_cluster_ids = cluster_ids[sizes[cluster_ids] >= min_cluster_size]
    small_cluster_ids = cluster_ids[sizes[cluster_ids] < min_cluster_size]
    if small_cluster_ids.size == 0:
        return base_labels

    if large_cluster_ids.size == 0:
        largest_size = int(np.max(sizes[cluster_ids]))
        largest_ids = cluster_ids[sizes[cluster_ids] == largest_size]
        target_id = int(np.min(largest_ids))
        return reindex_cluster_labels(np.full_like(base_labels, target_id))

    merged = base_labels.copy()
    cluster_cells = {
        int(cluster_id): np.where(base_labels == int(cluster_id))[0]
        for cluster_id in cluster_ids.tolist()
    }
    for small_id in small_cluster_ids:
        small_cells = cluster_cells[int(small_id)]
        if small_cells.size == 0:
            continue
        row_sum = np.asarray(snn_graph[small_cells].sum(axis=0)).ravel().astype(float, copy=False)
        cluster_connectivity = np.bincount(
            base_labels,
            weights=row_sum,
            minlength=int(sizes.size),
        )
        candidate_scores = cluster_connectivity[large_cluster_ids] / (
            float(small_cells.size) * sizes[large_cluster_ids].astype(float)
        )
        if candidate_scores.size == 0 or np.all(~np.isfinite(candidate_scores)):
            best_target = int(np.min(large_cluster_ids))
        else:
            max_score = float(np.nanmax(candidate_scores))
            tied_candidates = large_cluster_ids[np.where(np.isclose(candidate_scores, max_score))[0]]
            best_target = int(np.min(tied_candidates))
        if best_target is None:
            best_target = int(np.min(large_cluster_ids))
        merged[base_labels == small_id] = best_target
    return reindex_cluster_labels(merged)


def summarize_trial_cluster_counts(cluster_labels: np.ndarray, min_cluster_size: int) -> tuple[int, int]:
    counts = summarize_cluster_labels(cluster_labels, min_cluster_size=min_cluster_size)
    return int(counts["raw_cluster_count"]), int(counts["effective_cluster_count"])


def gamma_seed_role_priority(seed_role: str) -> int:
    priorities = {"selected": 1, "left": 2, "right": 2, "exact": 3, "near": 4, "seed": 5}
    return int(priorities.get(seed_role, 99))


def normalize_gamma_seed_table(gamma_seed_values: Any, gamma_range: tuple[float, float]):
    import pandas as pd

    empty = pd.DataFrame(columns=["gamma", "seed_role", "final_cluster_count", "raw_cluster_count"])
    if gamma_seed_values is None:
        return empty
    if isinstance(gamma_seed_values, pd.DataFrame):
        seed_table = gamma_seed_values.copy()
    elif isinstance(gamma_seed_values, dict) and "gamma" in gamma_seed_values:
        seed_table = pd.DataFrame(
            {
                "gamma": np.asarray(gamma_seed_values.get("gamma", []), dtype=float),
                "seed_role": gamma_seed_values.get("seed_role", None),
                "final_cluster_count": gamma_seed_values.get("final_cluster_count", np.nan),
                "raw_cluster_count": gamma_seed_values.get("raw_cluster_count", np.nan),
            }
        )
    else:
        gamma_values = np.asarray(gamma_seed_values, dtype=float)
        seed_table = pd.DataFrame(
            {
                "gamma": gamma_values,
                "seed_role": ["seed"] * len(gamma_values),
                "final_cluster_count": np.nan,
                "raw_cluster_count": np.nan,
            }
        )
    for column in ["gamma", "seed_role", "final_cluster_count", "raw_cluster_count"]:
        if column not in seed_table.columns:
            seed_table[column] = "seed" if column == "seed_role" else np.nan
    lower, upper = sorted((float(gamma_range[0]), float(gamma_range[1])))
    tolerance = max(np.sqrt(np.finfo(float).eps), abs(upper - lower) * 1e-8)
    seed_table = seed_table[np.isfinite(seed_table["gamma"])]
    seed_table = seed_table[(seed_table["gamma"] >= lower - tolerance) & (seed_table["gamma"] <= upper + tolerance)].copy()
    if seed_table.empty:
        return empty
    seed_table["seed_role"] = seed_table["seed_role"].fillna("seed").astype(str)
    seed_table["role_priority"] = seed_table["seed_role"].map(gamma_seed_role_priority)
    seed_table = seed_table.sort_values(["gamma", "role_priority"]).drop_duplicates(["gamma", "seed_role"]).drop(columns=["role_priority"]).reset_index(drop=True)
    return seed_table


def thin_gamma_candidates_by_gap(
    values: np.ndarray,
    protected_values: np.ndarray | None = None,
    objective_function: str = "CPM",
    min_log_gap: float = 0.08,
) -> np.ndarray:
    values = np.asarray(sorted(set(map(float, np.asarray(values)[np.isfinite(values)]))), dtype=float)
    if protected_values is None:
        protected = np.asarray([], dtype=float)
    else:
        protected_array = np.asarray(protected_values, dtype=float)
        protected = np.asarray(
            sorted(set(map(float, protected_array[np.isfinite(protected_array)]))),
            dtype=float,
        )
    if values.size <= 1 or objective_function != "CPM":
        return values
    keep: list[float] = []
    for value in values:
        compare_against = np.asarray(sorted(set([*protected.tolist(), *keep])), dtype=float)
        if compare_against.size == 0:
            keep.append(float(value))
            continue
        log_distance = np.min(np.abs(np.log(value) - np.log(compare_against)))
        if not np.isfinite(log_distance) or log_distance >= min_log_gap:
            keep.append(float(value))
    return np.asarray(sorted(set(keep)), dtype=float)


def select_evenly_spaced_gamma_values(values: np.ndarray, n_keep: int) -> np.ndarray:
    values = np.asarray(sorted(set(map(float, np.asarray(values)[np.isfinite(values)]))), dtype=float)
    n_keep = int(n_keep)
    if values.size <= n_keep or n_keep <= 0:
        return values
    keep_positions = np.unique(np.round(np.linspace(0, values.size - 1, num=n_keep)).astype(int))
    return values[keep_positions]


def build_even_interior_gamma_points(gamma_range: tuple[float, float], n_points: int, objective_function: str) -> np.ndarray:
    n_points = int(n_points)
    if n_points <= 0:
        return np.asarray([], dtype=float)
    lower, upper = sorted((float(gamma_range[0]), float(gamma_range[1])))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        return np.repeat(lower, n_points).astype(float)
    if objective_function == "CPM":
        points = np.exp(np.linspace(np.log(lower), np.log(upper), n_points + 2))
    else:
        points = np.linspace(lower, upper, n_points + 2)
    return points[1:-1].astype(float)


def fill_gamma_values_to_budget(existing_values: np.ndarray, budget: int, gamma_range: tuple[float, float], objective_function: str) -> np.ndarray:
    existing_values = np.asarray(sorted(set(map(float, np.asarray(existing_values)[np.isfinite(existing_values)]))), dtype=float)
    budget = int(budget)
    if budget <= existing_values.size:
        return existing_values
    candidates = build_even_interior_gamma_points(gamma_range, max(0, budget * 2), objective_function)
    if objective_function == "CPM":
        candidates = thin_gamma_candidates_by_gap(candidates, protected_values=existing_values, objective_function=objective_function)
    else:
        candidates = np.asarray(sorted(set(map(float, candidates))), dtype=float)
    candidates = select_evenly_spaced_gamma_values(candidates, max(0, budget - existing_values.size))
    merged = existing_values.tolist()
    for candidate in candidates:
        if not any(abs(candidate - value) <= np.sqrt(np.finfo(float).eps) for value in merged):
            merged.append(float(candidate))
    return np.asarray(sorted(set(merged)), dtype=float)


def build_secondary_gamma_points(primary_values: np.ndarray, gamma_range: tuple[float, float], objective_function: str, n_points: int) -> np.ndarray:
    n_points = int(n_points)
    if n_points <= 0:
        return np.asarray([], dtype=float)
    lower, upper = sorted((float(gamma_range[0]), float(gamma_range[1])))
    current_values = np.asarray(sorted(set(map(float, np.asarray(primary_values)[np.isfinite(primary_values)]))), dtype=float)
    if not np.any(np.isclose(current_values, lower)):
        current_values = np.sort(np.unique(np.append(current_values, lower)))
    if not np.any(np.isclose(current_values, upper)):
        current_values = np.sort(np.unique(np.append(current_values, upper)))

    secondary_values: list[float] = []
    transform = np.log if objective_function == "CPM" else (lambda x: x)
    inverse_transform = np.exp if objective_function == "CPM" else (lambda x: x)

    for _ in range(n_points):
        sorted_values = np.asarray(sorted(set([*current_values.tolist(), *secondary_values])), dtype=float)
        if sorted_values.size < 2:
            break
        transformed = transform(sorted_values)
        gap_widths = np.diff(transformed)
        gap_idx = int(np.argmax(gap_widths))
        if not np.isfinite(gap_widths[gap_idx]) or gap_widths[gap_idx] <= 0:
            break
        midpoint = float(inverse_transform(np.mean(transformed[[gap_idx, gap_idx + 1]])))
        if not np.isfinite(midpoint):
            break
        secondary_values.append(midpoint)
        current_values = np.sort(np.unique(np.append(current_values, midpoint)))
    return np.asarray([value for value in sorted(set(secondary_values)) if value not in primary_values], dtype=float)


def build_local_recovery_gamma_points(
    gamma_results: list[dict[str, Any]],
    gamma_range: tuple[float, float],
    objective_function: str,
    target_clusters: int,
    resolution_tolerance: float,
    n_points: int = 4,
) -> np.ndarray:
    if not gamma_results:
        return np.asarray([], dtype=float)

    supporting = [
        float(result.get("gamma", np.nan))
        for result in gamma_results
        if (
            int(result.get("hit_count", 0)) > 0
            or abs(float(result.get("final_cluster_median", np.nan)) - float(target_clusters)) <= 1.0
            or abs(float(result.get("raw_cluster_median", np.nan)) - float(target_clusters)) <= 1.0
        )
        and np.isfinite(float(result.get("gamma", np.nan)))
    ]
    if not supporting:
        return np.asarray([], dtype=float)

    evaluated = np.asarray(
        sorted(
            {
                float(result.get("gamma", np.nan))
                for result in gamma_results
                if np.isfinite(float(result.get("gamma", np.nan)))
            }
        ),
        dtype=float,
    )
    lower_bound = float(min(gamma_range))
    upper_bound = float(max(gamma_range))
    left = float(max(lower_bound, min(supporting)))
    right = float(min(upper_bound, max(supporting)))
    if np.isclose(left, right) and evaluated.size:
        insert_pos = int(np.searchsorted(evaluated, left))
        neighbor_values: list[float] = [left]
        if insert_pos - 1 >= 0:
            neighbor_values.append(float(evaluated[insert_pos - 1]))
        if insert_pos < evaluated.size:
            neighbor_values.append(float(evaluated[min(insert_pos, evaluated.size - 1)]))
        left = float(max(lower_bound, min(neighbor_values)))
        right = float(min(upper_bound, max(neighbor_values)))
    if not np.isfinite(left) or not np.isfinite(right) or left >= right:
        return np.asarray([], dtype=float)

    candidates = build_even_interior_gamma_points((left, right), n_points=max(1, int(n_points)), objective_function=objective_function)
    if candidates.size == 0:
        candidates = build_gamma_sequence_for_range(
            gamma_range=(left, right),
            objective_function=objective_function,
            resolution_tolerance=resolution_tolerance,
            n_steps=max(2, int(n_points) + 1),
        )
    if candidates.size == 0:
        return np.asarray([], dtype=float)

    tolerance = max(np.sqrt(np.finfo(float).eps), abs(right - left) * 1e-8, 1e-12)
    recovery_points = [
        float(candidate)
        for candidate in np.asarray(candidates, dtype=float).tolist()
        if np.isfinite(candidate) and not np.any(np.isclose(evaluated, candidate, atol=tolerance, rtol=0.0))
    ]
    return np.asarray(sorted(set(recovery_points)), dtype=float)


def build_optimization_gamma_batches(
    gamma_range: tuple[float, float],
    gamma_seed_values: Any,
    target_clusters: int,
    objective_function: str,
    resolution_tolerance: float,
    n_vertices: int,
    primary_budget: int = 8,
    secondary_budget: int = 4,
) -> dict[str, Any]:
    seed_table = normalize_gamma_seed_table(gamma_seed_values, gamma_range)
    anchors = np.asarray(sorted(gamma_range), dtype=float)
    exact_values = np.asarray([], dtype=float)
    near_values = np.asarray([], dtype=float)
    generic_seed_values = np.asarray([], dtype=float)
    if not seed_table.empty:
        anchor_mask = seed_table["seed_role"].isin(["left", "right", "selected"])
        anchors = np.asarray(sorted(set([*anchors.tolist(), *seed_table.loc[anchor_mask, "gamma"].astype(float).tolist()])), dtype=float)
        exact_values = np.asarray(sorted(set(seed_table.loc[seed_table["seed_role"] == "exact", "gamma"].astype(float).tolist())), dtype=float)
        near_values = np.asarray(sorted(set(seed_table.loc[seed_table["seed_role"] == "near", "gamma"].astype(float).tolist())), dtype=float)
        generic_seed_values = np.asarray(sorted(set(seed_table.loc[seed_table["seed_role"] == "seed", "gamma"].astype(float).tolist())), dtype=float)

    primary_values = anchors.copy()
    remaining_slots = max(0, int(primary_budget) - primary_values.size)
    if remaining_slots > 0 and exact_values.size:
        exact_candidates = exact_values[~np.isin(exact_values, primary_values)]
        primary_values = np.sort(np.unique(np.concatenate([primary_values, select_evenly_spaced_gamma_values(exact_candidates, remaining_slots)])))

    remaining_slots = max(0, int(primary_budget) - primary_values.size)
    if remaining_slots > 0 and near_values.size:
        near_candidates = near_values[~np.isin(near_values, primary_values)]
        near_candidates = thin_gamma_candidates_by_gap(near_candidates, protected_values=primary_values, objective_function=objective_function)
        primary_values = np.sort(np.unique(np.concatenate([primary_values, select_evenly_spaced_gamma_values(near_candidates, remaining_slots)])))

    remaining_slots = max(0, int(primary_budget) - primary_values.size)
    if remaining_slots > 0 and generic_seed_values.size:
        generic_candidates = generic_seed_values[~np.isin(generic_seed_values, primary_values)]
        generic_candidates = thin_gamma_candidates_by_gap(generic_candidates, protected_values=primary_values, objective_function=objective_function)
        primary_values = np.sort(np.unique(np.concatenate([primary_values, select_evenly_spaced_gamma_values(generic_candidates, remaining_slots)])))

    primary_values = fill_gamma_values_to_budget(primary_values, int(primary_budget), gamma_range, objective_function)
    primary_values = primary_values[(primary_values >= min(gamma_range)) & (primary_values <= max(gamma_range))]
    secondary_values = build_secondary_gamma_points(primary_values, gamma_range, objective_function, int(secondary_budget))
    return {"primary_gammas": np.sort(np.unique(primary_values)), "secondary_gammas": np.sort(np.unique(secondary_values)), "seed_table": seed_table}


def derive_gamma_admission_state(
    gamma_results: list[dict[str, Any]],
    target_clusters: int,
    min_cluster_size: int = 1,
    verbose: bool = False,
    worker_id: str = "OPTIMIZER",
) -> dict[str, Any]:
    if not gamma_results:
        return {
            "valid_indices": [],
            "admission_mode": "none",
            "exact_hit_gamma_count": 0,
            "selected_raw_gaps": np.asarray([], dtype=float),
            "best_raw_gap": math.inf,
        }

    strict_flags = np.asarray([bool(x.get("strict_valid", False)) for x in gamma_results], dtype=bool)
    relaxed_flags = np.asarray([bool(x.get("relaxed_valid", False)) for x in gamma_results], dtype=bool)
    raw_strict_flags = np.asarray([bool(x.get("raw_strict_valid", False)) for x in gamma_results], dtype=bool)
    raw_relaxed_flags = np.asarray([bool(x.get("raw_relaxed_valid", False)) for x in gamma_results], dtype=bool)
    soft_guard_flags = np.asarray([bool(x.get("raw_guard_soft", True)) for x in gamma_results], dtype=bool)
    hard_guard_flags = np.asarray([bool(x.get("raw_guard_hard", True)) for x in gamma_results], dtype=bool)
    hit_counts = np.asarray([int(x.get("hit_count", 0)) for x in gamma_results], dtype=int)

    if verbose:
        logger.info(
            "%s: strict=%s relaxed=%s raw_strict=%s raw_relaxed=%s soft_guard=%s hard_guard=%s",
            worker_id,
            int(np.sum(strict_flags)),
            int(np.sum(relaxed_flags)),
            int(np.sum(raw_strict_flags)),
            int(np.sum(raw_relaxed_flags)),
            int(np.sum(soft_guard_flags)),
            int(np.sum(hard_guard_flags)),
        )

    admission = select_gamma_admission(
        strict_flags=strict_flags,
        relaxed_flags=relaxed_flags,
        soft_guard_flags=soft_guard_flags,
        hard_guard_flags=hard_guard_flags,
        raw_strict_flags=raw_strict_flags,
        raw_relaxed_flags=raw_relaxed_flags,
    )
    refined = refine_gamma_candidates_by_raw_gap(
        valid_indices=admission["indices"],
        admission_mode=admission["mode"],
        gamma_results=gamma_results,
        target_clusters=target_clusters,
        min_cluster_size=min_cluster_size,
    )
    return {
        "valid_indices": refined["indices"],
        "admission_mode": refined["mode"],
        "exact_hit_gamma_count": int(np.sum(hit_counts > 0)),
        "selected_raw_gaps": refined["raw_gaps"],
        "best_raw_gap": refined["best_raw_gap"],
    }


def should_expand_phase1_secondary(valid_indices: list[int], admission_mode: str, exact_hit_gamma_count: int) -> bool:
    guarded_modes = {"raw_strict_soft", "strict_soft", "relaxed_soft", "strict_hard", "relaxed_hard"}
    if not valid_indices:
        return True
    if exact_hit_gamma_count > 0:
        return False
    if admission_mode in guarded_modes:
        return False
    return admission_mode in {"relaxed_unguarded", "raw_relaxed_unguarded"}


def should_skip_phase4_refinement(candidate_count: int, best_ic: float, exact_hit_gamma_count: int) -> bool:
    return int(candidate_count) <= 2 and np.isfinite(best_ic) and best_ic <= 1.005 and int(exact_hit_gamma_count) > 0


def phase4_iteration_cap_for_mode(admission_mode: str) -> int:
    return 2 if admission_mode in {"relaxed_unguarded", "raw_relaxed_unguarded"} else 3


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
    min_cluster_size: int = 1,
    verbose: bool = False,
    worker_id: str = "OPTIMIZER",
    runtime_context=None,
) -> dict[str, Any]:
    best_clustering = load_cluster_matrix(matrix_ref)
    heartbeat = create_heartbeat_logger(verbose=verbose, context=worker_id)
    n_trials = best_clustering.shape[0]
    preferred_trial_indices = sorted(set(int(idx) for idx in (preferred_trial_indices or []) if 0 <= int(idx) < n_trials))
    if not preferred_trial_indices and target_clusters is not None:
        final_clusters_vec = np.asarray(
            [
                len(np.unique(merge_small_clusters_to_neighbors(best_clustering[idx], snn_graph, min_cluster_size)))
                if min_cluster_size > 1
                else len(np.unique(best_clustering[idx]))
                for idx in range(n_trials)
            ],
            dtype=int,
        )
        preferred_trial_indices = np.where(final_clusters_vec == int(target_clusters))[0].tolist()

    bootstrap_start = time.time()
    if verbose:
        logger.info("%s: Phase 5 - Bootstrap analysis with %s iterations", worker_id, n_bootstrap)
    bootstrap_workers = cap_workers_by_memory(max(1, int(n_workers)), estimate_trial_matrix_bytes(best_clustering.shape[1], n_trials, 1), runtime_context)
    bootstrap_workers = min(bootstrap_workers, max(1, int(n_bootstrap)))
    bootstrap_log_every = max(1, int(math.floor(max(1, int(n_bootstrap)) / 5)))

    def should_log_bootstrap_step(step_idx: int) -> bool:
        return step_idx == 1 or step_idx == int(n_bootstrap) or (step_idx % bootstrap_log_every) == 0

    def run_single_bootstrap(bootstrap_idx: int) -> float:
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

    trial_cluster_counts = [summarize_trial_cluster_counts(row, min_cluster_size=min_cluster_size) for row in cluster_matrix]
    raw_clusters_vec = np.asarray([item[0] for item in trial_cluster_counts], dtype=int)
    effective_clusters_vec = np.asarray([item[1] for item in trial_cluster_counts], dtype=int)
    if min_cluster_size > 1:
        merged_labels = [merge_small_clusters_to_neighbors(row, snn_graph=snn_graph, min_cluster_size=min_cluster_size) for row in cluster_matrix]
        final_clusters_vec = np.asarray([np.unique(row).size for row in merged_labels], dtype=int)
    else:
        final_clusters_vec = raw_clusters_vec.copy()

    median_effective_clusters = float(np.median(effective_clusters_vec))
    raw_cluster_median = float(np.median(raw_clusters_vec))
    final_cluster_median = float(np.median(final_clusters_vec))
    hit_trials = np.where(final_clusters_vec == int(target_clusters))[0]
    raw_hit_trials = np.where(raw_clusters_vec == int(target_clusters))[0]
    hit_count = int(hit_trials.size)
    raw_hit_count = int(raw_hit_trials.size)
    median_gap = abs(final_cluster_median - float(target_clusters))
    raw_median_gap = abs(raw_cluster_median - float(target_clusters))
    within_median_window = median_gap <= 1
    strict_valid = int(final_cluster_median) == int(target_clusters)
    relaxed_valid = hit_count >= 1 and within_median_window
    raw_within_median_window = raw_median_gap <= 1
    raw_strict_valid = int(raw_cluster_median) == int(target_clusters)
    raw_relaxed_valid = raw_hit_count >= 1 and raw_within_median_window
    raw_guard_soft = bool(passes_raw_cluster_guard(raw_cluster_median, target_clusters, min_cluster_size=min_cluster_size, level="soft"))
    raw_guard_hard = bool(passes_raw_cluster_guard(raw_cluster_median, target_clusters, min_cluster_size=min_cluster_size, level="hard"))
    gamma_admitted = strict_valid or relaxed_valid or raw_strict_valid or raw_relaxed_valid

    result = {
        "valid": gamma_admitted,
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
        "hit_trials": hit_trials.tolist(),
    }
    if not gamma_admitted:
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
    result["matrix_ref"] = store_cluster_matrix(cluster_matrix, runtime_context=runtime_context, prefix=f"k{target_clusters}_g{abs(hash((target_clusters, gamma_val))) % 100000}")
    if log_this_gamma and gamma_idx is not None and gamma_total is not None:
        logger.info(
            "%s: Phase 1 progress gamma %s/%s completed in %.3f seconds - median_effective = %.6g - median_final = %.6g - median_raw = %.6g - median gap = %.3f - final hit trials = %s/%s - raw hit trials = %s/%s - strict_valid = %s - relaxed_valid = %s - raw_strict_valid = %s - raw_relaxed_valid = %s - raw_guard_soft = %s - raw_guard_hard = %s - IC (all trials) = %.4f",
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
        )
    return result


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
) -> dict[str, Any]:
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

    def compute_phase1_nested_workers(batch_gamma_count: int) -> int:
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
        return step_idx == 1 or (step_idx % phase1_log_every) == 0

    def evaluate_gamma_batch(gamma_sequence: np.ndarray, batch_label: str) -> dict[str, Any]:
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

    phase1_expected_runs = int((len(primary_gamma_sequence) + len(secondary_gamma_sequence)) * max(1, int(n_trials)))
    if verbose:
        logger.info("%s: Phase 1 maximum expected Leiden runs: %s", worker_id, f"{phase1_expected_runs:,}")

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
    effective_cluster_medians = np.asarray([result.get("median_effective_clusters", np.nan) for result in valid_results], dtype=float)
    final_cluster_medians = np.asarray([result.get("final_cluster_median", np.nan) for result in valid_results], dtype=float)
    raw_cluster_medians = np.asarray([result.get("raw_cluster_median", np.nan) for result in valid_results], dtype=float)
    preferred_hit_trials = [result.get("hit_trials", []) for result in valid_results]

    best_index = int(np.where(ic_scores == 1.0)[0][0]) if np.any(ic_scores == 1.0) else int(np.argmin(ic_scores))
    best_gamma = float(gamma_sequence[best_index])
    best_ref = clustering_refs[best_index]
    best_preferred_trials = preferred_hit_trials[best_index]
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
            ic_history = np.tile(current_ic[:, None], (1, 10))
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
                    final_clusters_vec = (
                        np.asarray([np.unique(merge_small_clusters_to_neighbors(row, snn_graph=snn_graph, min_cluster_size=min_cluster_size)).size for row in new_matrix], dtype=int)
                        if min_cluster_size > 1
                        else np.asarray([np.unique(row).size for row in new_matrix], dtype=int)
                    )
                    extracted = extract_clustering_array(new_matrix)
                    new_results.append(
                        {
                            "matrix_ref": store_cluster_matrix(new_matrix, runtime_context=runtime_context, prefix=f"k{target_clusters}_iter{k}_g{gamma_idx}"),
                            "ic": calculate_ic_from_extracted(extracted, n_workers=1),
                            "exact_hit_count": int(np.sum(final_clusters_vec == int(target_clusters))),
                            "preferred_trials": np.where(final_clusters_vec == int(target_clusters))[0].tolist(),
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
                    )
                    for idx in range(len(new_results))
                ]
                candidate_refs = new_refs
                candidate_ic = new_ic
                candidate_gammas = current_gammas
                candidate_preferred_trials = [item["preferred_trials"] for item in new_results]
                candidate_history = np.concatenate([ic_history[:, 1:], new_ic[:, None]], axis=1)
                candidate_exact_hit_flags = new_exact_hit_counts > 0
                candidate_origin_indices = np.arange(len(new_results), dtype=int)

                if prefer_exact_hits:
                    if not np.any(candidate_exact_hit_flags):
                        for row_idx in range(len(phase4_rows)):
                            phase4_rows[row_idx]["phase4_keep"] = False
                            phase4_rows[row_idx]["phase4_prune_reason"] = "missing_exact_hit_support"
                        gamma_diagnostics_rows.extend(phase4_rows)
                        release_cluster_matrix_refs(new_refs)
                        best_index = int(np.where(current_ic == 1.0)[0][0]) if np.any(current_ic == 1.0) else int(np.argmin(current_ic))
                        best_gamma = float(current_gammas[best_index])
                        best_ref = current_refs[best_index]
                        best_preferred_trials = current_preferred_trials[best_index]
                        release_cluster_matrix_refs([ref for idx, ref in enumerate(current_refs) if idx != best_index])
                        converged = True
                        break
                    if not np.all(candidate_exact_hit_flags):
                        release_cluster_matrix_refs([ref for idx, ref in enumerate(candidate_refs) if not candidate_exact_hit_flags[idx]])
                        for row_idx, keep_flag in enumerate(candidate_exact_hit_flags.tolist()):
                            if not keep_flag:
                                phase4_rows[row_idx]["phase4_keep"] = False
                                phase4_rows[row_idx]["phase4_prune_reason"] = "missing_exact_hit_support"
                        keep_idx = np.where(candidate_exact_hit_flags)[0]
                        candidate_refs = [candidate_refs[idx] for idx in keep_idx]
                        candidate_ic = candidate_ic[keep_idx]
                        candidate_gammas = candidate_gammas[keep_idx]
                        candidate_history = candidate_history[keep_idx]
                        candidate_preferred_trials = [candidate_preferred_trials[idx] for idx in keep_idx]
                        candidate_origin_indices = candidate_origin_indices[keep_idx]

                release_cluster_matrix_refs(current_refs)
                stable_indices = np.asarray([len(np.unique(row)) == 1 for row in candidate_history], dtype=bool)
                perfect_indices = np.where(candidate_ic == 1.0)[0]

                if perfect_indices.size:
                    best_index = int(perfect_indices[0])
                    origin_best = int(candidate_origin_indices[best_index])
                    best_gamma = float(candidate_gammas[best_index])
                    best_ref = candidate_refs[best_index]
                    best_preferred_trials = candidate_preferred_trials[best_index]
                    for row_idx in candidate_origin_indices.tolist():
                        phase4_rows[int(row_idx)]["phase4_keep"] = int(row_idx) == origin_best
                        phase4_rows[int(row_idx)]["phase4_prune_reason"] = None if int(row_idx) == origin_best else "perfect_ic_superseded"
                    release_cluster_matrix_refs([ref for idx, ref in enumerate(candidate_refs) if idx != best_index])
                    gamma_diagnostics_rows.extend(phase4_rows)
                    converged = True
                    break
                if np.all(stable_indices):
                    best_index = int(np.argmin(candidate_ic))
                    origin_best = int(candidate_origin_indices[best_index])
                    best_gamma = float(candidate_gammas[best_index])
                    best_ref = candidate_refs[best_index]
                    best_preferred_trials = candidate_preferred_trials[best_index]
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
                if len(keep_pool) > keep_limit:
                    best_global_idx = int(np.argmin(candidate_ic))
                    stable_pool = np.where(stable_indices)[0]
                    stable_best_idx = int(stable_pool[np.argmin(candidate_ic[stable_pool])]) if stable_pool.size else None
                    ordered_pool = [idx for idx in np.argsort(np.lexsort((candidate_gammas, candidate_ic))).tolist() if idx in keep_pool]
                    merged_pool = [best_global_idx]
                    if stable_best_idx is not None:
                        merged_pool.append(stable_best_idx)
                    merged_pool.extend(ordered_pool)
                    keep_pool = list(dict.fromkeys(merged_pool))[:keep_limit]
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
                ic_history = candidate_history[keep_mask]

            if not converged:
                best_index = int(np.argmin(current_ic))
                best_gamma = float(current_gammas[best_index])
                best_ref = current_refs[best_index]
                best_preferred_trials = current_preferred_trials[best_index]
                release_cluster_matrix_refs([ref for idx, ref in enumerate(current_refs) if idx != best_index])

            phase4_elapsed_sec = time.time() - iterative_start
    else:
        release_cluster_matrix_refs([ref for idx, ref in enumerate(clustering_refs) if idx != best_index])

    best_gamma_diag_index = int(np.argmin(np.abs(gamma_sequence - best_gamma)))
    finalized = finalize_selected_clustering(
        matrix_ref=best_ref,
        gamma=best_gamma,
        effective_cluster_median=float(effective_cluster_medians[best_gamma_diag_index]),
        raw_cluster_median=float(raw_cluster_medians[best_gamma_diag_index]),
        final_cluster_median=float(final_cluster_medians[best_gamma_diag_index]),
        admission_mode=admission_mode,
        cluster_seed=cluster_seed,
        n_bootstrap=n_bootstrap,
        n_workers=phase1_nested_workers,
        snn_graph=snn_graph,
        target_clusters=target_clusters,
        preferred_trial_indices=best_preferred_trials,
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
        return step_idx == 1 or step_idx == int(n_trials) or (step_idx % phase1_log_every) == 0

    if verbose:
        logger.info("%s: Phase 1 - Evaluating fixed resolution with %s trials", worker_id, n_trials)
        if in_parallel_context:
            logger.info("%s: Running in parallel context with worker budget %s", worker_id, n_workers)
        logger.info("%s: Trial worker budget: %s", worker_id, trial_workers)
    phase1_start = time.time()

    def run_single_trial(trial_idx: int) -> np.ndarray:
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

    trial_cluster_counts = [summarize_trial_cluster_counts(row, min_cluster_size=min_cluster_size) for row in cluster_matrix]
    raw_clusters_vec = np.asarray([item[0] for item in trial_cluster_counts], dtype=int)
    effective_clusters_vec = np.asarray([item[1] for item in trial_cluster_counts], dtype=int)
    if min_cluster_size > 1:
        final_clusters_vec = np.asarray([np.unique(merge_small_clusters_to_neighbors(row, snn_graph=snn_graph, min_cluster_size=min_cluster_size)).size for row in cluster_matrix], dtype=int)
    else:
        final_clusters_vec = raw_clusters_vec.copy()

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
        min_cluster_size=min_cluster_size,
        verbose=verbose,
        worker_id=worker_id,
        runtime_context=runtime_context,
    )
    finalized["phase1_ic"] = phase1_ic
    finalized["n_iterations"] = int(n_iterations)
    finalized["k"] = int(n_iterations)
    return finalized
