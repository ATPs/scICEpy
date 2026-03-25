"""Gamma candidate construction, admission, and ranking helpers for optimization."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .runtime import logger
from .search_bounds import build_gamma_sequence_for_range


def extract_raw_median_gap(result: dict[str, Any], target_clusters: int) -> float:
    """Return the stored raw-gap metric, falling back to the result mean when needed."""
    if "raw_median_gap" not in result or not np.isfinite(result["raw_median_gap"]):
        return abs(float(result.get("mean_clusters", np.nan)) - float(target_clusters))
    return float(result["raw_median_gap"])


def select_gamma_admission(
    strict_flags: np.ndarray,
    relaxed_flags: np.ndarray,
    soft_guard_flags: np.ndarray,
    hard_guard_flags: np.ndarray,
    raw_strict_flags: np.ndarray | None = None,
    raw_relaxed_flags: np.ndarray | None = None,
    exact_hit_flags: np.ndarray | None = None,
) -> dict[str, Any]:
    """Choose the highest-priority non-empty admission bucket for the current gamma candidates."""
    strict_flags = np.asarray(strict_flags, dtype=bool)
    relaxed_flags = np.asarray(relaxed_flags, dtype=bool)
    soft_guard_flags = np.asarray(soft_guard_flags, dtype=bool)
    hard_guard_flags = np.asarray(hard_guard_flags, dtype=bool)
    raw_strict_flags = (
        np.zeros_like(strict_flags) if raw_strict_flags is None else np.asarray(raw_strict_flags, dtype=bool)
    )
    raw_relaxed_flags = (
        np.zeros_like(strict_flags) if raw_relaxed_flags is None else np.asarray(raw_relaxed_flags, dtype=bool)
    )
    exact_hit_flags = (
        np.zeros_like(strict_flags) if exact_hit_flags is None else np.asarray(exact_hit_flags, dtype=bool)
    )
    exact_hit_supported_flags = (
        strict_flags | relaxed_flags | exact_hit_flags
        if np.any(exact_hit_flags) and np.any(strict_flags | relaxed_flags)
        else np.zeros_like(strict_flags)
    )

    candidate_sets = {
        "raw_strict_soft": np.where(raw_strict_flags & soft_guard_flags)[0],
        "strict_soft": np.where(strict_flags & soft_guard_flags)[0],
        "relaxed_soft": np.where(relaxed_flags & soft_guard_flags)[0],
        "strict_hard": np.where(strict_flags & hard_guard_flags)[0],
        "relaxed_hard": np.where(relaxed_flags & hard_guard_flags)[0],
        "exact_hit_supported": np.where(exact_hit_supported_flags)[0],
        "relaxed_unguarded": np.where(relaxed_flags)[0],
        "raw_relaxed_soft": np.where(raw_relaxed_flags & soft_guard_flags)[0],
        "raw_relaxed_hard": np.where(raw_relaxed_flags & hard_guard_flags)[0],
        "raw_relaxed_unguarded": np.where(raw_relaxed_flags)[0],
    }
    for mode, indices in candidate_sets.items():
        if indices.size:
            return {"indices": indices.tolist(), "mode": mode}
    return {"indices": [], "mode": "none"}


def refine_gamma_candidates_by_raw_gap(
    valid_indices: list[int],
    admission_mode: str,
    gamma_results: list[dict[str, Any]],
    target_clusters: int,
    min_cluster_size: int = 1,
) -> dict[str, Any]:
    """Keep the admitted gamma candidates with the smallest raw-median gap to the target cluster count."""
    if not valid_indices or int(min_cluster_size) <= 1:
        return {
            "indices": valid_indices,
            "mode": admission_mode,
            "raw_gaps": np.asarray([], dtype=float),
            "best_raw_gap": math.inf,
        }

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
    best_raw_gap = (
        float(np.min(selected_raw_gaps[np.isfinite(selected_raw_gaps)]))
        if np.any(np.isfinite(selected_raw_gaps))
        else math.inf
    )
    if len(valid_indices) > 1 and np.any(np.isfinite(selected_raw_gaps)):
        keep_mask = np.isfinite(selected_raw_gaps) & (selected_raw_gaps == best_raw_gap)
        if np.any(keep_mask) and int(np.sum(keep_mask)) < len(valid_indices):
            valid_indices = [valid_indices[idx] for idx in np.where(keep_mask)[0]]
            selected_raw_gaps = selected_raw_gaps[keep_mask]
    return {
        "indices": valid_indices,
        "mode": admission_mode,
        "raw_gaps": selected_raw_gaps,
        "best_raw_gap": best_raw_gap,
    }


def gamma_seed_role_priority(seed_role: str) -> int:
    """Rank gamma seed roles so interval anchors and exact hits win during deduplication."""
    priorities = {"selected": 1, "left": 2, "right": 2, "exact": 3, "near": 4, "seed": 5}
    return int(priorities.get(seed_role, 99))


def normalize_gamma_seed_table(gamma_seed_values: Any, gamma_range: tuple[float, float]) -> pd.DataFrame:
    """Normalize user-supplied gamma seed inputs into a bounded, deduplicated DataFrame."""
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
    seed_table = seed_table[
        (seed_table["gamma"] >= lower - tolerance) & (seed_table["gamma"] <= upper + tolerance)
    ].copy()
    if seed_table.empty:
        return empty
    seed_table["seed_role"] = seed_table["seed_role"].fillna("seed").astype(str)
    seed_table["role_priority"] = seed_table["seed_role"].map(gamma_seed_role_priority)
    seed_table = (
        seed_table.sort_values(["gamma", "role_priority"])
        .drop_duplicates(["gamma", "seed_role"])
        .drop(columns=["role_priority"])
        .reset_index(drop=True)
    )
    return seed_table


def thin_gamma_candidates_by_gap(
    values: np.ndarray,
    protected_values: np.ndarray | None = None,
    objective_function: str = "CPM",
    min_log_gap: float = 0.08,
) -> np.ndarray:
    """Prune nearby CPM gamma values while preserving protected anchors."""
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
    """Subsample candidate gamma values at evenly spaced ranks."""
    values = np.asarray(sorted(set(map(float, np.asarray(values)[np.isfinite(values)]))), dtype=float)
    n_keep = int(n_keep)
    if values.size <= n_keep or n_keep <= 0:
        return values
    keep_positions = np.unique(np.round(np.linspace(0, values.size - 1, num=n_keep)).astype(int))
    return values[keep_positions]


def build_even_interior_gamma_points(
    gamma_range: tuple[float, float],
    n_points: int,
    objective_function: str,
) -> np.ndarray:
    """Generate evenly spaced interior gamma values inside the requested interval."""
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


def fill_gamma_values_to_budget(
    existing_values: np.ndarray,
    budget: int,
    gamma_range: tuple[float, float],
    objective_function: str,
) -> np.ndarray:
    """Pad a gamma set with evenly distributed interior values until the batch budget is filled."""
    existing_values = np.asarray(
        sorted(set(map(float, np.asarray(existing_values)[np.isfinite(existing_values)]))),
        dtype=float,
    )
    budget = int(budget)
    if budget <= existing_values.size:
        return existing_values
    candidates = build_even_interior_gamma_points(gamma_range, max(0, budget * 2), objective_function)
    if objective_function == "CPM":
        candidates = thin_gamma_candidates_by_gap(
            candidates,
            protected_values=existing_values,
            objective_function=objective_function,
        )
    else:
        candidates = np.asarray(sorted(set(map(float, candidates))), dtype=float)
    candidates = select_evenly_spaced_gamma_values(candidates, max(0, budget - existing_values.size))
    merged = existing_values.tolist()
    for candidate in candidates:
        if not any(abs(candidate - value) <= np.sqrt(np.finfo(float).eps) for value in merged):
            merged.append(float(candidate))
    return np.asarray(sorted(set(merged)), dtype=float)


def build_secondary_gamma_points(
    primary_values: np.ndarray,
    gamma_range: tuple[float, float],
    objective_function: str,
    n_points: int,
) -> np.ndarray:
    """Split the widest remaining gaps between primary gamma anchors to build the secondary batch."""
    n_points = int(n_points)
    if n_points <= 0:
        return np.asarray([], dtype=float)
    lower, upper = sorted((float(gamma_range[0]), float(gamma_range[1])))
    current_values = np.asarray(
        sorted(set(map(float, np.asarray(primary_values)[np.isfinite(primary_values)]))),
        dtype=float,
    )
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
    return np.asarray(
        [value for value in sorted(set(secondary_values)) if value not in primary_values],
        dtype=float,
    )


def build_local_recovery_gamma_points(
    gamma_results: list[dict[str, Any]],
    gamma_range: tuple[float, float],
    objective_function: str,
    target_clusters: int,
    resolution_tolerance: float,
    n_points: int = 4,
) -> np.ndarray:
    """Build a local recovery batch around promising Phase 1 gammas when exact-hit support is still weak."""
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

    candidates = build_even_interior_gamma_points(
        (left, right),
        n_points=max(1, int(n_points)),
        objective_function=objective_function,
    )
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


def _cluster_gap(value: Any, target_clusters: int) -> float:
    """Return the absolute gap to the target cluster count, using infinity for missing values."""
    numeric = float(value) if value is not None else np.nan
    if not np.isfinite(numeric):
        return math.inf
    return abs(numeric - float(target_clusters))


def gamma_candidate_sort_key(
    result: dict[str, Any],
    target_clusters: int,
    *,
    exact_support: bool = False,
    prefer_right_exact_hits: bool = False,
) -> tuple[Any, ...]:
    """Build a stable ranking key for choosing which admitted gamma to finalize first."""
    gamma = float(result.get("gamma", np.nan))
    ic = float(result.get("ic", np.nan))
    hit_count = int(result.get("hit_count", 0))
    strict_valid = bool(result.get("strict_valid", False))
    relaxed_valid = bool(result.get("relaxed_valid", False))
    raw_strict_valid = bool(result.get("raw_strict_valid", False))
    raw_relaxed_valid = bool(result.get("raw_relaxed_valid", False))
    final_gap = _cluster_gap(result.get("final_cluster_median", np.nan), target_clusters)
    effective_gap = _cluster_gap(
        result.get("median_effective_clusters", result.get("effective_cluster_median", np.nan)),
        target_clusters,
    )
    raw_gap = extract_raw_median_gap(result, target_clusters)
    perfect_ic = bool(np.isfinite(ic) and ic == 1.0)
    ic_key = float(ic) if np.isfinite(ic) else math.inf
    gamma_desc_key = -float(gamma) if np.isfinite(gamma) else math.inf

    if exact_support and prefer_right_exact_hits:
        return (
            -int(perfect_ic),
            -int(strict_valid),
            -int(hit_count),
            -int(relaxed_valid),
            -int(raw_strict_valid),
            -int(raw_relaxed_valid),
            gamma_desc_key,
            ic_key,
            final_gap,
            effective_gap,
            raw_gap,
        )
    return (
        -int(perfect_ic),
        -int(strict_valid),
        -int(relaxed_valid),
        ic_key,
        final_gap,
        effective_gap,
        raw_gap,
        -int(hit_count),
        -int(raw_strict_valid),
        -int(raw_relaxed_valid),
        gamma_desc_key,
    )


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
    """Build the primary and secondary gamma batches used by Phase 1 optimization."""
    del target_clusters, resolution_tolerance, n_vertices

    seed_table = normalize_gamma_seed_table(gamma_seed_values, gamma_range)
    anchors = np.asarray(sorted(gamma_range), dtype=float)
    exact_values = np.asarray([], dtype=float)
    near_values = np.asarray([], dtype=float)
    generic_seed_values = np.asarray([], dtype=float)
    if not seed_table.empty:
        anchor_mask = seed_table["seed_role"].isin(["left", "right", "selected"])
        anchors = np.asarray(
            sorted(
                set(
                    [
                        *anchors.tolist(),
                        *seed_table.loc[anchor_mask, "gamma"].astype(float).tolist(),
                    ]
                )
            ),
            dtype=float,
        )
        exact_values = np.asarray(
            sorted(set(seed_table.loc[seed_table["seed_role"] == "exact", "gamma"].astype(float).tolist())),
            dtype=float,
        )
        near_values = np.asarray(
            sorted(set(seed_table.loc[seed_table["seed_role"] == "near", "gamma"].astype(float).tolist())),
            dtype=float,
        )
        generic_seed_values = np.asarray(
            sorted(set(seed_table.loc[seed_table["seed_role"] == "seed", "gamma"].astype(float).tolist())),
            dtype=float,
        )

    primary_values = anchors.copy()
    remaining_slots = max(0, int(primary_budget) - primary_values.size)
    if remaining_slots > 0 and exact_values.size:
        exact_candidates = exact_values[~np.isin(exact_values, primary_values)]
        primary_values = np.sort(
            np.unique(
                np.concatenate(
                    [primary_values, select_evenly_spaced_gamma_values(exact_candidates, remaining_slots)]
                )
            )
        )

    remaining_slots = max(0, int(primary_budget) - primary_values.size)
    if remaining_slots > 0 and near_values.size:
        near_candidates = near_values[~np.isin(near_values, primary_values)]
        near_candidates = thin_gamma_candidates_by_gap(
            near_candidates,
            protected_values=primary_values,
            objective_function=objective_function,
        )
        primary_values = np.sort(
            np.unique(
                np.concatenate(
                    [primary_values, select_evenly_spaced_gamma_values(near_candidates, remaining_slots)]
                )
            )
        )

    remaining_slots = max(0, int(primary_budget) - primary_values.size)
    if remaining_slots > 0 and generic_seed_values.size:
        generic_candidates = generic_seed_values[~np.isin(generic_seed_values, primary_values)]
        generic_candidates = thin_gamma_candidates_by_gap(
            generic_candidates,
            protected_values=primary_values,
            objective_function=objective_function,
        )
        primary_values = np.sort(
            np.unique(
                np.concatenate(
                    [primary_values, select_evenly_spaced_gamma_values(generic_candidates, remaining_slots)]
                )
            )
        )

    primary_values = fill_gamma_values_to_budget(
        primary_values,
        int(primary_budget),
        gamma_range,
        objective_function,
    )
    primary_values = primary_values[
        (primary_values >= min(gamma_range)) & (primary_values <= max(gamma_range))
    ]
    secondary_values = build_secondary_gamma_points(
        primary_values,
        gamma_range,
        objective_function,
        int(secondary_budget),
    )
    return {
        "primary_gammas": np.sort(np.unique(primary_values)),
        "secondary_gammas": np.sort(np.unique(secondary_values)),
        "seed_table": seed_table,
    }


def derive_gamma_admission_state(
    gamma_results: list[dict[str, Any]],
    target_clusters: int,
    min_cluster_size: int = 1,
    verbose: bool = False,
    worker_id: str = "OPTIMIZER",
) -> dict[str, Any]:
    """Summarize which gamma candidates are admissible after Phase 1 evaluation."""
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
        exact_hit_flags=hit_counts > 0,
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


def should_expand_phase1_secondary(
    valid_indices: list[int],
    admission_mode: str,
    exact_hit_gamma_count: int,
) -> bool:
    """Decide whether Phase 1 should evaluate the secondary gamma batch."""
    guarded_modes = {"raw_strict_soft", "strict_soft", "relaxed_soft", "strict_hard", "relaxed_hard"}
    if not valid_indices:
        return True
    if exact_hit_gamma_count > 0:
        return False
    if admission_mode in guarded_modes:
        return False
    return admission_mode in {"relaxed_unguarded", "raw_relaxed_unguarded"}


def should_skip_phase4_refinement(
    candidate_count: int,
    best_ic: float,
    exact_hit_gamma_count: int,
) -> bool:
    """Skip expensive Phase 4 refinement when the admitted frontier is already decisive."""
    return (
        int(candidate_count) <= 2
        and np.isfinite(best_ic)
        and best_ic <= 1.005
        and int(exact_hit_gamma_count) > 0
    )


def phase4_iteration_cap_for_mode(admission_mode: str) -> int:
    """Cap refinement depth based on how weak the Phase 1 admission mode was."""
    return 2 if admission_mode in {"relaxed_unguarded", "raw_relaxed_unguarded"} else 3


def preferred_trial_flags(
    preferred_trials: list[list[int]] | list[tuple[int, ...]] | None,
    size: int | None = None,
) -> np.ndarray:
    """Convert preferred-trial index groups into a boolean mask."""
    if preferred_trials is None:
        return np.zeros(0 if size is None else int(size), dtype=bool)
    flags = np.asarray([bool(trials) for trials in preferred_trials], dtype=bool)
    if size is not None and flags.size != int(size):
        raise ValueError("preferred_trials length must match the requested size.")
    return flags


def order_gamma_candidate_indices(
    results: list[dict[str, Any]],
    target_clusters: int,
    *,
    exact_support_flags: np.ndarray | None = None,
    prefer_right_exact_hits: bool = False,
    finalizable_flags: np.ndarray | None = None,
) -> list[int]:
    """Order gamma candidates so the most promising finalization targets are visited first."""
    if not results:
        return []
    if exact_support_flags is None:
        exact_support_flags = np.zeros(len(results), dtype=bool)
    else:
        exact_support_flags = np.asarray(exact_support_flags, dtype=bool)
        if exact_support_flags.size != len(results):
            raise ValueError("exact_support_flags must match the number of candidate results.")
    if finalizable_flags is None:
        finalizable_flags = np.zeros(len(results), dtype=bool)
    else:
        finalizable_flags = np.asarray(finalizable_flags, dtype=bool)
        if finalizable_flags.size != len(results):
            raise ValueError("finalizable_flags must match the number of candidate results.")

    primary_pool = np.where(finalizable_flags)[0].tolist()
    primary_pool_set = set(primary_pool)
    secondary_pool = [idx for idx in range(len(results)) if idx not in primary_pool_set]

    primary_order = sorted(
        primary_pool,
        key=lambda idx: gamma_candidate_sort_key(
            results[idx],
            target_clusters,
            exact_support=False,
            prefer_right_exact_hits=False,
        ),
    )
    secondary_order = sorted(
        secondary_pool,
        key=lambda idx: gamma_candidate_sort_key(
            results[idx],
            target_clusters,
            exact_support=bool(exact_support_flags[idx]),
            prefer_right_exact_hits=prefer_right_exact_hits,
        ),
    )
    return [*primary_order, *secondary_order]
