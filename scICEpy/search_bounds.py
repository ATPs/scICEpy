"""Resolution search helpers for scICEpy."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import pandas as pd

from .leiden_wrapper import cached_leiden_clustering
from .runtime import (
    cap_workers_by_memory,
    estimate_trial_matrix_bytes,
    get_parallel_context,
    initialize_parallel_state,
    logger,
    parallel_map_threads,
)

_SEARCH_PROBE_STATE: dict[str, Any] = {}



def build_gamma_sequence_for_range(
    gamma_range: tuple[float, float],
    objective_function: str,
    resolution_tolerance: float,
    n_vertices: int | None = None,
    n_steps: int | None = None,
) -> np.ndarray:
    lower, upper = sorted((float(gamma_range[0]), float(gamma_range[1])))
    if n_steps is None:
        range_width = abs(upper - lower)
        n_steps = 5 if n_vertices is not None and n_vertices >= 200000 and range_width <= max(resolution_tolerance * 10, np.finfo(float).eps) else 11
    n_steps = max(2, int(n_steps))

    if objective_function == "modularity":
        if abs(upper - lower) > resolution_tolerance:
            return np.linspace(lower, upper, n_steps)
        return np.linspace(lower - resolution_tolerance, lower + resolution_tolerance, n_steps)

    lower = max(lower, np.finfo(float).tiny)
    upper = max(upper, np.finfo(float).tiny)
    if abs(upper - lower) > max(resolution_tolerance, lower * 1e-6):
        return np.exp(np.linspace(np.log(lower), np.log(upper), n_steps))
    delta_log = max(resolution_tolerance, 1e-4)
    return np.exp(np.linspace(np.log(lower) - delta_log, np.log(lower) + delta_log, n_steps))

def stabilize_probe_raw_medians(raw_cluster_medians: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_cluster_medians, dtype=float)
    if values.size <= 1:
        return values
    finite_mask = np.isfinite(values)
    values[finite_mask] = np.maximum.accumulate(values[finite_mask])
    return values

def stabilize_monotone_probe_counts(values: np.ndarray) -> np.ndarray:
    return stabilize_probe_raw_medians(values)

def clamp_gamma_range_to_raw_plateau(
    gamma_sequence: np.ndarray,
    raw_cluster_medians: np.ndarray,
    target_clusters: int,
) -> dict[str, Any]:
    gamma_sequence = np.asarray(gamma_sequence, dtype=float)
    raw_cluster_medians = np.asarray(raw_cluster_medians, dtype=float)
    if gamma_sequence.size == 0 or gamma_sequence.size != raw_cluster_medians.size:
        raise ValueError("gamma_sequence and raw_cluster_medians must have the same non-zero length.")

    stabilized = stabilize_probe_raw_medians(raw_cluster_medians)
    exact_indices = np.where(stabilized == int(target_clusters))[0]
    if exact_indices.size:
        return {
            "bounds": np.asarray([gamma_sequence[exact_indices.min()], gamma_sequence[exact_indices.max()]], dtype=float),
            "mode": "raw_exact",
            "indices": exact_indices + 1,
            "stabilized_raw_medians": stabilized,
        }

    left_raw = stabilized[:-1]
    right_raw = stabilized[1:]
    crossing_indices = np.where(
        ((left_raw < target_clusters) & (right_raw > target_clusters))
        | ((left_raw > target_clusters) & (right_raw < target_clusters))
    )[0]
    if crossing_indices.size:
        widths = np.abs(gamma_sequence[crossing_indices + 1] - gamma_sequence[crossing_indices])
        best_pair_idx = int(crossing_indices[np.argmin(widths)])
        return {
            "bounds": np.sort(gamma_sequence[[best_pair_idx, best_pair_idx + 1]]).astype(float),
            "mode": "raw_bracket",
            "indices": np.asarray([best_pair_idx + 1, best_pair_idx + 2], dtype=int),
            "stabilized_raw_medians": stabilized,
        }

    near_target_indices = np.where(np.abs(stabilized - target_clusters) <= 1)[0]
    if near_target_indices.size:
        return {
            "bounds": np.asarray(
                [gamma_sequence[near_target_indices.min()], gamma_sequence[near_target_indices.max()]],
                dtype=float,
            ),
            "mode": "raw_near_target",
            "indices": near_target_indices + 1,
            "stabilized_raw_medians": stabilized,
        }

    return {
        "bounds": np.asarray([gamma_sequence.min(), gamma_sequence.max()], dtype=float),
        "mode": "coarse",
        "indices": np.arange(1, gamma_sequence.size + 1, dtype=int),
        "stabilized_raw_medians": stabilized,
    }

def global_resolution_search_midpoint(left: float, right: float, objective_function: str) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    if objective_function == "CPM":
        return float(np.exp((np.log(left) + np.log(right)) / 2.0))
    return float((left + right) / 2.0)

def global_resolution_search_interval_small(
    left: float,
    right: float,
    objective_function: str,
    resolution_tolerance: float,
) -> bool:
    if not np.isfinite(left) or not np.isfinite(right) or left >= right:
        return True
    tolerance = max(resolution_tolerance, np.finfo(float).eps * 100)
    if objective_function == "CPM":
        return bool(abs(np.log(right) - np.log(left)) <= tolerance)
    return bool(abs(right - left) <= tolerance)

def global_resolution_search_interval_width(left: float, right: float, objective_function: str) -> float:
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
    n_points = max(0, int(n_points))
    if n_points <= 0 or not np.isfinite(left) or not np.isfinite(right) or left >= right:
        return np.asarray([], dtype=float)
    fractions = np.arange(1, n_points + 1, dtype=float) / float(n_points + 1)
    if objective_function == "CPM":
        return np.exp(np.log(left) + fractions * (np.log(right) - np.log(left)))
    return left + fractions * (right - left)

def build_refinement_probe_plan(
    unresolved_intervals: dict[int, tuple[float, float]],
    objective_function: str,
    resolution_tolerance: float,
    active_probe_workers: int,
    existing_gamma_values: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    existing = np.asarray(existing_gamma_values if existing_gamma_values is not None else [], dtype=float)
    interval_rows: list[dict[str, Any]] = []
    for target, interval in unresolved_intervals.items():
        left, right = sorted((float(interval[0]), float(interval[1])))
        if global_resolution_search_interval_small(left, right, objective_function, resolution_tolerance):
            continue
        interval_rows.append(
            {
                "refinement_interval_id": int(target),
                "gamma_left": left,
                "gamma_right": right,
                "refinement_interval_width": global_resolution_search_interval_width(left, right, objective_function),
            }
        )

    intervals_dt = pd.DataFrame(interval_rows)
    if intervals_dt.empty:
        return {"probe_metadata": pd.DataFrame(), "interval_summary": pd.DataFrame()}

    intervals_dt = intervals_dt.sort_values(["refinement_interval_width", "refinement_interval_id"], ascending=[False, True]).reset_index(drop=True)
    intervals_dt["refinement_points_per_interval"] = 1
    n_intervals = len(intervals_dt)
    if n_intervals < active_probe_workers:
        max_points_per_interval = min(8, max(1, active_probe_workers // n_intervals))
        remaining_points = min(max(0, active_probe_workers - n_intervals), n_intervals * (max_points_per_interval - 1))
        while remaining_points > 0 and (intervals_dt["refinement_points_per_interval"] < max_points_per_interval).any():
            for idx in intervals_dt.index:
                if remaining_points <= 0:
                    break
                if int(intervals_dt.at[idx, "refinement_points_per_interval"]) < max_points_per_interval:
                    intervals_dt.at[idx, "refinement_points_per_interval"] += 1
                    remaining_points -= 1

    probe_rows: list[dict[str, Any]] = []
    for row in intervals_dt.itertuples(index=False):
        gammas = global_resolution_search_internal_points(
            row.gamma_left,
            row.gamma_right,
            objective_function,
            int(row.refinement_points_per_interval),
        )
        for gamma in gammas:
            probe_rows.append(
                {
                    "gamma": float(gamma),
                    "refinement_interval_id": int(row.refinement_interval_id),
                    "refinement_interval_width": float(row.refinement_interval_width),
                    "refinement_points_per_interval": int(row.refinement_points_per_interval),
                }
            )
    probe_dt = pd.DataFrame(probe_rows)
    if probe_dt.empty:
        return {"probe_metadata": pd.DataFrame(), "interval_summary": intervals_dt}

    probe_dt = probe_dt.drop_duplicates(subset=["gamma"]).sort_values("gamma").reset_index(drop=True)
    if existing.size:
        probe_dt = probe_dt[~probe_dt["gamma"].isin(existing)].reset_index(drop=True)
    return {"probe_metadata": probe_dt, "interval_summary": intervals_dt}

def derive_shared_gamma_intervals(
    probes_df: pd.DataFrame,
    cluster_range: np.ndarray,
    gamma_bounds: tuple[float, float],
    objective_function: str,
) -> dict[str, Any]:
    if probes_df.empty:
        return {
            "gamma_dict": {},
            "optimization_ready_targets": [],
            "unresolved_targets": list(map(int, cluster_range)),
            "unresolved_intervals": {int(k): tuple(sorted(gamma_bounds)) for k in cluster_range},
            "selected_gamma_values": np.asarray([], dtype=float),
            "target_gamma_seeds": {str(int(k)): [] for k in cluster_range},
            "target_interval_details": {str(int(k)): {"mode": "missing"} for k in cluster_range},
            "annotated_probes_df": probes_df.copy(),
        }

    probes_df = probes_df.sort_values("gamma").reset_index(drop=True)
    gamma_values = probes_df["gamma"].to_numpy(dtype=float)
    final_counts = stabilize_monotone_probe_counts(probes_df["final_cluster_count"].to_numpy(dtype=float))
    raw_counts = stabilize_monotone_probe_counts(probes_df["raw_cluster_count"].to_numpy(dtype=float))
    probes_df = probes_df.copy()
    probes_df["stabilized_final_cluster_count"] = final_counts
    probes_df["stabilized_raw_cluster_count"] = raw_counts

    gamma_dict: dict[int, tuple[float, float]] = {}
    optimization_ready_targets: list[int] = []
    unresolved_targets: list[int] = []
    unresolved_intervals: dict[int, tuple[float, float]] = {}
    selected_gamma_values: list[float] = []
    target_gamma_seeds: dict[str, list[float]] = {}
    target_interval_details: dict[str, dict[str, Any]] = {}
    final_exact_targets = [set() for _ in range(len(probes_df))]
    final_near_targets = [set() for _ in range(len(probes_df))]
    final_bracket_targets = [set() for _ in range(len(probes_df))]
    raw_exact_targets = [set() for _ in range(len(probes_df))]
    raw_near_targets = [set() for _ in range(len(probes_df))]
    raw_bracket_targets = [set() for _ in range(len(probes_df))]

    upper_tail_threshold = int(np.max(np.asarray(cluster_range, dtype=int))) - 3

    def _expand_interval_indices(indices: np.ndarray, left_steps: int = 0, right_steps: int = 0) -> np.ndarray:
        indices = np.asarray(indices, dtype=int)
        if indices.size == 0:
            return indices
        expanded = set(indices.tolist())
        left_idx = int(indices.min())
        right_idx = int(indices.max())
        for step in range(1, max(0, int(left_steps)) + 1):
            candidate = left_idx - step
            if candidate >= 0:
                expanded.add(candidate)
        for step in range(1, max(0, int(right_steps)) + 1):
            candidate = right_idx + step
            if candidate < gamma_values.size:
                expanded.add(candidate)
        return np.asarray(sorted(expanded), dtype=int)

    def _expand_exact_interval(indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=int)
        if indices.size == 0:
            return indices
        if indices.size > 1:
            return indices
        expanded = indices.tolist()
        idx = int(indices[0])
        if idx - 1 >= 0:
            expanded.append(idx - 1)
        if idx + 1 < len(gamma_values):
            expanded.append(idx + 1)
        return np.asarray(sorted(set(expanded)), dtype=int)

    def _find_bracket_indices(counts: np.ndarray, target: int) -> np.ndarray:
        below_indices = np.where(counts < target)[0]
        above_indices = np.where(counts > target)[0]
        if below_indices.size and above_indices.size:
            left_idx = int(below_indices.max())
            right_idx = int(above_indices.min())
            if left_idx < right_idx:
                return np.asarray([left_idx, right_idx], dtype=int)
        return np.asarray([], dtype=int)

    def _combine_bounds(
        primary_bounds: tuple[float, float] | None,
        secondary_bounds: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if primary_bounds is None:
            return secondary_bounds
        if secondary_bounds is None:
            return primary_bounds
        left = max(float(primary_bounds[0]), float(secondary_bounds[0]))
        right = min(float(primary_bounds[1]), float(secondary_bounds[1]))
        if np.isfinite(left) and np.isfinite(right) and left < right:
            return (left, right)
        return (
            float(min(primary_bounds[0], secondary_bounds[0])),
            float(max(primary_bounds[1], secondary_bounds[1])),
        )

    for target in np.asarray(cluster_range, dtype=int):
        key = str(int(target))
        upper_tail_target = bool(int(target) >= upper_tail_threshold)
        exact_idx = np.where(final_counts == int(target))[0]
        near_idx = np.where(np.abs(final_counts - int(target)) <= 1)[0]
        final_bracket_idx = _find_bracket_indices(final_counts, int(target))
        raw_state = clamp_gamma_range_to_raw_plateau(gamma_values, raw_counts, target_clusters=int(target))
        raw_state_mode = str(raw_state.get("mode", "coarse"))
        raw_idx = np.asarray(raw_state.get("indices", []), dtype=int)
        raw_idx = raw_idx - 1 if raw_idx.size else raw_idx
        raw_idx = raw_idx[(raw_idx >= 0) & (raw_idx < gamma_values.size)]
        raw_exact_idx = np.where(raw_counts == int(target))[0]
        raw_near_idx = np.where(np.abs(raw_counts - int(target)) <= 1)[0]
        mode = "missing"
        bounds: tuple[float, float] | None = None
        seed_values: list[float] = []
        exact_probe_values: list[float] = []
        near_probe_values: list[float] = []
        raw_exact_probe_values = gamma_values[raw_exact_idx].astype(float).tolist()
        raw_near_probe_values = gamma_values[raw_near_idx].astype(float).tolist()
        raw_bracket_probe_values = gamma_values[raw_idx].astype(float).tolist() if raw_state_mode == "raw_bracket" else []
        interval_indices = np.asarray([], dtype=int)
        bracketed = False
        optimization_ready = False

        if exact_idx.size:
            interval_indices = _expand_exact_interval(exact_idx)
            if upper_tail_target:
                interval_indices = _expand_interval_indices(interval_indices, right_steps=1)
            bounds = (float(gamma_values[interval_indices.min()]), float(gamma_values[interval_indices.max()]))
            exact_probe_values = gamma_values[exact_idx].astype(float).tolist()
            seed_values = gamma_values[np.unique(np.concatenate([interval_indices, exact_idx, near_idx, raw_idx]))].astype(float).tolist()
            mode = "final_exact"
            bracketed = bool(np.isfinite(bounds[0]) and np.isfinite(bounds[1]) and bounds[0] < bounds[1])
            optimization_ready = bracketed
        elif final_bracket_idx.size:
            interval_indices = final_bracket_idx
            if upper_tail_target:
                interval_indices = _expand_interval_indices(interval_indices, right_steps=1)
            bounds = tuple(sorted((float(gamma_values[final_bracket_idx[0]]), float(gamma_values[final_bracket_idx[1]]))))
            if interval_indices.size:
                bounds = (float(gamma_values[interval_indices.min()]), float(gamma_values[interval_indices.max()]))
            seed_values = gamma_values[np.unique(np.concatenate([interval_indices, near_idx, raw_idx]))].astype(float).tolist()
            midpoint = global_resolution_search_midpoint(bounds[0], bounds[1], objective_function)
            if np.isfinite(midpoint):
                seed_values.append(float(midpoint))
            mode = "final_bracket"
            bracketed = True
            optimization_ready = True
        elif near_idx.size:
            near_bounds = (float(gamma_values[near_idx.min()]), float(gamma_values[near_idx.max()]))
            near_probe_values = gamma_values[near_idx].astype(float).tolist()
            raw_bounds = None
            if np.all(np.isfinite(raw_state.get("bounds", np.asarray([np.nan, np.nan], dtype=float)))):
                raw_bounds_arr = np.asarray(raw_state["bounds"], dtype=float)
                raw_bounds = (float(raw_bounds_arr[0]), float(raw_bounds_arr[1]))
            bounds = _combine_bounds(near_bounds, raw_bounds if raw_state_mode != "coarse" else None)
            if bounds is not None:
                interval_indices = np.asarray(sorted(set([*near_idx.tolist(), *raw_idx.tolist()])), dtype=int)
                if upper_tail_target:
                    interval_indices = _expand_interval_indices(interval_indices, right_steps=1)
                    bounds = (
                        float(min(bounds[0], gamma_values[interval_indices.min()])),
                        float(max(bounds[1], gamma_values[interval_indices.max()])),
                    )
                seed_values = gamma_values[np.unique(np.concatenate([interval_indices]))].astype(float).tolist()
                midpoint = global_resolution_search_midpoint(bounds[0], bounds[1], objective_function)
                if np.isfinite(midpoint):
                    seed_values.append(float(midpoint))
                mode = f"final_near_{raw_state_mode}" if raw_state_mode != "coarse" else "final_near"
                bracketed = bool(np.isfinite(bounds[0]) and np.isfinite(bounds[1]) and bounds[0] < bounds[1])
                optimization_ready = bracketed

        if near_idx.size and not near_probe_values:
            near_probe_values = gamma_values[near_idx].astype(float).tolist()

        for idx in exact_idx.tolist():
            final_exact_targets[int(idx)].add(int(target))
        for idx in near_idx.tolist():
            final_near_targets[int(idx)].add(int(target))
        for idx in final_bracket_idx.tolist():
            final_bracket_targets[int(idx)].add(int(target))
        for idx in raw_exact_idx.tolist():
            raw_exact_targets[int(idx)].add(int(target))
        for idx in raw_near_idx.tolist():
            raw_near_targets[int(idx)].add(int(target))
        for idx in raw_idx.tolist():
            raw_bracket_targets[int(idx)].add(int(target))

        if bounds is not None and optimization_ready:
            gamma_dict[int(target)] = (float(bounds[0]), float(bounds[1]))
            optimization_ready_targets.append(int(target))
            selected_gamma_values.extend([float(bounds[0]), float(bounds[1]), *seed_values])
        else:
            unresolved_targets.append(int(target))
            below = np.where(final_counts < target)[0]
            above = np.where(final_counts > target)[0]
            left = float(gamma_values[below.max()]) if below.size else float(min(gamma_bounds))
            right = float(gamma_values[above.min()]) if above.size else float(max(gamma_bounds))
            if raw_idx.size:
                raw_left = float(np.min(gamma_values[raw_idx]))
                raw_right = float(np.max(gamma_values[raw_idx]))
                left = min(left, raw_left)
                right = max(right, raw_right)
            if left >= right:
                left, right = float(min(gamma_bounds)), float(max(gamma_bounds))
            unresolved_intervals[int(target)] = (left, right)

        target_gamma_seeds[key] = sorted(set(map(float, seed_values)))
        target_interval_details[key] = {
            "mode": mode,
            "bracketed": bool(bracketed),
            "optimization_ready": bool(optimization_ready),
            "has_exact_probe": bool(exact_idx.size > 0),
            "gamma_left": np.nan if bounds is None else float(bounds[0]),
            "gamma_right": np.nan if bounds is None else float(bounds[1]),
            "seed_gamma_values": sorted(set(map(float, seed_values))),
            "exact_probe_values": exact_probe_values,
            "near_probe_values": near_probe_values,
            "final_bracket_probe_values": gamma_values[final_bracket_idx].astype(float).tolist(),
            "raw_interval_mode": raw_state_mode,
            "raw_exact_probe_values": raw_exact_probe_values,
            "raw_near_probe_values": raw_near_probe_values,
            "raw_bracket_probe_values": raw_bracket_probe_values,
            "raw_cluster_count": np.nan if near_idx.size == 0 else float(np.nanmedian(raw_counts[near_idx])),
        }

    def _serialize_target_sets(values: list[set[int]]) -> list[str]:
        return [",".join(map(str, sorted(item))) if item else "" for item in values]

    probes_df["final_exact_targets"] = _serialize_target_sets(final_exact_targets)
    probes_df["final_near_targets"] = _serialize_target_sets(final_near_targets)
    probes_df["final_bracket_targets"] = _serialize_target_sets(final_bracket_targets)
    probes_df["raw_exact_targets"] = _serialize_target_sets(raw_exact_targets)
    probes_df["raw_near_targets"] = _serialize_target_sets(raw_near_targets)
    probes_df["raw_bracket_targets"] = _serialize_target_sets(raw_bracket_targets)

    return {
        "gamma_dict": gamma_dict,
        "optimization_ready_targets": optimization_ready_targets,
        "unresolved_targets": unresolved_targets,
        "unresolved_intervals": unresolved_intervals,
        "selected_gamma_values": np.asarray(sorted(set(selected_gamma_values)), dtype=float),
        "target_gamma_seeds": target_gamma_seeds,
        "target_interval_details": target_interval_details,
        "annotated_probes_df": probes_df,
    }
