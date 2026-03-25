"""Shared cluster-label utilities used across search and optimization code."""

from __future__ import annotations

from typing import Any

import numpy as np


def reindex_cluster_labels(labels: np.ndarray) -> np.ndarray:
    """Remap arbitrary cluster ids to a compact zero-based integer range."""
    labels = np.asarray(labels, dtype=np.int32)
    if labels.size == 0:
        return labels
    unique_ids, inverse = np.unique(labels, return_inverse=True)
    expected = np.arange(unique_ids.size, dtype=np.int32)
    if np.array_equal(unique_ids, expected):
        return labels
    return inverse.astype(np.int32, copy=False)


def summarize_cluster_labels(labels: np.ndarray, min_cluster_size: int = 1) -> dict[str, int]:
    """Report raw and effective cluster counts for one clustering vector."""
    labels = np.asarray(labels, dtype=np.int32)
    min_cluster_size = max(1, int(min_cluster_size))
    if labels.size == 0:
        return {"raw_cluster_count": 0, "effective_cluster_count": 0}
    _, counts = np.unique(labels, return_counts=True)
    raw_cluster_count = int(counts.size)
    if min_cluster_size <= 1:
        return {
            "raw_cluster_count": raw_cluster_count,
            "effective_cluster_count": raw_cluster_count,
        }
    return {
        "raw_cluster_count": raw_cluster_count,
        "effective_cluster_count": int(np.sum(counts >= min_cluster_size)),
    }


def count_effective_clusters(labels: np.ndarray, min_cluster_size: int = 1) -> int:
    """Return the number of clusters that satisfy the minimum size constraint."""
    return int(summarize_cluster_labels(labels, min_cluster_size=min_cluster_size)["effective_cluster_count"])


def raw_cluster_guard_limits(target_clusters: int) -> dict[str, int]:
    """Return soft and hard upper bounds for acceptable raw cluster over-fragmentation."""
    target_clusters = max(1, int(target_clusters))
    return {
        "soft": int(max(target_clusters + 3, np.ceil(target_clusters * 1.1))),
        "hard": int(max(target_clusters + 5, np.ceil(target_clusters * 1.5))),
    }


def raw_cluster_search_upper(target_clusters: int) -> int:
    """Return the raw-cluster ceiling used during coarse resolution search."""
    target_clusters = max(1, int(target_clusters))
    if target_clusters <= 10:
        return int(target_clusters + 1)
    return int(max(target_clusters + 2, np.ceil(target_clusters * 1.05)))


def passes_raw_cluster_guard(
    raw_cluster_median: float | np.ndarray,
    target_clusters: int,
    min_cluster_size: int = 1,
    level: str = "soft",
) -> np.ndarray | bool:
    """Check whether raw cluster counts stay within the configured over-fragmentation guard."""
    min_cluster_size = max(1, int(min_cluster_size))
    if min_cluster_size <= 1 or int(target_clusters) <= 1:
        values = np.asarray(raw_cluster_median)
        mask = np.ones_like(values, dtype=bool)
        return bool(mask.item()) if mask.ndim == 0 else mask

    limits = raw_cluster_guard_limits(target_clusters)
    upper_bound = limits["soft" if level == "soft" else "hard"]
    values = np.asarray(raw_cluster_median, dtype=float)
    mask = np.isfinite(values) & (values <= upper_bound)
    return bool(mask.item()) if mask.ndim == 0 else mask


def classify_resolution_search_state(
    raw_cluster_median: float,
    effective_cluster_median: float,
    target_clusters: int,
    min_cluster_size: int = 1,
) -> dict[str, Any]:
    """Classify one search probe to decide how the next resolution interval should move."""
    raw_guard_soft = bool(
        passes_raw_cluster_guard(
            raw_cluster_median,
            target_clusters,
            min_cluster_size=min_cluster_size,
            level="soft",
        )
    )
    raw_guard_search = True if min_cluster_size <= 1 or int(target_clusters) <= 1 else bool(
        np.isfinite(raw_cluster_median) and raw_cluster_median <= raw_cluster_search_upper(target_clusters)
    )
    raw_below = bool(np.isfinite(raw_cluster_median) and raw_cluster_median < target_clusters)
    raw_above_soft = (not raw_below) and (not raw_guard_search)
    raw_in_band = (not raw_below) and (not raw_above_soft)
    effective_meets_target = bool(
        np.isfinite(effective_cluster_median) and effective_cluster_median >= target_clusters
    )
    over_fragmented = (not raw_below) and (not effective_meets_target)
    raw_class = "raw_below" if raw_below else ("raw_above_soft" if raw_above_soft else "raw_in_band")
    lower_action = "increase_gamma" if raw_below else "decrease_gamma"
    upper_action = "increase_gamma" if raw_below or (raw_in_band and effective_meets_target) else "decrease_gamma"
    return {
        "raw_class": raw_class,
        "raw_below": raw_below,
        "raw_in_band": raw_in_band,
        "raw_above_soft": raw_above_soft,
        "raw_guard_soft": raw_guard_soft,
        "raw_guard_search": raw_guard_search,
        "effective_meets_target": effective_meets_target,
        "over_fragmented": over_fragmented,
        "lower_action": lower_action,
        "upper_action": upper_action,
    }


def merge_small_clusters_to_neighbors(
    labels: np.ndarray,
    snn_graph,
    min_cluster_size: int = 1,
) -> np.ndarray:
    """Merge undersized clusters into their best-connected large neighbors on the SNN graph."""
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
        merged[base_labels == small_id] = best_target
    return reindex_cluster_labels(merged)
