"""Metrics for scICEpy."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


def _prepare_label_pairs(labels_a: np.ndarray, labels_b: np.ndarray) -> dict[str, np.ndarray]:
    labels_a = np.asarray(labels_a, dtype=np.int32)
    labels_b = np.asarray(labels_b, dtype=np.int32)
    if labels_a.ndim != 1 or labels_b.ndim != 1 or labels_a.size == 0 or labels_a.size != labels_b.size:
        raise ValueError("labels_a and labels_b must be non-empty 1D arrays of the same length.")

    _, inverse_a, counts_a = np.unique(labels_a, return_inverse=True, return_counts=True)
    _, inverse_b, counts_b = np.unique(labels_b, return_inverse=True, return_counts=True)
    n_b = int(len(counts_b))
    pair_codes = inverse_a.astype(np.int64) * max(1, n_b) + inverse_b.astype(np.int64)
    unique_pairs, pair_inverse, pair_counts = np.unique(
        pair_codes,
        return_inverse=True,
        return_counts=True,
    )
    pair_a = (unique_pairs // max(1, n_b)).astype(np.int32, copy=False)
    pair_b = (unique_pairs % max(1, n_b)).astype(np.int32, copy=False)
    return {
        "pair_inverse": pair_inverse.astype(np.int32, copy=False),
        "pair_counts": pair_counts.astype(float, copy=False),
        "pair_a": pair_a,
        "pair_b": pair_b,
        "counts_a": counts_a.astype(float, copy=False),
        "counts_b": counts_b.astype(float, copy=False),
    }


def calculate_ecs(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    d: float = 0.9,
    return_vector: bool = False,
) -> float | np.ndarray:
    d = float(d)
    if not np.isfinite(d) or d <= 0:
        raise ValueError("d must be a finite positive value.")

    prepared = _prepare_label_pairs(labels_a, labels_b)
    pair_counts = prepared["pair_counts"]
    pair_a = prepared["pair_a"]
    pair_b = prepared["pair_b"]
    counts_a = prepared["counts_a"][pair_a]
    counts_b = prepared["counts_b"][pair_b]
    c_size_a = d / counts_a
    c_size_b = d / counts_b
    escore = (
        (counts_a - pair_counts) * c_size_a
        + (counts_b - pair_counts) * c_size_b
        + pair_counts * np.abs(c_size_a - c_size_b)
    )
    pair_similarity = np.clip(1.0 - (1.0 / (2.0 * d)) * escore, 0.0, 1.0)
    similarity_vector = pair_similarity[prepared["pair_inverse"]]
    if return_vector:
        return similarity_vector.astype(float, copy=False)
    return float(np.mean(similarity_vector))


def extract_clustering_array(clustering_matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(clustering_matrix, dtype=np.int32)
    if matrix.ndim != 2:
        raise ValueError("clustering_matrix must be a 2D array with trials on rows.")

    unique_rows, counts = np.unique(matrix, axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    arr = [np.asarray(unique_rows[idx], dtype=np.int32) for idx in order.tolist()]
    ordered_counts = counts[order].astype(float, copy=False)
    prob = ordered_counts / ordered_counts.sum()
    return {"arr": arr, "prob": prob, "parr": prob}


def _pairwise_similarity_matrix(extracted: dict[str, Any]) -> np.ndarray:
    cached = extracted.get("_pairwise_similarity_matrix")
    if isinstance(cached, np.ndarray) and cached.ndim == 2:
        return np.asarray(cached, dtype=float)

    clusterings = extracted["arr"]
    n_clusterings = len(clusterings)
    similarities = np.eye(n_clusterings, dtype=float)
    for i, j in combinations(range(n_clusterings), 2):
        sim = float(calculate_ecs(clusterings[i], clusterings[j]))
        similarities[i, j] = sim
        similarities[j, i] = sim
    extracted["_pairwise_similarity_matrix"] = similarities
    return similarities


def calculate_ic_from_extracted(extracted: dict[str, Any], n_workers: int = 1) -> float:
    del n_workers
    clusterings = extracted["arr"]
    prob_arr = np.asarray(extracted.get("prob", extracted.get("parr")), dtype=float)
    if len(clusterings) == 1:
        return 1.0

    similarities = _pairwise_similarity_matrix(extracted)
    consistency = float(np.dot(similarities @ prob_arr, prob_arr))
    return float(1.0 / consistency) if consistency > 0 else float("inf")


def get_best_clustering(extracted: dict[str, Any]) -> np.ndarray:
    clusterings = extracted["arr"]
    if len(clusterings) == 1:
        return np.asarray(clusterings[0], dtype=np.int32)

    similarities = _pairwise_similarity_matrix(extracted)
    best_idx = int(np.argmax(similarities.sum(axis=1)))
    return np.asarray(clusterings[best_idx], dtype=np.int32)


def calculate_mei_from_array(extracted: dict[str, Any], n_workers: int = 1) -> np.ndarray:
    del n_workers
    clusterings = extracted["arr"]
    prob_arr = np.asarray(extracted.get("prob", extracted.get("parr")), dtype=float)
    if len(clusterings) == 1:
        return np.ones_like(clusterings[0], dtype=float)

    n_elements = len(clusterings[0])
    weighted_scores = np.zeros(n_elements, dtype=float)
    total_weight = 0.0
    for i, j in combinations(range(len(clusterings)), 2):
        pair_weight = 2.0 * prob_arr[i] * prob_arr[j]
        if pair_weight <= 0:
            continue
        weighted_scores += pair_weight * np.asarray(
            calculate_ecs(clusterings[i], clusterings[j], return_vector=True),
            dtype=float,
        )
        total_weight += pair_weight

    if total_weight <= 0:
        return np.ones(n_elements, dtype=float)
    return np.clip(weighted_scores / total_weight, 0.0, 1.0)
