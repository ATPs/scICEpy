"""Input normalization and graph-loading helpers for the public scICE entry point."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .leiden_wrapper import graph_to_igraph
from .runtime import logger, summarize_adjacency_matrix

def _normalize_cluster_range(cluster_range: Any) -> np.ndarray:
    """Normalize the requested cluster range into a sorted unique integer array."""
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
    """Normalize manual resolution inputs into an ordered unique float array."""
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
    """Validate the common public-entry inputs before any clustering work begins."""
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
    """Load the requested adjacency matrix and convert it to igraph, with optional logging."""
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

def _format_cluster_values(values: Any) -> str:
    """Render a cluster-count or resolution array as a compact comma-separated string."""
    if values is None:
        return "none"
    arr = np.asarray(values)
    if arr.size == 0:
        return "none"
    return ", ".join(map(str, arr.tolist()))
