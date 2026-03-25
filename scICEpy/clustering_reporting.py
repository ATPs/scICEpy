"""Result reporting helpers for the public scICE clustering entry point."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .clustering_inputs import _format_cluster_values
from .runtime import logger


def _safe_len(value: Any) -> int:
    """Return `len(value)` as an integer, falling back to zero for missing or scalar inputs."""
    if value is None:
        return 0
    try:
        return int(len(value))
    except TypeError:
        return 0


def _log_results_summary(
    results: dict[str, Any],
    resolution_mode: bool,
    requested_cluster_range: np.ndarray | None,
    resolution_values: np.ndarray | None,
    ic_threshold: float,
    total_time: float,
) -> None:
    """Log a detailed summary of the final scICE result payload."""
    logger.info("-" * 80)
    logger.info("RESULTS ANALYSIS:")
    logger.info("  IC threshold for consistency: %s", ic_threshold)
    logger.info("  Results structure:")
    logger.info("    - gamma length: %s", _safe_len(results.get("gamma")))
    logger.info("    - labels length: %s", _safe_len(results.get("labels")))
    logger.info("    - ic length: %s", _safe_len(results.get("ic")))
    logger.info("    - n_cluster length: %s", _safe_len(results.get("n_cluster")))
    logger.info(
        "    - effective_cluster_median length: %s",
        _safe_len(results.get("effective_cluster_median")),
    )
    logger.info("    - raw_cluster_median length: %s", _safe_len(results.get("raw_cluster_median")))
    logger.info("    - final_cluster_median length: %s", _safe_len(results.get("final_cluster_median")))
    logger.info("    - admission_mode length: %s", _safe_len(results.get("admission_mode")))
    logger.info(
        "    - best_labels_raw_cluster_count length: %s",
        _safe_len(results.get("best_labels_raw_cluster_count")),
    )
    logger.info(
        "    - best_labels_final_cluster_count length: %s",
        _safe_len(results.get("best_labels_final_cluster_count")),
    )
    logger.info("    - source_target_cluster length: %s", _safe_len(results.get("source_target_cluster")))

    resolution_diag = results.get("resolution_diagnostics")
    search_diag = results.get("resolution_search_diagnostics")
    target_diag = results.get("target_diagnostics")
    if isinstance(target_diag, pd.DataFrame):
        logger.info("    - target diagnostics rows: %s", len(target_diag))
    if isinstance(search_diag, pd.DataFrame):
        logger.info("    - resolution search diagnostics rows: %s", len(search_diag))
        logger.info("    - search coverage complete: %s", bool(results.get("search_coverage_complete", False)))
    if isinstance(resolution_diag, pd.DataFrame):
        logger.info("    - manual resolution diagnostics rows: %s", len(resolution_diag))

    consistent_clusters = np.asarray(results.get("consistent_clusters", []), dtype=int)
    logger.info("  Returned final clusters: %s", _format_cluster_values(results.get("n_cluster")))
    logger.info("  Consistent clusters: %s", _format_cluster_values(consistent_clusters))

    if resolution_mode:
        logger.info("  Analysis mode: manual resolution")
        logger.info("  Manual resolutions evaluated: %s", _safe_len(resolution_values))
        if isinstance(resolution_diag, pd.DataFrame):
            superseded = int((~resolution_diag["selected"]).sum()) if "selected" in resolution_diag.columns else 0
            logger.info("  Manual resolutions retained after per-cluster IC selection: %s", _safe_len(results.get("n_cluster")))
            if superseded > 0:
                logger.info("  Manual resolutions superseded by lower-IC matches: %s", superseded)
    else:
        logger.info("  Analysis mode: cluster range search")
        logger.info("  Requested final cluster targets: %s", _format_cluster_values(requested_cluster_range))
        logger.info(
            "  Searched target clusters: %s",
            _format_cluster_values(results.get("searched_target_cluster_range")),
        )
        logger.info("  Search coverage complete: %s", bool(results.get("search_coverage_complete", False)))
        logger.info("  Coverage complete: %s", bool(results.get("coverage_complete", False)))
        uncovered_targets = np.asarray(results.get("uncovered_targets", []), dtype=int)
        if uncovered_targets.size:
            logger.info("  Uncovered targets: %s", _format_cluster_values(uncovered_targets))

    best_cluster = results.get("best_cluster", np.nan)
    best_resolution = results.get("best_resolution", np.nan)
    if np.isfinite(best_cluster):
        logger.info("  Best cluster: %s", int(best_cluster))
    if np.isfinite(best_resolution):
        logger.info("  Best resolution: %.6g", float(best_resolution))

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("  Total execution time: %.3f seconds", total_time)
