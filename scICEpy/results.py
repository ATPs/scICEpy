"""Result assembly helpers for scICEpy."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd


def empty_results_dict() -> dict[str, Any]:
    return {
        "gamma": np.asarray([], dtype=float),
        "labels": [],
        "ic": np.asarray([], dtype=float),
        "ic_vec": [],
        "n_cluster": np.asarray([], dtype=int),
        "best_labels": [],
        "effective_cluster_median": np.asarray([], dtype=float),
        "raw_cluster_median": np.asarray([], dtype=float),
        "final_cluster_median": np.asarray([], dtype=float),
        "admission_mode": np.asarray([], dtype=object),
        "best_labels_raw_cluster_count": np.asarray([], dtype=int),
        "best_labels_final_cluster_count": np.asarray([], dtype=int),
        "source_target_cluster": np.asarray([], dtype=float),
        "n_iter": np.asarray([], dtype=int),
        "mei": [],
        "k": np.asarray([], dtype=int),
        "excluded": np.asarray([], dtype=bool),
        "exclusion_reason": np.asarray([], dtype=object),
        "result_status": np.asarray([], dtype=object),
        "phase1_primary_gamma_count": np.asarray([], dtype=int),
        "phase1_secondary_gamma_count": np.asarray([], dtype=int),
        "phase1_total_gamma_count": np.asarray([], dtype=int),
        "phase1_elapsed_sec": np.asarray([], dtype=float),
        "phase1_leiden_runs": np.asarray([], dtype=int),
        "secondary_phase1_used": np.asarray([], dtype=bool),
        "exact_hit_gamma_count": np.asarray([], dtype=int),
        "phase4_iterations": np.asarray([], dtype=int),
        "phase4_elapsed_sec": np.asarray([], dtype=float),
        "phase5_elapsed_sec": np.asarray([], dtype=float),
        "optimization_elapsed_sec": np.asarray([], dtype=float),
        "consistent_clusters": np.asarray([], dtype=int),
        "analysis_mode": "cluster_range",
        "resolution_input": None,
        "resolution_diagnostics": None,
        "requested_cluster_range": None,
        "searched_target_cluster_range": None,
        "coverage_complete": True,
        "search_coverage_complete": True,
        "resolution_search_diagnostics": pd.DataFrame(),
        "optimization_diagnostics": pd.DataFrame(),
        "discovered_upper_gamma": np.nan,
        "upper_cap_stop_reason": None,
        "coarse_probe_count": np.nan,
        "target_diagnostics": pd.DataFrame(),
        "target_gamma_seeds": {},
        "target_interval_details": {},
        "plateau_stop": False,
        "uncovered_targets": np.asarray([], dtype=int),
        "search_uncovered_targets": np.asarray([], dtype=int),
        "best_cluster": np.nan,
        "best_resolution": np.nan,
        "min_cluster_size": 1,
        "cell_names": np.asarray([], dtype=object),
        "graph_key": None,
        "graph_name": None,
        "beta": np.nan,
        "beta_supported": False,
        "beta_applied": False,
        "beta_support_reason": None,
        "parallel_layout": None,
        "cluster_range_tested": np.asarray([], dtype=int),
    }


def _as_numpy(values: Iterable[Any], dtype=None) -> np.ndarray:
    return np.asarray(list(values), dtype=dtype)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None or not np.isfinite(value):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def cluster_results_to_dict(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return empty_results_dict()

    result = empty_results_dict()
    result["gamma"] = _as_numpy((x["gamma"] for x in results), dtype=float)
    result["labels"] = [x["labels"] for x in results]
    result["ic"] = _as_numpy((x["ic_median"] for x in results), dtype=float)
    result["ic_vec"] = [np.asarray(x["ic_bootstrap"], dtype=float) for x in results]
    result["n_cluster"] = _as_numpy((x["cluster_number"] for x in results), dtype=int)
    result["best_labels"] = [np.asarray(x["best_labels"], dtype=np.int32) for x in results]
    result["effective_cluster_median"] = _as_numpy(
        (x.get("effective_cluster_median", np.nan) for x in results),
        dtype=float,
    )
    result["raw_cluster_median"] = _as_numpy(
        (x.get("raw_cluster_median", np.nan) for x in results),
        dtype=float,
    )
    result["final_cluster_median"] = _as_numpy(
        (x.get("final_cluster_median", np.nan) for x in results),
        dtype=float,
    )
    result["admission_mode"] = _as_numpy(
        (x.get("admission_mode", "none") for x in results),
        dtype=object,
    )
    result["best_labels_raw_cluster_count"] = _as_numpy(
        (x.get("best_labels_raw_cluster_count", -1) for x in results),
        dtype=int,
    )
    result["best_labels_final_cluster_count"] = _as_numpy(
        (x.get("best_labels_final_cluster_count", -1) for x in results),
        dtype=int,
    )
    result["source_target_cluster"] = _as_numpy(
        (x.get("source_target_cluster", np.nan) for x in results),
        dtype=float,
    )
    result["n_iter"] = _as_numpy((x.get("n_iterations", 0) for x in results), dtype=int)
    result["mei"] = [np.asarray(x["mei"], dtype=float) for x in results]
    result["k"] = _as_numpy((x.get("k", 0) for x in results), dtype=int)
    result["excluded"] = _as_numpy((bool(x.get("excluded", False)) for x in results), dtype=bool)
    result["exclusion_reason"] = _as_numpy(
        (x.get("exclusion_reason", "none") for x in results),
        dtype=object,
    )
    result["result_status"] = _as_numpy(
        (x.get("result_status", "selected_main_result") for x in results),
        dtype=object,
    )
    for field, dtype in [
        ("phase1_primary_gamma_count", int),
        ("phase1_secondary_gamma_count", int),
        ("phase1_total_gamma_count", int),
        ("phase1_elapsed_sec", float),
        ("phase1_leiden_runs", int),
        ("secondary_phase1_used", bool),
        ("exact_hit_gamma_count", int),
        ("phase4_iterations", int),
        ("phase4_elapsed_sec", float),
        ("phase5_elapsed_sec", float),
        ("optimization_elapsed_sec", float),
    ]:
        result[field] = _as_numpy((x.get(field, 0) for x in results), dtype=dtype)
    return result


def rekey_target_results_by_final_cluster(
    target_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not target_results:
        return [], []

    copied = [dict(item) for item in target_results]
    for item in copied:
        item.setdefault("source_target_cluster", item.get("cluster_number", np.nan))
        item["selected_main_result"] = False

    candidates_by_final_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in copied:
        final_cluster = item.get("best_labels_final_cluster_count")
        source_target = _safe_int(item.get("source_target_cluster", item.get("cluster_number", np.nan)))
        if item.get("excluded") or final_cluster is None or int(final_cluster) < 0:
            continue
        if source_target < 0 or int(final_cluster) != int(source_target):
            item["excluded"] = True
            item["exclusion_reason"] = "final_cluster_mismatch"
            item["result_status"] = "final_cluster_mismatch"
            continue
        candidates_by_final_cluster[int(final_cluster)].append(item)

    selected_ids: set[int] = set()
    main_results: list[dict[str, Any]] = []
    for final_cluster in sorted(candidates_by_final_cluster):
        candidates = candidates_by_final_cluster[final_cluster]
        chosen = min(
            candidates,
            key=lambda x: (float(x.get("ic_median", np.inf)), float(x.get("gamma", np.inf))),
        )
        chosen["selected_main_result"] = True
        chosen["cluster_number"] = int(final_cluster)
        chosen["excluded"] = False
        chosen["exclusion_reason"] = "none"
        chosen["result_status"] = "selected_main_result"
        selected_ids.add(id(chosen))
        main_results.append(chosen)

    for item in copied:
        if item.get("selected_main_result"):
            continue
        if item.get("excluded"):
            item["result_status"] = item.get("exclusion_reason", "excluded")
        else:
            item["result_status"] = "deduplicated"
    copied.sort(key=lambda x: (float(x.get("source_target_cluster", np.inf)), float(x.get("gamma", np.inf))))
    main_results.sort(key=lambda x: (int(x["cluster_number"]), float(x.get("source_target_cluster", np.inf)), float(x["gamma"])))
    return main_results, copied


def build_target_diagnostics_df(
    target_results: list[dict[str, Any]],
    requested_cluster_range: np.ndarray,
    gamma_dict: dict[int, tuple[float, float]] | None = None,
    target_interval_details: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    requested_cluster_range = np.asarray(requested_cluster_range, dtype=int)
    rows: list[dict[str, Any]] = []
    lookup = {
        int(item.get("source_target_cluster", item.get("cluster_number", -1))): item
        for item in target_results
    }
    selected_by_final_cluster = {
        _safe_int(item.get("best_labels_final_cluster_count")): item
        for item in target_results
        if bool(item.get("selected_main_result", False))
        and np.isfinite(item.get("best_labels_final_cluster_count", np.nan))
        and _safe_int(item.get("best_labels_final_cluster_count", -1)) >= 0
    }
    for target in requested_cluster_range:
        detail = (target_interval_details or {}).get(str(int(target)), {})
        bounds = gamma_dict.get(int(target), (np.nan, np.nan)) if gamma_dict else (np.nan, np.nan)
        item = lookup.get(int(target))
        if item is None:
            exclusion_reason = "resolution_search_failed" if not np.isfinite(bounds[0]) else "optimization_admission_failed"
            rows.append(
                {
                    "requested_target_cluster": int(target),
                    "searched_target_cluster": int(target),
                    "requested_by_user": True,
                    "gamma_left": float(bounds[0]),
                    "gamma_right": float(bounds[1]),
                    "gamma": np.nan,
                    "ic": np.nan,
                    "effective_cluster_median": np.nan,
                    "raw_cluster_median": np.nan,
                    "final_cluster_median": np.nan,
                    "best_labels_raw_cluster_count": np.nan,
                    "best_labels_final_cluster_count": np.nan,
                    "phase1_primary_gamma_count": np.nan,
                    "phase1_secondary_gamma_count": np.nan,
                    "phase1_total_gamma_count": np.nan,
                    "phase1_elapsed_sec": np.nan,
                    "phase1_leiden_runs": np.nan,
                    "secondary_phase1_used": np.nan,
                    "exact_hit_gamma_count": np.nan,
                    "phase4_iterations": np.nan,
                    "phase4_elapsed_sec": np.nan,
                    "phase5_elapsed_sec": np.nan,
                    "optimization_elapsed_sec": np.nan,
                    "source_target_cluster": int(target),
                    "returned_final_cluster": np.nan,
                    "returned_in_main_result": False,
                    "selected_result_cluster": np.nan,
                    "superseded_by_source_target_cluster": np.nan,
                    "target_interval_mode": detail.get("mode", "missing"),
                    "search_bracketed": bool(detail.get("bracketed", False)),
                    "search_optimization_ready": bool(detail.get("optimization_ready", False)),
                    "search_has_exact_probe": bool(detail.get("has_exact_probe", False)),
                    "final_exact_probe_count": int(len(detail.get("exact_probe_values", []) or [])),
                    "final_near_probe_count": int(len(detail.get("near_probe_values", []) or [])),
                    "raw_interval_mode": detail.get("raw_interval_mode", "missing"),
                    "raw_exact_probe_count": int(len(detail.get("raw_exact_probe_values", []) or [])),
                    "raw_near_probe_count": int(len(detail.get("raw_near_probe_values", []) or [])),
                    "seed_gamma_count": int(len(detail.get("seed_gamma_values", []) or [])),
                    "excluded": True,
                    "exclusion_reason": exclusion_reason,
                    "selected_main_result": False,
                    "result_status": exclusion_reason,
                }
            )
            continue

        rows.append(
            {
                "requested_target_cluster": int(target),
                "searched_target_cluster": int(target),
                "requested_by_user": True,
                "gamma_left": float(bounds[0]),
                "gamma_right": float(bounds[1]),
                "gamma": float(item.get("gamma", np.nan)),
                "ic": float(item.get("ic_median", np.nan)),
                "effective_cluster_median": float(item.get("effective_cluster_median", np.nan)),
                "raw_cluster_median": float(item.get("raw_cluster_median", np.nan)),
                "final_cluster_median": float(item.get("final_cluster_median", np.nan)),
                "best_labels_raw_cluster_count": int(item.get("best_labels_raw_cluster_count", -1)),
                "best_labels_final_cluster_count": int(item.get("best_labels_final_cluster_count", -1)),
                "phase1_primary_gamma_count": int(item.get("phase1_primary_gamma_count", 0)),
                "phase1_secondary_gamma_count": int(item.get("phase1_secondary_gamma_count", 0)),
                "phase1_total_gamma_count": int(item.get("phase1_total_gamma_count", 0)),
                "phase1_elapsed_sec": float(item.get("phase1_elapsed_sec", np.nan)),
                "phase1_leiden_runs": int(item.get("phase1_leiden_runs", 0)),
                "secondary_phase1_used": bool(item.get("secondary_phase1_used", False)),
                "exact_hit_gamma_count": int(item.get("exact_hit_gamma_count", 0)),
                "phase4_iterations": int(item.get("phase4_iterations", 0)),
                "phase4_elapsed_sec": float(item.get("phase4_elapsed_sec", np.nan)),
                "phase5_elapsed_sec": float(item.get("phase5_elapsed_sec", np.nan)),
                "optimization_elapsed_sec": float(item.get("optimization_elapsed_sec", np.nan)),
                "source_target_cluster": float(item.get("source_target_cluster", np.nan)),
                "returned_final_cluster": float(item.get("best_labels_final_cluster_count", np.nan)),
                "returned_in_main_result": bool(item.get("selected_main_result", False)),
                "selected_result_cluster": float(item.get("best_labels_final_cluster_count", np.nan))
                if bool(item.get("selected_main_result", False))
                else (
                    float(item.get("best_labels_final_cluster_count", np.nan))
                    if _safe_int(item.get("best_labels_final_cluster_count", -1)) in selected_by_final_cluster
                    else np.nan
                ),
                "superseded_by_source_target_cluster": (
                    float(selected_by_final_cluster[_safe_int(item.get("best_labels_final_cluster_count", -1))].get("source_target_cluster", np.nan))
                    if (not bool(item.get("selected_main_result", False))
                        and not bool(item.get("excluded", False))
                        and _safe_int(item.get("best_labels_final_cluster_count", -1)) in selected_by_final_cluster)
                    else np.nan
                ),
                "target_interval_mode": detail.get("mode", "missing"),
                "search_bracketed": bool(detail.get("bracketed", False)),
                "search_optimization_ready": bool(detail.get("optimization_ready", False)),
                "search_has_exact_probe": bool(detail.get("has_exact_probe", False)),
                "final_exact_probe_count": int(len(detail.get("exact_probe_values", []) or [])),
                "final_near_probe_count": int(len(detail.get("near_probe_values", []) or [])),
                "raw_interval_mode": detail.get("raw_interval_mode", "missing"),
                "raw_exact_probe_count": int(len(detail.get("raw_exact_probe_values", []) or [])),
                "raw_near_probe_count": int(len(detail.get("raw_near_probe_values", []) or [])),
                "seed_gamma_count": int(len(detail.get("seed_gamma_values", []) or [])),
                "excluded": bool(item.get("excluded", False)),
                "exclusion_reason": item.get("exclusion_reason", "none"),
                "selected_main_result": bool(item.get("selected_main_result", False)),
                "result_status": item.get("result_status", "selected_main_result"),
            }
        )
    return pd.DataFrame(rows)


def build_optimization_diagnostics_df(target_results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for item in target_results:
        diagnostics = item.get("optimization_diagnostics")
        if diagnostics is None or not isinstance(diagnostics, pd.DataFrame) or diagnostics.empty:
            continue
        diagnostics = diagnostics.copy()
        diagnostics["requested_target_cluster"] = int(item.get("source_target_cluster", item.get("cluster_number", -1)))
        diagnostics["result_status"] = item.get("result_status", "candidate")
        rows.append(diagnostics)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True, sort=False)
    order_columns = [
        "requested_target_cluster",
        "phase",
        "gamma",
        "admission_selected",
        "phase4_keep",
        "selected_best_gamma",
        "result_status",
    ]
    sort_columns = [column for column in order_columns if column in combined.columns]
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return combined


def attach_summary_fields(result: dict[str, Any], ic_threshold: float) -> dict[str, Any]:
    ic = np.asarray(result.get("ic", []), dtype=float)
    clusters = np.asarray(result.get("n_cluster", []), dtype=int)
    gamma = np.asarray(result.get("gamma", []), dtype=float)
    if ic.size == 0:
        result["consistent_clusters"] = np.asarray([], dtype=int)
        result["best_cluster"] = np.nan
        result["best_resolution"] = np.nan
        result["cluster_range_tested"] = np.asarray(result.get("n_cluster", []), dtype=int)
        return result

    consistent_indices = np.where(ic < ic_threshold)[0]
    result["consistent_clusters"] = clusters[consistent_indices]
    best_idx = int(np.nanargmin(ic))
    result["best_cluster"] = int(clusters[best_idx])
    result["best_resolution"] = float(gamma[best_idx])
    result["cluster_range_tested"] = np.asarray(result.get("n_cluster", []), dtype=int)
    return result


def finalize_cluster_range_results(
    target_results: list[dict[str, Any]],
    requested_cluster_range: np.ndarray,
    searched_target_cluster_range: np.ndarray,
    search_coverage_complete: bool,
    gamma_dict: dict[int, tuple[float, float]] | None = None,
    resolution_search_diagnostics: pd.DataFrame | None = None,
    plateau_stop: bool = False,
    search_uncovered_targets: np.ndarray | None = None,
    discovered_upper_gamma: float | None = None,
    upper_cap_stop_reason: str | None = None,
    coarse_probe_count: int | None = None,
    target_gamma_seeds: dict[str, list[float]] | None = None,
    target_interval_details: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    main_results, full_target_results = rekey_target_results_by_final_cluster(target_results)
    result = cluster_results_to_dict(main_results)
    result["requested_cluster_range"] = np.asarray(requested_cluster_range, dtype=int)
    result["searched_target_cluster_range"] = np.asarray(searched_target_cluster_range, dtype=int)
    result["search_coverage_complete"] = bool(search_coverage_complete)
    result["resolution_search_diagnostics"] = (
        resolution_search_diagnostics if resolution_search_diagnostics is not None else pd.DataFrame()
    )
    result["optimization_diagnostics"] = build_optimization_diagnostics_df(full_target_results)
    result["plateau_stop"] = bool(plateau_stop)
    result["search_uncovered_targets"] = np.asarray(
        [] if search_uncovered_targets is None else search_uncovered_targets,
        dtype=int,
    )
    result["target_gamma_seeds"] = {} if target_gamma_seeds is None else dict(target_gamma_seeds)
    result["target_interval_details"] = {} if target_interval_details is None else dict(target_interval_details)
    result["discovered_upper_gamma"] = np.nan if discovered_upper_gamma is None else float(discovered_upper_gamma)
    result["upper_cap_stop_reason"] = upper_cap_stop_reason
    result["coarse_probe_count"] = np.nan if coarse_probe_count is None else int(coarse_probe_count)
    result["uncovered_targets"] = np.asarray(
        sorted(set(map(int, requested_cluster_range)) - set(map(int, result["n_cluster"]))),
        dtype=int,
    )
    result["coverage_complete"] = bool(result["uncovered_targets"].size == 0)
    result["target_diagnostics"] = build_target_diagnostics_df(
        full_target_results,
        requested_cluster_range=np.asarray(requested_cluster_range, dtype=int),
        gamma_dict=gamma_dict,
        target_interval_details=target_interval_details,
    )
    return result
