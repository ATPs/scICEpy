"""Result assembly helpers for scICEpy."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

_STORAGE_KIND_KEY = "__scicepy_storage_kind__"
_STORAGE_SEQUENCE_KIND = "sequence"


def _empty_float_array() -> np.ndarray:
    return np.asarray([], dtype=float)


def _empty_int_array() -> np.ndarray:
    return np.asarray([], dtype=int)


def _empty_object_array() -> np.ndarray:
    return np.asarray([], dtype=object)


def _is_simple_h5ad_scalar(value: Any) -> bool:
    return value is None or isinstance(
        value,
        (str, bytes, bool, int, float, np.bool_, np.integer, np.floating),
    )


def serialize_results_for_h5ad(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Index):
        return np.asarray(value)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): serialize_results_for_h5ad(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if all(_is_simple_h5ad_scalar(item) for item in value):
            return [serialize_results_for_h5ad(item) for item in value]
        return {
            _STORAGE_KIND_KEY: _STORAGE_SEQUENCE_KIND,
            "items": {
                str(idx): serialize_results_for_h5ad(item)
                for idx, item in enumerate(value)
            },
        }
    return value


def restore_results_from_h5ad(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get(_STORAGE_KIND_KEY) == _STORAGE_SEQUENCE_KIND:
            items = value.get("items", {})
            return [
                restore_results_from_h5ad(items[key])
                for key in sorted(items, key=lambda item: int(item))
            ]
        return {key: restore_results_from_h5ad(item) for key, item in value.items()}
    return value


def _as_array(results: list[dict[str, Any]], key: str, dtype, default) -> np.ndarray:
    return np.asarray(
        [item.get(key, default) if item.get(key) is not None else default for item in results],
        dtype=dtype,
    )


def _as_array_list(results: list[dict[str, Any]], key: str, dtype, default=None) -> list[np.ndarray]:
    fallback = [] if default is None else default
    return [
        np.asarray(item.get(key, fallback) if item.get(key) is not None else fallback, dtype=dtype)
        for item in results
    ]


def build_target_result_record(
    cluster_num: int,
    result: dict[str, Any] | None = None,
    *,
    source_target_cluster: float | None = None,
    excluded: bool = False,
    exclusion_reason: str | None = None,
    selected_main_result: bool = False,
    result_status: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    payload = {} if result is None else dict(result)
    payload.update(overrides)
    exclusion_reason = "excluded" if exclusion_reason is None and excluded else (exclusion_reason or "none")
    result_status = exclusion_reason if result_status is None and excluded else (result_status or "candidate")
    labels = None if excluded else payload.get("labels")
    best_labels = None if excluded or payload.get("best_labels") is None else np.asarray(payload.get("best_labels"), dtype=np.int32)
    ic_bootstrap = (
        _empty_float_array()
        if excluded
        else np.asarray(payload.get("ic_bootstrap", []), dtype=float)
    )
    mei = _empty_float_array() if excluded else np.asarray(payload.get("mei", []), dtype=float)
    optimization_diagnostics = payload.get("optimization_diagnostics")
    if not isinstance(optimization_diagnostics, pd.DataFrame):
        optimization_diagnostics = pd.DataFrame()
    admission_mode = payload.get("admission_mode", exclusion_reason if excluded else "none")
    return {
        "cluster_number": int(cluster_num),
        "gamma": float(payload.get("gamma", np.nan)),
        "labels": labels,
        "ic_median": np.nan if excluded else float(payload.get("ic_median", np.nan)),
        "ic_bootstrap": ic_bootstrap,
        "best_labels": best_labels,
        "effective_cluster_median": float(payload.get("effective_cluster_median", np.nan)),
        "raw_cluster_median": float(payload.get("raw_cluster_median", np.nan)),
        "final_cluster_median": float(payload.get("final_cluster_median", np.nan)),
        "admission_mode": str(admission_mode),
        "best_labels_raw_cluster_count": int(payload.get("best_labels_raw_cluster_count", -1)),
        "best_labels_final_cluster_count": int(payload.get("best_labels_final_cluster_count", -1)),
        "n_iterations": int(payload.get("n_iterations", 0)),
        "mei": mei,
        "k": int(payload.get("k", 0)),
        "source_target_cluster": float(cluster_num) if source_target_cluster is None else float(source_target_cluster),
        "excluded": bool(excluded),
        "exclusion_reason": str(exclusion_reason),
        "selected_main_result": bool(selected_main_result),
        "result_status": str(result_status),
        "phase1_primary_gamma_count": int(payload.get("phase1_primary_gamma_count", 0)),
        "phase1_secondary_gamma_count": int(payload.get("phase1_secondary_gamma_count", 0)),
        "phase1_total_gamma_count": int(payload.get("phase1_total_gamma_count", 0)),
        "phase1_elapsed_sec": float(payload.get("phase1_elapsed_sec", 0.0)),
        "phase1_leiden_runs": int(payload.get("phase1_leiden_runs", 0)),
        "secondary_phase1_used": bool(payload.get("secondary_phase1_used", False)),
        "exact_hit_gamma_count": int(payload.get("exact_hit_gamma_count", 0)),
        "phase4_iterations": int(payload.get("phase4_iterations", 0)),
        "phase4_elapsed_sec": float(payload.get("phase4_elapsed_sec", 0.0)),
        "phase5_elapsed_sec": float(payload.get("phase5_elapsed_sec", 0.0)),
        "optimization_elapsed_sec": float(payload.get("optimization_elapsed_sec", 0.0)),
        "optimization_diagnostics": optimization_diagnostics,
    }


def empty_results_dict() -> dict[str, Any]:
    return {
        "gamma": _empty_float_array(),
        "labels": [],
        "ic": _empty_float_array(),
        "ic_vec": [],
        "n_cluster": _empty_int_array(),
        "best_labels": [],
        "effective_cluster_median": _empty_float_array(),
        "raw_cluster_median": _empty_float_array(),
        "final_cluster_median": _empty_float_array(),
        "admission_mode": _empty_object_array(),
        "best_labels_raw_cluster_count": _empty_int_array(),
        "best_labels_final_cluster_count": _empty_int_array(),
        "source_target_cluster": _empty_float_array(),
        "n_iter": _empty_int_array(),
        "mei": [],
        "k": _empty_int_array(),
        "excluded": np.asarray([], dtype=bool),
        "exclusion_reason": _empty_object_array(),
        "result_status": _empty_object_array(),
        "phase1_primary_gamma_count": _empty_int_array(),
        "phase1_secondary_gamma_count": _empty_int_array(),
        "phase1_total_gamma_count": _empty_int_array(),
        "phase1_elapsed_sec": _empty_float_array(),
        "phase1_leiden_runs": _empty_int_array(),
        "secondary_phase1_used": np.asarray([], dtype=bool),
        "exact_hit_gamma_count": _empty_int_array(),
        "phase4_iterations": _empty_int_array(),
        "phase4_elapsed_sec": _empty_float_array(),
        "phase5_elapsed_sec": _empty_float_array(),
        "optimization_elapsed_sec": _empty_float_array(),
        "consistent_clusters": _empty_int_array(),
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
        "uncovered_targets": _empty_int_array(),
        "search_uncovered_targets": _empty_int_array(),
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
        "cluster_range_tested": _empty_int_array(),
    }


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
    result["gamma"] = _as_array(results, "gamma", float, np.nan)
    result["labels"] = [item.get("labels") for item in results]
    result["ic"] = _as_array(results, "ic_median", float, np.nan)
    result["ic_vec"] = _as_array_list(results, "ic_bootstrap", float)
    result["n_cluster"] = _as_array(results, "cluster_number", int, -1)
    result["best_labels"] = _as_array_list(results, "best_labels", np.int32)
    result["effective_cluster_median"] = _as_array(results, "effective_cluster_median", float, np.nan)
    result["raw_cluster_median"] = _as_array(results, "raw_cluster_median", float, np.nan)
    result["final_cluster_median"] = _as_array(results, "final_cluster_median", float, np.nan)
    result["admission_mode"] = _as_array(results, "admission_mode", object, "none")
    result["best_labels_raw_cluster_count"] = _as_array(results, "best_labels_raw_cluster_count", int, -1)
    result["best_labels_final_cluster_count"] = _as_array(results, "best_labels_final_cluster_count", int, -1)
    result["source_target_cluster"] = _as_array(results, "source_target_cluster", float, np.nan)
    result["n_iter"] = _as_array(results, "n_iterations", int, 0)
    result["mei"] = _as_array_list(results, "mei", float)
    result["k"] = _as_array(results, "k", int, 0)
    result["excluded"] = _as_array(results, "excluded", bool, False)
    result["exclusion_reason"] = _as_array(results, "exclusion_reason", object, "none")
    result["result_status"] = _as_array(results, "result_status", object, "selected_main_result")
    result["phase1_primary_gamma_count"] = _as_array(results, "phase1_primary_gamma_count", int, 0)
    result["phase1_secondary_gamma_count"] = _as_array(results, "phase1_secondary_gamma_count", int, 0)
    result["phase1_total_gamma_count"] = _as_array(results, "phase1_total_gamma_count", int, 0)
    result["phase1_elapsed_sec"] = _as_array(results, "phase1_elapsed_sec", float, 0.0)
    result["phase1_leiden_runs"] = _as_array(results, "phase1_leiden_runs", int, 0)
    result["secondary_phase1_used"] = _as_array(results, "secondary_phase1_used", bool, False)
    result["exact_hit_gamma_count"] = _as_array(results, "exact_hit_gamma_count", int, 0)
    result["phase4_iterations"] = _as_array(results, "phase4_iterations", int, 0)
    result["phase4_elapsed_sec"] = _as_array(results, "phase4_elapsed_sec", float, 0.0)
    result["phase5_elapsed_sec"] = _as_array(results, "phase5_elapsed_sec", float, 0.0)
    result["optimization_elapsed_sec"] = _as_array(results, "optimization_elapsed_sec", float, 0.0)
    return result


def rekey_target_results_by_final_cluster(
    target_results: list[dict[str, Any]],
    *,
    require_matching_source_target: bool = True,
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
        if require_matching_source_target and (source_target < 0 or int(final_cluster) != int(source_target)):
            item["excluded"] = True
            item["exclusion_reason"] = "final_cluster_mismatch"
            item["result_status"] = "final_cluster_mismatch"
            continue
        candidates_by_final_cluster[int(final_cluster)].append(item)

    main_results: list[dict[str, Any]] = []
    for final_cluster in sorted(candidates_by_final_cluster):
        chosen = min(
            candidates_by_final_cluster[final_cluster],
            key=lambda x: (float(x.get("ic_median", np.inf)), float(x.get("gamma", np.inf))),
        )
        chosen["selected_main_result"] = True
        chosen["cluster_number"] = int(final_cluster)
        chosen["excluded"] = False
        chosen["exclusion_reason"] = "none"
        chosen["result_status"] = "selected_main_result"
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


def _target_result_diagnostic_values(item: dict[str, Any] | None = None) -> dict[str, Any]:
    if item is None:
        return {
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
            "source_target_cluster": np.nan,
        }
    return {
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
    }


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
        row = {
            "requested_target_cluster": int(target),
            "searched_target_cluster": int(target),
            "requested_by_user": True,
            "gamma_left": float(bounds[0]),
            "gamma_right": float(bounds[1]),
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
        }
        if item is None:
            exclusion_reason = "resolution_search_failed" if not np.isfinite(bounds[0]) else "optimization_admission_failed"
            row.update(_target_result_diagnostic_values())
            row["source_target_cluster"] = int(target)
            row["excluded"] = True
            row["exclusion_reason"] = exclusion_reason
            row["selected_main_result"] = False
            row["result_status"] = exclusion_reason
            rows.append(row)
            continue

        row.update(_target_result_diagnostic_values(item))
        row["returned_final_cluster"] = float(item.get("best_labels_final_cluster_count", np.nan))
        row["returned_in_main_result"] = bool(item.get("selected_main_result", False))
        row["selected_result_cluster"] = float(item.get("best_labels_final_cluster_count", np.nan)) if bool(item.get("selected_main_result", False)) else (
            float(item.get("best_labels_final_cluster_count", np.nan))
            if _safe_int(item.get("best_labels_final_cluster_count", -1)) in selected_by_final_cluster
            else np.nan
        )
        row["superseded_by_source_target_cluster"] = (
            float(selected_by_final_cluster[_safe_int(item.get("best_labels_final_cluster_count", -1))].get("source_target_cluster", np.nan))
            if (not bool(item.get("selected_main_result", False))
                and not bool(item.get("excluded", False))
                and _safe_int(item.get("best_labels_final_cluster_count", -1)) in selected_by_final_cluster)
            else np.nan
        )
        row["excluded"] = bool(item.get("excluded", False))
        row["exclusion_reason"] = item.get("exclusion_reason", "none")
        row["selected_main_result"] = bool(item.get("selected_main_result", False))
        row["result_status"] = item.get("result_status", "selected_main_result")
        rows.append(row)
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
    sort_columns = [
        column
        for column in (
            "requested_target_cluster",
            "phase",
            "gamma",
            "admission_selected",
            "phase4_keep",
            "selected_best_gamma",
            "result_status",
        )
        if column in combined.columns
    ]
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return combined


def attach_summary_fields(result: dict[str, Any], ic_threshold: float) -> dict[str, Any]:
    ic = np.asarray(result.get("ic", []), dtype=float)
    clusters = np.asarray(result.get("n_cluster", []), dtype=int)
    gamma = np.asarray(result.get("gamma", []), dtype=float)
    if ic.size == 0:
        result["consistent_clusters"] = _empty_int_array()
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
