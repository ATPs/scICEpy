"""Runtime helpers for scICEpy."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


clustering_cache_env: dict[str, np.ndarray] = {}


def clear_clustering_cache() -> None:
    clustering_cache_env.clear()


def get_cache_stats() -> dict[str, int]:
    return {"cache_entries": len(clustering_cache_env)}


def cross_platform_mclapply(
    items: Sequence[Any],
    func: Callable[[Any], Any],
    mc_cores: int = 1,
    desc: str | None = None,
    verbose: bool = False,
) -> list[Any]:
    if mc_cores <= 1 or len(items) <= 1:
        iterable = tqdm(items, desc=desc, disable=not verbose) if desc else items
        return [func(item) for item in iterable]

    with mp.Pool(processes=mc_cores) as pool:
        if verbose and desc:
            return list(tqdm(pool.imap(func, items), total=len(items), desc=desc))
        return pool.map(func, items)


def parallel_map_threads(
    items: Sequence[Any],
    func: Callable[[Any], Any],
    max_workers: int = 1,
) -> list[Any]:
    items = list(items)
    workers = max(1, min(int(max_workers), len(items)))
    if workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, items))


def resolve_effective_workers(requested_workers: int) -> dict[str, int]:
    requested = max(1, int(requested_workers))
    detected = os.cpu_count() or 1
    max_workers = detected - 1 if detected > 1 else 1
    effective = 1 if os.name == "nt" else min(requested, max_workers)
    return {
        "requested": requested,
        "detected": int(detected),
        "effective": max(1, int(effective)),
    }


def estimate_outer_worker_bytes(
    n_cells: int,
    n_trials: int,
    n_bootstrap: int,
    expected_gamma_count: int = 11,
) -> float:
    n_cells = max(1, int(n_cells))
    n_trials = max(1, int(n_trials))
    n_bootstrap = max(1, int(n_bootstrap))
    expected_gamma_count = max(1, int(expected_gamma_count))

    base_matrix_bytes = estimate_trial_matrix_bytes(
        n_cells,
        max(n_trials, n_bootstrap),
        expected_gamma_count,
    )
    if n_cells >= 200000:
        graph_replication_factor = 8.0
    elif n_cells >= 100000:
        graph_replication_factor = 4.0
    elif n_cells >= 50000:
        graph_replication_factor = 2.5
    else:
        graph_replication_factor = 1.5
    return float(base_matrix_bytes * graph_replication_factor)


def resolve_nested_worker_layout(
    total_workers: int,
    task_count: int,
    n_cells: int,
    n_trials: int,
    n_bootstrap: int,
    runtime_context: "RuntimeContext | None" = None,
    outer_workers: int | None = None,
    inner_workers: int | None = None,
    expected_gamma_count: int = 11,
) -> dict[str, int]:
    total_workers = max(1, int(total_workers))
    task_count = max(1, int(task_count))
    n_cells = max(1, int(n_cells))
    n_trials = max(1, int(n_trials))
    n_bootstrap = max(1, int(n_bootstrap))
    expected_gamma_count = max(1, int(expected_gamma_count))
    max_inner_from_work = max(1, min(total_workers, max(n_trials, n_bootstrap)))
    large_graph = n_cells >= 200000

    def _default_large_graph_inner_workers() -> int:
        # Large graphs scale much better through outer process parallelism than through
        # deep nested thread pools in Phase 1. Bias toward smaller inner budgets so the
        # scheduler can keep more target optimizers active concurrently.
        phase1_parallel_ceiling = max(1, min(n_trials, expected_gamma_count))
        preferred_inner = min(max_inner_from_work, max(1, min(4, phase1_parallel_ceiling)))
        if phase1_parallel_ceiling <= 4:
            preferred_inner = min(preferred_inner, 3 if total_workers >= 12 else 2)
        if n_bootstrap <= 20:
            preferred_inner = min(preferred_inner, 3)
        return max(1, int(preferred_inner))

    if inner_workers is not None:
        resolved_inner = max(1, min(int(inner_workers), max_inner_from_work))
    elif outer_workers is not None:
        resolved_outer = max(1, min(int(outer_workers), task_count, total_workers))
        resolved_inner = max(1, min(max_inner_from_work, total_workers // resolved_outer))
    else:
        if large_graph:
            preferred_inner = _default_large_graph_inner_workers()
        else:
            preferred_inner = min(max_inner_from_work, max(1, int(round(total_workers ** 0.5))))
        resolved_outer = max(1, min(task_count, total_workers // max(1, preferred_inner)))
        resolved_inner = max(1, min(max_inner_from_work, total_workers // max(1, resolved_outer)))

    per_outer_bytes = estimate_outer_worker_bytes(
        n_cells=n_cells,
        n_trials=n_trials,
        n_bootstrap=n_bootstrap,
        expected_gamma_count=expected_gamma_count,
    )
    if outer_workers is not None:
        resolved_outer = max(1, min(int(outer_workers), task_count))
    else:
        resolved_outer = max(1, min(task_count, total_workers // max(1, resolved_inner)))
    resolved_outer = cap_workers_by_memory(
        resolved_outer,
        per_outer_bytes,
        runtime_context=runtime_context,
    )
    resolved_outer = max(1, min(resolved_outer, task_count))

    if inner_workers is not None:
        resolved_inner = max(1, min(int(inner_workers), max_inner_from_work))
    else:
        resolved_inner = max(1, min(max_inner_from_work, total_workers // max(1, resolved_outer)))
    resolved_inner = cap_workers_by_memory(
        resolved_inner,
        estimate_trial_matrix_bytes(n_cells, 1, 1),
        runtime_context=runtime_context,
    )
    resolved_inner = max(1, min(resolved_inner, max_inner_from_work))

    return {
        "total_workers": int(total_workers),
        "task_count": int(task_count),
        "outer_workers": int(max(1, min(resolved_outer, task_count))),
        "inner_workers": int(max(1, resolved_inner)),
        "estimated_bytes_per_outer_worker": int(max(1.0, per_outer_bytes)),
    }


def get_heartbeat_interval_seconds(default_seconds: float = 60.0) -> float:
    env_value = os.environ.get("SCICEPY_HEARTBEAT_SECONDS")
    if env_value:
        try:
            configured = float(env_value)
            if configured > 0:
                return configured
        except ValueError:
            pass
    return float(default_seconds)


def create_heartbeat_logger(
    verbose: bool,
    context: str = "",
    interval_seconds: float | None = None,
) -> Callable[[str | Callable[[], str]], bool]:
    enabled = bool(verbose)
    interval = interval_seconds if interval_seconds is not None else get_heartbeat_interval_seconds()
    interval = interval if interval and interval > 0 else 60.0
    last_emit = time.monotonic()
    prefix = f"{context} " if context else ""

    def _emit(message_text: str | Callable[[], str]) -> bool:
        nonlocal last_emit
        if not enabled:
            return False
        now = time.monotonic()
        if now - last_emit < interval:
            return False
        last_emit = now
        if callable(message_text):
            message_text = message_text()
        logger.info("%sHEARTBEAT (%ss): %s", prefix, int(round(interval)), message_text)
        return True

    return _emit


def estimate_trial_matrix_bytes(n_cells: int, n_trials: int, n_gamma: int = 1) -> float:
    return float(n_cells) * float(max(1, int(n_trials))) * float(max(1, int(n_gamma))) * 4.0


def detect_memory_budget_bytes(default_bytes: float = 4 * 1024**3) -> float:
    env_value = os.environ.get("SCICEPY_MEMORY_BUDGET_BYTES")
    if env_value:
        try:
            parsed = float(env_value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1]) * 1024.0 * 0.7
                    except ValueError:
                        break
    return float(default_bytes)


def cap_workers_by_memory(
    requested_workers: int,
    bytes_per_task: float,
    runtime_context: "RuntimeContext | None" = None,
) -> int:
    workers = max(1, int(requested_workers))
    if not np.isfinite(bytes_per_task) or bytes_per_task <= 0:
        return workers

    memory_budget = (
        runtime_context.memory_budget_bytes
        if runtime_context is not None
        else detect_memory_budget_bytes()
    )
    if not np.isfinite(memory_budget) or memory_budget <= 0:
        return workers

    max_workers = max(1, int(memory_budget // bytes_per_task))
    return max(1, min(workers, max_workers))


def summarize_adjacency_matrix(adjacency: Any) -> dict[str, Any]:
    shape = getattr(adjacency, "shape", (None, None))
    n_rows = int(shape[0]) if shape is not None and len(shape) >= 1 and shape[0] is not None else None
    n_cols = int(shape[1]) if shape is not None and len(shape) >= 2 and shape[1] is not None else None
    dtype = str(getattr(adjacency, "dtype", "unknown"))
    nnz = None
    sparsity = np.nan
    weight_min = np.nan
    weight_max = np.nan
    weight_mean = np.nan

    if hasattr(adjacency, "nnz"):
        try:
            nnz = int(adjacency.nnz)
        except Exception:
            nnz = None
    elif n_rows is not None and n_cols is not None and (n_rows * n_cols) <= 10_000_000:
        try:
            nnz = int(np.count_nonzero(np.asarray(adjacency)))
        except Exception:
            nnz = None

    if nnz is not None and n_rows is not None and n_cols is not None and n_rows > 0 and n_cols > 0:
        total_entries = float(n_rows) * float(n_cols)
        sparsity = (1.0 - (float(nnz) / total_entries)) * 100.0

    weight_values = None
    if hasattr(adjacency, "data"):
        try:
            weight_values = np.asarray(adjacency.data, dtype=float)
        except Exception:
            weight_values = None

    if weight_values is not None and weight_values.size:
        finite_weights = weight_values[np.isfinite(weight_values)]
        if finite_weights.size:
            weight_min = float(np.min(finite_weights))
            weight_max = float(np.max(finite_weights))
            weight_mean = float(np.mean(finite_weights))

    return {
        "shape": (n_rows, n_cols),
        "dtype": dtype,
        "nnz": nnz,
        "sparsity_percent": sparsity,
        "weight_min": weight_min,
        "weight_max": weight_max,
        "weight_mean": weight_mean,
    }


@dataclass
class RuntimeContext:
    spill_threshold_bytes: float = 2 * 1024**3
    memory_budget_bytes: float = detect_memory_budget_bytes()
    spill_dir: str | None = None
    spill_active: bool = False
    spill_announced: bool = False


def create_runtime_context() -> RuntimeContext:
    threshold = os.environ.get("SCICEPY_SPILL_THRESHOLD_BYTES")
    spill_threshold_bytes = 2 * 1024**3
    if threshold:
        try:
            parsed = float(threshold)
            if parsed > 0:
                spill_threshold_bytes = parsed
        except ValueError:
            pass
    return RuntimeContext(spill_threshold_bytes=spill_threshold_bytes)


def should_enable_spill(runtime_context: RuntimeContext | None, estimated_bytes: float) -> bool:
    if runtime_context is None:
        return False
    return bool(np.isfinite(estimated_bytes) and estimated_bytes >= runtime_context.spill_threshold_bytes)


def activate_runtime_spill(runtime_context: RuntimeContext | None, estimated_bytes: float | None = None) -> bool:
    if runtime_context is None:
        return False
    if runtime_context.spill_active and runtime_context.spill_dir:
        return True

    spill_dir = tempfile.mkdtemp(prefix="scicepy_spill_")
    runtime_context.spill_dir = spill_dir
    runtime_context.spill_active = True
    if not runtime_context.spill_announced:
        if estimated_bytes is not None and np.isfinite(estimated_bytes):
            logger.info(
                "scICEpy temp spill dir: %s (estimated matrix footprint %.2f GiB)",
                spill_dir,
                estimated_bytes / 1024**3,
            )
        else:
            logger.info("scICEpy temp spill dir: %s", spill_dir)
        runtime_context.spill_announced = True
    return True


def cleanup_runtime_spill(runtime_context: RuntimeContext | None) -> None:
    if runtime_context is None or not runtime_context.spill_dir:
        return
    if os.path.isdir(runtime_context.spill_dir):
        shutil.rmtree(runtime_context.spill_dir, ignore_errors=True)
    runtime_context.spill_active = False
    runtime_context.spill_dir = None


def store_cluster_matrix(
    cluster_matrix: np.ndarray,
    runtime_context: RuntimeContext | None,
    prefix: str = "cluster_matrix",
) -> dict[str, Any]:
    if runtime_context is None or not should_enable_spill(runtime_context, float(cluster_matrix.nbytes)):
        return {"type": "memory", "matrix": np.asarray(cluster_matrix, dtype=np.int32)}

    activate_runtime_spill(runtime_context, estimated_bytes=float(cluster_matrix.nbytes))
    assert runtime_context.spill_dir is not None
    file_path = tempfile.mktemp(prefix=f"{prefix}_", suffix=".npy", dir=runtime_context.spill_dir)
    np.save(file_path, np.asarray(cluster_matrix, dtype=np.int32), allow_pickle=False)
    return {"type": "spill", "path": file_path}


def load_cluster_matrix(matrix_ref: dict[str, Any]) -> np.ndarray:
    if matrix_ref["type"] == "memory":
        return np.asarray(matrix_ref["matrix"], dtype=np.int32)
    return np.load(matrix_ref["path"], allow_pickle=False)


def release_cluster_matrix(matrix_ref: dict[str, Any] | None) -> None:
    if not matrix_ref:
        return
    if matrix_ref.get("type") == "spill" and matrix_ref.get("path"):
        try:
            os.remove(matrix_ref["path"])
        except FileNotFoundError:
            pass


def release_cluster_matrix_refs(matrix_refs: Iterable[dict[str, Any] | None]) -> None:
    for ref in matrix_refs:
        release_cluster_matrix(ref)
