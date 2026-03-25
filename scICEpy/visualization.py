"""Visualization and label extraction for scICEpy."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .results import restore_results_from_h5ad


def _resolve_results(data_or_results: Any) -> dict[str, Any]:
    if isinstance(data_or_results, dict):
        return restore_results_from_h5ad(data_or_results)
    if hasattr(data_or_results, "uns") and "scICE" in data_or_results.uns:
        return restore_results_from_h5ad(data_or_results.uns["scICE"])
    raise ValueError("No scICE results found. Run scICE_clustering() first.")


def _resolve_obs_names(data_or_results: Any) -> pd.Index:
    if hasattr(data_or_results, "obs_names"):
        return pd.Index(data_or_results.obs_names)
    results = _resolve_results(data_or_results)
    cell_names = results.get("cell_names")
    if cell_names is None or len(cell_names) == 0:
        raise ValueError("Cell names are required when plotting or extracting labels from a raw results dict.")
    return pd.Index(cell_names)


def get_robust_labels(data_or_results, threshold: float = 1.005, return_adata: bool = False):
    results = _resolve_results(data_or_results)
    obs_names = _resolve_obs_names(data_or_results)
    ic = np.asarray(results.get("ic", []), dtype=float)
    valid_idx = np.where(ic < threshold)[0]

    if valid_idx.size == 0:
        warnings.warn(f"No clusterings found below IC threshold {threshold}")
        if return_adata and hasattr(data_or_results, "obs"):
            return data_or_results
        return pd.DataFrame(index=obs_names)

    label_dict: dict[str, Any] = {}
    cluster_numbers = np.asarray(results.get("n_cluster", []), dtype=int)
    best_labels = results.get("best_labels", [])
    for idx in valid_idx:
        cluster_num = int(cluster_numbers[idx])
        label_dict[f"scICE_k_{cluster_num}"] = pd.Categorical(np.asarray(best_labels[idx], dtype=np.int32))

    labels_df = pd.DataFrame(label_dict, index=obs_names)
    if return_adata:
        if not hasattr(data_or_results, "obs"):
            raise ValueError("return_adata=True requires an AnnData-like object.")
        for column_name, values in label_dict.items():
            data_or_results.obs[column_name] = values
        return data_or_results
    return labels_df


def plot_ic(
    data_or_results,
    threshold: float = 1.005,
    figsize: tuple[float, float] = (10, 6),
    title: str = "Clustering Consistency Analysis",
    show_threshold: bool = True,
    show_gamma: bool = True,
):
    import matplotlib.pyplot as plt

    results = _resolve_results(data_or_results)
    ic_vec = results.get("ic_vec", [])
    if len(ic_vec) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        ax.text(0.5, 0.5, "NO consistent cluster numbers found", ha="center", va="center", color="red", fontweight="bold")
        ax.set_title(title)
        return fig, ax

    cluster_numbers = np.asarray(results.get("n_cluster", []), dtype=int)
    ic = np.asarray(results.get("ic", []), dtype=float)
    excluded_targets = []
    target_diag = results.get("target_diagnostics")
    if isinstance(target_diag, pd.DataFrame) and {"searched_target_cluster", "excluded"} <= set(target_diag.columns):
        excluded_targets = target_diag.loc[target_diag["excluded"], "searched_target_cluster"].astype(int).tolist()

    valid_indices = [idx for idx, vec in enumerate(ic_vec) if len(vec) > 0 and np.isfinite(ic[idx])]
    if not valid_indices:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        ax.text(0.5, 0.5, "NO consistent cluster numbers found", ha="center", va="center", color="red", fontweight="bold")
        ax.set_title(title)
        return fig, ax

    cluster_numbers = cluster_numbers[valid_indices]
    ic = ic[valid_indices]
    box_data = [np.asarray(ic_vec[idx], dtype=float) for idx in valid_indices]
    consistent_mask = ic < threshold

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(box_data, positions=np.arange(len(cluster_numbers)), widths=0.6, patch_artist=True)
    for patch, is_consistent in zip(bp["boxes"], consistent_mask):
        patch.set_facecolor("lightgreen" if is_consistent else "lightgray")
        patch.set_alpha(0.7)

    for pos, values in enumerate(box_data):
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=len(values))
        ax.scatter(np.full(len(values), pos) + jitter, values, alpha=0.35, s=12, color="black")

    tick_labels = [str(int(k)) for k in cluster_numbers]
    if show_gamma:
        gamma = np.asarray(results.get("gamma", []), dtype=float)[valid_indices]
        tick_labels = [
            f"{int(k)}\n{('NA' if not np.isfinite(g) else f'{g:.2e}')}"
            for k, g in zip(cluster_numbers, gamma)
        ]

    ax.set_xticks(np.arange(len(cluster_numbers)))
    ax.set_xticklabels(tick_labels, rotation=45 if show_gamma else 0, ha="right" if show_gamma else "center")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inconsistency (IC) Score")
    subtitle_parts = [f"Lower IC scores indicate more consistent clustering. Threshold: {threshold}"]
    if excluded_targets:
        subtitle_parts.append(f"Excluded searched targets: {', '.join(map(str, excluded_targets))}")
    ax.set_title(f"{title}\n" + "\n".join(subtitle_parts))
    ax.grid(axis="y", alpha=0.3)

    if show_threshold:
        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1)
        ax.text(max(len(cluster_numbers) - 1, 0) * 0.8, threshold + 0.001, f"Threshold = {threshold}", color="red")

    fig.tight_layout()
    return fig, ax
